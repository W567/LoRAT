"""Live (online) tracker wrapper for trackit one_stream_tracker pipelines."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Optional, Tuple

import numpy as np
import torch

from trackit._resources import get_default_config_root
from trackit.core.operator.numpy.bbox.rasterize import bbox_rasterize
from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
from trackit.core.runtime.global_constant import get_global_constant
from trackit.core.runtime.utils.custom_yaml_loader import load_yaml
from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
from trackit.core.utils.siamfc_cropping import (
    apply_siamfc_cropping,
    apply_siamfc_cropping_to_boxes,
    get_siamfc_cropping_params,
    reverse_siamfc_cropping_params,
)
from trackit.models import ModelImplementationSuggestions, _load_state_dict_from_file
from trackit.models.methods.builder import create_model_build_context
from trackit.runners.evaluation.common.siamfc_search_region_cropping_params_provider.simple import (
    SiamFCCroppingParameterSimpleProvider,
)
from trackit.runners.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import (
    PostProcessing_BoxWithScoreMap,
)


__all__ = ['LiveTracker', 'load_pipeline_config']


def load_pipeline_config(method: str, config: str,
                         config_root: Optional[str] = None) -> dict:
    """Load ``<config_root>/<method>/<config>/config.yaml`` with trackit's loader."""
    root = config_root or os.environ.get('LORAT_CONFIG_PATH') or get_default_config_root()
    config_path = os.path.join(root, method, config, 'config.yaml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"trackit config not found: {config_path}")
    return load_yaml(config_path, get_global_constant())


def _find_eval_pipeline_config(config: dict) -> dict:
    for cfg in config['run']['runner'].values():
        if cfg.get('type') == 'default_eval':
            return cfg['evaluator']['pipeline']
    raise RuntimeError("No 'default_eval' runner found under run.runner.*")


def _find_template_area_factor(config: dict) -> float:
    for cfg in config['run']['data'].values():
        if cfg.get('type') != 'siamese_tracker_eval':
            continue
        transform = cfg.get('transform', {})
        if 'template_area_factor' in transform:
            return float(transform['template_area_factor'])
    raise RuntimeError("template_area_factor not found in any siamese_tracker_eval data section")


def _require_type(cfg: dict, expected: str, label: str) -> None:
    actual = cfg.get('type')
    if actual != expected:
        raise RuntimeError(f"LiveTracker only supports {label} type {expected!r}, got {actual!r}")


class LiveTracker:
    """Online tracker over a trackit one_stream_tracker pipeline.

    Lifecycle: construct → ``init(rgb, bbox_xyxy)`` once → ``track(rgb)`` per frame.
    """

    def __init__(self, config: dict, weight_path: str,
                 device: torch.device, dtype: torch.dtype,
                 window_penalty_override: Optional[float] = None,
                 use_autocast: bool = True,
                 use_compile: bool = False,
                 warmup_iters: int = 3):
        common = config['common']
        self.template_size: Tuple[int, int] = tuple(common['template_size'])
        self.search_region_size: Tuple[int, int] = tuple(common['search_region_size'])
        self.template_feat_size: Tuple[int, int] = tuple(common['template_feat_size'])
        self.response_map_size: Tuple[int, int] = tuple(common['response_map_size'])
        self.interpolation_mode: str = common['interpolation_mode']
        self.interpolation_align_corners: bool = common['interpolation_align_corners']

        pipeline_cfg = _find_eval_pipeline_config(config)
        _require_type(pipeline_cfg, 'one_stream_tracker', 'pipeline')

        crop_cfg = pipeline_cfg['search_region_cropping']
        _require_type(crop_cfg, 'simple', 'search_region_cropping')
        self.search_area_factor = float(crop_cfg['area_factor'])
        self.min_object_size = crop_cfg.get('min_object_size')

        pp_cfg = pipeline_cfg['post_process']
        _require_type(pp_cfg, 'box_with_score_map', 'post_process')
        wp = window_penalty_override if window_penalty_override is not None else pp_cfg.get('window_penalty', 0.0)
        self.post_process = PostProcessing_BoxWithScoreMap(
            device, self.response_map_size, self.search_region_size,
            window_penalty_ratio=float(wp))
        self.post_process.start()

        self.template_area_factor = _find_template_area_factor(config)

        impl = ModelImplementationSuggestions(
            device=device, dtype=dtype,
            optimize_for_inference=True, load_pretrained=True)
        self.model = create_model_build_context(config).create_fn(impl)
        _load_state_dict_from_file(self.model, weight_path, device, strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if use_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"[LiveTracker] torch.compile failed, falling back: {e}")

        self.image_normalize_ = get_dataset_norm_stats_transform(common['normalization'], inplace=True)
        self.device = device
        self.dtype = dtype
        self.warmup_iters = max(0, warmup_iters)

        self._is_cuda = device.type == 'cuda'
        if use_autocast and self._is_cuda and dtype != torch.float32:
            self._autocast_ctx = lambda: torch.autocast(device_type='cuda', dtype=dtype)
        else:
            self._autocast_ctx = nullcontext

        self.cropping_provider = SiamFCCroppingParameterSimpleProvider(
            self.search_area_factor, self.min_object_size)

        self._template: Optional[torch.Tensor] = None
        self._template_image_mean: Optional[torch.Tensor] = None
        self._z_feat_mask: Optional[torch.Tensor] = None
        self._upload_buffer: Optional[torch.Tensor] = None  # uint8 HxWx3 on device

    def _upload_frame(self, image_rgb: np.ndarray) -> torch.Tensor:
        # Upload as uint8 (3x cheaper than float32) and reuse the buffer.
        if not image_rgb.flags['C_CONTIGUOUS']:
            image_rgb = np.ascontiguousarray(image_rgb)
        H, W, _ = image_rgb.shape
        buf = self._upload_buffer
        if buf is None or buf.shape[0] != H or buf.shape[1] != W:
            buf = torch.empty((H, W, 3), dtype=torch.uint8, device=self.device)
            self._upload_buffer = buf
        buf.copy_(torch.from_numpy(image_rgb), non_blocking=self._is_cuda)
        return buf.permute(2, 0, 1).to(torch.float32)

    def _crop_and_preprocess(self, image_chw: torch.Tensor, target_size_np: np.ndarray,
                             crop_params, image_mean=None):
        out, mean, crop_params = apply_siamfc_cropping(
            image_chw, target_size_np, crop_params,
            self.interpolation_mode, self.interpolation_align_corners,
            image_mean=image_mean)
        out = out.div(255.)
        self.image_normalize_(out)
        return out.unsqueeze(0).to(self.dtype), mean, crop_params

    @torch.inference_mode()
    def init(self, image_rgb: np.ndarray, bbox_xyxy: np.ndarray) -> None:
        image_chw = self._upload_frame(image_rgb)
        z_bbox = np.asarray(bbox_xyxy, dtype=np.float64)
        template_size_np = np.array(self.template_size, dtype=np.int64)

        crop_params = get_siamfc_cropping_params(z_bbox, self.template_area_factor, template_size_np)
        z, z_image_mean, crop_params = self._crop_and_preprocess(image_chw, template_size_np, crop_params)
        self._template = z
        self._template_image_mean = z_image_mean

        stride_w = self.template_size[0] / self.template_feat_size[0]
        stride_h = self.template_size[1] / self.template_feat_size[1]
        cropped = apply_siamfc_cropping_to_boxes(z_bbox, crop_params).copy()
        cropped[0::2] /= stride_w
        cropped[1::2] /= stride_h
        cropped = bbox_rasterize(cropped, dtype=np.int64)
        bbox_clip_to_image_boundary_(cropped, np.array(self.template_feat_size, dtype=np.int64))
        mask = torch.zeros(self.template_feat_size[1], self.template_feat_size[0], dtype=torch.long)
        mask[cropped[1]:cropped[3], cropped[0]:cropped[2]] = 1
        self._z_feat_mask = mask.unsqueeze(0).to(self.device)

        self.cropping_provider = SiamFCCroppingParameterSimpleProvider(
            self.search_area_factor, self.min_object_size)
        self.cropping_provider.initialize(z_bbox)

        self._warmup(image_chw)

    @torch.inference_mode()
    def _warmup(self, image_chw: torch.Tensor) -> None:
        if self.warmup_iters <= 0:
            return
        search_size_np = np.array(self.search_region_size, dtype=np.int64)
        crop_params = self.cropping_provider.get(search_size_np)
        for _ in range(self.warmup_iters):
            x, _, _ = self._crop_and_preprocess(
                image_chw, search_size_np, crop_params,
                image_mean=self._template_image_mean)
            with self._autocast_ctx():
                _ = self.model(z=self._template, x=x, z_feat_mask=self._z_feat_mask)
        if self._is_cuda:
            torch.cuda.synchronize()

    @torch.inference_mode()
    def track(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        if self._template is None:
            raise RuntimeError("LiveTracker.init() must be called before track()")

        image_chw = self._upload_frame(image_rgb)
        H, W = image_chw.shape[-2:]
        image_size = np.array((W, H), dtype=np.int64)

        search_size_np = np.array(self.search_region_size, dtype=np.int64)
        crop_params = self.cropping_provider.get(search_size_np)
        x, _, crop_params = self._crop_and_preprocess(
            image_chw, search_size_np, crop_params,
            image_mean=self._template_image_mean)

        with self._autocast_ctx():
            outputs = self.model(z=self._template, x=x, z_feat_mask=self._z_feat_mask)
        post = self.post_process(outputs)
        score = post['confidence'].detach().cpu().item()
        bbox = post['box'].detach().cpu().to(torch.float64).numpy()[0]

        bbox = apply_siamfc_cropping_to_boxes(bbox, reverse_siamfc_cropping_params(crop_params))
        bbox_clip_to_image_boundary_(bbox, image_size.astype(np.int32))

        self.cropping_provider.update(score, bbox, image_size)
        return bbox, score