"""Bundled, read-only resources shipped with the wheel.

The original repo kept ``config/`` and ``consts.yaml.template`` next to the
top-level scripts and discovered them by walking up from ``__file__``. That
breaks the moment the package is ``pip install``ed and the repo root no
longer exists. Everything that used to live there now lives here, so the
package can find its defaults regardless of how it was installed.

External overrides (env vars, explicit paths) are still supported by the
callers in ``trackit.core.runtime.global_constant`` and the CLI scripts.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def get_default_config_root() -> str:
    """Absolute path to the bundled ``config/`` tree."""
    return os.path.join(_HERE, 'config')


def get_default_consts_path() -> str:
    """Absolute path to the bundled ``consts.yaml.template`` file."""
    return os.path.join(_HERE, 'consts.yaml.template')