"""
Hyperparameter spaces for RandomizedSearchCV, keyed by (model_key, fe_mode, log).

Loaded from `hpt_config.yaml`.

Notes:
- Param names must match the Pipeline structure used in model_eval._build_pipeline:
  - log=False -> estimator is at step 'model'        -> prefix 'model__'
  - log=True  -> estimator wrapped in TTR at 'model' -> prefix 'model__regressor__'
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "Missing dependency: PyYAML. Install with `pip install pyyaml`."
    ) from exc

Space = Dict[str, list]
Key = Tuple[str, str, bool]  # (model_key, fe_mode, log)

HPT_CONFIG_PATH = Path(__file__).with_suffix(".yaml")


def _pfx(log: bool, name: str) -> str:
    return f"model__regressor__{name}" if log else f"model__{name}"


def _load_config(path: Path = HPT_CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"HPT config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("HPT config must be a mapping at the top level.")
    return data


def _merge_params(base: dict, override: dict | None) -> dict:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def _build_spaces(cfg: dict) -> Dict[Key, Space]:
    spaces: Dict[Key, Space] = {}
    fe_modes = cfg.get("fe_modes", ["no_fe", "fe"])
    models_cfg = cfg.get("models", {})

    for model_key, model_cfg in models_cfg.items():
        base_params = model_cfg.get("params", {})
        params_no_log = model_cfg.get("params_no_log", {})
        params_log = model_cfg.get("params_log", {})

        for fe_mode in fe_modes:
            for log in (False, True):
                params = _merge_params(
                    base_params, params_log if log else params_no_log
                )
                space = {_pfx(log, k): v for k, v in params.items()}
                spaces[(model_key, fe_mode, log)] = space

    return spaces


HPT_SPACES: Dict[Key, Space] = _build_spaces(_load_config())


def get_space(model_key: str, fe_mode: str, log: bool) -> Space:
    """Return the parameter space for RandomizedSearchCV for the given tuple."""
    key = (model_key.lower(), fe_mode, bool(log))
    return HPT_SPACES[key]


def has_space(model_key: str, fe_mode: str, log: bool) -> bool:
    return (model_key.lower(), fe_mode, bool(log)) in HPT_SPACES
