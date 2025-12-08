"""
Hyperparameter spaces for RandomizedSearchCV, keyed by (model_key, fe_mode, log).

Notes
- Param names must match the Pipeline structure used in model_eval._build_pipeline:
  - log=False -> estimator is at step 'model'        -> prefix 'model__'
  - log=True  -> estimator wrapped in TTR at 'model' -> prefix 'model__regressor__'
- Define small, high-yield spaces; expand only if needed.
"""

from __future__ import annotations

from typing import Dict, Tuple


Space = Dict[str, list]
Key = Tuple[str, str, bool]  # (model_key, fe_mode, log)


def _pfx(log: bool, name: str) -> str:
    return f"model__regressor__{name}" if log else f"model__{name}"


def _rf_space(log: bool) -> Space:
    return {
        _pfx(log, "n_estimators"): [300, 500],
        _pfx(log, "min_samples_leaf"): [1, 2, 5],
        _pfx(log, "max_features"): ["sqrt", 0.7],
        _pfx(log, "max_depth"): [None, 12],
    }


def _hgbr_space(log: bool) -> Space:
    return {
        _pfx(log, "learning_rate"): [0.05, 0.1],
        _pfx(log, "max_iter"): [500, 800, 1000],
        _pfx(log, "max_leaf_nodes"): [31, 63],
        _pfx(log, "min_samples_leaf"): [10, 20],
        _pfx(log, "l2_regularization"): [0.0, 0.5, 1.0],
    }


def _gbr_space(log: bool) -> Space:
    return {
        _pfx(log, "n_estimators"): [300, 500],
        _pfx(log, "learning_rate"): [0.05, 0.1],
        _pfx(log, "max_depth"): [3, 4],
        _pfx(log, "alpha"): [0.85, 0.9],
        _pfx(log, "subsample"): [0.7, 0.9],
    }


def _xgb_space(log: bool) -> Space:
    return {
        _pfx(log, "max_depth"): [4, 6],
        _pfx(log, "min_child_weight"): [1, 5],
        _pfx(log, "subsample"): [0.7, 0.9],
        _pfx(log, "colsample_bytree"): [0.7, 0.9],
        _pfx(log, "reg_lambda"): [0, 1],
        _pfx(log, "learning_rate"): [0.05, 0.1],
        _pfx(log, "n_estimators"): [400, 800],
    }


def _cbr_space(log: bool) -> Space:
    return {
        _pfx(log, "iterations"): [600, 800],
        _pfx(log, "learning_rate"): [0.03, 0.07],
        _pfx(log, "depth"): [6, 8],
        _pfx(log, "l2_leaf_reg"): [3, 10],
        # Optional sampling
        # _pfx(log, "bootstrap_type"): ["Bernoulli"],
        # _pfx(log, "subsample"): [0.7, 0.9],
    }


def _lgbm_space(log: bool, tweedie: bool = False) -> Space:
    space = {
        _pfx(log, "num_leaves"): [31, 63, 127],
        _pfx(log, "min_child_samples"): [5, 10, 20],
        _pfx(log, "feature_fraction"): [0.7, 0.9],
        _pfx(log, "bagging_fraction"): [0.7, 0.9],
        _pfx(log, "learning_rate"): [0.05, 0.1],
        _pfx(log, "n_estimators"): [600, 1000],
    }
    if tweedie:
        space[_pfx(log, "tweedie_variance_power")] = [1.2, 1.5]
    return space


# Master mapping: (model_key, fe_mode, log) -> param space
HPT_SPACES: Dict[Key, Space] = {}

for fe_mode in ("no_fe", "fe"):
    # Random Forest
    HPT_SPACES[("rf", fe_mode, False)] = _rf_space(log=False)
    HPT_SPACES[("rf", fe_mode, True)] = _rf_space(log=True)

    # HGBR (raw typically Poisson; log typically squared_error)
    HPT_SPACES[("hgbr", fe_mode, False)] = _hgbr_space(log=False)
    HPT_SPACES[("hgbr", fe_mode, True)] = _hgbr_space(log=True)

    # GBR (use with log=True typically)
    HPT_SPACES[("gbr", fe_mode, False)] = _gbr_space(log=False)
    HPT_SPACES[("gbr", fe_mode, True)] = _gbr_space(log=True)

    # XGB
    HPT_SPACES[("xgb", fe_mode, False)] = _xgb_space(log=False)
    HPT_SPACES[("xgb", fe_mode, True)] = _xgb_space(log=True)

    # CatBoost
    HPT_SPACES[("cbr", fe_mode, False)] = _cbr_space(log=False)
    HPT_SPACES[("cbr", fe_mode, True)] = _cbr_space(log=True)

    # LightGBM (provide both poisson/tweedie options via two keys if desired)
    HPT_SPACES[("lgbm", fe_mode, False)] = _lgbm_space(log=False, tweedie=True)
    HPT_SPACES[("lgbm", fe_mode, True)] = _lgbm_space(log=True, tweedie=False)


def get_space(model_key: str, fe_mode: str, log: bool) -> Space:
    """Return the parameter space for RandomizedSearchCV for the given tuple.

    Raises KeyError if not found.
    """
    key = (model_key.lower(), fe_mode, bool(log))
    return HPT_SPACES[key]


def has_space(model_key: str, fe_mode: str, log: bool) -> bool:
    return (model_key.lower(), fe_mode, bool(log)) in HPT_SPACES

