"""
Model factory helpers.

Provides a single `make_model()` function that returns a preconfigured regressor by name.
Uses lazy imports so optional dependencies (xgboost, catboost, lightgbm) are only imported when requested

Seeding: override the global default by setting env var `MODEL_SEED`, or
pass `seed=` to `make_model()` for one-off experiments.
"""

import os
from typing import Any

# Global default seed (env override: MODEL_SEED)
SEED = int(os.getenv("MODEL_SEED", "5674381"))


def make_model(name: str, seed: int | None = None) -> Any:
    """
    Return a preconfigured regressor by name.

    Supported names (with lazy imports):
    - 'rf'   : sklearn `RandomForestRegressor`
    - 'hgbr' : sklearn `HistGradientBoostingRegressor`
    - 'gbr'  : sklearn `GradientBoostingRegressor`
    - 'xgb'  : `XGBoost XGBRegressor` (optional dependency)
    - 'cbr'  : `CatBoostRegressor` (optional dependency)
    - 'lgbm' : LightGBM `LGBMRegressor` (optional dependency)
    """
    name = name.lower()
    seed = SEED if seed is None else int(seed)

    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=50,
            min_samples_leaf=2,
            criterion='squared_error', # 'squared_error', 'friedman_mse', 'poisson'
            random_state=seed,
        )
    if name == "hgbr":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            loss='poisson', # 'poisson', 'squared_error' (may give negative predictions)
            learning_rate=0.075,
            max_iter=750,
            early_stopping=True,
            random_state=seed,
        )
    if name == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            loss='huber', # smooth and robust to spikes/outliers. Losses: 'squared_error', 'absolute_error', 'huber', 'quantile'. MSE can give neg. pred.
            alpha=0.85, # outlier sensitivity (.85-.95 typical)
            learning_rate=0.05,
            n_estimators=50,
            max_depth=3,
            random_state=seed,
        )
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:tweedie', #'reg:squarederror', 'count:poisson', 'reg:tweedie'
            tree_method='hist',
            random_state=seed,
            n_jobs=-1,
        )
    if name == "cbr":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=50,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',#'RMSE', 'Poisson'
            random_seed=seed,
            verbose=False,
        )
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=50,
            learning_rate=0.05,
            num_leaves=63, # more leaf capacity
            min_child_samples=10, # allow smaller leaves
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='tweedie', # 'rmse', 'poisson', 'tweedie', 'huber
            force_col_wise=True, # remove col/row test overhead message
            verbosity=-1, # silence LightGBM logs
            random_state=seed,
        )

    raise ValueError(f"Unknown model '{name}'")
