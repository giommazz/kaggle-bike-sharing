"""
Model factory helpers.

Provides a single `make_model()` function that returns a preconfigured regressor by name.
Uses lazy imports so optional dependencies (xgboost, catboost, lightgbm) are only imported when requested
"""

from typing import Any


def make_model(name: str) -> Any:
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

    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=50,
            min_samples_leaf=2,
            criterion='squared_error',# 'friedman_mse', 'poisson'
            random_state=42,
        )
    if name == "hgbr":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            loss="poisson", # good for counts, emphasizes relative errors
            learning_rate=0.075,
            max_iter=750,
            early_stopping=True,
            random_state=42,
        )
    if name == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            loss="huber", # smooth and robust to spikes/outliers
            alpha=0.85, # outlier sensitivity (.85-.95 typical)
            learning_rate=0.05,
            n_estimators=500,
            max_depth=3,
            random_state=42,
        )
    if name == "xgb": # Raw: objective='count:poisson', n_estimators=400–800, learning_rate=0.05, max_depth=4–6, subsample=0.7–0.8, colsample_bytree=0.7–0.8, min_child_weight=1–5, reg_lambda=0–1
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='count:poisson', #'reg:squarederror',
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
        )
    if name == "cbr": # Raw: loss_function='Poisson' (for counts), iterations=600–800, learning_rate=0.05–0.1, depth=6–8, l2_leaf_reg=3–10.
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            loss_function='Poisson',#'RMSE',
            random_seed=42,
            verbose=False,
        )
    if name == "lgbm": # Raw: objective='poisson' (or 'tweedie' with tweedie_variance_power≈1.3–1.5), n_estimators=600–1000, learning_rate=0.05–0.1, num_leaves=31–63, min_child_samples=5–20, subsample=0.7–0.9, colsample_bytree=0.7–0.9.
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63, # more leaf capacity
            min_child_samples=10, # allow smaller leaves
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='rmse', # 'rmse', 'poisson', 'tweedie'
            force_col_wise=True, # remove col/row test overhead message
            verbosity=-1, # silence LightGBM logs
            random_state=42,
        )

    raise ValueError(f"Unknown model '{name}'")

