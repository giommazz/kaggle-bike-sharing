"""
Hyperparameter tuning on the Last-30-days holdout.

Usage (examples):
  python tune_last30.py --model lgbm --fe-mode fe --log true --n-iter 25

Notes
- Inner CV uses a TimeSeriesSplit on the train portion (everything except the
  last 30 samples). Final score is computed on the Last30 holdout.
- Parameter spaces are pulled from `hpt_config.get_space(model, fe_mode, log)`.
- When log=True the pipeline wraps the estimator in `TransformedTargetRegressor`,
  so parameter names must be prefixed with `model__regressor__...`.
"""

from __future__ import annotations

import argparse
import json
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from eda_utils import tukey_outliers
from hpt_config import get_space, has_space
from ml_utils import (
    make_timeseries_split,
    Last30DaysSplit,
    make_no_fe_preprocess,
    BikeFeatureEngineer,
    rmsle_scorer,
)
from models import make_model, SEED


def load_hour_data() -> pd.DataFrame:
    df = pd.read_csv("data/hour.csv")
    # Minimal cleaning to match main script
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.set_index("timestamp").sort_index()
    df = df.drop(columns=["instant", "dteday", "casual", "registered"])  # avoid leakage
    # Fix impossible humidity zeros (carry from neighbors)
    df.loc[df["hum"] == 0, "hum"] = np.nan
    df["hum"] = df["hum"].ffill().bfill()
    # Hourly outlier flag (per-hour Tukey, permissive)
    flags = []
    for hour, g in df.groupby("hr"):
        m = tukey_outliers(g["cnt"], 20, 80, 2.0)
        flags.append(pd.Series(m, index=g.index))
    out = pd.concat(flags).reindex(df.index).fillna(False)
    df["hr_cnt_outlier"] = out.astype(int)
    return df


def build_pipeline(model_key: str, fe_mode: str, log: bool, *, X_full: pd.DataFrame) -> Pipeline:
    est = make_model(model_key)
    steps = []
    if fe_mode == "no_fe":
        preprocess, to_df, _ = make_no_fe_preprocess(X_full, target_col="cnt")
        steps.append(("features", preprocess))
        # Keep names for LGBM downstream (as in the main script)
        if model_key == "lgbm" and to_df is not None:
            steps.append(("to_df", to_df))
    elif fe_mode == "fe":
        steps.append(("features", BikeFeatureEngineer()))
    else:
        raise ValueError(f"Unsupported fe_mode '{fe_mode}'. Use 'no_fe' or 'fe'.")

    final_est = est if not log else TransformedTargetRegressor(
        regressor=est, func=np.log1p, inverse_func=np.expm1
    )
    steps.append(("model", final_est))
    return Pipeline(steps)


def split_last30(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    splitter = Last30DaysSplit()
    tr_idx, te_idx = next(splitter.split(X))
    return tr_idx, te_idx


def run_tuning(model_key: str,
               fe_mode: str,
               log: bool,
               *,
               n_iter: int = 25,
               cv_splits: int = 3,
               cv_test_size: int = 30,
               n_jobs: int = -1,
               seed: int = SEED,
               save: str | None = None) -> Dict[str, Any]:
    """
    Run RandomizedSearchCV on the train portion (Last30 holdout) with an inner
    TimeSeriesSplit, then refit best on full train and evaluate on Last30.

    Returns a dict with best_params, CV score, and holdout score, and prints a
    concise report to stdout. Optionally saves JSON to `save` path.
    """
    model_key = model_key.lower()
    if not has_space(model_key, fe_mode, log):
        raise KeyError(f"No HPT space for ({model_key}, {fe_mode}, log={log}).")
    space = get_space(model_key, fe_mode, log)

    # Load and split data
    df = load_hour_data()
    X = df.copy()
    y = X["cnt"].copy()
    tr_idx, te_idx = split_last30(X)
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

    pipe = build_pipeline(model_key, fe_mode, log, X_full=X_tr)
    inner_cv = make_timeseries_split(n_splits=int(cv_splits), test_size=int(cv_test_size))

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=space,
        n_iter=int(n_iter),
        scoring=rmsle_scorer,
        cv=inner_cv,
        random_state=int(seed),
        n_jobs=int(n_jobs),
        verbose=1,
        refit=True,
        error_score=np.nan,
    )
    search.fit(X_tr, y_tr)

    best_cv_rmsle = float(-search.best_score_)  # scorer is negative
    best_params = search.best_params_

    # Evaluate on Last30 holdout
    best = search.best_estimator_
    y_hat = best.predict(X_te)
    from ml_utils import _rmsle  # safe import here
    last30_rmsle = float(_rmsle(y_te, y_hat))

    # Print concise report
    print("\n=== RandomizedSearchCV Results (inner CV) ===")
    print(f"Best RMSLE (mean CV): {best_cv_rmsle:.6f}")
    print("Best params:")
    print(json.dumps(best_params, indent=2))
    print(f"\n=== Last30 holdout RMSLE ===\n{last30_rmsle:.6f}")

    result = {
        "model": model_key,
        "fe_mode": fe_mode,
        "log": bool(log),
        "best_params": best_params,
        "best_cv_rmsle": best_cv_rmsle,
        "last30_rmsle": last30_rmsle,
        "n_iter": int(n_iter),
        "cv_splits": int(cv_splits),
        "cv_test_size": int(cv_test_size),
        "seed": int(seed),
    }

    if save:
        with open(save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved best config to {save}")

    return result


def main():
    ap = argparse.ArgumentParser(description="Hyperparameter tuning on Last-30 holdout")
    ap.add_argument("--model", required=True, choices=["rf", "hgbr", "gbr", "xgb", "cbr", "lgbm"], help="model key")
    ap.add_argument("--fe-mode", required=True, choices=["no_fe", "fe"], help="feature mode")
    ap.add_argument("--log", required=True, type=str, choices=["true", "false"], help="use log-transform (true/false)")
    ap.add_argument("--n-iter", type=int, default=25, help="RandomizedSearchCV iterations")
    ap.add_argument("--cv-splits", type=int, default=3, help="inner TimeSeriesSplit folds on train")
    ap.add_argument("--cv-test-size", type=int, default=30, help="inner TSS test size")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel jobs where supported")
    ap.add_argument("--seed", type=int, default=SEED, help="random seed for the search")
    ap.add_argument("--save", type=str, default=None, help="optional path to save best params as JSON")
    args = ap.parse_args()
    run_tuning(
        args.model,
        args.fe_mode,
        args.log.lower() == "true",
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        cv_test_size=args.cv_test_size,
        n_jobs=args.n_jobs,
        seed=args.seed,
        save=args.save,
    )


if __name__ == "__main__":
    main()
