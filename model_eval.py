import numpy as np
import pandas as pd
from collections import deque
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from ml_utils import _rmsle, rmsle_scorer, BikeFeatureEngineer
from models import make_model


def _evaluate_no_ar(pipe, X, y, cv, scorer=rmsle_scorer):
    """
    Helper: Evaluate a pipeline without AR features via cross-validation.

    Returns positive RMSLE (mean across folds).
    """
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scorer)
    return float(-scores.mean())  # scorer is negative RMSLE


def _evaluate_ar_walkforward(pipe, X, y, cv, rmsle_func=_rmsle):
    """
    Helper: Walk-forward evaluation for pipelines with AR features.

    Builds test-time AR features using previous test-step predictions to avoid leakage.
    Looks for a feature-engineering step named 'fe' or 'features', and a final 'model' step.
    Returns mean RMSLE across splits.
    """
    scores = []
    for tr_idx, te_idx in cv.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        estimator = clone(pipe).fit(X_tr, y_tr)
        fe_step_name = 'fe' if 'fe' in estimator.named_steps else ('features' if 'features' in estimator.named_steps else None)
        if fe_step_name is None:
            raise ValueError("_evaluate_ar_walkforward requires a 'fe' or 'features' step (BikeFeatureEngineer).")
        fe = estimator.named_steps[fe_step_name]
        model = estimator.named_steps['model']

        preds = []
        max_window = getattr(fe, "_max_ar_window_", 0) or 0
        for k in range(len(X_te)):
            start = max(0, k + 1 - max_window) if max_window > 0 else k
            te_window = X_te.iloc[start:k+1].copy()
            past_len = len(te_window) - 1
            preds_slice = preds[-past_len:] if past_len > 0 else []
            target_proxy = pd.Series(preds_slice + [np.nan], index=te_window.index, dtype='float64')
            te_window['cnt'] = target_proxy.values

            last_row_features = fe.transform(te_window).iloc[[-1]]
            y_hat = model.predict(last_row_features).item()
            preds.append(float(y_hat))

        scores.append(rmsle_func(y_te, pd.Series(preds, index=y_te.index)))
    return float(np.mean(scores))


def ar_baseline_scores(y, splitter, *, lags=None, rolls=None, rmsle_func=_rmsle):
    """
    Compute RMSLE for autoregressive baselines over a splitter.

    Baselines (all step-based):
    - lag-K: predict value from K steps ago, using only predictions to roll forward in test set
    - roll-W: predict median of last W values, using only predictions to roll forward in test set

    Input:
    - `y`: Series of labels
    - `splitter`: cross-validator yielding (train_idx, test_idx)
    - `lags`: list[int] | None (e.g., [1, 24, 168])
    - `rolls`: list[int] | None (e.g., [24, 168])
    - `rmsle_func`: function to compute RMSLE (defaults to `_rmsle`)

    Output:
    - dict[str, float]: mapping like {"lag-1": 0.42, "roll-24": 0.38}
    """
    y = y.copy()
    lag_list = sorted(set(lags)) if lags else []
    roll_list = sorted(set(rolls)) if rolls else []

    # Storage of per-baseline scores across folds (dict-unpack: merge lag/roll baselines into one dict)
    scores = {**{f"lag-{k}": [] for k in lag_list}, **{f"roll-{w}": [] for w in roll_list}}

    for tr_idx, te_idx in splitter.split(y.to_frame()):
        tr_idx, te_idx = np.asarray(tr_idx) ,np.asarray(te_idx)
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        # Initialize buffers for each baseline from the end of training segment
        
        # For each baseline to be evaluated (each K in `lags`, each W in `rolls`)
        # -> create a fixed-length deque populated with last K (or W) training labels.
        # This is the baseline's "state" at test window start: predictions are obtained by "reading K steps back".
        lag_bufs = {k: deque(y_tr.tail(k).tolist(), maxlen=k) for k in lag_list}
        roll_bufs = {w: deque(y_tr.tail(w).tolist(), maxlen=w) for w in roll_list}

        # Prediction holders per baseline
        preds_lag = {k: [] for k in lag_list}
        preds_roll = {w: [] for w in roll_list}

        # Walk forward through test indices, one step at a time
        for _ in te_idx:
            # lag-K: take the oldest value in the length-K buffer (i.e., K steps back)
            for k in lag_list:
                buf = lag_bufs[k]
                p = float(buf[0]) if len(buf) == k and k > 0 else float(buf[-1]) # predict
                preds_lag[k].append(p)
                buf.append(p)
            # roll-W: take median of current buffer
            for w in roll_list:
                buf = roll_bufs[w]
                p = float(np.median(buf)) if len(buf) > 0 else float(y_tr.median()) # predict
                preds_roll[w].append(p)
                buf.append(p)

        # Score each baseline for this fold
        for k in lag_list:
            scores[f"lag-{k}"].append(rmsle_func(y_te, pd.Series(preds_lag[k], index=y_te.index)))
        for w in roll_list:
            scores[f"roll-{w}"].append(rmsle_func(y_te, pd.Series(preds_roll[w], index=y_te.index)))

    # Average across folds
    return {name: float(np.mean(vals)) for name, vals in scores.items()}


def _build_pipeline(model_key: str,
                    estimator,
                    fe_mode: str,
                    *,
                    log: bool,
                    preprocess=None,
                    to_df=None,
                    ar_lags=None,
                    ar_rolls=None) -> Pipeline:
    """
    Build a unified pipeline with a single 'features' step and final 'model' step.

    fe_mode: one of {'no_fe','fe','fe_ar'}
    - 'no_fe': uses provided `preprocess` (ColumnTransformer). Adds `to_df` for LightGBM only.
    - 'fe'   : uses BikeFeatureEngineer() (no AR)
    - 'fe_ar': uses BikeFeatureEngineer(ar_lags, ar_rolls)
    """
    steps = []
    if fe_mode == 'no_fe':
        if preprocess is None:
            raise ValueError("preprocess must be provided when fe_mode='no_fe'")
        steps.append(('features', preprocess))
        if model_key.lower() == 'lgbm' and to_df is not None:
            steps.append(('to_df', to_df))
    elif fe_mode == 'fe':
        steps.append(('features', BikeFeatureEngineer()))
    elif fe_mode == 'fe_ar':
        steps.append(('features', BikeFeatureEngineer(ar_lags=ar_lags, ar_rolls=ar_rolls)))
    else:
        raise ValueError(f"Unknown fe_mode: {fe_mode}")

    final_est = estimator if not log else TransformedTargetRegressor(
        regressor=estimator, func=np.log1p, inverse_func=np.expm1
    )
    steps.append(('model', final_est))
    return Pipeline(steps)


def evaluate_pipeline(models,
                      fe_mode: str,
                      *,
                      X,
                      y,
                      cv_last30,
                      cv_ts_no,
                      cv_ts_ar,
                      preprocess=None,
                      to_df=None,
                      ar_lags=None,
                      ar_rolls=None,
                      logs=(False, True)) -> pd.DataFrame:
    """
    Orchestrate evaluation over a suite of models and log-transform settings.

    - models: list[str] of model keys for `make_model()`
    - fe_mode: 'no_fe' | 'fe' | 'fe_ar'
    - evaluates on Last30 split and the appropriate TS-CV (no-AR -> cv_ts_no, AR -> cv_ts_ar)

    Returns a DataFrame with columns: ['model','fe_mode','log','split','rmsle']
    """
    rows = []
    if fe_mode == 'fe_ar':
        eval_fn = _evaluate_ar_walkforward
        ts_cv = cv_ts_ar
    else:
        eval_fn = _evaluate_no_ar
        ts_cv = cv_ts_no

    for key in models:
        est = make_model(key)
        for log in logs:
            pipe = _build_pipeline(
                key, est, fe_mode,
                log=log,
                preprocess=preprocess,
                to_df=to_df,
                ar_lags=ar_lags,
                ar_rolls=ar_rolls,
            )
            r_last30 = eval_fn(pipe, X, y, cv_last30)
            r_ts = eval_fn(pipe, X, y, ts_cv)
            rows.append({'model': key, 'fe_mode': fe_mode, 'log': log, 'split': 'last30', 'rmsle': r_last30})
            rows.append({'model': key, 'fe_mode': fe_mode, 'log': log, 'split': 'ts', 'rmsle': r_ts})

    return pd.DataFrame(rows)
