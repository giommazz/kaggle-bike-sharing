import numpy as np
import pandas as pd
from collections import deque
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from ml_utils import _rmsle, rmsle_scorer


def eval_pipeline(pipe, X, y, cv, scorer=rmsle_scorer):
    """
    Evaluate a pipeline via cross-validation using a scorer (default: RMSLE).

    Input:
    - pipe: sklearn Pipeline/estimator to evaluate
    - X: features (DataFrame or array-like)
    - y: target values (Series/array-like)
    - cv: cross-validator yielding train/test splits
    - scorer: sklearn scorer (defaults to `rmsle_scorer` which returns negative values)

    Output:
    - float: mean score across folds; for RMSLE returns positive value
    """
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scorer)
    # Our default scorer returns negative RMSLE; flip sign so smaller is better but positive
    return float(-scores.mean())


def eval_pipeline_walkforward(pipe, X, y, cv, rmsle_func=_rmsle):
    """
    Walk-forward evaluation for pipelines with autoregressive (AR) features.

    - Builds test-time AR features using previous test-step predictions, not true labels,
      mirroring deployment and preventing look-ahead leakage.
    - Requires the pipeline to expose:
        - step `fe`: a feature engineering transformer (e.g., `BikeFeatureEngineer`)
        - step `model`: the final estimator used for prediction

    Input:
    - pipe: sklearn Pipeline with steps named 'fe' and 'model'
    - X: DataFrame of features including 'cnt' used only to construct AR features
    - y: Series of labels aligned with X
    - cv: splitter yielding train/test indices in time order
    - rmsle_func: function to compute RMSLE (defaults to `_rmsle`)

    Output:
    - float: mean RMSLE across splits
    """
    scores = []
    for tr_idx, te_idx in cv.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        estimator = clone(pipe).fit(X_tr, y_tr)
        if 'fe' not in estimator.named_steps:
            raise ValueError("eval_pipeline_walkforward requires a 'fe' step (BikeFeatureEngineer).")
        fe = estimator.named_steps['fe']
        model = estimator.named_steps['model']

        preds = []
        for k in range(len(X_te)):
            te_prefix = X_te.iloc[:k+1].copy()
            cnt_proxy = pd.Series(preds + [np.nan], index=te_prefix.index, dtype='float64')
            te_prefix['cnt'] = cnt_proxy.values

            last_row_features = fe.transform(te_prefix).iloc[[-1]]
            y_hat = model.predict(last_row_features).item()
            preds.append(float(y_hat))

        scores.append(rmsle_func(y_te, pd.Series(preds, index=y_te.index)))
    return float(np.mean(scores))


def ar_baseline_scores(y, splitter, *, lags=None, rolls=None, rmsle_func=_rmsle):
    """
    Compute RMSLE for autoregressive baselines over a splitter.

    Baselines (all step-based):
    - lag-K: predict value from K steps ago, using only predictions to roll forward
    - roll-W: predict median of last W values, using only predictions to roll forward

    Input:
    - y: Series of labels
    - splitter: cross-validator yielding (train_idx, test_idx)
    - lags: list[int] | None (e.g., [1, 24, 168])
    - rolls: list[int] | None (e.g., [24, 168])
    - rmsle_func: function to compute RMSLE (defaults to `_rmsle`)

    Output:
    - dict[str, float]: mapping like {"lag-1": 0.42, "roll-24": 0.38}
    """
    y = y.copy()
    lag_list = sorted(set(lags)) if lags else []
    roll_list = sorted(set(rolls)) if rolls else []

    # Storage of per-baseline scores across folds
    scores = {**{f"lag-{k}": [] for k in lag_list}, **{f"roll-{w}": [] for w in roll_list}}

    for tr_idx, te_idx in splitter.split(y.to_frame()):
        tr_idx = np.asarray(tr_idx)
        te_idx = np.asarray(te_idx)
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]

        # Initialize buffers for each baseline from the end of training segment
        lag_bufs = {k: deque(y_tr.tail(k).tolist(), maxlen=k) for k in lag_list}
        roll_bufs = {w: deque(y_tr.tail(w).tolist(), maxlen=w) for w in roll_list}

        # Prediction holders per baseline
        preds_lag = {k: [] for k in lag_list}
        preds_roll = {w: [] for w in roll_list}

        # Walk forward through test indices, one step at a time
        for _ in te_idx:
            # lag-K: take the oldest value in the length-K buffer (i.e., K steps back), then append prediction
            for k in lag_list:
                buf = lag_bufs[k]
                p = float(buf[0]) if len(buf) == k and k > 0 else float(buf[-1])
                preds_lag[k].append(p)
                buf.append(p)
            # roll-W: take median of current buffer, then append prediction
            for w in roll_list:
                buf = roll_bufs[w]
                p = float(np.median(buf)) if len(buf) > 0 else float(y_tr.median())
                preds_roll[w].append(p)
                buf.append(p)

        # Score each baseline for this fold
        for k in lag_list:
            scores[f"lag-{k}"].append(rmsle_func(y_te, pd.Series(preds_lag[k], index=y_te.index)))
        for w in roll_list:
            scores[f"roll-{w}"].append(rmsle_func(y_te, pd.Series(preds_roll[w], index=y_te.index)))

    # Average across folds
    return {name: float(np.mean(vals)) for name, vals in scores.items()}
