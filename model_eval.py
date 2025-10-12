import numpy as np
import pandas as pd
from collections import deque
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from ml_utils import _rmsle, rmsle_scorer


def eval_pipeline(pipe, X, y, cv, scorer=rmsle_scorer):
    """
    Evaluate a pipeline without autoregressive (AR) features, via cross-validation, using a scorer (default: RMSLE).

    Input:
    - `pipe`: sklearn Pipeline/estimator to evaluate
    - `X`: features (DataFrame or array-like)
    - `y`: target values (Series/array-like)
    - `cv`: cross-validator yielding train/test splits
    - `scorer`: sklearn scorer (defaults to `rmsle_scorer` which returns negative values)

    Output:
    - float: mean score across folds. Return positive value for RMSLE
    """
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scorer)
    # Our default scorer returns negative RMSLE: flip sign so smaller is better but positive
    return float(-scores.mean())


def eval_pipeline_walkforward(pipe, X, y, cv, rmsle_func=_rmsle):
    """
    Walk-forward evaluation for pipelines with autoregressive (AR) features, via cross validation, using a scoroes (default: RMSLE)

    - Builds test-time AR features using previous test-step predictions, not true labels
      -> mirrors deployment + prevents look-ahead leakage.
    - Requires the pipeline to expose/have the following steps:
        - `fe`: a feature engineering transformer (e.g., `BikeFeatureEngineer`)
        - `model`: the final estimator used for prediction

    Input:
    - `pipe`: sklearn Pipeline with steps named 'fe' and 'model'
    - `X`: DataFrame of features including 'cnt' used only to construct AR features
    - `y`: Series of labels aligned with X
    - `cv`: splitter yielding train/test indices in time order
    - `rmsle_func`: function to compute RMSLE (defaults to `_rmsle`)

    Output:
    - float: mean RMSLE across splits
    """
    scores = []
    for tr_idx, te_idx in cv.split(X): # select train and test set based on CV splits
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        estimator = clone(pipe).fit(X_tr, y_tr)
        if 'fe' not in estimator.named_steps:
            raise ValueError("eval_pipeline_walkforward requires a 'fe' step (BikeFeatureEngineer).")
        fe = estimator.named_steps['fe']
        model = estimator.named_steps['model']

        # AR features depend on past true labels `cnt` to compute lags/rolls. In production, `cnt` is unavailable, so can only use prev. preds.
        # -> so instead of using `cnt` to compute AR features (which would be cheating) use curr. preds as "proxy labels" (`target_proxy`)
        preds = []
        # Optimization: use only a sliding window up to max FE-AR span
        max_window = getattr(fe, "_max_ar_window_", 0) or 0
        for k in range(len(X_te)):
            start = max(0, k + 1 - max_window) if max_window > 0 else k # Only info from last `max_window` row (when AR is used)...
            te_window = X_te.iloc[start:k+1].copy() # ...up to and including current step
            # use prev. preds (restricted to window) + set current step to NaN (AR features use shift(k) with k≥1)
            past_len = len(te_window) - 1
            preds_slice = preds[-past_len:] if past_len > 0 else []
            target_proxy = pd.Series(preds_slice + [np.nan], index=te_window.index, dtype='float64')
            te_window['cnt'] = target_proxy.values # enables AR feature construction without leakage

            last_row_features = fe.transform(te_window).iloc[[-1]] # engineer features for current step (which is the last row `iloc[-1]`)
            y_hat = model.predict(last_row_features).item() # prediction
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
