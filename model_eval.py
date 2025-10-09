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


def ar_baseline_scores(y, splitter, window=7, rmsle_func=_rmsle):
    """
    Compute RMSLE for two simple autoregressive baselines over a splitter:
      - lag-1: predict previous step's prediction
      - roll-window: predict median of last `window` predictions/observations

    Baselines initialize from the last available training values and never use
    test labels when generating predictions.

    Input:
    - y: Series of labels
    - splitter: cross-validator yielding (train_idx, test_idx)
    - window: int, window size for rolling median
    - rmsle_func: function to compute RMSLE (defaults to `_rmsle`)

    Output:
    - float: mean RMSLE for lag-1 baseline
    - float: mean RMSLE for roll-window baseline
    """
    lag_scores, roll_scores = [], []
    y = y.copy()

    for tr_idx, te_idx in splitter.split(y.to_frame()):
        tr_idx = np.asarray(tr_idx)
        te_idx = np.asarray(te_idx)
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]

        buf_lag = deque(y_tr.tail(1).tolist(), maxlen=window)
        buf_roll = deque(y_tr.tail(window).tolist(), maxlen=window)

        preds_lag, preds_roll = [], []
        for _ in te_idx:
            p_lag = float(buf_lag[-1])
            preds_lag.append(p_lag)
            buf_lag.append(p_lag)

            p_roll = float(np.median(buf_roll))
            preds_roll.append(p_roll)
            buf_roll.append(p_roll)

        lag_scores.append(rmsle_func(y_te, pd.Series(preds_lag, index=y_te.index)))
        roll_scores.append(rmsle_func(y_te, pd.Series(preds_roll, index=y_te.index)))

    return float(np.mean(lag_scores)), float(np.mean(roll_scores))

