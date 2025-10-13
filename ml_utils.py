# ml_utils.py
import numpy as np 
import pandas as pd
import inspect
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


##########################################
# EVALUATION SCORERS
##########################################
def _rmsle(y_true, y_pred):
    """
    Compute root mean squared log error (RMSLE) between `y_true` and `y_pred`.
    """
    # Compute RMSLE using sklearn's `mean_squared_log_error`
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Create `rmsle_scorer` for use in sklearn model evaluation (`False` because lower is better)
rmsle_scorer = make_scorer(_rmsle, greater_is_better=False)


##########################################
# CUSTOM SPLITTERS
##########################################
def make_timeseries_split(n_splits=5, test_size=30, *, lags=None, rolls=None):
    """
    Create `TimeSeriesSplit` with `gap` (in steps) sized to largest AR dependency supplied.

    Behavior:
    - If `lags` and/or `rolls` (lists of step sizes) are provided, `gap` is
      set to `max(max(lags or [0]), max(rolls or [0]))`.
    - If both are `None` or empty, `gap` is `0`.

    Input:
    - `n_splits`: int, number of splits
    - `test_size`: int, size of each test fold
    - `lags`: list[int] | `None`, explicit lag steps
    - `rolls`: list[int] | `None`, explicit rolling window steps

    Output:
    - `TimeSeriesSplit` configured with the computed `gap`
    """
    max_lag = max(lags) if lags else 0
    max_roll = max(rolls) if rolls else 0
    gap = int(max(max_lag, max_roll))
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

class Last30DaysSplit(BaseCrossValidator):
    """
    Single split: train on rows [0 ... N-31], test on rows [N-30 ... N-1].
    Assumes `X`,`y` are already time-ordered from oldest to newest.
    """

    # Determine number of many pairs `split()` yields
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    # Generator
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        if n_samples < 31:
            raise ValueError('Need at least 31 rows for a 30-day hold-out.')

        split_point = n_samples - 30  # first index in the test block
        train_idx = np.arange(0, split_point)  # all integers btw 0 and `split_point`
        test_idx = np.arange(split_point, n_samples)

        yield train_idx, test_idx


##########################################
# FEATURE ENGINEERING
##########################################
# Helper for one-hot encoding, implements `pd.get_dummies(..., drop_first=True)` and handles compatibility issues
def onehot_no_sparse():
    """
    Return `OneHotEncoder` with dense output, compatible with all sklearn versions.
    """
    # Grab signature of sklearn's `OneHotEncoder` constructor and pull its
    #   arguments, to check if newer `sparse_output` parameter exists
    params = inspect.signature(OneHotEncoder).parameters
    if 'sparse_output' in params: # sklearn >= 1.2
        # Drop first value to avoid collinearity, sparse bc few values and small dataset
        return OneHotEncoder(drop='first', sparse_output=False, dtype=np.float32)
    return OneHotEncoder(drop='first', sparse=False, dtype=np.float32) # sklearn < 1.2

class BikeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer:
    - drops `atemp` (highly correlated with `temp`)
    - one-hot encodes `weathersit`
    - adds sin/cos pairs for `mnth`, `season`, `weekday`, and optionally `hr`; then drops originals
    - optionally adds autoregressive (AR) features from past data and then drops `cnt`

    AR configuration:
    - Provide `ar_lags` and/or `ar_rolls` (lists of steps). If both None/empty -> no AR features added.

    Returns a `DataFrame` with preserved column names.
    
    Sklearn compliance:
    - __init__ only stores parameters as provided (no mutation), so the estimator is clone-safe.
    - learned attributes are created in `fit` and suffixed with `_`.
    """
    def __init__(self, ar_lags=None, ar_rolls=None):
        # store params as-is; do not convert to list here (clone-safety)
        self.ar_lags = ar_lags
        self.ar_rolls = ar_rolls

    # Use only training data, nothing to learn except one-hot encoder
    def fit(self, X, y=None):
        X_ = X.copy()
        # encoder learned on train; stored with trailing underscore
        self.weather_cols_ = ['weathersit']
        self.ohe_ = onehot_no_sparse()
        self.ohe_.fit(X_[self.weather_cols_])
        # train-only stats/state for AR features and transforms
        self.cnt_median_ = X_['cnt'].median()
        self._last_seen_train_time_ = X_.index.max()
        # determine max AR dependency window length
        lags = list(self.ar_lags) if self.ar_lags is not None else []
        rolls = list(self.ar_rolls) if self.ar_rolls is not None else []
        max_lag = max(lags) if lags else 0
        max_roll = max(rolls) if rolls else 0
        self._max_ar_window_ = int(max(max_lag, max_roll))
        # carry last `cnt` values from train to bootstrap test-time AR features
        self._carry_ = (
            X_['cnt'].tail(self._max_ar_window_).to_numpy()
            if self._max_ar_window_ > 0 else np.array([], dtype=float)
        )
        return self

    # Feature engineering
    def transform(self, X):
        X_ = X.copy()

        # 1) Drop highly-correlated `atemp`
        if 'atemp' in X_.columns: 
            X_ = X_.drop(columns='atemp')

        # 2) One-hot encode weather
        weather_ohe = self.ohe_.transform(X_[self.weather_cols_])
        X_.drop(columns=self.weather_cols_, inplace=True) # Drop og column, not needed anymore
        # Fetch colnames that the fitted OHE will output for the encoded weather feature
        X_[self.ohe_.get_feature_names_out(self.weather_cols_)] = weather_ohe # add one-hot matrix to `X_`

        # 3) Cyclic calendar features
        X_['mnth_sin'] = np.sin(2*np.pi*X_['mnth'] / 12)
        X_['mnth_cos'] = np.cos(2*np.pi*X_['mnth'] / 12)
        X_['season_sin'] = np.sin(2*np.pi*X_['season'] / 4)
        X_['season_cos'] = np.cos(2*np.pi*X_['season'] / 4)
        X_['weekday_sin'] = np.sin(2*np.pi*X_['weekday']/ 7)
        X_['weekday_cos'] = np.cos(2*np.pi*X_['weekday']/ 7)
        if 'hr' in X_.columns:
            X_['hr_sin'] = np.sin(2*np.pi*X_['hr'] / 24)
            X_['hr_cos'] = np.cos(2*np.pi*X_['hr'] / 24)
        X_.drop(columns=['mnth', 'season', 'weekday', 'hr'], inplace=True, errors='ignore')

        # 4) Autoregressive features (optional)
        lags = list(self.ar_lags) if self.ar_lags is not None else []
        rolls = list(self.ar_rolls) if self.ar_rolls is not None else []
        want_ar = (len(lags) > 0) or (len(rolls) > 0)
        if want_ar:
            # Set `use_carry=True` if processing test data (all indices after `_last_seen_train_time_`)
            use_carry = X.index.min() > self._last_seen_train_time_
            if use_carry: # building AR features on test/holdout
                feats = self._build_ar_general_test(X)
            else: # building AR features on train
                feats = self._build_ar_general_train(X)
            for k, v in feats.items():
                X_[k] = v

        # Drop current-day `cnt` to avoid leakage
        if 'cnt' in X_.columns:
            X_.drop(columns='cnt', inplace=True)
        return X_

    def _build_ar_general_train(self, X):
        """
        Build AR features for train data with arbitrary `lags` and rolling windows `rolls`.

        Returns dict: `{feature_name: Series}`
        """
        feats = {}
        lags = self.ar_lags if self.ar_lags is not None else []
        rolls = self.ar_rolls if self.ar_rolls is not None else []
        if lags:
            for lag in lags:
                feature_series = X['cnt'].shift(int(lag)).fillna(self.cnt_median_)
                feats[f'cnt_lag{lag}'] = feature_series
        if rolls:
            shifted_cnt = X['cnt'].shift(1)
            for window in rolls:
                feature_series = shifted_cnt.rolling(int(window), min_periods=1).median().fillna(self.cnt_median_)
                feats[f'cnt_roll{window}'] = feature_series
        return feats

    def _build_ar_general_test(self, X):
        """
        Build AR features for test data with arbitrary `lags` and rolling windows `rolls` using carryover from training.

        Returns dict: `{feature_name: np.ndarray}`
        """
        feats = {}
        lags = (self.ar_lags if self.ar_lags is not None else [])
        rolls = (self.ar_rolls if self.ar_rolls is not None else [])
        if (not lags) and (not rolls):
            return feats
        # concat train carry and test `cnt` as one history
        series = pd.Series(np.r_[self._carry_, X['cnt'].to_numpy()], index=None)
        off = len(self._carry_)  # offset to drop the prepended carry when slicing back to test rows
        if lags:
            for lag in lags:
                shifted_series = series.shift(int(lag)) # produce lagged view over carry+test history
                # align to test rows, fill edge NaNs
                feats[f'cnt_lag{lag}'] = shifted_series.iloc[off:].fillna(self.cnt_median_).to_numpy()
        if rolls:
            shifted_series = series.shift(1) # exclude current step from rolling window
            for window in rolls:
                rolled_series = shifted_series.rolling(int(window), min_periods=1).median()
                # drop carry part, fill edge NaNs
                feats[f'cnt_roll{window}'] = rolled_series.iloc[off:].fillna(self.cnt_median_).to_numpy()
        return feats


##########################################
# PREPROCESSING helpers (no feature engineering)
##########################################
def make_no_fe_preprocess(X, target_col='cnt'):
    """
    Build a passthrough preprocessor (no feature engineering): features are passed straight through
        as-is using `ColumnTransformer`.

    Input:
    - `X`: `DataFrame` to derive feature columns from
    - `target_col`: column to exclude (default `cnt`)

    Output:
    - `preprocess`: `ColumnTransformer` that passes features through unchanged
    - `to_df`: `FunctionTransformer` that wraps array output back into a `DataFrame`
              with original feature names (handy for LightGBM to keep names)
    - `feat_cols`: list of feature column names
    """
    feat_cols = [c for c in X.columns if c != target_col]  # all columns but `cnt`
    preprocess = ColumnTransformer([('keep_all', 'passthrough', feat_cols)], remainder='drop')  # "Preprocess-only" block:  no FE
    # Convert array output back to DataFrame with original `feat_cols` names
    # -> only needed for `lightgbm` to preserve feature names and avoid warning
    to_df = FunctionTransformer(
        lambda A: pd.DataFrame(A, columns=feat_cols),
        feature_names_out=lambda self, input_features=None: np.array(feat_cols),
    )
    return preprocess, to_df, feat_cols
