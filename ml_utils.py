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
def make_timeseries_split(add_lag1: bool, add_roll7: bool, n_splits=5, test_size=30):
    """
    Create `TimeSeriesSplit` with correct `gap` (in steps) to prevent data leakage when using autoregressive features.
    -> keeps order of data, splits train/test sets sequentially to avoid data leakage from future to past

    # Example: with n_samples=100, n_splits=3, test_size=10, gap=7
    # Split 1: train [0 ... 65], gap [66 ... 72], test [73 ... 82]
    # Split 2: train [0 ... 75], gap [76 ... 82], test [83 ... 92]
    # Split 3: train [0 ... 85], gap [86 ... 92], test [93 ... 99]

    Input:
    - `add_lag1`: bool, whether to add a 1-step lag feature
    - `add_roll7`: bool, whether to add a 7-step rolling feature
    - `n_splits`: int, number of splits (default 5)
    - `test_size`: int, size of test set for each split (default 30)

    Output:
    - `TimeSeriesSplit` object with correct `gap` parameter
    """
    # Compute `gap` (in steps) as largest window needed for AR features: 1 for `add_lag1`, 7 for `add_roll7`, 0 if no AR features
    # `gap` ensures that, for each test split, at least `gap` rows before test set are excluded from training
    # This prevents training on rows that would be used to compute AR features for the test set
    gap = max(1 if add_lag1 else 0, 7 if add_roll7 else 0)
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

def make_timeseries_split_with_gap(max_window_in_steps: int, n_splits=5, test_size=30):
    """
    Create `TimeSeriesSplit` with an explicit `gap` in steps (rows) to cover the
    largest autoregressive window used by your features/baselines.

    Input:
    - `max_window_in_steps`: maximum of all AR lags and rolling windows (in steps)
    - `n_splits`: number of splits (default 5)
    - `test_size`: size of test set for each split (default 30)

    Output:
    - `TimeSeriesSplit` configured with the provided `gap`
    """
    if max_window_in_steps < 0:
        raise ValueError("max_window_in_steps must be non-negative")
    return TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=int(max_window_in_steps))

class First19DaysTrainSplit(BaseCrossValidator):
    """
    Single split based on calendar day-of-month:
    - Train indices: all rows where day-of-month ∈ [1..19] across all months/years
    - Test indices:  all rows where day-of-month ∈ [20..end] across all months/years

    Assumptions:
    - `X` (and optionally `y`) are indexed by a `pd.DatetimeIndex` at hourly or daily granularity.
    - Order of rows follows time increasing (not strictly required for index selection here).

    Notes:
    - Works for hourly data: uses `index.day` to derive day-of-month for each timestamp.
    - This split intentionally mixes months (not a walk-forward split); use for diagnostic/ablation only.
    """

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        # Require a DatetimeIndex so we can compute day-of-month robustly
        if not hasattr(X, 'index') or not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("First19DaysTrainSplit requires X.index to be a pd.DatetimeIndex.")

        # Day-of-month for each row (1..31 depending on month)
        dom = X.index.day

        # Train: days 1..19; Test: days 20..end (vectorized boolean masks)
        train_idx = np.where(dom <= 19)[0]
        test_idx  = np.where(dom >= 20)[0]

        yield train_idx, test_idx

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

        split_point = n_samples - 30 # first index in the test block
        train_idx = np.arange(0, split_point) # all integers btw 0 and `split_point`
        test_idx = np.arange(split_point, n_samples)

        yield train_idx, test_idx


##########################################
# FEATURE ENGINEERING engineering
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
    Feature engineering class, inherits from `BaseEstimator`, `TransformerMixin`:
    - drops `atemp`: highly correlated with `temp`
    - one-hot encodes 'weathersit' (drops first binary column)
    - adds sin/cos transformation pairs for month, season, weekday, and optionally hour; then drops original cols
    - optionally adds autoregressive (AR) features from past data and then drops `cnt`

    AR configuration (non-breaking):
    - Default booleans keep old behavior (1-step lag and 7-step rolling median) when
      `ar_lags`/`ar_roll_windows` are not provided.
    - Pass explicit `ar_lags` and/or `ar_roll_windows` (both in steps) for custom setups.

    Works with/returns a Pandas DataFrame, so column names survive
    """
    def __init__(self, add_lag1=True, add_roll7=True, ar_lags=None, ar_roll_windows=None):
        self.ohe = onehot_no_sparse() # Set up a `OneHotEncoder`
        self.weather_cols_ = ['weathersit'] # Stores column to OH encode
        self.add_lag1 = add_lag1
        self.add_roll7 = add_roll7
        self.ar_lags = None if ar_lags is None else list(ar_lags)
        self.ar_roll_windows = None if ar_roll_windows is None else list(ar_roll_windows)

    # Use only training data, nothing to learn except one-hot encoder
    def fit(self, X, y=None):
        X_ = X.copy() # Don't alter og data
        self.ohe.fit(X_[self.weather_cols_]) # learn OHE variable mapping
        self.cnt_median_ = X_['cnt'].median() # compute median of `cnt` from training data for use in `fillna()`
        self._last_seen_train_time_ = X_.index.max() # Store last timestamp seen in training (`DatetimeIndex` value)
        # Determine carry length from configured windows (in steps)
        if self.ar_lags is not None or self.ar_roll_windows is not None:
            max_lag = max(self.ar_lags) if self.ar_lags else 0
            max_roll = max(self.ar_roll_windows) if self.ar_roll_windows else 0
            self._max_ar_window_ = int(max(max_lag, max_roll))
        else:
            self._max_ar_window_ = 7 if self.add_roll7 else (1 if self.add_lag1 else 0)
        self._carry_ = X_['cnt'].tail(self._max_ar_window_).to_numpy() if self._max_ar_window_ > 0 else np.array([], dtype=float)
        return self

    def _build_ar(self, X):
        """
        Build autoregressive features for training data.

        Input:
        - `X`: DataFrame with `cnt` column

        Output:
        - `lag1`: Series with previous step's `cnt`
        - `roll7`: Series with 7-step rolling median of `cnt`
        """
        # Compute lag-1 feature (previous day's `cnt`)
        lag1 = X['cnt'].shift(1)
        # Compute rolling 7-day median (excluding current day)
        roll7 = X['cnt'].shift(1).rolling(7, min_periods=1).median()
        # Replace missing values with median to handle NaNs from `shift()` and `rolling()`
        return lag1.fillna(self.cnt_median_), roll7.fillna(self.cnt_median_)

    def _build_ar_with_carry(self, X):
        """
        Build autoregressive features for test data using carryover from training.

        Input:
        - `X`: DataFrame with `cnt` column

        Output:
        - `lag1`: Numpy array with previous step's `cnt` (using carryover)
        - `roll7`: Numpy array with 7-step rolling median of `cnt` (using carryover)
        """
        # Combine last 7 training `cnt` values with test data `cnt` values
        # `np.r_` concatenates arrays: join `self._carry_` (last 7 train `cnt`) and `X['cnt']` (test `cnt`)
        # This ensures lag/rolling features for test set use correct previous values from training
        series = pd.Series(np.r_[self._carry_, X['cnt'].to_numpy()], index=None)
        off = len(self._carry_)
        lag1 = series.shift(1)
        roll7 = series.shift(1).rolling(7, min_periods=1).median()
        # Only return features for test set rows, not prepended training values
        return (
            lag1.iloc[off:].fillna(self.cnt_median_).to_numpy(),
            roll7.iloc[off:].fillna(self.cnt_median_).to_numpy()
        )
    
    # Feature engineering
    def transform(self, X):
        X_ = X.copy()

        # 1) Drop highly-correlated `atemp`
        if 'atemp' in X_.columns: 
            X_ = X_.drop(columns='atemp')

        # 2) One-hot encode weather
        weather_ohe = self.ohe.transform(X_[self.weather_cols_])
        X_.drop(columns=self.weather_cols_, inplace=True) # Drop og column, not needed anymore
        # Fetch colnames that the fitted OHE will output for the encoded weather feature
        X_[self.ohe.get_feature_names_out(self.weather_cols_)] = weather_ohe # add one-hot matrix to `X_`

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
        want_ar = (self.ar_lags is not None and len(self.ar_lags) > 0) or (self.ar_roll_windows is not None and len(self.ar_roll_windows) > 0) or self.add_lag1 or self.add_roll7
        if want_ar:
            # Set `use_carry` True if processing test data (all indices after last train time)
            use_carry = X.index.min() > self._last_seen_train_time_
            if use_carry:
                feats = self._build_ar_general_test(X)
            else:
                feats = self._build_ar_general_train(X)
            for k, v in feats.items():
                X_[k] = v

        # Drop current-day `cnt` to avoid leakage
        if 'cnt' in X_.columns:
            X_.drop(columns='cnt', inplace=True)
        return X_

    def _build_ar_general_train(self, X):
        """
        Build AR features for training data with arbitrary lags and rolling windows (all in steps).

        Returns dict: {feature_name: Series}
        """
        feats = {}
        lags = (self.ar_lags if self.ar_lags is not None else ([1] if self.add_lag1 else []))
        rolls = (self.ar_roll_windows if self.ar_roll_windows is not None else ([7] if self.add_roll7 else []))
        if lags:
            for k in lags:
                s = X['cnt'].shift(int(k)).fillna(self.cnt_median_)
                feats[f'cnt_lag{k}'] = s
        if rolls:
            base = X['cnt'].shift(1)
            for w in rolls:
                s = base.rolling(int(w), min_periods=1).median().fillna(self.cnt_median_)
                feats[f'cnt_roll{w}'] = s
        return feats

    def _build_ar_general_test(self, X):
        """
        Build AR features for test data with arbitrary lags and rolling windows using carryover from training.

        Returns dict: {feature_name: np.ndarray}
        """
        feats = {}
        lags = (self.ar_lags if self.ar_lags is not None else ([1] if self.add_lag1 else []))
        rolls = (self.ar_roll_windows if self.ar_roll_windows is not None else ([7] if self.add_roll7 else []))
        if (not lags) and (not rolls):
            return feats
        series = pd.Series(np.r_[self._carry_, X['cnt'].to_numpy()], index=None)
        off = len(self._carry_)
        if lags:
            for k in lags:
                s = series.shift(int(k))
                feats[f'cnt_lag{k}'] = s.iloc[off:].fillna(self.cnt_median_).to_numpy()
        if rolls:
            base = series.shift(1)
            for w in rolls:
                s = base.rolling(int(w), min_periods=1).median()
                feats[f'cnt_roll{w}'] = s.iloc[off:].fillna(self.cnt_median_).to_numpy()
        return feats


##########################################
# PREPROCESSING helpers (no feature engineering)
##########################################
def make_no_fe_preprocess(X, target_col='cnt'):
    """
    Build a passthrough preprocessor (no feature engineering): features are passed straight through
        as-is using `ColumnTransformer`.    

    Input:
    - X: DataFrame to derive feature columns from
    - target_col: column to exclude (default 'cnt')

    Output:
    - preprocess: ColumnTransformer that passes features through unchanged
    - to_df: FunctionTransformer that wraps array output back into a DataFrame
             with original feature names (handy for LightGBM to keep names)
    - feat_cols: list of feature column names
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
