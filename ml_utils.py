# ml_utils.py
import numpy as np 
import inspect
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


##########################################
# Evaluation scorers
def _rmsle(y_true, y_pred):
    """
    Compute root mean squared log error (RMSLE) between `y_true` and `y_pred`.
    """
    # Compute RMSLE using sklearn's `mean_squared_log_error`
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Create `rmsle_scorer` for use in sklearn model evaluation (`False` because lower is better)
rmsle_scorer = make_scorer(_rmsle, greater_is_better=False)


##########################################
# Custom splitters
def make_timeseries_split(add_lag1: bool, add_roll7: bool, n_splits=5, test_size=30):
    """
    Create `TimeSeriesSplit` with correct `gap` to prevent data leakage when using autoregressive features.
    -> keeps order of data, splits train/test sets sequentially to avoid data leakage from future to past

    # Example: with n_samples=100, n_splits=3, test_size=10, gap=7
    # Split 1: train [0 ... 65], gap [66 ... 72], test [73 ... 82]
    # Split 2: train [0 ... 75], gap [76 ... 82], test [83 ... 92]
    # Split 3: train [0 ... 85], gap [86 ... 92], test [93 ... 99]

    Input:
    - `add_lag1`: bool, whether to add 1-day lag feature
    - `add_roll7`: bool, whether to add 7-day rolling feature
    - `n_splits`: int, number of splits (default 5)
    - `test_size`: int, size of test set for each split (default 30)

    Output:
    - `TimeSeriesSplit` object with correct `gap` parameter
    """
    # Compute `gap` as largest window needed for AR features: 1 for `add_lag1`, 7 for `add_roll7`, 0 if no AR features
    # `gap` ensures that, for each test split, at least `gap` rows before test set are excluded from training
    # This prevents training on rows that would be used to compute AR features for the test set
    gap = max(1 if add_lag1 else 0, 7 if add_roll7 else 0)
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

        split_point = n_samples - 30 # first index in the test block
        train_idx = np.arange(0, split_point) # all integers btw 0 and `split_point`
        test_idx = np.arange(split_point, n_samples)

        yield train_idx, test_idx


##########################################
# Feature engineering
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
    - drops 'atemp'
    - one-hot encodes 'weathersit' (drops first binary column)
    - adds sin/cos transformation pairs for month, season, weekday. Then drops ogs
    - creates 1-day lag and rolling 7-day median from past data + drops `cnt`

    Works with/returns a Pandas DataFrame, so column names survive
    """
    def __init__(self, add_lag1=True, add_roll7=True):
        self.ohe = onehot_no_sparse() # Set up a `OneHotEncoder`
        self.weather_cols_ = ['weathersit'] # Stores column to OH encode
        self.add_lag1 = add_lag1
        self.add_roll7 = add_roll7

    # Use only training data, nothing to learn except one-hot encoder
    def fit(self, X, y=None):
        X_ = X.copy() # Don't alter og data
        self.ohe.fit(X_[self.weather_cols_]) # learn OHE variable mapping
        self.cnt_median_ = X_['cnt'].median() # compute median of `cnt` from training data for use in `fillna()`
        self._last_seen_train_time_ = X_.index.max()  # Store last timestamp seen in training (`DatetimeIndex` value)
        self._carry_ = X_['cnt'].tail(7).to_numpy()   # Store last 7 `cnt` values for use in test-time AR features
        return self

    def _build_ar(self, X):
        """
        Build autoregressive features for training data.

        Input:
        - `X`: DataFrame with `cnt` column

        Output:
        - `lag1`: Series with previous day's `cnt`
        - `roll7`: Series with 7-day rolling median of `cnt`
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
            - `lag1`: Numpy array with previous day's `cnt` (using carryover)
            - `roll7`: Numpy array with 7-day rolling median of `cnt` (using carryover)
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
        X_.drop(columns=['mnth', 'season', 'weekday', 'hr'], inplace=True)

        # 4) Autoregressive features
        if self.add_lag1 or self.add_roll7:
            # Set `use_carry` True if processing test data (all indices after last train time)
            use_carry = X.index.min() > self._last_seen_train_time_
            if use_carry: # test data
                lag1, roll7 = self._build_ar_with_carry(X) # use carryover from training
            else: # train data
                lag1, roll7 = self._build_ar(X) # use median from training
            if self.add_lag1:
                X_['cnt_lag1']  = lag1
            if self.add_roll7:
                X_['cnt_roll7'] = roll7

        # Drop current-day `cnt` to avoid leakage
        if 'cnt' in X_.columns:
            X_.drop(columns='cnt', inplace=True)
        return X_
