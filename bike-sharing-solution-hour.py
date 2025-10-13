# -*- coding: utf-8 -*-
# bike-sharing-solution-hour.py
# Kaggle's "Bike sharing demand" challenge

"""
##########################################
# LOAD DATA FROM UCI REPOSITORY
##########################################
# Create `data` directory, if it doesn't exist
!mkdir -p data

# Download the zipped archive from UCI
!wget -q --show-progress -P data http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip

# Unzip to get `hour.csv`, `day.csv`, `Readme.txt`
!unzip -q data/Bike-Sharing-Dataset.zip -d data
"""

##########################################
# IMPORTS
##########################################
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-GUI backend, safe for headless scripts
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
FIGDIR = Path.cwd() / "plots_hour" # save all figures under ./plots
FIGDIR.mkdir(parents=True, exist_ok=True) # create the folder if missing
from utils import savefig_pdf
from eda_utils import iqr_mask
from ml_utils import make_timeseries_split, make_no_fe_preprocess, Last30DaysSplit
from model_eval import evaluate_pipeline, ar_baseline_scores
from stat_analysis import diagnose_multicollinearity


##########################################
# DATA CLEANING
##########################################

##########################################
# Read raw data
hour_df = pd.read_csv('data/hour.csv')
print(f"Hourly dataset: {hour_df.shape[0]} rows x {hour_df.shape[1]} cols")
print(f"Its columns are {hour_df.columns.values}")
print(f"Raw data:\n{hour_df.head(2)}\n")

##########################################
# Clean data: handle missing values, set DateTime index, drop leaky columns
na = hour_df.isna().sum() # Sanity-check: check for missing values
if na.any():
    raise ValueError(f"Missing values in key columns for index: {na[na > 0].to_dict()}")
hour_df['timestamp'] = pd.to_datetime(hour_df['dteday']) + pd.to_timedelta(hour_df['hr'], unit='h') # combine date + hour
hour_df = hour_df.set_index('timestamp').sort_index() # set as DataFrame index
# Drop 'instant' (just an ID), 'dteday' (now redundant), 'casual' and 'registered' (risk of target leakage)
hour_df = hour_df.drop(columns=['instant', 'dteday', 'casual', 'registered'])
print(f'Columns after dropping uninformative and leaky ones: {hour_df.columns}')
print()
# Statistics
pd.set_option("display.max_columns", None) # show all columns
print(hour_df.describe())
print()

##########################################
# Outlier detection `humidity` column
iqr_mask(hour_df['hum'], 25, 75, 1.5, label="hum")
# Replace impossible humidity zeros using previous/next hour.
outlier_humidity_mask = hour_df['hum'] == 0  # mask to flag implausible 0.0 humidity
hour_df.loc[outlier_humidity_mask, 'hum'] = np.nan  # set to NaN to mark as missing
# - `ffill()`: carry forward the last valid value in time (previous hour)
# - `bfill()`: use the next valid value in time (next hour) if needed
hour_df['hum'] = hour_df['hum'].ffill().bfill() # fill from previous hour, else from next hour
print()


##########################################
# EXPLORATORY DATA ANALYSIS
##########################################

##########################################
# Visualize rentals of bikes per day: plot mean, rentals, rolling rentals
# `resample('D').sum()` does groupby+sum but on DateTimeIndex data
daily_cnt = hour_df['cnt'].resample('D').sum() # aggregate hourly counts -> into daily totals
mean_cnt = daily_cnt.mean() # average daily rentals over full period
window = 30 # rolling window (days)
# Smoothing: 30d rolling mean (centered instead of right-aligned)
rolling_mean = daily_cnt.rolling(window=window, center=True).mean()
plt.figure(figsize=(12, 6)) # plot size
# `alpha`: set transparency for daily count curve
plt.plot(daily_cnt.index, daily_cnt, label='Daily count', alpha=0.5) # Daily totals
plt.plot(daily_cnt.index, rolling_mean, label=f'{window}-day rolling mean', linewidth=2) # 30d rolling mean
plt.axhline(mean_cnt, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_cnt:.0f}') # overall mean
plt.title('Daily bike rentals')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend() # show legend
plt.tight_layout() # avoid label overlap
savefig_pdf("fig_daily_rentals", FIGDIR) # save to "plots/fig_daily_rentals.pdf"

##########################################
# Plot marginal distribution (histogram + KDE) of daily `cnt`
# Use the daily totals computed earlier (daily_cnt)
sns.histplot(daily_cnt, bins=30, kde=True, kde_kws={'bw_adjust': 0.7})
plt.title('Distribution of Daily Bike Rentals')
plt.xlabel('Daily rentals (cnt)')
plt.ylabel('Frequency')
plt.tight_layout()
savefig_pdf("fig_hist_cnt", FIGDIR)
##########################################
# Plot marginal distribution (histogram + KDE) of daily `cnt` after log-transform
transformed = np.log1p(daily_cnt)
sns.histplot(transformed, bins=30, kde=True, kde_kws={'bw_adjust': 0.7})
plt.title('Distribution of Daily Bike Rentals (post log-transf.)')
plt.xlabel('Daily rentals (cnt)')
plt.ylabel('Frequency')
plt.tight_layout()
savefig_pdf("fig_hist_log_cnt", FIGDIR)

##########################################
# Outlier detection and elimination: day granularity (all hours in each day)
iqr_mask(daily_cnt, 25, 75, 1.5, label="daily cnt") # standard Tukey's outlier range
iqr_mask(daily_cnt, 25, 75, 1.0, label="daily cnt") # narrower range -> more outliers detected
print()
##########################################
# Outlier detection and elimination: hour granularity (all days in each hour, 1am, 2am, ..., 12am)
# -> makes sense because, e.g., 3AM demand should be very different from 6PM demand
hour_masks = []
for hour, hour_group in hour_df.groupby('hr'):
    mask = iqr_mask(hour_group['cnt'], 25, 75, 2 ,label=f'hr={hour}') # `coeff=2` more permissive -> fewer outliers, no overflagging
    # Wrap mask in a Bool Series aligned with group's index -> can later merge masks across groups and preserve timestamps
    hour_masks.append(pd.Series(mask, index=hour_group.index))
# Concatenate all per-hour masks into one, `reindex` to align with og DataFrame `hour_df` (no lost rows)
# -> fill missing values with `False` (not outliers)
outliers_per_hour = pd.concat(hour_masks).reindex(hour_df.index).fillna(False)
hour_df['hr_cnt_outlier'] = outliers_per_hour.astype(int) # new column: 1 if hour-level outlier, else 0

##########################################
# Visualize total bike rental count per weekday
plot_daily_df = daily_cnt.to_frame('cnt').copy()
plot_daily_df['day_of_week'] = plot_daily_df.index.day_name()
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
plt.figure(figsize=(10,5))
sns.barplot(
    data = plot_daily_df,
    x = 'day_of_week',
    y = 'cnt',
    estimator = sum, # aggregate by sum of daily total rentals across the period
    order = dow_order, # keep calendar order
    hue='day_of_week',
    dodge=False, # avoid side-by-side bars
    palette = sns.color_palette("colorblind", 7)
)
plt.xlabel('Day of the week')
plt.ylabel('Total rentals (2011-2012)')
plt.title('Bike-sharing demand by weekday (sum over two years)')
plt.xticks(rotation=45)
plt.tight_layout()
savefig_pdf("fig_weekday_totals", FIGDIR)

##########################################
# Visualize total bike rental count per month
plot_month_df = daily_cnt.to_frame('cnt').copy()
plot_month_df['month_name'] = plot_month_df.index.month_name()
month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
plt.figure(figsize=(12,5))
sns.barplot(
    data=plot_month_df,
    x='month_name',
    y='cnt',
    estimator=sum,
    order=month_order,
    hue='month_name',
    dodge=False,
    legend=False,
    palette=sns.color_palette("colorblind", 12)
)
plt.xlabel('Month')
plt.ylabel('Total rentals (2011-2012)')
plt.title('Bike-sharing demand by month (sum over two years)')
plt.xticks(rotation=45)
plt.tight_layout()
savefig_pdf("fig_month_totals", FIGDIR)

##########################################
# Visualize total bike rental count per season (daily aggregation)
season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'} # Map numeric season codes to names
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
# Build daily dataframe with totals and season label (season is constant within a day)
plot_season_df = daily_cnt.to_frame('cnt').join(
    hour_df['season'].resample('D').max().to_frame('season')
)
plot_season_df['season_name'] = plot_season_df['season'].map(season_map)
plt.figure(figsize=(8,5))
sns.barplot(
    data=plot_season_df,
    x='season_name',
    y='cnt',
    estimator=sum,
    order=season_order,
    hue='season_name',
    dodge=False,
    legend=False,
    palette=sns.color_palette("colorblind", 4)
)
plt.xlabel('Season')
plt.ylabel('Total rentals (2011-2012)')
plt.title('Bike-sharing demand by season (sum over two years)')
plt.tight_layout()
savefig_pdf("fig_season_totals", FIGDIR)

##########################################
# Explore multimodality in `cnt`: visualize distplot (KDE) and histplot (histograms)
# Aggregate hourly -> daily to keep consistency with "Daily rental" interpretation
plot_season_df = daily_cnt.to_frame('cnt').join(
    hour_df['season'].resample('D').max().to_frame('season') # season is constant within a day
)
# KDE for each season == smoothed estimated histogram distribution (distplot). Idea: show PDF for each season
g = sns.displot(
    data=plot_season_df.assign(season=plot_season_df['season'].map(season_map)),
    x='cnt',
    hue='season',
    kind='kde',
    clip=(0, None),
    fill=False,
    common_norm=False,
    height=4,
    aspect=1.4
).set(title='Daily rental distribution by season')
g.savefig(FIGDIR / "fig_kde_by_season.pdf")
plt.close(g.fig)
g = sns.FacetGrid( # Histograms for each seasonal distribution (histplot)
        plot_season_df.assign(season=plot_season_df['season'].map(season_map)),
        col='season',
        col_wrap=2,
        height=3.2
    )
g.map(sns.histplot, 'cnt', bins=20, color='steelblue')
g.set_axis_labels('Daily rentals (cnt)', 'N. of days')
g.savefig(FIGDIR / "fig_hist_by_season.pdf")
plt.close(g.fig)

##########################################
# Assume that each bike has exactly maximum 12 rentals per day:
# a) Find max number of bicycles nmax that was needed in any one day
# b) Find 95th percentile of bicycles n95 that was needed in any one day
# a) Compute fleet size needed per day from daily totals (hourly data aggregated to days): each bike can be rented ≤12 times/day
daily_fleet_size = np.ceil(daily_cnt / 12).astype(int)
nmax = int(daily_fleet_size.max()) # fleet size needed to cover demand on "busiest" day
n95  = int(np.ceil(np.percentile(daily_fleet_size, 95))) # 95th percentile fleet size (smallest inventory covering 95% days)
n50  = int(np.ceil(np.percentile(daily_fleet_size, 50)))
print(f"Number of bikes needed on busiest day (nmax): {nmax}")
print(f"Smallest number of bikes to cover 95% of days (n95): {n95}")
print()

##########################################
# Visualize distribution of the covered days depending on number of available bicycles.
# E.g. `nmax` bicycles would cover 100% of days, `n95` covers 95%, etc.

# Build "coverage" curve using daily fleet size: for every value k, compute proportion of days with demand ≤k.
# Basically, compute empirical CDF (in %): P("bikes needed" ≤ k) ∈ [0, 100]
coverage_df = (
    daily_fleet_size
    .value_counts()  # for each k, count number of days that required exactly k bikes
    .sort_index()    # sort by bike count k, ascending
    .cumsum()        # cumulative days where ≤k bikes were needed
    / len(daily_fleet_size) * 100  # convert into percentage ~CDF
).reset_index()
coverage_df.columns = ['k', 'emp_CDF']
plt.figure(figsize=(10, 6)) # Plot: needed fleet size `k` vs %of days covered `emp_CDF`
# Empirical CDF is flat until a new value of k is attained, then jumps up:
#   - `plt.step`: draws that step-curve
#   - `where='post'`: hold current CDF value until next x-tick (i.e., next CDF value is attained), then jump
plt.step(coverage_df['k'], coverage_df['emp_CDF'], where='post')
# Vertical dashed line at 50th, 95th, 100th (max) percentiles of fleet size
plt.axvline(n50, linestyle='--', label=f'n50 = {n50}', color='#0072B2') 
plt.axvline(n95, linestyle='--', label=f'n95 = {n95}', color='#D55E00')
plt.axvline(nmax, linestyle='--', label=f'nmax = {nmax}', color='#009E73')
plt.xlabel('N. needed bicycles (k)')
plt.ylabel('% of fully served days (demand ≤ k)')
plt.title(f'Daily coverage for different bicycle fleet sizes k, over {len(daily_fleet_size)} days')
plt.grid(True, linestyle=':') # light dotted grid
plt.legend()
plt.tight_layout()
savefig_pdf("fig_coverage_ecdf", FIGDIR)


##########################################
# CORRELATION MATRIX
##########################################
print(hour_df.columns)
# Include `hr` (hour-of-day) and any engineered flags (e.g., hourly outlier column)
corr_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
             'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
if 'hr_cnt_outlier' in hour_df.columns: # Optionally include outlier flag column if present
    corr_cols.append('hr_cnt_outlier')
corr = hour_df[corr_cols].corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig, ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corr, mask=mask, vmax=.8, square=True, annot=True)
plt.tight_layout()
savefig_pdf("fig_corr_pre_FE", FIGDIR, fig)


##########################################
# PREDICTION MODELS
##########################################

##########################################
# Data
X = hour_df.copy(deep=True)
# Drop helper columns if present (safe-guard if computed earlier)
drop_cols = [c for c in ['daily_fleet_size'] if c in X.columns]
if drop_cols:
    X = X.drop(columns=drop_cols)
y = X['cnt']
print(f'Columns before any feature engineering: {X.columns.values}')
print()

##########################################
# Build simple passthrough preprocessor (no FE)
preprocess, to_df, feat_cols = make_no_fe_preprocess(X, target_col='cnt')

##########################################
# Instantiate splitters
cv_last30 = Last30DaysSplit()
hourly_ar_lags = [1, 24, 168]
hourly_ar_rolls = [24, 168]
cv_ts_ar  = make_timeseries_split(n_splits=5, test_size=30, lags=hourly_ar_lags, rolls=hourly_ar_rolls)
cv_ts_no  = make_timeseries_split(n_splits=5, test_size=30)

##########################################
# Autoregressive baselines
base_last30 = ar_baseline_scores(y, cv_last30, lags=hourly_ar_lags, rolls=hourly_ar_rolls)
base_ts     = ar_baseline_scores(y, cv_ts_ar,  lags=hourly_ar_lags, rolls=hourly_ar_rolls)
print('Last-30 split baselines (RMSLE):')
for name, val in sorted(base_last30.items()):
    print(f'  {name}: {val:.6f}')
print('Time-series split baselines (RMSLE):')
for name, val in sorted(base_ts.items()):
    print(f'  {name}: {val:.6f}')
print()

##########################################
# Our base regressors
# Models are referenced by key in `models` list below; actual estimators are built inside `evaluate_pipeline`.
models = ["rf", "hgbr", "gbr", "xgb", "cbr", "lgbm"]





##########################################
# MODEL EVALUATION
##########################################
# No FE (passthrough)
results_no_fe = evaluate_pipeline(
    models,
    'no_fe',
    X=X, y=y,
    cv_last30=cv_last30,
    cv_ts_no=cv_ts_no,
    cv_ts_ar=cv_ts_ar,
    preprocess=preprocess,
    to_df=to_df,
)
print("\n=== Evaluation: no FE ===")
print(results_no_fe.sort_values(["split", "model", "log"]).to_string(index=False))

# FE, no AR
results_fe = evaluate_pipeline(
    models,
    'fe',
    X=X, y=y,
    cv_last30=cv_last30,
    cv_ts_no=cv_ts_no,
    cv_ts_ar=cv_ts_ar,
)
print("\n=== Evaluation: FE (no AR) ===")
print(results_fe.sort_values(["split", "model", "log"]).to_string(index=False))

# FE + AR (walk-forward pipeline evaluation)
results_fe_ar = evaluate_pipeline(
    models,
    'fe_ar',
    X=X, y=y,
    cv_last30=cv_last30,
    cv_ts_no=cv_ts_no,
    cv_ts_ar=cv_ts_ar,
    ar_lags=hourly_ar_lags,
    ar_rolls=hourly_ar_rolls,
)
print("\n=== Evaluation: FE + AR ===")
print(results_fe_ar.sort_values(["split", "model", "log"]).to_string(index=False))





##########################################
# 3.2.10) Diagnostics: multicollinearity on engineered features (TRAIN ONLY)
##########################################
# Why train-only: avoid peeking at the test distribution and keep diagnostics fold-consistent.

# Run diagnostics for both settings you evaluate with CV
print("\n=== Multicollinearity diagnostics: NO AR features ===")
vif_no_ar  = diagnose_multicollinearity(X, cv=cv_ts_no, ar_lags=None, ar_rolls=None, figdir=FIGDIR)

print("\n=== Multicollinearity diagnostics: WITH AR features ===")
vif_with_ar = diagnose_multicollinearity(X, cv=cv_ts_ar, ar_lags=hourly_ar_lags,  ar_rolls=hourly_ar_rolls, figdir=FIGDIR)
