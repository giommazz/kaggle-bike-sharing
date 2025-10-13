# Comments.md
This document contains comments about choices taken in the code.


## Data cleaning
Instead of dropping `casual` and `registered`, one could, alternatively:
- train two separate models, one to predict `registered` and the other one to predict `casual`;
- then, sum the two predictions to yield `cnt`

Pros: potentially better adaptability.
Cons:
- double training time. This might be negligible for decision trees, but this depend on their size (e.g. number of estimators) or training criteria (e.g.impurity criterion)
- prediction errors might add up more than by just predicting the single target `cnt`


## Outlier detection
- the `iqr_mask()` function could be computed with Pandas' `quantile([0.25, 0.75])` as well
- the only column with a seemingly weird value seems to be `hum`, with a row being `0.0`: Washington is rearely so dry. Outlier flag!


## Marginal distributions
- The distribution is almost normal-like. However...
- ...multimodality detected (3 peaks/clusters): potentially corresponding to 3 seasons? (maybe winter, summer, shoulder season);
- Presence of some extreme lows/highs, which might be outliers, or not: they could correspond to extremely pleasant days, or very snowy days. When I applied the Tukey's IQR rule to exclude potential outliers, I found none: I assumed the few extreme points to plausibly correspond to exceptionally bad or good weather days
- log-transform is left-skewed (log compresses high values and sparsifies small ones)


## Visualize bike rental count per weekday
It doesn't look like `weekday` should have a large predictive power, as a feature. Later, I tried removing it and training RFs without it. Since it improves the error, although very marginally, I decided to keep it


## Correlation
- `atemp` is so highly corr. with `temp` (0.99) that they are basically the same $\to$ drop
- `season` is highly corr. with `month` (0.83). However, since: a) RFs don't suffer from highly correlated features that much, and b) I experimented with dropping `season` and this worsened the final test score $\to$ I decided to keep it


## Feature engineering, model evaluation, splits

### Log transform (+revert)
Reasons why I chose it
- **Compresses outliers**, while reducing their relative importance $\to$ now big and small rental days (e.g. vv hot or vv cold days) are comparable on an "additive scale", and additive effects should be more easily"measured"/"estimated" by models than multiplicative effect. So, our trees will tolerate more error on these outliers while being more accurate on the bulk of data.
- **Var is more uniform/stabilized**, while not changing the ranking of the values $\to$ improved split-point detection in decision trees. What happens is that, in RFs, each tree make splits by minimizing the variance of the children (impurity-based); in the end, since the label/prediction at the leaves is a mean, on the leaves RF basically minimize the sum of variances. Now, if the target is skewed, outliers will affect the variances and worsen splits: split points are pushed towards the outliers, as a tree tries to "isolate" the tail from the remaining points $\to$ trees will be more affected by outliers if data is not transformed, i.e., they will predict outliers more accurately

Bonus: empirical performance with log-transform is often superior

### Test metric for validating model prediction
Choice: **Root Mean Squared Log Error** (RMSLE):
$$\sqrt{\dfrac{1}{n}\sum_{i \in [n]}\Bigl(\log(\hat{y}_i+1) - \log(y_i+1)\Bigr)^2}\,,$$
so each summand in the loss is roughly equivalent to $≈\log^2\left(\dfrac{\hat{y_i}}{y_i}\right)$, with $\hat{y_i}$ a prediction and $y_i$ a label

Reasons:
- **asymmetric costs**: RMSLE yields a larger penalty for under-predictions: basically grows exponentially for underpredictions and less than logarithmically for overprediction. This is more useful than, say, MAE and MSE, which are "cost-symmetric" (no difference between under and over predictions). In fact, while over-predicting might not a big issue (e.g., if we prepared more bikes at a spot for one day, we could always move some during the night...or have them at the ready for the day after, the assumption being that people are always willing to move a bit to get to a bike), but under-predicting is bad because we cannot satisfy demand and thus lose money
- **robustness to outliers**: the log makes it robust to outliers, with similar considerations as the ones made about the log-transform. Useful considering that the data in `cnt` has a wide range of values (almost 3 orders of magnitude, from 22 to 8714). Also useful because it can predict relative errors well, and we are more interested in them than absolute errors (for instance, predicting 9 instead of 10 has the same RMSLE as predicting 900 instead of 1000).

### About Pipelines
I used sklearn's Pipelines, and called `Pipeline` objects in my `eval_pipeline` to:
- perform a train/test split $\to$ see `make_timeseries_split` and `Last30DaysSplit` (both avoid data leakage while training)
- perform feature engineering and target transformation $\to$ see `BikeFeatureEngineer`. In particular, `BikeFeatureEngineer`:
    - drops highly correlated features
    - computes one-hot encodings
    - compute cyclical sin/cos transformation features
    - computes autoregressive features using past data withouth leaking info
- fit a predictor, then evaluate it on a test set $\to$ see `eval_pipeline`


## About model evaluation (autoregressive baselines VS simple RF baseline)
- `cv_last30` split: autocorrelation is strong and, in a sense, bike rental demand is persistent to the 1-day lag. In fact:
  - the 1-day lag autoregressive baseline is good, actually better than our naive random forest with og-scaled target and no feature engineering. But already when we log-transform we are better than naive baseline, and then adding FE we improve even more.
  - note that our RF is better on this split when autoregressive feature are used WRT no AR features
- `cv_20_10` split: our RF is already better than the autoregressive baselines on this "more challenging split". Our calendar+weather RF improves once we stabilise variance (with log-target) and add basic feature engineering. Autoregressive features worsen the test scores. It would be interesting to perform experiments on a larger dataset
- the log-transform (transform `y_train`, then backtransform `y_pred` before computing the RMSLE) always yields better test results

### Takeaways
- `cv_last30`: our best model (with feature engineering including autoregressive features, and target log-transform) reduces the 1-day lag baseline error by about $18\%$, and the 7-day rolling median baseline by about $33\%$.
- `cv_last30`: on this tougher split, our best model (with feature engineering not including autoregressive features, and target log-transform) reduces the 1-day lag baseline error by about $19\%$, and the 7-day rolling median baseline by about $14\%$.
- **log-transform**: always produces improvements in test error: at least $2\%$ and as much $16\%$ WRT un-transformed target
- **autoregressive features**: could be useful but to be verified with larger dataset


## Potential improvements
- Hyperparameter tuning of the RF
- Add more features:
  - autoregressive: rolling mean instead of median, or as well as rolling median; lag-based feature, not only computed from `cnt` but also from `casual` and `registered`(if not heavily correlated with `cnt`)
  - weather: "pleasant_weather", for example if temp > 0.5 and hum < 0.6. For this, I would read in detail the paper that created the UCI dataset, plus articles on how weather and bike rental forecasts are related, etc.
- Explore feature importance
- Try new models
  - Gradient-boosted trees: they keep learning from the errors of previous trees, thus reducing bias and capturing subtler patterns than an RF, which just averages many independent trees
  - Time-series specific (maybe, if not even GBT works?): something like Prophet? Would need further exploration...
- The dataset is very tiny so the model might not be learning a lot: try retraining after gathering more data