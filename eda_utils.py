# eda_utils.py
import numpy as np

def iqr_mask(pd_series, pval_low, pval_high, coeff):
    """
    Return a Boolean mask that flags Tukey outliers (typically values outside [Q1 - c*IQR, Q3 + c*IQR]).
    """
    assert pval_low < pval_high, "`pval_low` should be smaller than `pval_hihg`"
    perc_low, perc_high = np.percentile(pd_series, [pval_low, pval_high])
    range = perc_high - perc_low # compute interequartile range
    # compute bounds: beyond them lie outliers
    lower, upper = perc_low - coeff * range, perc_high + coeff * range
    outlier_mask = (pd_series < lower) | (pd_series > upper)
    print(f"Tukey outliers with ± {coeff}*{range} ({pval_low}-{pval_high}):\
          {outlier_mask.sum()} days ({outlier_mask.mean():.2%} of data)")
    print(f'The outliers are: {pd_series[outlier_mask].values}')