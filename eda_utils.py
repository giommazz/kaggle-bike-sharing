# eda_utils.py
import numpy as np

def iqr_mask(pd_series, pval_low, pval_high, coeff, label=None):
    """
    Return a Boolean mask that flags Tukey outliers (typically values outside [Q1 - coeff*IQR, Q3 + coeff*IQR]).
    """
    assert pval_low < pval_high, "`pval_low` should be smaller than `pval_hihg`"
    perc_low, perc_high = np.percentile(pd_series, [pval_low, pval_high])
    range = perc_high - perc_low # compute interequartile range
    # compute bounds: beyond them lie outliers
    lower, upper = perc_low - coeff * range, perc_high + coeff * range
    outlier_mask = (pd_series < lower) | (pd_series > upper)
    if label == None:
        print(f"Tukey outliers with ± {coeff}*({pval_low}-{pval_high}), i.e., [<{lower}, >{upper}]:\
            \n\t{outlier_mask.sum()} data points ({outlier_mask.mean():.2%} of data)")
        print(f'\tOutliers are: {pd_series[outlier_mask].values}')
        print()
    else:
        print(f"Tukey outliers for column \"{label}\" with ± {coeff}*({pval_low}-{pval_high}), i.e., [<{lower}, >{upper}]:\
            \n\t{outlier_mask.sum()} data points ({outlier_mask.mean():.2%} of data)")
        print(f'\tOutliers for column \"{label}\" are: {pd_series[outlier_mask].values}')
        print()
    return outlier_mask