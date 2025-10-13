# eda_utils.py
import numpy as np

def tukey_outliers(pd_series, pval_low, pval_high, coeff, label=None):
    """
    Flag outliers via generalized Tukey's rule on a 1D series.

    Marks values outside [P_low - coeff*(P_high - P_low), P_high + coeff*(P_high - P_low)],
    where P_low and P_high are user-provided percentiles (e.g., 25 and 75 for IQR).

    Parameters
    - pd_series: 1D array-like (Pandas Series recommended)
    - pval_low, pval_high: lower/upper percentiles (0-100)
    - coeff: multiplier for the percentile range
    - label: optional column/series label for printing

    Returns
    - Boolean mask of the same length as `pd_series` (True = outlier)
    """
    assert pval_low < pval_high, "`pval_low` should be smaller than `pval_high`"
    perc_low, perc_high = np.percentile(pd_series, [pval_low, pval_high])
    iqr_range = perc_high - perc_low
    lower, upper = perc_low - coeff * iqr_range, perc_high + coeff * iqr_range
    outlier_mask = (pd_series < lower) | (pd_series > upper)
    if label is None:
        print(
            f"Tukey outliers with ± {coeff}*({pval_low}-{pval_high}), i.e., [<{lower}, >{upper}]:\n"
            f"\t{outlier_mask.sum()} data points ({outlier_mask.mean():.2%} of data)"
        )
        print(f"\tOutliers are: {pd_series[outlier_mask].values}")
        print()
    else:
        print(
            f"Tukey outliers for column \"{label}\" with ± {coeff}*({pval_low}-{pval_high}), i.e., [<{lower}, >{upper}]:\n"
            f"\t{outlier_mask.sum()} data points ({outlier_mask.mean():.2%} of data)"
        )
        print(f"\tOutliers for column \"{label}\" are: {pd_series[outlier_mask].values}")
        print()
    return outlier_mask

# Backward-compatibility alias (deprecated name)
iqr_mask = tukey_outliers
