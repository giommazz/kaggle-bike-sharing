"""
Statistical diagnostics utilities.

Provides:
- compute_vif: compute (multivariate) Variance Inflation Factors on a numeric matrix.
- diagnose_multicollinearity: fit feature engineering on train-only (first fold),
  plot a correlation heatmap, print top VIFs, and return the VIF table.

Notes
- Designed as diagnostics (train-only) to avoid leakage and keep costs low.
- For tree models, treat collinearity findings as informational; for linear/GLM
  models, consider pruning or regularization when VIFs are very large.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from ml_utils import BikeFeatureEngineer
from utils import savefig_pdf


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) per numeric feature in `df`.

    - Adds an explicit intercept column to emulate OLS-style VIF.
    - Returns a DataFrame sorted by VIF descending with columns ['feature','VIF'].

    Parameters
    - df: DataFrame with engineered features (numeric and non-numeric allowed).

    Returns
    - DataFrame with VIF per numeric feature (intercept excluded).
    """
    Xn = df.select_dtypes(include=[np.number]).copy()
    if Xn.empty:
        return pd.DataFrame(columns=["feature", "VIF"])  # nothing to compute

    Xn.insert(0, "_intercept", 1.0)  # explicit intercept for OLS-style computation
    vifs = []
    A = Xn.to_numpy()
    cols = Xn.columns.tolist()
    for j, col in enumerate(cols[1:], start=1):  # skip intercept at 0
        yj = A[:, j]
        Xj = np.delete(A, j, axis=1)
        r2 = LinearRegression(fit_intercept=False).fit(Xj, yj).score(Xj, yj)
        vif = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)  # guard against numerical 1
        vifs.append((col, float(vif)))
    return pd.DataFrame(vifs, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)


def diagnose_multicollinearity(
    X: pd.DataFrame,
    cv,
    *,
    ar_lags=None,
    ar_rolls=None,
    figdir=None,
    title_prefix: str = "Post-FE correlations (train-only)"
) -> pd.DataFrame:
    """
    Run train-only multicollinearity diagnostics for one representative fold:
    - fit BikeFeatureEngineer on TRAIN rows,
    - compute and plot correlation heatmap (lower-triangle shown),
    - compute VIF on numeric engineered features,
    - return the VIF table.

    Parameters
    - X: full feature DataFrame (time-ordered) including 'cnt' for AR construction.
    - cv: a splitter yielding (train_idx, test_idx); first fold is used.
    - ar_lags, ar_rolls: optional AR configuration passed to BikeFeatureEngineer.
    - figdir: directory path to save the heatmap PDF (if provided).
    - title_prefix: title prefix for the correlation heatmap.

    Returns
    - VIF DataFrame for engineered TRAIN matrix.
    """
    tr_idx, _ = next(cv.split(X))  # first fold only, TRAIN rows
    X_tr = X.iloc[tr_idx].copy()

    fe = BikeFeatureEngineer(ar_lags=ar_lags, ar_rolls=ar_rolls)
    Z_tr = clone(fe).fit(X_tr).transform(X_tr)  # engineered train matrix

    # 1) Correlation heatmap (pairwise) with lower triangle displayed
    corr = Z_tr.corr(numeric_only=True)
    if corr.size:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # hide upper triangle
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            corr, mask=mask, cmap="vlag", center=0, vmin=-1, vmax=1,
            square=True, linewidths=.5, linecolor='white',
            annot=True, fmt=".2f", annot_kws={"size":8}, ax=ax
        )
        ax.set_title(f"{title_prefix}. AR: lags={ar_lags}, rolls={ar_rolls}")
        plt.tight_layout()
        if figdir is not None:
            savefig_pdf(
                f"fig_corr_train_only_AR_{str(ar_lags).replace(' ','')}_{str(ar_rolls).replace(' ','')}",
                figdir,
                fig,
            )

    # 2) VIF (multivariate collinearity) on numeric features
    vif_df = compute_vif(Z_tr)
    print("Top VIFs (train-only):")
    print(vif_df.head(20).to_string(index=False))
    return vif_df

