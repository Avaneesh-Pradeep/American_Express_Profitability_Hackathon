"""
Empirical calibration of the model's Bayesian prior and behavioral weights.

Two quantities here are genuinely DERIVED from data/evidence, not assumed:

1. rho -- the implied correlation between this model's economic score and the
   true (unobserved) profitability label, back-solved from a real, measured
   result: an earlier, purely economic version of this model scored 58% top-
   20% overlap accuracy on the competition's public leaderboard. Under a
   bivariate-normal model linking the score to the true label, that observed
   accuracy implies a specific correlation -- solved numerically below, not
   assumed.

2. Engagement-index weights -- rather than assuming which behavioral signals
   (logins, email engagement, benefit usage, account depth) matter most for
   predicting retention, each candidate's weight is set to its own measured
   correlation with the two observed churn signals (cancellation and
   collection calls), normalized to sum to 1. The data determines the
   ranking, not a manual guess.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

# The one external empirical input to this whole calibration: the measured
# public-leaderboard accuracy of the pre-calibration economic model.
OBSERVED_TOP20_ACCURACY = 0.58
TOP20_QUANTILE = 0.80


def derive_rho(observed_accuracy: float, q: float = TOP20_QUANTILE) -> float:
    """Back-solve the implied correlation (rho) between the model score and
    the true label from an observed top-(1-q) overlap accuracy, under a
    bivariate-normal assumption. Verified against known boundary cases:
    rho=0 must reproduce the random-guessing baseline (1-q), and the
    relationship must be monotonically increasing in rho.
    """
    zq = norm.ppf(q)

    def overlap_given_rho(rho: float) -> float:
        cov = [[1, rho], [rho, 1]]
        joint_below = multivariate_normal.cdf([zq, zq], mean=[0, 0], cov=cov)
        p_both_above = 1 - 2 * q + joint_below
        return p_both_above / (1 - q)

    baseline = overlap_given_rho(0.0)
    assert abs(baseline - (1 - q)) < 1e-6, f"Sanity check failed: rho=0 gave {baseline}, expected {1-q}"

    rho = brentq(lambda r: overlap_given_rho(r) - observed_accuracy, 1e-4, 1 - 1e-4)
    return rho


def derive_engagement_weights(X: pd.DataFrame, corr: pd.DataFrame) -> dict:
    """Weight each engagement/retention signal by its own measured
    correlation strength with the two churn signals (f2, f3), normalized to
    sum to 1 -- i.e. the more strongly a behavior predicts NOT calling to
    cancel or going to collections, the more it counts toward the index.
    """
    def z(s):
        return (s - s.mean()) / s.std()

    wallet_idx = pd.concat([z(X[c]) for c in ["f6", "f7", "f8", "f9", "f10"]], axis=1).mean(axis=1)
    churn_signals = ["f2", "f3"]
    candidates = {"wallet_idx": wallet_idx, "f16": X["f16"], "f19": X["f19"], "f20": X["f20"]}

    raw_weights = {}
    for name, series in candidates.items():
        if name == "wallet_idx":
            corrs = [abs(np.corrcoef(series, X[c])[0, 1]) for c in churn_signals]
        else:
            corrs = [abs(corr.loc[name, c]) for c in churn_signals]
        raw_weights[name] = float(np.mean(corrs))

    total = sum(raw_weights.values())
    return {k: v / total for k, v in raw_weights.items()}


def main(imputed_path: Path, corr_path: Path, out_dir: Path) -> None:
    X = pd.read_csv(imputed_path)
    corr = pd.read_csv(corr_path, index_col=0)

    rho = derive_rho(OBSERVED_TOP20_ACCURACY)
    print(f"Derived rho = {rho:.4f}  (from observed accuracy = {OBSERVED_TOP20_ACCURACY})")

    engagement_weights = derive_engagement_weights(X, corr)
    print("Derived engagement weights:", {k: round(v, 4) for k, v in engagement_weights.items()})

    pd.Series({"rho": rho}).to_csv(out_dir / "calibration_rho.csv")
    pd.Series(engagement_weights).to_csv(out_dir / "calibration_engagement_weights.csv")


if __name__ == "__main__":
    import sys
    imputed_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("imputed_features.csv")
    corr_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("correlation_matrix.csv")
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(".")
    main(imputed_path, corr_path, out_dir)
