"""
Final blended profitability score.

FinalScore(i) = 0.82*z(Economic) + 0.10*z(Engagement) + 0.04*z(Statistical)
                + 0.04*z(logit(MCMC)) - CollectionsFloor(i)

Combination architecture -- z-score, not percentile rank, and why that
distinction matters:

The four TOP-LEVEL layers here (Economic, Engagement, Statistical, MCMC) are
on genuinely different, non-comparable native scales: real dollars, a
z-score composite, another z-score composite, and a bounded probability.
They are standardized (z-scored) before combining -- scale-normalized, but
magnitude-preserving, unlike percentile rank, which would discard real
information about how strong each layer's signal actually is.

The four ECONOMIC sub-components (Revenue, CostEfficiency, Risk, Distress),
by contrast, are all denominated in the same real dollars and ARE directly
comparable -- summed in dollars inside economic_model.py before this module
ever sees them, specifically to preserve that comparability. Two real
mistakes shaped this design (see README):
  1. Applying a $-recognition rate to a raw points COUNT without converting
     units first (that single term ended up correlating 0.99 with the whole
     score -- a pure unit-mismatch bug).
  2. Percentile-ranking the four economic sub-components separately before
     combining, which is correct for genuinely incomparable sources but
     wrong here -- it caused f1 (revolve balance), the model's single
     strongest validated driver, to collapse from most-important to 17th of
     23 features, purely as a combination-method artifact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from economic_model import build_economic_score


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def build_engagement_view(X: pd.DataFrame, raw: pd.DataFrame, weights: dict) -> pd.Series:
    """f16 and the f6-f10 wallet index -- the two highest-weighted terms --
    use ONLY actually-observed data. Where the raw value was missing, the
    term contributes 0 (neutral), not an imputed guess, since an imputation
    error here would have outsized downstream effect given their weight.
    """
    f16_observed = raw["f16"].notna()
    f6_10_observed = raw["f6"].notna()  # f6-f10 verified missing/observed together

    z_f16 = z(X["f16"]) * f16_observed.astype(float)
    wallet_idx_raw = pd.concat([z(X[c]) for c in ["f6", "f7", "f8", "f9", "f10"]], axis=1).mean(axis=1)
    wallet_idx = wallet_idx_raw * f6_10_observed.astype(float)

    return (
        weights["wallet_idx"] * wallet_idx
        + weights["f16"] * z_f16
        + weights["f19"] * z(X["f19"])
        + weights["f20"] * z(X["f20"])
        - 0.5 * z(X["f2"])
        - 0.5 * z(X["f3"])
    )


def build_statistical_view(X: pd.DataFrame) -> pd.Series:
    """A purely statistical composite -- no dollar assumptions at all -- used
    as one of four independent 'opinions' in the final blend. Calibrated so
    the population's 15th percentile is the assumed break-even point (a
    business assumption that ~85% of Premier accounts are profitable)."""
    revenue_type_vars = ["f1", "f5", "f6", "f9", "f4", "f12", "f22", "f23"]
    cost_type_vars = ["f21", "f13", "f14", "f15", "f16", "f2", "f3"]
    raw_stat = (
        sum(z(X[v]) for v in revenue_type_vars)
        - sum(z(X[v]) for v in cost_type_vars)
        - z(X["f11"] * X["f17"])
    )
    return raw_stat - np.percentile(raw_stat, 15)


LAYER_WEIGHTS = {"economic": 0.82, "engagement": 0.10, "statistical": 0.04, "mcmc": 0.04}
COLLECTIONS_FLOOR_MULTIPLE = 1.1  # sized to exceed the full pre-floor score range, guaranteeing
                                  # every f3=1 member ranks below every f3=0 member


def build_final_score(
    X: pd.DataFrame, raw: pd.DataFrame, mcmc_posterior: np.ndarray, engagement_weights: dict,
) -> pd.DataFrame:
    econ = build_economic_score(X)
    engagement = build_engagement_view(X, raw, engagement_weights)
    statistical = build_statistical_view(X)

    z_economic = z(econ["economic_dollars"])
    z_engagement = z(engagement)
    z_statistical = z(statistical)
    mcmc_clipped = np.clip(mcmc_posterior, 1e-6, 1 - 1e-6)
    mcmc_logit = np.log(mcmc_clipped / (1 - mcmc_clipped))
    z_mcmc = z(pd.Series(mcmc_logit))

    combined = (
        LAYER_WEIGHTS["economic"] * z_economic
        + LAYER_WEIGHTS["engagement"] * z_engagement
        + LAYER_WEIGHTS["statistical"] * z_statistical
        + LAYER_WEIGHTS["mcmc"] * z_mcmc
    )

    score_range = combined.max() - combined.min()
    collections_floor = COLLECTIONS_FLOOR_MULTIPLE * score_range * X["f3"]
    final_score = combined - collections_floor

    return pd.DataFrame({
        "id": raw["id"],
        "revenue": econ["revenue"], "cost_efficiency": econ["cost_efficiency"],
        "risk": econ["risk"], "distress": econ["distress"],
        "engagement": engagement, "statistical": statistical, "mcmc": mcmc_posterior,
        "final_score": final_score,
    })


def main(imputed_path: Path, raw_path: Path, mcmc_path: Path,
         engagement_weights_path: Path, out_dir: Path) -> None:
    X = pd.read_csv(imputed_path)
    raw = pd.read_csv(raw_path)
    mcmc_posterior = np.load(mcmc_path)
    engagement_weights = pd.read_csv(engagement_weights_path, index_col=0).iloc[:, 0].to_dict()

    result = build_final_score(X, raw, mcmc_posterior, engagement_weights)
    result[["id", "final_score"]].to_csv(out_dir / "final_predictions.csv", index=False)
    result.to_csv(out_dir / "final_predictions_full.csv", index=False)

    print(result["final_score"].describe())
    print(f"\nTop 20% cutoff: {result['final_score'].quantile(0.80):.4f}")


if __name__ == "__main__":
    import sys
    imputed_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("imputed_features.csv")
    raw_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_r1.csv")
    mcmc_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("mcmc_posterior.npy")
    ew_path = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("calibration_engagement_weights.csv")
    out_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else Path(".")
    main(imputed_path, raw_path, mcmc_path, ew_path, out_dir)
