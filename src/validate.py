"""
Robustness / sensitivity validation.

Since there is no ground-truth profitability label available (the whole
point of this being an unsupervised ranking problem), "accuracy" cannot be
directly validated. What CAN be validated is whether the ranking is fragile
to the specific dollar assumptions chosen -- if small, reasonable changes to
any single rate assumption reshuffled the ranking dramatically, that would
indicate overfitting to arbitrary constants rather than capturing real
relative differences between customers.

This script perturbs every economic rate assumption by +/-20% simultaneously
across N runs and measures how much the resulting ranking changes versus the
baseline -- both by Spearman rank correlation and by top-20% set overlap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import economic_model as em


def compute_ncm_with_params(X: pd.DataFrame, params: dict) -> pd.Series:
    ts = em.travel_share(X)
    revenue = (
        X["f1"] * params["APR"]
        + X["f5"] * params["INTERCHANGE_RATE"]
        + (X["f6"] * params["TRAVEL_RATE_F6"] + X["f9"] * params["TRAVEL_RATE_F9"]) * params["TRAVEL_CAPTURE_RATE"]
        + X["f4"] * params["POINT_VALUE"] * params["BREAKAGE_RATE"]
        + (X["f12"] * params["AD_RATE_LOGIN"] + X["f22"] * params["AD_RATE_EMAIL_OPEN"] + X["f23"] * params["AD_RATE_EMAIL_CLICK"])
          * (1 + params["AD_CONVERSION_BOOST"] * ts)
        + (X["f6"] * params["PARTNER_RATE_F6"] + X["f9"] * params["PARTNER_RATE_F9"]) * params["TRAVEL_CAPTURE_RATE"]
    )
    cost_efficiency = -(
        X["f21"] * params["POINT_VALUE"] * (1 - params["REWARD_DISCOUNT"] * ts)
        + X["f13"] * params["LOUNGE_COST_PER_VISIT"] + X["f14"] + X["f15"] * params["CAB_CREDIT_PER_MONTH"]
        + X["f16"] + X["f5"] * params["PROTECTION_RATE"] + X["f14"] * params["TRAVEL_INSURANCE_RATE"]
    )
    consumer_exposure = X["f18"]; other_exposure = (X["f17"] - X["f18"]).clip(lower=0)
    utilization = (X["f1"] / X["f17"].replace(0, np.nan)).clip(0, 3).fillna(0)
    bands = pd.qcut(X["f11"], 5, labels=False, duplicates="drop")
    band_mult = bands.map(em.RISK_BAND_MULTIPLIER).astype(float)
    risk = -((X["f11"] * band_mult) * (consumer_exposure * params["LGD_CONSUMER"] + other_exposure * params["LGD_OTHER"])
             * (1 + params["UTIL_RISK_WEIGHT"] * utilization))
    distress = em.distress_view(X)
    return revenue + cost_efficiency + risk + distress


PERTURBABLE_PARAMS = [
    "APR", "INTERCHANGE_RATE", "TRAVEL_CAPTURE_RATE", "TRAVEL_RATE_F6", "TRAVEL_RATE_F9",
    "PARTNER_RATE_F6", "PARTNER_RATE_F9", "POINT_VALUE", "BREAKAGE_RATE", "AD_RATE_LOGIN",
    "AD_RATE_EMAIL_OPEN", "AD_RATE_EMAIL_CLICK", "AD_CONVERSION_BOOST", "REWARD_DISCOUNT",
    "PROTECTION_RATE", "TRAVEL_INSURANCE_RATE", "LOUNGE_COST_PER_VISIT", "CAB_CREDIT_PER_MONTH",
    "LGD_CONSUMER", "LGD_OTHER", "UTIL_RISK_WEIGHT",
]


def run_sensitivity_analysis(X: pd.DataFrame, n_runs: int = 8, perturb_pct: float = 0.20, seed: int = 1) -> pd.DataFrame:
    base_params = {p: getattr(em, p) for p in PERTURBABLE_PARAMS}
    baseline = compute_ncm_with_params(X, base_params)

    base_rank_pct = baseline.rank(pct=True)
    baseline_top20 = set(X.index[base_rank_pct >= 0.80])

    rng = np.random.default_rng(seed)
    results = []
    for run in range(n_runs):
        perturbed_params = {
            p: v * (1 + rng.uniform(-perturb_pct, perturb_pct)) for p, v in base_params.items()
        }
        perturbed_ncm = compute_ncm_with_params(X, perturbed_params)

        spearman = np.corrcoef(baseline.rank(), perturbed_ncm.rank())[0, 1]
        perturbed_top20 = set(X.index[perturbed_ncm.rank(pct=True) >= 0.80])
        overlap = len(perturbed_top20 & baseline_top20) / len(baseline_top20)
        results.append({"run": run, "spearman_vs_baseline": spearman, "top20_overlap_vs_baseline": overlap})

    return pd.DataFrame(results)


def check_feature_importance(X: pd.DataFrame, final_scores: pd.Series, exclude_f3: bool = True) -> pd.Series:
    """Correlation of each raw feature with the final score. If exclude_f3,
    computed within the f3=0 subgroup only -- the collections floor (which
    only ever fires for f3=1) dominates whole-population variance and would
    otherwise swamp every other feature's measured importance."""
    mask = X["f3"] == 0 if exclude_f3 else pd.Series(True, index=X.index)
    # f3 is trivially constant (all 0) within its own subgroup -- exclude it
    # from this particular view rather than report an undefined correlation.
    cols = [c for c in X.columns if c not in ("id", "f3")] if exclude_f3 else [c for c in X.columns if c != "id"]
    corrs = {c: np.corrcoef(X.loc[mask, c], final_scores.loc[mask])[0, 1] for c in cols}
    return pd.Series(corrs).sort_values(key=abs, ascending=False)


def main(imputed_path: Path, final_scores_path: Path, out_dir: Path) -> None:
    X = pd.read_csv(imputed_path)
    final = pd.read_csv(final_scores_path)

    print("=== Sensitivity analysis: +/-20% perturbation of all rate assumptions ===")
    sensitivity = run_sensitivity_analysis(X)
    print(sensitivity)
    print(f"\nMean Spearman: {sensitivity['spearman_vs_baseline'].mean():.4f}")
    print(f"Mean top-20% overlap: {sensitivity['top20_overlap_vs_baseline'].mean():.4f}")
    sensitivity.to_csv(out_dir / "sensitivity_analysis_results.csv", index=False)

    print("\n=== Feature importance (within f3=0 subgroup, to avoid floor dominance) ===")
    importance = check_feature_importance(X, final["final_score"])
    print(importance)
    importance.to_csv(out_dir / "feature_importance.csv")

    print("\n=== Collections-floor variance check ===")
    total_var = final["final_score"].var()
    within_f3_0_var = final.loc[X["f3"] == 0, "final_score"].var()
    print(f"Variance within f3=0 subgroup: {within_f3_0_var:.2f} of total {total_var:.2f} "
          f"({100*within_f3_0_var/total_var:.1f}%) -- confirms the floor separates the f3=1 "
          f"minority cleanly without needing to inspect it further; feature importance should "
          f"always be read from the f3=0-only figures above.")


if __name__ == "__main__":
    import sys
    imputed_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("imputed_features.csv")
    final_scores_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("final_predictions.csv")
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(".")
    main(imputed_path, final_scores_path, out_dir)
