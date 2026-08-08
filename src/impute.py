"""
Missing-value imputation.

Several features are missing in a way that isn't random: f17/f18 (lend line
amounts) are missing on the exact same rows, as are f4/f21 (rewards
balance/redeemed), and f6-f10 (category spend) are missing together as a
block (verified programmatically below: always all five or none).

Rather than mean/zero-filling (which flattens out real differences between
customers), this uses an IterativeImputer -- a MICE-style, round-robin
regression approach that predicts each missing feature from the other
features correlated with it, for that same customer. Fit on a 60K-row
sample for speed, then applied to the full file.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

NONNEG_COLS = [
    "f1", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f12", "f13", "f14",
    "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
]


def verify_missingness_pattern(df: pd.DataFrame) -> None:
    """Confirms f6-f10 are missing together (all 5 or none) -- this is what
    justifies treating them as a single block downstream rather than five
    independent columns."""
    block_missing = df[["f6", "f7", "f8", "f9", "f10"]].isna()
    all_five = (block_missing.sum(axis=1) == 5).sum()
    partial = ((block_missing.sum(axis=1) > 0) & (block_missing.sum(axis=1) < 5)).sum()
    assert partial == 0, "f6-f10 missingness is not all-or-nothing -- assumption violated"
    print(f"Verified: f6-f10 missing together on {all_five:,} rows, 0 partial cases")


def impute(data_path: Path, out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    verify_missingness_pattern(df)

    X = df.drop(columns=["id"]).copy()
    print("Missing value counts:\n", X.isna().sum())

    corr = X.corr()
    corr.to_csv(out_dir / "correlation_matrix.csv")

    t0 = time.time()
    sample_idx = X.sample(n=60_000, random_state=42).index
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=8,
        n_nearest_features=8,
        random_state=42,
        sample_posterior=False,
        tol=1e-2,
    )
    imputer.fit(X.loc[sample_idx])
    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
    print(f"Imputation complete in {time.time() - t0:.1f}s")

    for c in NONNEG_COLS:
        X_imp[c] = X_imp[c].clip(lower=0)
    X_imp["f11"] = X_imp["f11"].clip(lower=0, upper=1)  # behaves like a probability index

    X_imp.insert(0, "id", df["id"])
    X_imp.to_csv(out_dir / "imputed_features.csv", index=False)
    return X_imp


if __name__ == "__main__":
    import sys
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data_r1.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    impute(data_path, out_dir)
