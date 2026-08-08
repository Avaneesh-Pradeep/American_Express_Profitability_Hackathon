"""
End-to-end pipeline runner.

    python run_pipeline.py --data data_r1.csv --out-dir output/

Runs, in order:
    1. Missing-value imputation (src/impute.py)
    2. Empirical calibration -- derives rho and engagement weights (src/calibration.py)
    3. Gibbs-sampled Bayesian mixture model (src/mcmc_mixture.py)
    4. Final blended score (src/final_score.py)
    5. Validation -- sensitivity analysis and feature importance (src/validate.py)

Each step's outputs are written to --out-dir and consumed by the next step,
so this is fully reproducible from raw data to final predictions and their
validation with a single command.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import impute
import calibration
import mcmc_mixture
import final_score
import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data_r1.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70 + "\nSTEP 1/5 -- Imputation\n" + "=" * 70)
    X_imputed = impute.impute(args.data, args.out_dir)

    print("\n" + "=" * 70 + "\nSTEP 2/5 -- Empirical calibration\n" + "=" * 70)
    calibration.main(
        args.out_dir / "imputed_features.csv",
        args.out_dir / "correlation_matrix.csv",
        args.out_dir,
    )

    print("\n" + "=" * 70 + "\nSTEP 3/5 -- MCMC mixture model\n" + "=" * 70)
    mcmc_mixture.main(args.out_dir / "imputed_features.csv", args.out_dir)

    print("\n" + "=" * 70 + "\nSTEP 4/5 -- Final blended score\n" + "=" * 70)
    final_score.main(
        args.out_dir / "imputed_features.csv",
        args.data,
        args.out_dir / "mcmc_posterior.npy",
        args.out_dir / "calibration_engagement_weights.csv",
        args.out_dir,
    )

    print("\n" + "=" * 70 + "\nSTEP 5/5 -- Validation\n" + "=" * 70)
    validate.main(
        args.out_dir / "imputed_features.csv",
        args.out_dir / "final_predictions.csv",
        args.out_dir,
    )

    print(f"\nPipeline complete. All outputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
