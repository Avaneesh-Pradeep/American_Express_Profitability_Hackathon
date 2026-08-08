"""
Gibbs-sampled Bayesian 2-component Gaussian mixture model.

Rather than a point-estimate (EM) fit, this draws repeated samples from the
actual posterior distribution over the mixture's parameters (means,
variances, mixing weight) via Gibbs sampling with conjugate priors, and
averages the resulting cluster-membership probability over all post-burn-in
draws. f2/f3 (cancellation/collection calls) are deliberately excluded from
the clustering features: their rare, extreme binary values dominate an
unsupervised fit and cause it to simply rediscover "collections vs not"
rather than a genuine profitability split -- they're handled explicitly
elsewhere in the pipeline.

Honest limitation, verified empirically in this repo's development: at
N=500K, this Gibbs sampler's posterior is nearly indistinguishable from a
plain EM point estimate, because posterior parameter uncertainty shrinks
with sample size (a standard large-sample statistics result -- the
Bernstein-von Mises phenomenon). The value here is a properly calibrated,
non-overconfident posterior, not a large accuracy gain from "being
Bayesian" per se.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIXTURE_FEATURES = ["f1", "f5", "f21", "f13"]  # + risk_exposure, wallet_idx (engineered below)
N_ITER = 150
BURN_IN = 60
PRIOR_MEAN_VAR = 4.0   # tau^2: prior variance on component means
PRIOR_IG_SHAPE, PRIOR_IG_SCALE = 2.0, 2.0   # inverse-gamma prior on per-dim variance
PRIOR_BETA_ALPHA = 1.0  # symmetric beta prior on the mixing weight
SEED = 42


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def build_mixture_matrix(X: pd.DataFrame) -> np.ndarray:
    risk_exposure = X["f11"] * X["f17"]
    wallet_idx = pd.concat([z(X[c]) for c in ["f6", "f7", "f8", "f9", "f10"]], axis=1).mean(axis=1)
    cols = [z(X[c]) for c in MIXTURE_FEATURES] + [z(risk_exposure), z(wallet_idx)]
    return np.column_stack([c.values for c in cols])


def gibbs_sample_mixture(Y: np.ndarray, n_iter: int = N_ITER, burn_in: int = BURN_IN, seed: int = SEED) -> np.ndarray:
    N, D = Y.shape
    rng = np.random.default_rng(seed)
    mu0 = np.zeros(D)

    z_assign = (Y[:, 0] > np.median(Y[:, 0])).astype(int)
    z_sum = np.zeros(N)
    n_kept = 0

    for it in range(n_iter):
        mus = np.zeros((2, D))
        sig2 = np.zeros((2, D))
        for k in range(2):
            mask = z_assign == k
            nk = mask.sum()
            yk = Y[mask]
            ybar = yk.mean(axis=0) if nk > 0 else np.zeros(D)
            post_var = 1.0 / (nk + 1.0 / PRIOR_MEAN_VAR)
            post_mean = post_var * (nk * ybar + mu0 / PRIOR_MEAN_VAR)
            mus[k] = rng.normal(post_mean, np.sqrt(post_var))
            ss = ((yk - mus[k]) ** 2).sum(axis=0) if nk > 0 else np.zeros(D)
            a_post = PRIOR_IG_SHAPE + nk / 2
            b_post = PRIOR_IG_SCALE + ss / 2
            sig2[k] = 1.0 / rng.gamma(a_post, 1.0 / np.maximum(b_post, 1e-6))

        pi1 = rng.beta(PRIOR_BETA_ALPHA + (z_assign == 1).sum(), PRIOR_BETA_ALPHA + (z_assign == 0).sum())

        logp0 = -0.5 * np.sum(((Y - mus[0]) ** 2) / sig2[0] + np.log(2 * np.pi * sig2[0]), axis=1) + np.log(1 - pi1)
        logp1 = -0.5 * np.sum(((Y - mus[1]) ** 2) / sig2[1] + np.log(2 * np.pi * sig2[1]), axis=1) + np.log(pi1)
        m = np.maximum(logp0, logp1)
        p1 = np.exp(logp1 - m) / (np.exp(logp0 - m) + np.exp(logp1 - m))
        z_assign = (rng.random(N) < p1).astype(int)

        if it >= burn_in:
            z_sum += p1
            n_kept += 1
        if it % 30 == 0:
            print(f"  iter {it}: pi1={pi1:.3f}, mean p1={p1.mean():.3f}")

    return z_sum / n_kept


def main(imputed_path: Path, out_dir: Path) -> None:
    X = pd.read_csv(imputed_path)
    Y = build_mixture_matrix(X)
    print(f"Mixture feature matrix: {Y.shape}")
    posterior = gibbs_sample_mixture(Y)
    print("Posterior probability describe:\n", pd.Series(posterior).describe())
    np.save(out_dir / "mcmc_posterior.npy", posterior)


if __name__ == "__main__":
    import sys
    imputed_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("imputed_features.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    main(imputed_path, out_dir)
