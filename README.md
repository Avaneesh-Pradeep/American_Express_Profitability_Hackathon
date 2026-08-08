# Customer Profitability Ranking — Unsupervised, Bayesian-Calibrated Scoring Model

A profitability ranking model built for a 500K-row anonymized credit card dataset with
**no ground-truth label available**. Ranks customers by estimated profitability by
combining a dollar-denominated economic P&L model with a Bayesian mixture model, using
an empirically-derived calibration to weight the two.

## The problem

Given ~23 anonymized customer attributes (spend, revolve balance, benefit usage,
risk score, engagement signals) and **no profitability label of any kind**, rank all
customers by estimated profitability. The only feedback available during development
was a single measured accuracy number from an external evaluation — everything else
had to be derived from the data itself or grounded in domain-standard financial
assumptions.

## Approach

**1. Correlation-aware imputation.** ~10-60% missingness on several features, with
non-random patterns (e.g. five columns are always missing together as a block, verified
programmatically rather than assumed). Uses an `IterativeImputer` (MICE-style,
regression-based) rather than mean/zero-fill, so a missing value is predicted from the
features actually correlated with it.

**2. A dollar-denominated economic model** — real net-contribution-margin logic
(interest income, interchange, rewards economics, risk-adjusted expected loss) with
explicit, documented rate assumptions grounded in industry benchmarks.

**3. A Gibbs-sampled Bayesian Gaussian mixture model**, implemented from conjugate
priors (not a library call) — a second, independent, purely unsupervised view of the
same customers, used as a component in the final blend.

**4. Empirical calibration, not guessing.** A single measured accuracy result (58%
top-20% overlap against a held-out evaluation) is converted into a Bayesian prior via a
bivariate-normal correlation model, solved numerically — this is the one place real
external feedback enters the pipeline, and it's converted into a precise, reusable
calibration constant rather than just noted as a fact. Engagement-index weights are
similarly derived from measured correlation with churn signals, not assumed.

**5. Magnitude-aware ensembling.** Four independent scoring "views" (dollar economics,
behavioral engagement, a statistical composite, and the Bayesian mixture posterior) are
combined by z-score standardization — deliberately *not* percentile rank, and not raw
units. This choice was reached after finding, and fixing, two real bugs during
development (see **Key Engineering Decisions** below) — both are documented because
they're the most interesting/transferable part of this project.

**6. Full validation as code, not prose.** Every dollar assumption is stress-tested by
simultaneous ±20% perturbation across repeated runs, with rank correlation and top-20%
overlap measured programmatically (`src/validate.py`) — the robustness claims in this
README are outputs of a script in this repo, not assertions.

## Key engineering decisions (the interesting part)

**Bug 1 — unit mismatch silently dominating the score.** An early version applied a
revenue-recognition rate directly to a raw *points count* instead of converting it to a
dollar value first. Because points balances run 10,000x larger than dollar figures,
that single mis-scaled term ended up correlating **0.99** with the entire model output
— the "model" was effectively just sorting by one column. Root-caused by checking each
term's correlation with the final score individually; fixed by enforcing consistent
unit conversion before any term touches the combined score.

**Bug 2 — the right robustness technique used on the wrong inputs.** A later version
combined four dollar-denominated sub-components (revenue, cost, risk, distress) by
percentile-ranking each one separately before summing them. Percentile rank is the
correct way to combine sources with *arbitrary, non-comparable* scales — but these four
sources are all real dollars and *are* directly comparable (a dollar of risk cost really
is worth a dollar of revenue). Rank-transforming them anyway silently discarded that
comparability, and caused the single strongest, most validated driver in the whole
model (revolve balance) to collapse from the most important feature to 17th of 23,
purely as an artifact of the combination method — confirmed directly by measuring each
feature's correlation with the output before and after. Fixed by summing dollar-scale
components in dollars, and reserving rank/z-score treatment for combining sources that
are *genuinely* on different scales (e.g. dollars vs. a bounded probability).

## Repo structure

```
├── run_pipeline.py         # single entry point, runs everything end-to-end
├── requirements.txt
└── src/
    ├── impute.py            # MICE-style missing value imputation
    ├── calibration.py       # derives the Bayesian prior (rho) and engagement weights
    ├── mcmc_mixture.py       # Gibbs-sampled Bayesian Gaussian mixture (from scratch)
    ├── economic_model.py     # dollar-denominated P&L: revenue, cost, risk, distress
    ├── final_score.py        # combines all four views into the final ranking
    └── validate.py           # sensitivity analysis + feature importance, as code
```

## Running it

```bash
pip install -r requirements.txt
python run_pipeline.py --data data_r1.csv --out-dir output/
```

Produces, in `output/`: imputed features, derived calibration constants, the MCMC
posterior, the final ranked predictions, and a full sensitivity/validation report —
reproducible end-to-end from raw data with one command.

## Stack

Python · pandas · NumPy · scikit-learn (`IterativeImputer`) · SciPy (numerical root-
finding for Bayesian calibration, bivariate-normal integration) · from-scratch Gibbs
sampling (no PyMC/Stan dependency — implemented directly to show the mechanics)
