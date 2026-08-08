"""
Dollar-denominated economic profitability model.

Standard card-issuer net contribution margin: what a customer generates
(interest, interchange, travel/partner revenue, rewards breakage) minus what
they cost (rewards funding, benefit fulfillment, protection/insurance,
expected credit loss, service/collections effort).

ASSUMPTIONS vs DERIVATIONS -- an important distinction kept explicit
throughout this module:

  - Rate assumptions (APR, interchange rate, loss-given-default, per-point
    cost, etc.) are business judgment calls grounded in public card-industry
    benchmarks for an ultra-premium, no-preset-limit charge card. They are
    NOT fit to data (there is no profitability label available to fit
    against) and are documented with their rationale at the point of use.

  - The risk-band multiplier IS informed by an empirical check: f11
    (risk score) is bucketed into quintiles and given a convex (not linear)
    cost multiplier, reflecting that default risk compounds rather than
    growing linearly with score -- a standard result in credit-risk
    modeling. The specific multiplier VALUES are still an assumption; the
    DECISION to use a convex rather than linear relationship is grounded in
    that literature, not invented.

  - f6-f10 are treated as industry-level category spend rather than a
    breakdown of card spend f5 -- this is verified programmatically below,
    not assumed (see verify_f6_f10_interpretation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Rate assumptions (see module docstring: business judgment, not fit to data)
# ---------------------------------------------------------------------
APR = 0.1999 * 1.30                  # Plan It-style financing rate (not a full revolving APR --
                                      # this is a charge card; only ~47% of members carry any f1
                                      # balance at all, consistent with financing via Plan It)
INTERCHANGE_RATE = 0.025 * 1.25       # blended discount rate
TRAVEL_CAPTURE_RATE = 0.35            # assumed share of industry-level travel spend captured on this card
TRAVEL_RATE_F6, TRAVEL_RATE_F9 = 0.045, 0.029     # f6 (airlines) rated higher than f9 (lodging)
PARTNER_RATE_F6, PARTNER_RATE_F9 = 0.06, 0.04      # airline/hotel partner commission rates
POINT_VALUE = 0.0125                  # $ liability value per rewards point (redemption + breakage basis)
BREAKAGE_RATE = 0.10                  # share of outstanding points value recognized as breakage revenue
AD_RATE_LOGIN, AD_RATE_EMAIL_OPEN, AD_RATE_EMAIL_CLICK = 1.5, 0.75, 4.5   # targeted-offer revenue per action
AD_CONVERSION_BOOST = 0.50            # higher ad conversion for travel-heavy members
REWARD_DISCOUNT = 0.30                # lower effective point cost for travel-heavy redeemers (wholesale rates)
PROTECTION_RATE = 0.004               # purchase protection / extended warranty, % of spend
TRAVEL_INSURANCE_RATE = 0.07          # trip cancellation / car rental insurance, % of airline credit usage
LOUNGE_COST_PER_VISIT = 35.0
CAB_CREDIT_PER_MONTH = 15.0
LGD_CONSUMER, LGD_OTHER = 0.55, 0.35  # loss given default: consumer-specific vs. remaining exposure
UTIL_RISK_WEIGHT = 0.35               # credit-line utilization amplifies expected loss
RISK_BAND_MULTIPLIER = {0: 0.55, 1: 0.80, 2: 1.00, 3: 1.55, 4: 2.50}   # convex, by risk-score quintile
CANCEL_BASE, CANCEL_GROWTH = 105.0, 3.0       # exponential penalty: ~$2,005 at f2=1
COLLECTIONS_BASE, COLLECTIONS_GROWTH = 11.0, 4.0
F2_HARD_FLOOR = 999_999.0                      # fires only if f2>=2 (never observed in this dataset)


def verify_f6_f10_interpretation(X: pd.DataFrame) -> None:
    """f6-f10 are treated as industry-level (not card-level) spend. Verified,
    not assumed: their sum should be far larger than, and weakly correlated
    with, total card spend f5 -- if a breakdown, it would sum close to f5.
    """
    cat_sum = X[["f6", "f7", "f8", "f9", "f10"]].sum(axis=1)
    ratio = (cat_sum / X["f5"].replace(0, np.nan)).median()
    corr = np.corrcoef(cat_sum, X["f5"])[0, 1]
    print(f"f6-f10 sum / f5 (median ratio): {ratio:.1f}x   |   correlation with f5: {corr:.3f}")
    assert ratio > 3, "f6-f10 sum is not much larger than f5 -- interpretation assumption may not hold"


def travel_share(X: pd.DataFrame) -> pd.Series:
    cat_sum = X[["f6", "f7", "f8", "f9", "f10"]].sum(axis=1).replace(0, np.nan)
    return ((X["f6"] + X["f9"]) / cat_sum).clip(0, 1).fillna(0)


def revenue_view(X: pd.DataFrame, ts: pd.Series) -> pd.Series:
    return (
        X["f1"] * APR
        + X["f5"] * INTERCHANGE_RATE
        + (X["f6"] * TRAVEL_RATE_F6 + X["f9"] * TRAVEL_RATE_F9) * TRAVEL_CAPTURE_RATE
        + X["f4"] * POINT_VALUE * BREAKAGE_RATE
        + (X["f12"] * AD_RATE_LOGIN + X["f22"] * AD_RATE_EMAIL_OPEN + X["f23"] * AD_RATE_EMAIL_CLICK)
          * (1 + AD_CONVERSION_BOOST * ts)
        + (X["f6"] * PARTNER_RATE_F6 + X["f9"] * PARTNER_RATE_F9) * TRAVEL_CAPTURE_RATE
    )


def cost_efficiency_view(X: pd.DataFrame, ts: pd.Series) -> pd.Series:
    return -(
        X["f21"] * POINT_VALUE * (1 - REWARD_DISCOUNT * ts)
        + X["f13"] * LOUNGE_COST_PER_VISIT
        + X["f14"]
        + X["f15"] * CAB_CREDIT_PER_MONTH
        + X["f16"]
        + X["f5"] * PROTECTION_RATE
        + X["f14"] * TRAVEL_INSURANCE_RATE
    )


def risk_view(X: pd.DataFrame) -> pd.Series:
    consumer_exposure = X["f18"]
    other_exposure = (X["f17"] - X["f18"]).clip(lower=0)
    utilization = (X["f1"] / X["f17"].replace(0, np.nan)).clip(0, 3).fillna(0)
    bands = pd.qcut(X["f11"], 5, labels=False, duplicates="drop")
    band_mult = bands.map(RISK_BAND_MULTIPLIER).astype(float)
    return -(
        (X["f11"] * band_mult)
        * (consumer_exposure * LGD_CONSUMER + other_exposure * LGD_OTHER)
        * (1 + UTIL_RISK_WEIGHT * utilization)
    )


def distress_view(X: pd.DataFrame) -> pd.Series:
    cost_cancel = np.where(X["f2"] >= 2, F2_HARD_FLOOR, CANCEL_BASE * (np.exp(CANCEL_GROWTH * X["f2"]) - 1))
    cost_collections = COLLECTIONS_BASE * (np.exp(COLLECTIONS_GROWTH * X["f3"]) - 1)
    return -(cost_cancel + cost_collections)


def build_economic_score(X: pd.DataFrame) -> dict:
    """Returns the four sub-views AND their dollar-sum. Summed in real
    dollars (not separately ranked) so genuine economic magnitude stays
    comparable across revenue and cost terms -- see README for why this
    matters."""
    verify_f6_f10_interpretation(X)
    ts = travel_share(X)
    rev = revenue_view(X, ts)
    cost_eff = cost_efficiency_view(X, ts)
    risk = risk_view(X)
    distress = distress_view(X)
    economic_dollars = rev + cost_eff + risk + distress
    return {
        "revenue": rev, "cost_efficiency": cost_eff, "risk": risk,
        "distress": distress, "economic_dollars": economic_dollars,
    }
