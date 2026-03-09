"""
Synthetic insurance dataset generator.

Generates a realistic motor insurance panel suitable for demonstrating
and testing the SDID estimator. The data-generating process follows a
factor model:

  Y_it = alpha_i + beta_t + tau * D_it + epsilon_it

where:
  alpha_i = segment-level fixed effect (size, risk profile)
  beta_t  = common time trend (market claims inflation)
  tau     = true ATT (known, for ground truth testing)
  D_it    = treatment indicator (1 if segment i received rate change by t)
  epsilon_it ~ N(0, sigma^2)

The rate change affects loss ratio because it changes premiums (denominator)
without immediately changing incurred claims (numerator). A rate increase of
+10% should, all else equal, reduce loss ratio by ~9% (100/110 - 1).

Segments are defined by {territory, age_band, channel} — realistic motor
insurance segmentation.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def make_synthetic_motor_panel(
    n_segments: int = 100,
    n_periods: int = 12,
    treat_fraction: float = 0.3,
    true_att: float = -0.08,
    treatment_period: int = 8,
    noise_sd: float = 0.05,
    base_loss_ratio: float = 0.72,
    trend_per_period: float = 0.005,
    random_seed: int = 42,
    staggered: bool = False,
    n_stagger_cohorts: int = 3,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Generate synthetic motor insurance data.

    Returns (policy_df, claims_df, rate_log_df) as Polars DataFrames.

    The true ATT is `true_att` — useful for checking estimator performance.

    Parameters
    ----------
    n_segments : int
        Number of segment × channel combinations. Default: 100.
    n_periods : int
        Number of quarterly periods. Default: 12 (3 years).
    treat_fraction : float
        Fraction of segments that receive the rate change. Default: 0.3.
    true_att : float
        True causal effect on loss ratio. Default: -0.08 (8pp reduction).
    treatment_period : int
        Period at which treatment starts (1-indexed). Default: 8.
    noise_sd : float
        Standard deviation of segment-period noise. Default: 0.05.
    base_loss_ratio : float
        Mean loss ratio before treatment. Default: 0.72.
    trend_per_period : float
        Market-wide claims inflation trend per period. Default: 0.005.
    random_seed : int
        Reproducibility seed. Default: 42.
    staggered : bool
        If True, generate staggered adoption (different segments treated at
        different periods). Default: False.
    n_stagger_cohorts : int
        Number of staggered cohorts (if staggered=True). Default: 3.

    Returns
    -------
    policy_df : Polars DataFrame
        Columns: segment_id, period, earned_premium, earned_exposure
    claims_df : Polars DataFrame
        Columns: segment_id, period, incurred_claims, claim_count
    rate_log_df : Polars DataFrame
        Columns: segment_id, first_treated_period
    """
    rng = np.random.default_rng(random_seed)

    # --- Segment metadata ---
    territories = ["north_west", "north_east", "yorkshire", "east_midlands",
                   "west_midlands", "south_east", "south_west", "london",
                   "wales", "scotland"]
    age_bands = ["17-25", "26-35", "36-50", "51-65", "66+"]
    channels = ["direct", "broker", "aggregator"]

    segment_ids = []
    for i in range(n_segments):
        t = territories[i % len(territories)]
        a = age_bands[(i // len(territories)) % len(age_bands)]
        c = channels[(i // (len(territories) * len(age_bands))) % len(channels)]
        segment_ids.append(f"{t}_{a}_{c}")

    # Deduplicate in case of overlap
    segment_ids = [f"seg_{i:03d}_{sid}" for i, sid in enumerate(segment_ids)]

    # Segment-level fixed effects (log-normal around 0)
    alpha = rng.normal(0, 0.08, n_segments)

    # Time trend (common market claims inflation)
    beta = np.arange(n_periods) * trend_per_period

    # Treatment assignment
    n_treated = max(1, int(n_segments * treat_fraction))
    treated_idx = rng.choice(n_segments, size=n_treated, replace=False)
    treated_set = set(treated_idx)

    # Staggered cohort assignment
    if staggered:
        cohort_periods = sorted(rng.choice(
            range(treatment_period, treatment_period + n_stagger_cohorts * 2, 2),
            size=n_stagger_cohorts,
            replace=False,
        ))
        cohort_assignments = {}
        treated_list = list(treated_idx)
        chunk_size = max(1, len(treated_list) // n_stagger_cohorts)
        for ci, cp in enumerate(cohort_periods):
            start = ci * chunk_size
            end = start + chunk_size if ci < n_stagger_cohorts - 1 else len(treated_list)
            for idx in treated_list[start:end]:
                cohort_assignments[idx] = cp
    else:
        cohort_assignments = {idx: treatment_period for idx in treated_idx}

    # --- Generate panel data ---
    policy_rows = []
    claims_rows = []

    for i, seg_id in enumerate(segment_ids):
        # Exposure (earned premium) — varies by segment and period
        base_premium = rng.uniform(200_000, 2_000_000)
        base_exposure = rng.uniform(500, 5_000)

        first_treated = cohort_assignments.get(i, None)

        for t in range(1, n_periods + 1):
            # Exposure grows slightly over time
            exposure_t = base_exposure * (1 + 0.01 * t) + rng.normal(0, base_exposure * 0.02)
            premium_t = base_premium * (1 + 0.02 * t) + rng.normal(0, base_premium * 0.03)
            exposure_t = max(exposure_t, 10.0)
            premium_t = max(premium_t, 10_000.0)

            # True loss ratio = base + segment FE + time trend + treatment + noise
            is_treated = (first_treated is not None) and (t >= first_treated)
            lr = (
                base_loss_ratio
                + alpha[i]
                + beta[t - 1]
                + (true_att if is_treated else 0.0)
                + rng.normal(0, noise_sd)
            )
            lr = max(lr, 0.05)  # floor at 5%

            incurred = lr * premium_t
            claim_count = int(max(0, rng.poisson(exposure_t * 0.08)))

            policy_rows.append({
                "segment_id": seg_id,
                "period": t,
                "earned_premium": round(premium_t, 2),
                "earned_exposure": round(exposure_t, 2),
            })
            claims_rows.append({
                "segment_id": seg_id,
                "period": t,
                "incurred_claims": round(incurred, 2),
                "claim_count": claim_count,
            })

    # Rate change log
    rate_log_rows = []
    for i, seg_id in enumerate(segment_ids):
        if i in cohort_assignments:
            rate_log_rows.append({
                "segment_id": seg_id,
                "first_treated_period": cohort_assignments[i],
            })

    policy_df = pl.DataFrame(policy_rows)
    claims_df = pl.DataFrame(claims_rows)
    rate_log_df = pl.DataFrame(rate_log_rows)

    return policy_df, claims_df, rate_log_df


def make_synthetic_panel_direct(
    n_control: int = 60,
    n_treated: int = 20,
    t_pre: int = 8,
    t_post: int = 4,
    true_att: float = -0.06,
    noise_sd: float = 0.03,
    base_lr: float = 0.70,
    trend: float = 0.003,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Generate a balanced panel directly (skipping raw policy/claims tables).

    Useful for unit testing the SDID estimator without going through the
    full panel construction pipeline.

    Returns a Polars DataFrame with columns:
      segment_id, period, loss_ratio, earned_premium, earned_exposure,
      incurred_claims, claim_count, first_treated_period, treated, cohort
    """
    rng = np.random.default_rng(random_seed)
    T = t_pre + t_post
    N = n_control + n_treated

    alpha = rng.normal(0, 0.06, N)
    beta = np.arange(T) * trend

    rows = []
    for i in range(N):
        is_treated_unit = i >= n_control
        first_treated = t_pre + 1 if is_treated_unit else None

        for t in range(1, T + 1):
            treated_cell = is_treated_unit and t > t_pre
            lr = base_lr + alpha[i] + beta[t - 1] + (true_att if treated_cell else 0.0) + rng.normal(0, noise_sd)
            lr = max(lr, 0.01)
            premium = 500_000 + rng.normal(0, 50_000)
            exposure = 2000 + rng.normal(0, 200)
            incurred = lr * premium
            rows.append({
                "segment_id": f"seg_{i:03d}",
                "period": t,
                "loss_ratio": lr,
                "earned_premium": max(premium, 10_000),
                "earned_exposure": max(exposure, 50),
                "incurred_claims": max(incurred, 0),
                "claim_count": int(max(0, rng.poisson(exposure * 0.08))),
                "first_treated_period": first_treated,
                "treated": 1 if treated_cell else 0,
                "cohort": first_treated,
            })

    return pl.DataFrame(rows)
