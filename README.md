# insurance-causal-policy

[![PyPI](https://img.shields.io/pypi/v/insurance-causal-policy)](https://pypi.org/project/insurance-causal-policy/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-causal-policy)](https://pypi.org/project/insurance-causal-policy/)
[![Downloads](https://img.shields.io/pypi/dm/insurance-causal-policy)](https://pypi.org/project/insurance-causal-policy/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/burning-cost/burning-cost-examples/blob/main/notebooks/burning-cost-in-30-minutes.ipynb)
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.org/github/burning-cost/insurance-causal-policy/blob/main/notebooks/quickstart.ipynb)

> Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-causal-policy/discussions). Found it useful? A star helps others find it.

**Your before-and-after comparison can't prove the rate change worked. SDID can.**

Pricing teams change rates. Loss ratios move. But did the rate change cause the movement, or was it market inflation, mix shift, or a concurrent regulatory change?

Standard before-and-after comparisons can't answer this. Neither can regression on mixed datasets. You need a control group and a method that handles the messy reality of insurance panels — staggered adoption across segments, varying exposures, IBNR lag, and market-wide shocks that hit everything at once.

This library implements Synthetic Difference-in-Differences (SDID) and Doubly Robust Synthetic Controls (DRSC) for insurance rate change evaluation. It converts policy/claims tables into segment × quarter panels, estimates causal effects with proper statistical inference, and produces structured output in a format consistent with FCA Consumer Duty evidence requirements.

**Blog post:** [Synthetic Difference-in-Differences for Rate Change Evaluation](https://burning-cost.github.io/2026/03/13/your-rate-change-didnt-prove-anything/)

## Why bother

Benchmarked against naive before-after and plain DiD on synthetic UK motor insurance panel data (100 segments, 12 quarterly periods, true ATT = -0.08 on loss ratio). Market-wide claims inflation of 0.5pp per period creates upward bias in naive estimators that do not use a control group. Results from `notebooks/benchmark_sdid.py` run 2026-03-17 on Databricks serverless.

Single-fit result (100 segments, 12 periods, 30 treated, 7 pre-treatment periods):

| Estimator | ATT estimate | Bias vs true | 95% CI | p-value |
|-----------|-------------|-------------|--------|---------|
| Naive before-after | -0.0422 | +0.0378 | none | none |
| Plain DiD | -0.0766 | +0.0034 | none | none |
| SDID | -0.0748 | +0.0052 | [-0.0899, -0.0597] | 0.000 |

| Capability | Naive before-after | Plain DiD | SDID |
|-----------|-------------------|-----------|------|
| Removes market trend bias | No | Partial | Yes |
| Produces confidence interval | No | No | Yes |
| Pre-treatment validation | No | No | Yes |
| Sensitivity analysis | No | No | Yes |
| SDID 95% CI coverage (50 simulations) | n/a | n/a | 98.0% |

The naive before-after bias (+3.8pp) arises because market claims inflation increased loss ratios across all segments during the post-treatment window. A before-after comparison on treated segments alone cannot separate the rate change effect from that market trend.

▶ [Run on Databricks](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/sdid_demo.py)

---

**Read more:** [Your Rate Change Didn't Prove Anything](https://burning-cost.github.io/blog/your-rate-change-didnt-prove-anything) — why before-and-after comparisons fail FCA scrutiny and how SDID fixes this.

## What is SDID?

SDID (Arkhangelsky et al., 2021, AER) combines Synthetic Control and Difference-in-Differences:

- **Synthetic Control** builds a weighted average of control segments that matched the treated segment's pre-treatment trend
- **Difference-in-Differences** uses the post-treatment divergence between treated and synthetic control as the causal estimate
- **SDID** adds an intercept term to the unit weight optimisation (absorbs level differences — unlike pure SC) and adds time weights to emphasise pre-treatment periods most predictive of the post-treatment window

The FCA has used causal DiD designs in its own market evaluations (for example, its GIPP remedies review). SDID belongs to the same methodological family — applying that class of reasoning to individual rate change evaluation. The FCA has not specifically endorsed SDID as an evidence standard.


## Doubly Robust Synthetic Controls (DRSC)

DRSC (Sant'Anna, Shaikh, Syrgkanis 2025, arXiv:2503.11375) is a formally doubly robust version of synthetic control for insurance panels. Use it instead of SDID when:

- Your donor pool is small (5-15 control segments) — the DR property provides a safety net if SC weights are poorly identified
- Your post-treatment window covers disrupted periods (COVID 2020, GIPP reform 2022) — parallel trends may hold approximately even when SC fit is imperfect
- You want a consistency guarantee under either assumption (SC or PT), not just informal robustness

The key difference from SDID: SC weights are estimated via unconstrained OLS (not CVXPY constrained quadratic program). Negative weights are valid and expected. No CVXPY dependency.

```python
from insurance_causal_policy import DoublyRobustSCEstimator, make_synthetic_panel_direct

# Generate a balanced panel directly (for testing)
panel = make_synthetic_panel_direct(
    n_control=15,    # small donor pool -- ideal DRSC use case
    n_treated=5,
    t_pre=8,
    t_post=4,
    true_att=-0.06,
)

est = DoublyRobustSCEstimator(
    panel,
    outcome="loss_ratio",
    inference="bootstrap",   # Exp(1)-1 multiplier bootstrap; valid under PT or SC
    n_replicates=500,
)
result = est.fit()
print(result.summary())
# -> DRSC estimate: -0.0614 decrease in loss_ratio (95% CI: -0.0924 to -0.0304, p=0.000)

# SC weights (unconstrained OLS -- negative weights allowed)
print(result.weights.sc_weights)

# Per-unit moment function (ATT = mean(phi))
print(result.phi.mean())  # should equal result.att

# FCA evidence summary
print(result.to_fca_summary(product_line="Motor", rate_change_date="2023-Q1"))
```

**SDID vs DRSC -- which to use:**

| Situation | Recommendation |
|-----------|---------------|
| Large donor pool (50+ controls), clean pre-trends | SDID (CVXPY weights well-identified) |
| Small donor pool (5-15 controls) | DRSC (DR property is the safety net) |
| Post-COVID or GIPP window, uncertain PT | DRSC (consistent under SC alone) |
| Need simplex-constrained (non-negative) weights | SDID |
| No CVXPY available | DRSC (pure numpy/scipy) |

## Installation

```bash
pip install insurance-causal-policy
```

Or with uv:

```bash
uv add insurance-causal-policy
```

With Callaway-Sant'Anna staggered adoption:

```bash
uv add insurance-causal-policy "differences>=0.2.0"
```

## Quick start

```python
from insurance_causal_policy import (
    PolicyPanelBuilder,
    SDIDEstimator,
    FCAEvidencePack,
    compute_sensitivity,
    make_synthetic_motor_panel,
)

# Generate synthetic data (or use your own policy/claims tables)
policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
    n_segments=100,
    n_periods=12,        # 3 years quarterly
    treat_fraction=0.25, # 25 segments receive the rate change
    true_att=-0.08,      # true causal effect (-8pp loss ratio)
    treatment_period=8,  # Q8 is the rate change
)

# Build balanced segment × period panel
builder = PolicyPanelBuilder(
    policy_df=policy_df,
    claims_df=claims_df,
    rate_log_df=rate_log_df,
    outcome="loss_ratio",
    exposure_col="earned_premium",
)
panel = builder.build()
print(builder.summary())

# Fit SDID
est = SDIDEstimator(panel, inference="placebo", n_replicates=200)
result = est.fit()
print(result.summary())
# → SDID estimate: -0.0748 decrease in loss_ratio (95% CI: -0.0899 to -0.0597, p=0.000)

# Sensitivity analysis
sens = compute_sensitivity(result, m_values=[0, 0.5, 1.0, 2.0])
print(sens.summary())
# → Result robust for all M tested (up to 2.0)

# FCA evidence pack
pack = FCAEvidencePack(
    result=result,
    sensitivity=sens,
    product_line="Motor",
    rate_change_date="2023-Q1",
    rate_change_magnitude="+8.5% technical premium",
    panel_summary=builder.summary(),
)
print(pack.to_markdown())
```

## Worked Example

[`causal_rate_change_evaluation.py`](https://github.com/burning-cost/burning-cost-examples/blob/main/examples/causal_rate_change_evaluation.py) walks through the full SDID workflow end-to-end: a synthetic motor portfolio with staggered rate changes, event study pre-treatment validation, HonestDiD sensitivity analysis, and final output as a structured FCA Consumer Duty evidence pack. It is the fastest way to see all four components working together on realistic data before wiring in your own policy and claims tables.


## Your data schema

`policy_df` — one row per segment per period:

| Column | Type | Description |
|--------|------|-------------|
| `segment_id` | str | Unique segment identifier (e.g., `north_east_26-35_direct`) |
| `period` | int | Period index (e.g., 202301 for 2023-Q1, or sequential 1..T) |
| `earned_premium` | float | Premium earned in this period (prorated) |
| `earned_exposure` | float | Earned exposure (policy years) |

`claims_df` — claims aggregated to segment × period:

| Column | Type | Description |
|--------|------|-------------|
| `segment_id` | str | Matches policy_df |
| `period` | int | Accident period matching policy_df.period |
| `incurred_claims` | float | Paid + IBNR reserve |
| `claim_count` | int | Number of claims in accident period |

`rate_log_df` — rate change log:

| Column | Type | Description |
|--------|------|-------------|
| `segment_id` | str | Segment that received rate change |
| `first_treated_period` | int | First period with new rate applied |

Segments absent from `rate_log_df` are classified as never-treated (valid control units).

## Staggered adoption

When different segments received rate changes at different times, use `StaggeredEstimator` instead.

Note: `StaggeredEstimator` uses the `differences` package (Callaway-Sant'Anna reference implementation). Install it first: `uv add "differences>=0.2.0"`

```python
from insurance_causal_policy import StaggeredEstimator

est = StaggeredEstimator(
    panel,
    outcome="loss_ratio",
    control_group="notyettreated",  # or "nevertreated"
)
result = est.fit()
# result.att_overall: overall ATT
# result.event_study: period-by-period effects
```

This implements Callaway & Sant'Anna (2021): estimates ATT(g, t) for each cohort g and period t separately, using only clean controls (never-treated or not-yet-treated). Avoids the contamination problem where already-treated segments pollute the control group for later-treated cohorts.

## Outcome metrics

```python
# Loss ratio (default)
builder = PolicyPanelBuilder(..., outcome="loss_ratio")

# Claim frequency
builder = PolicyPanelBuilder(..., outcome="frequency")

# Retention rate (requires policies_renewed and policies_due columns)
builder = PolicyPanelBuilder(..., outcome="retention")

# Use paid claims only (reduces IBNR bias for recent periods)
builder = PolicyPanelBuilder(..., paid_only=True)
```

## Parallel trends: what to do when it fails

The key assumption is approximate parallel trends: in the absence of the rate change, treated and control segments would have evolved similarly (after SDID reweighting). The event study pre-treatment coefficients test this.

If the pre-treatment test fails (p < 0.10), investigate before using results for regulatory evidence:

- **Market-wide Ogden rate change**: no internal control exists; benchmark against ABI market data
- **COVID lockdown in the window**: Q2 2020 affected all segments; treat as a structural break
- **GIPP (PS21/5, Jan 2022)**: market-wide renewal pricing change; affects all segments
- **Differential claims inflation**: young drivers and high-value vehicles may have had disproportionate severity increases 2021–2023

The sensitivity analysis quantifies how robust the conclusion is: "ATT remains significant even if post-treatment parallel trends violations are twice as large as the pre-period variation."

## Warnings about IBNR

Loss ratio for periods within 18 months of the analysis date understates ultimate claims because IBNR is not fully developed. If your post-treatment window is recent:

- Set `paid_only=True` to use paid claims (less IBNR noise, slower development)
- Or use `outcome="frequency"` — frequency is far less sensitive to IBNR lag
- Or restrict analysis to accident periods with >18 months of development

## FCA regulatory context

The FCA's multi-firm review of Consumer Duty implementation (2024) found that most insurers failed to demonstrate causal attribution between rate changes and outcomes. They showed data before and after but did not show that the rate change caused the change, rather than external factors.

The FCA has used causal DiD designs in its own market evaluations. SDID belongs to the same methodological family, which means the underlying reasoning — constructing a control group and attributing the difference causally — is the same approach that regulators find credible. The FCA has not specifically endorsed SDID as an evidence standard for rate change evaluation.

Good evidence for a Consumer Duty outcome monitoring pack:
- Pre-treatment parallel trends test (visual and statistical)
- ATT with confidence interval from a recognised causal method
- Sensitivity analysis showing robustness to assumption violations
- Structured narrative linking rate change to outcome change

This library produces all of it.

## Performance

Benchmarked against naive before-after and plain DiD on synthetic UK motor insurance panel data (100 segments, 12 quarterly periods, true ATT = -0.08 on loss ratio). Market-wide claims inflation of 0.5pp per period creates upward bias in naive estimators that do not use a control group. Results from `notebooks/benchmark_sdid.py` run 2026-03-17 on Databricks serverless. See `## Why bother` above for the comparison table.

**Fit time:** 0.013s (naive before-after) vs 2.50s (SDID with 200 placebo replicates) on a 100-segment, 12-period panel. SDID runtime scales with the number of control segments × pre-treatment periods and with `n_replicates`. For a 200-segment panel with 300 replicates, expect 15–30 seconds.

**Sensitivity (Rambachan-Roth 2023):** result robust for all M values tested (0 to 2.0). Post-treatment parallel trends violations would need to exceed twice the observed pre-period variation before the conclusion changes sign — this exceeds the FCA's typical robustness threshold.

**SDID 95% CI coverage across 50 simulations:** 98.0% (target: 95%). Coverage is slightly conservative — the placebo-based interval is slightly wider than the minimum necessary. In practice this means the stated confidence intervals do not systematically exclude the true effect.

**Where SDID adds most value:**
- Market-wide claims inflation present in the panel window (as in this DGP: +3.8pp naive bias)
- Rate change applied to a subset of segments, not book-wide
- FCA Consumer Duty evidence pack requires credible causal attribution
- Mix shift or regulatory changes coincide with the rate window

**When plain DiD is sufficient:** panels where treated and control segments have demonstrably parallel pre-trends and the only confound is an additive market shock. On balanced synthetic panels, plain DiD is approximately unbiased but still lacks formal inference and pre-treatment validation.

Run `notebooks/benchmark_sdid.py` on Databricks to reproduce.


## DRSC vs SDID: benchmark results

100-simulation Monte Carlo on synthetic motor insurance panels (true ATT = -0.06,
factor model DGP, 8 pre-treatment + 4 post-treatment periods, 5 treated units).
Run on Databricks serverless 2026-03-21. See `notebooks/drsc_vs_sdid.py` for
full methodology.

SDID uses jackknife inference. DRSC uses multiplier bootstrap (Exp(1)-1), valid
under either SC identification or parallel trends.

**Few donors (N_co = 6):**

| Metric              | SDID    | DRSC    |
|---------------------|---------|---------|
| Mean ATT (true=-0.06) | -0.0588 | -0.0590 |
| Absolute bias       | 0.0012  | 0.0010  |
| RMSE                | 0.0137  | 0.0104  |
| Std dev             | 0.0137  | 0.0104  |
| 95% CI coverage     | 97%     | 93%     |

DRSC reduces RMSE by 24% at N_co=6. Both estimators recover the ATT with similar
low bias under the factor model DGP. The DRSC advantage is variance reduction
from the OLS-weight + multiplier bootstrap combination, which doesn't force weights
into the simplex.

**Many donors (N_co = 40):**

| Metric              | SDID    | DRSC    |
|---------------------|---------|---------|
| Mean ATT (true=-0.06) | -0.0601 | -0.0601 |
| RMSE                | 0.0088  | 0.0088  |
| 95% CI coverage     | 96%     | 91%     |

At N_co=40, both methods perform identically on point estimates and RMSE. SDID has
slightly better CI coverage (96% vs 91%), likely because jackknife SE is better
calibrated than the bootstrap SE for large symmetric problems.

**Decision rule:**
- N_co < 10: use `DoublyRobustSCEstimator`
- N_co >= 20: use `SDIDEstimator`
- N_co 10–20: run both; if estimates diverge by more than one SE, investigate SC weight diagnostics

## Dependencies

- `polars` — panel construction (faster than pandas for large books)
- `pandas` — result objects (compatibility with econometrics libraries)
- `numpy`, `scipy` — numerical computation
- `cvxpy` — SDID weight optimisation (native implementation, no R dependency)
- `matplotlib` — plots
- `differences` (optional) — Callaway-Sant'Anna reference implementation

## Worked Example

[`causal_rate_change_evaluation.py`](https://github.com/burning-cost/burning-cost-examples/blob/main/examples/causal_rate_change_evaluation.py) — full SDID rate change evaluation on synthetic motor portfolio: event study, HonestDiD sensitivity, FCA evidence pack.

A Databricks-importable version is also available: [Databricks notebook](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/causal_rate_change_evaluation.py).



## References

- Sant'Anna, Shaikh, Syrgkanis (2025). *Doubly Robust Synthetic Controls*. arXiv:2503.11375.
- Arkhangelsky, Athey, Hirshberg, Imbens, Wager (2021). *Synthetic Difference-in-Differences*. American Economic Review 111(12): 4088–4118.
- Callaway, Sant'Anna (2021). *Difference-in-Differences with Multiple Time Periods*. Journal of Econometrics 225(2): 200–230.
- Rambachan, Roth (2023). *A More Credible Approach to Parallel Trends*. Review of Economic Studies, rdad018.
- FCA Multi-Firm Review of Consumer Duty Implementation (2024).
- FCA (2025). Evaluation of GIPP remedies (internal evaluation paper, causal DiD methodology).

## Related Libraries

| Library | Description |
|---------|-------------|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for individual-level causal effects — complements portfolio-level SDID with risk-level treatment effect estimation |
| [insurance-trend](https://github.com/burning-cost/insurance-trend) | Trend analysis with structural break detection — separates genuine market trends from the effects of pricing actions |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | PSI and A/E drift detection — for detecting when a rate change is needed |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — Consumer Duty evidence pack often requires both causal outcome monitoring and fairness audit |
## Licence

BSD-3

---
