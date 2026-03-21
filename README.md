# insurance-causal-policy

[![PyPI](https://img.shields.io/pypi/v/insurance-causal-policy)](https://pypi.org/project/insurance-causal-policy/)
[![Python](https://img.shields.io/pypi/pyversions/insurance-causal-policy)](https://pypi.org/project/insurance-causal-policy/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD--3-blue)]()

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-causal-policy/discussions). Found it useful? A ⭐ helps others find it.

**Your before-and-after comparison can't prove the rate change worked. SDID can.**

Pricing teams change rates. Loss ratios move. But did the rate change cause the movement, or was it market inflation, mix shift, or a concurrent regulatory change?

Standard before-and-after comparisons can't answer this. Neither can regression on mixed datasets. You need a control group and a method that handles the messy reality of insurance panels — staggered adoption across segments, varying exposures, IBNR lag, and market-wide shocks that hit everything at once.

This library implements Synthetic Difference-in-Differences (SDID) for insurance rate change evaluation. It converts policy/claims tables into segment × quarter panels, estimates causal effects with proper statistical inference, and produces structured output in a format consistent with FCA Consumer Duty evidence requirements.

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

## Installation

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

FCA TR24/2 (2024) found that most insurers failed to demonstrate causal attribution between rate changes and outcomes. They showed data before and after but did not show that the rate change caused the change, rather than external factors.

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


## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for causal rating factor analysis — the within-model equivalent of this library's between-period approach |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | PSI and A/E drift detection for detecting when a rate change is needed |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Demand modelling and ENBP compliance — the commercial complement to loss ratio causal analysis |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — Consumer Duty evidence pack often requires both causal outcome monitoring and fairness audit |

[All Burning Cost libraries →](https://burning-cost.github.io)

## References

- Arkhangelsky, Athey, Hirshberg, Imbens, Wager (2021). *Synthetic Difference-in-Differences*. American Economic Review 111(12): 4088–4118.
- Callaway, Sant'Anna (2021). *Difference-in-Differences with Multiple Time Periods*. Journal of Econometrics 225(2): 200–230.
- Rambachan, Roth (2023). *A More Credible Approach to Parallel Trends*. Review of Economic Studies, rdad018.
- FCA TR24/2 (2024). Insurance multi-firm outcomes monitoring review.
- FCA (2025). Evaluation of GIPP remedies (internal evaluation paper, causal DiD methodology).

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for individual-level causal effects — complements portfolio-level SDID with risk-level treatment effect estimation |
| [insurance-trend](https://github.com/burning-cost/insurance-trend) | Trend analysis with structural break detection — separates genuine market trends from the effects of pricing actions |

## Licence

BSD-3

---
