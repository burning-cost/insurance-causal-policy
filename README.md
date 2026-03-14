# insurance-causal-policy
[![Tests](https://github.com/burning-cost/insurance-causal-policy/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-causal-policy/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-causal-policy)](https://pypi.org/project/insurance-causal-policy/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Pricing teams change rates. Loss ratios move. But did the rate change cause the movement, or was it market inflation, mix shift, or a concurrent regulatory change?

Standard before-and-after comparisons can't answer this. Neither can regression on mixed datasets. You need a control group and a method that handles the messy reality of insurance panels — staggered adoption across segments, varying exposures, IBNR lag, and market-wide shocks that hit everything at once.

This library implements Synthetic Difference-in-Differences (SDID) for insurance rate change evaluation. It converts policy/claims tables into segment × quarter panels, estimates causal effects with proper statistical inference, and produces structured output that satisfies FCA Consumer Duty evidence requirements.

## The problem

Your motor team raised rates by 8% in Q1 2023 for direct channel, young drivers. Loss ratios subsequently fell. Two questions:

1. How much of the fall is attributable to the rate change versus market claims inflation flattening and mix shift towards lower-risk business?
2. Would the FCA's economics team accept this as credible evidence of a fair value outcome under Consumer Duty?

A raw before-and-after gives you a number. SDID gives you a number with a confidence interval, a pre-treatment validation, and a sensitivity analysis showing how robust the conclusion is to violations of the key identifying assumption.

## What is SDID?

SDID (Arkhangelsky et al., 2021, AER) combines Synthetic Control and Difference-in-Differences:

- **Synthetic Control** builds a weighted average of control segments that matched the treated segment's pre-treatment trend
- **Difference-in-Differences** uses the post-treatment divergence between treated and synthetic control as the causal estimate
- **SDID** adds an intercept term to the unit weight optimisation (absorbs level differences — unlike pure SC) and adds time weights to emphasise pre-treatment periods most predictive of the post-treatment window

The FCA used a causal DiD design in its own evaluation of GIPP remedies (EP25/2). SDID is the same class of method, applied to individual rate change evaluation.

## Installation

```bash
pip install insurance-causal-policy
```

With Callaway-Sant'Anna staggered adoption:

```bash
pip install insurance-causal-policy "differences>=0.2.0"
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
# → SDID estimate: -0.0792 decrease in loss_ratio (95% CI: -0.1103 to -0.0481, p=0.000)

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

Note: `StaggeredEstimator` uses the `differences` package (Callaway-Sant'Anna reference implementation). Install it first: `pip install "differences>=0.2.0"`

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

FCA EP25/2 (2025) — the FCA's own evaluation of GIPP remedies — used a causal DiD design with an Average Causal Response interpretation. SDID is the same class of method the FCA uses in its own evaluation work.

Good evidence for a Consumer Duty outcome monitoring pack:
- Pre-treatment parallel trends test (visual and statistical)
- ATT with confidence interval from a recognised causal method
- Sensitivity analysis showing robustness to assumption violations
- Structured narrative linking rate change to outcome change

This library produces all of it.

## Performance

Benchmarked against naive before-after and plain DiD on synthetic UK motor insurance
panel data (100 segments, 12 quarterly periods, true ATT = -0.08 on loss ratio).
Market-wide claims inflation of 0.5pp per period creates upward bias in naive estimators
that do not use a control group. 50-simulation Monte Carlo measures estimator bias and
confidence interval coverage. See `notebooks/benchmark_sdid.py` for full methodology.

| Metric                              | Naive before-after | Plain DiD  | SDID       |
|-------------------------------------|--------------------|------------|------------|
| Mean estimated ATT (true = -0.080)  | varies by sim      | varies     | ~-0.079    |
| Bias direction                      | positive (upward)  | near-zero  | near-zero  |
| 95% CI coverage                     | n/a                | n/a        | ~93-95%    |
| Produces confidence interval        | No                 | No         | Yes        |
| Pre-treatment validation            | No                 | No         | Yes        |
| Sensitivity analysis                | No                 | No         | Yes        |

The naive before-after bias scales with the length of the post-treatment window and the
rate of market claims inflation. On a 4-quarter post window with 0.5pp quarterly inflation,
the naive estimate overstates the rate change benefit by roughly 2pp. SDID eliminates this
bias by constructing a synthetic control that tracks the same inflation as the treated group.

Plain DiD has near-zero bias in this DGP because treated and control segments have similar
pre-trends by construction. In real portfolios with segment-level mix shift or different
business vintages, SDID's synthetic control reweighting does meaningful additional work.

## Dependencies

- `polars` — panel construction (faster than pandas for large books)
- `pandas` — result objects (compatibility with econometrics libraries)
- `numpy`, `scipy` — numerical computation
- `cvxpy` — SDID weight optimisation (native implementation, no R dependency)
- `matplotlib` — plots
- `differences` (optional) — Callaway-Sant'Anna reference implementation

## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for causal rating factor analysis — the within-model equivalent of this library's between-period approach |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | PSI and A/E drift detection for detecting when a rate change is needed |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Demand modelling and ENBP compliance — the commercial complement to loss ratio causal analysis |
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — Consumer Duty evidence pack often requires both causal outcome monitoring and fairness audit |

[All Burning Cost libraries →](https://burning-cost.github.io)

## Read more

[Your Rate Change Didn't Prove Anything](https://burning-cost.github.io/blog/your-rate-change-didnt-prove-anything) — why before-and-after comparisons fail FCA scrutiny and how SDID fixes this.

## References

- Arkhangelsky, Athey, Hirshberg, Imbens, Wager (2021). *Synthetic Difference-in-Differences*. American Economic Review 111(12): 4088–4118.
- Callaway, Sant'Anna (2021). *Difference-in-Differences with Multiple Time Periods*. Journal of Econometrics 225(2): 200–230.
- Rambachan, Roth (2023). *A More Credible Approach to Parallel Trends*. Review of Economic Studies, rdad018.
- FCA TR24/2 (2024). Insurance multi-firm outcomes monitoring review.
- FCA EP25/2 (2025). Evaluation of GIPP remedies.

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for individual-level causal effects — complements portfolio-level SDID with risk-level treatment effect estimation |
| [insurance-trend](https://github.com/burning-cost/insurance-trend) | Trend analysis with structural break detection — separates genuine market trends from the effects of pricing actions |

