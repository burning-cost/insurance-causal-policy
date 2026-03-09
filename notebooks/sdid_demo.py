# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-causal-policy: SDID Demo
# MAGIC
# MAGIC Demonstrates the full workflow for causal evaluation of insurance rate changes
# MAGIC using Synthetic Difference-in-Differences (Arkhangelsky et al. 2021).
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Generates synthetic motor insurance data (100 segments, 12 quarters)
# MAGIC 2. Builds a balanced segment × period panel
# MAGIC 3. Fits the SDID estimator
# MAGIC 4. Runs sensitivity analysis
# MAGIC 5. Generates an FCA evidence pack

# COMMAND ----------

# MAGIC %pip install insurance-causal-policy

# COMMAND ----------

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from insurance_causal_policy import (
    PolicyPanelBuilder,
    SDIDEstimator,
    StaggeredEstimator,
    FCAEvidencePack,
    compute_sensitivity,
    plot_event_study,
    plot_unit_weights,
    plot_sensitivity,
    pre_trend_summary,
    make_synthetic_motor_panel,
    make_synthetic_panel_direct,
)

print("insurance-causal-policy loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic motor insurance data
# MAGIC
# MAGIC The data-generating process has a known true ATT of -0.08 (8pp reduction in
# MAGIC loss ratio). We'll see how well SDID recovers this.

# COMMAND ----------

TRUE_ATT = -0.08

policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
    n_segments=100,
    n_periods=12,
    treat_fraction=0.25,  # 25 segments receive the rate change
    true_att=TRUE_ATT,
    treatment_period=8,   # Rate change at Q8 (2 years pre-treatment, 1 year post)
    noise_sd=0.04,
    random_seed=42,
)

print(f"Policy table: {len(policy_df)} rows, {policy_df['segment_id'].n_unique()} segments")
print(f"Claims table: {len(claims_df)} rows")
print(f"Rate log: {len(rate_log_df)} treated segments")
print()
print("Policy table sample:")
print(policy_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build balanced panel
# MAGIC
# MAGIC PolicyPanelBuilder aggregates policy-level data to segment × period,
# MAGIC joins claims (with zero-fill for missing periods), computes exposure-weighted
# MAGIC outcome metrics, and joins the treatment indicator.

# COMMAND ----------

builder = PolicyPanelBuilder(
    policy_df=policy_df,
    claims_df=claims_df,
    rate_log_df=rate_log_df,
    outcome="loss_ratio",
    exposure_col="earned_premium",
    min_exposure=50.0,
)

panel = builder.build()
summary = builder.summary()

print("Panel summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")

print(f"\nPanel shape: {len(panel)} rows × {len(panel.columns)} columns")
print("\nPanel sample (treated segment):")
treated_segs = panel.filter(pl.col("first_treated_period").is_not_null())["segment_id"].unique()[:2].to_list()
print(panel.filter(pl.col("segment_id") == treated_segs[0]).select(
    ["segment_id", "period", "loss_ratio", "treated", "first_treated_period"]
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit SDID estimator
# MAGIC
# MAGIC SDID computes:
# MAGIC - Unit weights omega_i: which control segments best match treated pre-trend
# MAGIC - Time weights lambda_t: which pre-treatment periods predict the post-treatment window
# MAGIC - ATT via weighted TWFE regression
# MAGIC - Inference via placebo (N_control > N_treated required)

# COMMAND ----------

est = SDIDEstimator(
    panel,
    outcome="loss_ratio",
    inference="placebo",
    n_replicates=200,
    random_seed=42,
)

result = est.fit()

print("=" * 60)
print("SDID RESULT")
print("=" * 60)
print(result.summary())
print()
print(f"True ATT: {TRUE_ATT:.4f}")
print(f"Estimated ATT: {result.att:.4f}")
print(f"Coverage: {'YES' if result.ci_low <= TRUE_ATT <= result.ci_high else 'NO'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Parallel trends diagnostics

# COMMAND ----------

trends = pre_trend_summary(result)
print("PARALLEL TRENDS ASSESSMENT")
print("-" * 40)
print(trends["interpretation"])
print(f"\nPre-period ATTs: {[f'{x:.4f}' for x in trends['pre_atts']]}")
print(f"Max absolute pre-ATT: {trends['max_abs_pre_att']:.4f}")

# COMMAND ----------

fig = plot_event_study(
    result,
    title="Motor Rate Change — Event Study (SDID)",
    annotate_pval=True,
)
display(fig)
plt.close()

# COMMAND ----------

fig = plot_unit_weights(result, n_top=15, title="Synthetic Control Composition")
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Sensitivity analysis (HonestDiD-style)
# MAGIC
# MAGIC How robust is the conclusion to violations of the parallel trends assumption?
# MAGIC M is expressed in multiples of the pre-period standard deviation.

# COMMAND ----------

sens = compute_sensitivity(result, m_values=[0, 0.5, 1.0, 1.5, 2.0, 3.0])

print("SENSITIVITY ANALYSIS")
print("-" * 40)
print(sens.summary())
print()
print(sens.to_dataframe().to_string())

# COMMAND ----------

fig = plot_sensitivity(sens, title="Breakdown Frontier: How Robust is the ATT?")
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. FCA evidence pack
# MAGIC
# MAGIC Structured regulatory narrative for Consumer Duty outcome monitoring.

# COMMAND ----------

pack = FCAEvidencePack(
    result=result,
    sensitivity=sens,
    product_line="Motor",
    rate_change_date="2023-Q1",
    rate_change_magnitude="+8.5% technical premium",
    analyst="Pricing Analytics Team",
    panel_summary=summary,
    additional_notes=(
        "Analysis excludes telematics segment (insufficient control history). "
        "No Ogden rate changes occurred during the analysis window."
    ),
)

md = pack.to_markdown()
print(md)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Staggered adoption demo
# MAGIC
# MAGIC When different segments received rate changes at different times,
# MAGIC use StaggeredEstimator (Callaway-Sant'Anna 2021).

# COMMAND ----------

policy_stag, claims_stag, rate_log_stag = make_synthetic_motor_panel(
    n_segments=80,
    n_periods=14,
    treat_fraction=0.4,
    true_att=-0.07,
    treatment_period=8,
    staggered=True,
    n_stagger_cohorts=3,
    random_seed=99,
)

panel_stag = PolicyPanelBuilder(
    policy_stag, claims_stag, rate_log_stag, outcome="loss_ratio"
).build()

stag_est = StaggeredEstimator(
    panel_stag,
    outcome="loss_ratio",
    control_group="notyettreated",
)
stag_result = stag_est.fit()

print("STAGGERED ADOPTION RESULT (CS21)")
print("=" * 60)
print(f"Overall ATT: {stag_result.att_overall:.4f}")
print(f"SE: {stag_result.se_overall:.4f}")
print(f"95% CI: [{stag_result.ci_low_overall:.4f}, {stag_result.ci_high_overall:.4f}]")
print(f"p-value: {stag_result.pval_overall:.4f}")
print(f"Cohorts: {stag_result.n_cohorts}")
print(f"Pre-trend p-value: {stag_result.pre_trend_pval}")
print()
print("Event study:")
print(stag_result.event_study.to_string())

# COMMAND ----------

fig = plot_event_study(
    stag_result,
    title="Staggered Rate Changes — CS21 Event Study",
)
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Method | ATT | SE | 95% CI | True ATT |
# MAGIC |--------|-----|----|--------|----------|
# MAGIC | SDID | {att:.4f} | {se:.4f} | [{ci_low:.4f}, {ci_high:.4f}] | {true_att:.4f} |
# MAGIC
# MAGIC The SDID estimator recovers the true ATT within sampling error.
# MAGIC
# MAGIC Key properties demonstrated:
# MAGIC - Pre-treatment parallel trends satisfied (p > 0.10)
# MAGIC - ATT is statistically significant
# MAGIC - Sensitivity analysis shows result is robust to parallel trends violations
# MAGIC - FCA evidence pack generated in regulatory narrative format

print("Demo complete.")
print(f"SDID ATT: {result.att:.4f} (true: {TRUE_ATT:.4f})")
print(f"Significant: {result.significant}")
print(f"Pre-trends pass: {result.pre_trends_pass}")
print(f"Sensitivity breakdown M*: {sens.m_breakdown:.2f}")
