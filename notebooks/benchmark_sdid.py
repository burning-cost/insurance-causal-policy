# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-causal-policy — SDID vs naive before-after
# MAGIC
# MAGIC **Library:** `insurance-causal-policy` v0.1.4 — Synthetic Difference-in-Differences
# MAGIC for causal evaluation of insurance rate changes.
# MAGIC
# MAGIC **Baseline:** Naive before-after comparison. This is what most pricing teams
# MAGIC actually do: compute the loss ratio pre and post rate change for treated segments,
# MAGIC compute the change, and call it the effect of the rate change. No control group,
# MAGIC no adjustment for market trends.
# MAGIC
# MAGIC **Dataset:** Synthetic UK motor insurance segment × quarter panel, 100 segments,
# MAGIC 12 periods (3 years quarterly). Known DGP with a 10% rate increase applied to
# MAGIC 30 randomly selected segments in period 8, producing a true ATT of -0.08 on
# MAGIC loss ratio (rate increase of ~10% reduces loss ratio by ~8pp mechanically).
# MAGIC Market-wide claims inflation of 0.5pp per quarter affects all segments.
# MAGIC
# MAGIC **Date:** 2026-03-14
# MAGIC
# MAGIC **Library version:** 0.1.4
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The benchmark tests the core proposition: naive before-after comparisons in
# MAGIC insurance panels are biased. Claims inflation and mix shift affect all segments
# MAGIC simultaneously. A method that uses control segments to isolate the counterfactual
# MAGIC trend will recover the true ATT; naive before-after will not — specifically, it
# MAGIC will overstate the effect because it attributes market-wide trend improvements
# MAGIC to the rate change.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-causal-policy cvxpy matplotlib numpy scipy polars pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from insurance_causal_policy import (
    PolicyPanelBuilder,
    SDIDEstimator,
    make_synthetic_motor_panel,
    make_synthetic_panel_direct,
    plot_event_study,
    plot_synthetic_trajectory,
    compute_sensitivity,
)

warnings.filterwarnings("ignore")

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Generation
# MAGIC
# MAGIC We generate a synthetic UK motor segment × quarter panel with a known causal structure.
# MAGIC
# MAGIC DGP:
# MAGIC - 100 segments (territory × age band × channel combinations)
# MAGIC - 12 quarterly periods; period 8 is the rate change implementation quarter
# MAGIC - 30% of segments receive a rate change (true ATT = -0.08 on loss ratio)
# MAGIC - Each segment has a fixed effect drawn from N(0, 0.06²)
# MAGIC - Market-wide claims inflation of 0.5pp per quarter
# MAGIC - Idiosyncratic noise: N(0, 0.05²) per segment-period
# MAGIC
# MAGIC The naive before-after estimator is biased because it cannot separate:
# MAGIC   (a) the mechanical effect of the rate change on loss ratio
# MAGIC   (b) the market-wide trend that would have happened regardless
# MAGIC
# MAGIC SDID builds a weighted synthetic control from untreated segments, eliminates the
# MAGIC market trend, and recovers the true ATT.

# COMMAND ----------

TRUE_ATT = -0.08
N_SEGMENTS = 100
N_PERIODS = 12
TREAT_FRACTION = 0.30
TREATMENT_PERIOD = 8
NOISE_SD = 0.05
TREND_PER_PERIOD = 0.005  # 0.5pp quarterly claims inflation

rng = np.random.default_rng(42)

policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
    n_segments=N_SEGMENTS,
    n_periods=N_PERIODS,
    treat_fraction=TREAT_FRACTION,
    true_att=TRUE_ATT,
    treatment_period=TREATMENT_PERIOD,
    noise_sd=NOISE_SD,
    base_loss_ratio=0.72,
    trend_per_period=TREND_PER_PERIOD,
    random_seed=42,
    staggered=False,
)

print(f"Policy data:  {policy_df.shape}")
print(f"Claims data:  {claims_df.shape}")
print(f"Rate changes: {rate_log_df.shape}")
print(f"Treated segments: {rate_log_df.height}")
print(f"Control segments: {N_SEGMENTS - rate_log_df.height}")
print(f"True ATT: {TRUE_ATT:+.3f} (i.e. -8pp loss ratio from the rate increase)")

# Build balanced panel
builder = PolicyPanelBuilder(
    policy_df=policy_df,
    claims_df=claims_df,
    rate_log_df=rate_log_df,
    outcome="loss_ratio",
    exposure_col="earned_premium",
)
panel = builder.build()
print()
print(builder.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline: Naive Before-After
# MAGIC
# MAGIC The naive estimator computes:
# MAGIC
# MAGIC   ATT_naive = mean(loss_ratio_treated, post) - mean(loss_ratio_treated, pre)
# MAGIC
# MAGIC This is equivalent to what a pricing team does when they look at the cohort of
# MAGIC policies that went through the rate change and compare their loss ratios before
# MAGIC and after. It ignores market trends entirely.
# MAGIC
# MAGIC The bias has a predictable direction: because claims inflation increases loss
# MAGIC ratios market-wide, the naive estimator will see a smaller reduction (or even
# MAGIC an increase) compared to the true effect. If the market was experiencing 2pp
# MAGIC annual claims inflation during the post-period, the naive estimate is biased
# MAGIC upward by that amount — understating the benefit of the rate change.

# COMMAND ----------

t0 = time.perf_counter()

# Extract panel data
panel_pd = panel.to_pandas()

# Identify treated segments and the treatment period
treated_segs = set(rate_log_df["segment_id"].to_list())
n_treated = len(treated_segs)

treated_panel = panel_pd[panel_pd["segment_id"].isin(treated_segs)].copy()

pre_periods = treated_panel[treated_panel["period"] < TREATMENT_PERIOD]
post_periods = treated_panel[treated_panel["period"] >= TREATMENT_PERIOD]

mean_lr_pre = pre_periods["loss_ratio"].mean()
mean_lr_post = post_periods["loss_ratio"].mean()

att_naive = mean_lr_post - mean_lr_pre

naive_time = time.perf_counter() - t0

print(f"Naive before-after estimator:")
print(f"  Mean LR (pre-treatment, treated segments):  {mean_lr_pre:.4f}")
print(f"  Mean LR (post-treatment, treated segments): {mean_lr_post:.4f}")
print(f"  Naive ATT: {att_naive:+.4f}")
print(f"  True ATT:  {TRUE_ATT:+.4f}")
print(f"  Bias:      {att_naive - TRUE_ATT:+.4f}")
print(f"  Fit time: {naive_time:.3f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Improved Naive: Before-After with Market Adjustment (DiD)
# MAGIC
# MAGIC A slightly better naive approach adjusts for market trends using control segments:
# MAGIC
# MAGIC   ATT_did = (treated_post - treated_pre) - (control_post - control_pre)
# MAGIC
# MAGIC This is plain difference-in-differences with equal weights across all control
# MAGIC segments. It improves on the naive estimator but does not optimally reweight
# MAGIC control segments to match the treated pre-trend — which is what SDID does.

# COMMAND ----------

control_segs = set(panel_pd["segment_id"].unique()) - treated_segs
control_panel = panel_pd[panel_pd["segment_id"].isin(control_segs)].copy()

control_pre = control_panel[control_panel["period"] < TREATMENT_PERIOD]["loss_ratio"].mean()
control_post = control_panel[control_panel["period"] >= TREATMENT_PERIOD]["loss_ratio"].mean()

att_did = (mean_lr_post - mean_lr_pre) - (control_post - control_pre)

print(f"Plain DiD estimator:")
print(f"  Treated:  pre={mean_lr_pre:.4f}, post={mean_lr_post:.4f}, change={mean_lr_post - mean_lr_pre:+.4f}")
print(f"  Control:  pre={control_pre:.4f}, post={control_post:.4f}, change={control_post - control_pre:+.4f}")
print(f"  DiD ATT:  {att_did:+.4f}")
print(f"  True ATT: {TRUE_ATT:+.4f}")
print(f"  Bias:     {att_did - TRUE_ATT:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Library: SDID Estimator
# MAGIC
# MAGIC SDID (Arkhangelsky et al., 2021) improves on plain DiD by:
# MAGIC
# MAGIC 1. **Unit weights** (omega): solve a constrained optimisation to find weights
# MAGIC    on control segments that make the weighted control pre-trend match the
# MAGIC    treated pre-trend as closely as possible. This is Synthetic Control.
# MAGIC
# MAGIC 2. **Time weights** (lambda): find pre-treatment periods that best predict
# MAGIC    the post-treatment outcome distribution. This upweights the most informative
# MAGIC    pre-treatment windows.
# MAGIC
# MAGIC 3. **Weighted TWFE**: estimate the ATT via a two-way fixed effects regression
# MAGIC    weighted by omega × lambda.
# MAGIC
# MAGIC Inference uses a placebo permutation test: randomly assign treatment to control
# MAGIC segments and recompute SDID. The empirical distribution of placebo ATTs gives
# MAGIC the standard error.

# COMMAND ----------

t0 = time.perf_counter()

est = SDIDEstimator(
    panel,
    outcome="loss_ratio",
    inference="placebo",
    n_replicates=200,
    random_seed=42,
)
result = est.fit()

sdid_time = time.perf_counter() - t0

print(f"SDID fit time: {sdid_time:.2f}s")
print()
print(result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Repeated Simulation: Bias and Coverage
# MAGIC
# MAGIC We run 50 independent simulations with different random seeds to measure
# MAGIC estimator bias and CI coverage empirically. This is the gold standard
# MAGIC for evaluating causal estimators: run the DGP many times, check whether
# MAGIC the true ATT falls inside the 95% CI at the right rate.
# MAGIC
# MAGIC Coverage: the fraction of simulations where the 95% CI contains the true ATT.
# MAGIC A well-calibrated estimator hits ~95%. A naive before-after has undefined
# MAGIC coverage because it produces no CI.

# COMMAND ----------

N_SIMS = 50
sdid_atts = []
sdid_cis = []
did_atts = []
naive_atts = []

print(f"Running {N_SIMS} simulations (true ATT = {TRUE_ATT:+.3f})...")

for sim_seed in range(N_SIMS):
    # Generate panel with different random seed (different segment fixed effects, noise)
    try:
        panel_sim = make_synthetic_panel_direct(
            n_control=70,
            n_treated=30,
            t_pre=7,
            t_post=5,
            true_att=TRUE_ATT,
            noise_sd=NOISE_SD,
            base_lr=0.72,
            trend=TREND_PER_PERIOD,
            random_seed=sim_seed + 100,
        )

        # Naive before-after (treated only)
        tr_pre = panel_sim.filter(
            (pl.col("treated") == 1) & (pl.col("period") <= 7)
        )["loss_ratio"].mean()
        tr_post = panel_sim.filter(
            (pl.col("treated") == 1) & (pl.col("period") > 7)
        )["loss_ratio"].mean()
        att_n = (tr_post or 0) - (tr_pre or 0)
        naive_atts.append(att_n)

        # Plain DiD
        co_pre = panel_sim.filter(
            (pl.col("treated") == 0) & (pl.col("period") <= 7)
        )["loss_ratio"].mean()
        co_post = panel_sim.filter(
            (pl.col("treated") == 0) & (pl.col("period") > 7)
        )["loss_ratio"].mean()
        att_d = att_n - ((co_post or 0) - (co_pre or 0))
        did_atts.append(att_d)

        # SDID
        est_sim = SDIDEstimator(
            panel_sim,
            outcome="loss_ratio",
            inference="bootstrap",  # faster than placebo for many sims
            n_replicates=100,
            random_seed=sim_seed,
        )
        res_sim = est_sim.fit()
        sdid_atts.append(res_sim.att)
        sdid_cis.append((res_sim.ci_low, res_sim.ci_high))

    except Exception as e:
        print(f"  Sim {sim_seed} failed: {e}")
        continue

sdid_atts = np.array(sdid_atts)
naive_atts = np.array(naive_atts)
did_atts = np.array(did_atts)

# Coverage
coverage = np.mean([lo <= TRUE_ATT <= hi for lo, hi in sdid_cis])

print(f"Completed {len(sdid_atts)} simulations.")
print()
print(f"{'Metric':<40} {'Naive B/A':>12} {'Plain DiD':>12} {'SDID':>12}")
print("=" * 80)
print(f"  {'Mean estimated ATT':<38} {np.mean(naive_atts):>12.4f} {np.mean(did_atts):>12.4f} {np.mean(sdid_atts):>12.4f}")
print(f"  {'True ATT':<38} {TRUE_ATT:>12.4f} {TRUE_ATT:>12.4f} {TRUE_ATT:>12.4f}")
print(f"  {'Bias (estimated - true)':<38} {np.mean(naive_atts)-TRUE_ATT:>12.4f} {np.mean(did_atts)-TRUE_ATT:>12.4f} {np.mean(sdid_atts)-TRUE_ATT:>12.4f}")
print(f"  {'RMSE':<38} {np.sqrt(np.mean((naive_atts-TRUE_ATT)**2)):>12.4f} {np.sqrt(np.mean((did_atts-TRUE_ATT)**2)):>12.4f} {np.sqrt(np.mean((sdid_atts-TRUE_ATT)**2)):>12.4f}")
print(f"  {'Std dev of estimates':<38} {np.std(naive_atts):>12.4f} {np.std(did_atts):>12.4f} {np.std(sdid_atts):>12.4f}")
print(f"  {'95% CI coverage (SDID only)':<38} {'n/a':>12} {'n/a':>12} {coverage:>12.3f}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Diagnostic Plots

# COMMAND ----------

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ── Plot 1: Distribution of ATT estimates across simulations ─────────────────
bins = np.linspace(-0.18, 0.02, 30)
ax1.hist(naive_atts, bins=bins, alpha=0.55, label=f"Naive B/A (bias={np.mean(naive_atts)-TRUE_ATT:+.3f})", color="tomato", edgecolor="white")
ax1.hist(did_atts, bins=bins, alpha=0.55, label=f"Plain DiD (bias={np.mean(did_atts)-TRUE_ATT:+.3f})", color="steelblue", edgecolor="white")
ax1.hist(sdid_atts, bins=bins, alpha=0.55, label=f"SDID (bias={np.mean(sdid_atts)-TRUE_ATT:+.3f})", color="seagreen", edgecolor="white")
ax1.axvline(TRUE_ATT, color="black", linewidth=2, linestyle="--", label=f"True ATT = {TRUE_ATT:+.3f}")
ax1.set_xlabel("Estimated ATT")
ax1.set_ylabel("Count (across 50 simulations)")
ax1.set_title("ATT Estimates: 50 Simulations")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Event study for the full-data SDID fit ────────────────────────────
if result.event_study is not None:
    es = result.event_study
    pre_mask = es["period_rel"] < 0
    post_mask = es["period_rel"] >= 0

    ax2.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax2.axvline(-0.5, color="grey", linewidth=1, linestyle=":", alpha=0.7)

    ax2.bar(es.loc[post_mask, "period_rel"], es.loc[post_mask, "att"],
            color="tomato", alpha=0.65, label="Post-treatment ATT", width=0.6)
    ax2.bar(es.loc[pre_mask, "period_rel"], es.loc[pre_mask, "att"],
            color="steelblue", alpha=0.65, label="Pre-treatment (placebo)", width=0.6)
    ax2.axhline(result.att, color="seagreen", linewidth=2, linestyle="-.",
                label=f"Overall SDID ATT = {result.att:+.4f}")
    ax2.axhline(TRUE_ATT, color="black", linewidth=1.5, linestyle="--",
                label=f"True ATT = {TRUE_ATT:+.3f}")

    ax2.set_xlabel("Period relative to treatment")
    ax2.set_ylabel("Period ATT estimate")
    ax2.set_title("Event Study: Pre/Post ATTs")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

# ── Plot 3: Bias comparison ────────────────────────────────────────────────────
methods = ["Naive\nB/A", "Plain\nDiD", "SDID"]
biases = [np.mean(naive_atts) - TRUE_ATT, np.mean(did_atts) - TRUE_ATT, np.mean(sdid_atts) - TRUE_ATT]
rmses = [
    np.sqrt(np.mean((naive_atts - TRUE_ATT) ** 2)),
    np.sqrt(np.mean((did_atts - TRUE_ATT) ** 2)),
    np.sqrt(np.mean((sdid_atts - TRUE_ATT) ** 2)),
]
colors = ["tomato", "steelblue", "seagreen"]

x = np.arange(3)
w = 0.35
bars_bias = ax3.bar(x - w / 2, biases, w, label="Bias", color=colors, alpha=0.7, edgecolor="white")
bars_rmse = ax3.bar(x + w / 2, rmses, w, label="RMSE", color=colors, alpha=0.45, edgecolor="white", hatch="//")
ax3.axhline(0, color="black", linewidth=1, linestyle="--")
ax3.set_xticks(x)
ax3.set_xticklabels(methods)
ax3.set_ylabel("Error (loss ratio points)")
ax3.set_title("Bias and RMSE Across 50 Simulations")
ax3.legend(fontsize=9)
ax3.grid(True, axis="y", alpha=0.3)

for bar, val in zip(bars_bias, biases):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f"{val:+.4f}", ha="center", fontsize=8)

# ── Plot 4: Unit weights from SDID ────────────────────────────────────────────
if result.weights is not None:
    omega = result.weights.unit_weights.sort_values(ascending=False)
    n_show = min(20, len(omega))
    top_omega = omega.head(n_show)
    ax4.barh(range(n_show), top_omega.values, color="steelblue", alpha=0.7)
    ax4.set_yticks(range(n_show))
    ax4.set_yticklabels([s[:25] for s in top_omega.index], fontsize=7)
    ax4.set_xlabel("Unit weight (omega)")
    ax4.set_title(f"SDID Unit Weights — Top {n_show} Control Segments")
    ax4.axvline(1.0 / len(omega), color="red", linewidth=1.5, linestyle="--",
                label=f"Equal weight = {1.0/len(omega):.4f}")
    ax4.legend(fontsize=9)
    ax4.grid(True, axis="x", alpha=0.3)

plt.suptitle(
    "insurance-causal-policy: SDID vs Naive Before-After — Benchmark",
    fontsize=13, fontweight="bold"
)
plt.savefig("/tmp/benchmark_sdid.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_sdid.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sensitivity Analysis
# MAGIC
# MAGIC The sensitivity analysis (Rambachan & Roth 2023) asks: how large would post-
# MAGIC treatment parallel trends violations have to be (as a multiple of the pre-
# MAGIC treatment variation) before the conclusion changes sign or loses significance?
# MAGIC
# MAGIC M=0: pure parallel trends assumed (standard SDID)
# MAGIC M=1: post-treatment violations as large as pre-treatment variation are allowed
# MAGIC M=2: post-treatment violations twice as large

# COMMAND ----------

try:
    sens = compute_sensitivity(result, m_values=[0.0, 0.5, 1.0, 1.5, 2.0])
    print(sens.summary())
except Exception as e:
    print(f"Sensitivity analysis: {e}")
    print("(Sensitivity requires sufficient pre-treatment periods)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verdict

# COMMAND ----------

print("=" * 72)
print("VERDICT: SDID vs Naive Before-After vs Plain DiD")
print("=" * 72)
print()
print(f"Single fit on 100-segment, 12-period panel:")
print(f"  True ATT:         {TRUE_ATT:+.4f}")
print(f"  Naive B/A:        {att_naive:+.4f}  (bias = {att_naive - TRUE_ATT:+.4f})")
print(f"  Plain DiD:        {att_did:+.4f}  (bias = {att_did - TRUE_ATT:+.4f})")
print(f"  SDID:             {result.att:+.4f}  (95% CI: [{result.ci_low:+.4f}, {result.ci_high:+.4f}])")
print(f"  SDID p-value:     {result.pval:.4f}")
print()
print(f"Across {len(sdid_atts)} simulations:")
print(f"  Naive B/A bias:   {np.mean(naive_atts)-TRUE_ATT:+.4f}  RMSE: {np.sqrt(np.mean((naive_atts-TRUE_ATT)**2)):.4f}")
print(f"  Plain DiD bias:   {np.mean(did_atts)-TRUE_ATT:+.4f}  RMSE: {np.sqrt(np.mean((did_atts-TRUE_ATT)**2)):.4f}")
print(f"  SDID bias:        {np.mean(sdid_atts)-TRUE_ATT:+.4f}  RMSE: {np.sqrt(np.mean((sdid_atts-TRUE_ATT)**2)):.4f}")
print(f"  SDID 95% CI coverage: {coverage:.1%}  (target: 95%)")
print()
print(f"Fit time: {naive_time:.3f}s (naive) vs {sdid_time:.2f}s (SDID)")
print()
print("Where SDID adds value:")
print("  - Market-wide claims inflation present in the panel window")
print("  - Rate change applied to a subset of segments (not book-wide)")
print("  - FCA Consumer Duty evidence pack requires credible causal attribution")
print("  - Mix shift or regulatory changes coincide with the rate window")
print()
print("Where naive before-after fails:")
print("  - Overstates or understates ATT by the market trend over the window")
print("  - No standard error -> no test of statistical significance")
print("  - No pre-treatment validation -> cannot detect selection bias")
print()
print("When SDID is unnecessary:")
print("  - Book-wide rate change with no control group available")
print("  - Only 1-2 pre-treatment periods (parallel trends untestable)")
print("  - <10 segments (weight solver becomes unreliable)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. README Performance Snippet

# COMMAND ----------

print("""
## Performance

Benchmarked against naive before-after and plain DiD on synthetic UK motor
insurance panel data (100 segments, 12 quarterly periods, true ATT = -0.08
on loss ratio). Market-wide claims inflation of 0.5pp per period is present
in the DGP to create bias in naive estimators. 50-simulation Monte Carlo
measures bias and confidence interval coverage.
See `notebooks/benchmark_sdid.py` for full methodology.
""")
print(f"| Metric                             | Naive before-after | Plain DiD   | SDID        |")
print(f"|------------------------------------|-------------------|-------------|-------------|")
print(f"| Mean estimated ATT (true = -0.080) | {np.mean(naive_atts):+.4f}          | {np.mean(did_atts):+.4f}      | {np.mean(sdid_atts):+.4f}      |")
print(f"| Bias                               | {np.mean(naive_atts)-TRUE_ATT:+.4f}          | {np.mean(did_atts)-TRUE_ATT:+.4f}      | {np.mean(sdid_atts)-TRUE_ATT:+.4f}      |")
print(f"| RMSE                               | {np.sqrt(np.mean((naive_atts-TRUE_ATT)**2)):.4f}           | {np.sqrt(np.mean((did_atts-TRUE_ATT)**2)):.4f}      | {np.sqrt(np.mean((sdid_atts-TRUE_ATT)**2)):.4f}      |")
print(f"| 95% CI coverage                    | n/a               | n/a         | {coverage:.1%}       |")
print(f"| Produces confidence interval       | No                | No          | Yes         |")
print(f"| Pre-treatment validation           | No                | No          | Yes         |")
print()
print("""SDID's advantage over plain DiD is concentrated in settings with
non-parallel pre-trends — where the synthetic control reweighting does meaningful
work. On balanced panels where control and treated segments have similar
pre-trends, plain DiD is approximately unbiased but still lacks formal inference.
SDID recovers the true ATT with well-calibrated confidence intervals and provides
a pre-treatment placebo test that the FCA now expects in causal evaluation evidence.
""")
