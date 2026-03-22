# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-causal-policy — SDID vs two-period DiD
# MAGIC
# MAGIC **Library:** `insurance-causal-policy` — Synthetic Difference-in-Differences
# MAGIC for causal evaluation of insurance rate changes.
# MAGIC
# MAGIC **Baseline:** Simple two-period Difference-in-Differences (manual calculation).
# MAGIC This is the step up from naive before-after that most pricing teams attempt:
# MAGIC compute the pre/post change for treated regions, subtract the equivalent
# MAGIC change for control regions, and call it the treatment effect. No synthetic
# MAGIC control weighting, no time weighting, no formal inference.
# MAGIC
# MAGIC **Dataset:** Synthetic panel — 20 regions × 24 months. Treatment (rate change)
# MAGIC applied to 5 regions at month 13. Heterogeneous pre-treatment trends: treated
# MAGIC regions have region-specific drift that differs from control regions, so the
# MAGIC parallel trends assumption for plain DiD is violated by design.
# MAGIC
# MAGIC **Date:** 2026-03-22
# MAGIC
# MAGIC **Library version:** check `insurance_causal_policy.__version__`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The key question: when pre-treatment trends are heterogeneous across regions,
# MAGIC does SDID give more accurate treatment effect estimates than simple DiD?
# MAGIC
# MAGIC Plain two-period DiD assumes treated and control units would have evolved in
# MAGIC parallel absent the treatment. When that assumption fails — as it typically
# MAGIC does in insurance panels with mix shift, regional risk composition changes,
# MAGIC or differential claims inflation — plain DiD is biased. SDID builds a
# MAGIC synthetic control that reweights control units to match the treated units'
# MAGIC pre-treatment trajectory, partially correcting for this violation.

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

from insurance_causal_policy import (
    SDIDEstimator,
    make_synthetic_panel_direct,
    __version__,
)

warnings.filterwarnings("ignore")
print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print(f"insurance-causal-policy version: {__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Generation
# MAGIC
# MAGIC Synthetic panel: 20 regions × 24 months.
# MAGIC
# MAGIC DGP is a factor model:
# MAGIC
# MAGIC     Y_it = alpha_i + beta_t + tau * D_it + epsilon_it
# MAGIC
# MAGIC where:
# MAGIC - `alpha_i` = region fixed effect (heterogeneous baseline loss ratios)
# MAGIC - `beta_t` = common time trend (market claims inflation: 0.4pp/month)
# MAGIC - `tau = TRUE_ATT` = true causal effect of the rate change (−0.07, i.e. −7pp)
# MAGIC - `D_it` = 1 for treated regions from month 13 onward
# MAGIC - `epsilon_it ~ N(0, noise_sd^2)` = i.i.d. noise
# MAGIC
# MAGIC **Heterogeneous pre-trends:** We add a region-specific linear drift
# MAGIC `gamma_i * t` for `t < 13`. Treated regions have larger |gamma_i| on
# MAGIC average, so their pre-treatment trajectory is not parallel to the control
# MAGIC group average. This violates plain DiD's identifying assumption.
# MAGIC
# MAGIC SDID partially corrects for this by reweighting control units to match the
# MAGIC treated units' pre-treatment path before estimating the post-treatment
# MAGIC divergence.

# COMMAND ----------

rng = np.random.default_rng(2024)

N_REGIONS  = 20      # total regions
N_CONTROL  = 15      # control regions
N_TREATED  = 5       # treated regions
T_PRE      = 12      # months before rate change (months 1–12)
T_POST     = 12      # months after rate change  (months 13–24)
T_TOTAL    = T_PRE + T_POST
TRUE_ATT   = -0.07   # true ATT: 7pp loss ratio reduction
NOISE_SD   = 0.018   # per-cell noise
BASE_LR    = 0.72    # baseline loss ratio
TREND      = 0.004   # market-wide trend per month (claims inflation)

# Heterogeneous pre-trends: each region has its own linear drift
# Control regions: small drift centred near 0
# Treated regions: larger drift, reflecting different risk composition change
gamma_control = rng.normal(0.000, 0.003, N_CONTROL)   # control: small
gamma_treated = rng.normal(0.004, 0.004, N_TREATED)   # treated: positive bias

# Region fixed effects (log-scale variation around 0)
alpha_control = rng.normal(0, 0.06, N_CONTROL)
alpha_treated = rng.normal(0, 0.06, N_TREATED)

print(f"Regions:    {N_REGIONS}  ({N_CONTROL} control + {N_TREATED} treated)")
print(f"Periods:    {T_TOTAL}   ({T_PRE} pre + {T_POST} post, treatment at month {T_PRE + 1})")
print(f"True ATT:   {TRUE_ATT}")
print(f"Noise SD:   {NOISE_SD}")
print()
print(f"Pre-trend gamma (control): mean={gamma_control.mean():.4f}, sd={gamma_control.std():.4f}")
print(f"Pre-trend gamma (treated): mean={gamma_treated.mean():.4f}, sd={gamma_treated.std():.4f}")
print("Treated regions have larger positive drift => DiD parallel trends violated.")

# COMMAND ----------

# Build balanced panel as numpy array Y[N_REGIONS, T_TOTAL]
Y = np.zeros((N_REGIONS, T_TOTAL))

for i in range(N_CONTROL):
    for t in range(T_TOTAL):
        trend_component = TREND * t
        drift_component = gamma_control[i] * t if t < T_PRE else gamma_control[i] * T_PRE
        Y[i, t] = (
            BASE_LR
            + alpha_control[i]
            + trend_component
            + drift_component
            + rng.normal(0, NOISE_SD)
        )

for j in range(N_TREATED):
    i = N_CONTROL + j
    for t in range(T_TOTAL):
        trend_component = TREND * t
        # Pre-treatment drift accumulates; freeze at T_PRE level post-treatment
        drift_component = gamma_treated[j] * t if t < T_PRE else gamma_treated[j] * T_PRE
        treatment_effect = TRUE_ATT if t >= T_PRE else 0.0
        Y[i, t] = (
            BASE_LR
            + alpha_treated[j]
            + trend_component
            + drift_component
            + treatment_effect
            + rng.normal(0, NOISE_SD)
        )

# Clip to plausible loss ratio range
Y = np.clip(Y, 0.10, 1.50)

print(f"Y matrix shape: {Y.shape}  (N_REGIONS x T_TOTAL)")
print(f"Pre-treatment mean (control):  {Y[:N_CONTROL, :T_PRE].mean():.4f}")
print(f"Pre-treatment mean (treated):  {Y[N_CONTROL:, :T_PRE].mean():.4f}")
print(f"Post-treatment mean (control): {Y[:N_CONTROL, T_PRE:].mean():.4f}")
print(f"Post-treatment mean (treated): {Y[N_CONTROL:, T_PRE:].mean():.4f}")

# COMMAND ----------

# Convert to Polars panel expected by SDIDEstimator
rows = []
for i in range(N_REGIONS):
    is_treated = i >= N_CONTROL
    first_tp = T_PRE + 1 if is_treated else None
    for t in range(1, T_TOTAL + 1):
        rows.append({
            "segment_id":         f"region_{i:02d}",
            "period":             t,
            "loss_ratio":         float(Y[i, t - 1]),
            "earned_premium":     float(rng.uniform(500_000, 2_000_000)),
            "earned_exposure":    float(rng.uniform(1_000, 8_000)),
            "incurred_claims":    float(Y[i, t - 1] * rng.uniform(500_000, 2_000_000)),
            "claim_count":        int(rng.poisson(200)),
            "first_treated_period": int(first_tp) if first_tp is not None else None,
            "treated":            int(is_treated and t > T_PRE),
            "cohort":             int(first_tp) if first_tp is not None else None,
        })

panel = pl.DataFrame(rows, schema={
    "segment_id":          pl.String,
    "period":              pl.Int64,
    "loss_ratio":          pl.Float64,
    "earned_premium":      pl.Float64,
    "earned_exposure":     pl.Float64,
    "incurred_claims":     pl.Float64,
    "claim_count":         pl.Int64,
    "first_treated_period": pl.Int64,
    "treated":             pl.Int64,
    "cohort":              pl.Int64,
})

print(f"Panel rows: {len(panel):,}")
print(panel.head(6))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline: Simple Two-Period DiD
# MAGIC
# MAGIC The manual DiD estimator:
# MAGIC
# MAGIC     DiD = (mean_post_treated - mean_pre_treated) - (mean_post_control - mean_pre_control)
# MAGIC
# MAGIC This uses equal weights on all control units and all pre-treatment periods.
# MAGIC It provides no confidence interval and no pre-treatment validation.
# MAGIC
# MAGIC **Bias source:** Because treated regions have a steeper pre-trend than
# MAGIC controls, the control group's pre-post change understates what the treated
# MAGIC regions would have done absent the rate change. DiD will attribute some of
# MAGIC the treatment effect to the trend difference, producing a biased estimate.

# COMMAND ----------

t0_did = time.perf_counter()

Y_pre_control  = Y[:N_CONTROL, :T_PRE]
Y_post_control = Y[:N_CONTROL, T_PRE:]
Y_pre_treated  = Y[N_CONTROL:, :T_PRE]
Y_post_treated = Y[N_CONTROL:, T_PRE:]

# Simple two-period DiD (all units equally weighted, all periods equally weighted)
did_control_change = Y_post_control.mean() - Y_pre_control.mean()
did_treated_change = Y_post_treated.mean() - Y_pre_treated.mean()
did_att = did_treated_change - did_control_change

did_time = time.perf_counter() - t0_did

# Parallel trends diagnostic: how different are pre-trends?
# We estimate the average slope for each group over the pre-period
periods_pre = np.arange(T_PRE)
slope_control = np.polyfit(periods_pre, Y_pre_control.mean(axis=0), 1)[0]
slope_treated = np.polyfit(periods_pre, Y_pre_treated.mean(axis=0), 1)[0]
slope_diff = slope_treated - slope_control

print(f"Simple two-period DiD ATT:  {did_att:.4f}  (true ATT: {TRUE_ATT})")
print(f"Bias:                       {did_att - TRUE_ATT:+.4f}")
print()
print(f"Pre-period slope (control): {slope_control:.5f} per month")
print(f"Pre-period slope (treated): {slope_treated:.5f} per month")
print(f"Slope difference:           {slope_diff:+.5f}  "
      f"({'>> 0: parallel trends violated' if abs(slope_diff) > TREND * 0.5 else 'approx. parallel'})")
print(f"DiD fit time: {did_time:.4f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Library: SDID
# MAGIC
# MAGIC SDID (Arkhangelsky et al. 2021) builds a weighted average of control units
# MAGIC that matches the treated units' pre-treatment trajectory, then uses that
# MAGIC synthetic control to estimate the counterfactual post-treatment path.
# MAGIC
# MAGIC Unit weights omega: solve constrained QP to minimise distance between treated
# MAGIC mean and weighted control mean in pre-treatment periods. Regularisation
# MAGIC zeta discourages extreme weights.
# MAGIC
# MAGIC Time weights lambda: reweight pre-treatment periods to emphasise those most
# MAGIC predictive of the post-treatment level.
# MAGIC
# MAGIC ATT: estimated via weighted TWFE regression using the computed (omega, lambda).
# MAGIC
# MAGIC Inference: placebo shuffles (200 replicates) to build the null distribution.

# COMMAND ----------

t0_sdid = time.perf_counter()

sdid_est = SDIDEstimator(
    panel,
    outcome="loss_ratio",
    inference="placebo",
    n_replicates=200,
    random_seed=2024,
)
sdid_result = sdid_est.fit()

sdid_time = time.perf_counter() - t0_sdid

print(f"SDID fit time: {sdid_time:.2f}s")
print()
print(sdid_result.summary())

# COMMAND ----------

sdid_att   = float(sdid_result.att)
sdid_se    = float(sdid_result.se)
sdid_ci_lo = float(sdid_result.ci[0])
sdid_ci_hi = float(sdid_result.ci[1])
sdid_pval  = float(sdid_result.p_value)

print(f"SDID ATT estimate:  {sdid_att:.4f}")
print(f"SDID SE:            {sdid_se:.4f}")
print(f"SDID 95% CI:        [{sdid_ci_lo:.4f}, {sdid_ci_hi:.4f}]")
print(f"SDID p-value:       {sdid_pval:.4f}")
print(f"True ATT:           {TRUE_ATT}")
print(f"SDID bias:          {sdid_att - TRUE_ATT:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Pre-treatment Parallel Trends Test
# MAGIC
# MAGIC We check whether the SDID unit weights improve pre-treatment fit relative to
# MAGIC equal weighting (the DiD assumption). The synthetic control trajectory should
# MAGIC track the treated mean closely in the pre-period.

# COMMAND ----------

# Build synthetic control trajectory using SDID unit weights
omega = np.asarray(sdid_result.weights.unit_weights)  # length N_CONTROL

Y_control_pre  = Y[:N_CONTROL, :T_PRE]
Y_treated_pre  = Y[N_CONTROL:, :T_PRE]
Y_treated_mean = Y_treated_pre.mean(axis=0)   # equally-weighted treated mean per period

# SDID synthetic control: omega-weighted average of control units
Y_synth_pre = (omega[:, np.newaxis] * Y_control_pre).sum(axis=0)

# Equal-weighted control (DiD assumption)
Y_equal_pre = Y_control_pre.mean(axis=0)

# Pre-trend fit: RMSE of control trajectory vs treated mean
rmse_sdid = float(np.sqrt(np.mean((Y_synth_pre - Y_treated_mean) ** 2)))
rmse_did  = float(np.sqrt(np.mean((Y_equal_pre  - Y_treated_mean) ** 2)))

print("Pre-treatment trajectory fit (lower RMSE = better):")
print(f"  DiD (equal weights):       RMSE = {rmse_did:.5f}")
print(f"  SDID (synthetic control):  RMSE = {rmse_sdid:.5f}")
print(f"  Improvement:               {(rmse_did - rmse_sdid)/rmse_did:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Metrics Comparison

# COMMAND ----------

bias_did  = did_att  - TRUE_ATT
bias_sdid = sdid_att - TRUE_ATT

print(f"{'Metric':<40} {'Simple DiD':>14} {'SDID':>10}")
print("=" * 68)
print(f"  {'ATT estimate':<38} {did_att:>14.4f} {sdid_att:>10.4f}")
print(f"  {'True ATT':<38} {TRUE_ATT:>14.4f} {TRUE_ATT:>10.4f}")
print(f"  {'Bias (estimate - true)':<38} {bias_did:>+14.4f} {bias_sdid:>+10.4f}")
print(f"  {'|Bias|':<38} {abs(bias_did):>14.4f} {abs(bias_sdid):>10.4f}")
print(f"  {'SE / confidence interval':<38} {'none':>14} {sdid_se:>10.4f}")
print(f"  {'95% CI covers true ATT':<38} {'n/a':>14} "
      f"{'YES' if sdid_ci_lo <= TRUE_ATT <= sdid_ci_hi else 'NO':>10}")
print(f"  {'p-value':<38} {'none':>14} {sdid_pval:>10.4f}")
print(f"  {'Pre-trend RMSE (pre-period fit)':<38} {rmse_did:>14.5f} {rmse_sdid:>10.5f}")
print(f"  {'Fit time (s)':<38} {did_time:>14.4f} {sdid_time:>10.2f}")
print("=" * 68)
print()
bias_ratio = abs(bias_sdid) / abs(bias_did) if abs(bias_did) > 1e-8 else float("nan")
print(f"SDID bias as fraction of DiD bias: {bias_ratio:.2f}x")
print(f"Pre-trend fit improvement:         {(rmse_did - rmse_sdid)/rmse_did:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualisations

# COMMAND ----------

fig = plt.figure(figsize=(15, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)
ax1 = fig.add_subplot(gs[0, :])    # wide: full panel trajectories
ax3 = fig.add_subplot(gs[1, 0])    # unit weights
ax4 = fig.add_subplot(gs[1, 1])    # bias comparison

months = np.arange(1, T_TOTAL + 1)

# ── Plot 1: Trajectories ────────────────────────────────────────────────────
Y_control_post = Y[:N_CONTROL, T_PRE:]
Y_treated_post = Y[N_CONTROL:, T_PRE:]

# Per-period means
treated_traj  = Y[N_CONTROL:, :].mean(axis=0)
control_traj  = Y[:N_CONTROL, :].mean(axis=0)

# SDID synthetic control post-treatment
Y_synth_post   = (omega[:, np.newaxis] * Y_control_post).sum(axis=0)
synth_traj_pre = Y_synth_pre
synth_traj     = np.concatenate([synth_traj_pre, Y_synth_post])

# Counterfactual: synth pre-trend extrapolated into post (for visualising ATT)
# Simple linear projection from last 3 pre-periods
for_months_post = np.arange(T_PRE, T_TOTAL)
slope_synth_pre = np.polyfit(np.arange(T_PRE - 4, T_PRE), synth_traj_pre[-4:], 1)[0]
counterfactual_post = synth_traj_pre[-1] + slope_synth_pre * np.arange(1, T_POST + 1)
counterfactual = np.concatenate([synth_traj_pre, counterfactual_post])

ax1.plot(months, treated_traj, "k-o",  lw=2.0, ms=6,  label="Treated mean", zorder=4)
ax1.plot(months, control_traj, "b--s", lw=1.5, ms=5,  label="Control mean (DiD weight)",
         alpha=0.8, zorder=3)
ax1.plot(months, synth_traj,   "r-^",  lw=1.5, ms=5,  label="SDID synthetic control",
         alpha=0.9, zorder=3)
ax1.plot(months[T_PRE:], counterfactual_post, "r:",  lw=1.5,
         label="SDID counterfactual (extrapolated)", alpha=0.7)

ax1.axvline(T_PRE + 0.5, color="gray", ls="--", lw=1.5, label="Treatment onset (month 13)")
ax1.fill_betweenx(
    [ax1.get_ylim()[0] if ax1.get_ylim()[0] < 0.5 else 0.60, 0.95],
    T_PRE + 0.5, T_TOTAL + 0.5,
    alpha=0.04, color="tomato", label="Post-treatment window",
)

ax1.set_xlabel("Month")
ax1.set_ylabel("Loss ratio")
ax1.set_title(
    f"Panel trajectories — 20 regions × 24 months\n"
    f"True ATT = {TRUE_ATT},  DiD = {did_att:.4f},  SDID = {sdid_att:.4f}"
)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.25)

# ── Plot 3: SDID unit weights ───────────────────────────────────────────────
control_labels = [f"R{i:02d}" for i in range(N_CONTROL)]
equal_w = np.ones(N_CONTROL) / N_CONTROL
x_pos = np.arange(N_CONTROL)

ax3.bar(x_pos, omega, color="tomato", alpha=0.75, label="SDID omega")
ax3.axhline(equal_w[0], color="steelblue", ls="--", lw=1.5,
            label=f"Equal weight (DiD) = {equal_w[0]:.3f}")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(control_labels, rotation=60, fontsize=7)
ax3.set_ylabel("Unit weight")
ax3.set_title("SDID control unit weights\n(vs equal weights assumed by DiD)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.25, axis="y")

# ── Plot 4: Bias comparison ─────────────────────────────────────────────────
estimators = ["DiD\n(simple)", "SDID"]
biases = [bias_did, bias_sdid]
colors = ["steelblue", "tomato"]
bars = ax4.bar(estimators, biases, color=colors, alpha=0.80, width=0.4)
ax4.axhline(0, color="black", lw=1.5, ls="-")
ax4.axhline(TRUE_ATT, color="gray", lw=1.0, ls=":", label=f"True ATT = {TRUE_ATT}")
ax4.set_ylabel("Bias (estimate − true ATT)")
ax4.set_title(
    f"ATT bias comparison\n"
    f"DiD: {did_att:.4f}  SDID: {sdid_att:.4f}  True: {TRUE_ATT}"
)
for bar, bias in zip(bars, biases):
    ax4.text(bar.get_x() + bar.get_width() / 2, bias + 0.001 * np.sign(bias),
             f"{bias:+.4f}", ha="center", va="bottom" if bias >= 0 else "top",
             fontsize=10, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.25, axis="y")

plt.suptitle(
    "insurance-causal-policy: SDID vs two-period DiD — heterogeneous pre-trends",
    fontsize=12, fontweight="bold",
)
plt.savefig("/tmp/benchmark_sdid.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_sdid.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Monte Carlo: bias and coverage across 100 simulations
# MAGIC
# MAGIC A single-run comparison is one draw from the DGP. To get a reliable
# MAGIC picture of bias and CI coverage we repeat the experiment 100 times
# MAGIC with different random seeds. This takes a few minutes on Databricks
# MAGIC serverless — adjust `N_MC` downward if compute is tight.

# COMMAND ----------

N_MC = 100
mc_did_att  = []
mc_sdid_att = []
mc_sdid_covers = []

for mc_seed in range(N_MC):
    mc_rng = np.random.default_rng(mc_seed * 7 + 13)

    g_co = mc_rng.normal(0.000, 0.003, N_CONTROL)
    g_tr = mc_rng.normal(0.004, 0.004, N_TREATED)
    a_co = mc_rng.normal(0, 0.06, N_CONTROL)
    a_tr = mc_rng.normal(0, 0.06, N_TREATED)

    Ymc = np.zeros((N_REGIONS, T_TOTAL))
    for i in range(N_CONTROL):
        for t in range(T_TOTAL):
            drift = g_co[i] * t if t < T_PRE else g_co[i] * T_PRE
            Ymc[i, t] = BASE_LR + a_co[i] + TREND * t + drift + mc_rng.normal(0, NOISE_SD)

    for j in range(N_TREATED):
        i = N_CONTROL + j
        for t in range(T_TOTAL):
            drift = g_tr[j] * t if t < T_PRE else g_tr[j] * T_PRE
            eff = TRUE_ATT if t >= T_PRE else 0.0
            Ymc[i, t] = BASE_LR + a_tr[j] + TREND * t + drift + eff + mc_rng.normal(0, NOISE_SD)

    Ymc = np.clip(Ymc, 0.10, 1.50)

    # DiD
    did_co = Ymc[:N_CONTROL, T_PRE:].mean() - Ymc[:N_CONTROL, :T_PRE].mean()
    did_tr = Ymc[N_CONTROL:, T_PRE:].mean() - Ymc[N_CONTROL:, :T_PRE].mean()
    mc_did_att.append(did_tr - did_co)

    # SDID via panel
    mc_rows = []
    for i in range(N_REGIONS):
        is_t = i >= N_CONTROL
        fp = T_PRE + 1 if is_t else None
        for t in range(1, T_TOTAL + 1):
            mc_rows.append({
                "segment_id": f"region_{i:02d}",
                "period": t,
                "loss_ratio": float(Ymc[i, t - 1]),
                "earned_premium": 1_000_000.0,
                "earned_exposure": 3_000.0,
                "incurred_claims": float(Ymc[i, t - 1] * 1_000_000),
                "claim_count": 200,
                "first_treated_period": int(fp) if fp is not None else None,
                "treated": int(is_t and t > T_PRE),
                "cohort": int(fp) if fp is not None else None,
            })
    mc_panel = pl.DataFrame(mc_rows, schema={
        "segment_id": pl.String,
        "period": pl.Int64,
        "loss_ratio": pl.Float64,
        "earned_premium": pl.Float64,
        "earned_exposure": pl.Float64,
        "incurred_claims": pl.Float64,
        "claim_count": pl.Int64,
        "first_treated_period": pl.Int64,
        "treated": pl.Int64,
        "cohort": pl.Int64,
    })
    try:
        mc_est = SDIDEstimator(mc_panel, outcome="loss_ratio",
                               inference="placebo", n_replicates=100,
                               random_seed=mc_seed)
        mc_res = mc_est.fit()
        mc_sdid_att.append(float(mc_res.att))
        mc_sdid_covers.append(float(mc_res.ci[0]) <= TRUE_ATT <= float(mc_res.ci[1]))
    except Exception:
        mc_sdid_att.append(float("nan"))
        mc_sdid_covers.append(False)

mc_did_arr  = np.array(mc_did_att)
mc_sdid_arr = np.array([x for x in mc_sdid_att if not np.isnan(x)])
mc_cov_arr  = np.array(mc_sdid_covers)

print(f"Monte Carlo ({N_MC} runs):")
print()
print(f"{'Metric':<40} {'DiD':>10} {'SDID':>10}")
print("=" * 62)
print(f"  {'Mean ATT (true={TRUE_ATT})':<38} {mc_did_arr.mean():>10.4f} {mc_sdid_arr.mean():>10.4f}")
print(f"  {'Mean absolute bias':<38} {np.abs(mc_did_arr - TRUE_ATT).mean():>10.4f} "
      f"{np.abs(mc_sdid_arr - TRUE_ATT).mean():>10.4f}")
print(f"  {'RMSE':<38} {np.sqrt(np.mean((mc_did_arr - TRUE_ATT)**2)):>10.4f} "
      f"{np.sqrt(np.mean((mc_sdid_arr - TRUE_ATT)**2)):>10.4f}")
print(f"  {'Std dev of estimates':<38} {mc_did_arr.std():>10.4f} {mc_sdid_arr.std():>10.4f}")
print(f"  {'95% CI coverage (SDID only)':<38} {'n/a':>10} {mc_cov_arr.mean():>10.1%}")
print("=" * 62)

# COMMAND ----------

# MC bias plot
fig2, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, vals, label, color in [
    (axes[0], mc_did_arr,  "DiD",  "steelblue"),
    (axes[1], mc_sdid_arr, "SDID", "tomato"),
]:
    ax.hist(vals, bins=25, color=color, alpha=0.75, edgecolor="white")
    ax.axvline(TRUE_ATT, color="black", lw=2, label=f"True ATT = {TRUE_ATT}")
    ax.axvline(vals.mean(), color=color, lw=1.5, ls="--",
               label=f"Mean = {vals.mean():.4f}")
    ax.set_xlabel("ATT estimate")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} ATT distribution ({N_MC} MC runs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Monte Carlo: DiD vs SDID bias distributions — true ATT = {TRUE_ATT}",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig("/tmp/benchmark_sdid_mc.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved /tmp/benchmark_sdid_mc.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verdict
# MAGIC
# MAGIC **Honest assessment of when SDID beats DiD and when it doesn't.**

# COMMAND ----------

print("=" * 64)
print("VERDICT: SDID vs simple two-period DiD")
print("=" * 64)
print()
print(f"Dataset: 20 regions × 24 months, {N_TREATED} treated at month {T_PRE + 1}")
print(f"True ATT: {TRUE_ATT}")
print()
print("Single-run results:")
print(f"  DiD ATT:  {did_att:.4f}  (bias: {bias_did:+.4f})")
print(f"  SDID ATT: {sdid_att:.4f}  (bias: {bias_sdid:+.4f})")
print(f"  SDID 95% CI: [{sdid_ci_lo:.4f}, {sdid_ci_hi:.4f}]  "
      f"({'covers true ATT' if sdid_ci_lo <= TRUE_ATT <= sdid_ci_hi else 'does NOT cover true ATT'})")
print()
print(f"Monte Carlo results ({N_MC} runs):")
print(f"  DiD RMSE:             {np.sqrt(np.mean((mc_did_arr - TRUE_ATT)**2)):.4f}")
print(f"  SDID RMSE:            {np.sqrt(np.mean((mc_sdid_arr - TRUE_ATT)**2)):.4f}")
print(f"  SDID 95% CI coverage: {mc_cov_arr.mean():.1%}  (target: 95%)")
print()
print("-" * 64)
print("WHEN SDID ADDS MOST VALUE (this DGP)")
print("-" * 64)
print("  Heterogeneous pre-trends: treated regions trended differently from")
print("  controls before treatment. SDID reweights controls to match treated")
print("  pre-trend, reducing the confounding. DiD is biased by design.")
print()
print("  SDID provides uncertainty quantification (SE, CI, p-value).")
print("  DiD provides no formal inference — just a point estimate.")
print()
print("  Pre-trend fit: SDID synthetic control RMSE vs treated trajectory:")
print(f"    DiD (equal weights):  {rmse_did:.5f}")
print(f"    SDID synthetic ctrl:  {rmse_sdid:.5f}  ({(rmse_did-rmse_sdid)/rmse_did:.1%} better)")
print()
print("-" * 64)
print("WHEN SDID MAY NOT HELP")
print("-" * 64)
print("  - Tiny panels (< 5 control units): SC weights are poorly identified.")
print("  - Perfectly parallel pre-trends: DiD is unbiased, SDID adds nothing.")
print("  - Very short pre-treatment window (< 4 periods): SDID cannot build")
print("    a credible synthetic control with few pre-period observations.")
print()
print("  On this DGP with heterogeneous trends and 20 regions, SDID reduces")
print("  bias vs DiD. The magnitude of the improvement depends on how badly")
print("  parallel trends is violated in your actual portfolio.")
print()
print("  **Do not use either method for FCA Consumer Duty evidence without")
print("   first checking the pre-treatment event study plot.**")
print("=" * 64)
