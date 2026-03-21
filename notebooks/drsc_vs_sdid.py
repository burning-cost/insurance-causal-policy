# Databricks notebook source
# MAGIC %md
# MAGIC # DRSC vs SDID: Benchmark under few donor units
# MAGIC
# MAGIC **Library:** `insurance-causal-policy` — Doubly Robust Synthetic Control (DRSC)
# MAGIC vs Synthetic Difference-in-Differences (SDID)
# MAGIC
# MAGIC ## The question
# MAGIC
# MAGIC SDID builds a weighted synthetic control from donor (control) units. When
# MAGIC you have 30+ donors, the weight optimisation is well-determined and SDID works
# MAGIC well. But in insurance you often have fewer:
# MAGIC
# MAGIC - A new product with only 5 comparable territories
# MAGIC - A niche channel with 7 comparable segments
# MAGIC - A regional product where only 4 other regions exist
# MAGIC
# MAGIC With few donors, the SDID CVXPY weight optimisation is underdetermined. The
# MAGIC optimiser concentrates mass on a handful of units and the resulting synthetic
# MAGIC control has high variance.
# MAGIC
# MAGIC **DRSC** (Sant'Anna, Shaikh, Syrgkanis 2025, arXiv:2503.11375) estimates SC
# MAGIC weights via unconstrained OLS (negative weights allowed) and uses a multiplier
# MAGIC bootstrap valid under EITHER SC identification OR parallel trends. This provides
# MAGIC a consistency guarantee without requiring the SC fit to be exact.
# MAGIC
# MAGIC ## Benchmark design
# MAGIC
# MAGIC - **Scenario 1 (few donors):** N_co = 6, N_tr = 5, 100 MC sims
# MAGIC - **Scenario 2 (many donors):** N_co = 40, N_tr = 5, 100 MC sims
# MAGIC - True ATT = -0.06, standard factor model DGP
# MAGIC - Compare: mean ATT, bias, RMSE, Std, SE, 95% CI coverage
# MAGIC
# MAGIC **Reference:** Sant'Anna, Shaikh, Syrgkanis (2025). Doubly Robust Synthetic
# MAGIC Controls. arXiv:2503.11375.
# MAGIC
# MAGIC **Date:** 2026-03-21  |  **Library version:** 0.2.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install "insurance-causal-policy>=0.2.0" cvxpy matplotlib numpy scipy polars pandas --quiet

# COMMAND ----------

import insurance_causal_policy
print(f"Package version: {insurance_causal_policy.__version__}")

from insurance_causal_policy import (
    SDIDEstimator,
    DoublyRobustSCEstimator,
    make_synthetic_panel_direct,
)
print("Imports OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. DGP parameters

# COMMAND ----------

import warnings, time, json, numpy as np
warnings.filterwarnings("ignore")

TRUE_ATT  = -0.06
T_PRE     = 8
T_POST    = 4
N_TREATED = 5
NOISE_SD  = 0.03
BASE_LR   = 0.70
TREND     = 0.003
N_SIMS    = 100

INFERENCE_SDID = "jackknife"   # fast for MC
INFERENCE_DRSC = "bootstrap"   # DRSC multiplier bootstrap (Exp(1)-1)

print(f"True ATT = {TRUE_ATT}  |  {N_SIMS} sims per scenario")
print(f"SDID inference: {INFERENCE_SDID}  |  DRSC inference: {INFERENCE_DRSC}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Monte Carlo runner

# COMMAND ----------

def run_scenario(label, n_co, n_sims):
    print(f"\nScenario: {label}  (N_co={n_co}, N_tr={N_TREATED})")
    sdid_atts, sdid_covers, sdid_ses = [], [], []
    drsc_atts, drsc_covers, drsc_ses = [], [], []

    for sim_idx in range(n_sims):
        seed = 1000 + sim_idx
        panel = make_synthetic_panel_direct(
            n_control=n_co, n_treated=N_TREATED, t_pre=T_PRE, t_post=T_POST,
            true_att=TRUE_ATT, noise_sd=NOISE_SD, base_lr=BASE_LR, trend=TREND,
            random_seed=seed,
        )

        try:
            r = SDIDEstimator(
                panel, outcome="loss_ratio",
                inference=INFERENCE_SDID, n_replicates=50, random_seed=seed,
            ).fit()
            sdid_atts.append(r.att); sdid_ses.append(r.se)
            sdid_covers.append(r.ci_low <= TRUE_ATT <= r.ci_high)
        except Exception: pass

        try:
            r = DoublyRobustSCEstimator(
                panel, outcome="loss_ratio",
                inference=INFERENCE_DRSC, n_replicates=100, random_seed=seed,
            ).fit()
            drsc_atts.append(r.att); drsc_ses.append(r.se)
            drsc_covers.append(r.ci_low <= TRUE_ATT <= r.ci_high)
        except Exception as e:
            pass

        if (sim_idx + 1) % 20 == 0:
            print(f"  {sim_idx + 1}/{n_sims}")

    def _s(atts, ses, covers, name):
        a = np.array(atts)
        if len(a) == 0:
            return dict(name=name, n=0, mean_att=float("nan"), abs_bias=float("nan"),
                        rmse=float("nan"), std=float("nan"), mean_se=float("nan"),
                        coverage=float("nan"))
        return dict(name=name, n=len(a),
                    mean_att=float(np.mean(a)),
                    abs_bias=float(abs(np.mean(a) - TRUE_ATT)),
                    rmse=float(np.sqrt(np.mean((a - TRUE_ATT)**2))),
                    std=float(np.std(a)),
                    mean_se=float(np.mean(ses)) if ses else float("nan"),
                    coverage=float(np.mean(covers)) if covers else float("nan"))

    return dict(label=label, n_co=n_co,
                sdid=_s(sdid_atts, sdid_ses, sdid_covers, "SDID"),
                drsc=_s(drsc_atts, drsc_ses, drsc_covers, "DRSC"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run scenario 1: few donors (N_co=6)

# COMMAND ----------

t0 = time.time()
res_few = run_scenario("Few donors (N_co=6)", n_co=6, n_sims=N_SIMS)
print(f"Completed in {time.time()-t0:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run scenario 2: many donors (N_co=40)

# COMMAND ----------

t0 = time.time()
res_many = run_scenario("Many donors (N_co=40)", n_co=40, n_sims=N_SIMS)
print(f"Completed in {time.time()-t0:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Results

# COMMAND ----------

def print_results(res):
    print(f"\n{'='*65}")
    print(f"  {res['label']}")
    print(f"{'='*65}")
    sdid, drsc = res["sdid"], res["drsc"]
    print(f"  {'Metric':<34} {'SDID':>9} {'DRSC':>9}  Winner")
    print(f"  {'-'*58}")
    for lbl, key in [("Mean ATT", "mean_att"), ("Abs Bias", "abs_bias"),
                     ("RMSE", "rmse"), ("Std Dev", "std"), ("Mean SE", "mean_se")]:
        sv, dv = sdid[key], drsc[key]
        w = ("DRSC" if dv < sv else ("SDID" if sv < dv else "tie")) if key != "mean_att" else ""
        sv_s = f"{sv:>9.4f}" if sv == sv else "      n/a"
        dv_s = f"{dv:>9.4f}" if dv == dv else "      n/a"
        print(f"  {lbl:<34} {sv_s} {dv_s}  {w}")
    sc, dc = sdid["coverage"], drsc["coverage"]
    w_cov = "DRSC" if abs(dc-0.95) < abs(sc-0.95) else "SDID"
    print(f"  {'95% CI coverage':<34} {sc:>8.1%}  {dc:>8.1%}  {w_cov}")
    print(f"  Valid sims: SDID={sdid['n']}  DRSC={drsc['n']}")

print_results(res_few)
print_results(res_many)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visual comparison

# COMMAND ----------

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"DRSC vs SDID: Performance by Donor Pool Size\nTrue ATT = {TRUE_ATT}",
    fontsize=13, fontweight="bold"
)
colors = {"SDID": "#1f77b4", "DRSC": "#ff7f0e"}

for ax, res in zip(axes, [res_few, res_many]):
    sdid, drsc = res["sdid"], res["drsc"]
    metrics = ["abs_bias", "rmse", "std"]
    labels_m = ["Abs Bias", "RMSE", "Std Dev"]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, [sdid[m] for m in metrics], w, label="SDID", color=colors["SDID"], alpha=0.85)
    ax.bar(x + w/2, [drsc[m] for m in metrics], w, label="DRSC", color=colors["DRSC"], alpha=0.85)
    ax.set_title(res["label"])
    ax.set_xticks(x); ax.set_xticklabels(labels_m)
    ax.set_ylabel("Value (loss ratio units)")
    ax.legend()
    sc, dc = sdid["coverage"], drsc["coverage"]
    ax.text(0.5, 0.97, f"Coverage: SDID={sc:.0%}  DRSC={dc:.0%}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("/tmp/drsc_vs_sdid_results.png", dpi=150, bbox_inches="tight")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Key findings and summary

# COMMAND ----------

few_bias_imp = (res_few["sdid"]["abs_bias"] - res_few["drsc"]["abs_bias"]) / max(res_few["sdid"]["abs_bias"], 1e-6) * 100
few_rmse_imp = (res_few["sdid"]["rmse"] - res_few["drsc"]["rmse"]) / max(res_few["sdid"]["rmse"], 1e-6) * 100
many_rmse_imp = (res_many["sdid"]["rmse"] - res_many["drsc"]["rmse"]) / max(res_many["sdid"]["rmse"], 1e-6) * 100

print("KEY FINDINGS")
print("=" * 65)
print(f"\nFew donors (N_co=6):")
print(f"  DRSC bias change vs SDID: {few_bias_imp:+.1f}%")
print(f"  DRSC RMSE change vs SDID: {few_rmse_imp:+.1f}%")
print(f"\nMany donors (N_co=40):")
print(f"  DRSC RMSE change vs SDID: {many_rmse_imp:+.1f}%")
print()
print("RECOMMENDATION")
print("  N_co < 10 : DoublyRobustSCEstimator (DRSC)")
print("  N_co >= 20: SDIDEstimator (SDID)")
print("  N_co 10-20: run both; compare if estimates diverge")

# Export results for notebook.exit() capture
summary = {
    "few_donors": {"n_co": res_few["n_co"], "sdid": res_few["sdid"], "drsc": res_few["drsc"]},
    "many_donors": {"n_co": res_many["n_co"], "sdid": res_many["sdid"], "drsc": res_many["drsc"]},
    "few_bias_improvement_pct": round(few_bias_imp, 2),
    "few_rmse_improvement_pct": round(few_rmse_imp, 2),
    "many_rmse_improvement_pct": round(many_rmse_imp, 2),
}

dbutils.notebook.exit(json.dumps(summary))
