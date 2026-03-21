"""
Benchmark: DRSC vs SDID — performance under few donor units.

The core question: when is Doubly Robust Synthetic Control (DRSC) worth
using over plain SDID?

Answer in short: when N_co is small (fewer than ~10 donor units). With few
control units, SC weight estimation is underdetermined — the optimiser
concentrates weight on a handful of units and the resulting synthetic control
has high variance. DRSC uses OLS weights (unconstrained, can go negative) plus
a multiplier bootstrap that is valid under EITHER SC identification OR parallel
trends, so the variance estimate remains reliable even when the SC fit is poor.

This benchmark tests two scenarios:
  1. Small N_co (6 control units): the fragile-weights regime.
  2. Large N_co (40 control units): the well-identified regime.

Both methods should recover the true ATT without systematic bias under the
factor model DGP. The question is: which has lower RMSE, and is the CI
coverage better calibrated?

DGP: standard factor model
  Y_it = alpha_i + beta_t + tau * D_it + epsilon_it

alpha_i ~ N(0, 0.06)  — unit fixed effects
beta_t  = 0.003 * t   — common time trend (claims inflation)
tau     = -0.06       — true ATT
epsilon ~ N(0, 0.03)

SDID uses: CVXPY constrained weight optimisation, placebo/bootstrap/jackknife
DRSC uses:  OLS unconstrained SC weights, multiplier bootstrap (Exp(1)-1)
            Both are consistent if SC OR parallel trends holds.

Reference: Sant'Anna, Shaikh, Syrgkanis (2025). Doubly Robust Synthetic
Controls. arXiv:2503.11375.

Run (locally — syntax check only; do not execute on Pi):
    cd /home/ralph/repos/insurance-causal-policy
    python -c "import ast; ast.parse(open('benchmarks/drsc_vs_sdid.py').read()); print('OK')"

On Databricks:
    Use notebooks/drsc_vs_sdid.py (pip-installs from PyPI, runs 100 MC sims).
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 72)
print("Benchmark: DRSC vs SDID (insurance-causal-policy)")
print("=" * 72)
print()

# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

try:
    from insurance_causal_policy import (
        SDIDEstimator,
        DoublyRobustSCEstimator,
        make_synthetic_panel_direct,
    )
    print("insurance-causal-policy imported OK")
    print()
except ImportError as e:
    print(f"ERROR: Could not import: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# DGP parameters
# ---------------------------------------------------------------------------

TRUE_ATT     = -0.06
T_PRE        = 8
T_POST       = 4
N_TREATED    = 5
NOISE_SD     = 0.03
BASE_LR      = 0.70
TREND        = 0.003

# Number of Monte Carlo simulations
# 30 is a quick check; use 100+ on Databricks for stable results
N_SIMS = int(getattr(sys, "_drsc_bench_n_sims", 30))

INFERENCE_SDID = "jackknife"   # fast for MC
INFERENCE_DRSC = "bootstrap"   # DRSC uses multiplier bootstrap (Exp(1)-1)


# ---------------------------------------------------------------------------
# Monte Carlo runner
# ---------------------------------------------------------------------------

def run_scenario(label: str, n_co: int, n_sims: int) -> dict:
    """Run N_SIMS Monte Carlo replications for both estimators."""
    print(f"Scenario: {label}  (N_co={n_co}, N_tr={N_TREATED}, {n_sims} sims)")

    sdid_atts, sdid_ses, sdid_covers = [], [], []
    drsc_atts, drsc_ses, drsc_covers = [], [], []

    for sim_idx in range(n_sims):
        seed = 1000 + sim_idx

        panel = make_synthetic_panel_direct(
            n_control=n_co,
            n_treated=N_TREATED,
            t_pre=T_PRE,
            t_post=T_POST,
            true_att=TRUE_ATT,
            noise_sd=NOISE_SD,
            base_lr=BASE_LR,
            trend=TREND,
            random_seed=seed,
        )

        # SDID
        try:
            r = SDIDEstimator(
                panel, outcome="loss_ratio",
                inference=INFERENCE_SDID, n_replicates=50, random_seed=seed,
            ).fit()
            sdid_atts.append(r.att)
            sdid_ses.append(r.se)
            sdid_covers.append(r.ci_low <= TRUE_ATT <= r.ci_high)
        except Exception:
            pass

        # DRSC
        try:
            r = DoublyRobustSCEstimator(
                panel, outcome="loss_ratio",
                inference=INFERENCE_DRSC, n_replicates=100, random_seed=seed,
            ).fit()
            drsc_atts.append(r.att)
            drsc_ses.append(r.se)
            drsc_covers.append(r.ci_low <= TRUE_ATT <= r.ci_high)
        except Exception:
            pass

        if (sim_idx + 1) % 10 == 0:
            print(f"  {sim_idx + 1}/{n_sims} sims complete")

    def _summarise(atts, ses, covers, name):
        arr = np.array(atts)
        if len(arr) == 0:
            return {k: float("nan") for k in ["name", "n_valid", "mean_att", "abs_bias", "rmse", "std", "mean_se", "coverage"]}
        return {
            "name": name,
            "n_valid": len(arr),
            "mean_att": float(np.mean(arr)),
            "abs_bias": float(abs(np.mean(arr) - TRUE_ATT)),
            "rmse": float(np.sqrt(np.mean((arr - TRUE_ATT) ** 2))),
            "std": float(np.std(arr)),
            "mean_se": float(np.mean(ses)) if ses else float("nan"),
            "coverage": float(np.mean(covers)) if covers else float("nan"),
        }

    return {
        "label": label,
        "n_co": n_co,
        "sdid": _summarise(sdid_atts, sdid_ses, sdid_covers, "SDID"),
        "drsc": _summarise(drsc_atts, drsc_ses, drsc_covers, "DRSC"),
    }


# ---------------------------------------------------------------------------
# Run scenarios
# ---------------------------------------------------------------------------

scenarios = [
    ("Few donors (N_co=6)",  6,  N_SIMS),
    ("Many donors (N_co=40)", 40, N_SIMS),
]

results = []
for label, n_co, n_sims in scenarios:
    res = run_scenario(label, n_co, n_sims)
    results.append(res)


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

def _pct(x):
    return f"{x:>8.1%}" if x == x else "     n/a"


print()
print("=" * 72)
print("RESULTS SUMMARY")
print("=" * 72)
print(f"True ATT = {TRUE_ATT:.4f}  |  SDID inference: {INFERENCE_SDID}  |  DRSC inference: {INFERENCE_DRSC}")
print()

for res in results:
    label = res["label"]
    sdid  = res["sdid"]
    drsc  = res["drsc"]

    print(f"--- {label} ---")
    print()
    print(f"  {'Metric':<35} {'SDID':>10} {'DRSC':>10}  Winner")
    print(f"  {'-'*60}")

    metrics = [
        ("Mean ATT", "mean_att", ".4f"),
        ("Absolute bias", "abs_bias", ".4f"),
        ("RMSE", "rmse", ".4f"),
        ("Std dev of estimates", "std", ".4f"),
        ("Mean reported SE", "mean_se", ".4f"),
    ]

    for label_m, key, fmt in metrics:
        sv, dv = sdid[key], drsc[key]
        if key in ("abs_bias", "rmse", "std", "mean_se"):
            w = "DRSC" if dv < sv else ("SDID" if sv < dv else "tie")
        else:
            w = ""
        sv_s = f"{sv:>10{fmt}}" if sv == sv else "       n/a"
        dv_s = f"{dv:>10{fmt}}" if dv == dv else "       n/a"
        print(f"  {label_m:<35} {sv_s} {dv_s}  {w}")

    sc, dc = sdid["coverage"], drsc["coverage"]
    w_cov = "DRSC" if abs(dc - 0.95) < abs(sc - 0.95) else "SDID"
    print(f"  {'95% CI coverage':<35} {_pct(sc):>10} {_pct(dc):>10}  closest to 95%: {w_cov}")
    print(f"  Valid replications: SDID={sdid['n_valid']}  DRSC={drsc['n_valid']}")
    print()


# ---------------------------------------------------------------------------
# Summary interpretation
# ---------------------------------------------------------------------------

few = results[0]
many = results[1]

few_bias_imp = (few["sdid"]["abs_bias"] - few["drsc"]["abs_bias"]) / max(few["sdid"]["abs_bias"], 1e-6) * 100
few_rmse_imp = (few["sdid"]["rmse"] - few["drsc"]["rmse"]) / max(few["sdid"]["rmse"], 1e-6) * 100
many_rmse_imp = (many["sdid"]["rmse"] - many["drsc"]["rmse"]) / max(many["sdid"]["rmse"], 1e-6) * 100

print("INTERPRETATION")
print("-" * 72)
print(f"\nFew donors (N_co={few['n_co']}):")
print(f"  SDID bias={few['sdid']['abs_bias']:.4f}  DRSC bias={few['drsc']['abs_bias']:.4f}")
print(f"  SDID RMSE={few['sdid']['rmse']:.4f}  DRSC RMSE={few['drsc']['rmse']:.4f}")
if abs(few_rmse_imp) > 5:
    winner = "DRSC" if few_rmse_imp > 0 else "SDID"
    print(f"  {winner} reduces RMSE by {abs(few_rmse_imp):.1f}% over the other.")
else:
    print(f"  RMSE difference is small ({abs(few_rmse_imp):.1f}%).")

print(f"\nMany donors (N_co={many['n_co']}):")
print(f"  SDID RMSE={many['sdid']['rmse']:.4f}  DRSC RMSE={many['drsc']['rmse']:.4f}")
if abs(many_rmse_imp) < 5:
    print(f"  Both methods perform equivalently at large N_co.")
else:
    winner = "DRSC" if many_rmse_imp > 0 else "SDID"
    print(f"  {winner} is {abs(many_rmse_imp):.1f}% better at large N_co.")

print()
print("PRACTICAL GUIDANCE")
print("  N_co < 10 : DRSC (doubly robust — valid under SC or PT)")
print("  N_co 10-20: run both; compare estimates and CI coverage")
print("  N_co > 20 : SDID (CVXPY weights stable, interpretation simpler)")
print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s")
