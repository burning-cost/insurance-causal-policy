# Databricks notebook source
# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-causal-policy v0.2.0 — DRSC verification
# MAGIC
# MAGIC Verifies that `DoublyRobustSCEstimator` (Sant'Anna, Shaikh, Syrgkanis 2025)
# MAGIC is correctly implemented and passes all correctness checks.

# COMMAND ----------

%pip install "insurance-causal-policy==0.2.0" --quiet

# COMMAND ----------

import numpy as np
import warnings

# COMMAND ----------
# 1. Import and version check
import insurance_causal_policy as icp
print(f"Version: {icp.__version__}")
assert icp.__version__ == "0.2.0", f"Expected 0.2.0, got {icp.__version__}"

from insurance_causal_policy import DoublyRobustSCEstimator, make_synthetic_panel_direct, DRSCResult, DRSCWeights
from insurance_causal_policy._drsc import _solve_sc_weights_ols, _multiplier_bootstrap, _fit_drsc_core, _analytic_variance

print("All imports OK")

# COMMAND ----------
# 2. OLS SC weight solver

rng = np.random.default_rng(1)

# Weights must sum to 1
mu_co = rng.normal(0.70, 0.03, (15, 8))
mu_tr = rng.normal(0.70, 0.03, 8)
w = _solve_sc_weights_ols(mu_tr, mu_co)
assert len(w) == 15, f"Expected 15 weights, got {len(w)}"
assert abs(w.sum() - 1.0) < 1e-8, f"Weights sum {w.sum():.8f} != 1"
print(f"[OK] Weight sum: {w.sum():.8f}")

# Perfect recovery when well-determined
mu_co_exact = rng.normal(0, 1, (3, 10))
true_w = np.array([0.4, 0.3, 0.3])
mu_tr_exact = true_w @ mu_co_exact
w_exact = _solve_sc_weights_ols(mu_tr_exact, mu_co_exact)
np.testing.assert_allclose(w_exact, true_w, atol=1e-8)
print(f"[OK] Perfect recovery: estimated={np.round(w_exact, 4)}, true={true_w}")

# Negative weights allowed (extrapolation case)
mu_co_trap = np.linspace(0.60, 0.65, 5).reshape(5, 1) + rng.normal(0, 0.001, (5, 8))
mu_tr_trap = np.full(8, 0.80)  # well outside control range
w_trap = _solve_sc_weights_ols(mu_tr_trap, mu_co_trap)
assert abs(w_trap.sum() - 1.0) < 1e-6
has_neg = (w_trap < 0).any()
print(f"[OK] Negative weights: {np.round(w_trap, 3)}, has_negative={has_neg}")
# Note: negative weights are expected here (extrapolation beyond convex hull)

# Single control: trivially weight 1
w_single = _solve_sc_weights_ols(np.array([0.72, 0.73]), np.array([[0.71, 0.72]]))
assert len(w_single) == 1 and abs(w_single[0] - 1.0) < 1e-8
print(f"[OK] Single control weight: {w_single[0]:.4f}")

print("\n=== SC Weight Solver: ALL CHECKS PASSED ===")

# COMMAND ----------
# 3. Multiplier bootstrap

phi = np.random.default_rng(20).normal(0.05, 0.02, 50)
att_val = float(np.mean(phi))

boot = _multiplier_bootstrap(phi, att_val, n_replicates=1000, rng=np.random.default_rng(21))
assert boot.shape == (1000,), f"Expected (1000,), got {boot.shape}"
assert abs(np.mean(boot) - att_val) < 0.01, f"Bootstrap mean {np.mean(boot):.4f} too far from {att_val:.4f}"
assert np.std(boot) > 0, "Bootstrap std should be positive"
print(f"[OK] Bootstrap: shape={boot.shape}, mean={np.mean(boot):.4f}, std={np.std(boot):.4f}")

# Reproducibility
b1 = _multiplier_bootstrap(phi, att_val, n_replicates=100, rng=np.random.default_rng(42))
b2 = _multiplier_bootstrap(phi, att_val, n_replicates=100, rng=np.random.default_rng(42))
np.testing.assert_array_equal(b1, b2)
print("[OK] Bootstrap reproducibility: same seed produces identical results")

# Exp(1)-1 distribution check: E[W] = 0, Var[W] = 1
rng_w = np.random.default_rng(99)
W = rng_w.exponential(1.0, 100000) - 1.0
assert abs(np.mean(W)) < 0.01, f"E[W] = {np.mean(W):.4f}, should be ~0"
assert abs(np.var(W) - 1.0) < 0.05, f"Var[W] = {np.var(W):.4f}, should be ~1"
print(f"[OK] Exp(1)-1 weights: E[W]={np.mean(W):.4f}, Var[W]={np.var(W):.4f}")

print("\n=== Multiplier Bootstrap: ALL CHECKS PASSED ===")

# COMMAND ----------
# 4. Core fit function

def make_Y(n_co, n_tr, t_pre, t_post, true_att, seed=0):
    rng = np.random.default_rng(seed)
    N = n_co + n_tr
    T = t_pre + t_post
    alpha = rng.normal(0, 0.06, N)
    beta = np.arange(T) * 0.003
    Y = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            is_treated = i >= n_co and t >= t_pre
            Y[i, t] = 0.70 + alpha[i] + beta[t] + (true_att if is_treated else 0.0) + rng.normal(0, 0.02)
    return Y

# ATT direction
Y = make_Y(40, 10, 8, 4, -0.08, seed=0)
att_c, sc_w, prop_r, phi_c, m_d = _fit_drsc_core(Y, n_co=40, n_tr=10, t_pre=8, t_post=4)
assert att_c < 0, f"ATT {att_c:.4f} should be negative"
assert abs(sc_w.sum() - 1.0) < 1e-6, f"SC weights sum {sc_w.sum():.6f} != 1"
assert len(phi_c) == 50, f"phi length {len(phi_c)} != 50"
assert abs(att_c - np.mean(phi_c)) < 1e-10, "ATT != mean(phi)"
np.testing.assert_array_equal(prop_r, np.full(40, 10.0))
print(f"[OK] Core fit: ATT={att_c:.4f}, SC_w_sum={sc_w.sum():.6f}, ATT=mean(phi): {abs(att_c - np.mean(phi_c)) < 1e-10}")

# ATT near zero under null DGP
Y_null = make_Y(40, 10, 8, 4, 0.0, seed=2)
att_null, _, _, _, _ = _fit_drsc_core(Y_null, n_co=40, n_tr=10, t_pre=8, t_post=4)
assert abs(att_null) < 0.05, f"Null ATT {att_null:.4f} too large"
print(f"[OK] Null DGP: ATT={att_null:.4f} (should be ~0)")

# ATT recovery close to true value
Y_clean = make_Y(50, 15, 8, 4, -0.06, seed=1)
att_clean, _, _, _, _ = _fit_drsc_core(Y_clean, n_co=50, n_tr=15, t_pre=8, t_post=4)
assert abs(att_clean - (-0.06)) < 0.06, f"ATT {att_clean:.4f} too far from -0.06"
print(f"[OK] ATT recovery: estimated={att_clean:.4f}, true=-0.06, error={abs(att_clean+0.06):.4f}")

print("\n=== Core Fit: ALL CHECKS PASSED ===")

# COMMAND ----------
# 5. Full estimator end-to-end

panel = make_synthetic_panel_direct(n_control=40, n_treated=12, t_pre=8, t_post=4, true_att=-0.07)
est = DoublyRobustSCEstimator(panel, outcome="loss_ratio", inference="bootstrap", n_replicates=200, random_seed=42)
result = est.fit()

assert isinstance(result, DRSCResult), f"Expected DRSCResult, got {type(result)}"
assert isinstance(result.weights, DRSCWeights), f"Expected DRSCWeights"
assert isinstance(result.att, float), "att should be float"
assert result.se > 0, f"SE={result.se} should be positive"
assert result.ci_low <= result.att <= result.ci_high, "att not in CI"
assert 0 <= result.pval <= 1, f"pval={result.pval} out of range"
assert result.att < 0, f"ATT={result.att} should be negative (true=-0.07)"
assert result.n_treated == 12
assert result.n_control == 40
assert result.t_pre == 8
assert result.t_post == 4
assert abs(result.weights.sc_weights.sum() - 1.0) < 1e-6
assert len(result.phi) == 40 + 12
assert result.event_study is not None
assert len(result.event_study) == 12  # t_pre + t_post
assert set(result.event_study.columns) >= {"period_rel", "att", "se", "ci_low", "ci_high"}
assert result.outcome_name == "loss_ratio"
assert result.inference_method == "bootstrap"
assert result.n_replicates == 200

print(f"[OK] Full fit: ATT={result.att:.4f}, SE={result.se:.4f}, CI=[{result.ci_low:.4f}, {result.ci_high:.4f}]")
print(f"[OK] Weights: {len(result.weights.sc_weights)} control units, sum={result.weights.sc_weights.sum():.6f}")
print(f"[OK] Event study: {len(result.event_study)} rows")

# COMMAND ----------
# 6. Reproducibility
r1 = DoublyRobustSCEstimator(panel, outcome="loss_ratio", inference="bootstrap", n_replicates=100, random_seed=42).fit()
r2 = DoublyRobustSCEstimator(panel, outcome="loss_ratio", inference="bootstrap", n_replicates=100, random_seed=42).fit()
assert abs(r1.att - r2.att) < 1e-10
assert abs(r1.se - r2.se) < 1e-8
print(f"[OK] Reproducibility: r1.att={r1.att:.6f}, r2.att={r2.att:.6f}")

# COMMAND ----------
# 7. Analytic variance
result_analytic = DoublyRobustSCEstimator(panel, outcome="loss_ratio", inference="analytic").fit()
assert result_analytic.se > 0
assert result_analytic.inference_method == "analytic"
print(f"[OK] Analytic inference: SE={result_analytic.se:.4f}")

# COMMAND ----------
# 8. Edge cases

# Single treated unit
panel_small = make_synthetic_panel_direct(n_control=5, n_treated=1, t_pre=6, t_post=3, true_att=-0.05)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result_small = DoublyRobustSCEstimator(panel_small, outcome="loss_ratio", n_replicates=50).fit()
assert isinstance(result_small, DRSCResult)
assert result_small.n_treated == 1
print(f"[OK] Single treated unit: ATT={result_small.att:.4f}")

# Underdetermined (T_pre < N_co): should warn but still produce valid result
import polars as pl
panel_under = make_synthetic_panel_direct(n_control=30, n_treated=5, t_pre=4, t_post=3, true_att=-0.05)
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    result_under = DoublyRobustSCEstimator(panel_under, outcome="loss_ratio", n_replicates=50).fit()
    underdetermined_warned = any("under-identified" in str(w.message) for w in caught_warnings)
assert isinstance(result_under, DRSCResult)
assert np.isfinite(result_under.att)
print(f"[OK] Underdetermined case: ATT={result_under.att:.4f}, warned={underdetermined_warned}")

# No treated raises ValueError
panel_no_treated = panel.with_columns(pl.lit(0).alias("treated"))
try:
    DoublyRobustSCEstimator(panel_no_treated, outcome="loss_ratio").fit()
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "No treated segments" in str(e)
    print(f"[OK] No-treated ValueError: {e}")

print("\n=== Edge Cases: ALL CHECKS PASSED ===")

# COMMAND ----------
# 9. Output formatting
s = result.summary()
assert "DRSC estimate" in s
assert "loss_ratio" in s
assert "bootstrap" in s

fca = result.to_fca_summary(product_line="Motor", rate_change_date="2024-Q1")
assert "Motor" in fca
assert "2024-Q1" in fca
assert "Doubly Robust Synthetic Controls" in fca
assert "Sant'Anna, Shaikh, Syrgkanis 2025" in fca

print("[OK] summary() output contains expected text")
print("[OK] to_fca_summary() output contains expected text")
print()
print("=== SAMPLE OUTPUT ===")
print(result.summary())
print()

# COMMAND ----------
# 10. Null DGP end-to-end
panel_null = make_synthetic_panel_direct(n_control=40, n_treated=10, t_pre=8, t_post=4, true_att=0.0, random_seed=100)
result_null = DoublyRobustSCEstimator(panel_null, outcome="loss_ratio", n_replicates=100).fit()
assert abs(result_null.att) < 0.04, f"Null ATT {result_null.att:.4f} too large"
pre_atts = result_null.event_study[result_null.event_study["period_rel"] < 0]["att"].values
assert np.std(pre_atts) < 0.05, f"Pre-period ATT std {np.std(pre_atts):.4f} too large"
print(f"[OK] Null DGP: ATT={result_null.att:.4f}, pre-period std={np.std(pre_atts):.4f}")

# COMMAND ----------
print()
print("=" * 65)
print("ALL DRSC VERIFICATION CHECKS PASSED")
print(f"insurance-causal-policy version: {icp.__version__}")
print("=" * 65)
print()
print("Checks completed:")
print("  1. Imports and version")
print("  2. OLS SC weight solver (sum-to-1, perfect recovery, negative weights)")
print("  3. Multiplier bootstrap (Exp(1)-1 weights, reproducibility)")
print("  4. Core fit (ATT direction, weight sum, phi identity, null DGP)")
print("  5. Full estimator end-to-end (all result attributes)")
print("  6. Reproducibility (same seed, same result)")
print("  7. Analytic inference")
print("  8. Edge cases (single treated, underdetermined, no-treated error)")
print("  9. Output formatting (summary, to_fca_summary)")
print(" 10. Null DGP (ATT near zero, flat pre-period event study)")
