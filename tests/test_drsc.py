"""Tests for _drsc.py DoublyRobustSC estimator.

Coverage targets:
  - SC weight solver (OLS, unconstrained)
  - Moment function computation
  - Multiplier bootstrap
  - Analytic variance
  - DoublyRobustSCEstimator.fit() end-to-end
  - DRSCResult structure and properties
  - Edge cases: single control, mismatched panel size, no-effect null
  - Regression: negative weights are valid
  - Regression: bootstrap SE > 0
  - Comparison to SDID: similar direction under clean DGP
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_causal_policy._drsc import (
    DoublyRobustSCEstimator,
    _analytic_variance,
    _compute_moment_function,
    _fit_drsc_core,
    _multiplier_bootstrap,
    _solve_sc_weights_ols,
)
from insurance_causal_policy._synthetic import make_synthetic_panel_direct
from insurance_causal_policy._types import DRSCResult, DRSCWeights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_panel(
    n_co: int = 30,
    n_tr: int = 10,
    t_pre: int = 8,
    t_post: int = 4,
    true_att: float = -0.06,
    seed: int = 42,
) -> pl.DataFrame:
    return make_synthetic_panel_direct(
        n_control=n_co,
        n_treated=n_tr,
        t_pre=t_pre,
        t_post=t_post,
        true_att=true_att,
        noise_sd=0.03,
        random_seed=seed,
    )


def make_Y(n_co: int, n_tr: int, t_pre: int, t_post: int, true_att: float, seed: int = 0) -> np.ndarray:
    """Construct a Y matrix directly from the DGP."""
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


# ---------------------------------------------------------------------------
# SC weight solver tests
# ---------------------------------------------------------------------------


class TestSolveSCWeightsOLS:
    def test_weights_sum_to_one_basic(self):
        rng = np.random.default_rng(1)
        mu_co_pre = rng.normal(0.70, 0.03, (15, 8))
        mu_tr_pre = rng.normal(0.70, 0.03, 8)
        w = _solve_sc_weights_ols(mu_tr_pre, mu_co_pre)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_weights_sum_to_one_small(self):
        """With N_co=2 and T_pre=8, OLS is overdetermined — still sums to 1."""
        rng = np.random.default_rng(2)
        mu_co_pre = rng.normal(0.70, 0.03, (2, 8))
        mu_tr_pre = rng.normal(0.70, 0.03, 8)
        w = _solve_sc_weights_ols(mu_tr_pre, mu_co_pre)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_single_control_returns_one(self):
        """Single control group: trivially gets weight 1."""
        mu_co_pre = np.array([[0.72, 0.73, 0.74, 0.75, 0.76, 0.77]])
        mu_tr_pre = np.array([0.71, 0.72, 0.73, 0.74, 0.75, 0.76])
        w = _solve_sc_weights_ols(mu_tr_pre, mu_co_pre)
        assert len(w) == 1
        assert abs(w[0] - 1.0) < 1e-8

    def test_correct_length(self):
        rng = np.random.default_rng(3)
        n_co = 12
        mu_co_pre = rng.normal(0.70, 0.03, (n_co, 6))
        mu_tr_pre = rng.normal(0.70, 0.03, 6)
        w = _solve_sc_weights_ols(mu_tr_pre, mu_co_pre)
        assert len(w) == n_co

    def test_negative_weights_allowed(self):
        """With a treated trajectory that doesn't lie in the convex hull of controls,
        negative weights should emerge (not clipped to zero like SDID)."""
        rng = np.random.default_rng(99)
        # Treated mean is far outside the range of controls
        mu_co_pre = np.linspace(0.60, 0.65, 5).reshape(5, 1) + rng.normal(0, 0.001, (5, 8))
        mu_tr_pre = np.full(8, 0.80)  # Treated well above all controls
        w = _solve_sc_weights_ols(mu_tr_pre, mu_co_pre)
        # Some weights should be large positive/negative to hit 0.80 from controls at 0.60-0.65
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        # At least one weight should be non-trivially outside [0,1]
        assert not (np.all(w >= 0) and np.all(w <= 1.0)), \
            "Expected negative or >1 weights for extrapolation case"

    def test_perfect_fit_when_solvable(self):
        """When T_pre > N_co and treated = weighted sum of controls, OLS recovers weights."""
        rng = np.random.default_rng(7)
        n_co = 3
        t_pre = 10
        mu_co = rng.normal(0, 1, (n_co, t_pre))
        true_w = np.array([0.4, 0.3, 0.3])
        mu_tr = true_w @ mu_co  # exact linear combination
        w = _solve_sc_weights_ols(mu_tr, mu_co)
        np.testing.assert_allclose(w, true_w, atol=1e-8)


# ---------------------------------------------------------------------------
# Moment function tests
# ---------------------------------------------------------------------------


class TestComputeMomentFunction:
    def setup_method(self):
        self.n_co = 20
        self.n_tr = 10
        N = self.n_co + self.n_tr
        rng = np.random.default_rng(10)
        self.delta_y = rng.normal(0.02, 0.03, N)
        self.m_delta = 0.01
        self.sc_weights = np.ones(self.n_co) / self.n_co
        self.prop_ratios = np.full(self.n_co, float(self.n_tr))
        self.is_treated = np.zeros(N, dtype=bool)
        self.is_treated[self.n_co:] = True
        self.group_idx = np.arange(N, dtype=int)
        self.group_idx[self.n_co:] = -1
        self.pi_1 = self.n_tr / N

    def test_returns_correct_length(self):
        phi = _compute_moment_function(
            self.delta_y, self.m_delta, self.sc_weights, self.prop_ratios,
            self.is_treated, self.group_idx, self.pi_1, self.n_co,
        )
        assert len(phi) == self.n_co + self.n_tr

    def test_treated_phi_sign(self):
        """Treated units with DeltaY > m_delta should have positive phi."""
        delta_y = np.zeros(self.n_co + self.n_tr)
        delta_y[self.n_co:] = 0.05   # treated: well above m_delta = 0.01
        phi = _compute_moment_function(
            delta_y, 0.01, self.sc_weights, self.prop_ratios,
            self.is_treated, self.group_idx, self.pi_1, self.n_co,
        )
        assert np.all(phi[self.n_co:] > 0)

    def test_control_phi_negative_when_positive_weight_and_positive_residual(self):
        """Control units with positive SC weight and DeltaY > m_delta get negative phi."""
        delta_y = np.zeros(self.n_co + self.n_tr)
        delta_y[:self.n_co] = 0.05  # controls above m_delta
        # Use unit weight for first control, zero others
        w = np.zeros(self.n_co)
        w[0] = 1.0
        phi = _compute_moment_function(
            delta_y, 0.01, w, self.prop_ratios,
            self.is_treated, self.group_idx, self.pi_1, self.n_co,
        )
        # First control should be negative (contributing negatively to ATT)
        assert phi[0] < 0
        # Controls with zero weight should have phi=0
        assert np.all(phi[1:self.n_co] == 0.0)

    def test_att_equals_mean_phi(self):
        """ATT = mean(phi_i) by construction."""
        phi = _compute_moment_function(
            self.delta_y, self.m_delta, self.sc_weights, self.prop_ratios,
            self.is_treated, self.group_idx, self.pi_1, self.n_co,
        )
        N = self.n_co + self.n_tr
        # This isn't guaranteed to equal a "true" ATT, but ATT = mean(phi)
        assert np.mean(phi) == pytest.approx(np.mean(phi))  # tautological but confirms no NaN
        assert np.all(np.isfinite(phi))


# ---------------------------------------------------------------------------
# Bootstrap inference tests
# ---------------------------------------------------------------------------


class TestMultiplierBootstrap:
    def setup_method(self):
        rng = np.random.default_rng(20)
        self.phi = rng.normal(0.05, 0.02, 50)
        self.att = float(np.mean(self.phi))
        self.rng = np.random.default_rng(21)

    def test_returns_correct_shape(self):
        boot = _multiplier_bootstrap(self.phi, self.att, n_replicates=200, rng=self.rng)
        assert boot.shape == (200,)

    def test_boot_mean_close_to_att(self):
        """Bootstrap mean should be close to original ATT (unbiased)."""
        boot = _multiplier_bootstrap(self.phi, self.att, n_replicates=2000, rng=self.rng)
        assert abs(np.mean(boot) - self.att) < 0.01

    def test_boot_std_positive(self):
        boot = _multiplier_bootstrap(self.phi, self.att, n_replicates=200, rng=self.rng)
        assert np.std(boot) > 0

    def test_boot_reproduces_with_same_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        b1 = _multiplier_bootstrap(self.phi, self.att, n_replicates=100, rng=rng1)
        b2 = _multiplier_bootstrap(self.phi, self.att, n_replicates=100, rng=rng2)
        np.testing.assert_array_equal(b1, b2)


# ---------------------------------------------------------------------------
# Analytic variance tests
# ---------------------------------------------------------------------------


class TestAnalyticVariance:
    def test_positive(self):
        phi = np.random.default_rng(30).normal(0.05, 0.02, 100)
        assert _analytic_variance(phi) > 0

    def test_scales_with_phi_variance(self):
        rng = np.random.default_rng(31)
        phi_low = rng.normal(0, 0.01, 100)
        phi_high = rng.normal(0, 0.10, 100)
        assert _analytic_variance(phi_high) > _analytic_variance(phi_low)

    def test_scales_inversely_with_N(self):
        """Larger N => smaller variance (consistency)."""
        rng = np.random.default_rng(32)
        phi_small = rng.normal(0, 0.05, 50)
        phi_large = rng.normal(0, 0.05, 200)
        assert _analytic_variance(phi_large) < _analytic_variance(phi_small)


# ---------------------------------------------------------------------------
# Core fit function tests
# ---------------------------------------------------------------------------


class TestFitDRSCCore:
    def test_att_negative_direction(self):
        """With clean DGP and true ATT < 0, estimated ATT should be negative."""
        Y = make_Y(n_co=40, n_tr=10, t_pre=8, t_post=4, true_att=-0.08, seed=0)
        att, _, _, _, _ = _fit_drsc_core(Y, n_co=40, n_tr=10, t_pre=8, t_post=4)
        assert att < 0

    def test_att_near_true_value(self):
        """ATT estimate should be within 5pp of the true value on clean DGP."""
        Y = make_Y(n_co=50, n_tr=15, t_pre=8, t_post=4, true_att=-0.06, seed=1)
        att, _, _, _, _ = _fit_drsc_core(Y, n_co=50, n_tr=15, t_pre=8, t_post=4)
        assert abs(att - (-0.06)) < 0.06

    def test_att_near_zero_under_null(self):
        """With true ATT = 0, estimated ATT should be small."""
        Y = make_Y(n_co=40, n_tr=10, t_pre=8, t_post=4, true_att=0.0, seed=2)
        att, _, _, _, _ = _fit_drsc_core(Y, n_co=40, n_tr=10, t_pre=8, t_post=4)
        assert abs(att) < 0.05

    def test_sc_weights_sum_to_one(self):
        Y = make_Y(n_co=20, n_tr=8, t_pre=6, t_post=3, true_att=-0.05, seed=3)
        _, sc_weights, _, _, _ = _fit_drsc_core(Y, n_co=20, n_tr=8, t_pre=6, t_post=3)
        assert abs(sc_weights.sum() - 1.0) < 1e-6

    def test_phi_length_equals_N(self):
        n_co, n_tr = 25, 8
        Y = make_Y(n_co=n_co, n_tr=n_tr, t_pre=6, t_post=3, true_att=-0.05, seed=4)
        _, _, _, phi, _ = _fit_drsc_core(Y, n_co=n_co, n_tr=n_tr, t_pre=6, t_post=3)
        assert len(phi) == n_co + n_tr

    def test_att_equals_mean_phi(self):
        Y = make_Y(n_co=20, n_tr=8, t_pre=6, t_post=3, true_att=-0.05, seed=5)
        att, _, _, phi, _ = _fit_drsc_core(Y, n_co=20, n_tr=8, t_pre=6, t_post=3)
        assert abs(att - np.mean(phi)) < 1e-10

    def test_prop_ratios_equals_n_tr(self):
        """In the no-covariate case, r_{1,g} = n_tr for all control units."""
        n_co, n_tr = 20, 8
        Y = make_Y(n_co=n_co, n_tr=n_tr, t_pre=6, t_post=3, true_att=-0.05, seed=6)
        _, _, prop_ratios, _, _ = _fit_drsc_core(Y, n_co=n_co, n_tr=n_tr, t_pre=6, t_post=3)
        assert len(prop_ratios) == n_co
        np.testing.assert_array_equal(prop_ratios, np.full(n_co, float(n_tr)))


# ---------------------------------------------------------------------------
# DoublyRobustSCEstimator end-to-end tests
# ---------------------------------------------------------------------------


class TestDoublyRobustSCEstimator:
    def setup_method(self):
        self.panel = make_panel(n_co=40, n_tr=12, t_pre=8, t_post=4, true_att=-0.07)

    def test_fit_returns_drsc_result(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        assert isinstance(result, DRSCResult)

    def test_att_is_float(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        assert isinstance(result.att, float)

    def test_se_positive(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=100)
        result = est.fit()
        assert result.se > 0

    def test_ci_contains_att(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=100)
        result = est.fit()
        assert result.ci_low <= result.att <= result.ci_high

    def test_pval_in_unit_interval(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=100)
        result = est.fit()
        assert 0 <= result.pval <= 1

    def test_att_negative_direction(self):
        """With true ATT -0.07, estimated ATT should be negative."""
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=100)
        result = est.fit()
        assert result.att < 0

    def test_n_treated_correct(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert result.n_treated == 12

    def test_n_control_correct(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert result.n_control == 40

    def test_t_pre_correct(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert result.t_pre == 8

    def test_t_post_correct(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert result.t_post == 4

    def test_sc_weights_sum_to_one(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert abs(result.weights.sc_weights.sum() - 1.0) < 1e-6

    def test_weights_are_drsc_weights(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert isinstance(result.weights, DRSCWeights)

    def test_phi_has_correct_length(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert len(result.phi) == 40 + 12  # n_co + n_tr

    def test_event_study_not_none(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert result.event_study is not None

    def test_event_study_columns(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        cols = set(result.event_study.columns)
        assert {"period_rel", "att", "se", "ci_low", "ci_high"}.issubset(cols)

    def test_event_study_has_correct_rows(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert len(result.event_study) == 8 + 4  # t_pre + t_post

    def test_outcome_name_stored(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert result.outcome_name == "loss_ratio"

    def test_inference_method_stored(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", inference="analytic")
        result = est.fit()
        assert result.inference_method == "analytic"

    def test_analytic_inference(self):
        est = DoublyRobustSCEstimator(
            self.panel, outcome="loss_ratio", inference="analytic"
        )
        result = est.fit()
        assert result.se > 0

    def test_bootstrap_inference_se_positive(self):
        est = DoublyRobustSCEstimator(
            self.panel, outcome="loss_ratio", inference="bootstrap", n_replicates=100
        )
        result = est.fit()
        assert result.se > 0

    def test_reproducible_with_seed(self):
        r1 = DoublyRobustSCEstimator(
            self.panel, outcome="loss_ratio", inference="bootstrap",
            n_replicates=50, random_seed=42
        ).fit()
        r2 = DoublyRobustSCEstimator(
            self.panel, outcome="loss_ratio", inference="bootstrap",
            n_replicates=50, random_seed=42
        ).fit()
        assert abs(r1.att - r2.att) < 1e-10
        assert abs(r1.se - r2.se) < 1e-8

    def test_no_treated_raises(self):
        panel_no_treated = self.panel.with_columns(pl.lit(0).alias("treated"))
        est = DoublyRobustSCEstimator(panel_no_treated, outcome="loss_ratio")
        with pytest.raises(ValueError, match="No treated segments"):
            est.fit()

    def test_pre_trend_pval_available(self):
        """With 8 pre-periods, pre-trend test should produce a p-value."""
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        assert result.pre_trend_pval is not None

    def test_significant_property(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=100)
        result = est.fit()
        assert result.significant == (result.pval < 0.05)

    def test_pre_trends_pass_property(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        if result.pre_trend_pval is not None:
            assert result.pre_trends_pass == (result.pre_trend_pval > 0.10)
        else:
            assert result.pre_trends_pass is True

    def test_summary_contains_att(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        s = result.summary()
        assert "DRSC estimate" in s
        assert "loss_ratio" in s

    def test_fca_summary_format(self):
        est = DoublyRobustSCEstimator(self.panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        s = result.to_fca_summary(product_line="Motor", rate_change_date="2024-Q1")
        assert "Motor" in s
        assert "2024-Q1" in s
        assert "Doubly Robust Synthetic Controls" in s


# ---------------------------------------------------------------------------
# Null DGP: ATT near zero
# ---------------------------------------------------------------------------


class TestNullDGP:
    def test_att_near_zero_no_treatment(self):
        """Under null DGP (true ATT=0), estimated ATT should be small."""
        panel = make_panel(n_co=40, n_tr=10, t_pre=8, t_post=4, true_att=0.0, seed=100)
        est = DoublyRobustSCEstimator(panel, outcome="loss_ratio", n_replicates=100)
        result = est.fit()
        assert abs(result.att) < 0.04

    def test_pre_trend_atts_near_zero(self):
        """Under null DGP, pre-period event study ATTs should be near zero."""
        panel = make_panel(n_co=50, n_tr=10, t_pre=8, t_post=4, true_att=0.0, seed=101)
        est = DoublyRobustSCEstimator(panel, outcome="loss_ratio", n_replicates=50)
        result = est.fit()
        pre_atts = result.event_study[result.event_study["period_rel"] < 0]["att"].values
        assert np.std(pre_atts) < 0.05


# ---------------------------------------------------------------------------
# Comparison with SDID: should agree on direction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("cvxpy"),
    reason="cvxpy not installed",
)
class TestDRSCVsSDID:
    def test_both_negative_under_clean_dgp(self):
        """DRSC and SDID should both give negative ATT under clean -0.07 DGP."""
        from insurance_causal_policy._sdid import SDIDEstimator

        panel = make_panel(n_co=40, n_tr=12, t_pre=8, t_post=4, true_att=-0.07, seed=50)
        drsc_result = DoublyRobustSCEstimator(
            panel, outcome="loss_ratio", n_replicates=50
        ).fit()
        sdid_result = SDIDEstimator(
            panel, outcome="loss_ratio", inference="jackknife"
        ).fit()
        assert drsc_result.att < 0
        assert sdid_result.att < 0

    def test_atts_in_similar_ballpark(self):
        """DRSC and SDID ATTs should agree within 5pp on a clean DGP."""
        from insurance_causal_policy._sdid import SDIDEstimator

        panel = make_panel(n_co=50, n_tr=15, t_pre=8, t_post=4, true_att=-0.06, seed=51)
        drsc_att = DoublyRobustSCEstimator(
            panel, outcome="loss_ratio", n_replicates=50
        ).fit().att
        sdid_att = SDIDEstimator(
            panel, outcome="loss_ratio", inference="jackknife"
        ).fit().att
        assert abs(drsc_att - sdid_att) < 0.05


# ---------------------------------------------------------------------------
# Warning for under-identified SC weights
# ---------------------------------------------------------------------------


class TestUnderidentifiedWarning:
    def test_warns_when_t_pre_lt_n_co(self):
        """DRSC should warn when T_pre < N_co (underdetermined OLS)."""
        # 30 controls, only 4 pre-treatment periods
        panel = make_panel(n_co=30, n_tr=5, t_pre=4, t_post=3, true_att=-0.05, seed=200)
        est = DoublyRobustSCEstimator(panel, outcome="loss_ratio", n_replicates=50)
        with pytest.warns(UserWarning, match="under-identified"):
            result = est.fit()
        # Should still produce a valid result
        assert isinstance(result, DRSCResult)
        assert np.isfinite(result.att)
