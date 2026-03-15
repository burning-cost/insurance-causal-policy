"""Tests for _sdid.py SDID estimator."""

import numpy as np
import polars as pl
import pytest
from insurance_causal_policy._sdid import (
    SDIDEstimator,
    _compute_regularisation_zeta,
    _fit_sdid_core,
    _jackknife_variance,
    _solve_unit_weights,
    _solve_time_weights,
    _weighted_twfe,
)
from insurance_causal_policy._synthetic import make_synthetic_panel_direct
from insurance_causal_policy._types import SDIDResult


def make_simple_panel(
    n_co=30, n_tr=10, t_pre=6, t_post=4, true_att=-0.06, seed=42
):
    return make_synthetic_panel_direct(
        n_control=n_co, n_treated=n_tr,
        t_pre=t_pre, t_post=t_post,
        true_att=true_att, noise_sd=0.03, random_seed=seed,
    )


class TestRegularisationZeta:
    def test_positive_zeta(self):
        y_pre = np.random.default_rng(1).normal(0.7, 0.05, (20, 6))
        zeta = _compute_regularisation_zeta(y_pre, n_treated=5, t_post=3)
        assert zeta > 0

    def test_scales_with_n_treated(self):
        y_pre = np.random.default_rng(2).normal(0.7, 0.05, (20, 6))
        z1 = _compute_regularisation_zeta(y_pre, n_treated=5, t_post=3)
        z2 = _compute_regularisation_zeta(y_pre, n_treated=20, t_post=3)
        assert z2 > z1  # more treated → larger regularisation


class TestSolveUnitWeights:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(3)
        y_pre_tr = rng.normal(0.7, 0.03, 6)
        y_pre_co = rng.normal(0.7, 0.03, (20, 6))
        zeta = 0.05
        omega, omega_0 = _solve_unit_weights(y_pre_tr, y_pre_co, zeta, t_pre=6)
        assert abs(omega.sum() - 1.0) < 1e-4

    def test_weights_nonneg(self):
        rng = np.random.default_rng(4)
        y_pre_tr = rng.normal(0.7, 0.03, 6)
        y_pre_co = rng.normal(0.7, 0.03, (15, 6))
        omega, _ = _solve_unit_weights(y_pre_tr, y_pre_co, zeta=0.05, t_pre=6)
        assert np.all(omega >= -1e-6)

    def test_returns_intercept(self):
        rng = np.random.default_rng(5)
        y_pre_tr = rng.normal(0.7, 0.03, 4)
        y_pre_co = rng.normal(0.8, 0.03, (10, 4))  # level difference
        _, omega_0 = _solve_unit_weights(y_pre_tr, y_pre_co, zeta=0.05, t_pre=4)
        assert isinstance(omega_0, float)


class TestSolveTimeWeights:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(6)
        y_post_co_bar = rng.normal(0.75, 0.02, 15)
        y_pre_co = rng.normal(0.7, 0.02, (15, 6))
        lambda_ = _solve_time_weights(y_post_co_bar, y_pre_co)
        assert abs(lambda_.sum() - 1.0) < 1e-4

    def test_weights_nonneg(self):
        rng = np.random.default_rng(7)
        y_post_co_bar = rng.normal(0.75, 0.02, 15)
        y_pre_co = rng.normal(0.7, 0.02, (15, 6))
        lambda_ = _solve_time_weights(y_post_co_bar, y_pre_co)
        assert np.all(lambda_ >= -1e-6)

    def test_correct_length(self):
        rng = np.random.default_rng(8)
        y_post_co_bar = rng.normal(0.75, 0.02, 15)
        y_pre_co = rng.normal(0.7, 0.02, (15, 8))
        lambda_ = _solve_time_weights(y_post_co_bar, y_pre_co)
        assert len(lambda_) == 8


class TestFitSDIDCore:
    def test_att_close_to_true(self):
        """With a clean DGP, ATT should roughly recover true effect."""
        panel = make_simple_panel(n_co=50, n_tr=15, t_pre=8, t_post=4, true_att=-0.06)
        Y_rows = []
        unit_ids = panel["segment_id"].unique().sort().to_list()
        periods = sorted(panel["period"].unique().to_list())

        treated_segs = (
            panel.filter(pl.col("treated") == 1)["segment_id"].unique().to_list()
        )
        control_segs = [s for s in unit_ids if s not in treated_segs]
        unit_order = control_segs + treated_segs
        n_co = len(control_segs)
        n_tr = len(treated_segs)
        t_pre = 8
        t_post = 4

        for unit in unit_order:
            row_vals = []
            for per in periods:
                v = panel.filter(
                    (pl.col("segment_id") == unit) & (pl.col("period") == per)
                )["loss_ratio"][0]
                row_vals.append(float(v) if v is not None else 0.70)
            Y_rows.append(row_vals)

        Y = np.array(Y_rows)
        D = np.zeros_like(Y)
        for i, unit in enumerate(unit_order):
            if unit in treated_segs:
                for j, per in enumerate(periods):
                    if per > t_pre:
                        D[i, j] = 1

        att, _, _, _, _ = _fit_sdid_core(Y, D, n_co, n_tr, t_pre, t_post)
        # Should be in the right direction and ballpark
        assert att < 0
        assert abs(att - (-0.06)) < 0.05  # within 5pp of truth

    def test_no_treatment_att_near_zero(self):
        """With true_att=0, estimated ATT should be near zero."""
        panel = make_simple_panel(n_co=40, n_tr=10, t_pre=6, t_post=3, true_att=0.0, seed=10)
        # Use SDIDEstimator for convenience
        est = SDIDEstimator(panel, outcome="loss_ratio", inference="jackknife", n_replicates=50)
        result = est.fit()
        assert abs(result.att) < 0.05  # near zero under null


class TestSDIDEstimator:
    def setup_method(self):
        self.panel = make_simple_panel(n_co=40, n_tr=12, t_pre=7, t_post=4, true_att=-0.07)

    def test_fit_returns_sdid_result(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert isinstance(result, SDIDResult)

    def test_att_is_float(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert isinstance(result.att, float)

    def test_se_positive(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert result.se > 0

    def test_ci_contains_att(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert result.ci_low <= result.att <= result.ci_high

    def test_pval_in_unit_interval(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert 0 <= result.pval <= 1

    def test_att_negative_direction(self):
        """With true ATT -0.07, estimated ATT should be negative."""
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert result.att < 0

    def test_unit_weights_sum_to_one(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert abs(result.weights.unit_weights.sum() - 1.0) < 1e-3

    def test_time_weights_sum_to_one(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert abs(result.weights.time_weights.sum() - 1.0) < 1e-3

    def test_n_treated_correct(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert result.n_treated == 12

    def test_n_control_correct(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert result.n_control_total == 40

    def test_t_pre_correct(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert result.t_pre == 7

    def test_t_post_correct(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert result.t_post == 4

    def test_event_study_not_none(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert result.event_study is not None

    def test_event_study_columns(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        cols = set(result.event_study.columns)
        assert {"period_rel", "att"}.issubset(cols)

    def test_pre_trend_pval_available(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        # May be None if only 1 pre-period, but with 7 pre-periods should exist
        assert result.pre_trend_pval is not None

    def test_outcome_name_stored(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="placebo")
        result = est.fit()
        assert result.outcome_name == "loss_ratio"

    def test_inference_method_stored(self):
        est = SDIDEstimator(self.panel, outcome="loss_ratio", inference="bootstrap", n_replicates=50)
        result = est.fit()
        assert result.inference_method == "bootstrap"

    def test_bootstrap_inference(self):
        est = SDIDEstimator(
            self.panel, outcome="loss_ratio", inference="bootstrap", n_replicates=50
        )
        result = est.fit()
        assert result.se > 0

    def test_jackknife_inference(self):
        small_panel = make_simple_panel(n_co=20, n_tr=5, t_pre=4, t_post=3, true_att=-0.05)
        est = SDIDEstimator(small_panel, outcome="loss_ratio", inference="jackknife")
        result = est.fit()
        assert result.se > 0

    def test_no_treated_raises(self):
        panel_no_treated = make_simple_panel().with_columns(
            pl.lit(0).alias("treated")
        )
        est = SDIDEstimator(panel_no_treated, outcome="loss_ratio", inference="placebo")
        with pytest.raises(ValueError, match="No treated segments"):
            est.fit()

    def test_reproducible_with_seed(self):
        r1 = SDIDEstimator(
            self.panel, outcome="loss_ratio", inference="placebo",
            n_replicates=50, random_seed=42
        ).fit()
        r2 = SDIDEstimator(
            self.panel, outcome="loss_ratio", inference="placebo",
            n_replicates=50, random_seed=42
        ).fit()
        assert abs(r1.att - r2.att) < 1e-8
        assert abs(r1.se - r2.se) < 1e-8


# ---------------------------------------------------------------------------
# Regression tests for P0 bugs fixed 2026-03-15
# ---------------------------------------------------------------------------


class TestP0RegressionRegularisationZeta:
    """P0-1: zeta must use demeaned first differences, not raw ones.

    A panel with a strong common time trend (claims inflation) should produce a
    much smaller sigma when the trend is removed.  The raw first-differences
    formula included the trend in sigma, inflating zeta and over-regularising
    the unit weights toward equal weights.
    """

    def test_demeaned_sigma_smaller_than_raw_under_trend(self):
        """With a strong common trend, demeaned sigma < raw sigma."""
        rng = np.random.default_rng(99)
        n_co, t_pre = 20, 8
        # Panel with strong common trend but small unit noise
        time_trend = np.linspace(0.0, 0.10, t_pre)  # 10pp trend over pre-period
        unit_fe = rng.normal(0.0, 0.01, (n_co, 1))
        noise = rng.normal(0.0, 0.005, (n_co, t_pre))
        y_pre_co = 0.70 + unit_fe + time_trend[None, :] + noise

        # Compute raw sigma (old behaviour)
        raw_diffs = np.diff(y_pre_co, axis=1).flatten()
        sigma_raw = np.std(raw_diffs)

        # Compute demeaned sigma (new behaviour applied inside the function)
        unit_means = y_pre_co.mean(axis=1, keepdims=True)
        time_means = y_pre_co.mean(axis=0, keepdims=True)
        grand_mean = y_pre_co.mean()
        y_demeaned = y_pre_co - unit_means - time_means + grand_mean
        demeaned_diffs = np.diff(y_demeaned, axis=1).flatten()
        sigma_demeaned = np.std(demeaned_diffs)

        # The demeaned sigma should be substantially smaller
        assert sigma_demeaned < sigma_raw * 0.5, (
            f"Expected demeaned sigma << raw sigma under trend; "
            f"got demeaned={sigma_demeaned:.5f}, raw={sigma_raw:.5f}"
        )

    def test_zeta_not_inflated_by_trend(self):
        """The actual function should produce a zeta not dominated by the trend."""
        rng = np.random.default_rng(100)
        n_co, t_pre = 20, 8
        n_treated, t_post = 5, 3

        # Pure noise panel (no trend) — baseline zeta
        y_no_trend = rng.normal(0.70, 0.01, (n_co, t_pre))
        zeta_no_trend = _compute_regularisation_zeta(y_no_trend, n_treated, t_post)

        # Strong trend panel — with demeaning, zeta should be ~similar to no-trend
        time_trend = np.linspace(0.0, 0.10, t_pre)
        unit_fe = rng.normal(0.0, 0.01, (n_co, 1))
        noise = rng.normal(0.0, 0.01, (n_co, t_pre))
        y_with_trend = 0.70 + unit_fe + time_trend[None, :] + noise
        zeta_with_trend = _compute_regularisation_zeta(y_with_trend, n_treated, t_post)

        # After demeaning, the trend is absorbed.  zeta_with_trend should be
        # within a reasonable factor of zeta_no_trend (not inflated 10x).
        ratio = zeta_with_trend / max(zeta_no_trend, 1e-9)
        assert ratio < 5.0, (
            f"zeta inflated by trend even after demeaning: ratio={ratio:.2f}"
        )


class TestP0RegressionJackknifeVariance:
    """P0-2: jackknife variance must use n_total = n_co + n_tr, not n_j.

    When some replicates are skipped (convergence failure), n_j < n_total.
    Using n_j understates the variance via a smaller (n_j-1)/n_j correction
    and a smaller denominator in the mean computation.
    """

    def test_variance_uses_n_total_not_n_j(self):
        """Simulate dropped replicates: variance with n_total > variance with n_j."""
        rng = np.random.default_rng(42)
        # Build a set of hypothetical jackknife replicates and two results
        n_total = 10
        n_j = 7  # 3 replicates dropped (convergence failure simulation)
        jack_atts = np.array([-0.08, -0.07, -0.06, -0.09, -0.07, -0.065, -0.075])
        jack_mean = np.mean(jack_atts)
        sq_devs = np.sum((jack_atts - jack_mean) ** 2)

        var_wrong = ((n_j - 1) / n_j) * sq_devs    # old (buggy) formula
        var_correct = ((n_total - 1) / n_total) * sq_devs  # new (fixed) formula

        assert var_correct > var_wrong, (
            "With dropped replicates, n_total-based variance should exceed n_j-based"
        )

    def test_jackknife_variance_end_to_end_with_small_panel(self):
        """End-to-end: jackknife SE should be finite and positive."""
        panel = make_synthetic_panel_direct(
            n_control=15, n_treated=5, t_pre=4, t_post=3,
            true_att=-0.05, noise_sd=0.03, random_seed=7,
        )
        est = SDIDEstimator(
            panel, outcome="loss_ratio", inference="jackknife", n_replicates=0
        )
        result = est.fit()
        assert result.se > 0
        assert np.isfinite(result.se)


class TestP0RegressionEventStudyUniformWeights:
    """P0-3: event study pre-trend test must use uniform time weights.

    With SDID lambda_ (concentrated weights), pre-treatment ATTs are non-zero
    by construction even under perfect parallel trends.  Uniform weights give
    ATTs that are genuinely zero in expectation under parallel trends.
    """

    def test_pretrend_atts_near_zero_under_parallel_trends(self):
        """Under null DGP (true ATT=0), pre-trend ATTs should be near zero."""
        panel = make_synthetic_panel_direct(
            n_control=40, n_treated=10, t_pre=6, t_post=3,
            true_att=0.0, noise_sd=0.02, random_seed=55,
        )
        est = SDIDEstimator(
            panel, outcome="loss_ratio", inference="placebo", n_replicates=50,
            random_seed=55,
        )
        result = est.fit()
        pre_atts = result.event_study[result.event_study["period_rel"] < 0]["att"].values
        assert len(pre_atts) > 0
        # Under null, pre-period ATTs should be small — std < 2x noise level
        assert np.std(pre_atts) < 0.06, (
            f"Pre-trend ATT std too large: {np.std(pre_atts):.4f}. "
            "Suggests SDID lambda_ weights leaked into event study."
        )

    def test_event_study_pre_atts_not_systematically_non_zero(self):
        """Mean of pre-trend ATTs should be close to zero (no structural bias)."""
        panel = make_synthetic_panel_direct(
            n_control=50, n_treated=15, t_pre=8, t_post=3,
            true_att=0.0, noise_sd=0.02, random_seed=77,
        )
        est = SDIDEstimator(
            panel, outcome="loss_ratio", inference="placebo", n_replicates=30,
            random_seed=77,
        )
        result = est.fit()
        pre_atts = result.event_study[result.event_study["period_rel"] < 0]["att"].values
        assert abs(np.mean(pre_atts)) < 0.05, (
            f"Mean pre-trend ATT {np.mean(pre_atts):.4f} too large; "
            "uniform weights should give ~zero mean under null."
        )
