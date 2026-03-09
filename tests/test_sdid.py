"""Tests for _sdid.py SDID estimator."""

import numpy as np
import polars as pl
import pytest
from insurance_causal_policy._sdid import (
    SDIDEstimator,
    _compute_regularisation_zeta,
    _fit_sdid_core,
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
