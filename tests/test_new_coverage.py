"""New tests targeting untested/undertested code paths.

Coverage targets:
  - plot_synthetic_trajectory (zero tests previously)
  - _weighted_twfe (never directly tested)
  - _find_breakdown_point (internal function)
  - compute_sensitivity with near-zero pre_period_sd (warning path)
  - compute_sensitivity with smooth method and few pre-periods
  - SensitivityResult.summary() with empty m_values
  - SensitivityResult.to_dataframe() empty case
  - DRSCResult.summary() and to_fca_summary()
  - StaggeredResult.pre_trends_pass edge cases
  - FCAEvidencePack with StaggeredResult input
  - FCAEvidencePack with no sensitivity section wording
  - FCAEvidencePack _header variants (analyst, magnitude)
  - FCAEvidencePack partial panel_summary keys
  - SDIDResult.to_fca_summary with no date
  - DRSCResult.to_fca_summary with failing parallel trends
  - make_synthetic_motor_panel: positive ATT, minimal segments
  - make_synthetic_panel_direct: zero noise, very short panel
  - SDIDEstimator: invalid inference method
  - _weighted_twfe manual computation
  - _compute_regularisation_zeta: short T_pre warning
  - Integration: DRSC result into FCAEvidencePack
  - StaggeredResult CI and significance properties
  - pre_trend_summary with StaggeredResult
  - pre_trend_summary with no pre-periods in event_study
  - build_panel_from_pandas with pandas float columns
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_causal_policy._types import (
    DRSCResult,
    DRSCWeights,
    SDIDResult,
    SDIDWeights,
    SensitivityResult,
    StaggeredResult,
)
from insurance_causal_policy._synthetic import (
    make_synthetic_motor_panel,
    make_synthetic_panel_direct,
)

# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------

try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")
    _MPL_OK = True
except Exception:
    _MPL_OK = False

_skip_mpl = pytest.mark.skipif(not _MPL_OK, reason="matplotlib not functional")


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_sdid_weights(n_units: int = 3) -> SDIDWeights:
    return SDIDWeights(
        unit_weights=pd.Series(
            [1.0 / n_units] * n_units,
            index=[f"ctrl_{i}" for i in range(n_units)],
        ),
        time_weights=pd.Series([0.5, 0.5], index=[1, 2]),
        unit_intercept=0.003,
        regularisation_zeta=0.02,
    )


def _make_sdid_result(att=-0.05, se=0.02, pre_trend_pval=0.40):
    event_df = pd.DataFrame({
        "period_rel": [-3, -2, -1, 0, 1, 2, 3],
        "att": [0.002, -0.001, 0.003, att, att * 1.1, att * 0.9, att * 1.05],
        "se": [0.01] * 7,
        "ci_low": [np.nan] * 7,
        "ci_high": [np.nan] * 7,
    })
    return SDIDResult(
        att=att, se=se,
        ci_low=att - 1.96 * se, ci_high=att + 1.96 * se,
        pval=0.01,
        weights=_make_sdid_weights(),
        pre_trend_pval=pre_trend_pval,
        event_study=event_df,
        n_treated=10, n_control=5, n_control_total=30,
        t_pre=3, t_post=3,
        outcome_name="loss_ratio",
        inference_method="placebo",
        n_replicates=200,
    )


def _make_drsc_weights(n_co: int = 10) -> DRSCWeights:
    return DRSCWeights(
        sc_weights=pd.Series(np.random.default_rng(7).normal(0, 0.1, n_co) + 1.0 / n_co,
                              index=[f"co_{i}" for i in range(n_co)]),
        propensity_ratios=pd.Series(np.full(n_co, 8.0), index=[f"co_{i}" for i in range(n_co)]),
        m_delta=0.005,
    )


def _make_drsc_result(att=-0.06, se=0.02, pre_trend_pval=0.35):
    n_co = 10
    n_tr = 8
    n_total = n_co + n_tr
    phi = np.full(n_total, att)
    event_df = pd.DataFrame({
        "period_rel": [-4, -3, -2, -1, 0, 1, 2, 3],
        "att": [0.001, -0.002, 0.001, 0.002, att, att * 1.1, att * 0.9, att],
        "se": [0.01] * 8,
        "ci_low": [np.nan] * 8,
        "ci_high": [np.nan] * 8,
    })
    return DRSCResult(
        att=att, se=se,
        ci_low=att - 1.96 * se, ci_high=att + 1.96 * se,
        pval=0.005,
        weights=_make_drsc_weights(n_co),
        phi=phi,
        pre_trend_pval=pre_trend_pval,
        event_study=event_df,
        n_treated=n_tr,
        n_control=n_co,
        t_pre=4, t_post=4,
        outcome_name="loss_ratio",
        inference_method="bootstrap",
        n_replicates=200,
    )


def _make_staggered_result(att=-0.06, pre_trend_pval=0.30):
    att_gt = pd.DataFrame({
        "cohort": [5, 5, 6, 6, 7, 7],
        "period": [5, 6, 6, 7, 7, 8],
        "att": [-0.05, -0.07, -0.04, -0.06, -0.03, -0.07],
        "se": [0.02] * 6,
        "ci_low": [-0.09, -0.11, -0.08, -0.10, -0.07, -0.11],
        "ci_high": [-0.01, -0.03, 0.00, -0.02, 0.01, -0.03],
    })
    event_df = pd.DataFrame({
        "period_rel": [-3, -2, -1, 0, 1, 2],
        "att": [0.002, -0.001, 0.003, att, att * 1.1, att * 0.9],
        "se": [0.01] * 6,
        "ci_low": [np.nan] * 6,
        "ci_high": [np.nan] * 6,
    })
    return StaggeredResult(
        att_overall=att,
        se_overall=0.018,
        ci_low_overall=att - 1.96 * 0.018,
        ci_high_overall=att + 1.96 * 0.018,
        pval_overall=0.001,
        att_gt=att_gt,
        event_study=event_df,
        pre_trend_pval=pre_trend_pval,
        n_cohorts=3,
        outcome_name="loss_ratio",
        control_group="notyettreated",
    )


# ---------------------------------------------------------------------------
# SensitivityResult: edge cases not covered
# ---------------------------------------------------------------------------


class TestSensitivityResultEmpty:
    def test_to_dataframe_empty_m_values(self):
        s = SensitivityResult(
            m_values=[], att_lower=[], att_upper=[],
            m_breakdown=0.0, pre_period_sd=0.01,
        )
        df = s.to_dataframe()
        assert list(df.columns) == ["m", "att_lower", "att_upper"]
        assert len(df) == 0

    def test_summary_empty_m_values(self):
        s = SensitivityResult(
            m_values=[], att_lower=[], att_upper=[],
            m_breakdown=0.0, pre_period_sd=0.01,
        )
        summary = s.summary()
        assert "No sensitivity values computed" in summary

    def test_to_dataframe_single_value(self):
        s = SensitivityResult(
            m_values=[0.0], att_lower=[-0.08], att_upper=[-0.02],
            m_breakdown=1.5, pre_period_sd=0.01,
        )
        df = s.to_dataframe()
        assert len(df) == 1
        assert df["m"].iloc[0] == 0.0

    def test_summary_reports_pre_period_sd(self):
        s = SensitivityResult(
            m_values=[0.0, 1.0], att_lower=[-0.08, -0.10],
            att_upper=[-0.02, 0.02], m_breakdown=0.8, pre_period_sd=0.0042,
        )
        txt = s.summary()
        assert "0.0042" in txt

    def test_summary_breakdown_within_range(self):
        """m_breakdown < max(m_values) => 'breaks down at M' text."""
        s = SensitivityResult(
            m_values=[0.0, 1.0, 2.0], att_lower=[-0.08, -0.10, -0.12],
            att_upper=[-0.02, 0.03, 0.08], m_breakdown=0.8, pre_period_sd=0.01,
        )
        txt = s.summary()
        assert "breaks down" in txt.lower() or "0.80" in txt


# ---------------------------------------------------------------------------
# DRSCResult: summary and fca_summary not tested elsewhere
# ---------------------------------------------------------------------------


class TestDRSCResultSummary:
    def test_summary_returns_string(self):
        r = _make_drsc_result()
        s = r.summary()
        assert isinstance(s, str)
        assert len(s) > 50

    def test_summary_contains_drsc(self):
        r = _make_drsc_result()
        assert "DRSC estimate" in r.summary()

    def test_summary_contains_outcome_name(self):
        r = _make_drsc_result()
        assert "loss_ratio" in r.summary()

    def test_summary_decrease_direction(self):
        r = _make_drsc_result(att=-0.06)
        assert "decrease" in r.summary()

    def test_summary_increase_direction(self):
        r = _make_drsc_result(att=0.04)
        assert "increase" in r.summary()

    def test_summary_pre_trend_pass(self):
        r = _make_drsc_result(pre_trend_pval=0.50)
        assert "PASS" in r.summary()

    def test_summary_pre_trend_warning(self):
        r = _make_drsc_result(pre_trend_pval=0.03)
        assert "WARNING" in r.summary()

    def test_fca_summary_returns_string(self):
        r = _make_drsc_result()
        s = r.to_fca_summary(product_line="Home", rate_change_date="2024-Q2")
        assert isinstance(s, str)
        assert len(s) > 100

    def test_fca_summary_product_line(self):
        r = _make_drsc_result()
        assert "Home" in r.to_fca_summary(product_line="Home")

    def test_fca_summary_date_str(self):
        r = _make_drsc_result()
        assert "2024-Q3" in r.to_fca_summary(rate_change_date="2024-Q3")

    def test_fca_summary_no_date(self):
        r = _make_drsc_result()
        s = r.to_fca_summary()  # no date — no parenthetical
        assert isinstance(s, str)
        assert "Rate Change Evaluation" in s

    def test_fca_summary_failing_parallel_trends(self):
        r = _make_drsc_result(pre_trend_pval=0.02)
        s = r.to_fca_summary()
        assert "WARNING" in s

    def test_fca_summary_contains_att(self):
        r = _make_drsc_result(att=-0.06)
        s = r.to_fca_summary()
        assert "0.06" in s

    def test_fca_summary_doubly_robust_text(self):
        r = _make_drsc_result()
        s = r.to_fca_summary()
        assert "Doubly Robust" in s

    def test_drsc_significant_property(self):
        r = _make_drsc_result()
        r2 = DRSCResult(**{**r.__dict__, "pval": 0.04})
        assert r2.significant is True

    def test_drsc_not_significant(self):
        r = _make_drsc_result()
        r2 = DRSCResult(**{**r.__dict__, "pval": 0.10})
        assert r2.significant is False

    def test_pre_trends_pass_none(self):
        r = _make_drsc_result()
        r2 = DRSCResult(**{**r.__dict__, "pre_trend_pval": None})
        assert r2.pre_trends_pass is True

    def test_pre_trends_pass_boundary(self):
        """Exactly p=0.10 should pass (> 0.10 is the condition)."""
        r = _make_drsc_result()
        r2 = DRSCResult(**{**r.__dict__, "pre_trend_pval": 0.10})
        assert r2.pre_trends_pass is False  # 0.10 is not > 0.10

    def test_pre_trends_pass_above_boundary(self):
        r = _make_drsc_result()
        r2 = DRSCResult(**{**r.__dict__, "pre_trend_pval": 0.101})
        assert r2.pre_trends_pass is True


# ---------------------------------------------------------------------------
# StaggeredResult: properties and edge cases
# ---------------------------------------------------------------------------


class TestStaggeredResultProperties:
    def test_pre_trends_pass_high_pval(self):
        r = _make_staggered_result(pre_trend_pval=0.50)
        assert r.pre_trends_pass is True

    def test_pre_trends_pass_low_pval(self):
        r = _make_staggered_result(pre_trend_pval=0.05)
        assert r.pre_trends_pass is False

    def test_pre_trends_pass_none(self):
        r = _make_staggered_result()
        r2 = StaggeredResult(**{**r.__dict__, "pre_trend_pval": None})
        assert r2.pre_trends_pass is True

    def test_att_gt_has_positive_length(self):
        r = _make_staggered_result()
        assert len(r.att_gt) > 0

    def test_event_study_has_period_rel(self):
        r = _make_staggered_result()
        assert "period_rel" in r.event_study.columns

    def test_ci_ordering(self):
        r = _make_staggered_result(att=-0.06)
        assert r.ci_low_overall <= r.att_overall <= r.ci_high_overall

    def test_n_cohorts_stored(self):
        r = _make_staggered_result()
        assert r.n_cohorts == 3

    def test_outcome_name_stored(self):
        r = _make_staggered_result()
        assert r.outcome_name == "loss_ratio"

    def test_control_group_stored(self):
        r = _make_staggered_result()
        assert r.control_group == "notyettreated"


# ---------------------------------------------------------------------------
# SDIDResult: to_fca_summary variants
# ---------------------------------------------------------------------------


class TestSDIDResultFCASummaryVariants:
    def test_to_fca_summary_no_date(self):
        r = _make_sdid_result()
        s = r.to_fca_summary(product_line="Motor")
        # No rate_change_date → no parenthetical in header
        assert "Motor" in s
        assert "()" not in s  # no empty parenthetical

    def test_to_fca_summary_warning_on_low_pval(self):
        r = _make_sdid_result(pre_trend_pval=0.03)
        s = r.to_fca_summary()
        assert "WARNING" in s

    def test_to_fca_summary_pass_on_high_pval(self):
        r = _make_sdid_result(pre_trend_pval=0.45)
        s = r.to_fca_summary()
        assert "PASS" in s

    def test_to_fca_summary_not_significant(self):
        r = _make_sdid_result()
        r2 = SDIDResult(**{**r.__dict__, "pval": 0.20})
        s = r2.to_fca_summary()
        assert "not statistically significant" in s

    def test_to_fca_summary_contains_estimator_name(self):
        r = _make_sdid_result()
        s = r.to_fca_summary()
        assert "Synthetic Difference-in-Differences" in s

    def test_summary_positive_att_says_increase(self):
        r = _make_sdid_result(att=0.03)
        assert "increase" in r.summary()


# ---------------------------------------------------------------------------
# _weighted_twfe: direct test
# ---------------------------------------------------------------------------


class TestWeightedTWFE:
    """Direct unit tests for _weighted_twfe."""

    def _make_Y_D(self, n_co=10, n_tr=5, t_pre=4, t_post=3, true_att=-0.06, seed=42):
        rng = np.random.default_rng(seed)
        N = n_co + n_tr
        T = t_pre + t_post
        alpha = rng.normal(0, 0.05, N)
        beta = np.arange(T) * 0.003
        Y = np.zeros((N, T))
        D = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                is_treated = i >= n_co and t >= t_pre
                Y[i, t] = 0.70 + alpha[i] + beta[t] + (true_att if is_treated else 0.0) + rng.normal(0, 0.01)
                D[i, t] = 1 if is_treated else 0
        return Y, D

    def test_returns_float(self):
        from insurance_causal_policy._sdid import _weighted_twfe
        Y, D = self._make_Y_D()
        n_co, n_tr, t_pre, t_post = 10, 5, 4, 3
        omega = np.ones(n_co) / n_co
        lambda_ = np.ones(t_pre) / t_pre
        tau = _weighted_twfe(Y, D, omega, lambda_, n_co, n_tr, t_pre, t_post)
        assert isinstance(tau, float)

    def test_negative_direction(self):
        """With true ATT < 0, weighted TWFE should give negative tau."""
        from insurance_causal_policy._sdid import _weighted_twfe
        Y, D = self._make_Y_D(n_co=20, n_tr=8, t_pre=6, t_post=4, true_att=-0.08)
        n_co, n_tr, t_pre, t_post = 20, 8, 6, 4
        omega = np.ones(n_co) / n_co
        lambda_ = np.ones(t_pre) / t_pre
        tau = _weighted_twfe(Y, D, omega, lambda_, n_co, n_tr, t_pre, t_post)
        assert tau < 0

    def test_near_zero_under_null(self):
        """With true ATT = 0, tau should be close to zero."""
        from insurance_causal_policy._sdid import _weighted_twfe
        Y, D = self._make_Y_D(n_co=30, n_tr=10, t_pre=6, t_post=4, true_att=0.0, seed=99)
        n_co, n_tr, t_pre, t_post = 30, 10, 6, 4
        omega = np.ones(n_co) / n_co
        lambda_ = np.ones(t_pre) / t_pre
        tau = _weighted_twfe(Y, D, omega, lambda_, n_co, n_tr, t_pre, t_post)
        assert abs(tau) < 0.05

    def test_concentrated_omega_changes_estimate(self):
        """Different omega vectors produce different tau — not all-equal result."""
        from insurance_causal_policy._sdid import _weighted_twfe
        Y, D = self._make_Y_D(n_co=20, n_tr=8, t_pre=6, t_post=3, true_att=-0.06, seed=10)
        n_co, n_tr, t_pre, t_post = 20, 8, 6, 3
        lambda_ = np.ones(t_pre) / t_pre

        omega_equal = np.ones(n_co) / n_co
        omega_concentrated = np.zeros(n_co)
        omega_concentrated[0] = 1.0

        tau_equal = _weighted_twfe(Y, D, omega_equal, lambda_, n_co, n_tr, t_pre, t_post)
        tau_conc = _weighted_twfe(Y, D, omega_concentrated, lambda_, n_co, n_tr, t_pre, t_post)
        # They won't be the same (different controls weighted differently)
        assert tau_equal != tau_conc

    def test_finite_output(self):
        """Output should be a finite float."""
        from insurance_causal_policy._sdid import _weighted_twfe
        Y, D = self._make_Y_D()
        n_co, n_tr, t_pre, t_post = 10, 5, 4, 3
        omega = np.ones(n_co) / n_co
        lambda_ = np.ones(t_pre) / t_pre
        tau = _weighted_twfe(Y, D, omega, lambda_, n_co, n_tr, t_pre, t_post)
        assert math.isfinite(tau)


# ---------------------------------------------------------------------------
# _compute_regularisation_zeta: short T_pre warning
# ---------------------------------------------------------------------------


class TestZetaShortTPreWarning:
    def test_warns_when_t_pre_lt_3(self):
        from insurance_causal_policy._sdid import _compute_regularisation_zeta
        rng = np.random.default_rng(1)
        y_pre = rng.normal(0.70, 0.05, (15, 2))  # only 2 pre-periods
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _compute_regularisation_zeta(y_pre, n_treated=5, t_post=3)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1
        assert "T_pre" in str(user_warnings[0].message) or "pre-treatment" in str(user_warnings[0].message).lower()

    def test_no_warning_when_t_pre_ge_3(self):
        from insurance_causal_policy._sdid import _compute_regularisation_zeta
        rng = np.random.default_rng(2)
        y_pre = rng.normal(0.70, 0.05, (15, 4))  # 4 pre-periods
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _compute_regularisation_zeta(y_pre, n_treated=5, t_post=3)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_zeta_still_positive_with_short_pre(self):
        from insurance_causal_policy._sdid import _compute_regularisation_zeta
        rng = np.random.default_rng(3)
        y_pre = rng.normal(0.70, 0.05, (10, 2))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zeta = _compute_regularisation_zeta(y_pre, n_treated=3, t_post=2)
        assert zeta > 0

    def test_minimum_zeta_floor(self):
        """Even with zero variance, zeta should be at least 1e-6."""
        from insurance_causal_policy._sdid import _compute_regularisation_zeta
        # All-identical panel: no variance in first differences
        y_pre = np.ones((10, 4)) * 0.70
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zeta = _compute_regularisation_zeta(y_pre, n_treated=5, t_post=3)
        assert zeta >= 1e-6


# ---------------------------------------------------------------------------
# compute_sensitivity: edge cases
# ---------------------------------------------------------------------------


class TestComputeSensitivityEdgeCases:
    def test_near_zero_pre_sd_warns(self):
        """Near-zero pre-period SD should trigger a UserWarning."""
        from insurance_causal_policy._sensitivity import compute_sensitivity
        # Build a result with perfectly flat pre-treatment estimates
        event_df = pd.DataFrame({
            "period_rel": [-2, -1, 0, 1],
            "att": [0.0, 0.0, -0.05, -0.05],  # zero pre-period variation
            "se": [0.01] * 4,
            "ci_low": [np.nan] * 4,
            "ci_high": [np.nan] * 4,
        })
        r = SDIDResult(
            att=-0.05, se=0.015,
            ci_low=-0.05 - 1.96 * 0.015, ci_high=-0.05 + 1.96 * 0.015,
            pval=0.001, weights=_make_sdid_weights(),
            pre_trend_pval=0.80, event_study=event_df,
            n_treated=10, n_control=5, n_control_total=30,
            t_pre=2, t_post=2, outcome_name="loss_ratio",
            inference_method="placebo", n_replicates=200,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sens = compute_sensitivity(r, m_values=[0.0, 1.0])
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        # Should warn about near-zero SD
        assert len(user_warnings) >= 1

    def test_smooth_method_with_two_pre_periods(self):
        """Smooth method with <3 pre-periods falls back to pre_period_sd."""
        from insurance_causal_policy._sensitivity import compute_sensitivity
        event_df = pd.DataFrame({
            "period_rel": [-2, -1, 0, 1],
            "att": [0.003, -0.002, -0.05, -0.05],
            "se": [0.01] * 4,
            "ci_low": [np.nan] * 4,
            "ci_high": [np.nan] * 4,
        })
        r = SDIDResult(
            att=-0.05, se=0.015,
            ci_low=-0.079, ci_high=-0.021,
            pval=0.001, weights=_make_sdid_weights(),
            pre_trend_pval=0.80, event_study=event_df,
            n_treated=10, n_control=5, n_control_total=30,
            t_pre=2, t_post=2, outcome_name="loss_ratio",
            inference_method="placebo", n_replicates=200,
        )
        # Should run without raising even though <3 pre-periods for smooth
        from insurance_causal_policy._types import SensitivityResult
        sens = compute_sensitivity(r, m_values=[0.0, 0.5, 1.0], method="smooth")
        assert isinstance(sens, SensitivityResult)

    def test_no_pre_treatment_periods_raises(self):
        """An event study with only post-treatment periods should raise."""
        from insurance_causal_policy._sensitivity import compute_sensitivity
        event_df = pd.DataFrame({
            "period_rel": [0, 1, 2],
            "att": [-0.05, -0.05, -0.06],
            "se": [0.01] * 3,
            "ci_low": [np.nan] * 3,
            "ci_high": [np.nan] * 3,
        })
        r = SDIDResult(
            att=-0.05, se=0.015,
            ci_low=-0.079, ci_high=-0.021,
            pval=0.001, weights=_make_sdid_weights(),
            pre_trend_pval=None, event_study=event_df,
            n_treated=10, n_control=5, n_control_total=30,
            t_pre=0, t_post=3, outcome_name="loss_ratio",
            inference_method="placebo", n_replicates=200,
        )
        with pytest.raises(ValueError, match="No pre-treatment periods"):
            compute_sensitivity(r, m_values=[0.0, 1.0])

    def test_single_pre_period_uses_abs_value(self):
        """With single pre-treatment period, SD is set to abs of that value."""
        from insurance_causal_policy._sensitivity import compute_sensitivity
        event_df = pd.DataFrame({
            "period_rel": [-1, 0, 1],
            "att": [0.008, -0.05, -0.05],
            "se": [0.01] * 3,
            "ci_low": [np.nan] * 3,
            "ci_high": [np.nan] * 3,
        })
        r = SDIDResult(
            att=-0.05, se=0.015,
            ci_low=-0.079, ci_high=-0.021,
            pval=0.001, weights=_make_sdid_weights(),
            pre_trend_pval=0.70, event_study=event_df,
            n_treated=10, n_control=5, n_control_total=30,
            t_pre=1, t_post=2, outcome_name="loss_ratio",
            inference_method="placebo", n_replicates=200,
        )
        from insurance_causal_policy._types import SensitivityResult
        sens = compute_sensitivity(r, m_values=[0.0, 1.0])
        assert isinstance(sens, SensitivityResult)
        # With single pre-period of 0.008, SD = abs(0.008) = 0.008
        assert abs(sens.pre_period_sd - 0.008) < 1e-6

    def test_breakdown_correctly_zero_when_already_insignificant(self):
        """When ATT CI already includes zero, breakdown should be 0."""
        from insurance_causal_policy._sensitivity import compute_sensitivity
        event_df = pd.DataFrame({
            "period_rel": [-2, -1, 0, 1],
            "att": [0.002, -0.001, -0.01, -0.01],  # ATT barely negative, wide CI
            "se": [0.01] * 4,
            "ci_low": [np.nan] * 4,
            "ci_high": [np.nan] * 4,
        })
        r = SDIDResult(
            att=-0.01, se=0.02,  # CI includes zero: -0.01 +/- 1.96*0.02
            ci_low=-0.01 - 1.96 * 0.02, ci_high=-0.01 + 1.96 * 0.02,
            pval=0.40, weights=_make_sdid_weights(),
            pre_trend_pval=0.60, event_study=event_df,
            n_treated=10, n_control=5, n_control_total=30,
            t_pre=2, t_post=2, outcome_name="loss_ratio",
            inference_method="placebo", n_replicates=200,
        )
        sens = compute_sensitivity(r, m_values=[0.0, 1.0, 2.0])
        assert sens.m_breakdown == 0.0


# ---------------------------------------------------------------------------
# FCAEvidencePack: StaggeredResult input and other variants
# ---------------------------------------------------------------------------


class TestFCAEvidencePackStaggered:
    def setup_method(self):
        from insurance_causal_policy._evidence import FCAEvidencePack
        self.FCAEvidencePack = FCAEvidencePack
        self.result = _make_staggered_result()

    def test_to_markdown_with_staggered_result(self):
        pack = self.FCAEvidencePack(
            result=self.result,
            product_line="Motor",
            rate_change_date="2023-Q2",
        )
        md = pack.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 100

    def test_markdown_contains_product_line(self):
        pack = self.FCAEvidencePack(result=self.result, product_line="Home")
        assert "Home" in pack.to_markdown()

    def test_markdown_staggered_estimator_name(self):
        pack = self.FCAEvidencePack(result=self.result, product_line="Motor")
        md = pack.to_markdown()
        # Should mention CS21 / Callaway / staggered
        assert "Callaway" in md or "Staggered" in md or "staggered" in md.lower()

    def test_to_dict_with_staggered(self):
        pack = self.FCAEvidencePack(result=self.result)
        d = pack.to_dict()
        assert "metadata" in d
        assert "estimation" in d
        # att_overall from StaggeredResult
        assert d["estimation"]["att"] is not None

    def test_to_json_valid_with_staggered(self):
        import json
        pack = self.FCAEvidencePack(result=self.result)
        s = pack.to_json()
        parsed = json.loads(s)
        assert "estimation" in parsed

    def test_parallel_trends_warning_when_failing(self):
        bad_result = _make_staggered_result(pre_trend_pval=0.02)
        pack = self.FCAEvidencePack(result=bad_result)
        md = pack.to_markdown()
        assert "WARNING" in md

    def test_no_sensitivity_section_text(self):
        """When no sensitivity supplied, markdown should mention running it."""
        pack = self.FCAEvidencePack(result=self.result, sensitivity=None)
        md = pack.to_markdown()
        assert "Sensitivity" in md
        assert "compute_sensitivity" in md

    def test_with_analyst_name(self):
        pack = self.FCAEvidencePack(
            result=self.result,
            analyst="Pricing Team Alpha",
        )
        md = pack.to_markdown()
        assert "Pricing Team Alpha" in md

    def test_with_rate_change_magnitude(self):
        pack = self.FCAEvidencePack(
            result=self.result,
            rate_change_magnitude="+7.5% technical premium",
        )
        md = pack.to_markdown()
        assert "7.5%" in md

    def test_additional_notes_included(self):
        pack = self.FCAEvidencePack(
            result=self.result,
            additional_notes="COVID lockdown period excluded from analysis",
        )
        md = pack.to_markdown()
        assert "COVID" in md

    def test_partial_panel_summary(self):
        """Panel summary with only some keys should not crash."""
        from insurance_causal_policy._evidence import FCAEvidencePack
        partial_ps = {"n_segments": 50, "outcome": "frequency"}
        pack = FCAEvidencePack(result=self.result, panel_summary=partial_ps)
        md = pack.to_markdown()
        assert isinstance(md, str)


class TestFCAEvidencePackHeaderVariants:
    """Test _header section text with various parameter combinations."""

    def setup_method(self):
        from insurance_causal_policy._evidence import FCAEvidencePack
        self.FCAEvidencePack = FCAEvidencePack

    def test_header_no_date_no_magnitude(self):
        pack = self.FCAEvidencePack(result=_make_sdid_result())
        md = pack.to_markdown()
        assert "Not specified" in md  # both rate_change and analyst should say this

    def test_header_with_all_fields(self):
        pack = self.FCAEvidencePack(
            result=_make_sdid_result(),
            product_line="Commercial",
            rate_change_date="2024-Q1",
            rate_change_magnitude="+5%",
            analyst="Actuarial Team",
        )
        md = pack.to_markdown()
        assert "Commercial" in md
        assert "2024-Q1" in md
        assert "+5%" in md
        assert "Actuarial Team" in md

    def test_footer_contains_package_name(self):
        pack = self.FCAEvidencePack(result=_make_sdid_result())
        md = pack.to_markdown()
        assert "insurance-causal-policy" in md

    def test_footer_contains_fca_reference(self):
        pack = self.FCAEvidencePack(result=_make_sdid_result())
        md = pack.to_markdown()
        assert "EP25/2" in md or "FCA" in md

    def test_caveats_section_present(self):
        pack = self.FCAEvidencePack(result=_make_sdid_result())
        md = pack.to_markdown()
        assert "Caveat" in md or "caveat" in md.lower() or "Limitation" in md

    def test_ibnr_mention_in_caveats(self):
        pack = self.FCAEvidencePack(result=_make_sdid_result())
        md = pack.to_markdown()
        assert "IBNR" in md


# ---------------------------------------------------------------------------
# Integration: DRSC result into FCAEvidencePack
# ---------------------------------------------------------------------------


class TestDRSCFCAPack:
    """FCAEvidencePack should handle DRSCResult as input without crashing."""

    def test_markdown_with_drsc_result(self):
        from insurance_causal_policy._evidence import FCAEvidencePack
        r = _make_drsc_result()
        pack = FCAEvidencePack(
            result=r,
            product_line="Motor",
            rate_change_date="2024-Q3",
        )
        md = pack.to_markdown()
        assert "Motor" in md
        assert isinstance(md, str)
        assert len(md) > 100

    def test_dict_with_drsc_result(self):
        from insurance_causal_policy._evidence import FCAEvidencePack
        r = _make_drsc_result(att=-0.06)
        pack = FCAEvidencePack(result=r)
        d = pack.to_dict()
        assert abs(d["estimation"]["att"] - (-0.06)) < 1e-6

    def test_json_with_drsc_result(self):
        import json
        from insurance_causal_policy._evidence import FCAEvidencePack
        r = _make_drsc_result()
        pack = FCAEvidencePack(result=r)
        parsed = json.loads(pack.to_json())
        assert "att" in parsed["estimation"]


# ---------------------------------------------------------------------------
# plot_synthetic_trajectory: not tested anywhere
# ---------------------------------------------------------------------------


@_skip_mpl
class TestPlotSyntheticTrajectory:
    def _make_panel_data(self, n_co=15, n_tr=5, t_pre=4, t_post=3, true_att=-0.06):
        rng = np.random.default_rng(42)
        N = n_co + n_tr
        T = t_pre + t_post
        Y = rng.normal(0.70, 0.03, (N, T))
        for i in range(n_co, N):
            Y[i, t_pre:] += true_att
        return Y, t_pre, t_post

    def test_returns_figure(self):
        from insurance_causal_policy._event_study import plot_synthetic_trajectory
        Y, t_pre, t_post = self._make_panel_data()
        n_co, n_tr = 15, 5
        period_ids = list(range(1, t_pre + t_post + 1))
        result = _make_sdid_result()
        # Adjust unit_weights to match n_co
        weights = SDIDWeights(
            unit_weights=pd.Series(np.ones(n_co) / n_co, index=[f"c{i}" for i in range(n_co)]),
            time_weights=pd.Series([0.5, 0.5, 0.0, 0.0], index=list(range(t_pre))),
            unit_intercept=0.0,
            regularisation_zeta=0.01,
        )
        r = SDIDResult(**{**result.__dict__, "weights": weights})
        fig = plot_synthetic_trajectory(
            r, Y, n_co, n_tr, t_pre, t_post, period_ids,
            outcome_name="loss_ratio",
        )
        assert fig is not None
        assert hasattr(fig, "savefig")

    def test_custom_title(self):
        from insurance_causal_policy._event_study import plot_synthetic_trajectory
        Y, t_pre, t_post = self._make_panel_data()
        n_co, n_tr = 15, 5
        period_ids = list(range(1, t_pre + t_post + 1))
        result = _make_sdid_result()
        weights = SDIDWeights(
            unit_weights=pd.Series(np.ones(n_co) / n_co, index=[f"c{i}" for i in range(n_co)]),
            time_weights=pd.Series([0.5, 0.5, 0.0, 0.0], index=list(range(t_pre))),
            unit_intercept=0.0,
            regularisation_zeta=0.01,
        )
        r = SDIDResult(**{**result.__dict__, "weights": weights})
        fig = plot_synthetic_trajectory(
            r, Y, n_co, n_tr, t_pre, t_post, period_ids,
            title="Custom Trajectory Title",
        )
        assert fig is not None

    def test_with_existing_ax(self):
        """Passing an existing Axes should use it, not create a new figure."""
        from insurance_causal_policy._event_study import plot_synthetic_trajectory
        import matplotlib.pyplot as plt
        fig0, ax0 = plt.subplots()
        Y, t_pre, t_post = self._make_panel_data()
        n_co, n_tr = 15, 5
        period_ids = list(range(1, t_pre + t_post + 1))
        result = _make_sdid_result()
        weights = SDIDWeights(
            unit_weights=pd.Series(np.ones(n_co) / n_co, index=[f"c{i}" for i in range(n_co)]),
            time_weights=pd.Series([0.5, 0.5, 0.0, 0.0], index=list(range(t_pre))),
            unit_intercept=0.0,
            regularisation_zeta=0.01,
        )
        r = SDIDResult(**{**result.__dict__, "weights": weights})
        fig = plot_synthetic_trajectory(
            r, Y, n_co, n_tr, t_pre, t_post, period_ids, ax=ax0
        )
        assert fig is fig0  # same figure returned


# ---------------------------------------------------------------------------
# pre_trend_summary with StaggeredResult and edge cases
# ---------------------------------------------------------------------------


class TestPreTrendSummaryVariants:
    def test_works_with_staggered_result(self):
        from insurance_causal_policy._event_study import pre_trend_summary
        r = _make_staggered_result()
        s = pre_trend_summary(r)
        assert isinstance(s, dict)
        assert "pval" in s
        assert "pass" in s

    def test_pre_atts_from_staggered(self):
        from insurance_causal_policy._event_study import pre_trend_summary
        r = _make_staggered_result()
        s = pre_trend_summary(r)
        assert isinstance(s["pre_atts"], list)
        assert len(s["pre_atts"]) == 3  # 3 pre-treatment periods in the helper

    def test_interpretation_mentions_ogden_when_failing(self):
        """Failing pre-trends should mention known insurance shocks."""
        from insurance_causal_policy._event_study import pre_trend_summary
        r = _make_sdid_result(pre_trend_pval=0.03)
        s = pre_trend_summary(r)
        assert not s["pass"]
        # The interpretation should mention known insurance disruptions
        interp = s["interpretation"]
        assert "Ogden" in interp or "COVID" in interp or "GIPP" in interp

    def test_interpretation_mentions_low_power_when_passing(self):
        from insurance_causal_policy._event_study import pre_trend_summary
        r = _make_sdid_result(pre_trend_pval=0.50)
        s = pre_trend_summary(r)
        assert s["pass"]
        assert "power" in s["interpretation"].lower() or "caution" in s["interpretation"].lower()

    def test_not_available_interpretation_when_no_event_study(self):
        from insurance_causal_policy._event_study import pre_trend_summary
        r = _make_sdid_result()
        r2 = SDIDResult(**{**r.__dict__, "event_study": None})
        s = pre_trend_summary(r2)
        assert "not available" in s["interpretation"].lower()

    def test_insufficient_periods_interpretation(self):
        """With pval=None but event_study present, should say 'insufficient'."""
        from insurance_causal_policy._event_study import pre_trend_summary
        r = _make_sdid_result(pre_trend_pval=None)
        s = pre_trend_summary(r)
        # pval is None but event_study is present
        assert s["pval"] is None
        assert "insufficient" in s["interpretation"].lower() or "not available" in s["interpretation"].lower()


# ---------------------------------------------------------------------------
# make_synthetic_motor_panel: additional variants
# ---------------------------------------------------------------------------


class TestSyntheticMotorPanelVariants:
    def test_positive_att(self):
        """true_att > 0 means rate decrease (loss ratio goes up)."""
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=30, n_periods=8, true_att=0.05, treatment_period=5, random_seed=11
        )
        assert isinstance(policy_df, pl.DataFrame)
        # Rate log should still have segments
        assert len(rate_log_df) > 0

    def test_zero_att(self):
        policy_df, claims_df, _ = make_synthetic_motor_panel(
            n_segments=20, n_periods=6, true_att=0.0, treatment_period=4, random_seed=22
        )
        assert policy_df["earned_premium"].min() > 0

    def test_minimal_two_segments(self):
        """Should work with n_segments=2."""
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=2, n_periods=4, treat_fraction=0.5, treatment_period=3, random_seed=33
        )
        assert policy_df["segment_id"].n_unique() == 2

    def test_high_noise(self):
        """Large noise_sd should produce valid (if noisy) data."""
        policy_df, _, _ = make_synthetic_motor_panel(
            n_segments=20, n_periods=6, noise_sd=0.30, random_seed=44
        )
        assert policy_df["earned_premium"].is_nan().sum() == 0

    def test_zero_treat_fraction(self):
        """treat_fraction=0 → no treated segments in rate_log."""
        _, _, rate_log_df = make_synthetic_motor_panel(
            n_segments=20, n_periods=6, treat_fraction=0.0, random_seed=55
        )
        assert len(rate_log_df) == 0

    def test_full_treat_fraction(self):
        """treat_fraction=1.0 → all segments treated."""
        policy_df, _, rate_log_df = make_synthetic_motor_panel(
            n_segments=10, n_periods=6, treat_fraction=1.0,
            treatment_period=4, random_seed=66
        )
        assert len(rate_log_df) == 10

    def test_staggered_cohort_periods_are_sequential(self):
        """Staggered cohort periods should all be >= treatment_period."""
        _, _, rate_log_df = make_synthetic_motor_panel(
            n_segments=60, n_periods=14,
            staggered=True, n_stagger_cohorts=3,
            treatment_period=7, random_seed=77
        )
        periods = rate_log_df["first_treated_period"].to_list()
        assert all(p >= 7 for p in periods)

    def test_custom_base_loss_ratio(self):
        """Custom base_loss_ratio should shift the overall mean LR."""
        from insurance_causal_policy._panel import PolicyPanelBuilder
        p1, c1, r1 = make_synthetic_motor_panel(
            n_segments=20, n_periods=4, base_loss_ratio=0.60, treat_fraction=0, random_seed=88
        )
        p2, c2, r2 = make_synthetic_motor_panel(
            n_segments=20, n_periods=4, base_loss_ratio=0.90, treat_fraction=0, random_seed=88
        )
        panel1 = PolicyPanelBuilder(p1, c1, r1).build()
        panel2 = PolicyPanelBuilder(p2, c2, r2).build()
        # Higher base_loss_ratio should produce higher mean loss ratio
        assert panel2["loss_ratio"].mean() > panel1["loss_ratio"].mean()


# ---------------------------------------------------------------------------
# make_synthetic_panel_direct: additional variants
# ---------------------------------------------------------------------------


class TestSyntheticPanelDirectVariants:
    def test_minimal_panel_one_pre_one_post(self):
        """Minimal viable panel: t_pre=1, t_post=1."""
        panel = make_synthetic_panel_direct(
            n_control=5, n_treated=3, t_pre=1, t_post=1,
            true_att=-0.05, random_seed=1
        )
        assert panel.height == 8 * 2  # (5+3) * 2 periods
        assert "loss_ratio" in panel.columns

    def test_zero_noise(self):
        """With noise_sd=0, all variation comes from unit FEs and trend."""
        panel = make_synthetic_panel_direct(
            n_control=10, n_treated=5, t_pre=4, t_post=2,
            true_att=-0.08, noise_sd=0.0, random_seed=2
        )
        # Treated post-treatment rows should all have the same ATT adjustment
        treated_post = panel.filter(
            (pl.col("first_treated_period").is_not_null()) & (pl.col("treated") == 1)
        )
        assert len(treated_post) > 0
        # Loss ratios should exist
        assert treated_post["loss_ratio"].is_nan().sum() == 0

    def test_all_rows_have_positive_loss_ratio(self):
        panel = make_synthetic_panel_direct(
            n_control=20, n_treated=8, t_pre=5, t_post=3,
            true_att=-0.05, noise_sd=0.02, random_seed=3
        )
        assert (panel["loss_ratio"] > 0).all()

    def test_treated_column_only_zeros_and_ones(self):
        panel = make_synthetic_panel_direct(
            n_control=15, n_treated=5, t_pre=4, t_post=3,
            true_att=-0.05, random_seed=4
        )
        vals = set(panel["treated"].to_list())
        assert vals.issubset({0, 1})

    def test_cohort_matches_first_treated_period(self):
        """cohort should equal first_treated_period for treated units."""
        panel = make_synthetic_panel_direct(
            n_control=10, n_treated=5, t_pre=4, t_post=3,
            true_att=-0.05, random_seed=5
        )
        treated = panel.filter(pl.col("first_treated_period").is_not_null())
        mismatches = treated.filter(pl.col("cohort") != pl.col("first_treated_period"))
        assert len(mismatches) == 0

    def test_claim_count_nonneg(self):
        panel = make_synthetic_panel_direct(
            n_control=10, n_treated=5, t_pre=4, t_post=3,
            true_att=-0.05, random_seed=6
        )
        assert (panel["claim_count"] >= 0).all()

    def test_large_positive_att_increases_loss_ratio(self):
        """With true_att > 0 (rate decrease), treated post should be higher."""
        panel = make_synthetic_panel_direct(
            n_control=30, n_treated=10, t_pre=6, t_post=4,
            true_att=0.10, noise_sd=0.01, random_seed=7
        )
        pre_tr = panel.filter(
            (pl.col("first_treated_period").is_not_null()) & (pl.col("treated") == 0)
        )["loss_ratio"].mean()
        post_tr = panel.filter(
            (pl.col("first_treated_period").is_not_null()) & (pl.col("treated") == 1)
        )["loss_ratio"].mean()
        pre_co = panel.filter(
            (pl.col("first_treated_period").is_null()) & (pl.col("period") <= 6)
        )["loss_ratio"].mean()
        post_co = panel.filter(
            (pl.col("first_treated_period").is_null()) & (pl.col("period") > 6)
        )["loss_ratio"].mean()
        naive_did = (post_tr - pre_tr) - (post_co - pre_co)
        assert naive_did > 0  # positive ATT

    def test_custom_trend(self):
        """With positive trend, later periods should have higher LR for controls."""
        panel = make_synthetic_panel_direct(
            n_control=20, n_treated=0, t_pre=8, t_post=0,
            true_att=0.0, noise_sd=0.005, trend=0.02, random_seed=8
        )
        early = panel.filter(pl.col("period") <= 2)["loss_ratio"].mean()
        late = panel.filter(pl.col("period") >= 7)["loss_ratio"].mean()
        assert late > early


# ---------------------------------------------------------------------------
# SDIDEstimator: edge cases
# ---------------------------------------------------------------------------


class TestSDIDEstimatorEdgeCases:
    def test_invalid_inference_method_raises(self):
        from insurance_causal_policy._sdid import SDIDEstimator
        panel = make_synthetic_panel_direct(
            n_control=20, n_treated=5, t_pre=4, t_post=3,
            true_att=-0.05, random_seed=10
        )
        with pytest.raises((ValueError, Exception)):
            SDIDEstimator(panel, outcome="loss_ratio", inference="invalid_method").fit()

    def test_frequency_outcome_runs(self):
        from insurance_causal_policy._sdid import SDIDEstimator
        panel = make_synthetic_panel_direct(
            n_control=25, n_treated=8, t_pre=5, t_post=3,
            true_att=-0.05, random_seed=11
        )
        # Manually add frequency column
        panel = panel.with_columns(
            (pl.col("claim_count") / pl.col("earned_exposure")).alias("frequency")
        )
        est = SDIDEstimator(panel, outcome="frequency", inference="jackknife")
        result = est.fit()
        assert result.outcome_name == "frequency"

    def test_small_panel_placebo_less_than_treated_raises(self):
        """When n_co <= n_tr, placebo inference should raise ValueError."""
        from insurance_causal_policy._sdid import SDIDEstimator
        # Create panel with more treated than control
        panel = make_synthetic_panel_direct(
            n_control=5, n_treated=10, t_pre=4, t_post=3,
            true_att=-0.05, random_seed=12
        )
        est = SDIDEstimator(panel, outcome="loss_ratio", inference="placebo", n_replicates=20)
        with pytest.raises((ValueError, RuntimeError)):
            est.fit()


# ---------------------------------------------------------------------------
# build_panel_from_pandas: additional dtype robustness
# ---------------------------------------------------------------------------


class TestBuildPanelFromPandasRobust:
    def test_float32_period_column(self):
        """Period columns as float32 (common in pandas) should still work."""
        from insurance_causal_policy._panel import build_panel_from_pandas
        policy = pd.DataFrame({
            "segment_id": ["A", "A", "B", "B"],
            "period": np.array([1, 2, 1, 2], dtype=np.float32),
            "earned_premium": [100_000.0, 110_000.0, 80_000.0, 85_000.0],
            "earned_exposure": [500.0, 520.0, 400.0, 410.0],
        })
        claims = pd.DataFrame({
            "segment_id": ["A", "A", "B", "B"],
            "period": np.array([1, 2, 1, 2], dtype=np.float32),
            "incurred_claims": [72_000.0, 75_000.0, 56_000.0, 60_000.0],
            "claim_count": [120, 125, 90, 95],
        })
        rate_log = pd.DataFrame({
            "segment_id": ["A"],
            "first_treated_period": np.array([2], dtype=np.float32),
        })
        result = build_panel_from_pandas(
            policy_df=policy, claims_df=claims, rate_log_df=rate_log
        )
        assert isinstance(result, pl.DataFrame)
        assert "loss_ratio" in result.columns

    def test_integer_columns_as_int32(self):
        """int32 columns in pandas should be accepted."""
        from insurance_causal_policy._panel import build_panel_from_pandas
        policy = pd.DataFrame({
            "segment_id": ["A", "B"],
            "period": np.array([1, 1], dtype=np.int32),
            "earned_premium": [100_000.0, 80_000.0],
            "earned_exposure": [500.0, 400.0],
        })
        claims = pd.DataFrame({
            "segment_id": ["A", "B"],
            "period": np.array([1, 1], dtype=np.int32),
            "incurred_claims": [70_000.0, 55_000.0],
            "claim_count": np.array([100, 80], dtype=np.int32),
        })
        rate_log = pd.DataFrame({
            "segment_id": pd.Series([], dtype=str),
            "first_treated_period": pd.Series([], dtype=np.int32),
        })
        result = build_panel_from_pandas(
            policy_df=policy, claims_df=claims, rate_log_df=rate_log
        )
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# Full pipeline: DRSC + sensitivity + FCA pack
# ---------------------------------------------------------------------------


class TestDRSCFullPipeline:
    def test_drsc_pipeline_to_fca_pack(self):
        """DRSC → sensitivity → FCA pack should all run without error."""
        from insurance_causal_policy._drsc import DoublyRobustSCEstimator
        from insurance_causal_policy._sensitivity import compute_sensitivity
        from insurance_causal_policy._evidence import FCAEvidencePack

        panel = make_synthetic_panel_direct(
            n_control=30, n_treated=10, t_pre=8, t_post=4,
            true_att=-0.07, noise_sd=0.03, random_seed=50
        )
        result = DoublyRobustSCEstimator(
            panel, outcome="loss_ratio",
            inference="bootstrap", n_replicates=50, random_seed=50
        ).fit()
        sens = compute_sensitivity(result, m_values=[0.0, 0.5, 1.0])
        pack = FCAEvidencePack(
            result=result, sensitivity=sens,
            product_line="Motor", rate_change_date="2024-Q2",
        )
        md = pack.to_markdown()
        assert "Motor" in md
        assert result.att < 0  # direction correct

    def test_drsc_summary_after_fit(self):
        from insurance_causal_policy._drsc import DoublyRobustSCEstimator

        panel = make_synthetic_panel_direct(
            n_control=25, n_treated=8, t_pre=6, t_post=3,
            true_att=-0.06, random_seed=51
        )
        result = DoublyRobustSCEstimator(
            panel, outcome="loss_ratio",
            inference="analytic"
        ).fit()
        s = result.summary()
        assert "DRSC estimate" in s
        assert "bootstrap" not in s.lower() or "analytic" in s


# ---------------------------------------------------------------------------
# SDIDResult: boundary conditions on properties
# ---------------------------------------------------------------------------


class TestSDIDResultBoundaryConditions:
    def test_significant_at_boundary_005(self):
        """pval exactly 0.05 should NOT be significant (strict < comparison)."""
        r = _make_sdid_result()
        r2 = SDIDResult(**{**r.__dict__, "pval": 0.05})
        assert r2.significant is False

    def test_significant_just_below_005(self):
        r = _make_sdid_result()
        r2 = SDIDResult(**{**r.__dict__, "pval": 0.0499})
        assert r2.significant is True

    def test_pre_trends_pass_boundary_010(self):
        """pre_trend_pval exactly 0.10 should NOT pass (strict > comparison)."""
        r = _make_sdid_result(pre_trend_pval=0.10)
        assert r.pre_trends_pass is False

    def test_pre_trends_pass_just_above_010(self):
        r = _make_sdid_result(pre_trend_pval=0.101)
        assert r.pre_trends_pass is True

    def test_large_ci_contains_att(self):
        r = _make_sdid_result(att=0.0, se=0.5)
        assert r.ci_low <= r.att <= r.ci_high

    def test_weights_regularisation_zeta_stored(self):
        r = _make_sdid_result()
        assert r.weights.regularisation_zeta == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# DRSCWeights: basic attributes
# ---------------------------------------------------------------------------


class TestDRSCWeightsAttributes:
    def test_sc_weights_accessible(self):
        w = _make_drsc_weights(n_co=5)
        assert len(w.sc_weights) == 5

    def test_propensity_ratios_accessible(self):
        w = _make_drsc_weights(n_co=5)
        assert len(w.propensity_ratios) == 5

    def test_m_delta_accessible(self):
        w = _make_drsc_weights()
        assert isinstance(w.m_delta, float)

    def test_sc_weights_can_be_negative(self):
        """Unlike SDID, DRSC weights can be negative — this is by design."""
        w = DRSCWeights(
            sc_weights=pd.Series([-0.3, 1.3], index=["co_0", "co_1"]),
            propensity_ratios=pd.Series([5.0, 5.0], index=["co_0", "co_1"]),
            m_delta=0.01,
        )
        assert w.sc_weights.min() < 0
        assert abs(w.sc_weights.sum() - 1.0) < 1e-6  # still sum to 1


# ---------------------------------------------------------------------------
# Import check: all public symbols from __all__ are importable
# ---------------------------------------------------------------------------


class TestPublicAPIComplete:
    def test_drsc_weights_importable(self):
        from insurance_causal_policy import DRSCWeights
        assert DRSCWeights is not None

    def test_drsc_result_importable(self):
        from insurance_causal_policy import DRSCResult
        assert DRSCResult is not None

    def test_staggered_result_importable(self):
        from insurance_causal_policy import StaggeredResult
        assert StaggeredResult is not None

    def test_plot_synthetic_trajectory_importable(self):
        from insurance_causal_policy import plot_synthetic_trajectory
        assert callable(plot_synthetic_trajectory)

    def test_version_semver_format(self):
        import insurance_causal_policy
        v = insurance_causal_policy.__version__
        parts = v.split(".")
        # Should be at least major.minor format
        assert len(parts) >= 2
        # First two parts should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()
