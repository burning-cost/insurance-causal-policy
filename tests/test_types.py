"""Tests for _types.py result dataclasses."""

import numpy as np
import pandas as pd
import pytest
from insurance_causal_policy._types import SDIDResult, SDIDWeights, SensitivityResult, StaggeredResult


def make_sdid_result(att=-0.05, se=0.02, pre_trend_pval=0.40, n_treated=20, n_control=15):
    weights = SDIDWeights(
        unit_weights=pd.Series([0.4, 0.35, 0.25], index=["seg_a", "seg_b", "seg_c"]),
        time_weights=pd.Series([0.3, 0.4, 0.3], index=[1, 2, 3]),
        unit_intercept=0.01,
        regularisation_zeta=0.05,
    )
    event_df = pd.DataFrame({
        "period_rel": [-3, -2, -1, 0, 1, 2],
        "att": [0.001, -0.002, 0.003, -0.04, -0.055, -0.06],
        "se": [np.nan] * 6,
        "ci_low": [np.nan] * 6,
        "ci_high": [np.nan] * 6,
    })
    return SDIDResult(
        att=att,
        se=se,
        ci_low=att - 1.96 * se,
        ci_high=att + 1.96 * se,
        pval=0.012,
        weights=weights,
        pre_trend_pval=pre_trend_pval,
        event_study=event_df,
        n_treated=n_treated,
        n_control=n_control,
        n_control_total=60,
        t_pre=8,
        t_post=4,
        outcome_name="loss_ratio",
        inference_method="placebo",
        n_replicates=200,
    )


class TestSDIDResult:
    def test_significant_true(self):
        r = make_sdid_result()
        r = r.__class__(**{**r.__dict__, "pval": 0.01})
        assert r.significant

    def test_significant_false(self):
        r = make_sdid_result()
        r = r.__class__(**{**r.__dict__, "pval": 0.20})
        assert not r.significant

    def test_pre_trends_pass_high_pval(self):
        r = make_sdid_result(pre_trend_pval=0.40)
        assert r.pre_trends_pass

    def test_pre_trends_fail_low_pval(self):
        r = make_sdid_result(pre_trend_pval=0.02)
        assert not r.pre_trends_pass

    def test_pre_trends_pass_none(self):
        r = make_sdid_result()
        r = r.__class__(**{**r.__dict__, "pre_trend_pval": None})
        assert r.pre_trends_pass  # None treated as pass

    def test_summary_returns_string(self):
        r = make_sdid_result()
        s = r.summary()
        assert isinstance(s, str)
        assert "loss_ratio" in s
        assert "placebo" in s

    def test_summary_contains_att(self):
        r = make_sdid_result(att=-0.05)
        assert "-0.05" in r.summary() or "decrease" in r.summary()

    def test_fca_summary_contains_product_line(self):
        r = make_sdid_result()
        txt = r.to_fca_summary(product_line="Motor", rate_change_date="2023-Q1")
        assert "Motor" in txt
        assert "2023-Q1" in txt

    def test_fca_summary_contains_att(self):
        r = make_sdid_result(att=-0.05)
        txt = r.to_fca_summary()
        assert "loss_ratio" in txt

    def test_fca_summary_contains_ci(self):
        r = make_sdid_result(att=-0.05, se=0.02)
        txt = r.to_fca_summary()
        assert "CI" in txt or "confidence" in txt.lower()

    def test_ci_bounds_correct(self):
        r = make_sdid_result(att=-0.05, se=0.02)
        assert abs(r.ci_low - (-0.05 - 1.96 * 0.02)) < 1e-6
        assert abs(r.ci_high - (-0.05 + 1.96 * 0.02)) < 1e-6

    def test_n_control_accessible(self):
        r = make_sdid_result(n_control=15)
        assert r.n_control == 15

    def test_weights_accessible(self):
        r = make_sdid_result()
        assert len(r.weights.unit_weights) == 3
        assert abs(r.weights.unit_weights.sum() - 1.0) < 0.01


class TestSDIDWeights:
    def test_weights_sum_to_one(self):
        w = SDIDWeights(
            unit_weights=pd.Series([0.5, 0.3, 0.2]),
            time_weights=pd.Series([0.4, 0.6]),
            unit_intercept=0.005,
            regularisation_zeta=0.02,
        )
        assert abs(w.unit_weights.sum() - 1.0) < 1e-6
        assert abs(w.time_weights.sum() - 1.0) < 1e-6

    def test_intercept_stored(self):
        w = SDIDWeights(
            unit_weights=pd.Series([1.0]),
            time_weights=pd.Series([1.0]),
            unit_intercept=0.123,
            regularisation_zeta=0.01,
        )
        assert w.unit_intercept == 0.123


class TestSensitivityResult:
    def test_to_dataframe(self):
        s = SensitivityResult(
            m_values=[0, 1, 2],
            att_lower=[-0.1, -0.15, -0.2],
            att_upper=[-0.02, 0.05, 0.1],
            m_breakdown=1.3,
            pre_period_sd=0.01,
        )
        df = s.to_dataframe()
        assert list(df.columns) == ["m", "att_lower", "att_upper"]
        assert len(df) == 3

    def test_summary_mentions_breakdown(self):
        s = SensitivityResult(
            m_values=[0, 1, 2],
            att_lower=[-0.1, -0.15, -0.2],
            att_upper=[-0.02, 0.05, 0.1],
            m_breakdown=1.3,
            pre_period_sd=0.01,
        )
        assert "1.3" in s.summary() or "breakdown" in s.summary().lower()

    def test_summary_robust_if_high_breakdown(self):
        s = SensitivityResult(
            m_values=[0, 1, 2],
            att_lower=[-0.1, -0.14, -0.18],
            att_upper=[-0.02, -0.01, 0.0],
            m_breakdown=3.0,
            pre_period_sd=0.01,
        )
        assert "robust" in s.summary().lower()
