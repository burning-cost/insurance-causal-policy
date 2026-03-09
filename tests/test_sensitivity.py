"""Tests for _sensitivity.py HonestDiD-style sensitivity analysis."""

import numpy as np
import pandas as pd
import pytest
from insurance_causal_policy._sensitivity import compute_sensitivity, plot_sensitivity
from insurance_causal_policy._types import SDIDResult, SDIDWeights, SensitivityResult


def make_result(att=-0.05, se=0.015, pre_trend_pval=0.40):
    event_df = pd.DataFrame({
        "period_rel": [-4, -3, -2, -1, 0, 1, 2, 3],
        "att": [0.002, -0.001, 0.003, -0.002, att * 0.8, att, att * 1.1, att],
        "se": [0.01] * 8,
        "ci_low": [np.nan] * 8,
        "ci_high": [np.nan] * 8,
    })
    weights = SDIDWeights(
        unit_weights=pd.Series([0.6, 0.4], index=["A", "B"]),
        time_weights=pd.Series([0.5, 0.5], index=[1, 2]),
        unit_intercept=0.002,
        regularisation_zeta=0.03,
    )
    return SDIDResult(
        att=att, se=se,
        ci_low=att - 1.96 * se, ci_high=att + 1.96 * se,
        pval=0.001,
        weights=weights,
        pre_trend_pval=pre_trend_pval,
        event_study=event_df,
        n_treated=10, n_control=5, n_control_total=30,
        t_pre=4, t_post=4,
        outcome_name="loss_ratio",
        inference_method="placebo",
        n_replicates=200,
    )


class TestComputeSensitivity:
    def test_returns_sensitivity_result(self):
        result = make_result()
        sens = compute_sensitivity(result)
        assert isinstance(sens, SensitivityResult)

    def test_default_m_values(self):
        result = make_result()
        sens = compute_sensitivity(result)
        assert len(sens.m_values) > 0
        assert 0.0 in sens.m_values

    def test_custom_m_values(self):
        result = make_result()
        m_vals = [0, 0.5, 1.0, 2.0]
        sens = compute_sensitivity(result, m_values=m_vals)
        assert sens.m_values == m_vals

    def test_att_lower_at_m0_equals_normal_ci(self):
        result = make_result(att=-0.05, se=0.015)
        sens = compute_sensitivity(result, m_values=[0.0, 1.0])
        expected_lower = -0.05 - 1.96 * 0.015
        assert abs(sens.att_lower[0] - expected_lower) < 1e-4

    def test_att_upper_increases_with_m(self):
        result = make_result()
        sens = compute_sensitivity(result, m_values=[0.0, 1.0, 2.0])
        assert sens.att_upper[1] >= sens.att_upper[0]
        assert sens.att_upper[2] >= sens.att_upper[1]

    def test_att_lower_decreases_with_m(self):
        result = make_result()
        sens = compute_sensitivity(result, m_values=[0.0, 1.0, 2.0])
        assert sens.att_lower[1] <= sens.att_lower[0]
        assert sens.att_lower[2] <= sens.att_lower[1]

    def test_pre_period_sd_positive(self):
        result = make_result()
        sens = compute_sensitivity(result)
        assert sens.pre_period_sd > 0

    def test_breakdown_nonneg(self):
        result = make_result()
        sens = compute_sensitivity(result)
        assert sens.m_breakdown >= 0

    def test_significant_result_has_positive_breakdown(self):
        # ATT is significantly negative — should survive some violations
        result = make_result(att=-0.05, se=0.01)
        sens = compute_sensitivity(result)
        assert sens.m_breakdown > 0

    def test_insignificant_result_low_breakdown(self):
        # ATT not significant (wide CI includes zero) → breakdown at M=0
        result = make_result(att=-0.01, se=0.02)
        sens = compute_sensitivity(result)
        assert sens.m_breakdown == 0.0

    def test_to_dataframe(self):
        result = make_result()
        sens = compute_sensitivity(result)
        df = sens.to_dataframe()
        assert "m" in df.columns
        assert "att_lower" in df.columns
        assert "att_upper" in df.columns

    def test_no_event_study_raises(self):
        result = make_result()
        result = result.__class__(**{**result.__dict__, "event_study": None})
        with pytest.raises(ValueError, match="event_study"):
            compute_sensitivity(result)

    def test_smooth_method(self):
        result = make_result()
        sens = compute_sensitivity(result, method="smooth")
        assert isinstance(sens, SensitivityResult)

    def test_invalid_method_raises(self):
        result = make_result()
        with pytest.raises(ValueError, match="Unknown sensitivity method"):
            compute_sensitivity(result, method="bad_method")


class TestPlotSensitivity:
    def test_returns_figure(self):
        result = make_result()
        sens = compute_sensitivity(result)
        fig = plot_sensitivity(sens)
        assert hasattr(fig, "savefig")

    def test_custom_title(self):
        result = make_result()
        sens = compute_sensitivity(result)
        fig = plot_sensitivity(sens, title="My Sensitivity Test")
        assert fig is not None
