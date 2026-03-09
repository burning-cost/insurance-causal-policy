"""Tests for _event_study.py diagnostic functions."""

import numpy as np
import pandas as pd
import pytest
from insurance_causal_policy._event_study import (
    plot_event_study,
    plot_unit_weights,
    pre_trend_summary,
)
from insurance_causal_policy._types import SDIDResult, SDIDWeights


def make_result_with_event_study(pre_trend_pval=0.40, att=-0.05):
    event_df = pd.DataFrame({
        "period_rel": [-4, -3, -2, -1, 0, 1, 2, 3],
        "att": [0.002, -0.003, 0.001, -0.002, -0.04, -0.05, -0.06, -0.055],
        "se": [0.01] * 8,
        "ci_low": [-0.018, -0.023, -0.019, -0.022, -0.06, -0.07, -0.08, -0.075],
        "ci_high": [0.022, 0.017, 0.021, 0.018, -0.02, -0.03, -0.04, -0.035],
    })
    weights = SDIDWeights(
        unit_weights=pd.Series(
            [0.4, 0.35, 0.15, 0.10],
            index=["north_west", "south_east", "midlands", "east"],
        ),
        time_weights=pd.Series([0.25, 0.3, 0.25, 0.2], index=[1, 2, 3, 4]),
        unit_intercept=0.005,
        regularisation_zeta=0.04,
    )
    return SDIDResult(
        att=att,
        se=0.015,
        ci_low=att - 1.96 * 0.015,
        ci_high=att + 1.96 * 0.015,
        pval=0.001,
        weights=weights,
        pre_trend_pval=pre_trend_pval,
        event_study=event_df,
        n_treated=15,
        n_control=3,
        n_control_total=40,
        t_pre=4,
        t_post=4,
        outcome_name="loss_ratio",
        inference_method="placebo",
        n_replicates=200,
    )


class TestPlotEventStudy:
    def test_returns_figure(self):
        import matplotlib
        result = make_result_with_event_study()
        fig = plot_event_study(result)
        assert hasattr(fig, "savefig")  # is a Figure

    def test_pre_trend_pval_annotated(self):
        result = make_result_with_event_study(pre_trend_pval=0.04)
        fig = plot_event_study(result, annotate_pval=True)
        # Just check it doesn't raise
        assert fig is not None

    def test_passing_pval_annotated(self):
        result = make_result_with_event_study(pre_trend_pval=0.42)
        fig = plot_event_study(result, annotate_pval=True)
        assert fig is not None

    def test_no_event_study_raises(self):
        result = make_result_with_event_study()
        result = result.__class__(**{**result.__dict__, "event_study": None})
        with pytest.raises(ValueError, match="Event study data"):
            plot_event_study(result)

    def test_custom_title(self):
        result = make_result_with_event_study()
        fig = plot_event_study(result, title="My Custom Title")
        assert fig is not None

    def test_returns_figure_not_none(self):
        result = make_result_with_event_study()
        fig = plot_event_study(result)
        assert fig is not None


class TestPlotUnitWeights:
    def test_returns_figure(self):
        result = make_result_with_event_study()
        fig = plot_unit_weights(result)
        assert hasattr(fig, "savefig")

    def test_custom_n_top(self):
        result = make_result_with_event_study()
        fig = plot_unit_weights(result, n_top=2)
        assert fig is not None

    def test_all_shown_when_n_top_large(self):
        result = make_result_with_event_study()
        fig = plot_unit_weights(result, n_top=100)
        assert fig is not None


class TestPreTrendSummary:
    def test_returns_dict(self):
        result = make_result_with_event_study()
        s = pre_trend_summary(result)
        assert isinstance(s, dict)

    def test_has_required_keys(self):
        result = make_result_with_event_study()
        s = pre_trend_summary(result)
        assert "pval" in s
        assert "pass" in s
        assert "pre_atts" in s
        assert "interpretation" in s

    def test_pass_status_correct_high_pval(self):
        result = make_result_with_event_study(pre_trend_pval=0.50)
        s = pre_trend_summary(result)
        assert s["pass"] is True

    def test_fail_status_correct_low_pval(self):
        result = make_result_with_event_study(pre_trend_pval=0.02)
        s = pre_trend_summary(result)
        assert s["pass"] is False

    def test_pre_atts_list(self):
        result = make_result_with_event_study()
        s = pre_trend_summary(result)
        assert isinstance(s["pre_atts"], list)
        assert len(s["pre_atts"]) == 4  # 4 pre-treatment periods

    def test_interpretation_is_string(self):
        result = make_result_with_event_study()
        s = pre_trend_summary(result)
        assert isinstance(s["interpretation"], str)
        assert len(s["interpretation"]) > 10

    def test_no_event_study_handled(self):
        result = make_result_with_event_study()
        result = result.__class__(**{**result.__dict__, "event_study": None})
        s = pre_trend_summary(result)
        assert s["pval"] is None
        assert "not available" in s["interpretation"]

    def test_max_abs_pre_att(self):
        result = make_result_with_event_study()
        s = pre_trend_summary(result)
        assert s["max_abs_pre_att"] is not None
        assert s["max_abs_pre_att"] >= 0
