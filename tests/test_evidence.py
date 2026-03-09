"""Tests for _evidence.py FCA evidence pack generator."""

import json

import pandas as pd
import pytest
from insurance_causal_policy._evidence import FCAEvidencePack
from insurance_causal_policy._sensitivity import compute_sensitivity
from insurance_causal_policy._types import SDIDResult, SDIDWeights, SensitivityResult


def make_result(att=-0.05):
    event_df = pd.DataFrame({
        "period_rel": [-3, -2, -1, 0, 1, 2],
        "att": [0.001, -0.002, 0.001, att, att * 1.1, att * 0.9],
        "se": [0.01] * 6,
        "ci_low": [-0.02] * 6,
        "ci_high": [0.02] * 6,
    })
    weights = SDIDWeights(
        unit_weights=pd.Series([0.6, 0.4], index=["A", "B"]),
        time_weights=pd.Series([0.5, 0.5], index=[1, 2]),
        unit_intercept=0.002,
        regularisation_zeta=0.03,
    )
    return SDIDResult(
        att=att, se=0.015,
        ci_low=att - 1.96 * 0.015,
        ci_high=att + 1.96 * 0.015,
        pval=0.001,
        weights=weights,
        pre_trend_pval=0.45,
        event_study=event_df,
        n_treated=10, n_control=5, n_control_total=30,
        t_pre=8, t_post=4,
        outcome_name="loss_ratio",
        inference_method="placebo",
        n_replicates=200,
    )


def make_sensitivity(result):
    return compute_sensitivity(result, m_values=[0, 0.5, 1.0, 2.0])


class TestFCAEvidencePackMarkdown:
    def setup_method(self):
        self.result = make_result()
        self.sens = make_sensitivity(self.result)
        self.pack = FCAEvidencePack(
            result=self.result,
            sensitivity=self.sens,
            product_line="Motor",
            rate_change_date="2023-Q1",
            rate_change_magnitude="+8.5%",
            analyst="Pricing Team",
            panel_summary={
                "n_segments": 40,
                "n_treated_segments": 10,
                "n_control_segments": 30,
                "n_periods": 12,
                "n_cells": 480,
                "pct_nonzero_exposure": 98.5,
                "outcome": "loss_ratio",
            },
        )

    def test_to_markdown_returns_string(self):
        md = self.pack.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 100

    def test_markdown_contains_product_line(self):
        md = self.pack.to_markdown()
        assert "Motor" in md

    def test_markdown_contains_rate_change_date(self):
        md = self.pack.to_markdown()
        assert "2023-Q1" in md

    def test_markdown_contains_att(self):
        md = self.pack.to_markdown()
        assert "-0.05" in md or "0.05" in md

    def test_markdown_contains_methodology(self):
        md = self.pack.to_markdown()
        assert "Synthetic Difference-in-Differences" in md or "SDID" in md

    def test_markdown_contains_parallel_trends(self):
        md = self.pack.to_markdown()
        assert "parallel trends" in md.lower() or "Parallel Trends" in md

    def test_markdown_contains_sensitivity(self):
        md = self.pack.to_markdown()
        assert "sensitivity" in md.lower() or "HonestDiD" in md

    def test_markdown_contains_data_quality(self):
        md = self.pack.to_markdown()
        assert "40" in md  # n_segments

    def test_markdown_contains_fca_reference(self):
        md = self.pack.to_markdown()
        assert "FCA" in md or "Consumer Duty" in md

    def test_markdown_contains_caveats(self):
        md = self.pack.to_markdown()
        assert "caveat" in md.lower() or "limitation" in md.lower()

    def test_pass_warning_in_markdown(self):
        md = self.pack.to_markdown()
        assert "PASS" in md or "pass" in md.lower()

    def test_fail_warning_in_markdown(self):
        # Make a result that fails parallel trends
        bad_result = make_result()
        bad_result = bad_result.__class__(**{**bad_result.__dict__, "pre_trend_pval": 0.02})
        pack = FCAEvidencePack(result=bad_result)
        md = pack.to_markdown()
        assert "WARNING" in md or "warning" in md.lower()

    def test_additional_notes_included(self):
        pack = FCAEvidencePack(
            result=self.result,
            additional_notes="Ogden rate changed during window",
        )
        md = pack.to_markdown()
        assert "Ogden" in md


class TestFCAEvidencePackJSON:
    def setup_method(self):
        self.result = make_result()
        self.sens = make_sensitivity(self.result)
        self.pack = FCAEvidencePack(
            result=self.result,
            sensitivity=self.sens,
            product_line="Home",
        )

    def test_to_dict_returns_dict(self):
        d = self.pack.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_metadata(self):
        d = self.pack.to_dict()
        assert "metadata" in d
        assert d["metadata"]["product_line"] == "Home"

    def test_to_dict_has_estimation(self):
        d = self.pack.to_dict()
        assert "estimation" in d
        assert "att" in d["estimation"]

    def test_to_dict_att_correct(self):
        d = self.pack.to_dict()
        assert abs(d["estimation"]["att"] - (-0.05)) < 1e-6

    def test_to_dict_has_parallel_trends(self):
        d = self.pack.to_dict()
        assert "parallel_trends" in d
        assert "pval" in d["parallel_trends"]

    def test_to_dict_has_sensitivity(self):
        d = self.pack.to_dict()
        assert "sensitivity" in d
        assert isinstance(d["sensitivity"], list)

    def test_to_json_valid(self):
        json_str = self.pack.to_json()
        parsed = json.loads(json_str)
        assert "metadata" in parsed

    def test_to_json_is_string(self):
        json_str = self.pack.to_json()
        assert isinstance(json_str, str)

    def test_no_sensitivity_handled(self):
        pack = FCAEvidencePack(result=self.result, sensitivity=None)
        d = pack.to_dict()
        assert d["sensitivity"] is None

    def test_no_panel_summary_handled(self):
        pack = FCAEvidencePack(result=self.result, panel_summary=None)
        md = pack.to_markdown()
        assert isinstance(md, str)  # should not crash
