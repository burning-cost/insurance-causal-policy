"""Tests for _staggered.py CS21 staggered estimator."""

import numpy as np
import polars as pl
import pytest
from insurance_causal_policy._staggered import (
    StaggeredEstimator,
    _aggregate_event_study,
    _estimate_att_gt_naive,
    _fit_native_cs21,
)
from insurance_causal_policy._synthetic import make_synthetic_motor_panel, make_synthetic_panel_direct
from insurance_causal_policy._types import StaggeredResult


def make_staggered_panel(n_segments=60, n_periods=14, seed=5):
    """Generate a staggered panel via the synthetic data generator."""
    policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
        n_segments=n_segments,
        n_periods=n_periods,
        treat_fraction=0.4,
        true_att=-0.07,
        treatment_period=8,
        staggered=True,
        n_stagger_cohorts=3,
        random_seed=seed,
    )
    from insurance_causal_policy._panel import PolicyPanelBuilder
    builder = PolicyPanelBuilder(policy_df, claims_df, rate_log_df)
    return builder.build()


class TestAggregateEventStudy:
    def test_returns_dataframe(self):
        import pandas as pd
        att_gt = pd.DataFrame({
            "cohort": [5, 5, 7, 7],
            "period": [4, 5, 6, 7],
            "att": [0.001, -0.05, 0.002, -0.06],
            "se": [0.01, 0.01, 0.01, 0.01],
        })
        event = _aggregate_event_study(att_gt)
        assert "period_rel" in event.columns
        assert "att" in event.columns

    def test_period_rel_computed(self):
        import pandas as pd
        att_gt = pd.DataFrame({
            "cohort": [5, 5],
            "period": [4, 6],
            "att": [0.001, -0.05],
            "se": [0.01, 0.01],
        })
        event = _aggregate_event_study(att_gt)
        period_rels = set(event["period_rel"].tolist())
        assert -1 in period_rels  # period 4, cohort 5 → 4-5 = -1
        assert 1 in period_rels   # period 6, cohort 5 → 6-5 = +1


class TestEstimateAttGtNaive:
    def test_returns_tuple(self):
        import pandas as pd
        panel = pd.DataFrame({
            "unit_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "period": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "outcome": [0.70, 0.72, 0.65, 0.68, 0.70, 0.71, 0.69, 0.71, 0.70],
            "first_treated_period": [3.0, 3.0, 3.0, None, None, None, None, None, None],
        })
        att, se = _estimate_att_gt_naive(
            panel, cohort=3, period=3,
            outcome_col="outcome", unit_col="unit_id", period_col="period",
        )
        assert isinstance(att, float)
        assert isinstance(se, float)

    def test_null_for_empty_cohort(self):
        import pandas as pd
        panel = pd.DataFrame({
            "unit_id": ["A"],
            "period": [1],
            "outcome": [0.70],
            "first_treated_period": [None],
        })
        att, se = _estimate_att_gt_naive(
            panel, cohort=5, period=2,
            outcome_col="outcome", unit_col="unit_id", period_col="period",
        )
        import math
        assert math.isnan(att)


class TestFitNativeCS21:
    def test_returns_dataframe(self):
        import pandas as pd
        from insurance_causal_policy._synthetic import make_synthetic_panel_direct
        panel = make_synthetic_panel_direct(
            n_control=20, n_treated=8, t_pre=5, t_post=3, random_seed=10
        ).to_pandas()
        panel = panel.rename(columns={
            "segment_id": "unit_id",
            "loss_ratio": "outcome",
        })
        panel["first_treated_period"] = panel["first_treated_period"].astype("float64")
        att_gt = _fit_native_cs21(
            panel,
            outcome_col="outcome",
            unit_col="unit_id",
            period_col="period",
            cohort_col="first_treated_period",
        )
        assert "cohort" in att_gt.columns
        assert "period" in att_gt.columns
        assert "att" in att_gt.columns
        assert len(att_gt) > 0


class TestStaggeredEstimator:
    def setup_method(self):
        # Use a smaller panel for speed
        self.panel = make_synthetic_panel_direct(
            n_control=30, n_treated=10,
            t_pre=6, t_post=4,
            true_att=-0.07, random_seed=15,
        )
        # Add staggered structure: split treated into 2 cohorts
        import polars as pl
        segs = self.panel.filter(
            pl.col("first_treated_period").is_not_null()
        )["segment_id"].unique().to_list()
        cohort_map = {}
        for i, s in enumerate(segs):
            cohort_map[s] = 7 if i < len(segs) // 2 else 8

        self.panel = self.panel.with_columns(
            pl.col("segment_id").map_elements(
                lambda s: cohort_map.get(s, None), return_dtype=pl.Float64
            ).alias("cohort"),
            pl.struct(["segment_id", "period"]).map_elements(
                lambda row: 1 if (
                    cohort_map.get(row["segment_id"]) is not None
                    and row["period"] >= cohort_map.get(row["segment_id"], 9999)
                ) else 0,
                return_dtype=pl.Int64,
            ).alias("treated"),
        )

    def test_fit_returns_staggered_result(self):
        est = StaggeredEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        assert isinstance(result, StaggeredResult)

    def test_att_overall_is_float(self):
        est = StaggeredEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        assert isinstance(result.att_overall, float)

    def test_se_overall_positive(self):
        est = StaggeredEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        assert result.se_overall > 0

    def test_att_gt_dataframe(self):
        est = StaggeredEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        assert len(result.att_gt) > 0
        assert "att" in result.att_gt.columns

    def test_event_study_dataframe(self):
        est = StaggeredEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        assert len(result.event_study) > 0
        assert "period_rel" in result.event_study.columns

    def test_n_cohorts_correct(self):
        est = StaggeredEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        assert result.n_cohorts == 2

    def test_outcome_name_stored(self):
        est = StaggeredEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        assert result.outcome_name == "loss_ratio"

    def test_control_group_stored(self):
        est = StaggeredEstimator(
            self.panel, outcome="loss_ratio", control_group="notyettreated"
        )
        result = est.fit()
        assert result.control_group == "notyettreated"

    def test_pre_trends_pass_attribute(self):
        est = StaggeredEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        # Should be bool or True (if pval is None)
        assert isinstance(result.pre_trends_pass, bool)

    def test_att_negative_direction(self):
        """With true ATT -0.07, estimated overall ATT should be negative."""
        est = StaggeredEstimator(self.panel, outcome="loss_ratio")
        result = est.fit()
        assert result.att_overall < 0
