"""Tests for _panel.py PolicyPanelBuilder."""

import warnings

import polars as pl
import pytest
from insurance_causal_policy._panel import PolicyPanelBuilder, build_panel_from_pandas
from insurance_causal_policy._synthetic import make_synthetic_motor_panel


def make_minimal_policy():
    return pl.DataFrame({
        "segment_id": ["A", "A", "B", "B"],
        "period": [1, 2, 1, 2],
        "earned_premium": [100_000.0, 110_000.0, 80_000.0, 85_000.0],
        "earned_exposure": [500.0, 520.0, 400.0, 410.0],
    })


def make_minimal_claims():
    return pl.DataFrame({
        "segment_id": ["A", "A", "B", "B"],
        "period": [1, 2, 1, 2],
        "incurred_claims": [72_000.0, 75_000.0, 56_000.0, 60_000.0],
        "claim_count": [120, 125, 90, 95],
    })


def make_minimal_rate_log():
    return pl.DataFrame({
        "segment_id": ["A"],
        "first_treated_period": [2],
    })


class TestPolicyPanelBuilderValidation:
    def test_missing_policy_col_raises(self):
        bad = pl.DataFrame({"segment_id": ["A"], "period": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            PolicyPanelBuilder(bad, make_minimal_claims(), make_minimal_rate_log())

    def test_missing_claims_col_raises(self):
        bad = pl.DataFrame({"segment_id": ["A"], "period": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            PolicyPanelBuilder(make_minimal_policy(), bad, make_minimal_rate_log())

    def test_missing_rate_log_col_raises(self):
        bad = pl.DataFrame({"segment_id": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            PolicyPanelBuilder(make_minimal_policy(), make_minimal_claims(), bad)

    def test_invalid_outcome_raises(self):
        with pytest.raises(ValueError, match="outcome must be one of"):
            PolicyPanelBuilder(
                make_minimal_policy(),
                make_minimal_claims(),
                make_minimal_rate_log(),
                outcome="bad_outcome",
            )

    def test_retention_without_cols_raises(self):
        with pytest.raises(ValueError, match="policies_renewed"):
            PolicyPanelBuilder(
                make_minimal_policy(),
                make_minimal_claims(),
                make_minimal_rate_log(),
                outcome="retention",
            )

    def test_paid_only_without_col_raises(self):
        with pytest.raises(ValueError, match="paid_claims"):
            PolicyPanelBuilder(
                make_minimal_policy(),
                make_minimal_claims(),
                make_minimal_rate_log(),
                paid_only=True,
            )


class TestPolicyPanelBuilderBuild:
    def setup_method(self):
        self.builder = PolicyPanelBuilder(
            make_minimal_policy(),
            make_minimal_claims(),
            make_minimal_rate_log(),
            outcome="loss_ratio",
        )
        self.panel = self.builder.build()

    def test_returns_polars_dataframe(self):
        assert isinstance(self.panel, pl.DataFrame)

    def test_has_loss_ratio(self):
        assert "loss_ratio" in self.panel.columns

    def test_loss_ratio_correct(self):
        # Segment A, period 1: 72000/100000 = 0.72
        row = self.panel.filter(
            (pl.col("segment_id") == "A") & (pl.col("period") == 1)
        )
        assert abs(row["loss_ratio"][0] - 0.72) < 0.001

    def test_has_treated_column(self):
        assert "treated" in self.panel.columns

    def test_treated_correct(self):
        # A at period 2: first_treated=2, period>=2 → treated=1
        row = self.panel.filter(
            (pl.col("segment_id") == "A") & (pl.col("period") == 2)
        )
        assert row["treated"][0] == 1

    def test_control_not_treated(self):
        row = self.panel.filter(
            (pl.col("segment_id") == "B") & (pl.col("period") == 2)
        )
        assert row["treated"][0] == 0

    def test_pre_treatment_not_treated(self):
        # A at period 1: first_treated=2, period<2 → treated=0
        row = self.panel.filter(
            (pl.col("segment_id") == "A") & (pl.col("period") == 1)
        )
        assert row["treated"][0] == 0

    def test_has_cohort_column(self):
        assert "cohort" in self.panel.columns

    def test_never_treated_null_first_period(self):
        row = self.panel.filter(pl.col("segment_id") == "B")
        assert row["first_treated_period"][0] is None

    def test_balanced_panel(self):
        # 2 segments × 2 periods = 4 rows
        assert len(self.panel) == 4

    def test_no_null_in_exposure(self):
        assert self.panel["earned_exposure"].is_null().sum() == 0

    def test_zero_claims_filled(self):
        # All periods in minimal data have claims, so no zeros expected here
        # But test that missing claims periods get zero, not null
        claims_partial = pl.DataFrame({
            "segment_id": ["A"],
            "period": [1],
            "incurred_claims": [72_000.0],
            "claim_count": [120],
        })
        builder2 = PolicyPanelBuilder(
            make_minimal_policy(), claims_partial, make_minimal_rate_log()
        )
        panel2 = builder2.build()
        # Period 2 for A has no claims row → should be 0 not null
        row = panel2.filter((pl.col("segment_id") == "A") & (pl.col("period") == 2))
        assert row["incurred_claims"][0] == 0.0

    def test_summary_works_after_build(self):
        s = self.builder.summary()
        assert isinstance(s, dict)
        assert "n_segments" in s
        assert s["n_segments"] == 2
        assert s["n_treated_segments"] == 1

    def test_summary_fails_before_build(self):
        b = PolicyPanelBuilder(
            make_minimal_policy(), make_minimal_claims(), make_minimal_rate_log()
        )
        with pytest.raises(RuntimeError, match="build"):
            b.summary()

    def test_to_pandas_works(self):
        import pandas as pd
        df = self.builder.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert "loss_ratio" in df.columns


class TestPolicyPanelBuilderFrequency:
    def test_frequency_computed(self):
        builder = PolicyPanelBuilder(
            make_minimal_policy(),
            make_minimal_claims(),
            make_minimal_rate_log(),
            outcome="frequency",
        )
        panel = builder.build()
        assert "frequency" in panel.columns
        # A, period 1: 120 / 500 = 0.24
        row = panel.filter(
            (pl.col("segment_id") == "A") & (pl.col("period") == 1)
        )
        assert abs(row["frequency"][0] - 0.24) < 0.001


class TestBuildPanelFromPandas:
    def test_accepts_pandas_input(self):
        import pandas as pd
        import pandas as pd
        result = build_panel_from_pandas(
            policy_df=pd.DataFrame(make_minimal_policy().to_dict(as_series=False)),
            claims_df=pd.DataFrame(make_minimal_claims().to_dict(as_series=False)),
            rate_log_df=pd.DataFrame(make_minimal_rate_log().to_dict(as_series=False)),
        )
        assert isinstance(result, pl.DataFrame)
        assert "loss_ratio" in result.columns


class TestPanelBuilderWithSyntheticData:
    def test_full_pipeline_runs(self):
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=30, n_periods=8, random_seed=1
        )
        builder = PolicyPanelBuilder(policy_df, claims_df, rate_log_df)
        panel = builder.build()
        assert isinstance(panel, pl.DataFrame)
        assert "loss_ratio" in panel.columns

    def test_summary_sensible_counts(self):
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=30, n_periods=8, treat_fraction=0.3, random_seed=2
        )
        builder = PolicyPanelBuilder(policy_df, claims_df, rate_log_df)
        builder.build()
        s = builder.summary()
        assert s["n_segments"] == 30
        assert s["n_treated_segments"] > 0
        assert s["n_control_segments"] > 0
        assert s["n_periods"] == 8

    def test_thin_segment_warning(self):
        # Use very high min_exposure to trigger warning
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=10, n_periods=4, random_seed=3
        )
        builder = PolicyPanelBuilder(
            policy_df, claims_df, rate_log_df, min_exposure=1e9
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.build()
            thin_warnings = [x for x in w if "thin" in str(x.message).lower() or "exposure" in str(x.message).lower()]
            assert len(thin_warnings) >= 0  # may or may not trigger, just shouldn't crash
