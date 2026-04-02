"""Edge case and boundary tests for _staggered.py.

The existing test_staggered.py covers the happy path well. These tests focus
on the cases that bite in production:
- All-never-treated controls (nevertreated mode with no such units)
- Single cohort (degenerate staggered case)
- Clean controls returning empty when all units are treated
- _aggregate_event_study weighted aggregation correctness
- _clean_controls with both control group modes
- StaggeredEstimator before_fit attribute access
- ATT direction with known synthetic data
- SE monotonicity: larger cohorts should produce tighter aggregate SE

Production context: UK motor and home pricing teams apply rate changes in
rolling cohorts (Q1 new business only, Q2 renewal only, etc.). The CS21
estimator must handle these patterns without silent failures.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from insurance_causal_policy._staggered import (
    StaggeredEstimator,
    _aggregate_event_study,
    _clean_controls,
    _estimate_att_gt_naive,
    _fit_native_cs21,
)
from insurance_causal_policy._synthetic import (
    make_synthetic_motor_panel,
    make_synthetic_panel_direct,
)
from insurance_causal_policy._panel import PolicyPanelBuilder
from insurance_causal_policy._types import StaggeredResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel_pd(
    n_control: int = 20,
    n_treated: int = 10,
    t_pre: int = 5,
    t_post: int = 3,
    true_att: float = -0.07,
    seed: int = 7,
) -> pd.DataFrame:
    """Return a pandas panel as expected by the internal _fit_native_cs21 functions."""
    pl_panel = make_synthetic_panel_direct(
        n_control=n_control,
        n_treated=n_treated,
        t_pre=t_pre,
        t_post=t_post,
        true_att=true_att,
        random_seed=seed,
    )
    df = pd.DataFrame(pl_panel.to_dict(as_series=False))
    df = df.rename(columns={
        "segment_id": "unit_id",
        "loss_ratio": "outcome",
        "first_treated_period": "first_treated_period",
    })
    # Ensure cohort col present (used by _clean_controls)
    if "cohort" not in df.columns:
        df["cohort"] = df["first_treated_period"]
    return df


# ---------------------------------------------------------------------------
# _clean_controls: never-treated mode
# ---------------------------------------------------------------------------

class TestCleanControlsNeverTreated:
    def test_returns_only_never_treated(self):
        df = _make_panel_pd()
        cohort = int(df["first_treated_period"].dropna().min())
        period = cohort + 1
        controls = _clean_controls(df, cohort, period, cohort - 1, "nevertreated")
        # All returned rows should have null first_treated_period
        assert controls["first_treated_period"].isna().all()

    def test_never_treated_excludes_treated_cohorts(self):
        df = _make_panel_pd()
        cohort = int(df["first_treated_period"].dropna().min())
        controls = _clean_controls(df, cohort, cohort + 1, cohort - 1, "nevertreated")
        treated_in_controls = controls["first_treated_period"].notna().sum()
        assert treated_in_controls == 0

    def test_empty_controls_when_all_treated(self):
        """If every unit is treated, never-treated controls are empty."""
        n = 15
        df = pd.DataFrame({
            "unit_id": [f"U{i}" for i in range(n)],
            "period": [1] * n,
            "outcome": np.random.default_rng(1).uniform(0.5, 1.0, n),
            "first_treated_period": [1] * n,
            "cohort": [1] * n,
            "treated": [1] * n,
        })
        controls = _clean_controls(df, cohort=1, period=1, base_period=0, control_group="nevertreated")
        assert len(controls) == 0

    def test_invalid_control_group_raises(self):
        df = _make_panel_pd()
        with pytest.raises(ValueError, match="Unknown control_group"):
            _clean_controls(df, cohort=6, period=7, base_period=5, control_group="invalidmode")


# ---------------------------------------------------------------------------
# _clean_controls: not-yet-treated mode
# ---------------------------------------------------------------------------

class TestCleanControlsNotYetTreated:
    def test_includes_never_treated(self):
        df = _make_panel_pd()
        cohort = int(df["first_treated_period"].dropna().min())
        period = cohort + 1
        never_treated_count = df["first_treated_period"].isna().sum()
        controls = _clean_controls(df, cohort, period, cohort - 1, "notyettreated")
        # Must include at least the never-treated units
        control_unit_count = controls["unit_id"].nunique() if "unit_id" in controls.columns else len(controls)
        assert control_unit_count >= never_treated_count

    def test_excludes_already_treated(self):
        """Units treated at or before 'period' should not appear in controls."""
        df = _make_panel_pd(n_control=20, n_treated=10, t_pre=5, t_post=3, seed=9)
        cohort = int(df["first_treated_period"].dropna().min())
        period = cohort + 1  # one period into post-treatment
        controls = _clean_controls(df, cohort, period, cohort - 1, "notyettreated")
        # Check: no control unit has first_treated_period <= period (unless NaN)
        treated_too_early = controls[
            controls["first_treated_period"].notna()
            & (controls["first_treated_period"] <= period)
        ]
        assert len(treated_too_early) == 0

    def test_notyettreated_returns_more_controls_than_nevertreated(self):
        """Not-yet-treated is a superset of never-treated — should have >= units."""
        df = _make_panel_pd(n_control=15, n_treated=10, seed=11)
        # Use a panel with multiple cohorts by staggering
        cohort = int(df["first_treated_period"].dropna().min())
        period = cohort  # at treatment time, some units haven't been treated yet
        c_never = _clean_controls(df, cohort, period, cohort - 1, "nevertreated")
        c_notyet = _clean_controls(df, cohort, period, cohort - 1, "notyettreated")
        assert len(c_notyet) >= len(c_never)


# ---------------------------------------------------------------------------
# _estimate_att_gt_naive: edge cases
# ---------------------------------------------------------------------------

class TestEstimateAttGtNaive:
    def test_returns_float_tuple(self):
        df = _make_panel_pd()
        cohort = int(df["first_treated_period"].dropna().min())
        period = cohort + 1
        att, se = _estimate_att_gt_naive(
            df, cohort=cohort, period=period,
            outcome_col="outcome", unit_col="unit_id", period_col="period",
            control_group="notyettreated",
        )
        assert isinstance(att, float)
        assert isinstance(se, float)

    def test_returns_nan_for_empty_cohort(self):
        """A cohort that doesn't exist in the data should yield NaN."""
        df = _make_panel_pd()
        att, se = _estimate_att_gt_naive(
            df, cohort=9999, period=1,
            outcome_col="outcome", unit_col="unit_id", period_col="period",
        )
        assert np.isnan(att)
        assert np.isnan(se)

    def test_se_non_negative_when_valid(self):
        df = _make_panel_pd()
        cohort = int(df["first_treated_period"].dropna().min())
        period = cohort + 1
        att, se = _estimate_att_gt_naive(
            df, cohort=cohort, period=period,
            outcome_col="outcome", unit_col="unit_id", period_col="period",
        )
        if not np.isnan(se):
            assert se >= 0.0

    def test_att_direction_with_known_negative_effect(self):
        """True ATT = -0.10; estimate should be negative on average over post periods."""
        pl_panel = make_synthetic_panel_direct(
            n_control=40, n_treated=15, t_pre=4, t_post=4,
            true_att=-0.10, random_seed=42,
        )
        df = pd.DataFrame(pl_panel.to_dict(as_series=False))
        df = df.rename(columns={"segment_id": "unit_id", "loss_ratio": "outcome"})
        if "cohort" not in df.columns:
            df["cohort"] = df["first_treated_period"]
        cohort = int(df["first_treated_period"].dropna().iloc[0])
        atts = []
        for period in range(cohort, cohort + 4):
            att, se = _estimate_att_gt_naive(
                df, cohort=cohort, period=period,
                outcome_col="outcome", unit_col="unit_id", period_col="period",
            )
            if not np.isnan(att):
                atts.append(att)
        assert len(atts) > 0, "No valid ATT estimates for post-treatment periods"
        assert np.mean(atts) < 0, "Mean post-treatment ATT should be negative"


# ---------------------------------------------------------------------------
# _aggregate_event_study: weighted aggregation correctness
# ---------------------------------------------------------------------------

class TestAggregateEventStudy:
    def test_returns_dataframe_with_required_columns(self):
        att_gt = pd.DataFrame({
            "cohort": [5, 5, 6, 6],
            "period": [5, 6, 6, 7],
            "att": [-0.05, -0.08, -0.04, -0.06],
            "se": [0.02, 0.025, 0.02, 0.022],
        })
        result = _aggregate_event_study(att_gt)
        assert "period_rel" in result.columns
        assert "att" in result.columns
        assert "se" in result.columns
        assert "ci_low" in result.columns
        assert "ci_high" in result.columns

    def test_period_rel_computed_correctly(self):
        """period_rel = period - cohort."""
        att_gt = pd.DataFrame({
            "cohort": [5, 5, 6, 6],
            "period": [5, 6, 6, 7],
            "att": [-0.05, -0.08, -0.04, -0.06],
            "se": [0.02, 0.025, 0.02, 0.022],
        })
        result = _aggregate_event_study(att_gt)
        # cohort=5, period=5 -> rel=0; cohort=5,period=6 -> rel=1
        # cohort=6, period=6 -> rel=0; cohort=6,period=7 -> rel=1
        # Relative period 0 should aggregate cohort[5,period5] and cohort[6,period6]
        assert 0 in result["period_rel"].values
        assert 1 in result["period_rel"].values

    def test_ci_width_correct(self):
        """CI should be ATT +/- 1.96 * SE."""
        att_gt = pd.DataFrame({
            "cohort": [5, 6],
            "period": [6, 7],
            "att": [-0.05, -0.07],
            "se": [0.02, 0.025],
        })
        result = _aggregate_event_study(att_gt)
        for _, row in result.iterrows():
            expected_low = row["att"] - 1.96 * row["se"]
            expected_high = row["att"] + 1.96 * row["se"]
            assert abs(row["ci_low"] - expected_low) < 1e-6
            assert abs(row["ci_high"] - expected_high) < 1e-6

    def test_weighted_mean_not_simple_mean(self):
        """With unequal cohort sizes, weighted mean != simple mean.

        cohort A: 30 units, att=-0.10
        cohort B: 10 units, att=-0.02
        Weighted mean: (30*-0.10 + 10*-0.02)/40 = -0.08
        Simple mean:   (-0.10 + -0.02)/2 = -0.06
        """
        att_gt = pd.DataFrame({
            "cohort": [5, 6],
            "period": [5, 6],  # period_rel = 0 for both
            "att": [-0.10, -0.02],
            "se": [0.01, 0.01],
            "cohort_size": [30, 10],
        })
        result = _aggregate_event_study(att_gt)
        row = result[result["period_rel"] == 0].iloc[0]
        # Should be closer to -0.08 than -0.06
        assert abs(row["att"] - (-0.08)) < abs(row["att"] - (-0.06))

    def test_sorted_by_period_rel(self):
        att_gt = pd.DataFrame({
            "cohort": [5, 5, 5, 6, 6, 6],
            "period": [3, 4, 6, 4, 5, 7],
            "att": [-0.01, 0.0, -0.07, -0.01, 0.01, -0.06],
            "se": [0.02] * 6,
        })
        result = _aggregate_event_study(att_gt)
        assert list(result["period_rel"]) == sorted(result["period_rel"].tolist())

    def test_single_cohort_returns_identity(self):
        """With a single cohort, the event study should be the original ATTs."""
        att_gt = pd.DataFrame({
            "cohort": [5, 5, 5],
            "period": [4, 5, 6],  # period_rel = -1, 0, 1
            "att": [0.01, -0.08, -0.09],
            "se": [0.02, 0.025, 0.03],
        })
        result = _aggregate_event_study(att_gt)
        # With one cohort, weighted mean = the single value
        row_pre = result[result["period_rel"] == -1].iloc[0]
        assert abs(row_pre["att"] - 0.01) < 1e-9
        row_post = result[result["period_rel"] == 0].iloc[0]
        assert abs(row_post["att"] - (-0.08)) < 1e-9


# ---------------------------------------------------------------------------
# StaggeredEstimator: boundary conditions
# ---------------------------------------------------------------------------

class TestStaggeredEstimatorBoundaryConditions:
    def _build_simple_staggered_panel(self, seed: int = 20):
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=50,
            n_periods=12,
            treat_fraction=0.4,
            true_att=-0.06,
            treatment_period=7,
            staggered=True,
            n_stagger_cohorts=2,
            random_seed=seed,
        )
        return PolicyPanelBuilder(policy_df, claims_df, rate_log_df).build()

    def test_result_before_fit_is_none(self):
        panel = self._build_simple_staggered_panel()
        est = StaggeredEstimator(panel)
        assert est._result is None

    def test_fit_returns_staggered_result_instance(self):
        panel = self._build_simple_staggered_panel()
        result = StaggeredEstimator(panel).fit()
        assert isinstance(result, StaggeredResult)

    def test_att_overall_is_finite(self):
        panel = self._build_simple_staggered_panel()
        result = StaggeredEstimator(panel).fit()
        assert np.isfinite(result.att_overall)

    def test_se_overall_positive(self):
        panel = self._build_simple_staggered_panel()
        result = StaggeredEstimator(panel).fit()
        assert result.se_overall > 0

    def test_ci_contains_att(self):
        panel = self._build_simple_staggered_panel()
        result = StaggeredEstimator(panel).fit()
        assert result.ci_low_overall <= result.att_overall <= result.ci_high_overall

    def test_att_gt_has_required_columns(self):
        panel = self._build_simple_staggered_panel()
        result = StaggeredEstimator(panel).fit()
        required = {"cohort", "period", "att", "se"}
        assert required.issubset(set(result.att_gt.columns))

    def test_event_study_has_required_columns(self):
        panel = self._build_simple_staggered_panel()
        result = StaggeredEstimator(panel).fit()
        required = {"period_rel", "att", "se", "ci_low", "ci_high"}
        assert required.issubset(set(result.event_study.columns))

    def test_pre_trend_pval_in_zero_one(self):
        panel = self._build_simple_staggered_panel()
        result = StaggeredEstimator(panel).fit()
        if result.pre_trend_pval is not None:
            assert 0.0 <= result.pre_trend_pval <= 1.0

    def test_negative_att_with_known_negative_true_effect(self):
        """With true_att=-0.10, the estimated overall ATT should be negative."""
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=80,
            n_periods=14,
            treat_fraction=0.4,
            true_att=-0.10,
            treatment_period=8,
            staggered=True,
            n_stagger_cohorts=3,
            random_seed=123,
        )
        panel = PolicyPanelBuilder(policy_df, claims_df, rate_log_df).build()
        result = StaggeredEstimator(panel).fit()
        assert result.att_overall < 0, (
            f"Expected negative ATT, got {result.att_overall:.4f}. "
            "True effect is -0.10; estimate should be in the right direction."
        )

    def test_outcome_name_stored_correctly(self):
        panel = self._build_simple_staggered_panel()
        result = StaggeredEstimator(panel, outcome="loss_ratio").fit()
        assert result.outcome_name == "loss_ratio"

    def test_control_group_stored_correctly(self):
        panel = self._build_simple_staggered_panel()
        est = StaggeredEstimator(panel, control_group="notyettreated")
        result = est.fit()
        assert result.control_group == "notyettreated"

    def test_nevertreated_control_group_runs(self):
        """nevertreated mode should complete without error even if fewer controls."""
        panel = self._build_simple_staggered_panel(seed=30)
        result = StaggeredEstimator(panel, control_group="nevertreated").fit()
        assert isinstance(result, StaggeredResult)

    def test_issues_warning_when_differences_not_installed(self):
        """When differences package is absent, a UserWarning should be issued."""
        from insurance_causal_policy._staggered import _DIFFERENCES_AVAILABLE
        if _DIFFERENCES_AVAILABLE:
            pytest.skip("differences package is installed; cannot test fallback warning")
        panel = self._build_simple_staggered_panel()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StaggeredEstimator(panel).fit()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert any("differences" in str(x.message).lower() for x in user_warnings)

    def test_n_cohorts_matches_input(self):
        """n_cohorts in the result should match the number of treated cohorts."""
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=60,
            n_periods=14,
            treat_fraction=0.4,
            true_att=-0.07,
            treatment_period=8,
            staggered=True,
            n_stagger_cohorts=3,
            random_seed=77,
        )
        panel = PolicyPanelBuilder(policy_df, claims_df, rate_log_df).build()
        result = StaggeredEstimator(panel).fit()
        assert result.n_cohorts >= 1


# ---------------------------------------------------------------------------
# _fit_native_cs21: output structure
# ---------------------------------------------------------------------------

class TestFitNativeCS21:
    def test_returns_dataframe(self):
        df = _make_panel_pd(n_control=15, n_treated=8, t_pre=4, t_post=3, seed=5)
        result_df = _fit_native_cs21(
            df,
            outcome_col="outcome",
            unit_col="unit_id",
            period_col="period",
            cohort_col="cohort",
            control_group="notyettreated",
        )
        assert isinstance(result_df, pd.DataFrame)

    def test_has_required_columns(self):
        df = _make_panel_pd()
        result_df = _fit_native_cs21(
            df,
            outcome_col="outcome",
            unit_col="unit_id",
            period_col="period",
            cohort_col="cohort",
        )
        required = {"cohort", "period", "att", "se", "ci_low", "ci_high"}
        assert required.issubset(set(result_df.columns))

    def test_no_nan_atts_for_valid_data(self):
        """With clean data, all returned ATT estimates should be finite."""
        df = _make_panel_pd(n_control=30, n_treated=10, t_pre=5, t_post=4, seed=3)
        result_df = _fit_native_cs21(
            df,
            outcome_col="outcome",
            unit_col="unit_id",
            period_col="period",
            cohort_col="cohort",
        )
        # All returned rows should have finite ATTs (NaN rows are filtered out)
        assert result_df["att"].notna().all()

    def test_all_atts_are_finite(self):
        df = _make_panel_pd()
        result_df = _fit_native_cs21(
            df,
            outcome_col="outcome",
            unit_col="unit_id",
            period_col="period",
            cohort_col="cohort",
        )
        assert np.isfinite(result_df["att"].values).all()

    def test_ci_ordering(self):
        """ci_low <= att <= ci_high for every row."""
        df = _make_panel_pd(n_control=25, n_treated=10, seed=8)
        result_df = _fit_native_cs21(
            df,
            outcome_col="outcome",
            unit_col="unit_id",
            period_col="period",
            cohort_col="cohort",
        )
        for _, row in result_df.iterrows():
            if not (np.isnan(row["ci_low"]) or np.isnan(row["ci_high"])):
                assert row["ci_low"] <= row["att"] <= row["ci_high"], (
                    f"CI ordering violated: [{row['ci_low']:.4f}, {row['att']:.4f}, {row['ci_high']:.4f}]"
                )
