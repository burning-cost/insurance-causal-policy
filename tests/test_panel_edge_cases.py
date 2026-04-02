"""Edge case tests for _panel.py PolicyPanelBuilder.

The existing test_panel.py covers the standard pipeline. These tests focus on
boundary conditions and failure modes that occur in real insurance data:

- Segments that exist in the policy table but have zero claims for all periods
- Zero-premium periods (mid-term cancellation runs, book acquisitions)
- paid_only=True mode and its interaction with incurred vs paid
- rate_log_df with segment IDs that don't match policy_df (no-match warning)
- Retention outcome computation
- Multiple segments with different treatment periods (cohort column correctness)
- High null-outcome rate warning threshold (>20%)
- summary() correctness after edge-case builds
- All-control panel (no treated segments) triggers warning

Production context: UK pricing teams routinely encounter these patterns when
building causal evaluation panels from raw policy admin extracts.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

from insurance_causal_policy._panel import PolicyPanelBuilder, build_panel_from_pandas
from insurance_causal_policy._synthetic import make_synthetic_motor_panel


# ---------------------------------------------------------------------------
# Minimal data factories
# ---------------------------------------------------------------------------

def _policy(segments=("A", "B"), periods=(1, 2, 3)):
    rows = []
    for seg in segments:
        for p in periods:
            rows.append({
                "segment_id": seg,
                "period": p,
                "earned_premium": 100_000.0,
                "earned_exposure": 500.0,
            })
    return pl.DataFrame(rows)


def _claims(segments=("A", "B"), periods=(1, 2, 3), lr=0.65):
    rows = []
    for seg in segments:
        for p in periods:
            rows.append({
                "segment_id": seg,
                "period": p,
                "incurred_claims": 100_000.0 * lr,
                "claim_count": 100,
            })
    return pl.DataFrame(rows)


def _rate_log(segment="A", first_treated_period=2):
    return pl.DataFrame({
        "segment_id": [segment],
        "first_treated_period": [first_treated_period],
    })


# ---------------------------------------------------------------------------
# Zero-claims segments
# ---------------------------------------------------------------------------

class TestZeroClaimsSegments:
    def test_segment_with_no_claims_rows_gets_zero_not_null(self):
        """A segment absent from claims_df entirely should have 0 incurred, not null."""
        policy = _policy(segments=("A", "B"), periods=(1, 2))
        claims = _claims(segments=("A",), periods=(1, 2))  # B has no claims
        rate_log = _rate_log("A", 2)
        panel = PolicyPanelBuilder(policy, claims, rate_log).build()
        b_rows = panel.filter(pl.col("segment_id") == "B")
        assert b_rows["incurred_claims"].sum() == 0.0
        assert b_rows["incurred_claims"].null_count() == 0

    def test_zero_claim_count_for_missing_claims(self):
        policy = _policy(segments=("A", "B"), periods=(1, 2))
        claims = _claims(segments=("A",), periods=(1, 2))
        rate_log = _rate_log("A", 2)
        panel = PolicyPanelBuilder(policy, claims, rate_log).build()
        b_rows = panel.filter(pl.col("segment_id") == "B")
        assert b_rows["claim_count"].sum() == 0

    def test_zero_claims_produces_zero_loss_ratio(self):
        """Zero claims / positive premium = 0.0 loss ratio, not null."""
        policy = _policy(segments=("A",), periods=(1, 2, 3))
        claims_none = pl.DataFrame({
            "segment_id": pl.Series([], dtype=pl.String),
            "period": pl.Series([], dtype=pl.Int64),
            "incurred_claims": pl.Series([], dtype=pl.Float64),
            "claim_count": pl.Series([], dtype=pl.Int32),
        })
        rate_log = pl.DataFrame({
            "segment_id": pl.Series([], dtype=pl.String),
            "first_treated_period": pl.Series([], dtype=pl.Int64),
        })
        panel = PolicyPanelBuilder(policy, claims_none, rate_log).build()
        assert panel["loss_ratio"].null_count() == 0
        assert (panel["loss_ratio"] == 0.0).all()


# ---------------------------------------------------------------------------
# Zero-premium cells (division-by-zero handling)
# ---------------------------------------------------------------------------

class TestZeroPremiumHandling:
    def test_zero_premium_produces_null_loss_ratio(self):
        """Division by zero premium must yield null, not inf or NaN."""
        policy = pl.DataFrame({
            "segment_id": ["A", "A"],
            "period": [1, 2],
            "earned_premium": [0.0, 100_000.0],  # period 1 has zero premium
            "earned_exposure": [0.0, 500.0],
        })
        claims = _claims(segments=("A",), periods=(1, 2))
        rate_log = _rate_log("A", 2)
        panel = PolicyPanelBuilder(policy, claims, rate_log).build()
        row_p1 = panel.filter((pl.col("segment_id") == "A") & (pl.col("period") == 1))
        assert row_p1["loss_ratio"][0] is None, "Zero premium should yield null LR"

    def test_nonzero_premium_loss_ratio_correct(self):
        """Adjacent period with premium should still have a valid LR."""
        policy = pl.DataFrame({
            "segment_id": ["A", "A"],
            "period": [1, 2],
            "earned_premium": [0.0, 80_000.0],
            "earned_exposure": [0.0, 400.0],
        })
        claims = pl.DataFrame({
            "segment_id": ["A"],
            "period": [2],
            "incurred_claims": [56_000.0],
            "claim_count": [90],
        })
        rate_log = pl.DataFrame({
            "segment_id": pl.Series([], dtype=pl.String),
            "first_treated_period": pl.Series([], dtype=pl.Int64),
        })
        panel = PolicyPanelBuilder(policy, claims, rate_log).build()
        row_p2 = panel.filter((pl.col("segment_id") == "A") & (pl.col("period") == 2))
        assert abs(row_p2["loss_ratio"][0] - 0.70) < 0.001

    def test_high_null_outcome_rate_warns(self):
        """If >20% of cells have null loss_ratio, a warning should fire."""
        # Create a policy with many zero-premium cells
        rows_policy = []
        for i in range(20):
            for p in range(1, 4):
                premium = 0.0 if p == 1 else 100_000.0  # period 1 always zero
                rows_policy.append({
                    "segment_id": f"S{i}",
                    "period": p,
                    "earned_premium": premium,
                    "earned_exposure": max(0.0, premium / 200),
                })
        policy = pl.DataFrame(rows_policy)
        claims = pl.DataFrame({
            "segment_id": pl.Series([], dtype=pl.String),
            "period": pl.Series([], dtype=pl.Int64),
            "incurred_claims": pl.Series([], dtype=pl.Float64),
            "claim_count": pl.Series([], dtype=pl.Int32),
        })
        rate_log = pl.DataFrame({
            "segment_id": pl.Series([], dtype=pl.String),
            "first_treated_period": pl.Series([], dtype=pl.Int64),
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PolicyPanelBuilder(policy, claims, rate_log).build()
        # 1/3 of cells have zero premium -> ~33% null -> should warn
        outcome_warnings = [
            x for x in w if "null" in str(x.message).lower() or "%" in str(x.message)
        ]
        assert len(outcome_warnings) >= 1


# ---------------------------------------------------------------------------
# paid_only mode
# ---------------------------------------------------------------------------

class TestPaidOnlyMode:
    def _make_paid_claims(self):
        return pl.DataFrame({
            "segment_id": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "incurred_claims": [70_000.0, 75_000.0, 55_000.0, 60_000.0],
            "paid_claims": [50_000.0, 55_000.0, 40_000.0, 45_000.0],
            "claim_count": [100, 110, 80, 85],
        })

    def test_paid_only_uses_paid_claims_column(self):
        policy = _policy(segments=("A", "B"), periods=(1, 2))
        claims = self._make_paid_claims()
        rate_log = _rate_log("A", 2)
        panel = PolicyPanelBuilder(policy, claims, rate_log, paid_only=True).build()
        row = panel.filter((pl.col("segment_id") == "A") & (pl.col("period") == 1))
        # paid_claims=50_000, premium=100_000 -> LR = 0.50
        assert abs(row["loss_ratio"][0] - 0.50) < 0.001

    def test_paid_only_false_uses_incurred(self):
        policy = _policy(segments=("A", "B"), periods=(1, 2))
        claims = self._make_paid_claims()
        rate_log = _rate_log("A", 2)
        panel = PolicyPanelBuilder(policy, claims, rate_log, paid_only=False).build()
        row = panel.filter((pl.col("segment_id") == "A") & (pl.col("period") == 1))
        # incurred=70_000, premium=100_000 -> LR = 0.70
        assert abs(row["loss_ratio"][0] - 0.70) < 0.001

    def test_paid_only_without_paid_column_raises(self):
        policy = _policy(segments=("A",), periods=(1, 2))
        claims = _claims(segments=("A",), periods=(1, 2))  # no paid_claims col
        rate_log = _rate_log("A", 2)
        with pytest.raises(ValueError, match="paid_claims"):
            PolicyPanelBuilder(policy, claims, rate_log, paid_only=True)

    def test_paid_loss_ratio_lower_than_incurred(self):
        """Paid < incurred always (IBNR reserve). Paid LR should be lower."""
        policy = _policy(segments=("A",), periods=(1, 2))
        claims = self._make_paid_claims().filter(pl.col("segment_id") == "A")
        rate_log = pl.DataFrame({
            "segment_id": pl.Series([], dtype=pl.String),
            "first_treated_period": pl.Series([], dtype=pl.Int64),
        })
        panel_paid = PolicyPanelBuilder(policy, claims, rate_log, paid_only=True).build()
        panel_incurred = PolicyPanelBuilder(policy, claims, rate_log, paid_only=False).build()
        avg_paid_lr = panel_paid["loss_ratio"].mean()
        avg_incurred_lr = panel_incurred["loss_ratio"].mean()
        assert avg_paid_lr < avg_incurred_lr


# ---------------------------------------------------------------------------
# No-match treatment segments (rate_log has unknown segment IDs)
# ---------------------------------------------------------------------------

class TestNoMatchTreatmentSegments:
    def test_rate_log_mismatch_warns(self):
        """If no policy segment matches the rate log, a warning should fire."""
        policy = _policy(segments=("A", "B"), periods=(1, 2))
        claims = _claims(segments=("A", "B"), periods=(1, 2))
        rate_log = pl.DataFrame({
            "segment_id": ["NONEXISTENT"],
            "first_treated_period": [2],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PolicyPanelBuilder(policy, claims, rate_log).build()
        # Should warn: no treated segments found
        treatment_warnings = [
            x for x in w if "treated" in str(x.message).lower()
        ]
        assert len(treatment_warnings) >= 1

    def test_all_segments_untreated_when_no_match(self):
        policy = _policy(segments=("A", "B"), periods=(1, 2))
        claims = _claims(segments=("A", "B"), periods=(1, 2))
        rate_log = pl.DataFrame({
            "segment_id": ["UNKNOWN_SEG"],
            "first_treated_period": [2],
        })
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            panel = PolicyPanelBuilder(policy, claims, rate_log).build()
        assert (panel["treated"] == 0).all()
        assert panel["first_treated_period"].null_count() == panel.height


# ---------------------------------------------------------------------------
# Cohort column correctness with multiple treatment cohorts
# ---------------------------------------------------------------------------

class TestCohortColumnWithMultipleCohorts:
    def test_different_treatment_times_stored_in_cohort(self):
        policy = _policy(segments=("A", "B", "C"), periods=(1, 2, 3, 4))
        claims = _claims(segments=("A", "B", "C"), periods=(1, 2, 3, 4))
        rate_log = pl.DataFrame({
            "segment_id": ["A", "B"],
            "first_treated_period": [2, 3],
        })
        panel = PolicyPanelBuilder(policy, claims, rate_log).build()
        seg_a = panel.filter(pl.col("segment_id") == "A")["cohort"].drop_nulls().to_list()
        seg_b = panel.filter(pl.col("segment_id") == "B")["cohort"].drop_nulls().to_list()
        seg_c = panel.filter(pl.col("segment_id") == "C")["cohort"].drop_nulls().to_list()
        assert all(c == 2 for c in seg_a)
        assert all(c == 3 for c in seg_b)
        assert len(seg_c) == 0  # never treated

    def test_treated_indicator_respects_cohort(self):
        """A segment treated at period 3 should have treated=0 for periods 1,2."""
        policy = _policy(segments=("A",), periods=(1, 2, 3, 4))
        claims = _claims(segments=("A",), periods=(1, 2, 3, 4))
        rate_log = pl.DataFrame({
            "segment_id": ["A"],
            "first_treated_period": [3],
        })
        panel = PolicyPanelBuilder(policy, claims, rate_log).build()
        pre_rows = panel.filter(pl.col("period") < 3)
        post_rows = panel.filter(pl.col("period") >= 3)
        assert (pre_rows["treated"] == 0).all()
        assert (post_rows["treated"] == 1).all()


# ---------------------------------------------------------------------------
# Retention outcome
# ---------------------------------------------------------------------------

class TestRetentionOutcome:
    def _policy_with_retention(self):
        return pl.DataFrame({
            "segment_id": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "earned_premium": [100_000.0, 110_000.0, 80_000.0, 85_000.0],
            "earned_exposure": [500.0, 520.0, 400.0, 410.0],
            "policies_renewed": [450, 480, 360, 370],
            "policies_due": [500, 500, 400, 400],
        })

    def test_retention_outcome_computed(self):
        policy = self._policy_with_retention()
        claims = _claims(segments=("A", "B"), periods=(1, 2))
        rate_log = _rate_log("A", 2)
        panel = PolicyPanelBuilder(
            policy, claims, rate_log, outcome="retention"
        ).build()
        assert "retention" in panel.columns
        row = panel.filter((pl.col("segment_id") == "A") & (pl.col("period") == 1))
        # 450/500 = 0.90
        assert abs(row["retention"][0] - 0.90) < 0.001

    def test_retention_without_renewal_columns_raises(self):
        policy = _policy(segments=("A",), periods=(1, 2))
        claims = _claims(segments=("A",), periods=(1, 2))
        rate_log = _rate_log("A", 2)
        with pytest.raises(ValueError, match="policies_renewed"):
            PolicyPanelBuilder(policy, claims, rate_log, outcome="retention")


# ---------------------------------------------------------------------------
# summary() edge cases
# ---------------------------------------------------------------------------

class TestSummaryEdgeCases:
    def test_summary_pct_nonzero_exposure_correct(self):
        """With some zero-exposure cells, pct_nonzero_exposure should be < 100."""
        policy = pl.DataFrame({
            "segment_id": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "earned_premium": [0.0, 100_000.0, 80_000.0, 85_000.0],
            "earned_exposure": [0.0, 500.0, 400.0, 410.0],
        })
        claims = _claims(segments=("A", "B"), periods=(1, 2))
        rate_log = _rate_log("A", 2)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            builder = PolicyPanelBuilder(policy, claims, rate_log)
            builder.build()
            s = builder.summary()
        # 1 of 4 cells has zero exposure
        assert s["pct_nonzero_exposure"] < 100.0

    def test_summary_pct_treated_cells_correct(self):
        """pct_treated_cells should reflect the fraction of post-treatment rows."""
        policy = _policy(segments=("A", "B"), periods=(1, 2, 3, 4))
        claims = _claims(segments=("A", "B"), periods=(1, 2, 3, 4))
        # A treated at period 3: post-treatment cells = periods 3,4 = 2/8 = 25%
        rate_log = pl.DataFrame({
            "segment_id": ["A"],
            "first_treated_period": [3],
        })
        panel = PolicyPanelBuilder(policy, claims, rate_log).build()
        builder = PolicyPanelBuilder(policy, claims, rate_log)
        builder.build()
        s = builder.summary()
        assert s["pct_treated_cells"] == pytest.approx(25.0, abs=0.1)


# ---------------------------------------------------------------------------
# build_panel_from_pandas: pandas round-trip
# ---------------------------------------------------------------------------

class TestBuildPanelFromPandas:
    def test_produces_same_result_as_polars_path(self):
        import pandas as pd
        policy = _policy(segments=("A", "B"), periods=(1, 2, 3))
        claims = _claims(segments=("A", "B"), periods=(1, 2, 3))
        rate_log = _rate_log("A", 2)

        panel_polars = PolicyPanelBuilder(policy, claims, rate_log).build()
        panel_pandas = build_panel_from_pandas(
            policy_df=pd.DataFrame(policy.to_dict(as_series=False)),
            claims_df=pd.DataFrame(claims.to_dict(as_series=False)),
            rate_log_df=pd.DataFrame(rate_log.to_dict(as_series=False)),
        )
        # Both paths should produce the same shape
        assert panel_pandas.shape == panel_polars.shape

    def test_pandas_path_produces_polars_dataframe(self):
        import pandas as pd
        policy = _policy()
        claims = _claims()
        rate_log = _rate_log()
        result = build_panel_from_pandas(
            policy_df=pd.DataFrame(policy.to_dict(as_series=False)),
            claims_df=pd.DataFrame(claims.to_dict(as_series=False)),
            rate_log_df=pd.DataFrame(rate_log.to_dict(as_series=False)),
        )
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# Synthetic data integration: full pipeline edge cases
# ---------------------------------------------------------------------------

class TestFullPipelineEdgeCases:
    def test_single_period_post_treatment(self):
        """Only one post-treatment period is valid for SDID (though weak)."""
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=20, n_periods=5, treat_fraction=0.3,
            true_att=-0.05, treatment_period=5, random_seed=50,
        )
        panel = PolicyPanelBuilder(policy_df, claims_df, rate_log_df).build()
        assert "loss_ratio" in panel.columns
        assert panel.height > 0

    def test_all_segments_treated_still_builds(self):
        """Even if every segment is treated, the panel should build (no SDID
        is possible, but the panel itself is valid)."""
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=10, n_periods=6, treat_fraction=1.0,
            true_att=-0.05, treatment_period=4, random_seed=51,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            panel = PolicyPanelBuilder(policy_df, claims_df, rate_log_df).build()
        assert panel.height > 0

    def test_very_large_book_builds_in_reasonable_shape(self):
        """A 500-segment panel should produce a correctly sized output."""
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=500, n_periods=8, treat_fraction=0.3,
            true_att=-0.06, treatment_period=5, random_seed=52,
        )
        panel = PolicyPanelBuilder(policy_df, claims_df, rate_log_df).build()
        assert panel["segment_id"].n_unique() == 500
        assert panel["period"].n_unique() == 8
        assert panel.height == 500 * 8
