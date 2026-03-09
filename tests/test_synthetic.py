"""Tests for _synthetic.py data generators."""

import polars as pl
import pytest
from insurance_causal_policy._synthetic import (
    make_synthetic_motor_panel,
    make_synthetic_panel_direct,
)


class TestMakeSyntheticMotorPanel:
    def setup_method(self):
        self.policy_df, self.claims_df, self.rate_log_df = make_synthetic_motor_panel(
            n_segments=50,
            n_periods=10,
            treat_fraction=0.3,
            true_att=-0.08,
            treatment_period=7,
            random_seed=99,
        )

    def test_returns_three_dataframes(self):
        assert isinstance(self.policy_df, pl.DataFrame)
        assert isinstance(self.claims_df, pl.DataFrame)
        assert isinstance(self.rate_log_df, pl.DataFrame)

    def test_policy_has_required_columns(self):
        required = {"segment_id", "period", "earned_premium", "earned_exposure"}
        assert required.issubset(set(self.policy_df.columns))

    def test_claims_has_required_columns(self):
        required = {"segment_id", "period", "incurred_claims", "claim_count"}
        assert required.issubset(set(self.claims_df.columns))

    def test_rate_log_has_required_columns(self):
        required = {"segment_id", "first_treated_period"}
        assert required.issubset(set(self.rate_log_df.columns))

    def test_n_segments_correct(self):
        n = self.policy_df["segment_id"].n_unique()
        assert n == 50

    def test_n_periods_correct(self):
        n = self.policy_df["period"].n_unique()
        assert n == 10

    def test_positive_premiums(self):
        assert self.policy_df["earned_premium"].min() > 0

    def test_positive_exposures(self):
        assert self.policy_df["earned_exposure"].min() > 0

    def test_positive_incurred(self):
        assert self.claims_df["incurred_claims"].min() >= 0

    def test_treat_fraction_approx(self):
        n_total = self.policy_df["segment_id"].n_unique()
        n_treated = self.rate_log_df["segment_id"].n_unique()
        frac = n_treated / n_total
        assert abs(frac - 0.3) < 0.1

    def test_treatment_period_in_rate_log(self):
        periods = self.rate_log_df["first_treated_period"].to_list()
        assert 7 in periods

    def test_reproducibility(self):
        p1, _, _ = make_synthetic_motor_panel(n_segments=20, random_seed=42)
        p2, _, _ = make_synthetic_motor_panel(n_segments=20, random_seed=42)
        assert p1["earned_premium"].to_list() == p2["earned_premium"].to_list()

    def test_different_seeds_differ(self):
        p1, _, _ = make_synthetic_motor_panel(n_segments=20, random_seed=1)
        p2, _, _ = make_synthetic_motor_panel(n_segments=20, random_seed=2)
        assert p1["earned_premium"].to_list() != p2["earned_premium"].to_list()

    def test_staggered_mode(self):
        _, _, rate_log = make_synthetic_motor_panel(
            n_segments=60, n_periods=14,
            staggered=True, n_stagger_cohorts=3,
            treatment_period=8,
            random_seed=42,
        )
        unique_periods = rate_log["first_treated_period"].n_unique()
        assert unique_periods >= 2  # at least 2 different cohort periods


class TestMakeSyntheticPanelDirect:
    def setup_method(self):
        self.panel = make_synthetic_panel_direct(
            n_control=40,
            n_treated=15,
            t_pre=6,
            t_post=3,
            true_att=-0.06,
            random_seed=7,
        )

    def test_returns_polars_dataframe(self):
        assert isinstance(self.panel, pl.DataFrame)

    def test_required_columns(self):
        required = {
            "segment_id", "period", "loss_ratio",
            "earned_premium", "earned_exposure",
            "incurred_claims", "claim_count",
            "first_treated_period", "treated", "cohort",
        }
        assert required.issubset(set(self.panel.columns))

    def test_n_rows(self):
        # (40 + 15) * (6 + 3) = 55 * 9 = 495
        assert len(self.panel) == 55 * 9

    def test_treated_col_binary(self):
        vals = set(self.panel["treated"].to_list())
        assert vals.issubset({0, 1})

    def test_positive_loss_ratios(self):
        assert self.panel["loss_ratio"].min() > 0

    def test_n_treated_segments(self):
        n_tr = self.panel.filter(pl.col("first_treated_period").is_not_null())["segment_id"].n_unique()
        assert n_tr == 15

    def test_n_control_segments(self):
        n_co = self.panel.filter(pl.col("first_treated_period").is_null())["segment_id"].n_unique()
        assert n_co == 40

    def test_true_att_direction(self):
        # Treated post should have lower LR than control trend would predict
        pre_tr = self.panel.filter(
            (pl.col("first_treated_period").is_not_null()) & (pl.col("treated") == 0)
        )["loss_ratio"].mean()
        post_tr = self.panel.filter(
            (pl.col("first_treated_period").is_not_null()) & (pl.col("treated") == 1)
        )["loss_ratio"].mean()
        pre_co = self.panel.filter(
            (pl.col("first_treated_period").is_null()) & (pl.col("period") <= 6)
        )["loss_ratio"].mean()
        post_co = self.panel.filter(
            (pl.col("first_treated_period").is_null()) & (pl.col("period") > 6)
        )["loss_ratio"].mean()

        naive_did = (post_tr - pre_tr) - (post_co - pre_co)
        # Naive DiD should roughly approximate true ATT of -0.06
        assert naive_did < 0  # correct direction
