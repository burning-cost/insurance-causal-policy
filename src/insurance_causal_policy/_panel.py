"""
PolicyPanelBuilder: convert insurance policy/claims tables into balanced panels.

The SDID estimator requires a balanced segment × period panel. Raw insurance
data is policy-level and inherently unbalanced — policies start and end,
exposures vary, and some segment-period cells have zero claims. This module
handles the aggregation and balancing step.

Design notes:
- Polars throughout for speed. Motor books with 2M+ policies are slow in pandas.
- Exposure weighting is central: loss ratio = incurred / earned_premium, not the
  mean of individual policy loss ratios. This matters for SDID weights.
- Treatment indicators are joined from a rate change log (segment → first treated
  period). Missing means never-treated.
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------

POLICY_REQUIRED_COLS = {"segment_id", "period", "earned_premium", "earned_exposure"}
CLAIMS_REQUIRED_COLS = {"segment_id", "period", "incurred_claims", "claim_count"}
RATE_LOG_REQUIRED_COLS = {"segment_id", "first_treated_period"}


def _check_cols(df: pl.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------


class PolicyPanelBuilder:
    """Build a balanced segment × period panel from raw insurance tables.

    The panel is the input to SDIDEstimator and StaggeredEstimator. Each row
    represents one segment in one period with the aggregated outcome metric
    and the treatment indicator.

    Parameters
    ----------
    policy_df : polars DataFrame
        Policy-level table. Required columns:
          - segment_id : str — unique segment identifier
          - period : int — period index (e.g., YYYYQQ or sequential integer)
          - earned_premium : float — premium earned in this period (prorated)
          - earned_exposure : float — policy years (or any exposure measure)
        Optional columns included as segment-level covariates if present.
    claims_df : polars DataFrame
        Claims table aggregated to segment × period. Required columns:
          - segment_id : str
          - period : int — accident period matching policy_df.period
          - incurred_claims : float — paid + IBNR reserve
          - claim_count : int
        Segments/periods absent from claims_df are treated as zero claims.
    rate_log_df : polars DataFrame
        Rate change log. Required columns:
          - segment_id : str
          - first_treated_period : int — period when rate change first applied
        Segments absent from rate_log_df are classified as never-treated
        (first_treated_period = None).
    outcome : str
        Outcome metric to compute. One of:
          'loss_ratio'  — incurred_claims / earned_premium
          'frequency'   — claim_count / earned_exposure
          'retention'   — requires policies_renewed and policies_due columns
        Default: 'loss_ratio'.
    exposure_col : str
        Column from policy_df to use as exposure weight in the panel. Used
        for RC-SDiD exposure weighting. Default: 'earned_premium'.
    min_exposure : float
        Minimum exposure per segment-period cell. Cells below this threshold
        are flagged as thin. A warning is issued if any thin cells exist.
        Default: 50.0.
    paid_only : bool
        If True, use paid_claims instead of incurred_claims for loss ratio.
        Reduces IBNR bias for recent periods. Requires 'paid_claims' column
        in claims_df. Default: False.
    """

    def __init__(
        self,
        policy_df: pl.DataFrame,
        claims_df: pl.DataFrame,
        rate_log_df: pl.DataFrame,
        outcome: str = "loss_ratio",
        exposure_col: str = "earned_premium",
        min_exposure: float = 50.0,
        paid_only: bool = False,
    ) -> None:
        _check_cols(policy_df, POLICY_REQUIRED_COLS, "policy_df")
        _check_cols(claims_df, CLAIMS_REQUIRED_COLS, "claims_df")
        _check_cols(rate_log_df, RATE_LOG_REQUIRED_COLS, "rate_log_df")

        valid_outcomes = {"loss_ratio", "frequency", "retention"}
        if outcome not in valid_outcomes:
            raise ValueError(f"outcome must be one of {valid_outcomes}, got '{outcome}'")

        if outcome == "retention" and (
            "policies_renewed" not in policy_df.columns
            or "policies_due" not in policy_df.columns
        ):
            raise ValueError(
                "outcome='retention' requires 'policies_renewed' and 'policies_due' "
                "columns in policy_df."
            )

        if paid_only and "paid_claims" not in claims_df.columns:
            raise ValueError(
                "paid_only=True requires 'paid_claims' column in claims_df."
            )

        self.policy_df = policy_df
        self.claims_df = claims_df
        self.rate_log_df = rate_log_df
        self.outcome = outcome
        self.exposure_col = exposure_col
        self.min_exposure = min_exposure
        self.paid_only = paid_only
        self._panel: Optional[pl.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self) -> pl.DataFrame:
        """Build and return the balanced panel.

        Returns a Polars DataFrame with columns:
          - segment_id : str
          - period : int
          - earned_premium : float
          - earned_exposure : float
          - incurred_claims : float
          - claim_count : int
          - {outcome} : float — the computed outcome metric
          - first_treated_period : int or null — null = never treated
          - treated : int — 1 if period >= first_treated_period, else 0
          - cohort : int or null — alias for first_treated_period
        """
        panel = self._aggregate_policy()
        panel = self._join_claims(panel)
        panel = self._compute_outcome(panel)
        panel = self._join_treatment(panel)
        panel = self._balance_panel(panel)
        panel = self._validate(panel)
        self._panel = panel
        return panel

    def summary(self) -> dict:
        """Return summary statistics about the panel.

        Raises RuntimeError if build() has not been called.
        """
        if self._panel is None:
            raise RuntimeError("Call build() before summary().")
        p = self._panel
        n_segments = p["segment_id"].n_unique()
        n_periods = p["period"].n_unique()
        n_treated_segs = p.filter(pl.col("first_treated_period").is_not_null())["segment_id"].n_unique()
        n_cells = len(p)
        n_cells_nonzero = p.filter(pl.col("earned_exposure") > 0).height
        pct_treated = 100 * p.filter(pl.col("treated") == 1).height / n_cells
        return {
            "n_segments": n_segments,
            "n_treated_segments": n_treated_segs,
            "n_control_segments": n_segments - n_treated_segs,
            "n_periods": n_periods,
            "n_cells": n_cells,
            "pct_nonzero_exposure": 100 * n_cells_nonzero / n_cells,
            "pct_treated_cells": round(pct_treated, 1),
            "outcome": self.outcome,
            "exposure_col": self.exposure_col,
        }

    def to_pandas(self) -> pd.DataFrame:
        """Return the panel as a pandas DataFrame (for econometrics libraries)."""
        if self._panel is None:
            raise RuntimeError("Call build() before to_pandas().")
        # Use dict-based conversion to avoid pyarrow dependency (Databricks serverless compat)
        return pd.DataFrame(self._panel.to_dict(as_series=False))

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _aggregate_policy(self) -> pl.DataFrame:
        """Aggregate policy table to segment × period."""
        agg_exprs = [
            pl.col("earned_premium").sum(),
            pl.col("earned_exposure").sum(),
        ]
        if self.outcome == "retention":
            agg_exprs += [
                pl.col("policies_renewed").sum(),
                pl.col("policies_due").sum(),
            ]
        return (
            self.policy_df
            .group_by(["segment_id", "period"])
            .agg(agg_exprs)
            .sort(["segment_id", "period"])
        )

    def _join_claims(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Left-join claims onto policy panel; fill nulls with zero."""
        claims_cols = ["segment_id", "period", "incurred_claims", "claim_count"]
        if self.paid_only:
            claims_cols.append("paid_claims")
        claims = self.claims_df.select(claims_cols)
        panel = panel.join(claims, on=["segment_id", "period"], how="left")
        fill_cols = ["incurred_claims", "claim_count"]
        if self.paid_only:
            fill_cols.append("paid_claims")
        panel = panel.with_columns(
            [pl.col(c).fill_null(0.0) for c in fill_cols]
        )
        return panel

    def _compute_outcome(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Compute the requested outcome metric."""
        if self.outcome == "loss_ratio":
            claims_col = "paid_claims" if self.paid_only else "incurred_claims"
            panel = panel.with_columns(
                pl.when(pl.col("earned_premium") > 0)
                .then(pl.col(claims_col) / pl.col("earned_premium"))
                .otherwise(None)
                .alias("loss_ratio")
            )
        elif self.outcome == "frequency":
            panel = panel.with_columns(
                pl.when(pl.col("earned_exposure") > 0)
                .then(pl.col("claim_count") / pl.col("earned_exposure"))
                .otherwise(None)
                .alias("frequency")
            )
        elif self.outcome == "retention":
            panel = panel.with_columns(
                pl.when(pl.col("policies_due") > 0)
                .then(pl.col("policies_renewed") / pl.col("policies_due"))
                .otherwise(None)
                .alias("retention")
            )
        return panel

    def _join_treatment(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Join rate change log to create treatment indicators."""
        rate_log = self.rate_log_df.select(["segment_id", "first_treated_period"])
        panel = panel.join(rate_log, on="segment_id", how="left")
        panel = panel.with_columns(
            pl.col("first_treated_period").cast(pl.Int64, strict=False)
        )
        panel = panel.with_columns(
            pl.when(
                pl.col("first_treated_period").is_not_null()
                & (pl.col("period") >= pl.col("first_treated_period"))
            )
            .then(1)
            .otherwise(0)
            .alias("treated")
        )
        panel = panel.with_columns(
            pl.col("first_treated_period").alias("cohort")
        )
        return panel

    def _balance_panel(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Ensure all segment × period combinations exist.

        Cells that exist in the cartesian product but not in the data are
        filled with zero exposure and null outcome. These indicate periods
        when a segment had no active policies — common for new segments.
        """
        all_segments = panel["segment_id"].unique()
        all_periods = panel["period"].unique()
        full_grid = pl.DataFrame({"segment_id": all_segments}).join(
            pl.DataFrame({"period": all_periods}), how="cross"
        )
        panel = full_grid.join(panel, on=["segment_id", "period"], how="left")
        panel = panel.with_columns(
            pl.col("earned_premium").fill_null(0.0),
            pl.col("earned_exposure").fill_null(0.0),
            pl.col("incurred_claims").fill_null(0.0),
            pl.col("claim_count").fill_null(0),
        )
        return panel.sort(["segment_id", "period"])

    def _validate(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Run data quality checks and emit warnings."""
        # Thin cells
        thin = panel.filter(
            (pl.col("earned_exposure") > 0)
            & (pl.col("earned_exposure") < self.min_exposure)
        )
        if thin.height > 0:
            warnings.warn(
                f"{thin.height} segment-period cells have exposure below {self.min_exposure} "
                f"({self.exposure_col}). Estimates may be unstable for thin segments. "
                "Consider aggregating to coarser segments.",
                UserWarning,
                stacklevel=3,
            )

        # Null outcomes (zero-exposure cells)
        n_null = panel.filter(pl.col(self.outcome).is_null()).height
        if n_null > 0:
            pct = 100 * n_null / len(panel)
            if pct > 20:
                warnings.warn(
                    f"{pct:.1f}% of panel cells have null {self.outcome} "
                    "(zero or missing exposure). High null rate may affect SDID validity. "
                    "Consider restricting to segments with continuous coverage.",
                    UserWarning,
                    stacklevel=3,
                )

        # Treatment consistency check
        treated_segs = panel.filter(pl.col("first_treated_period").is_not_null())
        if treated_segs.height == 0:
            warnings.warn(
                "No treated segments found. Check that rate_log_df segment_ids "
                "match policy_df segment_ids.",
                UserWarning,
                stacklevel=3,
            )

        return panel


# ---------------------------------------------------------------------------
# Convenience function: build panel from pandas DataFrames
# ---------------------------------------------------------------------------


def build_panel_from_pandas(
    policy_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    rate_log_df: pd.DataFrame,
    outcome: str = "loss_ratio",
    exposure_col: str = "earned_premium",
    min_exposure: float = 50.0,
    paid_only: bool = False,
) -> pl.DataFrame:
    """Convenience wrapper accepting pandas DataFrames.

    Converts inputs to Polars and calls PolicyPanelBuilder.build().
    Returns a Polars DataFrame.
    """
    def _pd_to_pl(df: pd.DataFrame) -> pl.DataFrame:
        # Avoid pyarrow-based conversion (Databricks serverless compat)
        return pl.DataFrame(df.to_dict("list"))

    return PolicyPanelBuilder(
        policy_df=_pd_to_pl(policy_df),
        claims_df=_pd_to_pl(claims_df),
        rate_log_df=_pd_to_pl(rate_log_df),
        outcome=outcome,
        exposure_col=exposure_col,
        min_exposure=min_exposure,
        paid_only=paid_only,
    ).build()
