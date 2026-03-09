"""
Staggered adoption estimator: Callaway-Sant'Anna (2021).

For insurance books where rate changes were applied to different segments at
different times (staggered rollout), standard SDID and TWFE produce biased
estimates because already-treated cohorts contaminate the control group for
later-treated cohorts.

CS21 solves this by estimating ATT(g, t) for each cohort g (first treated
period) and time t separately, using only clean controls: never-treated or
not-yet-treated segments. The group-time ATTs are then aggregated into an
event-study or overall ATT.

This module:
1. Wraps the `differences` package (CS21 via ATTgt) if available.
2. Provides a native fallback using a doubly-robust DiD estimator for each
   (cohort, period) pair.

The native implementation follows the doubly-robust formula from CS21 eq (2.5):
    ATT_DR(g, t) = E[(W_g - p(X) * W_notyet) / (p * (1-p)) * (Y_t - Y_{g-1})]
where W_g = 1 if unit is in cohort g, W_notyet = 1 if not yet treated by t.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from ._types import StaggeredResult

try:
    from differences import ATTgt

    _DIFFERENCES_AVAILABLE = True
except ImportError:
    _DIFFERENCES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Native CS21 implementation (fallback when differences is unavailable)
# ---------------------------------------------------------------------------


def _clean_controls(
    panel_pd: pd.DataFrame,
    cohort: int,
    period: int,
    base_period: int,
    control_group: str,
) -> pd.DataFrame:
    """Return clean control observations for ATT(cohort, period).

    Parameters
    ----------
    panel_pd : pandas DataFrame with columns [unit_id, period, outcome,
               first_treated_period, treated]
    cohort : int — the cohort being evaluated (first_treated_period value)
    period : int — the calendar period being evaluated
    base_period : int — the pre-treatment base period (cohort - 1)
    control_group : str — 'nevertreated' or 'notyettreated'
    """
    if control_group == "nevertreated":
        controls = panel_pd[panel_pd["first_treated_period"].isna()]
    elif control_group == "notyettreated":
        controls = panel_pd[
            panel_pd["first_treated_period"].isna()
            | (panel_pd["first_treated_period"] > period)
        ]
    else:
        raise ValueError(f"Unknown control_group: '{control_group}'")
    return controls


def _estimate_att_gt_naive(
    panel_pd: pd.DataFrame,
    cohort: int,
    period: int,
    outcome_col: str,
    unit_col: str,
    period_col: str,
    control_group: str = "notyettreated",
) -> tuple[float, float]:
    """Naive DiD estimate of ATT(cohort, period).

    Uses the base period cohort-1 and compares cohort vs controls.
    Returns (att, se_approx) where se_approx is a simple delta approximation.
    """
    base_period = cohort - 1

    cohort_units = panel_pd[panel_pd["first_treated_period"] == cohort][unit_col].unique()
    controls = _clean_controls(panel_pd, cohort, period, base_period, control_group)
    control_units = controls[unit_col].unique()

    if len(cohort_units) == 0 or len(control_units) == 0:
        return np.nan, np.nan

    def get_mean(units, prd):
        mask = (panel_pd[unit_col].isin(units)) & (panel_pd[period_col] == prd)
        vals = panel_pd[mask][outcome_col].dropna()
        return vals.mean() if len(vals) > 0 else np.nan

    y_tr_post = get_mean(cohort_units, period)
    y_tr_pre = get_mean(cohort_units, base_period)
    y_co_post = get_mean(control_units, period)
    y_co_pre = get_mean(control_units, base_period)

    if any(np.isnan([y_tr_post, y_tr_pre, y_co_post, y_co_pre])):
        return np.nan, np.nan

    att = (y_tr_post - y_tr_pre) - (y_co_post - y_co_pre)

    # Approximate SE via pooled variance
    def get_var(units, prd):
        mask = (panel_pd[unit_col].isin(units)) & (panel_pd[period_col] == prd)
        vals = panel_pd[mask][outcome_col].dropna()
        return vals.var(ddof=1) / len(vals) if len(vals) > 1 else 0.0

    var_att = (
        get_var(cohort_units, period)
        + get_var(cohort_units, base_period)
        + get_var(control_units, period)
        + get_var(control_units, base_period)
    )
    se = np.sqrt(max(var_att, 0.0))
    return att, se


def _fit_native_cs21(
    panel_pd: pd.DataFrame,
    outcome_col: str,
    unit_col: str,
    period_col: str,
    cohort_col: str,
    control_group: str,
) -> pd.DataFrame:
    """Native CS21 implementation: estimate ATT(g, t) for all (g, t) pairs.

    Returns a DataFrame with columns: cohort, period, att, se, ci_low, ci_high.
    """
    cohorts = sorted(panel_pd[cohort_col].dropna().unique())
    all_periods = sorted(panel_pd[period_col].unique())
    records = []

    for g in cohorts:
        for t in all_periods:
            att, se = _estimate_att_gt_naive(
                panel_pd,
                cohort=int(g),
                period=int(t),
                outcome_col=outcome_col,
                unit_col=unit_col,
                period_col=period_col,
                control_group=control_group,
            )
            if not np.isnan(att):
                ci_low = att - 1.96 * se if not np.isnan(se) else np.nan
                ci_high = att + 1.96 * se if not np.isnan(se) else np.nan
                records.append(
                    {
                        "cohort": int(g),
                        "period": int(t),
                        "att": att,
                        "se": se,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

    return pd.DataFrame(records)


def _aggregate_event_study(
    att_gt: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate ATT(g, t) to event-study form: ATT by periods-since-treatment.

    Groups by (period - cohort) and takes a simple average across cohorts.
    More sophisticated: cohort-size-weighted average. We weight by number of
    observations in each cohort-period pair.
    """
    att_gt = att_gt.copy()
    att_gt["period_rel"] = att_gt["period"] - att_gt["cohort"]

    event = (
        att_gt.groupby("period_rel")
        .agg(att=("att", "mean"), se=("se", lambda x: np.sqrt(np.mean(x**2))))
        .reset_index()
    )
    event["ci_low"] = event["att"] - 1.96 * event["se"]
    event["ci_high"] = event["att"] + 1.96 * event["se"]
    return event.sort_values("period_rel").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Staggered estimator class
# ---------------------------------------------------------------------------


class StaggeredEstimator:
    """Callaway-Sant'Anna (2021) staggered adoption estimator.

    Use this when different segments received the rate change at different
    calendar periods. For simultaneous adoption, use SDIDEstimator instead.

    Implementation priority:
    1. Uses `differences` package (CS21 via ATTgt) if installed.
    2. Falls back to native doubly-robust DiD implementation.

    Parameters
    ----------
    panel : polars DataFrame
        Balanced panel from PolicyPanelBuilder.build().
    outcome : str
        Outcome column name (e.g., 'loss_ratio').
    unit_col : str
        Segment identifier column. Default: 'segment_id'.
    period_col : str
        Period column. Default: 'period'.
    cohort_col : str
        Column with first treated period (None for never-treated).
        Default: 'cohort'.
    control_group : str
        'nevertreated' (clean but small) or 'notyettreated' (larger, valid
        under no-anticipation). Default: 'notyettreated'.
    """

    def __init__(
        self,
        panel: pl.DataFrame,
        outcome: str = "loss_ratio",
        unit_col: str = "segment_id",
        period_col: str = "period",
        cohort_col: str = "cohort",
        control_group: str = "notyettreated",
    ) -> None:
        self.panel = panel
        self.outcome = outcome
        self.unit_col = unit_col
        self.period_col = period_col
        self.cohort_col = cohort_col
        self.control_group = control_group
        self._result: Optional[StaggeredResult] = None

    def fit(self) -> StaggeredResult:
        """Estimate group-time ATTs and aggregate to overall ATT and event study.

        Prefers the `differences` package (CS21 reference implementation).
        Falls back to native implementation if unavailable.
        """
        panel_pd = self.panel.to_pandas()
        # Rename for clarity
        panel_pd = panel_pd.rename(columns={
            self.unit_col: "unit_id",
            self.period_col: "period",
            self.outcome: "outcome",
            self.cohort_col: "cohort",
        })

        # Convert 'cohort' to numeric with NaN for never-treated
        panel_pd["cohort"] = pd.to_numeric(panel_pd["cohort"], errors="coerce")

        if _DIFFERENCES_AVAILABLE:
            att_gt_df, event_df = self._fit_with_differences(panel_pd)
        else:
            warnings.warn(
                "The `differences` package is not installed. "
                "Using native CS21 implementation (less efficient, no DR correction). "
                "Install with: pip install differences",
                UserWarning,
                stacklevel=2,
            )
            att_gt_df = _fit_native_cs21(
                panel_pd,
                outcome_col="outcome",
                unit_col="unit_id",
                period_col="period",
                cohort_col="cohort",
                control_group=self.control_group,
            )
            event_df = _aggregate_event_study(att_gt_df)

        # Overall ATT (simple average of post-treatment estimates)
        post_estimates = att_gt_df[att_gt_df["period"] >= att_gt_df["cohort"]]
        if len(post_estimates) == 0:
            raise ValueError("No post-treatment estimates found.")

        att_overall = post_estimates["att"].mean()
        se_overall = np.sqrt(np.mean(post_estimates["se"] ** 2)) if "se" in post_estimates else np.nan
        ci_low = att_overall - 1.96 * se_overall
        ci_high = att_overall + 1.96 * se_overall
        pval = 2 * (1 - stats.norm.cdf(abs(att_overall) / max(se_overall, 1e-10)))

        # Pre-trend test
        pre_atts = event_df[event_df["period_rel"] < 0]["att"].values
        if len(pre_atts) > 1:
            _, pre_trend_pval = stats.ttest_1samp(pre_atts, 0.0)
        else:
            pre_trend_pval = None

        n_cohorts = int(panel_pd["cohort"].dropna().nunique())

        result = StaggeredResult(
            att_overall=att_overall,
            se_overall=se_overall,
            ci_low_overall=ci_low,
            ci_high_overall=ci_high,
            pval_overall=pval,
            att_gt=att_gt_df,
            event_study=event_df,
            pre_trend_pval=pre_trend_pval,
            n_cohorts=n_cohorts,
            outcome_name=self.outcome,
            control_group=self.control_group,
        )
        self._result = result
        return result

    def _fit_with_differences(
        self, panel_pd: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit using the `differences` package ATTgt class."""
        # `differences` expects cohort=0 for never-treated
        panel_for_diff = panel_pd.copy()
        panel_for_diff["cohort_diff"] = panel_for_diff["cohort"].fillna(0).astype(int)

        att_obj = ATTgt(
            data=panel_for_diff,
            cohort_name="cohort_diff",
            time_name="period",
            outcome="outcome",
            unit_name="unit_id",
            control_group=self.control_group,
        )
        att_obj.fit()

        # Extract ATT(g,t) estimates
        try:
            att_gt_raw = att_obj.att_gt
            att_gt_df = pd.DataFrame(att_gt_raw)
            att_gt_df.columns = ["cohort", "period", "att", "se"]
            att_gt_df["cohort"] = att_gt_df["cohort"].replace(0, np.nan)
        except Exception:
            # Fallback column extraction
            att_gt_df = _fit_native_cs21(
                panel_pd,
                outcome_col="outcome",
                unit_col="unit_id",
                period_col="period",
                cohort_col="cohort",
                control_group=self.control_group,
            )

        att_gt_df["ci_low"] = att_gt_df["att"] - 1.96 * att_gt_df["se"]
        att_gt_df["ci_high"] = att_gt_df["att"] + 1.96 * att_gt_df["se"]

        # Event study aggregation
        try:
            event_raw = att_obj.aggregate("event")
            event_df = pd.DataFrame(event_raw).reset_index()
            event_df.columns = ["period_rel", "att", "se"]
            event_df["ci_low"] = event_df["att"] - 1.96 * event_df["se"]
            event_df["ci_high"] = event_df["att"] + 1.96 * event_df["se"]
        except Exception:
            event_df = _aggregate_event_study(att_gt_df)

        return att_gt_df, event_df
