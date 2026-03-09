"""
HonestDiD-style sensitivity analysis.

Implements a Python version of the breakdown frontier analysis from:
  Rambachan & Roth (2023). 'A More Credible Approach to Parallel Trends.'
  Review of Economic Studies, rdad018.

The key question: how large must a violation of the parallel trends assumption
be (post-treatment) before the sign of the ATT reverses? This is the
'breakdown frontier' — the maximum M such that the identified set for ATT
still excludes zero.

The sensitivity parameter M is expressed in multiples of the pre-period
standard deviation of the event-study estimates. This makes M interpretable:
M=1 means 'the post-treatment violation could be as large as the typical
pre-treatment variation we observed'.

Implementation follows the linear extrapolation restriction (Assumption 1 in
Rambachan & Roth): the post-treatment parallel trends violation is bounded by
M * max(|delta_{pre}|) where delta_{pre} are the pre-treatment event-study
residuals.

Note: The official HonestDiD package is R-only. This is a Python approximation
that captures the key intuition for regulatory evidence packs. For formal
econometric analysis, use the R package.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from ._types import SDIDResult, SensitivityResult


# ---------------------------------------------------------------------------
# Breakdown frontier computation
# ---------------------------------------------------------------------------


def compute_sensitivity(
    result: SDIDResult,
    m_values: Optional[list[float]] = None,
    method: str = "linear",
) -> SensitivityResult:
    """Compute HonestDiD-style sensitivity bounds for the ATT.

    For each value of M, computes the identified set [ATT_lower, ATT_upper]
    under the assumption that the post-treatment parallel trends violation is
    bounded by M * pre_period_sd.

    The identified set widens as M increases. The 'breakdown point' M* is the
    smallest M at which 0 enters the identified set.

    Parameters
    ----------
    result : SDIDResult
        Fitted SDID result with event study.
    m_values : list[float]
        Sensitivity parameter values to test. Default: [0, 0.5, 1.0, 1.5, 2.0].
    method : str
        Bound construction method:
        'linear' — linear extrapolation of pre-trend (Assumption 1)
        'smooth' — smoothness restriction (second differences bounded)
        Default: 'linear'.

    Returns
    -------
    SensitivityResult
    """
    if result.event_study is None:
        raise ValueError("Sensitivity analysis requires event_study data.")

    if m_values is None:
        m_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    event_df = result.event_study
    pre = event_df[event_df["period_rel"] < 0]

    if len(pre) == 0:
        raise ValueError("No pre-treatment periods in event study for sensitivity analysis.")

    pre_atts = pre["att"].dropna().values
    pre_period_sd = float(np.std(pre_atts)) if len(pre_atts) > 1 else float(np.abs(pre_atts[0])) if len(pre_atts) == 1 else 1e-6

    if pre_period_sd < 1e-8:
        warnings.warn(
            "Pre-period standard deviation is near zero. "
            "Sensitivity analysis may be degenerate (ATT already very robust).",
            UserWarning,
            stacklevel=2,
        )
        pre_period_sd = max(pre_period_sd, 1e-6)

    att = result.att
    se = result.se

    att_lower_list = []
    att_upper_list = []

    for M in m_values:
        delta = M * pre_period_sd
        if method == "linear":
            # Under linear extrapolation: the post-treatment violation could be
            # up to delta in either direction. The identified set for ATT is
            # [att - delta - 1.96*se, att + delta + 1.96*se].
            # The 'robust' confidence interval widens by +/- delta.
            lower = att - delta - 1.96 * se
            upper = att + delta + 1.96 * se
        elif method == "smooth":
            # Smoothness restriction: post-treatment deviations bounded by
            # M * (pre-period second difference). More conservative.
            max_second_diff = float(np.max(np.abs(np.diff(pre_atts, 2)))) if len(pre_atts) >= 3 else pre_period_sd
            delta_smooth = M * max_second_diff
            lower = att - delta_smooth - 1.96 * se
            upper = att + delta_smooth + 1.96 * se
        else:
            raise ValueError(f"Unknown sensitivity method: '{method}'")

        att_lower_list.append(lower)
        att_upper_list.append(upper)

    # Breakdown point: smallest M such that 0 is in [lower, upper]
    # i.e. the sign is no longer certain
    m_breakdown = _find_breakdown_point(att, se, m_values, pre_period_sd, method)

    return SensitivityResult(
        m_values=m_values,
        att_lower=att_lower_list,
        att_upper=att_upper_list,
        m_breakdown=m_breakdown,
        pre_period_sd=pre_period_sd,
    )


def _find_breakdown_point(
    att: float,
    se: float,
    m_values: list[float],
    pre_period_sd: float,
    method: str,
) -> float:
    """Find M* where zero first enters the identified set.

    Binary search over M in [0, max(m_values) * 2].
    """
    if pre_period_sd < 1e-8:
        return float("inf")

    max_m = max(m_values) * 2
    lo, hi = 0.0, max_m

    for _ in range(50):
        mid = (lo + hi) / 2
        delta = mid * pre_period_sd
        lower = att - delta - 1.96 * se
        upper = att + delta + 1.96 * se

        # Does zero fall inside [lower, upper]?
        zero_in_set = lower <= 0 <= upper

        if zero_in_set:
            hi = mid
        else:
            lo = mid

        if hi - lo < 1e-4:
            break

    m_star = (lo + hi) / 2

    # If already zero is in the set at M=0, return 0
    lower_0 = att - 1.96 * se
    upper_0 = att + 1.96 * se
    if lower_0 <= 0 <= upper_0:
        return 0.0

    return m_star


# ---------------------------------------------------------------------------
# Sensitivity plot
# ---------------------------------------------------------------------------


def plot_sensitivity(
    sens: SensitivityResult,
    title: str = "Sensitivity Analysis: Parallel Trends Violations",
    ax: Optional[object] = None,
    figsize: tuple[int, int] = (8, 5),
) -> object:
    """Plot the breakdown frontier — identified set width vs M.

    Shows how the confidence interval for ATT widens as we allow larger
    violations of the parallel trends assumption. The breakdown frontier
    M* is annotated.

    Returns matplotlib Figure.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        raise ImportError("matplotlib required for sensitivity plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    m_arr = np.array(sens.m_values)
    lower = np.array(sens.att_lower)
    upper = np.array(sens.att_upper)

    ax.fill_between(m_arr, lower, upper, alpha=0.25, color="#2166ac", label="Identified set")
    ax.plot(m_arr, lower, color="#2166ac", linewidth=1.5)
    ax.plot(m_arr, upper, color="#2166ac", linewidth=1.5)
    ax.axhline(0, color="#666666", linewidth=1, linestyle="--", label="ATT = 0")

    # Breakdown point
    m_star = sens.m_breakdown
    if m_star <= max(sens.m_values):
        ax.axvline(m_star, color="#d6604d", linewidth=1.5, linestyle=":", label=f"Breakdown M* = {m_star:.2f}")

    ax.set_xlabel("M (multiple of pre-period SD)", fontsize=11)
    ax.set_ylabel("Identified set for ATT", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate pre-period SD
    ax.text(
        0.98, 0.02,
        f"Pre-period SD = {sens.pre_period_sd:.4f}",
        transform=ax.transAxes, fontsize=8, ha="right", color="#666666"
    )

    fig.tight_layout()
    return fig
