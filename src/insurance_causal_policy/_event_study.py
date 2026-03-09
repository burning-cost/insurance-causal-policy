"""
Event study diagnostics and visualisation.

Provides functions for:
1. Pre-treatment parallel trends test (formal and visual)
2. Event study plot (dynamic treatment effects)
3. Synthetic outcome trajectory plot (actual vs synthetic control)
4. Unit weight bar chart (which control segments drive the synthetic control)

Design note: all plot functions return matplotlib Figure objects rather than
showing them — callers control display. This makes the library usable in
Databricks notebooks, Jupyter, and scripted batch runs without GUI issues.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    matplotlib.use("Agg")
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from ._types import SDIDResult, StaggeredResult


def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plots. Install with: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Event study plot
# ---------------------------------------------------------------------------


def plot_event_study(
    result: "SDIDResult | StaggeredResult",
    title: str = "Event Study: Dynamic Treatment Effects",
    ax: Optional["plt.Axes"] = None,
    figsize: tuple[int, int] = (10, 5),
    zero_line_color: str = "#666666",
    pre_color: str = "#2166ac",
    post_color: str = "#d6604d",
    annotate_pval: bool = True,
) -> "plt.Figure":
    """Plot event study with pre-treatment and post-treatment ATT estimates.

    Points to the left of the vertical dashed line (period_rel < 0) are
    pre-treatment estimates — they should hug zero if parallel trends holds.
    Points to the right (period_rel >= 0) show the dynamic treatment effect.

    Parameters
    ----------
    result : SDIDResult or StaggeredResult
    title : str — plot title
    ax : existing matplotlib Axes (creates new figure if None)
    figsize : figure size
    zero_line_color : colour of the zero reference line
    pre_color : colour for pre-treatment estimates
    post_color : colour for post-treatment estimates
    annotate_pval : whether to annotate the pre-treatment p-value

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    event_df = result.event_study
    if event_df is None:
        raise ValueError("Event study data not available. Check result.event_study.")

    pre = event_df[event_df["period_rel"] < 0]
    post = event_df[event_df["period_rel"] >= 0]

    # Pre-treatment points
    ax.errorbar(
        pre["period_rel"],
        pre["att"],
        yerr=1.96 * pre["se"].fillna(0) if "se" in pre and pre["se"].notna().any() else None,
        fmt="o",
        color=pre_color,
        capsize=4,
        label="Pre-treatment (parallel trends test)",
        zorder=3,
    )

    # Post-treatment points
    ax.errorbar(
        post["period_rel"],
        post["att"],
        yerr=1.96 * post["se"].fillna(0) if "se" in post and post["se"].notna().any() else None,
        fmt="s",
        color=post_color,
        capsize=4,
        label="Post-treatment (treatment effect)",
        zorder=3,
    )

    # Reference lines
    ax.axhline(0, color=zero_line_color, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(-0.5, color="black", linewidth=1.0, linestyle=":", alpha=0.5, label="Treatment start")

    # Annotation
    if annotate_pval and result.pre_trend_pval is not None:
        pval_str = f"Pre-trend test: p = {result.pre_trend_pval:.3f}"
        verdict = "PASS" if result.pre_trends_pass else "FAIL"
        colour = "#2ca02c" if result.pre_trends_pass else "#d62728"
        ax.text(
            0.02, 0.97,
            f"{pval_str} ({verdict})",
            transform=ax.transAxes,
            fontsize=9,
            color=colour,
            verticalalignment="top",
        )

    ax.set_xlabel("Periods relative to treatment", fontsize=11)
    ax.set_ylabel(getattr(result, "outcome_name", "Outcome"), fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Synthetic outcome trajectory (actual vs synthetic control)
# ---------------------------------------------------------------------------


def plot_synthetic_trajectory(
    result: SDIDResult,
    Y: np.ndarray,
    n_co: int,
    n_tr: int,
    t_pre: int,
    t_post: int,
    period_ids: list,
    outcome_name: str = "outcome",
    title: str = "Actual vs Synthetic Control",
    ax: Optional["plt.Axes"] = None,
    figsize: tuple[int, int] = (10, 5),
) -> "plt.Figure":
    """Plot the treated average vs SDID-weighted synthetic control over time.

    The pre-treatment overlap (visual parallel trends after weighting) and
    post-treatment divergence (the estimated effect) are the key features.

    Parameters
    ----------
    result : SDIDResult
    Y : (N, T) outcome matrix with control rows first
    n_co, n_tr, t_pre, t_post : panel dimensions
    period_ids : list of period identifiers (x-axis labels)
    outcome_name : label for y-axis
    """
    _require_mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    omega = result.weights.unit_weights.values
    Y_co = Y[:n_co]
    Y_tr = Y[n_co:]

    # Synthetic control: omega-weighted average of control outcomes
    synthetic = omega @ Y_co  # shape (T,)
    actual = np.mean(Y_tr, axis=0)  # shape (T,)

    periods = list(range(len(period_ids)))
    ax.plot(periods, actual, color="#d6604d", linewidth=2, label="Treated (actual)")
    ax.plot(periods, synthetic, color="#2166ac", linewidth=2, linestyle="--", label="Synthetic control")

    # Shade post-treatment region
    ax.axvspan(t_pre - 0.5, len(periods) - 0.5, alpha=0.08, color="orange", label="Post-treatment")
    ax.axvline(t_pre - 0.5, color="black", linewidth=1, linestyle=":", alpha=0.5)

    # X-axis ticks
    tick_labels = [str(p) for p in period_ids]
    step = max(1, len(periods) // 8)
    ax.set_xticks(periods[::step])
    ax.set_xticklabels(tick_labels[::step], rotation=30, ha="right", fontsize=8)

    ax.set_ylabel(outcome_name, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Unit weight bar chart
# ---------------------------------------------------------------------------


def plot_unit_weights(
    result: SDIDResult,
    n_top: int = 20,
    title: str = "Unit Weights: Synthetic Control Composition",
    ax: Optional["plt.Axes"] = None,
    figsize: tuple[int, int] = (10, 5),
    color: str = "#4393c3",
) -> "plt.Figure":
    """Bar chart of unit weights for the top control segments.

    High-weight segments are the synthetic control backbone. In insurance
    context: 'Synthetic control = 42% south-east, 31% midlands, ...'

    Parameters
    ----------
    result : SDIDResult
    n_top : int — show top N segments by weight (others collapsed to 'other')
    """
    _require_mpl()

    omega = result.weights.unit_weights.sort_values(ascending=False)
    top = omega.head(n_top)
    other_weight = omega.iloc[n_top:].sum() if len(omega) > n_top else 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    labels = list(top.index.astype(str))
    values = list(top.values)

    if other_weight > 0.001:
        labels.append("(other)")
        values.append(other_weight)

    # Zero-weight segments
    n_zero = int((omega <= 0.001).sum())

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=color, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Weight (omega)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, max(values) * 1.15)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8
        )

    if n_zero > 0:
        ax.text(
            0.98, 0.02,
            f"{n_zero} segments excluded (zero weight)",
            transform=ax.transAxes, fontsize=8, ha="right",
            color="#666666",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Parallel trends summary function
# ---------------------------------------------------------------------------


def pre_trend_summary(result: "SDIDResult | StaggeredResult") -> dict:
    """Return a structured summary of the parallel trends test.

    Keys:
      pval : float — joint p-value for H0: all pre-treatment ATTs = 0
      pass : bool — True if p > 0.10
      pre_atts : list[float] — individual pre-period estimates
      max_abs_pre_att : float — maximum absolute deviation in pre-treatment
      interpretation : str — plain-English summary
    """
    event_df = result.event_study
    if event_df is None:
        return {
            "pval": None,
            "pass": None,
            "pre_atts": [],
            "max_abs_pre_att": None,
            "interpretation": "Event study not available.",
        }

    pre = event_df[event_df["period_rel"] < 0]
    pre_atts = pre["att"].dropna().tolist()
    max_abs = float(np.max(np.abs(pre_atts))) if pre_atts else None
    pval = result.pre_trend_pval
    passes = result.pre_trends_pass

    if pval is None:
        interp = "Insufficient pre-treatment periods for joint test."
    elif passes:
        interp = (
            f"Parallel trends test passes (p={pval:.3f}). "
            "Pre-treatment ATTs are jointly indistinguishable from zero. "
            "Caution: low power with few pre-treatment periods."
        )
    else:
        interp = (
            f"Parallel trends test FAILS (p={pval:.3f}). "
            "Pre-treatment ATTs show systematic deviation from zero. "
            "SDID estimates may be biased. Investigate external shocks "
            "(Ogden rate changes, COVID lockdowns, GIPP reforms)."
        )

    return {
        "pval": pval,
        "pass": passes,
        "pre_atts": pre_atts,
        "max_abs_pre_att": max_abs,
        "interpretation": interp,
    }
