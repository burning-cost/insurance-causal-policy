"""
Result dataclasses for insurance-causal-policy.

All results carry insurance-vocabulary attributes rather than generic
econometric ones — ATT is described in terms of the outcome metric
(loss ratio, frequency, retention) rather than abstract treatment effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SDIDWeights:
    """Unit and time weights computed by the SDID optimiser.

    unit_weights: Series indexed by control unit identifier, values in [0,1],
        sum to 1. High-weight segments are the synthetic control backbone.
    time_weights: Series indexed by pre-treatment period, values in [0,1],
        sum to 1. High-weight periods are most informative for trend matching.
    unit_intercept: Scalar intercept from unit weight optimisation (absorbs
        level differences between treated and control — the key SDID relaxation
        vs pure synthetic control).
    regularisation_zeta: The zeta value used in L2 regularisation of unit
        weights. Theory-motivated: zeta = (N_tr * T_post)^{1/4} * sigma(Delta).
    """

    unit_weights: pd.Series
    time_weights: pd.Series
    unit_intercept: float
    regularisation_zeta: float


@dataclass
class SDIDResult:
    """Full SDID estimation result.

    Attributes
    ----------
    att : float
        Average Treatment Effect on the Treated. In the insurance context this
        is the estimated causal change in the outcome (e.g., loss ratio) caused
        by the rate change, expressed in the outcome's natural units.
    se : float
        Standard error of the ATT estimate.
    ci_low, ci_high : float
        95% confidence interval bounds (normal approximation).
    pval : float
        Two-sided p-value for H0: ATT = 0.
    weights : SDIDWeights
        Unit and time weights used in the estimation.
    pre_trend_pval : float or None
        Joint p-value for H0: all pre-treatment ATTs = 0. None if event study
        was not computed. Low values indicate parallel trends violation.
    event_study : DataFrame or None
        Event-study table with columns [period_rel, att, se, ci_low, ci_high].
        period_rel is quarters relative to treatment (negative = pre-treatment).
    n_treated : int
        Number of treated units (segments).
    n_control : int
        Number of control units contributing non-zero unit weight.
    n_control_total : int
        Total control units available (including zero-weight ones).
    t_pre : int
        Number of pre-treatment periods.
    t_post : int
        Number of post-treatment periods.
    outcome_name : str
        Name of the outcome variable analysed.
    inference_method : str
        Variance estimation method: 'placebo', 'bootstrap', or 'jackknife'.
    n_replicates : int
        Number of placebo/bootstrap replicates used in inference.
    """

    att: float
    se: float
    ci_low: float
    ci_high: float
    pval: float
    weights: SDIDWeights
    pre_trend_pval: Optional[float]
    event_study: Optional[pd.DataFrame]
    n_treated: int
    n_control: int
    n_control_total: int
    t_pre: int
    t_post: int
    outcome_name: str
    inference_method: str
    n_replicates: int

    @property
    def significant(self) -> bool:
        """Returns True if ATT is significant at the 5% level."""
        return self.pval < 0.05

    @property
    def pre_trends_pass(self) -> bool:
        """Returns True if pre-treatment parallel trends test passes (p > 0.10).

        Note: a passing test does not prove parallel trends — with few pre-
        treatment periods the test has low power. 8+ quarters is the minimum
        for a credible test.
        """
        if self.pre_trend_pval is None:
            return True
        return self.pre_trend_pval > 0.10

    def summary(self) -> str:
        """One-paragraph plain-English summary of the result."""
        direction = "decrease" if self.att < 0 else "increase"
        sig_text = "statistically significant" if self.significant else "not statistically significant"
        trend_text = (
            "Pre-treatment parallel trends: PASS"
            if self.pre_trends_pass
            else f"Pre-treatment parallel trends: WARNING (p={self.pre_trend_pval:.3f})"
        )
        return (
            f"SDID estimate: {self.att:+.4f} {direction} in {self.outcome_name} "
            f"(95% CI: {self.ci_low:+.4f} to {self.ci_high:+.4f}, p={self.pval:.3f}). "
            f"The effect is {sig_text}. "
            f"Based on {self.n_treated} treated and {self.n_control} control segments "
            f"with {self.t_pre} pre-treatment and {self.t_post} post-treatment periods. "
            f"Inference via {self.inference_method} ({self.n_replicates} replicates). "
            f"{trend_text}."
        )

    def to_fca_summary(
        self,
        product_line: str = "Motor",
        rate_change_date: str = "",
    ) -> str:
        """Formatted regulatory narrative matching FCA evidence pack requirements.

        Produces text suitable for inclusion in a Consumer Duty outcome monitoring
        report or an FCA evidence pack for rate change evaluation.

        Parameters
        ----------
        product_line : str
            Product line affected by the rate change (e.g., 'Motor', 'Home').
        rate_change_date : str
            Date the rate change was implemented (for narrative context).
        """
        trend_status = (
            "PASS — pre-treatment coefficients jointly indistinguishable from zero"
            if self.pre_trends_pass
            else (
                f"WARNING — joint pre-treatment test p={self.pre_trend_pval:.3f}, "
                "indicating possible parallel trends violation"
            )
        )
        sig_statement = (
            "The effect is statistically significant at the 5% level."
            if self.significant
            else "The effect is not statistically significant at the 5% level."
        )
        date_str = f" ({rate_change_date})" if rate_change_date else ""
        return f"""Rate Change Evaluation — {product_line}{date_str}
{'=' * 60}
Outcome metric      : {self.outcome_name}
Estimated ATT       : {self.att:+.4f} (in natural units of {self.outcome_name})
95% Confidence interval: [{self.ci_low:+.4f}, {self.ci_high:+.4f}]
p-value             : {self.pval:.4f}
{sig_statement}

Panel dimensions
  Treated segments  : {self.n_treated}
  Control segments  : {self.n_control} (of {self.n_control_total} available, non-zero weight)
  Pre-treatment     : {self.t_pre} periods
  Post-treatment    : {self.t_post} periods

Parallel trends test: {trend_status}

Estimator: Synthetic Difference-in-Differences (Arkhangelsky et al. 2021)
  Native CVXPY implementation; weights estimated via constrained least squares.
Inference: {self.inference_method} with {self.n_replicates} replicates

Notes
  - SDID uses unit weights omega_i and time weights lambda_t to construct
    a synthetic control that matches pre-treatment trends in the treated segments.
  - The intercept term in unit weight optimisation absorbs level differences
    between treated and control (unlike pure synthetic control).
  - Results should be interpreted as causal estimates conditional on approximate
    parallel trends holding after reweighting. External shocks (Ogden rate changes,
    COVID lockdowns, GIPP reforms) should be assessed separately.
"""


@dataclass
class DRSCWeights:
    """SC weights and propensity ratios computed by the DRSC estimator.

    sc_weights: Series indexed by control unit identifier. Unlike SDID unit
        weights, these are unconstrained — negative values are valid and indicate
        extrapolation beyond the convex hull of control units. They are identified
        via OLS on pre-treatment group means (Eq. 10 of arXiv:2503.11375), not
        via a constrained quadratic program.
    propensity_ratios: Series indexed by control unit identifier. In the no-
        covariate (aggregate panel) case, r_{1,g} = n_treated for all control
        units (simplification of pi_1 / pi_g when each control is its own group
        of size 1). With covariates, these would be estimated via local polynomial.
    m_delta: Pooled control trend — mean(DeltaY_co) across all control units.
        This is the no-covariate estimate of E[Y_T(0) - Y_{T-1}(0) | G != 1].
    """

    sc_weights: pd.Series
    propensity_ratios: pd.Series
    m_delta: float


@dataclass
class DRSCResult:
    """Full DRSC estimation result.

    Attributes
    ----------
    att : float
        Average Treatment Effect on the Treated. Estimated via the doubly
        robust moment function (Eq. 3 of arXiv:2503.11375). Consistent if
        EITHER SC weights are correctly specified OR parallel trends holds.
    se : float
        Standard error of the ATT estimate.
    ci_low, ci_high : float
        95% confidence interval bounds (normal approximation).
    pval : float
        Two-sided p-value for H0: ATT = 0.
    weights : DRSCWeights
        SC weights, propensity ratios, and pooled control trend.
    phi : np.ndarray
        Per-unit moment function values (N,). ATT = mean(phi). Useful for
        custom bootstrap procedures or diagnostic analysis.
    pre_trend_pval : float or None
        Joint p-value for H0: all pre-treatment ATTs = 0.
    event_study : DataFrame or None
        Columns: [period_rel, att, se, ci_low, ci_high].
    n_treated : int
        Number of treated segments.
    n_control : int
        Number of control segments.
    t_pre : int
        Number of pre-treatment periods.
    t_post : int
        Number of post-treatment periods.
    outcome_name : str
        Name of the outcome variable analysed.
    inference_method : str
        Variance estimation: 'bootstrap' or 'analytic'.
    n_replicates : int
        Number of bootstrap replicates (if inference='bootstrap').
    """

    att: float
    se: float
    ci_low: float
    ci_high: float
    pval: float
    weights: DRSCWeights
    phi: np.ndarray
    pre_trend_pval: Optional[float]
    event_study: Optional[pd.DataFrame]
    n_treated: int
    n_control: int
    t_pre: int
    t_post: int
    outcome_name: str
    inference_method: str
    n_replicates: int

    @property
    def significant(self) -> bool:
        """Returns True if ATT is significant at the 5% level."""
        return self.pval < 0.05

    @property
    def pre_trends_pass(self) -> bool:
        """Returns True if pre-treatment parallel trends test passes (p > 0.10)."""
        if self.pre_trend_pval is None:
            return True
        return self.pre_trend_pval > 0.10

    def summary(self) -> str:
        """One-paragraph plain-English summary of the result."""
        direction = "decrease" if self.att < 0 else "increase"
        sig_text = "statistically significant" if self.significant else "not statistically significant"
        trend_text = (
            "Pre-treatment parallel trends: PASS"
            if self.pre_trends_pass
            else f"Pre-treatment parallel trends: WARNING (p={self.pre_trend_pval:.3f})"
        )
        return (
            f"DRSC estimate: {self.att:+.4f} {direction} in {self.outcome_name} "
            f"(95% CI: {self.ci_low:+.4f} to {self.ci_high:+.4f}, p={self.pval:.3f}). "
            f"The effect is {sig_text}. "
            f"Based on {self.n_treated} treated and {self.n_control} control segments "
            f"with {self.t_pre} pre-treatment and {self.t_post} post-treatment periods. "
            f"Inference via {self.inference_method} ({self.n_replicates} replicates). "
            f"{trend_text}."
        )

    def to_fca_summary(
        self,
        product_line: str = "Motor",
        rate_change_date: str = "",
    ) -> str:
        """Formatted regulatory narrative for FCA evidence packs.

        Parameters
        ----------
        product_line : str
            Product line affected by the rate change.
        rate_change_date : str
            Date of rate change implementation (for narrative context).
        """
        trend_status = (
            "PASS — pre-treatment coefficients jointly indistinguishable from zero"
            if self.pre_trends_pass
            else (
                f"WARNING — joint pre-treatment test p={self.pre_trend_pval:.3f}, "
                "indicating possible parallel trends violation"
            )
        )
        sig_statement = (
            "The effect is statistically significant at the 5% level."
            if self.significant
            else "The effect is not statistically significant at the 5% level."
        )
        date_str = f" ({rate_change_date})" if rate_change_date else ""
        return f"""Rate Change Evaluation — {product_line}{date_str}
{'=' * 60}
Outcome metric      : {self.outcome_name}
Estimated ATT       : {self.att:+.4f} (in natural units of {self.outcome_name})
95% Confidence interval: [{self.ci_low:+.4f}, {self.ci_high:+.4f}]
p-value             : {self.pval:.4f}
{sig_statement}

Panel dimensions
  Treated segments  : {self.n_treated}
  Control segments  : {self.n_control}
  Pre-treatment     : {self.t_pre} periods
  Post-treatment    : {self.t_post} periods

Parallel trends test: {trend_status}

Estimator: Doubly Robust Synthetic Controls (Sant'Anna, Shaikh, Syrgkanis 2025)
  SC weights via unconstrained OLS on pre-treatment group means.
  Doubly robust: consistent under EITHER correct SC weights OR parallel trends.
Inference: {self.inference_method} ({self.n_replicates} replicates)

Notes
  - DRSC SC weights are unconstrained — negative weights indicate extrapolation
    beyond the convex hull of control units. This is expected and valid.
  - In the no-covariate case (aggregate segment panel), propensity ratios
    simplify to group size ratios; no local polynomial estimation is needed.
  - The doubly robust property provides insurance against misspecification of
    either the synthetic control or the parallel trends assumption.
  - Results should be interpreted as causal estimates. External shocks
    (Ogden rate changes, COVID lockdowns, GIPP reforms) should be assessed
    separately through sensitivity analysis.
"""


@dataclass
class StaggeredResult:
    """Result from Callaway-Sant'Anna staggered adoption estimator.

    Contains group-time ATTs and their aggregated event-study form.
    Use when different segments received the rate change at different times.
    """

    att_overall: float
    se_overall: float
    ci_low_overall: float
    ci_high_overall: float
    pval_overall: float
    att_gt: pd.DataFrame  # columns: cohort, period, att, se, ci_low, ci_high
    event_study: pd.DataFrame  # columns: period_rel, att, se, ci_low, ci_high
    pre_trend_pval: Optional[float]
    n_cohorts: int
    outcome_name: str
    control_group: str

    @property
    def pre_trends_pass(self) -> bool:
        if self.pre_trend_pval is None:
            return True
        return bool(self.pre_trend_pval > 0.10)


@dataclass
class SensitivityResult:
    """HonestDiD-style sensitivity analysis result.

    Reports the identified set for the ATT under bounded violations of
    the parallel trends assumption. The key question: how large must the
    violation be before the sign of the ATT reverses?
    """

    m_values: list[float]  # Size of parallel trends violation tested
    att_lower: list[float]  # Lower bound of identified set at each M
    att_upper: list[float]  # Upper bound of identified set at each M
    m_breakdown: float  # Smallest M at which zero enters the identified set
    pre_period_sd: float  # SD of pre-period estimates (benchmark for M)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "m": self.m_values,
                "att_lower": self.att_lower,
                "att_upper": self.att_upper,
            }
        )

    def summary(self) -> str:
        if self.m_breakdown >= max(self.m_values):
            robustness = f"robust for all M tested (up to {max(self.m_values):.1f})"
        else:
            robustness = f"breaks down at M ≈ {self.m_breakdown:.2f}"
        return (
            f"Sensitivity: result {robustness}. "
            f"M is expressed as multiples of the pre-period SD ({self.pre_period_sd:.4f}). "
            f"M=1.0 means the post-treatment violation could be as large as the "
            f"largest observed pre-treatment deviation."
        )
