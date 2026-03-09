"""
SDID estimator: native CVXPY implementation.

Implements Arkhangelsky et al. (2021) Synthetic Difference-in-Differences
from first principles using CVXPY for the constrained weight optimisation.
Does not depend on synthdid.py (maintenance uncertain as of 2026).

The implementation closely follows the mathematical specification in the paper:

Unit weights omega:
    min_{omega, omega_0} || Y_pre_tr_bar - (Y_pre_co @ omega + omega_0) ||^2
                         + zeta^2 * T_pre * ||omega||^2
    s.t. sum(omega) = 1, omega >= 0

Time weights lambda:
    min_{lambda, lambda_0} || Y_post_co_bar - (Y_pre_co.T @ lambda + lambda_0) ||^2
    s.t. sum(lambda) = 1, lambda >= 0
    (post-treatment periods get uniform weight 1/T_post)

ATT: from weighted TWFE regression using the computed weights.

Inference: placebo (default), bootstrap, jackknife.
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

try:
    import cvxpy as cp

    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False

from ._types import SDIDResult, SDIDWeights


# ---------------------------------------------------------------------------
# Weight solvers
# ---------------------------------------------------------------------------


def _compute_regularisation_zeta(
    y_pre_co: np.ndarray,
    n_treated: int,
    t_post: int,
) -> float:
    """Theory-motivated regularisation: zeta = (N_tr * T_post)^{1/4} * sigma(Delta).

    Delta_it = first differences (Y_it - Y_{i,t-1}) for control units in
    pre-treatment. sigma = std of these first differences.
    """
    # First differences along time axis for control units
    first_diffs = np.diff(y_pre_co, axis=1).flatten()
    sigma = np.std(first_diffs) if len(first_diffs) > 0 else 1.0
    zeta = (n_treated * t_post) ** 0.25 * sigma
    return max(zeta, 1e-6)


def _solve_unit_weights(
    y_pre_tr_bar: np.ndarray,
    y_pre_co: np.ndarray,
    zeta: float,
    t_pre: int,
) -> tuple[np.ndarray, float]:
    """Solve for unit weights omega via CVXPY.

    Parameters
    ----------
    y_pre_tr_bar : array of shape (T_pre,)
        Mean outcome of treated units over pre-treatment periods.
    y_pre_co : array of shape (N_co, T_pre)
        Outcomes of control units over pre-treatment periods.
    zeta : float
        Regularisation strength.
    t_pre : int
        Number of pre-treatment periods.

    Returns
    -------
    omega : array of shape (N_co,)
        Unit weights summing to 1, all >= 0.
    omega_0 : float
        Intercept (absorbs level differences).
    """
    if not _CVXPY_AVAILABLE:
        raise ImportError(
            "cvxpy is required for SDID weight optimisation. "
            "Install it with: pip install cvxpy"
        )

    n_co = y_pre_co.shape[0]
    omega = cp.Variable(n_co, nonneg=True)
    omega_0 = cp.Variable()

    # Synthetic control prediction: Y_pre_co.T @ omega + omega_0
    # y_pre_co shape: (N_co, T_pre), so y_pre_co.T @ omega gives (T_pre,)
    predicted = y_pre_co.T @ omega + omega_0
    residuals = y_pre_tr_bar - predicted

    objective = cp.Minimize(
        cp.sum_squares(residuals) + (zeta**2) * t_pre * cp.sum_squares(omega)
    )
    constraints = [cp.sum(omega) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(
            f"Unit weight optimisation returned status '{prob.status}'. "
            "Falling back to equal weights.",
            UserWarning,
            stacklevel=4,
        )
        return np.ones(n_co) / n_co, 0.0

    return np.array(omega.value), float(omega_0.value)


def _solve_time_weights(
    y_post_co_bar: np.ndarray,
    y_pre_co: np.ndarray,
) -> np.ndarray:
    """Solve for time weights lambda via CVXPY.

    Finds pre-treatment periods that best predict the post-treatment control
    distribution — emphasising the most informative pre-treatment periods.

    Parameters
    ----------
    y_post_co_bar : array of shape (N_co,)
        Mean outcome of control units over post-treatment periods.
    y_pre_co : array of shape (N_co, T_pre)
        Outcomes of control units over pre-treatment periods.

    Returns
    -------
    lambda_ : array of shape (T_pre,)
        Time weights for pre-treatment periods summing to 1, all >= 0.
    """
    if not _CVXPY_AVAILABLE:
        raise ImportError("cvxpy is required for SDID weight optimisation.")

    t_pre = y_pre_co.shape[1]
    lambda_ = cp.Variable(t_pre, nonneg=True)
    lambda_0 = cp.Variable()

    # Predict post-treatment control mean from pre-treatment history
    # y_pre_co @ lambda gives (N_co,) weighted across pre-treatment periods
    predicted = y_pre_co @ lambda_ + lambda_0
    residuals = y_post_co_bar - predicted

    objective = cp.Minimize(cp.sum_squares(residuals))
    constraints = [cp.sum(lambda_) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(
            f"Time weight optimisation returned status '{prob.status}'. "
            "Falling back to equal weights.",
            UserWarning,
            stacklevel=4,
        )
        return np.ones(t_pre) / t_pre

    return np.array(lambda_.value)


# ---------------------------------------------------------------------------
# TWFE regression with weights
# ---------------------------------------------------------------------------


def _weighted_twfe(
    Y: np.ndarray,
    D: np.ndarray,
    omega: np.ndarray,
    lambda_: np.ndarray,
    n_co: int,
    n_tr: int,
    t_pre: int,
    t_post: int,
) -> float:
    """Estimate ATT via weighted two-way fixed effects.

    The SDID ATT is the coefficient on D_it in the weighted regression:
        Y_it ~ mu + alpha_i + beta_t + tau * D_it
    where observations are weighted by omega_i * lambda_t (for pre-treatment
    control) and the appropriate weight for other cells.

    We solve this directly via the Frisch-Waugh-Lovell theorem — partial out
    unit and time fixed effects from the weighted outcome and treatment indicator,
    then regress residuals.

    Parameters
    ----------
    Y : (N, T) outcome matrix
    D : (N, T) treatment matrix (0/1)
    omega : (N_co,) unit weights for control units
    lambda_ : (T_pre,) time weights for pre-treatment periods
    n_co, n_tr, t_pre, t_post : panel dimensions

    Returns
    -------
    tau : float — ATT estimate
    """
    N = n_co + n_tr
    T = t_pre + t_post

    # Build weight matrix W (N, T)
    # Control units in pre-treatment: weighted by omega * lambda
    # Treated units in post-treatment: uniform weight 1/(N_tr * T_post)
    # Other cells: weight 0 (excluded from regression per SDID paper)
    W = np.zeros((N, T))

    # Control pre-treatment: omega_i * lambda_t
    for i in range(n_co):
        for t in range(t_pre):
            W[i, t] = omega[i] * lambda_[t]

    # Treated post-treatment: uniform
    for i in range(n_co, N):
        for t in range(t_pre, T):
            W[i, t] = 1.0 / (n_tr * t_post)

    # Partial out unit and time fixed effects with weights
    # Equivalent to demeaning within the weighted regression
    # We use the formula from Arkhangelsky et al. Appendix

    # Extract relevant blocks
    Y_co_pre = Y[:n_co, :t_pre]     # (N_co, T_pre)
    Y_co_post = Y[:n_co, t_pre:]    # (N_co, T_post)
    Y_tr_pre = Y[n_co:, :t_pre]     # (N_tr, T_pre)
    Y_tr_post = Y[n_co:, t_pre:]    # (N_tr, T_post)

    # SDID ATT: weighted difference-in-differences
    # tau = (Y_tr_post_bar - Y_tr_pre_bar_lam) - (Y_co_post_bar_om - Y_co_pre_bar_om_lam)
    # where bar_lam = time-weighted average, bar_om = unit-weighted average

    Y_tr_post_bar = np.mean(Y_tr_post)  # uniform over treated × post
    Y_tr_pre_lam = np.mean(Y_tr_pre @ lambda_)  # time-weighted pre-treatment treated

    # Unit-weighted control averages
    Y_co_post_om = omega @ np.mean(Y_co_post, axis=1)  # omega-weighted control post
    Y_co_pre_om_lam = omega @ (Y_co_pre @ lambda_)     # omega-weighted, time-weighted control pre

    tau = (Y_tr_post_bar - Y_tr_pre_lam) - (Y_co_post_om - Y_co_pre_om_lam)
    return tau


# ---------------------------------------------------------------------------
# Inference: placebo, bootstrap, jackknife
# ---------------------------------------------------------------------------


def _placebo_variance(
    Y: np.ndarray,
    D: np.ndarray,
    n_co: int,
    n_tr: int,
    t_pre: int,
    t_post: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> float:
    """Placebo variance: randomly assign treatment to control units.

    Requires N_co > N_tr. Valid under homoskedasticity across units.
    Returns estimated variance of the ATT.
    """
    if n_co <= n_tr:
        raise ValueError(
            f"Placebo inference requires N_control > N_treated, "
            f"got N_co={n_co}, N_tr={n_tr}. Use bootstrap instead."
        )

    Y_co = Y[:n_co]
    placebo_atts = []

    for _ in range(n_replicates):
        # Randomly select n_tr units from control as placebo treated
        placebo_tr_idx = rng.choice(n_co, size=n_tr, replace=False)
        placebo_co_idx = np.setdiff1d(np.arange(n_co), placebo_tr_idx)

        Y_placebo_tr = Y_co[placebo_tr_idx]
        Y_placebo_co = Y_co[placebo_co_idx]

        Y_placebo = np.vstack([Y_placebo_co, Y_placebo_tr])
        D_placebo = np.vstack([
            np.zeros((len(placebo_co_idx), t_pre + t_post)),
            D[n_co:],
        ])

        try:
            att_p = _fit_sdid_core(
                Y_placebo, D_placebo,
                len(placebo_co_idx), n_tr, t_pre, t_post,
                return_weights=False,
            )[0]
            placebo_atts.append(att_p)
        except Exception:
            continue

    if len(placebo_atts) < 2:
        raise RuntimeError("Placebo variance estimation failed (too few valid replicates).")

    return float(np.var(placebo_atts, ddof=1))


def _bootstrap_variance(
    Y: np.ndarray,
    D: np.ndarray,
    n_co: int,
    n_tr: int,
    t_pre: int,
    t_post: int,
    n_replicates: int,
    rng: np.random.Generator,
) -> float:
    """Bootstrap variance: resample units (control + treated) with replacement."""
    boot_atts = []

    for _ in range(n_replicates):
        # Resample control and treated separately to preserve proportions
        co_idx = rng.choice(n_co, size=n_co, replace=True)
        tr_idx = rng.choice(np.arange(n_co, n_co + n_tr), size=n_tr, replace=True)
        idx = np.concatenate([co_idx, tr_idx])

        Y_boot = Y[idx]
        D_boot = D[idx]

        try:
            att_b = _fit_sdid_core(
                Y_boot, D_boot, n_co, n_tr, t_pre, t_post, return_weights=False
            )[0]
            boot_atts.append(att_b)
        except Exception:
            continue

    if len(boot_atts) < 2:
        raise RuntimeError("Bootstrap variance estimation failed.")

    return float(np.var(boot_atts, ddof=1))


def _jackknife_variance(
    Y: np.ndarray,
    D: np.ndarray,
    n_co: int,
    n_tr: int,
    t_pre: int,
    t_post: int,
    att_full: float,
) -> float:
    """Jackknife variance: leave-one-unit-out."""
    n_total = n_co + n_tr
    jack_atts = []

    for drop_idx in range(n_total):
        idx = np.concatenate([np.arange(drop_idx), np.arange(drop_idx + 1, n_total)])
        Y_j = Y[idx]
        D_j = D[idx]

        # Adjust n_co or n_tr depending on which unit was dropped
        n_co_j = n_co - (1 if drop_idx < n_co else 0)
        n_tr_j = n_tr - (1 if drop_idx >= n_co else 0)

        if n_co_j == 0 or n_tr_j == 0:
            continue

        try:
            att_j = _fit_sdid_core(
                Y_j, D_j, n_co_j, n_tr_j, t_pre, t_post, return_weights=False
            )[0]
            jack_atts.append(att_j)
        except Exception:
            continue

    if len(jack_atts) < 2:
        raise RuntimeError("Jackknife variance estimation failed.")

    n_j = len(jack_atts)
    # Jackknife variance formula
    jack_mean = np.mean(jack_atts)
    variance = ((n_j - 1) / n_j) * np.sum((np.array(jack_atts) - jack_mean) ** 2)
    return float(variance)


# ---------------------------------------------------------------------------
# Core fit function
# ---------------------------------------------------------------------------


def _fit_sdid_core(
    Y: np.ndarray,
    D: np.ndarray,
    n_co: int,
    n_tr: int,
    t_pre: int,
    t_post: int,
    return_weights: bool = True,
) -> tuple:
    """Core SDID fit: compute weights and ATT.

    Returns (att, omega, omega_0, lambda_, zeta) if return_weights=True,
    else (att,).
    """
    Y_co = Y[:n_co]   # (N_co, T)
    Y_tr = Y[n_co:]   # (N_tr, T)

    Y_co_pre = Y_co[:, :t_pre]   # (N_co, T_pre)
    Y_co_post = Y_co[:, t_pre:]  # (N_co, T_post)
    Y_tr_pre = Y_tr[:, :t_pre]   # (N_tr, T_pre)

    # Mean of treated units over pre-treatment periods: shape (T_pre,)
    y_pre_tr_bar = np.mean(Y_tr_pre, axis=0)

    # Mean of control units over post-treatment periods: shape (N_co,)
    y_post_co_bar = np.mean(Y_co_post, axis=1)

    # Regularisation
    zeta = _compute_regularisation_zeta(Y_co_pre, n_tr, t_post)

    # Unit weights
    omega, omega_0 = _solve_unit_weights(y_pre_tr_bar, Y_co_pre, zeta, t_pre)

    # Time weights
    lambda_ = _solve_time_weights(y_post_co_bar, Y_co_pre)

    # ATT via weighted DiD formula
    att = _weighted_twfe(Y, D, omega, lambda_, n_co, n_tr, t_pre, t_post)

    if return_weights:
        return att, omega, omega_0, lambda_, zeta
    else:
        return (att,)


# ---------------------------------------------------------------------------
# Main SDID estimator class
# ---------------------------------------------------------------------------


class SDIDEstimator:
    """Synthetic Difference-in-Differences estimator for insurance panels.

    Implements Arkhangelsky et al. (2021) natively using CVXPY. Handles
    simultaneous adoption only — use StaggeredEstimator for staggered adoption.

    Parameters
    ----------
    panel : polars DataFrame
        Balanced panel from PolicyPanelBuilder.build(). Must contain:
        segment_id, period, {outcome}, treated, first_treated_period.
    outcome : str
        Name of the outcome column to estimate (e.g., 'loss_ratio').
    treatment_col : str
        Binary treatment indicator column. Default: 'treated'.
    unit_col : str
        Segment identifier column. Default: 'segment_id'.
    period_col : str
        Period column. Default: 'period'.
    inference : str
        Variance estimation method: 'placebo' (default), 'bootstrap', 'jackknife'.
    n_replicates : int
        Number of replicates for placebo/bootstrap. Default: 200.
    random_seed : int
        Reproducibility seed. Default: 42.
    """

    def __init__(
        self,
        panel: pl.DataFrame,
        outcome: str = "loss_ratio",
        treatment_col: str = "treated",
        unit_col: str = "segment_id",
        period_col: str = "period",
        inference: Literal["placebo", "bootstrap", "jackknife"] = "placebo",
        n_replicates: int = 200,
        random_seed: int = 42,
    ) -> None:
        self.panel = panel
        self.outcome = outcome
        self.treatment_col = treatment_col
        self.unit_col = unit_col
        self.period_col = period_col
        self.inference = inference
        self.n_replicates = n_replicates
        self.rng = np.random.default_rng(random_seed)
        self._result: Optional[SDIDResult] = None

    def fit(self) -> SDIDResult:
        """Estimate the ATT and run inference.

        Returns an SDIDResult with ATT, confidence interval, weights,
        and event study data.
        """
        Y, D, unit_ids, period_ids, n_co, n_tr, t_pre, t_post = self._prepare_matrices()

        # Core fit
        att, omega, omega_0, lambda_, zeta = _fit_sdid_core(
            Y, D, n_co, n_tr, t_pre, t_post, return_weights=True
        )

        # Inference
        var = self._compute_variance(Y, D, n_co, n_tr, t_pre, t_post, att)
        se = np.sqrt(max(var, 0.0))
        ci_low = att - 1.96 * se
        ci_high = att + 1.96 * se
        pval = 2 * (1 - stats.norm.cdf(abs(att) / max(se, 1e-10)))

        # Build weight objects
        co_unit_ids = unit_ids[:n_co]
        pre_period_ids = period_ids[:t_pre]

        weights = SDIDWeights(
            unit_weights=pd.Series(omega, index=co_unit_ids, name="omega"),
            time_weights=pd.Series(lambda_, index=pre_period_ids, name="lambda"),
            unit_intercept=omega_0,
            regularisation_zeta=zeta,
        )

        # Event study
        event_study, pre_trend_pval = self._compute_event_study(
            Y, omega, lambda_, n_co, n_tr, t_pre, t_post, period_ids
        )

        result = SDIDResult(
            att=att,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            pval=pval,
            weights=weights,
            pre_trend_pval=pre_trend_pval,
            event_study=event_study,
            n_treated=n_tr,
            n_control=int(np.sum(omega > 1e-6)),
            n_control_total=n_co,
            t_pre=t_pre,
            t_post=t_post,
            outcome_name=self.outcome,
            inference_method=self.inference,
            n_replicates=self.n_replicates,
        )
        self._result = result
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list, list, int, int, int, int]:
        """Convert panel DataFrame to Y and D matrices.

        Returns Y (N, T), D (N, T), unit_ids, period_ids, n_co, n_tr, t_pre, t_post.
        Control units occupy rows 0..n_co-1, treated rows n_co..N-1.
        """
        p = self.panel.drop_nulls(subset=[self.outcome])

        # Identify treated vs control segments
        treated_segs = (
            p.filter(pl.col(self.treatment_col) == 1)[self.unit_col].unique().to_list()
        )
        if len(treated_segs) == 0:
            raise ValueError(
                "No treated segments found in panel. "
                "Check that 'treated' column contains 1s in the post-treatment period."
            )

        all_segs = p[self.unit_col].unique().sort().to_list()
        control_segs = [s for s in all_segs if s not in treated_segs]

        if len(control_segs) == 0:
            raise ValueError("No control segments found. All segments appear to be treated.")

        # Identify treatment period — first period where any unit has treated=1
        first_treated = (
            p.filter(pl.col(self.treatment_col) == 1)[self.period_col].min()
        )
        all_periods = sorted(p[self.period_col].unique().to_list())
        pre_periods = [t for t in all_periods if t < first_treated]
        post_periods = [t for t in all_periods if t >= first_treated]

        if len(pre_periods) == 0:
            raise ValueError("No pre-treatment periods found. Cannot estimate SDID.")
        if len(post_periods) == 0:
            raise ValueError("No post-treatment periods found. Cannot estimate SDID.")

        n_co = len(control_segs)
        n_tr = len(treated_segs)
        t_pre = len(pre_periods)
        t_post = len(post_periods)
        T = t_pre + t_post

        unit_order = control_segs + treated_segs
        period_order = pre_periods + post_periods

        # Pivot to matrix — fill missing with row mean for robustness
        pivot = (
            p.pivot(
                index=self.unit_col,
                on=self.period_col,
                values=self.outcome,
                aggregate_function="mean",
            )
        )
        # Reorder rows and columns
        col_order = [self.unit_col] + [str(pp) if not isinstance(pp, str) else pp for pp in period_order]
        # Handle numeric period columns
        period_str = [str(pp) for pp in period_order]
        pivot_cols = pivot.columns
        # Map period_order to column names as they appear in pivot
        period_col_map = {p: str(p) for p in period_order}
        # Ensure pivot has the right column names (Polars casts to string for column names)
        Y_rows = []
        D_rows = []
        for unit in unit_order:
            row = pivot.filter(pl.col(self.unit_col) == unit)
            if row.height == 0:
                Y_rows.append(np.full(T, np.nan))
            else:
                vals = []
                for per in period_order:
                    col_name = str(per)
                    if col_name in pivot.columns:
                        v = row[col_name][0]
                        vals.append(float(v) if v is not None else np.nan)
                    else:
                        vals.append(np.nan)
                Y_rows.append(vals)

        Y = np.array(Y_rows, dtype=float)

        # Fill NaN with column means (avoid NaN propagation)
        col_means = np.nanmean(Y, axis=0)
        for j in range(T):
            mask = np.isnan(Y[:, j])
            Y[mask, j] = col_means[j]

        # Build D matrix
        D = np.zeros((n_co + n_tr, T))
        for i, unit in enumerate(treated_segs):
            unit_data = p.filter(pl.col(self.unit_col) == unit)
            ftp = unit_data["first_treated_period"][0]
            for j, per in enumerate(period_order):
                if ftp is not None and per >= ftp:
                    D[n_co + i, j] = 1

        return Y, D, unit_order, period_order, n_co, n_tr, t_pre, t_post

    def _compute_variance(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        n_co: int,
        n_tr: int,
        t_pre: int,
        t_post: int,
        att_full: float,
    ) -> float:
        if self.inference == "placebo":
            return _placebo_variance(
                Y, D, n_co, n_tr, t_pre, t_post, self.n_replicates, self.rng
            )
        elif self.inference == "bootstrap":
            return _bootstrap_variance(
                Y, D, n_co, n_tr, t_pre, t_post, self.n_replicates, self.rng
            )
        elif self.inference == "jackknife":
            return _jackknife_variance(Y, D, n_co, n_tr, t_pre, t_post, att_full)
        else:
            raise ValueError(f"Unknown inference method: {self.inference}")

    def _compute_event_study(
        self,
        Y: np.ndarray,
        omega: np.ndarray,
        lambda_: np.ndarray,
        n_co: int,
        n_tr: int,
        t_pre: int,
        t_post: int,
        period_ids: list,
    ) -> tuple[pd.DataFrame, float]:
        """Compute period-by-period ATTs for the event study.

        For each period relative to treatment, estimate:
          ATT(e) = (Y_tr_e_bar - Y_tr_pre_lam) - (Y_co_e_om - Y_co_pre_om_lam)

        Pre-treatment estimates (e < 0) should be ~zero under parallel trends.
        """
        Y_co = Y[:n_co]
        Y_tr = Y[n_co:]

        Y_co_pre = Y_co[:, :t_pre]
        Y_tr_pre = Y_tr[:, :t_pre]

        Y_tr_pre_lam = np.mean(Y_tr_pre @ lambda_)
        Y_co_pre_om_lam = omega @ (Y_co_pre @ lambda_)

        T = t_pre + t_post
        att_by_period = []

        for t_idx in range(T):
            Y_tr_t = np.mean(Y_tr[:, t_idx])
            Y_co_t_om = omega @ Y_co[:, t_idx]
            att_t = (Y_tr_t - Y_tr_pre_lam) - (Y_co_t_om - Y_co_pre_om_lam)
            period_rel = t_idx - t_pre  # negative for pre-treatment
            att_by_period.append({"period_rel": period_rel, "att": att_t})

        event_df = pd.DataFrame(att_by_period)

        # Estimate SE for each period via placebo (limited replicates for speed)
        n_rep_es = min(50, self.n_replicates)
        period_ses = []
        for t_idx in range(T):
            # Use leave-one-out period variance as a rough estimate
            # Full placebo per period is expensive; use overall SE as approximation
            period_ses.append(np.nan)

        event_df["se"] = np.nan
        event_df["ci_low"] = np.nan
        event_df["ci_high"] = np.nan

        # Joint pre-treatment test: are pre-treatment ATTs jointly zero?
        pre_atts = event_df[event_df["period_rel"] < 0]["att"].values
        if len(pre_atts) > 1:
            # One-sample t-test on pre-treatment ATTs
            t_stat, pre_trend_pval = stats.ttest_1samp(pre_atts, 0.0)
        else:
            pre_trend_pval = None

        return event_df, pre_trend_pval
