"""
DoublyRobustSC estimator: OLS synthetic control with doubly robust ATT.

Implements Sant'Anna, Shaikh, Syrgkanis (2025) "Doubly Robust Synthetic
Controls" (arXiv:2503.11375) from first principles. No reference implementation
exists — built directly from the paper.

The estimator is doubly robust: it produces a consistent ATT estimate if
EITHER the synthetic control weights correctly reproduce the treated unit's
counterfactual trend, OR if parallel trends holds (possibly after covariate
reweighting via propensity scores).

Key differences from SDID:
  - SC weights estimated via OLS on group conditional mean matrix (not CVXPY
    constrained quadratic program). Negative weights are valid and expected.
  - Explicit propensity score component: ratio r_{1,g} = P(G=1|X) / P(G=g|X).
    In the no-covariate case (aggregate panel), this simplifies to N_tr/N_g.
  - Inference uses multiplier bootstrap with Exp(1)-1 weights (Bayesian
    bootstrap), valid under EITHER parallel trends or SC identification regime.
  - No CVXPY dependency.

The no-covariate (aggregate) special case:
  When X is absent, propensity ratios r_{1,g} = n_tr / n_g (group size ratios),
  conditional means m_{g,t} are sample means within each control group-period
  cell, and m_Delta(x) = mean(DeltaY_co) is the pooled control trend. Cross-
  fitting has no effect in this case (nuisance objects are sample means, not
  estimated via local polynomial).

ATT moment function (per-observation, unit-normalised):
  phi_i = (1/pi_1) * [G_{1,i} - sum_g w_g * r_{1,g} * G_{g,i}] * (DeltaY_i - m_delta)

  For treated unit i:  phi_i = (DeltaY_i - m_delta) / pi_1
  For control unit j in group g:  phi_j = -w_g * r_{1,g} * (DeltaY_j - m_delta) / pi_1

  ATT = mean(phi_i over all units)

Bootstrap:
  theta_boot_b = att + mean(W_i * phi_i)
  where W_i ~ Exp(1) - 1, iid. Valid for both V_PT and V_SC variance regimes.
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from ._types import DRSCResult, DRSCWeights


# ---------------------------------------------------------------------------
# SC weight solver (OLS, unconstrained)
# ---------------------------------------------------------------------------


def _solve_sc_weights_ols(
    mu_tr_pre: np.ndarray,
    mu_co_pre: np.ndarray,
) -> np.ndarray:
    """Estimate SC weights via OLS on pre-treatment group conditional means.

    Implements Equation 10 from Sant'Anna, Shaikh, Syrgkanis (2025).

    The OLS system is:
        mu_tr[t] ~ sum_{g=1}^{N_co-1} w_g * (mu_co_g[t] - mu_co_last[t])
                   + w_last * mu_co_last[t]

    Rearranged: solve for w_0 = (w_1, ..., w_{N_co-1}) via least squares on
    the matrix M (T_pre-1 x N_co-1) of demeaned control means, with the
    constraint w_last = 1 - sum(w_0).

    This simplification handles the N_co = 1 case (no freedom — trivially
    solved) and N_co >= 2.

    Negative weights are valid and indicate extrapolation beyond the convex
    hull of the control group — unlike SDID, we do NOT constrain omega >= 0.

    Parameters
    ----------
    mu_tr_pre : array of shape (T_pre,)
        Pre-treatment group mean of the treated units, period by period.
    mu_co_pre : array of shape (N_co, T_pre)
        Pre-treatment group means of each control unit (or group).

    Returns
    -------
    w : array of shape (N_co,)
        SC weights summing to 1. May contain negative values.
    """
    n_co, t_pre = mu_co_pre.shape

    if n_co == 1:
        # Trivial: single control group gets weight 1
        return np.array([1.0])

    # Use all T_pre periods (not T_pre - 1 as in the abstract formulation)
    # The paper uses T_pre - 1 pre-periods after differencing, but with
    # aggregate data the full pre-treatment mean matrix is more stable.
    # We solve: w @ mu_co_pre.T ~ mu_tr_pre  (T_pre equations, N_co unknowns)
    # subject to sum(w) = 1.
    #
    # Impose the sum constraint by parametrisation:
    #   w_g = w_0_g for g = 1,...,N_co-1
    #   w_{N_co} = 1 - sum(w_0)
    # Substituting: mu_co_pre[:-1,:].T @ w_0 + mu_co_pre[-1,:] * (1 - sum(w_0)) ~ mu_tr_pre
    # => (mu_co_pre[:-1,:] - mu_co_pre[-1:,:]).T @ w_0 ~ mu_tr_pre - mu_co_pre[-1,:]
    # This is the M matrix formulation in Eq. 10.

    mu_last = mu_co_pre[-1, :]      # (T_pre,)
    M = (mu_co_pre[:-1, :] - mu_co_pre[-1:, :]).T  # (T_pre, N_co-1)
    rhs = mu_tr_pre - mu_last        # (T_pre,)

    # OLS: w_0 = (M'M)^{-1} M' rhs  — use lstsq for numerical stability
    w_0, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

    w_last = 1.0 - w_0.sum()
    w = np.concatenate([w_0, [w_last]])

    return w


# ---------------------------------------------------------------------------
# Moment function
# ---------------------------------------------------------------------------


def _compute_moment_function(
    delta_y: np.ndarray,
    m_delta: float,
    sc_weights: np.ndarray,
    prop_ratios: np.ndarray,
    is_treated: np.ndarray,
    group_idx: np.ndarray,
    pi_1: float,
    n_co: int,
) -> np.ndarray:
    """Compute the per-unit DRSC moment function phi_i.

    For each unit i:
      phi_i = (1/pi_1) * [1{G=1,i} - sum_g w_g * r_{1,g} * 1{G=g,i}] * (DeltaY_i - m_delta)

    In the no-covariate case with aggregate panel (one observation per unit):
      Treated unit i:    phi_i = (DeltaY_i - m_delta) / pi_1
      Control unit j:    phi_j = -w_{g(j)} * r_{1,g(j)} * (DeltaY_j - m_delta) / pi_1

    ATT = mean(phi_i over all N units)

    Parameters
    ----------
    delta_y : (N,) outcome differences (post mean - last pre period)
    m_delta : float, mean control trend (pooled across all controls)
    sc_weights : (N_co,) OLS SC weights
    prop_ratios : (N_co,) propensity ratios r_{1,g} = n_tr / n_g for each control
    is_treated : (N,) bool indicator for treated units
    group_idx : (N,) int, group index for each unit (0..N_co-1 for controls)
    pi_1 : float, proportion of treated units
    n_co : int, number of control units

    Returns
    -------
    phi : (N,) moment function values
    """
    N = len(delta_y)
    phi = np.zeros(N)
    residual = delta_y - m_delta

    # Treated contribution
    phi[is_treated] = residual[is_treated] / pi_1

    # Control contribution
    control_mask = ~is_treated
    control_idx_arr = np.where(control_mask)[0]
    for k, unit_idx in enumerate(control_idx_arr):
        g = group_idx[unit_idx]   # which control group (0-based index)
        phi[unit_idx] = -sc_weights[g] * prop_ratios[g] * residual[unit_idx] / pi_1

    return phi


# ---------------------------------------------------------------------------
# Core fit function
# ---------------------------------------------------------------------------


def _fit_drsc_core(
    Y: np.ndarray,
    n_co: int,
    n_tr: int,
    t_pre: int,
    t_post: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """Core DRSC fit: compute SC weights, ATT, and per-unit moment function.

    Implements the no-covariate special case where each control unit is its
    own group. Propensity ratios simplify to r_{1,g} = n_tr / 1 = n_tr for
    per-unit panel.

    Actually: in the aggregate panel setting, N_g = 1 (each control is its own
    group of size 1). So pi_g = 1/N and pi_1 = n_tr/N. Then:
      r_{1,g} = pi_1 / pi_g = (n_tr/N) / (1/N) = n_tr

    For a unit-count balanced interpretation, we normalise so that the weights
    encode the SC matching without double-counting the propensity ratio.
    In practice: the moment function contribution of control j simplifies to:
      phi_j = -w_g * n_tr * (DeltaY_j - m_delta) / (n_tr/N)
            = -w_g * N * (DeltaY_j - m_delta)

    But since ATT = mean(phi_i), taking mean over N units:
      ATT = (1/N) * [sum_tr (DeltaY_i - m_delta) / pi_1
                     - sum_co w_{g(j)} * r_{1,j} * (DeltaY_j - m_delta) / pi_1]

    With pi_1 = n_tr/N and r_{1,j} = n_tr (per unit), this simplifies to:
      ATT = mean(DeltaY_tr) - m_delta_tr_adj - (SC-weighted mean DeltaY_co - m_delta)
    where the m_delta terms cancel (they're the pooled control mean), leaving:
      ATT = mean(DeltaY_tr) - w @ mean(DeltaY_co per unit)

    This is equivalent to the standard SC-DiD formula with OLS weights.

    Returns
    -------
    att : float
    sc_weights : (N_co,) SC weights
    prop_ratios : (N_co,) propensity ratios
    phi : (N,) per-unit moment function
    m_delta : float pooled control trend
    """
    N = n_co + n_tr

    Y_co = Y[:n_co]    # (N_co, T)
    Y_tr = Y[n_co:]    # (N_tr, T)

    Y_co_pre = Y_co[:, :t_pre]
    Y_co_post = Y_co[:, t_pre:]
    Y_tr_pre = Y_tr[:, :t_pre]
    Y_tr_post = Y_tr[:, t_pre:]

    # Pre-treatment group means
    # In the unit-level panel: each control is its own group (N_g=1 per group)
    mu_co_pre = Y_co_pre    # (N_co, T_pre) — each row is a "group" mean
    mu_tr_pre = np.mean(Y_tr_pre, axis=0)   # (T_pre,) treated group mean

    # Post/pre summary: DeltaY = mean(post) - last pre
    # Using the last pre-period as the base period for DiD
    last_pre = t_pre - 1
    delta_y_tr = np.mean(Y_tr_post, axis=1) - Y_tr[:, last_pre]    # (N_tr,)
    delta_y_co = np.mean(Y_co_post, axis=1) - Y_co[:, last_pre]    # (N_co,)

    # SC weights via OLS
    sc_weights = _solve_sc_weights_ols(mu_tr_pre, mu_co_pre)

    # Pooled control trend (m_delta = mean(DeltaY_co), no covariates)
    m_delta = float(np.mean(delta_y_co))

    # Propensity ratios: r_{1,g} = pi_1 / pi_g
    # In unit-level panel: pi_1 = n_tr/N, pi_g = 1/N (each control is 1 unit)
    # => r_{1,g} = n_tr for all control units
    pi_1 = n_tr / N
    prop_ratios = np.full(n_co, float(n_tr))   # r_{1,g} = n_tr for each unit

    # Per-unit DeltaY array (controls first, then treated — same ordering as Y)
    delta_y_all = np.concatenate([delta_y_co, delta_y_tr])  # (N,)

    # is_treated indicator
    is_treated = np.zeros(N, dtype=bool)
    is_treated[n_co:] = True

    # group index for each control: control j belongs to group j (its own)
    group_idx = np.arange(N, dtype=int)
    group_idx[n_co:] = -1    # treated: not used in control contribution

    # Moment function
    phi = _compute_moment_function(
        delta_y_all, m_delta, sc_weights, prop_ratios,
        is_treated, group_idx, pi_1, n_co,
    )

    att = float(np.mean(phi))

    return att, sc_weights, prop_ratios, phi, m_delta


# ---------------------------------------------------------------------------
# Inference: multiplier bootstrap (Bayesian bootstrap)
# ---------------------------------------------------------------------------


def _multiplier_bootstrap(
    phi: np.ndarray,
    att: float,
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Multiplier bootstrap for DRSC inference.

    Uses Exp(1)-1 weights (Bayesian bootstrap), which satisfies E[W]=0,
    Var[W]=1. Valid under both PT and SC identification regimes.

    theta_boot_b = att + mean(W_i * phi_i)

    where W_i ~ Exp(1) - 1, iid across units, independent across bootstrap
    replicates.

    Parameters
    ----------
    phi : (N,) per-unit moment function values
    att : float, original ATT estimate (= mean(phi))
    n_replicates : int
    rng : numpy Generator

    Returns
    -------
    boot_atts : (n_replicates,) bootstrap ATT replicates
    """
    N = len(phi)
    # Shape: (n_replicates, N)
    W = rng.exponential(scale=1.0, size=(n_replicates, N)) - 1.0
    # Boot ATT: att + mean(W_i * phi_i) per replicate
    boot_atts = att + np.mean(W * phi[np.newaxis, :], axis=1)
    return boot_atts


def _analytic_variance(
    phi: np.ndarray,
) -> float:
    """Analytic variance of the ATT under parallel trends (V_PT).

    V_PT = Var(phi_i) / N

    Valid when parallel trends holds and nuisance functions are correctly
    specified. Under SC-only identification, this understates the variance
    (no Neyman orthogonality), so the bootstrap is preferred in practice.

    Parameters
    ----------
    phi : (N,) per-unit moment function values

    Returns
    -------
    variance : float
    """
    N = len(phi)
    return float(np.var(phi, ddof=1) / N)


# ---------------------------------------------------------------------------
# Main DoublyRobustSC class
# ---------------------------------------------------------------------------


class DoublyRobustSCEstimator:
    """Doubly Robust Synthetic Control estimator for insurance panels.

    Implements Sant'Anna, Shaikh, Syrgkanis (2025) arXiv:2503.11375 for
    insurance segment panels. Handles simultaneous adoption only — use
    StaggeredEstimator for staggered adoption.

    The estimator is doubly robust: the ATT is consistently estimated if
    EITHER the synthetic control weights w correctly reproduce the treated
    group's counterfactual trend (SC identification), OR parallel trends
    holds after covariate adjustment. You don't need both.

    In the no-covariate (aggregate panel) case implemented here, SC weights
    are estimated via unconstrained OLS. Negative weights are expected and
    valid. Propensity score ratios simplify to group size ratios.

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
        Variance estimation method: 'bootstrap' (default) or 'analytic'.
        Bootstrap is preferred — it is valid under both PT and SC regimes.
        Analytic variance is valid only under parallel trends.
    n_replicates : int
        Number of bootstrap replicates. Default: 500.
    random_seed : int
        Reproducibility seed. Default: 42.

    References
    ----------
    Sant'Anna, P., Shaikh, A., Syrgkanis, V. (2025). Doubly Robust Synthetic
    Controls. arXiv:2503.11375.
    """

    def __init__(
        self,
        panel: pl.DataFrame,
        outcome: str = "loss_ratio",
        treatment_col: str = "treated",
        unit_col: str = "segment_id",
        period_col: str = "period",
        inference: Literal["bootstrap", "analytic"] = "bootstrap",
        n_replicates: int = 500,
        random_seed: int = 42,
    ) -> None:
        self.panel = panel
        self.outcome = outcome
        self.treatment_col = treatment_col
        self.unit_col = unit_col
        self.period_col = period_col
        self.inference = inference
        self.n_replicates = n_replicates
        self._random_seed = random_seed
        self._result: Optional[DRSCResult] = None

    def fit(self) -> DRSCResult:
        """Estimate the ATT and run inference.

        Returns a DRSCResult with ATT, confidence interval, SC weights,
        propensity ratios, and event study data.
        """
        # Fresh RNG each fit() call ensures reproducibility regardless of call order
        self.rng = np.random.default_rng(self._random_seed)
        Y, D, unit_ids, period_ids, n_co, n_tr, t_pre, t_post = self._prepare_matrices()

        # Core fit
        att, sc_weights, prop_ratios, phi, m_delta = _fit_drsc_core(
            Y, n_co, n_tr, t_pre, t_post
        )

        # Inference
        var = self._compute_variance(phi, att)
        se = np.sqrt(max(var, 0.0))
        ci_low = att - 1.96 * se
        ci_high = att + 1.96 * se
        pval = 2 * (1 - stats.norm.cdf(abs(att) / max(se, 1e-10)))

        # Build weight object
        co_unit_ids = unit_ids[:n_co]
        weights = DRSCWeights(
            sc_weights=pd.Series(sc_weights, index=co_unit_ids, name="w"),
            propensity_ratios=pd.Series(prop_ratios, index=co_unit_ids, name="r"),
            m_delta=m_delta,
        )

        # Event study
        event_study, pre_trend_pval = self._compute_event_study(
            Y, sc_weights, n_co, n_tr, t_pre, t_post, period_ids
        )

        result = DRSCResult(
            att=att,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            pval=pval,
            weights=weights,
            phi=phi,
            pre_trend_pval=pre_trend_pval,
            event_study=event_study,
            n_treated=n_tr,
            n_control=n_co,
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

        treated_segs = sorted(
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
        first_treated = p.filter(pl.col(self.treatment_col) == 1)[self.period_col].min()
        all_periods = sorted(p[self.period_col].unique().to_list())
        pre_periods = [t for t in all_periods if t < first_treated]
        post_periods = [t for t in all_periods if t >= first_treated]

        if len(pre_periods) == 0:
            raise ValueError("No pre-treatment periods found. Cannot estimate DRSC.")
        if len(post_periods) == 0:
            raise ValueError("No post-treatment periods found. Cannot estimate DRSC.")

        n_co = len(control_segs)
        n_tr = len(treated_segs)
        t_pre = len(pre_periods)
        t_post = len(post_periods)
        T = t_pre + t_post

        # SC requirement: T_pre >= N_co for OLS to be well-identified.
        # With T_pre < N_co the system is underdetermined; lstsq still works
        # (minimum-norm solution) but warn the user.
        if t_pre < n_co:
            warnings.warn(
                f"DRSC SC weights are under-identified: T_pre={t_pre} < N_co={n_co}. "
                "OLS minimum-norm solution used. Results may be unstable. "
                "Consider using fewer control units or more pre-treatment periods.",
                UserWarning,
                stacklevel=3,
            )

        unit_order = control_segs + treated_segs
        period_order = pre_periods + post_periods

        # Pivot to matrix
        pivot = p.pivot(
            index=self.unit_col,
            on=self.period_col,
            values=self.outcome,
            aggregate_function="mean",
        )

        Y_rows = []
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

        # Fill NaN with column means
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

    def _compute_variance(self, phi: np.ndarray, att: float) -> float:
        if self.inference == "bootstrap":
            boot_atts = _multiplier_bootstrap(phi, att, self.n_replicates, self.rng)
            return float(np.var(boot_atts, ddof=1))
        elif self.inference == "analytic":
            return _analytic_variance(phi)
        else:
            raise ValueError(f"Unknown inference method: {self.inference!r}")

    def _compute_event_study(
        self,
        Y: np.ndarray,
        sc_weights: np.ndarray,
        n_co: int,
        n_tr: int,
        t_pre: int,
        t_post: int,
        period_ids: list,
    ) -> tuple[pd.DataFrame, Optional[float]]:
        """Compute period-by-period ATTs for the event study.

        For each period t, DiD estimate using SC-weighted control:
          ATT(t) = mean(Y_tr[:, t]) - w @ mean(Y_co[:, t])
                   - [mean(Y_tr_pre_mean) - w @ mean(Y_co_pre_mean)]

        where Y_tr_pre_mean and Y_co_pre_mean are averages over all pre-periods,
        providing a clean baseline analogous to standard event study design.
        """
        Y_co = Y[:n_co]
        Y_tr = Y[n_co:]

        T = t_pre + t_post

        # Pre-treatment baselines (uniform average over all pre-periods)
        Y_tr_pre_mean = float(np.mean(Y_tr[:, :t_pre]))
        Y_co_pre_mean = float(sc_weights @ np.mean(Y_co[:, :t_pre], axis=1))

        att_by_period = []
        for t_idx in range(T):
            Y_tr_t = float(np.mean(Y_tr[:, t_idx]))
            Y_co_t_w = float(sc_weights @ Y_co[:, t_idx])
            att_t = (Y_tr_t - Y_tr_pre_mean) - (Y_co_t_w - Y_co_pre_mean)
            period_rel = t_idx - t_pre
            att_by_period.append({"period_rel": period_rel, "att": att_t})

        event_df = pd.DataFrame(att_by_period)

        # Bootstrap SE per period — resample phi and refit each period
        # For the event study, use the simple bootstrap on the period-ATT
        # quantities rather than the full multiplier bootstrap (too slow
        # period-by-period). We resample the outcome matrix instead.
        n_rep_es = min(100, self.n_replicates)
        period_boot_atts = np.zeros((n_rep_es, T))

        for b in range(n_rep_es):
            co_idx = self.rng.choice(n_co, size=n_co, replace=True)
            tr_idx = self.rng.choice(n_tr, size=n_tr, replace=True)

            Y_co_b = Y_co[co_idx]
            Y_tr_b = Y_tr[tr_idx]

            # Recompute SC weights on bootstrap sample
            mu_co_pre_b = Y_co_b[:, :t_pre]
            mu_tr_pre_b = np.mean(Y_tr_b[:, :t_pre], axis=0)
            try:
                w_b = _solve_sc_weights_ols(mu_tr_pre_b, mu_co_pre_b)
            except Exception:
                w_b = sc_weights

            bl_tr = float(np.mean(Y_tr_b[:, :t_pre]))
            bl_co = float(w_b @ np.mean(Y_co_b[:, :t_pre], axis=1))
            for t_idx in range(T):
                Y_tr_t = float(np.mean(Y_tr_b[:, t_idx]))
                Y_co_t = float(w_b @ Y_co_b[:, t_idx])
                period_boot_atts[b, t_idx] = (Y_tr_t - bl_tr) - (Y_co_t - bl_co)

        period_ses = np.std(period_boot_atts, axis=0, ddof=1)
        event_df["se"] = period_ses
        z_crit = 1.96
        event_df["ci_low"] = event_df["att"] - z_crit * period_ses
        event_df["ci_high"] = event_df["att"] + z_crit * period_ses

        # Joint pre-trend test
        pre_atts = event_df[event_df["period_rel"] < 0]["att"].values
        if len(pre_atts) > 1:
            _, pre_trend_pval = stats.ttest_1samp(pre_atts, 0.0)
        else:
            pre_trend_pval = None

        return event_df, pre_trend_pval
