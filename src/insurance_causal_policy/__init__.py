"""
insurance-causal-policy
=======================

Causal evaluation of insurance pricing interventions using Synthetic
Difference-in-Differences (SDID), Callaway-Sant'Anna (2021), and Doubly
Robust Synthetic Controls (DRSC, Sant'Anna, Shaikh, Syrgkanis 2025).

Did your rate change actually reduce your loss ratio? This library provides
the tools to measure it causally.

Quick start — SDID
------------------
>>> from insurance_causal_policy import (
...     PolicyPanelBuilder,
...     SDIDEstimator,
...     make_synthetic_motor_panel,
... )
>>> policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
...     n_segments=100, n_periods=12, true_att=-0.08
... )
>>> builder = PolicyPanelBuilder(policy_df, claims_df, rate_log_df)
>>> panel = builder.build()
>>> est = SDIDEstimator(panel, outcome="loss_ratio", inference="placebo")
>>> result = est.fit()
>>> print(result.summary())

Quick start — DRSC (doubly robust)
-----------------------------------
>>> from insurance_causal_policy import DoublyRobustSCEstimator, make_synthetic_panel_direct
>>> panel = make_synthetic_panel_direct(n_control=30, n_treated=10, t_pre=8, t_post=4, true_att=-0.06)
>>> est = DoublyRobustSCEstimator(panel, outcome="loss_ratio", inference="bootstrap")
>>> result = est.fit()
>>> print(result.summary())

Use DRSC over SDID when:
  - Small donor pool (5-15 control segments) — DR property provides a safety net
  - Post-disruption periods (COVID, GIPP) where parallel trends is uncertain
  - You want formal double robustness guarantees (consistent under EITHER SC OR PT)

References
----------
- Arkhangelsky, Athey, Hirshberg, Imbens, Wager (2021).
  Synthetic Difference-in-Differences. AER 111(12): 4088-4118.
- Callaway & Sant'Anna (2021). DiD with Multiple Time Periods.
  Journal of Econometrics 225(2): 200-230.
- Rambachan & Roth (2023). A More Credible Approach to Parallel Trends.
  Review of Economic Studies, rdad018.
- Sant'Anna, Shaikh, Syrgkanis (2025). Doubly Robust Synthetic Controls.
  arXiv:2503.11375.
- FCA TR24/2 (2024). Insurance multi-firm outcomes monitoring review.
- FCA EP25/2 (2025). Evaluation of GIPP remedies.
"""

from ._panel import PolicyPanelBuilder, build_panel_from_pandas
from ._sdid import SDIDEstimator
from ._drsc import DoublyRobustSCEstimator
from ._staggered import StaggeredEstimator
from ._event_study import (
    plot_event_study,
    plot_synthetic_trajectory,
    plot_unit_weights,
    pre_trend_summary,
)
from ._sensitivity import compute_sensitivity, plot_sensitivity
from ._evidence import FCAEvidencePack
from ._synthetic import make_synthetic_motor_panel, make_synthetic_panel_direct
from ._types import SDIDResult, SDIDWeights, DRSCResult, DRSCWeights, StaggeredResult, SensitivityResult

__version__ = "0.2.0"

__all__ = [
    # Panel construction
    "PolicyPanelBuilder",
    "build_panel_from_pandas",
    # Estimators
    "SDIDEstimator",
    "DoublyRobustSCEstimator",
    "StaggeredEstimator",
    # Diagnostics
    "plot_event_study",
    "plot_synthetic_trajectory",
    "plot_unit_weights",
    "pre_trend_summary",
    # Sensitivity
    "compute_sensitivity",
    "plot_sensitivity",
    # Evidence
    "FCAEvidencePack",
    # Synthetic data
    "make_synthetic_motor_panel",
    "make_synthetic_panel_direct",
    # Types
    "SDIDResult",
    "SDIDWeights",
    "DRSCResult",
    "DRSCWeights",
    "StaggeredResult",
    "SensitivityResult",
    # Version
    "__version__",
]
