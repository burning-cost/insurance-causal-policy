"""
insurance-causal-policy
=======================

Causal evaluation of insurance pricing interventions using Synthetic
Difference-in-Differences (SDID) and Callaway-Sant'Anna (2021).

Did your rate change actually reduce your loss ratio? This library provides
the tools to measure it causally.

Quick start
-----------
>>> from insurance_causal_policy import (
...     PolicyPanelBuilder,
...     SDIDEstimator,
...     StaggeredEstimator,
...     FCAEvidencePack,
...     compute_sensitivity,
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

References
----------
- Arkhangelsky, Athey, Hirshberg, Imbens, Wager (2021).
  Synthetic Difference-in-Differences. AER 111(12): 4088-4118.
- Callaway & Sant'Anna (2021). DiD with Multiple Time Periods.
  Journal of Econometrics 225(2): 200-230.
- Rambachan & Roth (2023). A More Credible Approach to Parallel Trends.
  Review of Economic Studies, rdad018.
- FCA TR24/2 (2024). Insurance multi-firm outcomes monitoring review.
- FCA EP25/2 (2025). Evaluation of GIPP remedies.
"""

from ._panel import PolicyPanelBuilder, build_panel_from_pandas
from ._sdid import SDIDEstimator
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
from ._types import SDIDResult, SDIDWeights, StaggeredResult, SensitivityResult

__version__ = "0.1.7"

__all__ = [
    # Panel construction
    "PolicyPanelBuilder",
    "build_panel_from_pandas",
    # Estimators
    "SDIDEstimator",
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
    "StaggeredResult",
    "SensitivityResult",
    # Version
    "__version__",
]
