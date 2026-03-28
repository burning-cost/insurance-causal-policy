"""
FCA evidence pack generator.

Produces structured regulatory output suitable for Consumer Duty outcome
monitoring reports and FCA evidence packs for rate change evaluation.

The FCA's multi-firm review of Consumer Duty implementation (2024) found that firms routinely failed to demonstrate causal
attribution between rate changes and observed outcomes. This module generates
the structured narrative and data quality documentation that addresses this gap.

Output formats:
- Markdown string (default) — suitable for embedding in reports
- JSON dict — for programmatic consumption or downstream formatting
- PDF via weasyprint (optional dependency)
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from ._types import SDIDResult, SensitivityResult, StaggeredResult


# ---------------------------------------------------------------------------
# Evidence pack builder
# ---------------------------------------------------------------------------


class FCAEvidencePack:
    """Generate FCA-compliant evidence pack for a rate change evaluation.

    Combines SDID results, sensitivity analysis, and data quality statistics
    into a structured document that addresses FCA Consumer Duty (PRIN 2A) requirements for
    causal attribution between rate changes and outcomes.

    Parameters
    ----------
    result : SDIDResult or StaggeredResult
        Primary estimation result.
    sensitivity : SensitivityResult or None
        Sensitivity analysis result (recommended for FCA evidence).
    product_line : str
        Product line (e.g., 'Motor', 'Home', 'Commercial').
    rate_change_date : str
        Date of rate change (ISO format or descriptive, e.g., '2023-Q1').
    rate_change_magnitude : str
        Description of the rate change (e.g., '+8.5% technical premium').
    analyst : str
        Name or team responsible for the analysis.
    panel_summary : dict or None
        Panel quality statistics from PolicyPanelBuilder.summary().
    additional_notes : str
        Any additional context (known threats to identification, caveats).
    """

    def __init__(
        self,
        result: "SDIDResult | StaggeredResult",
        sensitivity: Optional[SensitivityResult] = None,
        product_line: str = "Insurance",
        rate_change_date: str = "",
        rate_change_magnitude: str = "",
        analyst: str = "",
        panel_summary: Optional[dict] = None,
        additional_notes: str = "",
    ) -> None:
        self.result = result
        self.sensitivity = sensitivity
        self.product_line = product_line
        self.rate_change_date = rate_change_date
        self.rate_change_magnitude = rate_change_magnitude
        self.analyst = analyst
        self.panel_summary = panel_summary
        self.additional_notes = additional_notes
        self._generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the evidence pack as a Markdown document."""
        sections = [
            self._header(),
            self._executive_summary(),
            self._methodology(),
            self._results_table(),
            self._parallel_trends_section(),
            self._sensitivity_section(),
            self._data_quality_section(),
            self._caveats_section(),
            self._footer(),
        ]
        return "\n\n".join(s for s in sections if s.strip())

    def to_dict(self) -> dict:
        """Render the evidence pack as a JSON-serialisable dictionary."""
        r = self.result
        return {
            "metadata": {
                "product_line": self.product_line,
                "rate_change_date": self.rate_change_date,
                "rate_change_magnitude": self.rate_change_magnitude,
                "analyst": self.analyst,
                "generated_at": self._generated_at,
            },
            "estimation": {
                "att": r.att if hasattr(r, "att") else getattr(r, "att_overall", None),
                "se": r.se if hasattr(r, "se") else getattr(r, "se_overall", None),
                "ci_low": r.ci_low if hasattr(r, "ci_low") else getattr(r, "ci_low_overall", None),
                "ci_high": r.ci_high if hasattr(r, "ci_high") else getattr(r, "ci_high_overall", None),
                "pval": r.pval if hasattr(r, "pval") else getattr(r, "pval_overall", None),
                "significant_5pct": (r.pval if hasattr(r, "pval") else getattr(r, "pval_overall", 1.0)) < 0.05,
                "outcome": r.outcome_name,
            },
            "panel": self.panel_summary or {},
            "parallel_trends": {
                "pval": r.pre_trend_pval,
                "pass": r.pre_trends_pass,
            },
            "sensitivity": self.sensitivity.to_dataframe().to_dict("records") if self.sensitivity else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Render the evidence pack as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(self) -> str:
        date_str = f" — {self.rate_change_date}" if self.rate_change_date else ""
        magnitude_str = f" ({self.rate_change_magnitude})" if self.rate_change_magnitude else ""
        return (
            f"# Rate Change Evaluation: {self.product_line}{date_str}\n"
            f"**Rate change**: {self.rate_change_magnitude or 'Not specified'}{date_str}\n"
            f"**Analyst**: {self.analyst or 'Not specified'} | "
            f"**Generated**: {self._generated_at}\n\n"
            f"---"
        )

    def _executive_summary(self) -> str:
        r = self.result
        att = r.att if hasattr(r, "att") else r.att_overall
        se = r.se if hasattr(r, "se") else r.se_overall
        ci_low = r.ci_low if hasattr(r, "ci_low") else r.ci_low_overall
        ci_high = r.ci_high if hasattr(r, "ci_high") else r.ci_high_overall
        pval = r.pval if hasattr(r, "pval") else r.pval_overall

        direction = "decrease" if att < 0 else "increase"
        significant = pval < 0.05
        sig_text = "**statistically significant at the 5% level**" if significant else "not statistically significant at the 5% level"

        trend_status = (
            "Pre-treatment parallel trends test: **PASS**"
            if r.pre_trends_pass
            else f"Pre-treatment parallel trends test: **WARNING** (p={r.pre_trend_pval:.3f})"
        )

        return (
            f"## Executive Summary\n\n"
            f"The Synthetic Difference-in-Differences (SDID) analysis estimates that the rate change "
            f"caused a **{att:+.4f} {direction}** in {r.outcome_name} "
            f"(95% CI: {ci_low:+.4f} to {ci_high:+.4f}, p={pval:.4f}). "
            f"The result is {sig_text}.\n\n"
            f"{trend_status}"
        )

    def _methodology(self) -> str:
        r = self.result
        estimator_type = "Synthetic Difference-in-Differences (SDID)"
        if hasattr(r, "n_cohorts"):
            estimator_type = f"Callaway-Sant'Anna (2021) Staggered DiD ({r.n_cohorts} cohorts)"

        inference_text = ""
        if hasattr(r, "inference_method"):
            inference_text = f"\n- **Inference**: {r.inference_method} with {r.n_replicates} replicates"

        return (
            f"## Methodology\n\n"
            f"- **Estimator**: {estimator_type}\n"
            f"- **Reference**: Arkhangelsky, Athey, Hirshberg, Imbens, Wager (2021). "
            f"*Synthetic Difference-in-Differences*. American Economic Review 111(12): 4088–4118."
            f"{inference_text}\n"
            f"- **Unit weights**: Computed via CVXPY constrained least squares. "
            f"Intercept term absorbs level differences between treated and control segments.\n"
            f"- **Regularisation**: Theory-motivated zeta = (N_tr × T_post)^(1/4) × σ(ΔY)"
        )

    def _results_table(self) -> str:
        r = self.result
        att = r.att if hasattr(r, "att") else r.att_overall
        se = r.se if hasattr(r, "se") else r.se_overall
        ci_low = r.ci_low if hasattr(r, "ci_low") else r.ci_low_overall
        ci_high = r.ci_high if hasattr(r, "ci_high") else r.ci_high_overall
        pval = r.pval if hasattr(r, "pval") else r.pval_overall

        n_tr = r.n_treated if hasattr(r, "n_treated") else "N/A"
        n_co = r.n_control if hasattr(r, "n_control") else "N/A"
        t_pre = r.t_pre if hasattr(r, "t_pre") else "N/A"
        t_post = r.t_post if hasattr(r, "t_post") else "N/A"

        return (
            f"## Results\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| ATT ({r.outcome_name}) | {att:+.4f} |\n"
            f"| Standard error | {se:.4f} |\n"
            f"| 95% CI lower | {ci_low:+.4f} |\n"
            f"| 95% CI upper | {ci_high:+.4f} |\n"
            f"| p-value | {pval:.4f} |\n"
            f"| Treated segments | {n_tr} |\n"
            f"| Control segments | {n_co} |\n"
            f"| Pre-treatment periods | {t_pre} |\n"
            f"| Post-treatment periods | {t_post} |\n"
        )

    def _parallel_trends_section(self) -> str:
        r = self.result
        pval = r.pre_trend_pval
        passes = r.pre_trends_pass

        if pval is None:
            status = "Insufficient pre-treatment periods for formal test."
        elif passes:
            status = (
                f"**PASS** (p={pval:.3f}). Pre-treatment ATTs are jointly "
                f"indistinguishable from zero. The parallel trends assumption "
                f"is not rejected by the data."
            )
        else:
            status = (
                f"**WARNING** (p={pval:.3f}). Pre-treatment ATTs show "
                f"systematic deviation from zero, suggesting parallel trends "
                f"may not hold. Results should be interpreted with caution. "
                f"Common causes: Ogden rate changes, COVID lockdowns, "
                f"GIPP reforms, differential claims inflation by segment."
            )

        caution = (
            "\n\n> **Note**: A passing pre-trend test does not prove parallel "
            "trends. With few pre-treatment periods, the test has low power. "
            "Eight or more quarters of pre-treatment data is recommended."
        )

        return f"## Parallel Trends Test\n\n{status}{caution}"

    def _sensitivity_section(self) -> str:
        if self.sensitivity is None:
            return (
                "## Sensitivity Analysis\n\n"
                "Sensitivity analysis was not performed. "
                "Run `compute_sensitivity(result)` and include in the evidence pack "
                "for FCA submissions."
            )

        sens = self.sensitivity
        summary = sens.summary()

        # Build sensitivity table
        df = sens.to_dataframe()
        table_rows = "\n".join(
            f"| {row.m:.1f} | {row.att_lower:+.4f} | {row.att_upper:+.4f} |"
            for row in df.itertuples()
        )

        return (
            f"## Sensitivity Analysis (HonestDiD-style)\n\n"
            f"{summary}\n\n"
            f"M is the maximum post-treatment parallel trends violation, expressed "
            f"as a multiple of the pre-period standard deviation "
            f"({sens.pre_period_sd:.4f} in {self.result.outcome_name}).\n\n"
            f"| M | ATT lower bound | ATT upper bound |\n"
            f"|---|-----------------|------------------|\n"
            f"{table_rows}\n\n"
            f"> Reference: Rambachan & Roth (2023). *A More Credible Approach to "
            f"Parallel Trends*. Review of Economic Studies, rdad018."
        )

    def _data_quality_section(self) -> str:
        if not self.panel_summary:
            return ""

        ps = self.panel_summary
        return (
            f"## Data Quality\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total segments | {ps.get('n_segments', 'N/A')} |\n"
            f"| Treated segments | {ps.get('n_treated_segments', 'N/A')} |\n"
            f"| Control segments | {ps.get('n_control_segments', 'N/A')} |\n"
            f"| Time periods | {ps.get('n_periods', 'N/A')} |\n"
            f"| Panel cells | {ps.get('n_cells', 'N/A')} |\n"
            f"| Non-zero exposure cells | {ps.get('pct_nonzero_exposure', 'N/A'):.1f}% |\n"
            f"| Outcome metric | {ps.get('outcome', 'N/A')} |\n"
        )

    def _caveats_section(self) -> str:
        base = (
            "## Caveats and Limitations\n\n"
            "1. **Parallel trends assumption**: Results are valid conditional on "
            "approximate parallel trends holding after SDID reweighting. "
            "This cannot be proven from data — it must be justified by domain knowledge.\n\n"
            "2. **IBNR bias**: Incurred loss ratios for periods within 18 months of "
            "the analysis date may understate ultimate claims due to IBNR lag. "
            "Frequency-based outcomes are less affected.\n\n"
            "3. **Market-wide shocks**: Simultaneous market events (Ogden rate changes, "
            "FCA GIPP reforms, COVID lockdowns) affect all segments. If such events "
            "occurred in the analysis window, their effect may be confounded with "
            "the rate change effect.\n\n"
            "4. **External validity**: Results apply to the analysed segments. "
            "Generalisation to unanalysed segments or future periods requires "
            "additional assumption.\n\n"
            "5. **Inference method**: Placebo-based variance is valid under "
            "homoskedasticity across units. Bootstrap variance is preferred "
            "when this assumption is doubtful."
        )

        if self.additional_notes:
            base += f"\n\n**Additional notes**: {self.additional_notes}"

        return base

    def _footer(self) -> str:
        return (
            "---\n\n"
            "*Analysis performed using `insurance-causal-policy` (Burning Cost). "
            "Estimator: Synthetic Difference-in-Differences (Arkhangelsky et al. 2021). "
            "This document is for internal evidence purposes. The FCA has not prescribed "
            "a specific causal methodology — SDID is used because it is the same class of "
            "method the FCA itself used in its own GIPP remedies evaluation (EP25/2, July 2025).*"
        )
