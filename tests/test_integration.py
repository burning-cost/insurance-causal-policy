"""Integration tests: full pipeline from raw data to FCA evidence pack."""

import polars as pl
import pytest
from insurance_causal_policy import (
    FCAEvidencePack,
    PolicyPanelBuilder,
    SDIDEstimator,
    StaggeredEstimator,
    compute_sensitivity,
    make_synthetic_motor_panel,
    make_synthetic_panel_direct,
    plot_event_study,
    plot_unit_weights,
    pre_trend_summary,
)


class TestFullSDIDPipeline:
    """End-to-end SDID pipeline from synthetic raw data to FCA pack."""

    def setup_method(self):
        self.policy_df, self.claims_df, self.rate_log_df = make_synthetic_motor_panel(
            n_segments=60,
            n_periods=12,
            treat_fraction=0.25,
            true_att=-0.06,
            treatment_period=8,
            noise_sd=0.04,
            random_seed=77,
        )

    def test_panel_builds(self):
        builder = PolicyPanelBuilder(
            self.policy_df, self.claims_df, self.rate_log_df
        )
        panel = builder.build()
        assert isinstance(panel, pl.DataFrame)

    def test_sdid_fits(self):
        builder = PolicyPanelBuilder(
            self.policy_df, self.claims_df, self.rate_log_df
        )
        panel = builder.build()
        est = SDIDEstimator(panel, inference="placebo", n_replicates=50)
        result = est.fit()
        assert result.att < 0  # correct direction

    def test_sensitivity_runs(self):
        panel = PolicyPanelBuilder(
            self.policy_df, self.claims_df, self.rate_log_df
        ).build()
        result = SDIDEstimator(panel, inference="placebo", n_replicates=50).fit()
        sens = compute_sensitivity(result)
        assert len(sens.m_values) > 0

    def test_fca_pack_generates(self):
        panel = PolicyPanelBuilder(
            self.policy_df, self.claims_df, self.rate_log_df
        ).build()
        result = SDIDEstimator(panel, inference="placebo", n_replicates=50).fit()
        sens = compute_sensitivity(result)
        pack = FCAEvidencePack(
            result=result,
            sensitivity=sens,
            product_line="Motor",
            rate_change_date="2023-Q1",
            rate_change_magnitude="+6% technical premium",
        )
        md = pack.to_markdown()
        assert "Motor" in md
        assert len(md) > 200

    def test_event_study_plot_runs(self):
        panel = PolicyPanelBuilder(
            self.policy_df, self.claims_df, self.rate_log_df
        ).build()
        result = SDIDEstimator(panel, inference="placebo", n_replicates=50).fit()
        fig = plot_event_study(result)
        assert fig is not None

    def test_unit_weights_plot_runs(self):
        panel = PolicyPanelBuilder(
            self.policy_df, self.claims_df, self.rate_log_df
        ).build()
        result = SDIDEstimator(panel, inference="placebo", n_replicates=50).fit()
        fig = plot_unit_weights(result)
        assert fig is not None

    def test_pre_trend_summary_runs(self):
        panel = PolicyPanelBuilder(
            self.policy_df, self.claims_df, self.rate_log_df
        ).build()
        result = SDIDEstimator(panel, inference="placebo", n_replicates=50).fit()
        s = pre_trend_summary(result)
        assert isinstance(s, dict)

    def test_full_pipeline_json_output(self):
        panel = PolicyPanelBuilder(
            self.policy_df, self.claims_df, self.rate_log_df
        ).build()
        result = SDIDEstimator(panel, inference="placebo", n_replicates=50).fit()
        sens = compute_sensitivity(result)
        pack = FCAEvidencePack(result=result, sensitivity=sens)
        import json
        d = json.loads(pack.to_json())
        assert "estimation" in d
        assert "att" in d["estimation"]


class TestFrequencyPipeline:
    """Test the pipeline using frequency rather than loss ratio."""

    def test_frequency_outcome(self):
        policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
            n_segments=40, n_periods=10, random_seed=88
        )
        panel = PolicyPanelBuilder(
            policy_df, claims_df, rate_log_df, outcome="frequency"
        ).build()
        assert "frequency" in panel.columns
        result = SDIDEstimator(
            panel, outcome="frequency", inference="placebo", n_replicates=30
        ).fit()
        assert isinstance(result.att, float)


class TestSmallPanelDirect:
    """Direct panel tests with controlled dimensions."""

    def test_minimal_viable_panel(self):
        """SDID should run on the smallest viable panel."""
        panel = make_synthetic_panel_direct(
            n_control=20, n_treated=5, t_pre=4, t_post=2,
            true_att=-0.05, random_seed=1
        )
        result = SDIDEstimator(
            panel, outcome="loss_ratio",
            inference="jackknife", n_replicates=30
        ).fit()
        assert isinstance(result.att, float)

    def test_bootstrap_inference_pipeline(self):
        panel = make_synthetic_panel_direct(
            n_control=30, n_treated=10, t_pre=6, t_post=3,
            true_att=-0.06, random_seed=2
        )
        result = SDIDEstimator(
            panel, outcome="loss_ratio",
            inference="bootstrap", n_replicates=30
        ).fit()
        assert result.se > 0

    def test_summary_method(self):
        panel = make_synthetic_panel_direct(
            n_control=25, n_treated=8, t_pre=5, t_post=3,
            true_att=-0.05, random_seed=3
        )
        result = SDIDEstimator(
            panel, outcome="loss_ratio", inference="placebo", n_replicates=30
        ).fit()
        summary = result.summary()
        assert "loss_ratio" in summary
        assert "placebo" in summary

    def test_fca_summary_method(self):
        panel = make_synthetic_panel_direct(
            n_control=25, n_treated=8, t_pre=5, t_post=3,
            true_att=-0.05, random_seed=4
        )
        result = SDIDEstimator(
            panel, outcome="loss_ratio", inference="placebo", n_replicates=30
        ).fit()
        txt = result.to_fca_summary(product_line="Home", rate_change_date="2024-Q1")
        assert "Home" in txt
        assert "loss_ratio" in txt


class TestImports:
    """Verify all public API is importable and accessible."""

    def test_all_public_classes_importable(self):
        from insurance_causal_policy import (
            FCAEvidencePack,
            PolicyPanelBuilder,
            SDIDEstimator,
            SDIDResult,
            SDIDWeights,
            SensitivityResult,
            StaggeredEstimator,
            StaggeredResult,
            build_panel_from_pandas,
            compute_sensitivity,
            make_synthetic_motor_panel,
            make_synthetic_panel_direct,
            plot_event_study,
            plot_sensitivity,
            plot_synthetic_trajectory,
            plot_unit_weights,
            pre_trend_summary,
        )

    def test_version_string(self):
        import insurance_causal_policy
        assert hasattr(insurance_causal_policy, "__version__")
        assert isinstance(insurance_causal_policy.__version__, str)
        assert "." in insurance_causal_policy.__version__
