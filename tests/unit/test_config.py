from __future__ import annotations

from pathlib import Path

from stock_analysis.config import apply_env_overrides, load_config


def test_load_config_defaults_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text("optimizer:\n  max_weight: 0.2\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.optimizer.max_weight == 0.2
    assert config.live_account.enabled is False
    assert config.live_account.cashflow_source == "scenario"
    assert config.supabase.enabled is False
    assert config.supabase.url_env == "SUPABASE_URL"
    assert config.supabase.key_env == "SUPABASE_SERVICE_ROLE_KEY"
    assert config.tableau.publish_enabled is False
    assert config.gcp.enabled is False
    assert config.gcp.bucket is None


def test_load_config_normalizes_gcp_bucket(tmp_path: Path) -> None:
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        """
gcp:
  enabled: true
  bucket: gs://stock-analysis-medallion-prod/
""".lstrip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.gcp.enabled is True
    assert config.gcp.bucket == "stock-analysis-medallion-prod"


def test_apply_env_overrides_supports_cloud_run_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        """
gcp:
  enabled: true
  bucket: stock-analysis-medallion-template
run:
  run_id: yaml-run
""".lstrip(),
        encoding="utf-8",
    )
    config = load_config(config_path)

    overridden = apply_env_overrides(
        config,
        {
            "STOCK_ANALYSIS_GCP_PROJECT_ID": "proyectodata",
            "STOCK_ANALYSIS_GCP_REGION": "us-central1",
            "STOCK_ANALYSIS_GCP_BUCKET": "gs://stock-analysis-proyectodata/",
            "STOCK_ANALYSIS_GCP_GCS_PREFIX": "cloud-runs",
            "STOCK_ANALYSIS_GCP_BIGQUERY_DATASET_GOLD": "stock_analysis_dashboard",
            "STOCK_ANALYSIS_GCP_MODEL_REGISTRY_PREFIX": "model-registry",
            "STOCK_ANALYSIS_GCP_MODEL_ARTIFACT_URI": "",
            "STOCK_ANALYSIS_RUN_ID": "env-run",
            "STOCK_ANALYSIS_RUN_AS_OF_DATE": "2026-05-01",
        },
    )

    assert overridden.gcp.project_id == "proyectodata"
    assert overridden.gcp.region == "us-central1"
    assert overridden.gcp.bucket == "stock-analysis-proyectodata"
    assert overridden.gcp.gcs_prefix == "cloud-runs"
    assert overridden.gcp.bigquery_dataset_gold == "stock_analysis_dashboard"
    assert overridden.gcp.model_registry_prefix == "model-registry"
    assert overridden.gcp.model_artifact_uri is None
    assert overridden.run.run_id == "env-run"
    assert overridden.run.as_of_date.isoformat() == "2026-05-01"
