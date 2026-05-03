from __future__ import annotations

from pathlib import Path

from stock_analysis.config import load_config


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
