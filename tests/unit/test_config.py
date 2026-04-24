from __future__ import annotations

from pathlib import Path

from stock_analysis.config import load_config


def test_load_config_defaults_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text("optimizer:\n  max_weight: 0.2\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.optimizer.max_weight == 0.2
    assert config.tableau.publish_enabled is False
