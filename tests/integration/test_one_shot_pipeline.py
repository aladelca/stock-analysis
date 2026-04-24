from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stock_analysis.pipeline.one_shot import run_one_shot
from stock_analysis.tableau.export import export_existing_run_for_tableau


def test_one_shot_pipeline_writes_outputs(
    sample_html,
    sample_config,
    static_price_provider,
) -> None:
    result = run_one_shot(
        sample_config,
        universe_html=sample_html,
        price_provider=static_price_provider,
    )
    output_root = Path(result.output_root)

    assert output_root == sample_config.run.output_root / "runs" / "test-run"
    assert (output_root / "raw" / "prices" / "static_prices.csv").exists()
    assert (output_root / "bronze" / "sp500_constituents.parquet").exists()
    assert (output_root / "silver" / "asset_daily_features.parquet").exists()
    assert (output_root / "silver" / "asset_daily_features_panel.parquet").exists()
    assert (output_root / "silver" / "spy_daily.parquet").exists()
    assert (output_root / "silver" / "benchmark_returns.parquet").exists()
    assert (output_root / "gold" / "labels_panel.parquet").exists()
    assert (output_root / "gold" / "portfolio_recommendations.parquet").exists()
    assert (output_root / "gold" / "csv" / "portfolio_recommendations.csv").exists()

    recommendations = pd.read_parquet(output_root / "gold" / "portfolio_recommendations.parquet")
    assert recommendations["target_weight"].sum() == pytest.approx(1.0)
    assert recommendations["target_weight"].min() >= 0
    assert recommendations["as_of_date"].nunique() == 1
    assert recommendations["as_of_date"].iat[0] == "2026-02-25"


def test_export_existing_run_for_tableau_does_not_rerun_pipeline(
    sample_html,
    sample_config,
    static_price_provider,
) -> None:
    run_one_shot(
        sample_config,
        universe_html=sample_html,
        price_provider=static_price_provider,
    )

    outputs = export_existing_run_for_tableau(sample_config, "test-run")

    assert "gold.portfolio_recommendations.csv" in outputs
    assert outputs["gold.portfolio_recommendations.csv"].exists()


def test_export_existing_run_for_tableau_creates_single_table_hyper(
    sample_html,
    sample_config,
    static_price_provider,
) -> None:
    sample_config.tableau.export_hyper = True
    run_one_shot(
        sample_config,
        universe_html=sample_html,
        price_provider=static_price_provider,
    )

    outputs = export_existing_run_for_tableau(sample_config, "test-run")

    assert outputs["gold.tableau_dashboard_mart.hyper"].exists()
    table_names = _hyper_table_names(outputs["gold.tableau_dashboard_mart.hyper"])
    assert len(table_names) == 1
    assert "portfolio_dashboard_mart" in table_names[0]


def _hyper_table_names(path: Path) -> list[str]:
    from tableauhyperapi import Connection, HyperProcess, Telemetry

    with (
        HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper,
        Connection(hyper.endpoint, path) as connection,
    ):
        return sorted(str(table.name) for table in connection.catalog.get_table_names("Extract"))
