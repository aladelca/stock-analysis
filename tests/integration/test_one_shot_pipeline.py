from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stock_analysis.config import (
    FeatureConfig,
    ForecastConfig,
    OptimizerConfig,
    PanelFeatureConfig,
    PortfolioConfig,
    PriceConfig,
    RunConfig,
    TableauConfig,
)
from stock_analysis.ingestion.prices import PriceDownload
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


def test_one_shot_pipeline_can_use_ml_forecast_engine(sample_html, tmp_path: Path) -> None:
    config = PortfolioConfig(
        run=RunConfig(
            as_of_date=date(2026, 4, 24),
            output_root=tmp_path / "data",
            run_id="ml-run",
        ),
        prices=PriceConfig(lookback_years=1, batch_size=10),
        features=FeatureConfig(
            min_history_days=30,
            momentum_windows=[5, 10],
            volatility_window=5,
            drawdown_window=10,
            moving_average_windows=[5, 10],
        ),
        panel_features=PanelFeatureConfig(
            min_history_days=30,
            momentum_windows=[21],
            volatility_windows=[21],
            drawdown_windows=[21],
            moving_average_windows=[10, 21],
            return_windows=[5, 21],
            volume_zscore_window=5,
        ),
        forecast=ForecastConfig(
            engine="ml",
            covariance_lookback_days=40,
            label_horizons=[5, 21],
            ml_max_assets=4,
            ml_feature_columns=["momentum_21d", "volatility_21d", "return_5d"],
            ml_lightgbm_nested_cv=False,
        ),
        optimizer=OptimizerConfig(max_weight=0.6, risk_aversion=1.0, min_trade_weight=0.001),
        tableau=TableauConfig(export_csv=True, export_hyper=False, publish_enabled=False),
    )

    result = run_one_shot(
        config,
        universe_html=sample_html,
        price_provider=_LongStaticPriceProvider(_ml_fixture_prices()),
    )
    output_root = Path(result.output_root)
    optimizer_input = pd.read_parquet(output_root / "gold" / "optimizer_input.parquet")
    recommendations = pd.read_parquet(output_root / "gold" / "portfolio_recommendations.parquet")
    metadata = pd.read_parquet(output_root / "gold" / "run_metadata.parquet")

    assert optimizer_input["forecast_engine"].eq("ml").all()
    assert optimizer_input["forecast_model_version"].eq("phase2-e8-ridge-lightgbm-blend-v1").all()
    assert recommendations["target_weight"].sum() == pytest.approx(1.0)
    assert metadata["forecast_engine"].iat[0] == "ml"
    assert not bool(metadata["expected_return_is_calibrated"].iat[0])


class _LongStaticPriceProvider:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices

    def get_daily_prices(self, tickers, start, end, as_of_date) -> PriceDownload:
        del start, end, as_of_date
        result = self.prices.loc[self.prices["provider_ticker"].isin(set(tickers))].copy()
        return PriceDownload(prices=result, raw_payloads={"static_prices.csv": result.to_csv()})


def _ml_fixture_prices() -> pd.DataFrame:
    provider_tickers = ["AAA", "BBB", "CCC", "BRK-B", "SPY"]
    dates = pd.bdate_range("2025-11-03", periods=110)
    rows: list[dict[str, object]] = []
    for ticker_idx, provider_ticker in enumerate(provider_tickers):
        ticker = provider_ticker.replace("-", ".")
        base = 80 + ticker_idx * 8
        for day_idx, current_date in enumerate(dates):
            trend = 0.0015 + ticker_idx * 0.0002
            cycle = 0.002 * ((day_idx % 9) - 4) / 4
            close = base * (1 + trend * day_idx + cycle)
            rows.append(
                {
                    "ticker": ticker,
                    "provider_ticker": provider_ticker,
                    "date": current_date.date().isoformat(),
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "adj_close": close,
                    "volume": int(1_000_000 + ticker_idx * 50_000 + day_idx * 500),
                    "as_of_date": dates[-1].date().isoformat(),
                }
            )
    return pd.DataFrame(rows)


def _hyper_table_names(path: Path) -> list[str]:
    from tableauhyperapi import Connection, HyperProcess, Telemetry

    with (
        HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper,
        Connection(hyper.endpoint, path) as connection,
    ):
        return sorted(str(table.name) for table in connection.catalog.get_table_names("Extract"))
