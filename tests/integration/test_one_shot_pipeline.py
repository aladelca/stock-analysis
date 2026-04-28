from __future__ import annotations

from dataclasses import replace
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
    PortfolioStateConfig,
    PriceConfig,
    RunConfig,
    TableauConfig,
)
from stock_analysis.ingestion.prices import PriceDownload
from stock_analysis.pipeline.one_shot import run_one_shot
from stock_analysis.storage.contracts import (
    AccountRecord,
    CashflowRecord,
    HoldingSnapshotRecord,
    PerformanceSnapshotRecord,
    PortfolioSnapshotRecord,
    RecommendationLineRecord,
    RecommendationRunRecord,
)
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
    assert {
        "current_weight",
        "trade_weight",
        "trade_abs_weight",
        "estimated_commission_weight",
        "rebalance_required",
        "action",
    } <= set(recommendations.columns)
    assert recommendations["target_weight"].sum() == pytest.approx(1.0)
    assert recommendations["target_weight"].min() >= 0
    assert recommendations["current_weight"].sum() == pytest.approx(0.0)
    assert recommendations["as_of_date"].nunique() == 1
    assert recommendations["as_of_date"].iat[0] == "2026-02-25"


def test_one_shot_pipeline_adds_spy_as_optimizer_candidate(
    sample_html,
    sample_config,
    sample_prices,
) -> None:
    result = run_one_shot(
        sample_config,
        universe_html=sample_html,
        price_provider=_LongStaticPriceProvider(_prices_with_spy(sample_prices)),
    )
    output_root = Path(result.output_root)

    constituents = pd.read_parquet(output_root / "bronze" / "sp500_constituents.parquet")
    optimizer_input = pd.read_parquet(output_root / "gold" / "optimizer_input.parquet")
    recommendations = pd.read_parquet(output_root / "gold" / "portfolio_recommendations.parquet")

    spy_constituent = constituents.set_index("ticker").loc["SPY"]
    spy_optimizer = optimizer_input.set_index("ticker").loc["SPY"]
    spy_recommendation = recommendations.set_index("ticker").loc["SPY"]

    assert bool(spy_constituent["is_benchmark_candidate"])
    assert spy_constituent["security"] == "SPDR S&P 500 ETF Trust"
    assert bool(spy_optimizer["eligible_for_optimization"])
    assert bool(spy_optimizer["is_benchmark_candidate"])
    assert spy_recommendation["reason_code"] != "current_holding_outside_optimizer_universe"


def test_one_shot_pipeline_emits_sell_for_current_holding_outside_universe(
    sample_html,
    sample_config,
    static_price_provider,
    tmp_path: Path,
) -> None:
    holdings_path = tmp_path / "current_holdings.csv"
    holdings_path.write_text("ticker,current_weight\nZZZ,0.10\nAAA,0.20\n", encoding="utf-8")
    sample_config.portfolio_state = PortfolioStateConfig(current_holdings_path=holdings_path)

    result = run_one_shot(
        sample_config,
        universe_html=sample_html,
        price_provider=static_price_provider,
    )

    recommendations = pd.read_parquet(
        Path(result.output_root) / "gold" / "portfolio_recommendations.parquet"
    )
    zzz = recommendations.set_index("ticker").loc["ZZZ"]
    assert zzz["action"] == "SELL"
    assert zzz["target_weight"] == 0
    assert zzz["trade_weight"] == pytest.approx(-0.10)
    assert zzz["estimated_commission_weight"] == pytest.approx(0.002)


def test_one_shot_pipeline_uses_actual_live_cashflows(
    sample_html,
    sample_config,
    static_price_provider,
) -> None:
    sample_config.live_account.enabled = True
    sample_config.live_account.account_slug = "main"
    sample_config.live_account.cashflow_source = "actual"
    sample_config.contributions.monthly_deposit_amount = 999.0
    repository = FakeAccountTrackingRepository()

    result = run_one_shot(
        sample_config,
        universe_html=sample_html,
        price_provider=static_price_provider,
        account_repository=repository,
    )

    recommendations = pd.read_parquet(
        Path(result.output_root) / "gold" / "portfolio_recommendations.parquet"
    )
    metadata = pd.read_parquet(Path(result.output_root) / "gold" / "run_metadata.parquet")
    cashflows = pd.read_parquet(Path(result.output_root) / "gold" / "cashflows.parquet")
    performance = pd.read_parquet(
        Path(result.output_root) / "gold" / "performance_snapshots.parquet"
    )
    recommendation_runs = pd.read_parquet(
        Path(result.output_root) / "gold" / "recommendation_runs.parquet"
    )
    recommendation_lines = pd.read_parquet(
        Path(result.output_root) / "gold" / "recommendation_lines.parquet"
    )

    assert recommendations["contribution_amount"].max() == pytest.approx(200.0)
    assert recommendations["portfolio_value_before_contribution"].max() == pytest.approx(1000.0)
    assert recommendations["portfolio_value_after_contribution"].max() == pytest.approx(1200.0)
    assert metadata["live_cashflow_source"].iat[0] == "actual"
    assert metadata["live_account_slug"].iat[0] == "main"
    assert metadata["live_unapplied_cashflow_amount"].iat[0] == pytest.approx(200.0)
    assert cashflows["is_applied_to_recommendation"].tolist() == [True]
    assert recommendation_runs["unapplied_cashflow_amount"].iat[0] == pytest.approx(200.0)
    assert recommendation_lines["recommendation_key"].str.startswith("test-run:").all()
    assert "spy_same_cashflow_value" in performance.columns
    assert repository.recommendation_runs[0].run_id == "test-run"
    assert repository.recommendation_runs[0].id == "recommendation-run-1"
    assert len(repository.recommendation_lines) == len(recommendations)
    assert repository.performance_snapshots[0].account_total_value == pytest.approx(1000.0)

    exports = export_existing_run_for_tableau(sample_config, "test-run")
    assert "gold.cashflows.csv" in exports
    assert "gold.performance_snapshots.csv" in exports


def test_export_existing_run_for_tableau_includes_account_history(
    sample_html,
    sample_config,
    static_price_provider,
) -> None:
    sample_config.live_account.enabled = True
    sample_config.live_account.account_slug = "main"
    sample_config.live_account.cashflow_source = "actual"
    repository = FakeAccountTrackingRepository()
    repository.cashflows = []
    historical_run = RecommendationRunRecord(
        id="historical-run-id",
        account_id="account-1",
        run_id="historical-run",
        as_of_date=date(2026, 1, 2),
        data_as_of_date=date(2026, 1, 2),
        model_version="e8-scale-0p5-contribution-aware-v1",
        ml_score_scale=0.5,
        config_hash="old",
    )
    repository.recommendation_runs.append(historical_run)
    repository.recommendation_lines.append(
        RecommendationLineRecord(
            id="historical-line-id",
            recommendation_run_id=historical_run.id or "",
            ticker="AAA",
            action="BUY",
            forecast_score=0.01,
            expected_return=0.01,
            target_weight=0.2,
        )
    )
    run_one_shot(
        sample_config,
        universe_html=sample_html,
        price_provider=static_price_provider,
        account_repository=repository,
    )

    outputs = export_existing_run_for_tableau(
        sample_config,
        "test-run",
        account_repository=repository,
    )

    assert "gold.recommendation_lines_history.csv" in outputs
    assert "gold.cashflows_history.csv" in outputs
    history = pd.read_parquet(
        sample_config.run.output_root
        / "runs"
        / "test-run"
        / "gold"
        / "recommendation_lines_history.parquet"
    )
    assert set(history["run_id"]) == {"historical-run", "test-run"}
    historical = history.loc[history["run_id"].eq("historical-run")].iloc[0]
    assert historical["outcome_status"] == "realized"
    assert historical["forecast_score"] == pytest.approx(0.01)
    assert historical["forecast_horizon_days"] == 5
    assert pd.notna(historical["realized_return"])
    cashflows_history = pd.read_parquet(
        sample_config.run.output_root / "runs" / "test-run" / "gold" / "cashflows_history.parquet"
    )
    assert cashflows_history.empty
    assert "account_slug" in cashflows_history.columns


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


def test_export_existing_live_run_for_tableau_creates_account_tracking_hyper_tables(
    sample_html,
    sample_config,
    static_price_provider,
) -> None:
    sample_config.tableau.export_hyper = True
    sample_config.live_account.enabled = True
    sample_config.live_account.account_slug = "main"
    sample_config.live_account.cashflow_source = "actual"
    repository = FakeAccountTrackingRepository()
    run_one_shot(
        sample_config,
        universe_html=sample_html,
        price_provider=static_price_provider,
        account_repository=repository,
    )

    outputs = export_existing_run_for_tableau(sample_config, "test-run")

    table_names = _hyper_table_names(outputs["gold.tableau_dashboard_mart.hyper"])
    assert all(
        any(expected in table_name for table_name in table_names)
        for expected in {"portfolio_dashboard_mart", "cashflows", "performance_snapshots"}
    )
    assert len(table_names) == 7


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
    assert optimizer_input["forecast_model_version"].eq("lightgbm_return_zscore").all()
    assert recommendations["target_weight"].sum() == pytest.approx(1.0)
    assert metadata["forecast_engine"].iat[0] == "ml"
    assert not bool(metadata["expected_return_is_calibrated"].iat[0])


class FakeAccountTrackingRepository:
    def __init__(self) -> None:
        self.account = AccountRecord(id="account-1", slug="main", display_name="Main")
        self.snapshot = PortfolioSnapshotRecord(
            id="snapshot-1",
            account_id="account-1",
            snapshot_date=date(2026, 2, 20),
            market_value=600.0,
            cash_balance=400.0,
            total_value=1000.0,
        )
        self.holdings = [
            HoldingSnapshotRecord(
                snapshot_id="snapshot-1",
                ticker="AAA",
                market_value=600.0,
            )
        ]
        self.cashflows = [
            CashflowRecord(
                account_id="account-1",
                cashflow_date=date(2026, 2, 24),
                amount=200.0,
                cashflow_type="deposit",
            )
        ]
        self.recommendation_runs: list[RecommendationRunRecord] = []
        self.recommendation_lines: list[RecommendationLineRecord] = []
        self.performance_snapshots: list[PerformanceSnapshotRecord] = []

    def get_account_by_slug(self, slug: str) -> AccountRecord | None:
        if slug == self.account.slug:
            return self.account
        return None

    def upsert_account(self, account: AccountRecord) -> AccountRecord:
        return account

    def insert_cashflow(self, cashflow: CashflowRecord) -> CashflowRecord:
        return cashflow

    def list_cashflows(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[CashflowRecord]:
        del account_id
        rows = self.cashflows
        if start_date is not None:
            rows = [row for row in rows if row.cashflow_date >= start_date]
        if end_date is not None:
            rows = [row for row in rows if row.cashflow_date <= end_date]
        return rows

    def insert_portfolio_snapshot(
        self,
        snapshot: PortfolioSnapshotRecord,
        holdings: list[HoldingSnapshotRecord] | None = None,
    ) -> PortfolioSnapshotRecord:
        del holdings
        return snapshot

    def latest_portfolio_snapshot(
        self,
        account_id: str,
        *,
        as_of_date: date,
    ) -> PortfolioSnapshotRecord | None:
        del account_id, as_of_date
        return self.snapshot

    def list_portfolio_snapshots(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[PortfolioSnapshotRecord]:
        del account_id, start_date, end_date
        return [self.snapshot]

    def list_holding_snapshots(self, snapshot_id: str) -> list[HoldingSnapshotRecord]:
        return [holding for holding in self.holdings if holding.snapshot_id == snapshot_id]

    def insert_recommendation_run(self, run: RecommendationRunRecord) -> RecommendationRunRecord:
        persisted = replace(run, id=f"recommendation-run-{len(self.recommendation_runs) + 1}")
        self.recommendation_runs.append(persisted)
        return persisted

    def insert_recommendation_lines(
        self,
        lines: list[RecommendationLineRecord],
    ) -> list[RecommendationLineRecord]:
        persisted = [
            replace(line, id=f"recommendation-line-{len(self.recommendation_lines) + index + 1}")
            for index, line in enumerate(lines)
        ]
        self.recommendation_lines.extend(persisted)
        return persisted

    def insert_performance_snapshot(
        self,
        snapshot: PerformanceSnapshotRecord,
    ) -> PerformanceSnapshotRecord:
        persisted = replace(
            snapshot,
            id=f"performance-snapshot-{len(self.performance_snapshots) + 1}",
        )
        self.performance_snapshots.append(persisted)
        return persisted

    def list_recommendation_runs(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[RecommendationRunRecord]:
        rows = [row for row in self.recommendation_runs if row.account_id == account_id]
        if start_date is not None:
            rows = [row for row in rows if row.data_as_of_date >= start_date]
        if end_date is not None:
            rows = [row for row in rows if row.data_as_of_date <= end_date]
        return sorted(rows, key=lambda row: row.data_as_of_date)

    def list_recommendation_lines(
        self,
        recommendation_run_ids: list[str],
    ) -> list[RecommendationLineRecord]:
        wanted = set(recommendation_run_ids)
        return [row for row in self.recommendation_lines if row.recommendation_run_id in wanted]

    def list_performance_snapshots(
        self,
        account_id: str,
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[PerformanceSnapshotRecord]:
        rows = [row for row in self.performance_snapshots if row.account_id == account_id]
        if start_date is not None:
            rows = [row for row in rows if row.as_of_date >= start_date]
        if end_date is not None:
            rows = [row for row in rows if row.as_of_date <= end_date]
        return sorted(rows, key=lambda row: row.as_of_date)


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


def _prices_with_spy(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.copy()
    dates = sorted(result["date"].drop_duplicates().astype(str))
    rows: list[dict[str, object]] = []
    for day, current_date in enumerate(dates):
        close = 400 * (1 + 0.001 * day)
        rows.append(
            {
                "ticker": "SPY",
                "provider_ticker": "SPY",
                "date": current_date,
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "adj_close": close,
                "volume": int(10_000_000 + day),
                "as_of_date": result["as_of_date"].iat[0],
            }
        )
    return pd.concat([result, pd.DataFrame(rows)], ignore_index=True)


def _hyper_table_names(path: Path) -> list[str]:
    from tableauhyperapi import Connection, HyperProcess, Telemetry

    with (
        HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper,
        Connection(hyper.endpoint, path) as connection,
    ):
        return sorted(str(table.name) for table in connection.catalog.get_table_names("Extract"))
