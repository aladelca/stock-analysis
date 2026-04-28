from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_analysis.config import PortfolioConfig
from stock_analysis.env import load_local_env
from stock_analysis.io.csv import write_csv
from stock_analysis.io.parquet import write_parquet
from stock_analysis.paths import ProjectPaths
from stock_analysis.storage.contracts import AccountTrackingRepository
from stock_analysis.storage.supabase import create_account_tracking_repository
from stock_analysis.tableau.account_history_marts import build_account_history_marts
from stock_analysis.tableau.dashboard_mart import build_dashboard_mart
from stock_analysis.tableau.hyper import export_hyper_if_available

TABLEAU_CSV_TABLES: tuple[tuple[str, str], ...] = (
    ("bronze", "sp500_constituents"),
    ("silver", "asset_daily_features"),
    ("gold", "portfolio_recommendations"),
    ("gold", "portfolio_risk_metrics"),
    ("gold", "sector_exposure"),
    ("gold", "run_metadata"),
)
TABLEAU_OPTIONAL_CSV_TABLES: tuple[tuple[str, str], ...] = (
    ("gold", "forecast_calibration_diagnostics"),
    ("gold", "forecast_calibration_predictions"),
    ("gold", "cashflows"),
    ("gold", "portfolio_snapshots"),
    ("gold", "holding_snapshots"),
    ("gold", "recommendation_runs"),
    ("gold", "recommendation_lines"),
    ("gold", "performance_snapshots"),
)


def export_existing_run_for_tableau(
    config: PortfolioConfig,
    run_id: str,
    *,
    account_repository: AccountTrackingRepository | None = None,
) -> dict[str, Path]:
    paths = ProjectPaths(config.run.output_root, run_id)
    outputs: dict[str, Path] = {}
    history_tables = _account_history_tables(config, paths, account_repository)
    for name, table in history_tables.items():
        outputs[f"gold.{name}.parquet"] = write_parquet(table, paths.gold_path(name))

    if config.tableau.export_csv:
        for layer, name in TABLEAU_CSV_TABLES:
            parquet_path = _table_path(paths, layer, name)
            if not parquet_path.exists():
                msg = f"Cannot export Tableau CSV; missing existing run table: {parquet_path}"
                raise FileNotFoundError(msg)
            outputs[f"{layer}.{name}.csv"] = write_csv(
                pd.read_parquet(parquet_path),
                paths.csv_mirror_path(layer, name),
            )
        for layer, name in TABLEAU_OPTIONAL_CSV_TABLES:
            parquet_path = _table_path(paths, layer, name)
            if parquet_path.exists():
                outputs[f"{layer}.{name}.csv"] = write_csv(
                    pd.read_parquet(parquet_path),
                    paths.csv_mirror_path(layer, name),
                )
        for name, table in history_tables.items():
            outputs[f"gold.{name}.csv"] = write_csv(table, paths.csv_mirror_path("gold", name))

    if config.tableau.export_hyper:
        performance_snapshots = _read_optional_gold_table(paths, "performance_snapshots")
        optional_gold_tables = _read_optional_gold_tables(paths)
        dashboard_mart = build_dashboard_mart(
            _read_gold_table(paths, "portfolio_recommendations"),
            _read_gold_table(paths, "portfolio_risk_metrics"),
            _read_gold_table(paths, "sector_exposure"),
            _read_gold_table(paths, "run_metadata"),
            performance_snapshots=performance_snapshots,
        )
        hyper_tables = {
            "portfolio_dashboard_mart": dashboard_mart,
            **optional_gold_tables,
            **history_tables,
        }
        hyper_path = paths.gold_path("tableau_dashboard_mart", "hyper")
        exported = export_hyper_if_available(hyper_tables, hyper_path)
        if exported is not None:
            outputs["gold.tableau_dashboard_mart.hyper"] = exported

    return outputs


def _table_path(paths: ProjectPaths, layer: str, name: str) -> Path:
    if layer == "bronze":
        return paths.bronze_path(name)
    if layer == "silver":
        return paths.silver_path(name)
    if layer == "gold":
        return paths.gold_path(name)
    msg = f"Unsupported medallion layer: {layer}"
    raise ValueError(msg)


def _read_gold_table(paths: ProjectPaths, name: str) -> pd.DataFrame:
    parquet_path = paths.gold_path(name)
    if not parquet_path.exists():
        msg = f"Cannot export Tableau Hyper; missing existing run table: {parquet_path}"
        raise FileNotFoundError(msg)
    return pd.read_parquet(parquet_path)


def _read_optional_gold_table(paths: ProjectPaths, name: str) -> pd.DataFrame | None:
    parquet_path = paths.gold_path(name)
    if not parquet_path.exists():
        return None
    return pd.read_parquet(parquet_path)


def _read_optional_gold_tables(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for layer, name in TABLEAU_OPTIONAL_CSV_TABLES:
        if layer != "gold":
            continue
        table = _read_optional_gold_table(paths, name)
        if table is not None:
            tables[name] = table
    return tables


def _account_history_tables(
    config: PortfolioConfig,
    paths: ProjectPaths,
    account_repository: AccountTrackingRepository | None,
) -> dict[str, pd.DataFrame]:
    if not config.live_account.enabled or not config.live_account.account_slug:
        return {}
    repository = account_repository
    if repository is None:
        if not config.supabase.enabled:
            return {}
        load_local_env()
        repository = create_account_tracking_repository(config.supabase)
    daily_prices_path = paths.bronze_path("daily_prices")
    daily_prices = (
        pd.read_parquet(daily_prices_path) if daily_prices_path.exists() else pd.DataFrame()
    )
    return build_account_history_marts(
        repository=repository,
        account_slug=config.live_account.account_slug,
        daily_prices=daily_prices,
        default_horizon_days=config.forecast.ml_horizon_days,
        benchmark_ticker=config.prices.benchmark_tickers[0],
    )
