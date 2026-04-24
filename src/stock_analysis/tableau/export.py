from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_analysis.config import PortfolioConfig
from stock_analysis.io.csv import write_csv
from stock_analysis.paths import ProjectPaths
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


def export_existing_run_for_tableau(config: PortfolioConfig, run_id: str) -> dict[str, Path]:
    paths = ProjectPaths(config.run.output_root, run_id)
    outputs: dict[str, Path] = {}

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

    if config.tableau.export_hyper:
        dashboard_mart = build_dashboard_mart(
            _read_gold_table(paths, "portfolio_recommendations"),
            _read_gold_table(paths, "portfolio_risk_metrics"),
            _read_gold_table(paths, "sector_exposure"),
            _read_gold_table(paths, "run_metadata"),
        )
        hyper_path = paths.gold_path("tableau_dashboard_mart", "hyper")
        exported = export_hyper_if_available(dashboard_mart, hyper_path)
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
