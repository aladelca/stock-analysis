from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from stock_analysis.benchmarks.spy import build_benchmark_returns, build_spy_daily
from stock_analysis.config import PortfolioConfig
from stock_analysis.domain.models import PipelineResult
from stock_analysis.features.panel import compute_asset_feature_panel
from stock_analysis.features.price_features import compute_asset_daily_features
from stock_analysis.forecasting.baseline import build_optimizer_inputs
from stock_analysis.forecasting.ml_forecast import build_ml_optimizer_inputs
from stock_analysis.ingestion.prices import PriceDownload, PriceProvider, YFinancePriceProvider
from stock_analysis.ingestion.raw_store import write_json, write_text
from stock_analysis.ingestion.universe import fetch_sp500_html, parse_sp500_constituents
from stock_analysis.io.csv import write_csv
from stock_analysis.io.parquet import write_parquet
from stock_analysis.medallion.bronze import write_bronze_constituents, write_bronze_prices
from stock_analysis.medallion.silver import (
    build_asset_daily_returns,
    build_asset_universe_snapshot,
    write_silver_table,
)
from stock_analysis.ml.labels import build_forward_return_labels
from stock_analysis.optimization.engine import optimize_long_only
from stock_analysis.optimization.recommendations import (
    build_recommendations,
    build_risk_metrics,
    build_sector_exposure,
)
from stock_analysis.paths import ProjectPaths
from stock_analysis.tableau.dashboard_mart import build_dashboard_mart
from stock_analysis.tableau.hyper import export_hyper_if_available

logger = logging.getLogger(__name__)


def run_one_shot(
    config: PortfolioConfig,
    *,
    universe_html: str | None = None,
    price_provider: PriceProvider | None = None,
) -> PipelineResult:
    requested_as_of_date = config.run.as_of_date or date.today()
    run_id = config.run.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    paths = ProjectPaths(config.run.output_root, run_id)

    logger.info("Starting one-shot run %s for %s", run_id, requested_as_of_date.isoformat())

    html = universe_html or fetch_sp500_html(config.universe.source_url)
    constituents = parse_sp500_constituents(html, requested_as_of_date)

    provider = price_provider or YFinancePriceProvider(batch_size=config.prices.batch_size)
    start = requested_as_of_date - timedelta(days=int(config.prices.lookback_years * 365.25))
    provider_tickers = constituents["provider_ticker"].dropna().astype(str).tolist()
    provider_tickers = list(dict.fromkeys([*provider_tickers, *config.prices.benchmark_tickers]))
    price_download = provider.get_daily_prices(
        provider_tickers,
        start,
        requested_as_of_date,
        requested_as_of_date,
    )
    price_download = _coerce_price_download(price_download)
    daily_prices = price_download.prices
    daily_prices = daily_prices.merge(
        constituents[["ticker", "provider_ticker"]],
        on="provider_ticker",
        how="left",
        suffixes=("_provider", ""),
    )
    daily_prices["ticker"] = daily_prices["ticker"].fillna(daily_prices["ticker_provider"])
    daily_prices = daily_prices.drop(columns=["ticker_provider"], errors="ignore")
    data_as_of_date = _latest_price_date(daily_prices)
    data_as_of_date_str = data_as_of_date.isoformat()
    daily_prices["as_of_date"] = data_as_of_date_str
    constituents["as_of_date"] = data_as_of_date_str

    universe_raw_dir = paths.raw_dir("sp500_constituents")
    write_text(universe_raw_dir / "source.html", html)
    write_json(
        universe_raw_dir / "metadata.json",
        {
            "source_url": config.universe.source_url,
            "requested_as_of_date": requested_as_of_date.isoformat(),
            "data_as_of_date": data_as_of_date_str,
        },
    )
    constituents = write_bronze_constituents(constituents, paths)

    prices_raw_dir = paths.raw_dir("prices")
    raw_price_files = []
    for filename, payload in price_download.raw_payloads.items():
        raw_price_files.append(str(write_text(prices_raw_dir / filename, payload)))
    write_json(
        prices_raw_dir / "metadata.json",
        {
            "provider": config.prices.provider,
            "start": start.isoformat(),
            "requested_end": requested_as_of_date.isoformat(),
            "data_as_of_date": data_as_of_date_str,
            "requested_tickers": len(provider_tickers),
            "returned_rows": len(daily_prices),
            "raw_payload_files": raw_price_files,
        },
    )
    daily_prices = write_bronze_prices(daily_prices, paths)
    asset_tickers = set(constituents["ticker"].astype(str))
    asset_daily_prices = daily_prices.loc[
        daily_prices["ticker"].astype(str).isin(asset_tickers)
    ].copy()

    spy_daily = build_spy_daily(daily_prices)
    write_silver_table(spy_daily, "spy_daily", paths)
    benchmark_returns = build_benchmark_returns(
        spy_daily,
        horizons=config.forecast.label_horizons,
    )
    write_silver_table(benchmark_returns, "benchmark_returns", paths)

    returns = build_asset_daily_returns(asset_daily_prices)
    write_silver_table(returns, "asset_daily_returns", paths)

    universe_snapshot = build_asset_universe_snapshot(constituents, asset_daily_prices)
    write_silver_table(universe_snapshot, "asset_universe_snapshot", paths)

    feature_panel = compute_asset_feature_panel(
        asset_daily_prices,
        constituents,
        config.panel_features,
        benchmark_returns=spy_daily,
    )
    write_silver_table(feature_panel, "asset_daily_features_panel", paths)
    labels_panel = build_forward_return_labels(
        asset_daily_prices,
        feature_panel,
        benchmark_returns=benchmark_returns,
        horizons=config.forecast.label_horizons,
    )
    _write_gold_with_csv(paths, "labels_panel", labels_panel)

    features = compute_asset_daily_features(asset_daily_prices, constituents, config.features)
    write_silver_table(features, "asset_daily_features", paths)

    if config.forecast.engine == "ml":
        optimizer_input, covariance = build_ml_optimizer_inputs(
            feature_panel,
            labels_panel,
            returns,
            config.forecast,
        )
    else:
        optimizer_input, covariance = build_optimizer_inputs(features, returns, config.forecast)
    write_parquet(optimizer_input, paths.gold_path("optimizer_input"))
    write_csv(optimizer_input, paths.csv_mirror_path("gold", "optimizer_input"))
    covariance.to_parquet(paths.gold_path("covariance_matrix"))

    weights = optimize_long_only(optimizer_input, covariance, config.optimizer)
    recommendations = build_recommendations(
        optimizer_input,
        weights,
        config.optimizer,
        data_as_of_date_str,
        run_id,
    )
    risk_metrics = build_risk_metrics(
        optimizer_input,
        covariance,
        weights,
        data_as_of_date_str,
        run_id,
    )
    sector_exposure = build_sector_exposure(optimizer_input, weights, data_as_of_date_str, run_id)
    run_metadata = _build_run_metadata(
        config,
        run_id,
        requested_as_of_date,
        data_as_of_date,
        constituents,
        daily_prices,
    )

    _write_gold_with_csv(paths, "portfolio_recommendations", recommendations)
    _write_gold_with_csv(paths, "portfolio_risk_metrics", risk_metrics)
    _write_gold_with_csv(paths, "sector_exposure", sector_exposure)
    _write_gold_with_csv(paths, "run_metadata", run_metadata)

    if config.tableau.export_hyper:
        hyper_path = paths.gold_path("tableau_dashboard_mart", "hyper")
        dashboard_mart = build_dashboard_mart(
            recommendations,
            risk_metrics,
            sector_exposure,
            run_metadata,
        )
        exported = export_hyper_if_available(
            dashboard_mart,
            hyper_path,
        )
        if exported is None:
            logger.warning("Tableau Hyper API is not installed; skipped Hyper export")

    logger.info("Completed one-shot run %s", run_id)
    return PipelineResult(
        run_id=run_id,
        as_of_date=data_as_of_date,
        output_root=str(paths.run_root),
        recommendations_path=str(paths.gold_path("portfolio_recommendations")),
        risk_metrics_path=str(paths.gold_path("portfolio_risk_metrics")),
        sector_exposure_path=str(paths.gold_path("sector_exposure")),
    )


def _write_gold_with_csv(paths: ProjectPaths, name: str, df: pd.DataFrame) -> None:
    write_parquet(df, paths.gold_path(name))
    write_csv(df, paths.csv_mirror_path("gold", name))


def _build_run_metadata(
    config: PortfolioConfig,
    run_id: str,
    requested_as_of_date: date,
    data_as_of_date: date,
    constituents: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    config_json = config.model_dump_json(exclude={"run": {"as_of_date"}})
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    payload = {
        "run_id": run_id,
        "requested_as_of_date": requested_as_of_date.isoformat(),
        "as_of_date": data_as_of_date.isoformat(),
        "data_as_of_date": data_as_of_date.isoformat(),
        "config_hash": config_hash,
        "forecast_engine": config.forecast.engine,
        "model_version": (
            config.forecast.ml_model_version if config.forecast.engine == "ml" else "heuristic"
        ),
        "model_family": ("ridge_lightgbm_blend" if config.forecast.engine == "ml" else "heuristic"),
        "expected_return_is_calibrated": False,
        "universe_count": int(len(constituents)),
        "price_row_count": int(len(daily_prices)),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_json": json.dumps(config.model_dump(mode="json"), sort_keys=True),
    }
    return pd.DataFrame([payload])


def _latest_price_date(daily_prices: pd.DataFrame) -> date:
    if daily_prices.empty:
        msg = "Cannot determine data_as_of_date from an empty price table"
        raise ValueError(msg)
    latest = pd.to_datetime(daily_prices["date"], errors="coerce").max()
    if pd.isna(latest):
        msg = "Cannot determine data_as_of_date because all price dates are invalid"
        raise ValueError(msg)
    return latest.date()


def _coerce_price_download(price_download: PriceDownload | pd.DataFrame) -> PriceDownload:
    if isinstance(price_download, PriceDownload):
        return price_download
    return PriceDownload(prices=price_download, raw_payloads={})
