from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from stock_analysis.benchmarks.spy import build_benchmark_returns, build_spy_daily
from stock_analysis.config import PortfolioConfig
from stock_analysis.domain.models import PipelineResult
from stock_analysis.env import load_local_env
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
from stock_analysis.ml.mlflow_tracking import log_portfolio_run
from stock_analysis.optimization.engine import optimize_long_only
from stock_analysis.optimization.recommendations import (
    build_recommendations,
    build_risk_metrics,
    build_sector_exposure,
)
from stock_analysis.paths import ProjectPaths
from stock_analysis.portfolio.holdings import PortfolioState, load_portfolio_state
from stock_analysis.portfolio.live_state import LivePortfolioState, build_live_portfolio_state
from stock_analysis.portfolio.rebalance import build_rebalance_context
from stock_analysis.storage.contracts import AccountTrackingRepository
from stock_analysis.storage.supabase import create_account_tracking_repository
from stock_analysis.tableau.account_tracking_marts import build_account_tracking_marts
from stock_analysis.tableau.dashboard_mart import build_dashboard_mart
from stock_analysis.tableau.hyper import export_hyper_if_available

logger = logging.getLogger(__name__)


def run_one_shot(
    config: PortfolioConfig,
    *,
    universe_html: str | None = None,
    price_provider: PriceProvider | None = None,
    account_repository: AccountTrackingRepository | None = None,
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

    portfolio_state, contribution_amount, live_state = _load_rebalance_state(
        config,
        data_as_of_date=data_as_of_date,
        account_repository=account_repository,
    )
    context_tickers = _rebalance_context_tickers(optimizer_input, portfolio_state)
    rebalance_context = build_rebalance_context(
        portfolio_state,
        context_tickers,
        contribution_amount=contribution_amount,
    )
    current_weights = rebalance_context.current_weights.reindex(
        optimizer_input["ticker"].astype(str)
    ).fillna(0.0)
    weights = optimize_long_only(
        optimizer_input,
        covariance,
        config.optimizer,
        w_prev=current_weights,
    )
    recommendations = build_recommendations(
        optimizer_input,
        weights,
        config.optimizer,
        data_as_of_date_str,
        run_id,
        current_weights=current_weights,
        rebalance_context=rebalance_context,
        no_trade_band=config.execution.no_trade_band,
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
        live_state=live_state,
    )

    _write_gold_with_csv(paths, "portfolio_recommendations", recommendations)
    _write_gold_with_csv(paths, "portfolio_risk_metrics", risk_metrics)
    _write_gold_with_csv(paths, "sector_exposure", sector_exposure)
    _write_gold_with_csv(paths, "run_metadata", run_metadata)
    account_tracking_tables: dict[str, pd.DataFrame] = {}
    if live_state is not None:
        account_tracking_tables = build_account_tracking_marts(
            live_state=live_state,
            recommendations=recommendations,
            run_metadata=run_metadata,
            spy_daily=spy_daily,
            commission_rate=config.optimizer.commission_rate,
        )
        for name, table in account_tracking_tables.items():
            _write_gold_with_csv(paths, name, table)

    artifact_paths = [
        paths.gold_path("portfolio_recommendations"),
        paths.csv_mirror_path("gold", "portfolio_recommendations"),
        paths.gold_path("portfolio_risk_metrics"),
        paths.csv_mirror_path("gold", "portfolio_risk_metrics"),
        paths.gold_path("sector_exposure"),
        paths.csv_mirror_path("gold", "sector_exposure"),
        paths.gold_path("run_metadata"),
        paths.csv_mirror_path("gold", "run_metadata"),
        paths.gold_path("optimizer_input"),
        paths.gold_path("covariance_matrix"),
    ]
    for name in account_tracking_tables:
        artifact_paths.extend(
            [
                paths.gold_path(name),
                paths.csv_mirror_path("gold", name),
            ]
        )

    if config.tableau.export_hyper:
        hyper_path = paths.gold_path("tableau_dashboard_mart", "hyper")
        dashboard_mart = build_dashboard_mart(
            recommendations,
            risk_metrics,
            sector_exposure,
            run_metadata,
            performance_snapshots=account_tracking_tables.get("performance_snapshots"),
        )
        exported = export_hyper_if_available(
            dashboard_mart,
            hyper_path,
        )
        if exported is None:
            logger.warning("Tableau Hyper API is not installed; skipped Hyper export")
        else:
            artifact_paths.append(exported)

    if config.mlflow.enabled:
        mlflow_run_id = log_portfolio_run(
            config,
            run_id=run_id,
            data_as_of_date=data_as_of_date_str,
            recommendations=recommendations,
            risk_metrics=risk_metrics,
            run_metadata=run_metadata,
            artifacts=artifact_paths,
            tracking_uri=config.mlflow.tracking_uri,
            experiment_name=config.mlflow.experiment_name,
        )
        logger.info("Logged one-shot run %s to MLflow run %s", run_id, mlflow_run_id)

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


def _load_portfolio_state(config: PortfolioConfig) -> PortfolioState:
    state = load_portfolio_state(
        config.portfolio_state.current_holdings_path,
        cash_balance=config.execution.cash_balance,
        portfolio_value=config.portfolio_state.portfolio_value,
    )
    if state.resolved_portfolio_value is not None:
        return state
    return load_portfolio_state(
        config.portfolio_state.current_holdings_path,
        cash_balance=config.execution.cash_balance,
        portfolio_value=config.contributions.initial_portfolio_value,
    )


def _load_rebalance_state(
    config: PortfolioConfig,
    *,
    data_as_of_date: date,
    account_repository: AccountTrackingRepository | None,
) -> tuple[PortfolioState, float, LivePortfolioState | None]:
    if config.live_account.cashflow_source != "actual":
        return _load_portfolio_state(config), config.contributions.monthly_deposit_amount, None
    if not config.live_account.enabled:
        msg = "Set live_account.enabled=true when live_account.cashflow_source is actual."
        raise ValueError(msg)
    if not config.live_account.account_slug:
        msg = "Set live_account.account_slug when live_account.cashflow_source is actual."
        raise ValueError(msg)

    repository = account_repository
    if repository is None:
        load_local_env()
        repository = create_account_tracking_repository(config.supabase)
    account = repository.get_account_by_slug(config.live_account.account_slug)
    if account is None:
        msg = f"Supabase account does not exist: {config.live_account.account_slug}"
        raise ValueError(msg)
    live_state = build_live_portfolio_state(
        repository,
        account,
        as_of_date=data_as_of_date,
    )
    logger.info(
        "Loaded live account %s snapshot %s with %.2f unapplied cashflow",
        account.slug,
        live_state.snapshot.snapshot_date.isoformat(),
        live_state.contribution_amount,
    )
    return live_state.state, live_state.contribution_amount, live_state


def _rebalance_context_tickers(
    optimizer_input: pd.DataFrame,
    portfolio_state: PortfolioState,
) -> pd.Index:
    tickers = optimizer_input["ticker"].astype(str).tolist()
    if not portfolio_state.weights.empty:
        tickers.extend(portfolio_state.weights.index.astype(str).tolist())
    if not portfolio_state.market_values.empty:
        tickers.extend(portfolio_state.market_values.index.astype(str).tolist())
    return pd.Index(list(dict.fromkeys(tickers)), name="ticker")


def _build_run_metadata(
    config: PortfolioConfig,
    run_id: str,
    requested_as_of_date: date,
    data_as_of_date: date,
    constituents: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    live_state: LivePortfolioState | None = None,
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
        "commission_rate": config.optimizer.commission_rate,
        "min_rebalance_trade_weight": config.optimizer.min_rebalance_trade_weight,
        "sector_max_weight": config.optimizer.sector_max_weight,
        "current_holdings_path": (
            str(config.portfolio_state.current_holdings_path)
            if config.portfolio_state.current_holdings_path is not None
            else ""
        ),
        "live_account_enabled": config.live_account.enabled,
        "live_account_slug": config.live_account.account_slug or "",
        "live_cashflow_source": config.live_account.cashflow_source,
        "live_snapshot_id": live_state.snapshot.id if live_state is not None else "",
        "live_snapshot_date": (
            live_state.snapshot.snapshot_date.isoformat() if live_state is not None else ""
        ),
        "live_unapplied_cashflow_amount": (
            live_state.net_cashflow_amount if live_state is not None else 0.0
        ),
        "live_unapplied_cashflow_count": (
            len(live_state.applied_cashflows) if live_state is not None else 0
        ),
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
