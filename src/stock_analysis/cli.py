from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import typer
from rich import print

from stock_analysis.backtest.runner import BacktestConfig
from stock_analysis.config import OptimizerConfig, load_config
from stock_analysis.env import load_local_env
from stock_analysis.logging import configure_logging
from stock_analysis.ml.autoresearch_eval import (
    AutoresearchEvalConfig,
    TurnoverSweepConfig,
    append_result_tsv,
    evaluate_candidate,
    evaluate_turnover_sweep,
    result_to_json,
)
from stock_analysis.ml.experiments import run_experiment_from_config
from stock_analysis.ml.mlflow_tracking import (
    DEFAULT_MLFLOW_EXPERIMENT_NAME,
    DEFAULT_MLFLOW_TRACKING_URI,
    log_autoresearch_result,
)
from stock_analysis.ml.phase2 import Phase2Config, run_phase2
from stock_analysis.pipeline.gcp_model_training import run_gcp_model_training
from stock_analysis.pipeline.gcp_one_shot import run_gcp_one_shot
from stock_analysis.pipeline.one_shot import run_one_shot
from stock_analysis.storage.contracts import (
    AccountRecord,
    AccountTrackingRepository,
    CashflowRecord,
    CashflowType,
    HoldingSnapshotRecord,
    PortfolioSnapshotRecord,
)
from stock_analysis.storage.supabase import (
    SupabaseConfigError,
    create_account_tracking_repository,
)
from stock_analysis.tableau.export import export_existing_run_for_tableau
from stock_analysis.tableau.publish import (
    publish_datasource_if_enabled,
    publish_workbook_if_enabled,
)
from stock_analysis.tableau.workbook import PortfolioWorkbookSpec, write_portfolio_workbook

app = typer.Typer(help="One-shot S&P 500 portfolio assistant.")
CONFIG_OPTION = typer.Option(Path("configs/portfolio.yaml"), "--config", "-c")
GCP_CONFIG_OPTION = typer.Option(Path("configs/portfolio.gcp.yaml"), "--config", "-c")
RUN_ID_OPTION = typer.Option(None, "--run-id")
DATASOURCE_ARGUMENT = typer.Argument(..., help="Path to a .hyper datasource to publish.")
WORKBOOK_ARGUMENT = typer.Argument(..., help="Path to a .twb workbook to publish.")
WORKBOOK_OUTPUT_OPTION = typer.Option(None, "--output", "-o")
FORCE_OPTION = typer.Option(False, "--force", help="Overwrite an existing experiment id.")
EXPERIMENT_CONFIG_OPTION = typer.Option(..., "--config", "-c", help="Experiment YAML config.")
FORECAST_ENGINE_OPTION = typer.Option(
    None,
    "--forecast-engine",
    help="Override forecast.engine for a one-shot run: heuristic or ml.",
)
INPUT_RUN_ROOT_OPTION = typer.Option(
    ...,
    "--input-run-root",
    help="Path to a Phase 1 run root containing silver/gold ML artifacts.",
)
EXPERIMENT_OUTPUT_DIR_OPTION = typer.Option(
    Path("docs/experiments"),
    "--output-dir",
    help="Directory where markdown experiment reports are written.",
)
MAX_REBALANCES_OPTION = typer.Option(
    None,
    "--max-rebalances",
    help="Limit completed rebalance dates for quicker experiment iterations.",
)
MAX_ASSETS_OPTION = typer.Option(
    None,
    "--max-assets",
    help="Limit experiments to the point-in-time most-liquid assets by dollar volume.",
)
EXPERIMENTS_OPTION = typer.Option(
    "E0,E1,E2,E3,E4,E5,E6,E7,E8",
    "--experiments",
    help="Comma-separated Phase 2 experiment ids.",
)
OPTIMIZER_MAX_WEIGHT_OPTION = typer.Option(
    None,
    "--optimizer-max-weight",
    help="Override the Phase 2 optimizer max-weight cap.",
)
AUTORESEARCH_CANDIDATE_OPTION = typer.Option(
    "e8_baseline",
    "--candidate",
    help="Autoresearch candidate id.",
)
AUTORESEARCH_INPUT_RUN_ROOT_OPTION = typer.Option(
    Path("data/runs/phase2-source-20260424"),
    "--input-run-root",
    help="Path to a Phase 1 run root containing silver/gold ML artifacts.",
)
AUTORESEARCH_RESULTS_TSV_OPTION = typer.Option(
    None,
    "--results-tsv",
    help="Optional append-only TSV ledger path.",
)
AUTORESEARCH_JSON_OUTPUT_OPTION = typer.Option(
    None,
    "--json-output",
    help="Optional path for the full JSON result.",
)
AUTORESEARCH_MAX_ASSETS_OPTION = typer.Option(
    100,
    "--max-assets",
    help="Limit each rebalance to the point-in-time most-liquid assets by liquidity column.",
)
AUTORESEARCH_MAX_REBALANCES_OPTION = typer.Option(
    48,
    "--max-rebalances",
    help="Limit completed rebalance dates for faster autoresearch iterations.",
)
AUTORESEARCH_MLFLOW_OPTION = typer.Option(
    False,
    "--mlflow",
    help="Log the evaluator result to MLflow.",
)
AUTORESEARCH_MLFLOW_TRACKING_URI_OPTION = typer.Option(
    None,
    "--mlflow-tracking-uri",
    help=f"MLflow tracking URI. Defaults to {DEFAULT_MLFLOW_TRACKING_URI}.",
)
AUTORESEARCH_MLFLOW_EXPERIMENT_NAME_OPTION = typer.Option(
    DEFAULT_MLFLOW_EXPERIMENT_NAME,
    "--mlflow-experiment-name",
    help="MLflow experiment name used when --mlflow is set.",
)
DEPOSIT_START_DATE_OPTION = typer.Option(
    None,
    "--deposit-start-date",
    help="First contribution date in YYYY-MM-DD format.",
)
TURNOVER_PENALTIES_OPTION = typer.Option(
    "0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5",
    "--turnover-penalties",
    help="Comma-separated lambda_turnover values to sweep.",
)
TURNOVER_OBJECTIVE_OPTION = typer.Option(
    "information_ratio",
    "--objective-metric",
    help="Summary metric used to select the best turnover penalty.",
)
HOLDINGS_PATH_OPTION = typer.Option(None, "--holdings")
CASHFLOW_TYPES: set[str] = {
    "deposit",
    "withdrawal",
    "dividend",
    "interest",
    "fee",
    "tax",
    "transfer",
}
INFLOW_CASHFLOW_TYPES = {"deposit", "dividend", "interest"}
OUTFLOW_CASHFLOW_TYPES = {"withdrawal", "fee", "tax"}


@app.command("run-one-shot")
def run_one_shot_command(
    config: Path = CONFIG_OPTION,
    forecast_engine: str | None = FORECAST_ENGINE_OPTION,
) -> None:
    configure_logging()
    portfolio_config = load_config(config)
    if forecast_engine is not None:
        if forecast_engine not in {"heuristic", "ml"}:
            print("[red]--forecast-engine must be either 'heuristic' or 'ml'.[/red]")
            raise typer.Exit(2)
        portfolio_config.forecast.engine = cast(Literal["heuristic", "ml"], forecast_engine)
    result = run_one_shot(portfolio_config)
    print(f"[green]Completed run[/green] {result.run_id}")
    print(f"Recommendations: {result.recommendations_path}")


@app.command("run-gcp-one-shot")
def run_gcp_one_shot_command(
    config: Path = GCP_CONFIG_OPTION,
    forecast_engine: str | None = FORECAST_ENGINE_OPTION,
) -> None:
    configure_logging()
    portfolio_config = load_config(config)
    if forecast_engine is not None:
        if forecast_engine not in {"heuristic", "ml"}:
            print("[red]--forecast-engine must be either 'heuristic' or 'ml'.[/red]")
            raise typer.Exit(2)
        portfolio_config.forecast.engine = cast(Literal["heuristic", "ml"], forecast_engine)
    try:
        result = run_gcp_one_shot(portfolio_config)
    except RuntimeError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except ValueError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    print(f"[green]Completed GCP run[/green] {result.pipeline.run_id}")
    print(f"GCS run root: {result.gcs_run_root}")
    print(f"Recommendations: {result.pipeline.recommendations_path}")
    if result.bigquery_tables:
        print("BigQuery tables:")
        for name, table_id in sorted(result.bigquery_tables.items()):
            print(f"  {name}: {table_id}")
    if result.model_artifact_uri:
        print(f"Model artifact: {result.model_artifact_uri}")


@app.command("train-gcp-model")
def train_gcp_model_command(
    config: Path = GCP_CONFIG_OPTION,
    forecast_engine: str | None = FORECAST_ENGINE_OPTION,
    promote: bool = typer.Option(
        True,
        "--promote/--no-promote",
        help="Copy the trained artifact to the production model path used by inference.",
    ),
) -> None:
    configure_logging()
    portfolio_config = load_config(config)
    if forecast_engine is not None:
        if forecast_engine not in {"heuristic", "ml"}:
            print("[red]--forecast-engine must be either 'heuristic' or 'ml'.[/red]")
            raise typer.Exit(2)
        portfolio_config.forecast.engine = cast(Literal["heuristic", "ml"], forecast_engine)
    try:
        result = run_gcp_model_training(portfolio_config, promote=promote)
    except RuntimeError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except ValueError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    print(f"[green]Completed GCP model training[/green] {result.run_id}")
    print(f"GCS run root: {result.gcs_run_root}")
    print(f"Model artifact: {result.model_uri}")
    if result.production_model_uri:
        print(f"Production model: {result.production_model_uri}")


@app.command("upsert-account")
def upsert_account_command(
    config: Path = CONFIG_OPTION,
    account_slug: str | None = typer.Option(None, "--account-slug"),
    display_name: str | None = typer.Option(None, "--display-name"),
    owner_id: str | None = typer.Option(None, "--owner-id"),
    base_currency: str = typer.Option("USD", "--base-currency"),
    benchmark_ticker: str = typer.Option("SPY", "--benchmark-ticker"),
) -> None:
    cfg = load_config(config)
    slug = _resolve_account_slug(cfg.live_account.account_slug, account_slug)
    account = _account_tracking_repository(cfg).upsert_account(
        AccountRecord(
            slug=slug,
            display_name=display_name or slug,
            owner_id=owner_id,
            base_currency=base_currency.upper(),
            benchmark_ticker=benchmark_ticker.upper(),
        )
    )
    print(f"[green]Upserted account[/green] {account.slug}")
    print(f"id: {account.id}")


@app.command("register-cashflow")
def register_cashflow_command(
    config: Path = CONFIG_OPTION,
    account_slug: str | None = typer.Option(None, "--account-slug"),
    cashflow_date: str = typer.Option(..., "--date", help="Cashflow date in YYYY-MM-DD format."),
    amount: float = typer.Option(..., "--amount", help="Absolute cashflow amount."),
    cashflow_type: str = typer.Option("deposit", "--type"),
    settled_date: str | None = typer.Option(None, "--settled-date"),
    currency: str = typer.Option("USD", "--currency"),
    source: str = typer.Option("manual", "--source"),
    external_ref: str | None = typer.Option(None, "--external-ref"),
    notes: str | None = typer.Option(None, "--notes"),
    included_in_snapshot_id: str | None = typer.Option(None, "--included-in-snapshot-id"),
) -> None:
    cfg = load_config(config)
    repo = _account_tracking_repository(cfg)
    account = _require_account(
        repo, _resolve_account_slug(cfg.live_account.account_slug, account_slug)
    )
    parsed_type = _parse_cashflow_type(cashflow_type)
    inserted = repo.insert_cashflow(
        CashflowRecord(
            account_id=account.id or "",
            cashflow_date=_parse_required_date(cashflow_date, "--date"),
            settled_date=_parse_date_option(settled_date, "--settled-date"),
            amount=_normalized_cashflow_amount(parsed_type, amount),
            cashflow_type=parsed_type,
            currency=currency.upper(),
            source=source,
            external_ref=external_ref,
            notes=notes,
            included_in_snapshot_id=included_in_snapshot_id,
        )
    )
    print(f"[green]Registered cashflow[/green] {inserted.id}")
    print(
        f"{inserted.cashflow_date.isoformat()} {inserted.cashflow_type} "
        f"{inserted.amount:.2f} {inserted.currency}"
    )


@app.command("list-cashflows")
def list_cashflows_command(
    config: Path = CONFIG_OPTION,
    account_slug: str | None = typer.Option(None, "--account-slug"),
    start_date: str | None = typer.Option(None, "--from"),
    end_date: str | None = typer.Option(None, "--to"),
) -> None:
    cfg = load_config(config)
    repo = _account_tracking_repository(cfg)
    account = _require_account(
        repo, _resolve_account_slug(cfg.live_account.account_slug, account_slug)
    )
    cashflows = repo.list_cashflows(
        account.id or "",
        start_date=_parse_date_option(start_date, "--from"),
        end_date=_parse_date_option(end_date, "--to"),
    )
    if not cashflows:
        print("[yellow]No cashflows found.[/yellow]")
        return
    for cashflow in cashflows:
        settled = cashflow.settled_date.isoformat() if cashflow.settled_date else "-"
        print(
            f"{cashflow.cashflow_date.isoformat()} settled={settled} "
            f"{cashflow.cashflow_type} {cashflow.amount:.2f} {cashflow.currency} "
            f"id={cashflow.id}"
        )


@app.command("import-portfolio-snapshot")
def import_portfolio_snapshot_command(
    config: Path = CONFIG_OPTION,
    account_slug: str | None = typer.Option(None, "--account-slug"),
    snapshot_date: str = typer.Option(..., "--date", help="Snapshot date in YYYY-MM-DD format."),
    holdings_path: Path | None = HOLDINGS_PATH_OPTION,
    market_value: float | None = typer.Option(None, "--market-value"),
    cash_balance: float = typer.Option(0.0, "--cash-balance"),
    total_value: float | None = typer.Option(None, "--total-value"),
    currency: str = typer.Option("USD", "--currency"),
    source: str = typer.Option("manual", "--source"),
) -> None:
    cfg = load_config(config)
    repo = _account_tracking_repository(cfg)
    account = _require_account(
        repo, _resolve_account_slug(cfg.live_account.account_slug, account_slug)
    )
    holdings = _read_holding_snapshot_rows(holdings_path, currency=currency.upper())
    resolved_market_value = _resolve_snapshot_market_value(market_value, holdings)
    resolved_total_value = total_value
    if resolved_total_value is None:
        resolved_total_value = resolved_market_value + cash_balance
    snapshot = repo.insert_portfolio_snapshot(
        PortfolioSnapshotRecord(
            account_id=account.id or "",
            snapshot_date=_parse_required_date(snapshot_date, "--date"),
            market_value=resolved_market_value,
            cash_balance=cash_balance,
            total_value=resolved_total_value,
            currency=currency.upper(),
            source=source,
        ),
        holdings=holdings,
    )
    print(f"[green]Imported portfolio snapshot[/green] {snapshot.id}")
    print(
        f"{snapshot.snapshot_date.isoformat()} total={snapshot.total_value:.2f} "
        f"cash={snapshot.cash_balance:.2f} holdings={len(holdings)}"
    )


@app.command("show-latest-portfolio-snapshot")
def show_latest_portfolio_snapshot_command(
    config: Path = CONFIG_OPTION,
    account_slug: str | None = typer.Option(None, "--account-slug"),
    as_of_date: str | None = typer.Option(None, "--as-of-date"),
) -> None:
    cfg = load_config(config)
    repo = _account_tracking_repository(cfg)
    account = _require_account(
        repo, _resolve_account_slug(cfg.live_account.account_slug, account_slug)
    )
    resolved_as_of = _parse_date_option(as_of_date, "--as-of-date") or date.today()
    snapshot = repo.latest_portfolio_snapshot(account.id or "", as_of_date=resolved_as_of)
    if snapshot is None:
        print("[yellow]No portfolio snapshot found.[/yellow]")
        return
    holdings = repo.list_holding_snapshots(snapshot.id or "")
    print(f"[green]Latest portfolio snapshot[/green] {snapshot.id}")
    print(
        f"{snapshot.snapshot_date.isoformat()} total={snapshot.total_value:.2f} "
        f"cash={snapshot.cash_balance:.2f} holdings={len(holdings)}"
    )


@app.command("run-experiment")
def run_experiment_command(
    config: Path = EXPERIMENT_CONFIG_OPTION,
    force: bool = FORCE_OPTION,
) -> None:
    configure_logging()
    run_dir = run_experiment_from_config(config, force=force)
    print(f"[green]Completed experiment[/green] {run_dir.name}")
    print(f"Artifacts: {run_dir}")


@app.command("run-phase2")
def run_phase2_command(
    input_run_root: Path = INPUT_RUN_ROOT_OPTION,
    output_dir: Path = EXPERIMENT_OUTPUT_DIR_OPTION,
    force: bool = FORCE_OPTION,
    max_rebalances: int | None = MAX_REBALANCES_OPTION,
    max_assets: int | None = MAX_ASSETS_OPTION,
    experiments: str = EXPERIMENTS_OPTION,
    optimizer_max_weight: float | None = OPTIMIZER_MAX_WEIGHT_OPTION,
    no_sweeps: bool = typer.Option(False, "--no-sweeps"),
    no_nested_cv: bool = typer.Option(False, "--no-nested-cv"),
) -> None:
    configure_logging()
    optimizer = (
        OptimizerConfig(max_weight=optimizer_max_weight)
        if optimizer_max_weight is not None
        else OptimizerConfig()
    )
    summary = run_phase2(
        Phase2Config(
            input_run_root=input_run_root,
            output_dir=output_dir,
            force=force,
            experiments=tuple(item.strip() for item in experiments.split(",") if item.strip()),
            max_assets=max_assets,
            optimizer=optimizer,
            backtest=BacktestConfig(max_rebalances=max_rebalances),
            run_sweeps=not no_sweeps,
            lightgbm_nested_cv=not no_nested_cv,
        )
    )
    print("[green]Completed Phase 2 experiment batch[/green]")
    print(summary.drop(columns=["metrics"], errors="ignore").to_string(index=False))


@app.command("autoresearch-eval")
def autoresearch_eval_command(
    candidate: str = AUTORESEARCH_CANDIDATE_OPTION,
    input_run_root: Path = AUTORESEARCH_INPUT_RUN_ROOT_OPTION,
    max_assets: int | None = AUTORESEARCH_MAX_ASSETS_OPTION,
    max_rebalances: int | None = AUTORESEARCH_MAX_REBALANCES_OPTION,
    optimizer_max_weight: float = typer.Option(0.30, "--optimizer-max-weight"),
    risk_aversion: float = typer.Option(10.0, "--risk-aversion"),
    min_trade_weight: float = typer.Option(0.005, "--min-trade-weight"),
    lambda_turnover: float = typer.Option(5.0, "--lambda-turnover"),
    commission_rate: float = typer.Option(
        0.02,
        "--commission-rate",
        help="Commission rate applied to absolute traded notional. Example: 0.02 means 2%.",
    ),
    max_trade_abs_weight: float | None = typer.Option(None, "--max-trade-abs-weight"),
    initial_portfolio_value: float = typer.Option(1000.0, "--initial-portfolio-value"),
    monthly_deposit_amount: float = typer.Option(0.0, "--monthly-deposit-amount"),
    deposit_frequency_days: int = typer.Option(30, "--deposit-frequency-days"),
    deposit_start_date: str | None = DEPOSIT_START_DATE_OPTION,
    rebalance_on_deposit_day: bool = typer.Option(
        True,
        "--rebalance-on-deposit-day/--no-rebalance-on-deposit-day",
    ),
    no_trade_band: float = typer.Option(0.0, "--no-trade-band"),
    horizon_days: int | None = typer.Option(None, "--horizon-days"),
    rebalance_step_days: int = typer.Option(5, "--rebalance-step-days"),
    embargo_days: int = typer.Option(15, "--embargo-days"),
    cost_bps: float = typer.Option(5.0, "--cost-bps"),
    covariance_lookback_days: int = typer.Option(252, "--covariance-lookback-days"),
    liquidity_column: str = typer.Option("dollar_volume_21d", "--liquidity-column"),
    iteration_id: str | None = typer.Option(None, "--iteration-id"),
    results_tsv: Path | None = AUTORESEARCH_RESULTS_TSV_OPTION,
    json_output: Path | None = AUTORESEARCH_JSON_OUTPUT_OPTION,
    enable_mlflow: bool = AUTORESEARCH_MLFLOW_OPTION,
    mlflow_tracking_uri: str | None = AUTORESEARCH_MLFLOW_TRACKING_URI_OPTION,
    mlflow_experiment_name: str = AUTORESEARCH_MLFLOW_EXPERIMENT_NAME_OPTION,
) -> None:
    configure_logging()
    result = evaluate_candidate(
        AutoresearchEvalConfig(
            candidate_id=candidate,
            input_run_root=input_run_root,
            max_assets=max_assets,
            max_rebalances=max_rebalances,
            optimizer_max_weight=optimizer_max_weight,
            risk_aversion=risk_aversion,
            min_trade_weight=min_trade_weight,
            lambda_turnover=lambda_turnover,
            commission_rate=commission_rate,
            max_trade_abs_weight=max_trade_abs_weight,
            horizon_days=horizon_days,
            rebalance_step_days=rebalance_step_days,
            embargo_days=embargo_days,
            cost_bps=cost_bps,
            covariance_lookback_days=covariance_lookback_days,
            liquidity_column=liquidity_column,
            iteration_id=iteration_id,
            initial_portfolio_value=initial_portfolio_value,
            monthly_deposit_amount=monthly_deposit_amount,
            deposit_frequency_days=deposit_frequency_days,
            deposit_start_date=_parse_date_option(deposit_start_date, "--deposit-start-date"),
            rebalance_on_deposit_day=rebalance_on_deposit_day,
            no_trade_band=no_trade_band,
        )
    )
    if results_tsv is not None:
        append_result_tsv(results_tsv, result)
    if enable_mlflow:
        artifacts = [results_tsv] if results_tsv is not None else []
        try:
            run_id = log_autoresearch_result(
                result,
                tracking_uri=mlflow_tracking_uri,
                experiment_name=mlflow_experiment_name,
                artifacts=artifacts,
            )
        except RuntimeError as exc:
            print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        result["mlflow"] = {
            "experiment_name": mlflow_experiment_name,
            "run_id": run_id,
            "tracking_uri": mlflow_tracking_uri or DEFAULT_MLFLOW_TRACKING_URI,
        }
    output = result_to_json(result)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(f"{output}\n", encoding="utf-8")
    print(output)


@app.command("tune-turnover")
def tune_turnover_command(
    candidate: str = AUTORESEARCH_CANDIDATE_OPTION,
    input_run_root: Path = AUTORESEARCH_INPUT_RUN_ROOT_OPTION,
    max_assets: int | None = AUTORESEARCH_MAX_ASSETS_OPTION,
    max_rebalances: int | None = AUTORESEARCH_MAX_REBALANCES_OPTION,
    optimizer_max_weight: float = typer.Option(0.30, "--optimizer-max-weight"),
    risk_aversion: float = typer.Option(10.0, "--risk-aversion"),
    min_trade_weight: float = typer.Option(0.005, "--min-trade-weight"),
    commission_rate: float = typer.Option(
        0.02,
        "--commission-rate",
        help="Commission rate applied to absolute traded notional. Example: 0.02 means 2%.",
    ),
    max_trade_abs_weight: float | None = typer.Option(None, "--max-trade-abs-weight"),
    initial_portfolio_value: float = typer.Option(1000.0, "--initial-portfolio-value"),
    monthly_deposit_amount: float = typer.Option(0.0, "--monthly-deposit-amount"),
    deposit_frequency_days: int = typer.Option(30, "--deposit-frequency-days"),
    deposit_start_date: str | None = DEPOSIT_START_DATE_OPTION,
    rebalance_on_deposit_day: bool = typer.Option(
        True,
        "--rebalance-on-deposit-day/--no-rebalance-on-deposit-day",
    ),
    no_trade_band: float = typer.Option(0.0, "--no-trade-band"),
    turnover_penalties: str = TURNOVER_PENALTIES_OPTION,
    objective_metric: str = TURNOVER_OBJECTIVE_OPTION,
    horizon_days: int | None = typer.Option(None, "--horizon-days"),
    rebalance_step_days: int = typer.Option(5, "--rebalance-step-days"),
    embargo_days: int = typer.Option(15, "--embargo-days"),
    cost_bps: float = typer.Option(5.0, "--cost-bps"),
    covariance_lookback_days: int = typer.Option(252, "--covariance-lookback-days"),
    liquidity_column: str = typer.Option("dollar_volume_21d", "--liquidity-column"),
    iteration_id: str | None = typer.Option(None, "--iteration-id"),
    json_output: Path | None = AUTORESEARCH_JSON_OUTPUT_OPTION,
) -> None:
    configure_logging()
    penalties = _parse_float_list(turnover_penalties)
    result = evaluate_turnover_sweep(
        TurnoverSweepConfig(
            base=AutoresearchEvalConfig(
                candidate_id=candidate,
                input_run_root=input_run_root,
                max_assets=max_assets,
                max_rebalances=max_rebalances,
                optimizer_max_weight=optimizer_max_weight,
                risk_aversion=risk_aversion,
                min_trade_weight=min_trade_weight,
                commission_rate=commission_rate,
                max_trade_abs_weight=max_trade_abs_weight,
                horizon_days=horizon_days,
                rebalance_step_days=rebalance_step_days,
                embargo_days=embargo_days,
                cost_bps=cost_bps,
                covariance_lookback_days=covariance_lookback_days,
                liquidity_column=liquidity_column,
                iteration_id=iteration_id,
                initial_portfolio_value=initial_portfolio_value,
                monthly_deposit_amount=monthly_deposit_amount,
                deposit_frequency_days=deposit_frequency_days,
                deposit_start_date=_parse_date_option(deposit_start_date, "--deposit-start-date"),
                rebalance_on_deposit_day=rebalance_on_deposit_day,
                no_trade_band=no_trade_band,
            ),
            penalties=penalties,
            objective_metric=objective_metric,
        )
    )
    output = result_to_json(result)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(f"{output}\n", encoding="utf-8")
    print(output)


@app.command("export-tableau")
def export_tableau_command(
    config: Path = CONFIG_OPTION,
    run_id: str | None = RUN_ID_OPTION,
) -> None:
    cfg = load_config(config)
    if run_id:
        cfg.run.run_id = run_id
    if not cfg.tableau.export_csv and not cfg.tableau.export_hyper:
        print("[yellow]No Tableau export is enabled in config.[/yellow]")
        raise typer.Exit(0)
    effective_run_id = run_id or cfg.run.run_id
    if not effective_run_id:
        print("[red]Provide --run-id or set run.run_id in the config.[/red]")
        raise typer.Exit(2)
    outputs = export_existing_run_for_tableau(cfg, effective_run_id)
    print(f"[green]Exported Tableau outputs for existing run[/green] {effective_run_id}")
    for key, path in outputs.items():
        print(f"{key}: {path}")


@app.command("publish-tableau")
def publish_tableau_command(
    datasource: Path = DATASOURCE_ARGUMENT,
    config: Path = CONFIG_OPTION,
) -> None:
    load_local_env()
    cfg = load_config(config)
    published_id = publish_datasource_if_enabled(cfg.tableau, datasource)
    if published_id is None:
        print("[yellow]Tableau publishing is disabled.[/yellow]")
    else:
        print(f"[green]Published Tableau datasource[/green] {published_id}")


@app.command("generate-tableau-workbook")
def generate_tableau_workbook_command(
    config: Path = CONFIG_OPTION,
    output: Path | None = WORKBOOK_OUTPUT_OPTION,
) -> None:
    load_local_env()
    cfg = load_config(config)
    server_url = cfg.tableau.server_url or os.getenv("TABLEAU_SERVER_URL")
    if not server_url:
        print("[red]Set tableau.server_url in config or TABLEAU_SERVER_URL in .env.[/red]")
        raise typer.Exit(2)
    workbook_path = output or cfg.tableau.workbook_output_path
    spec = PortfolioWorkbookSpec(
        server_url=server_url,
        datasource_name=cfg.tableau.datasource_name,
        workbook_name=cfg.tableau.workbook_name,
        project_name=cfg.tableau.project_name,
        site_name=cfg.tableau.site_name,
    )
    written = write_portfolio_workbook(spec, workbook_path)
    print(f"[green]Generated Tableau workbook[/green] {written}")


@app.command("publish-tableau-workbook")
def publish_tableau_workbook_command(
    workbook: Path = WORKBOOK_ARGUMENT,
    config: Path = CONFIG_OPTION,
) -> None:
    load_local_env()
    cfg = load_config(config)
    published_id = publish_workbook_if_enabled(cfg.tableau, workbook)
    if published_id is None:
        print("[yellow]Tableau publishing is disabled.[/yellow]")
    else:
        print(f"[green]Published Tableau workbook[/green] {published_id}")


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError as exc:
            print(f"[red]Invalid float in comma-separated list: {stripped}[/red]")
            raise typer.Exit(2) from exc
    if not values:
        print("[red]Provide at least one numeric value.[/red]")
        raise typer.Exit(2)
    return tuple(values)


def _parse_date_option(raw: str | None, option_name: str) -> date | None:
    if raw is None:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        print(f"[red]Invalid date for {option_name}: {raw}. Use YYYY-MM-DD.[/red]")
        raise typer.Exit(2) from exc


def _parse_required_date(raw: str, option_name: str) -> date:
    parsed = _parse_date_option(raw, option_name)
    if parsed is None:
        print(f"[red]Provide {option_name} in YYYY-MM-DD format.[/red]")
        raise typer.Exit(2)
    return parsed


def _account_tracking_repository(config) -> AccountTrackingRepository:
    load_local_env()
    try:
        return create_account_tracking_repository(config.supabase)
    except SupabaseConfigError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


def _resolve_account_slug(config_slug: str | None, override_slug: str | None) -> str:
    slug = override_slug or config_slug
    if not slug:
        print("[red]Provide --account-slug or set live_account.account_slug in the config.[/red]")
        raise typer.Exit(2)
    return slug


def _require_account(repository: AccountTrackingRepository, slug: str) -> AccountRecord:
    account = repository.get_account_by_slug(slug)
    if account is None:
        print(f"[red]Account does not exist in Supabase: {slug}. Run upsert-account first.[/red]")
        raise typer.Exit(2)
    if account.id is None:
        print(f"[red]Account row for {slug} did not include an id.[/red]")
        raise typer.Exit(2)
    return account


def _parse_cashflow_type(raw: str) -> CashflowType:
    parsed = raw.strip().lower()
    if parsed not in CASHFLOW_TYPES:
        allowed = ", ".join(sorted(CASHFLOW_TYPES))
        print(f"[red]Invalid --type: {raw}. Allowed values: {allowed}.[/red]")
        raise typer.Exit(2)
    return cast(CashflowType, parsed)


def _normalized_cashflow_amount(cashflow_type: CashflowType, amount: float) -> float:
    if amount == 0:
        print("[red]Cashflow amount cannot be zero.[/red]")
        raise typer.Exit(2)
    if cashflow_type in INFLOW_CASHFLOW_TYPES:
        return abs(float(amount))
    if cashflow_type in OUTFLOW_CASHFLOW_TYPES:
        return -abs(float(amount))
    return float(amount)


def _read_holding_snapshot_rows(
    holdings_path: Path | None,
    *,
    currency: str,
) -> list[HoldingSnapshotRecord]:
    if holdings_path is None:
        return []
    if not holdings_path.exists():
        print(f"[red]Holdings file does not exist: {holdings_path}[/red]")
        raise typer.Exit(2)
    holdings = _read_snapshot_frame(holdings_path)
    required_columns = {"ticker", "market_value"}
    missing = required_columns - set(holdings.columns)
    if missing:
        print(f"[red]Holdings file is missing columns: {sorted(missing)}[/red]")
        raise typer.Exit(2)
    market_values = pd.to_numeric(holdings["market_value"], errors="coerce")
    if market_values.isna().any() or market_values.lt(0).any():
        print("[red]Holdings market_value must be nonnegative numbers.[/red]")
        raise typer.Exit(2)
    quantities = _optional_numeric_series(holdings, "quantity")
    prices = _optional_numeric_series(holdings, "price")
    rows: list[HoldingSnapshotRecord] = []
    normalized = holdings.reset_index(drop=True)
    for index in range(len(normalized)):
        ticker = str(normalized.at[index, "ticker"]).strip()
        if not ticker:
            print("[red]Holdings ticker values cannot be blank.[/red]")
            raise typer.Exit(2)
        rows.append(
            HoldingSnapshotRecord(
                snapshot_id="pending",
                ticker=ticker,
                market_value=float(market_values.iloc[index]),
                quantity=_optional_series_value(quantities, index),
                price=_optional_series_value(prices, index),
                currency=currency,
            )
        )
    return rows


def _read_snapshot_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    print(f"[red]Unsupported holdings file format: {path.suffix}. Use CSV or Parquet.[/red]")
    raise typer.Exit(2)


def _optional_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().lt(0).any():
        print(f"[red]Holdings {column} values cannot be negative.[/red]")
        raise typer.Exit(2)
    return values


def _optional_series_value(values: pd.Series | None, index: int) -> float | None:
    if values is None:
        return None
    value = values.iloc[index]
    if pd.isna(value):
        return None
    return float(value)


def _resolve_snapshot_market_value(
    market_value: float | None,
    holdings: list[HoldingSnapshotRecord],
) -> float:
    if market_value is not None:
        if market_value < 0:
            print("[red]--market-value cannot be negative.[/red]")
            raise typer.Exit(2)
        return market_value
    if holdings:
        return float(sum(holding.market_value for holding in holdings))
    print("[red]Provide --market-value or --holdings.[/red]")
    raise typer.Exit(2)
