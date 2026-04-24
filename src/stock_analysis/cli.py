from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

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
from stock_analysis.pipeline.one_shot import run_one_shot
from stock_analysis.tableau.export import export_existing_run_for_tableau
from stock_analysis.tableau.publish import (
    publish_datasource_if_enabled,
    publish_workbook_if_enabled,
)
from stock_analysis.tableau.workbook import PortfolioWorkbookSpec, write_portfolio_workbook

app = typer.Typer(help="One-shot S&P 500 portfolio assistant.")
CONFIG_OPTION = typer.Option(Path("configs/portfolio.yaml"), "--config", "-c")
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
    help="Limit experiments to the latest most-liquid assets by dollar volume.",
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
    help="Limit to the latest most-liquid assets by liquidity column.",
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
            horizon_days=horizon_days,
            rebalance_step_days=rebalance_step_days,
            embargo_days=embargo_days,
            cost_bps=cost_bps,
            covariance_lookback_days=covariance_lookback_days,
            liquidity_column=liquidity_column,
            iteration_id=iteration_id,
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
                horizon_days=horizon_days,
                rebalance_step_days=rebalance_step_days,
                embargo_days=embargo_days,
                cost_bps=cost_bps,
                covariance_lookback_days=covariance_lookback_days,
                liquidity_column=liquidity_column,
                iteration_id=iteration_id,
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
