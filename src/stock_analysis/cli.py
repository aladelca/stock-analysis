from __future__ import annotations

import os
from pathlib import Path

import typer
from rich import print

from stock_analysis.config import load_config
from stock_analysis.env import load_local_env
from stock_analysis.logging import configure_logging
from stock_analysis.ml.experiments import run_experiment_from_config
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


@app.command("run-one-shot")
def run_one_shot_command(
    config: Path = CONFIG_OPTION,
) -> None:
    configure_logging()
    result = run_one_shot(load_config(config))
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
) -> None:
    configure_logging()
    summary = run_phase2(
        Phase2Config(
            input_run_root=input_run_root,
            output_dir=output_dir,
            force=force,
        )
    )
    print("[green]Completed Phase 2 experiment batch[/green]")
    print(summary.drop(columns=["metrics"], errors="ignore").to_string(index=False))


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
