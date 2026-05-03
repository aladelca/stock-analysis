from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

from typer.testing import CliRunner

from stock_analysis import cli
from stock_analysis.domain.models import PipelineResult
from stock_analysis.pipeline.gcp_model_training import GcpModelTrainingResult
from stock_analysis.pipeline.gcp_one_shot import GcpPipelineResult
from stock_analysis.storage.contracts import (
    AccountRecord,
    CashflowRecord,
    HoldingSnapshotRecord,
    PerformanceSnapshotRecord,
    PortfolioSnapshotRecord,
    RecommendationLineRecord,
    RecommendationRunRecord,
)


def test_upsert_account_command_uses_config_slug(tmp_path: Path, monkeypatch) -> None:
    repository = FakeAccountTrackingRepository()
    monkeypatch.setattr(cli, "create_account_tracking_repository", lambda _config: repository)

    result = runner.invoke(
        cli.app,
        [
            "upsert-account",
            "--config",
            str(_config_path(tmp_path)),
            "--display-name",
            "Main Brokerage",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Upserted account" in result.output
    assert repository.account == AccountRecord(
        id="account-1",
        slug="main",
        display_name="Main Brokerage",
        base_currency="USD",
        benchmark_ticker="SPY",
    )


def test_register_cashflow_command_normalizes_outflows(tmp_path: Path, monkeypatch) -> None:
    repository = FakeAccountTrackingRepository()
    repository.account = AccountRecord(id="account-1", slug="main", display_name="Main")
    monkeypatch.setattr(cli, "create_account_tracking_repository", lambda _config: repository)

    result = runner.invoke(
        cli.app,
        [
            "register-cashflow",
            "--config",
            str(_config_path(tmp_path)),
            "--date",
            "2026-04-24",
            "--amount",
            "100",
            "--type",
            "withdrawal",
        ],
    )

    assert result.exit_code == 0, result.output
    assert repository.cashflows[0].amount == -100.0
    assert repository.cashflows[0].cashflow_type == "withdrawal"


def test_list_cashflows_command_prints_registered_rows(tmp_path: Path, monkeypatch) -> None:
    repository = FakeAccountTrackingRepository()
    repository.account = AccountRecord(id="account-1", slug="main", display_name="Main")
    repository.cashflows = [
        CashflowRecord(
            id="cashflow-1",
            account_id="account-1",
            cashflow_date=date(2026, 4, 24),
            amount=500.0,
            cashflow_type="deposit",
        )
    ]
    monkeypatch.setattr(cli, "create_account_tracking_repository", lambda _config: repository)

    result = runner.invoke(
        cli.app,
        ["list-cashflows", "--config", str(_config_path(tmp_path))],
    )

    assert result.exit_code == 0, result.output
    assert "2026-04-24" in result.output
    assert "deposit 500.00 USD" in result.output


def test_import_portfolio_snapshot_reads_holdings_file(tmp_path: Path, monkeypatch) -> None:
    repository = FakeAccountTrackingRepository()
    repository.account = AccountRecord(id="account-1", slug="main", display_name="Main")
    monkeypatch.setattr(cli, "create_account_tracking_repository", lambda _config: repository)
    holdings_path = tmp_path / "holdings.csv"
    holdings_path.write_text(
        "ticker,market_value,quantity,price\nAAPL,400,2,200\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        [
            "import-portfolio-snapshot",
            "--config",
            str(_config_path(tmp_path)),
            "--date",
            "2026-04-24",
            "--cash-balance",
            "25",
            "--holdings",
            str(holdings_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert repository.snapshot == PortfolioSnapshotRecord(
        id="snapshot-1",
        account_id="account-1",
        snapshot_date=date(2026, 4, 24),
        market_value=400.0,
        cash_balance=25.0,
        total_value=425.0,
    )
    assert repository.holdings == [
        HoldingSnapshotRecord(
            snapshot_id="snapshot-1",
            ticker="AAPL",
            market_value=400.0,
            quantity=2.0,
            price=200.0,
        )
    ]


def test_run_gcp_one_shot_command_prints_cloud_outputs(tmp_path: Path, monkeypatch) -> None:
    def fake_run_gcp_one_shot(config):
        assert config.gcp.enabled is True
        return GcpPipelineResult(
            pipeline=PipelineResult(
                run_id="gcp-run-1",
                as_of_date=date(2026, 4, 24),
                output_root="gs://bucket/runs/gcp-run-1",
                recommendations_path=(
                    "gs://bucket/runs/gcp-run-1/gold/portfolio_recommendations.parquet"
                ),
                risk_metrics_path="gs://bucket/runs/gcp-run-1/gold/portfolio_risk_metrics.parquet",
                sector_exposure_path="gs://bucket/runs/gcp-run-1/gold/sector_exposure.parquet",
            ),
            gcs_run_root="gs://bucket/runs/gcp-run-1",
            bigquery_tables={"portfolio_dashboard_mart": "project.gold.portfolio_dashboard_mart"},
        )

    monkeypatch.setattr(cli, "run_gcp_one_shot", fake_run_gcp_one_shot)
    config_path = tmp_path / "portfolio.gcp.yaml"
    config_path.write_text(
        """
gcp:
  enabled: true
  project_id: project
  bucket: bucket
""".lstrip(),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        ["run-gcp-one-shot", "--config", str(config_path), "--forecast-engine", "ml"],
    )

    assert result.exit_code == 0, result.output
    assert "Completed GCP run" in result.output
    assert "gs://bucket/runs/gcp-run-1" in result.output
    assert "project.gold.portfolio_dashboard_mart" in result.output


def test_train_gcp_model_command_prints_model_artifact(tmp_path: Path, monkeypatch) -> None:
    def fake_run_gcp_model_training(config, *, promote):
        assert config.gcp.enabled is True
        assert promote is True
        return GcpModelTrainingResult(
            run_id="train-run-1",
            gcs_run_root="gs://bucket/runs/train-run-1",
            model_uri="gs://bucket/models/runs/train-run-1/model.cloudpickle",
            production_model_uri="gs://bucket/models/production/current.json",
            artifact_uris=[],
        )

    monkeypatch.setattr(cli, "run_gcp_model_training", fake_run_gcp_model_training)
    config_path = tmp_path / "portfolio.gcp.yaml"
    config_path.write_text(
        """
forecast:
  engine: ml
gcp:
  enabled: true
  project_id: project
  bucket: bucket
""".lstrip(),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        ["train-gcp-model", "--config", str(config_path), "--forecast-engine", "ml"],
    )

    assert result.exit_code == 0, result.output
    assert "Completed GCP model training" in result.output
    assert "gs://bucket/models/production/current.json" in result.output


runner = CliRunner()


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        """
live_account:
  account_slug: main
supabase:
  enabled: true
""".lstrip(),
        encoding="utf-8",
    )
    return path


class FakeAccountTrackingRepository:
    def __init__(self) -> None:
        self.account: AccountRecord | None = None
        self.cashflows: list[CashflowRecord] = []
        self.snapshot: PortfolioSnapshotRecord | None = None
        self.holdings: list[HoldingSnapshotRecord] = []

    def get_account_by_slug(self, slug: str) -> AccountRecord | None:
        if self.account is not None and self.account.slug == slug:
            return self.account
        return None

    def upsert_account(self, account: AccountRecord) -> AccountRecord:
        self.account = replace(account, id=account.id or "account-1")
        return self.account

    def insert_cashflow(self, cashflow: CashflowRecord) -> CashflowRecord:
        inserted = replace(cashflow, id=cashflow.id or f"cashflow-{len(self.cashflows) + 1}")
        self.cashflows.append(inserted)
        return inserted

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
        self.snapshot = replace(snapshot, id=snapshot.id or "snapshot-1")
        self.holdings = [
            replace(holding, snapshot_id=self.snapshot.id or "") for holding in holdings or []
        ]
        return self.snapshot

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
        return [self.snapshot] if self.snapshot is not None else []

    def list_holding_snapshots(self, snapshot_id: str) -> list[HoldingSnapshotRecord]:
        return [holding for holding in self.holdings if holding.snapshot_id == snapshot_id]

    def insert_recommendation_run(self, run: RecommendationRunRecord) -> RecommendationRunRecord:
        return run

    def insert_recommendation_lines(
        self,
        lines: list[RecommendationLineRecord],
    ) -> list[RecommendationLineRecord]:
        return lines

    def insert_performance_snapshot(
        self,
        snapshot: PerformanceSnapshotRecord,
    ) -> PerformanceSnapshotRecord:
        return snapshot
