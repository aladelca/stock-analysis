from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest

from stock_analysis.config import SupabaseConfig
from stock_analysis.storage.contracts import (
    AccountRecord,
    CashflowRecord,
    HoldingSnapshotRecord,
    PerformanceSnapshotRecord,
    PortfolioSnapshotRecord,
    RecommendationLineRecord,
    RecommendationRunRecord,
)
from stock_analysis.storage.supabase import (
    SupabaseAccountTrackingRepository,
    SupabaseConfigError,
    SupabaseRepositoryError,
    create_account_tracking_repository,
    create_supabase_client,
)


def test_repository_factory_requires_enabled_supabase() -> None:
    with pytest.raises(SupabaseConfigError, match="disabled"):
        create_account_tracking_repository(SupabaseConfig(), environ={})


def test_create_supabase_client_requires_credentials() -> None:
    config = SupabaseConfig(enabled=True)

    with pytest.raises(SupabaseConfigError, match="SUPABASE_URL"):
        create_supabase_client(config, environ={})

    with pytest.raises(SupabaseConfigError, match="SUPABASE_SERVICE_ROLE_KEY"):
        create_supabase_client(config, environ={"SUPABASE_URL": "https://example.supabase.co"})


def test_repository_upserts_account_and_filters_cashflows() -> None:
    client = FakeSupabaseClient()
    repo = SupabaseAccountTrackingRepository(client, SupabaseConfig())

    account = repo.upsert_account(
        AccountRecord(
            slug="main",
            display_name="Main Brokerage",
            owner_id="user-1",
        )
    )
    assert account.id == "accounts-1"
    assert account.owner_id == "user-1"
    assert repo.get_account_by_slug("main") == account

    inserted = repo.insert_cashflow(
        CashflowRecord(
            account_id=account.id or "",
            cashflow_date=date(2026, 4, 10),
            settled_date=date(2026, 4, 11),
            amount=500.0,
            cashflow_type="deposit",
        )
    )
    repo.insert_cashflow(
        CashflowRecord(
            account_id=account.id or "",
            cashflow_date=date(2026, 5, 10),
            amount=250.0,
            cashflow_type="deposit",
        )
    )

    assert inserted.id == "cashflows-1"
    assert client.tables["cashflows"][0]["cashflow_date"] == "2026-04-10"
    assert "notes" not in client.tables["cashflows"][0]

    cashflows = repo.list_cashflows(
        account.id or "",
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 30),
    )

    assert [cashflow.amount for cashflow in cashflows] == [500.0]
    assert cashflows[0].cashflow_date == date(2026, 4, 10)
    assert cashflows[0].cashflow_type == "deposit"


def test_repository_writes_snapshot_and_holding_rows() -> None:
    client = FakeSupabaseClient()
    repo = SupabaseAccountTrackingRepository(client, SupabaseConfig())

    snapshot = repo.insert_portfolio_snapshot(
        PortfolioSnapshotRecord(
            account_id="account-1",
            snapshot_date=date(2026, 4, 24),
            market_value=900.0,
            cash_balance=100.0,
            total_value=1000.0,
        ),
        holdings=[
            HoldingSnapshotRecord(
                snapshot_id="pending",
                ticker="AAPL",
                quantity=2.0,
                market_value=400.0,
                price=200.0,
            )
        ],
    )

    assert snapshot.id == "portfolio_snapshots-1"
    latest = repo.latest_portfolio_snapshot("account-1", as_of_date=date(2026, 4, 30))
    assert latest == snapshot
    snapshots = repo.list_portfolio_snapshots("account-1", end_date=date(2026, 4, 30))
    assert snapshots == [snapshot]

    holdings = repo.list_holding_snapshots(snapshot.id or "")
    assert holdings == [
        HoldingSnapshotRecord(
            id="holding_snapshots-1",
            snapshot_id=snapshot.id or "",
            ticker="AAPL",
            quantity=2.0,
            market_value=400.0,
            price=200.0,
        )
    ]


def test_repository_deletes_partial_snapshot_when_holding_insert_fails() -> None:
    client = FakeSupabaseClient()
    client.fail_inserts_for.add("holding_snapshots")
    repo = SupabaseAccountTrackingRepository(client, SupabaseConfig())

    with pytest.raises(SupabaseRepositoryError, match="deleted the partial portfolio snapshot"):
        repo.insert_portfolio_snapshot(
            PortfolioSnapshotRecord(
                account_id="account-1",
                snapshot_date=date(2026, 4, 24),
                market_value=900.0,
                cash_balance=100.0,
                total_value=1000.0,
            ),
            holdings=[
                HoldingSnapshotRecord(
                    snapshot_id="pending",
                    ticker="AAPL",
                    market_value=400.0,
                )
            ],
        )

    assert client.tables["portfolio_snapshots"] == []


def test_repository_writes_recommendations_and_performance_snapshot() -> None:
    client = FakeSupabaseClient()
    repo = SupabaseAccountTrackingRepository(client, SupabaseConfig())

    run = repo.insert_recommendation_run(
        RecommendationRunRecord(
            account_id="account-1",
            run_id="run-1",
            as_of_date=date(2026, 4, 24),
            data_as_of_date=date(2026, 4, 24),
            model_version="e8-scale-0p5-contribution-aware-v1",
            ml_score_scale=0.5,
            config_hash="abc123",
            expected_return_is_calibrated=True,
            optimizer_return_unit="5d_return",
            calibration_enabled=True,
            calibration_method="isotonic",
            calibration_target="fwd_return_5d",
            calibration_model_version="lightgbm_return_zscore:isotonic",
            calibration_status="calibrated",
            calibration_trained_through_date=date(2026, 4, 23),
            calibration_observations=250,
            calibration_mae=0.015,
            calibration_rmse=0.02,
            calibration_rank_ic=0.12,
        )
    )
    lines = repo.insert_recommendation_lines(
        [
            RecommendationLineRecord(
                recommendation_run_id=run.id or "",
                ticker="SPY",
                target_weight=0.3,
                action="BUY",
                forecast_score=0.2,
                expected_return=0.2,
                forecast_horizon_days=5,
                forecast_start_date=date(2026, 4, 24),
                outcome_status="pending",
            )
        ]
    )
    performance = repo.insert_performance_snapshot(
        PerformanceSnapshotRecord(
            account_id="account-1",
            as_of_date=date(2026, 4, 24),
            account_total_value=1000.0,
            total_deposits=500.0,
            net_external_cashflow=500.0,
            initial_value=500.0,
            invested_capital=1000.0,
            return_on_invested_capital=0.0,
            spy_same_cashflow_value=980.0,
        )
    )

    assert run.id == "recommendation_runs-1"
    assert run.expected_return_is_calibrated is True
    assert run.optimizer_return_unit == "5d_return"
    assert run.calibration_enabled is True
    assert run.calibration_status == "calibrated"
    assert run.calibration_trained_through_date == date(2026, 4, 23)
    assert run.calibration_observations == 250
    assert run.calibration_mae == pytest.approx(0.015)
    assert client.tables["recommendation_runs"][0]["calibration_method"] == "isotonic"
    assert lines[0].id == "recommendation_lines-1"
    assert lines[0].forecast_score == pytest.approx(0.2)
    assert lines[0].forecast_horizon_days == 5
    assert lines[0].forecast_start_date == date(2026, 4, 24)
    assert repo.list_recommendation_runs("account-1") == [run]
    assert repo.list_recommendation_lines([run.id or ""]) == lines
    assert performance.id == "performance_snapshots-1"
    assert performance.initial_value == pytest.approx(500.0)
    assert performance.invested_capital == pytest.approx(1000.0)
    assert client.tables["performance_snapshots"][0]["as_of_date"] == "2026-04-24"
    assert repo.list_performance_snapshots("account-1") == [performance]


def test_repository_paginates_recommendation_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stock_analysis.storage.supabase.SUPABASE_PAGE_SIZE", 2)
    client = FakeSupabaseClient()
    repo = SupabaseAccountTrackingRepository(client, SupabaseConfig())
    run = repo.insert_recommendation_run(
        RecommendationRunRecord(
            account_id="account-1",
            run_id="run-1",
            as_of_date=date(2026, 4, 24),
            data_as_of_date=date(2026, 4, 24),
            model_version="model",
            ml_score_scale=1.0,
            config_hash="abc123",
        )
    )
    repo.insert_recommendation_lines(
        [
            RecommendationLineRecord(
                recommendation_run_id=run.id or "",
                ticker=f"T{index}",
            )
            for index in range(5)
        ]
    )

    lines = repo.list_recommendation_lines([run.id or ""])

    assert [line.ticker for line in lines] == ["T0", "T1", "T2", "T3", "T4"]


@dataclass
class FakeResponse:
    data: list[dict[str, Any]]


class FakeSupabaseClient:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.fail_inserts_for: set[str] = set()

    def table(self, name: str) -> FakeQuery:
        return FakeQuery(self, name)


class FakeQuery:
    def __init__(self, client: FakeSupabaseClient, table_name: str) -> None:
        self.client = client
        self.table_name = table_name
        self.action = "select"
        self.payload: dict[str, Any] | list[dict[str, Any]] | None = None
        self.filters: list[tuple[str, str, Any]] = []
        self.orders: list[tuple[str, bool]] = []
        self.row_limit: int | None = None
        self.row_range: tuple[int, int] | None = None
        self.on_conflict: str | None = None

    def select(self, _columns: str) -> FakeQuery:
        self.action = "select"
        return self

    def insert(self, payload: dict[str, Any] | list[dict[str, Any]]) -> FakeQuery:
        self.action = "insert"
        self.payload = payload
        return self

    def upsert(
        self,
        payload: dict[str, Any] | list[dict[str, Any]],
        *,
        on_conflict: str,
    ) -> FakeQuery:
        self.action = "upsert"
        self.payload = payload
        self.on_conflict = on_conflict
        return self

    def delete(self) -> FakeQuery:
        self.action = "delete"
        return self

    def eq(self, column: str, value: Any) -> FakeQuery:
        self.filters.append(("eq", column, value))
        return self

    def gte(self, column: str, value: Any) -> FakeQuery:
        self.filters.append(("gte", column, value))
        return self

    def lte(self, column: str, value: Any) -> FakeQuery:
        self.filters.append(("lte", column, value))
        return self

    def in_(self, column: str, values: list[Any]) -> FakeQuery:
        self.filters.append(("in", column, values))
        return self

    def order(self, column: str, *, desc: bool = False) -> FakeQuery:
        self.orders.append((column, desc))
        return self

    def limit(self, row_limit: int) -> FakeQuery:
        self.row_limit = row_limit
        return self

    def range(self, start: int, end: int) -> FakeQuery:
        self.row_range = (start, end)
        return self

    def execute(self) -> FakeResponse:
        if self.action == "insert":
            return FakeResponse(self._insert_rows())
        if self.action == "upsert":
            return FakeResponse(self._upsert_rows())
        if self.action == "delete":
            return FakeResponse(self._delete_rows())
        return FakeResponse(self._select_rows())

    def _insert_rows(self) -> list[dict[str, Any]]:
        if self.table_name in self.client.fail_inserts_for:
            raise RuntimeError(f"forced insert failure for {self.table_name}")
        payload = self.payload
        if payload is None:
            raise AssertionError("insert payload is required")
        rows = payload if isinstance(payload, list) else [payload]
        inserted: list[dict[str, Any]] = []
        for row in rows:
            stored = dict(row)
            stored.setdefault("id", f"{self.table_name}-{len(self._table()) + 1}")
            self._table().append(stored)
            inserted.append(stored)
        return inserted

    def _upsert_rows(self) -> list[dict[str, Any]]:
        if self.payload is None:
            raise AssertionError("upsert payload is required")
        rows = self.payload if isinstance(self.payload, list) else [self.payload]
        return [self._upsert_row(row) for row in rows]

    def _upsert_row(self, payload: dict[str, Any]) -> dict[str, Any]:
        columns = [column.strip() for column in (self.on_conflict or "").split(",")]
        for index, row in enumerate(self._table()):
            if all(row.get(column) == payload.get(column) for column in columns):
                stored = {**row, **payload}
                self._table()[index] = stored
                return stored
        stored = dict(payload)
        stored.setdefault("id", f"{self.table_name}-{len(self._table()) + 1}")
        self._table().append(stored)
        return stored

    def _delete_rows(self) -> list[dict[str, Any]]:
        rows = self._select_rows()
        self.client.tables[self.table_name] = [row for row in self._table() if row not in rows]
        return rows

    def _select_rows(self) -> list[dict[str, Any]]:
        rows = list(self._table())
        for operator, column, value in self.filters:
            if operator == "eq":
                rows = [row for row in rows if row.get(column) == value]
            elif operator == "gte":
                rows = [row for row in rows if row.get(column) >= value]
            elif operator == "lte":
                rows = [row for row in rows if row.get(column) <= value]
            elif operator == "in":
                rows = [row for row in rows if row.get(column) in set(value)]
        for column, desc in reversed(self.orders):
            rows = sorted(rows, key=lambda row: row.get(column), reverse=desc)
        if self.row_limit is not None:
            rows = rows[: self.row_limit]
        if self.row_range is not None:
            start, end = self.row_range
            rows = rows[start : end + 1]
        return rows

    def _table(self) -> list[dict[str, Any]]:
        return self.client.tables.setdefault(self.table_name, [])
