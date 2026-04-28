from __future__ import annotations

from datetime import date

import pytest

from stock_analysis.portfolio.live_state import build_live_portfolio_state
from stock_analysis.storage.contracts import (
    AccountRecord,
    CashflowRecord,
    HoldingSnapshotRecord,
    PerformanceSnapshotRecord,
    PortfolioSnapshotRecord,
    RecommendationLineRecord,
    RecommendationRunRecord,
)


def test_live_state_applies_settled_cashflows_after_latest_snapshot() -> None:
    repository = FakeLiveRepository(
        cashflows=[
            CashflowRecord(
                account_id="account-1",
                cashflow_date=date(2026, 4, 24),
                settled_date=date(2026, 4, 24),
                amount=250.0,
                cashflow_type="deposit",
            ),
            CashflowRecord(
                account_id="account-1",
                cashflow_date=date(2026, 4, 25),
                settled_date=date(2026, 4, 27),
                amount=100.0,
                cashflow_type="deposit",
            ),
        ]
    )

    live_state = build_live_portfolio_state(
        repository,
        AccountRecord(id="account-1", slug="main", display_name="Main"),
        as_of_date=date(2026, 4, 25),
    )

    assert live_state.contribution_amount == 250.0
    assert live_state.net_cashflow_amount == 250.0
    assert live_state.state.portfolio_value == 1000.0
    assert live_state.state.cash_balance == 100.0
    assert live_state.state.market_values.to_dict() == {"AAPL": 900.0}
    assert live_state.state.weights.to_dict() == {"AAPL": 0.9}


def test_live_state_does_not_apply_same_day_cashflow_against_latest_snapshot() -> None:
    repository = FakeLiveRepository(
        cashflows=[
            CashflowRecord(
                account_id="account-1",
                cashflow_date=date(2026, 4, 28),
                amount=300.0,
                cashflow_type="deposit",
            ),
            CashflowRecord(
                account_id="account-1",
                cashflow_date=date(2026, 4, 28),
                amount=150.0,
                cashflow_type="deposit",
                included_in_snapshot_id="snapshot-1",
            ),
        ]
    )
    repository.snapshot = PortfolioSnapshotRecord(
        id="snapshot-1",
        account_id="account-1",
        snapshot_date=date(2026, 4, 28),
        market_value=900.0,
        cash_balance=100.0,
        total_value=1000.0,
    )

    live_state = build_live_portfolio_state(
        repository,
        AccountRecord(id="account-1", slug="main", display_name="Main"),
        as_of_date=date(2026, 4, 28),
    )

    assert live_state.contribution_amount == 0.0
    assert live_state.applied_cashflows == []


def test_live_state_requires_fresh_snapshot_after_negative_net_cashflow() -> None:
    repository = FakeLiveRepository(
        cashflows=[
            CashflowRecord(
                account_id="account-1",
                cashflow_date=date(2026, 4, 24),
                amount=-50.0,
                cashflow_type="fee",
            )
        ]
    )

    with pytest.raises(ValueError, match="negative net cashflows"):
        build_live_portfolio_state(
            repository,
            AccountRecord(id="account-1", slug="main", display_name="Main"),
            as_of_date=date(2026, 4, 25),
        )


def test_live_state_requires_holdings_for_positive_market_value_snapshot() -> None:
    repository = FakeLiveRepository(cashflows=[])
    repository.holdings = []

    with pytest.raises(ValueError, match="holding rows"):
        build_live_portfolio_state(
            repository,
            AccountRecord(id="account-1", slug="main", display_name="Main"),
            as_of_date=date(2026, 4, 25),
        )


class FakeLiveRepository:
    def __init__(self, *, cashflows: list[CashflowRecord]) -> None:
        self.cashflows = cashflows
        self.snapshot = PortfolioSnapshotRecord(
            id="snapshot-1",
            account_id="account-1",
            snapshot_date=date(2026, 4, 23),
            market_value=900.0,
            cash_balance=100.0,
            total_value=1000.0,
        )
        self.holdings = [
            HoldingSnapshotRecord(
                snapshot_id="snapshot-1",
                ticker="AAPL",
                market_value=900.0,
            )
        ]

    def get_account_by_slug(self, slug: str) -> AccountRecord | None:
        return AccountRecord(id="account-1", slug=slug, display_name="Main")

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
