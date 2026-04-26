from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from stock_analysis.portfolio.holdings import PortfolioState
from stock_analysis.storage.contracts import (
    AccountRecord,
    AccountTrackingRepository,
    CashflowRecord,
    PortfolioSnapshotRecord,
)


@dataclass(frozen=True)
class LivePortfolioState:
    account: AccountRecord
    snapshot: PortfolioSnapshotRecord
    state: PortfolioState
    contribution_amount: float
    applied_cashflows: list[CashflowRecord]

    @property
    def net_cashflow_amount(self) -> float:
        return float(sum(cashflow.amount for cashflow in self.applied_cashflows))


def build_live_portfolio_state(
    repository: AccountTrackingRepository,
    account: AccountRecord,
    *,
    as_of_date: date,
) -> LivePortfolioState:
    if account.id is None:
        msg = f"Account row for {account.slug} does not include an id."
        raise ValueError(msg)
    snapshot = repository.latest_portfolio_snapshot(account.id, as_of_date=as_of_date)
    if snapshot is None:
        msg = f"No portfolio snapshot found for account {account.slug} on or before {as_of_date}."
        raise ValueError(msg)
    if snapshot.id is None:
        msg = "Latest portfolio snapshot does not include an id."
        raise ValueError(msg)

    unapplied_cashflows = _unapplied_cashflows_after_snapshot(
        repository,
        account_id=account.id,
        snapshot=snapshot,
        as_of_date=as_of_date,
    )
    net_cashflow = float(sum(cashflow.amount for cashflow in unapplied_cashflows))
    if net_cashflow < 0:
        msg = (
            "Live one-shot does not support negative net cashflows after the latest snapshot. "
            "Import a fresh portfolio snapshot after withdrawals, fees, or taxes."
        )
        raise ValueError(msg)

    holdings = repository.list_holding_snapshots(snapshot.id)
    market_values = pd.Series(
        {holding.ticker: float(holding.market_value) for holding in holdings},
        dtype=float,
        name="market_value",
    ).sort_index()
    weights = (market_values / float(snapshot.total_value)).rename("current_weight")
    weights = weights[weights.gt(0)]

    return LivePortfolioState(
        account=account,
        snapshot=snapshot,
        state=PortfolioState(
            weights=weights,
            market_values=market_values.reindex(weights.index).fillna(0.0),
            cash_balance=float(snapshot.cash_balance),
            portfolio_value=float(snapshot.total_value),
        ),
        contribution_amount=net_cashflow,
        applied_cashflows=unapplied_cashflows,
    )


def _unapplied_cashflows_after_snapshot(
    repository: AccountTrackingRepository,
    *,
    account_id: str,
    snapshot: PortfolioSnapshotRecord,
    as_of_date: date,
) -> list[CashflowRecord]:
    start_date = snapshot.snapshot_date + timedelta(days=1)
    cashflows = repository.list_cashflows(
        account_id,
        start_date=start_date,
        end_date=as_of_date,
    )
    return [
        cashflow
        for cashflow in cashflows
        if cashflow.included_in_snapshot_id is None and _is_settled(cashflow, as_of_date)
    ]


def _is_settled(cashflow: CashflowRecord, as_of_date: date) -> bool:
    settlement_date = cashflow.settled_date or cashflow.cashflow_date
    return settlement_date <= as_of_date
