from __future__ import annotations

from datetime import date
from typing import Any, cast

import pandas as pd

from stock_analysis.backtest.cashflows import cumulative_time_weighted_return, money_weighted_return
from stock_analysis.portfolio.live_state import LivePortfolioState
from stock_analysis.storage.contracts import CashflowRecord, PortfolioSnapshotRecord


def build_account_tracking_marts(
    *,
    live_state: LivePortfolioState,
    recommendations: pd.DataFrame,
    run_metadata: pd.DataFrame,
    spy_daily: pd.DataFrame,
    commission_rate: float,
) -> dict[str, pd.DataFrame]:
    run_context = _run_context(run_metadata)
    cashflows = _cashflows_table(live_state, run_context)
    snapshots = _portfolio_snapshots_table(live_state, run_context)
    holdings = _holding_snapshots_table(live_state, run_context)
    recommendation_runs = _recommendation_runs_table(live_state, run_context)
    recommendation_lines = _recommendation_lines_table(live_state, recommendations, run_context)
    performance = _performance_snapshots_table(
        live_state,
        spy_daily=spy_daily,
        commission_rate=commission_rate,
        run_context=run_context,
    )
    return {
        "cashflows": cashflows,
        "portfolio_snapshots": snapshots,
        "holding_snapshots": holdings,
        "recommendation_runs": recommendation_runs,
        "recommendation_lines": recommendation_lines,
        "performance_snapshots": performance,
    }


def latest_performance_fields(performance_snapshots: pd.DataFrame) -> dict[str, object]:
    if performance_snapshots.empty:
        return {}
    sorted_frame = performance_snapshots.copy()
    sorted_frame["as_of_date"] = pd.to_datetime(sorted_frame["as_of_date"], errors="coerce")
    row = sorted_frame.sort_values("as_of_date").iloc[-1].to_dict()
    return {
        "account_total_value": row.get("account_total_value"),
        "account_initial_value": row.get("initial_value"),
        "account_total_deposits": row.get("total_deposits"),
        "account_invested_capital": row.get("invested_capital"),
        "account_net_external_cashflow": row.get("net_external_cashflow"),
        "account_return_on_invested_capital": row.get("return_on_invested_capital"),
        "account_time_weighted_return": row.get("account_time_weighted_return"),
        "account_money_weighted_return": row.get("account_money_weighted_return"),
        "spy_same_cashflow_value": row.get("spy_same_cashflow_value"),
        "spy_time_weighted_return": row.get("spy_time_weighted_return"),
        "spy_money_weighted_return": row.get("spy_money_weighted_return"),
        "active_value": row.get("active_value"),
        "active_return": row.get("active_return"),
    }


def _run_context(run_metadata: pd.DataFrame) -> dict[str, object]:
    if run_metadata.empty:
        return {}
    row = run_metadata.iloc[0].to_dict()
    return {
        "run_id": row.get("run_id"),
        "run_as_of_date": row.get("data_as_of_date") or row.get("as_of_date"),
        "run_requested_as_of_date": row.get("requested_as_of_date"),
        "model_version": row.get("model_version"),
        "expected_return_is_calibrated": row.get("expected_return_is_calibrated"),
        "config_hash": row.get("config_hash"),
        "created_at_utc": row.get("created_at_utc"),
    }


def _cashflows_table(
    live_state: LivePortfolioState,
    run_context: dict[str, object],
) -> pd.DataFrame:
    rows = [
        {
            **run_context,
            "account_id": live_state.account.id,
            "account_slug": live_state.account.slug,
            "cashflow_id": cashflow.id,
            "cashflow_date": cashflow.cashflow_date.isoformat(),
            "settled_date": (
                cashflow.settled_date.isoformat() if cashflow.settled_date is not None else ""
            ),
            "amount": cashflow.amount,
            "currency": cashflow.currency,
            "cashflow_type": cashflow.cashflow_type,
            "source": cashflow.source,
            "external_ref": cashflow.external_ref or "",
            "notes": cashflow.notes or "",
            "included_in_snapshot_id": cashflow.included_in_snapshot_id or "",
            "is_applied_to_recommendation": cashflow in live_state.applied_cashflows,
        }
        for cashflow in live_state.cashflows
    ]
    return _frame(rows, _cashflow_columns())


def _portfolio_snapshots_table(
    live_state: LivePortfolioState,
    run_context: dict[str, object],
) -> pd.DataFrame:
    rows = [
        {
            **run_context,
            "account_id": live_state.account.id,
            "account_slug": live_state.account.slug,
            "snapshot_id": snapshot.id,
            "snapshot_date": snapshot.snapshot_date.isoformat(),
            "market_value": snapshot.market_value,
            "cash_balance": snapshot.cash_balance,
            "total_value": snapshot.total_value,
            "currency": snapshot.currency,
            "source": snapshot.source,
            "is_latest_for_run": snapshot.id == live_state.snapshot.id,
            "run_unapplied_cashflow_amount": live_state.net_cashflow_amount,
            "run_projected_total_value": (
                snapshot.total_value + live_state.net_cashflow_amount
                if snapshot.id == live_state.snapshot.id
                else snapshot.total_value
            ),
        }
        for snapshot in live_state.snapshots
    ]
    return _frame(rows, _portfolio_snapshot_columns())


def _holding_snapshots_table(
    live_state: LivePortfolioState,
    run_context: dict[str, object],
) -> pd.DataFrame:
    rows = [
        {
            **run_context,
            "account_id": live_state.account.id,
            "account_slug": live_state.account.slug,
            "snapshot_id": holding.snapshot_id,
            "ticker": holding.ticker,
            "quantity": holding.quantity,
            "market_value": holding.market_value,
            "price": holding.price,
            "currency": holding.currency,
            "is_latest_for_run": holding.snapshot_id == live_state.snapshot.id,
        }
        for holding in live_state.holdings
    ]
    return _frame(rows, _holding_snapshot_columns())


def _recommendation_runs_table(
    live_state: LivePortfolioState,
    run_context: dict[str, object],
) -> pd.DataFrame:
    return _frame(
        [
            {
                **run_context,
                "account_id": live_state.account.id,
                "account_slug": live_state.account.slug,
                "benchmark_ticker": live_state.account.benchmark_ticker,
                "base_currency": live_state.account.base_currency,
                "snapshot_id": live_state.snapshot.id,
                "snapshot_date": live_state.snapshot.snapshot_date.isoformat(),
                "unapplied_cashflow_amount": live_state.net_cashflow_amount,
                "unapplied_cashflow_count": len(live_state.applied_cashflows),
            }
        ],
        _recommendation_run_columns(),
    )


def _recommendation_lines_table(
    live_state: LivePortfolioState,
    recommendations: pd.DataFrame,
    run_context: dict[str, object],
) -> pd.DataFrame:
    if recommendations.empty:
        return _frame([], _recommendation_line_columns())
    lines = recommendations.copy()
    for column, value in run_context.items():
        lines[column] = [value] * len(lines)  # type: ignore[assignment]
    lines["account_id"] = live_state.account.id
    lines["account_slug"] = live_state.account.slug
    lines["recommendation_key"] = lines["run_id"].astype(str) + ":" + lines["ticker"].astype(str)
    for column in _recommendation_line_columns():
        if column not in lines.columns:
            lines[column] = pd.NA
    return lines[_recommendation_line_columns()].reset_index(drop=True)


def _performance_snapshots_table(
    live_state: LivePortfolioState,
    *,
    spy_daily: pd.DataFrame,
    commission_rate: float,
    run_context: dict[str, object],
) -> pd.DataFrame:
    snapshots = sorted(live_state.snapshots, key=lambda snapshot: snapshot.snapshot_date)
    if not snapshots:
        return _frame([], _performance_snapshot_columns())

    first_snapshot = snapshots[0]
    twr_returns: list[float] = []
    rows: list[dict[str, object]] = []
    previous_snapshot = first_snapshot
    for index, snapshot in enumerate(snapshots):
        if index > 0:
            period_cashflow = _net_cashflow_between(
                live_state.cashflows,
                previous_snapshot.snapshot_date,
                snapshot.snapshot_date,
            )
            if previous_snapshot.total_value > 0:
                twr_returns.append(
                    float((snapshot.total_value - period_cashflow) / previous_snapshot.total_value)
                    - 1
                )
            previous_snapshot = snapshot
        cashflows_to_date = [
            cashflow
            for cashflow in live_state.cashflows
            if first_snapshot.snapshot_date < cashflow.cashflow_date <= snapshot.snapshot_date
        ]
        total_deposits = _total_deposits(cashflows_to_date)
        invested_capital = float(first_snapshot.total_value) + total_deposits
        account_mwr_cashflows = [
            (first_snapshot.snapshot_date, -float(first_snapshot.total_value)),
            *[(cashflow.cashflow_date, -float(cashflow.amount)) for cashflow in cashflows_to_date],
            (snapshot.snapshot_date, float(snapshot.total_value)),
        ]
        benchmark = _same_cashflow_spy_benchmark(
            spy_daily,
            first_snapshot=first_snapshot,
            as_of_date=snapshot.snapshot_date,
            cashflows=cashflows_to_date,
            commission_rate=commission_rate,
        )
        active_value = _safe_difference(
            snapshot.total_value,
            benchmark.get("spy_same_cashflow_value"),
        )
        spy_value = benchmark.get("spy_same_cashflow_value")
        active_return = float(snapshot.total_value / spy_value - 1) if spy_value else None
        rows.append(
            {
                **run_context,
                "account_id": live_state.account.id,
                "account_slug": live_state.account.slug,
                "as_of_date": snapshot.snapshot_date.isoformat(),
                "source_snapshot_id": snapshot.id,
                "account_total_value": snapshot.total_value,
                "initial_value": first_snapshot.total_value,
                "total_deposits": total_deposits,
                "invested_capital": invested_capital,
                "net_external_cashflow": sum(cashflow.amount for cashflow in cashflows_to_date),
                "return_on_invested_capital": (
                    float(snapshot.total_value / invested_capital - 1)
                    if invested_capital > 0
                    else None
                ),
                "account_time_weighted_return": cumulative_time_weighted_return(twr_returns),
                "account_money_weighted_return": money_weighted_return(account_mwr_cashflows),
                "spy_same_cashflow_value": benchmark.get("spy_same_cashflow_value"),
                "spy_time_weighted_return": benchmark.get("spy_time_weighted_return"),
                "spy_money_weighted_return": benchmark.get("spy_money_weighted_return"),
                "active_value": active_value,
                "active_return": active_return,
            }
        )
    return _frame(rows, _performance_snapshot_columns())


def _same_cashflow_spy_benchmark(
    spy_daily: pd.DataFrame,
    *,
    first_snapshot: PortfolioSnapshotRecord,
    as_of_date: date,
    cashflows: list[CashflowRecord],
    commission_rate: float,
) -> dict[str, float | None]:
    value = float(first_snapshot.total_value)
    if as_of_date <= first_snapshot.snapshot_date:
        return {
            "spy_same_cashflow_value": value,
            "spy_time_weighted_return": 0.0,
            "spy_money_weighted_return": None,
        }
    spy = spy_daily.copy()
    if spy.empty or "return_1d" not in spy.columns:
        return {}
    spy["date"] = pd.to_datetime(spy["date"], errors="coerce")
    spy["return_1d"] = pd.to_numeric(spy["return_1d"], errors="coerce").fillna(0.0)
    mask = (spy["date"].dt.date > first_snapshot.snapshot_date) & (
        spy["date"].dt.date <= as_of_date
    )
    spy = spy.loc[mask].sort_values("date")
    cashflow_by_date = _cashflow_amount_by_trading_date(
        cashflows,
        trading_dates=[pd.Timestamp(value).date() for value in spy["date"]],
        as_of_date=as_of_date,
    )
    investor_cashflows = [(first_snapshot.snapshot_date, -value)]
    twr_returns: list[float] = []
    for _, row in spy.iterrows():
        current_date = pd.Timestamp(row["date"]).date()
        cashflow_amount = cashflow_by_date.get(current_date, 0.0)
        if cashflow_amount:
            investor_cashflows.append((current_date, -cashflow_amount))
            value += cashflow_amount
            if cashflow_amount > 0:
                value -= cashflow_amount * float(commission_rate)
        period_base = value
        period_return = float(row["return_1d"])
        value *= 1 + period_return
        if period_base > 0:
            twr_returns.append(value / period_base - 1)
    investor_cashflows.append((as_of_date, value))
    return {
        "spy_same_cashflow_value": value,
        "spy_time_weighted_return": cumulative_time_weighted_return(twr_returns),
        "spy_money_weighted_return": money_weighted_return(investor_cashflows),
    }


def _net_cashflow_between(
    cashflows: list[CashflowRecord],
    start_date: date,
    end_date: date,
) -> float:
    return float(
        sum(
            cashflow.amount
            for cashflow in cashflows
            if start_date < cashflow.cashflow_date <= end_date
        )
    )


def _cashflow_amount_by_trading_date(
    cashflows: list[CashflowRecord],
    *,
    trading_dates: list[date],
    as_of_date: date,
) -> dict[date, float]:
    result: dict[date, float] = {}
    sorted_trading_dates = sorted(set(trading_dates))
    for cashflow in cashflows:
        mapped_date = _next_trading_date(cashflow.cashflow_date, sorted_trading_dates)
        if mapped_date is None or mapped_date > as_of_date:
            continue
        result[mapped_date] = result.get(mapped_date, 0.0) + float(cashflow.amount)
    return result


def _next_trading_date(cashflow_date: date, trading_dates: list[date]) -> date | None:
    for trading_date in trading_dates:
        if trading_date >= cashflow_date:
            return trading_date
    return None


def _total_deposits(cashflows: list[CashflowRecord]) -> float:
    return float(
        sum(
            cashflow.amount
            for cashflow in cashflows
            if cashflow.cashflow_type == "deposit" and cashflow.amount > 0
        )
    )


def _safe_difference(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return float(cast(Any, left)) - float(cast(Any, right))


def _frame(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def _cashflow_columns() -> list[str]:
    return [
        "run_id",
        "run_as_of_date",
        "account_id",
        "account_slug",
        "cashflow_id",
        "cashflow_date",
        "settled_date",
        "amount",
        "currency",
        "cashflow_type",
        "source",
        "external_ref",
        "notes",
        "included_in_snapshot_id",
        "is_applied_to_recommendation",
    ]


def _portfolio_snapshot_columns() -> list[str]:
    return [
        "run_id",
        "run_as_of_date",
        "account_id",
        "account_slug",
        "snapshot_id",
        "snapshot_date",
        "market_value",
        "cash_balance",
        "total_value",
        "currency",
        "source",
        "is_latest_for_run",
        "run_unapplied_cashflow_amount",
        "run_projected_total_value",
    ]


def _holding_snapshot_columns() -> list[str]:
    return [
        "run_id",
        "run_as_of_date",
        "account_id",
        "account_slug",
        "snapshot_id",
        "ticker",
        "quantity",
        "market_value",
        "price",
        "currency",
        "is_latest_for_run",
    ]


def _recommendation_run_columns() -> list[str]:
    return [
        "run_id",
        "run_as_of_date",
        "run_requested_as_of_date",
        "account_id",
        "account_slug",
        "benchmark_ticker",
        "base_currency",
        "model_version",
        "config_hash",
        "expected_return_is_calibrated",
        "created_at_utc",
        "snapshot_id",
        "snapshot_date",
        "unapplied_cashflow_amount",
        "unapplied_cashflow_count",
    ]


def _recommendation_line_columns() -> list[str]:
    return [
        "recommendation_key",
        "run_id",
        "run_as_of_date",
        "account_id",
        "account_slug",
        "ticker",
        "security",
        "gics_sector",
        "forecast_score",
        "expected_return",
        "calibrated_expected_return",
        "expected_return_is_calibrated",
        "volatility",
        "forecast_horizon_days",
        "forecast_start_date",
        "forecast_end_date",
        "realized_return",
        "realized_spy_return",
        "realized_active_return",
        "forecast_error",
        "forecast_hit",
        "outcome_status",
        "current_weight",
        "target_weight",
        "trade_weight",
        "trade_abs_weight",
        "trade_notional",
        "commission_amount",
        "deposit_used_amount",
        "cash_after_trade_amount",
        "action",
        "reason_code",
    ]


def _performance_snapshot_columns() -> list[str]:
    return [
        "run_id",
        "run_as_of_date",
        "account_id",
        "account_slug",
        "as_of_date",
        "source_snapshot_id",
        "account_total_value",
        "initial_value",
        "total_deposits",
        "invested_capital",
        "net_external_cashflow",
        "return_on_invested_capital",
        "account_time_weighted_return",
        "account_money_weighted_return",
        "spy_same_cashflow_value",
        "spy_time_weighted_return",
        "spy_money_weighted_return",
        "active_value",
        "active_return",
    ]
