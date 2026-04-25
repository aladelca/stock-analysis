from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContributionSchedule:
    amount: float = 0.0
    frequency_days: int = 30
    start_date: date | None = None


def contributions_for_rebalance_dates(
    rebalance_dates: pd.DatetimeIndex,
    schedule: ContributionSchedule,
) -> dict[pd.Timestamp, float]:
    """Map calendar contribution dates to the next available rebalance date."""

    if schedule.amount <= 0 or len(rebalance_dates) == 0:
        return {pd.Timestamp(rebalance_date): 0.0 for rebalance_date in rebalance_dates}
    if schedule.frequency_days < 1:
        msg = "Contribution frequency must be at least one day"
        raise ValueError(msg)

    sorted_dates = pd.DatetimeIndex(sorted(pd.Timestamp(value) for value in rebalance_dates))
    first_rebalance = sorted_dates[0].date()
    next_deposit = schedule.start_date or (
        first_rebalance + timedelta(days=schedule.frequency_days)
    )
    result = {pd.Timestamp(rebalance_date): 0.0 for rebalance_date in sorted_dates}

    for rebalance_date in sorted_dates:
        current_day = rebalance_date.date()
        while current_day >= next_deposit:
            result[pd.Timestamp(rebalance_date)] += float(schedule.amount)
            next_deposit = next_deposit + timedelta(days=schedule.frequency_days)
    return result


def cumulative_time_weighted_return(returns: list[float]) -> float:
    if not returns:
        return 0.0
    return float(np.prod(1 + np.asarray(returns, dtype=float)) - 1)


def money_weighted_return(cashflows: list[tuple[date, float]]) -> float | None:
    """Return annualized XIRR from dated cash flows, or None when not solvable."""

    valid = [(cashflow_date, float(amount)) for cashflow_date, amount in cashflows if amount != 0]
    if not valid:
        return None
    has_inflow = any(amount > 0 for _, amount in valid)
    has_outflow = any(amount < 0 for _, amount in valid)
    if not has_inflow or not has_outflow:
        return None

    low = -0.999999
    high = 1.0
    low_value = _xnpv(low, valid)
    high_value = _xnpv(high, valid)
    expansion_count = 0
    while low_value * high_value > 0 and expansion_count < 64:
        high *= 2
        high_value = _xnpv(high, valid)
        expansion_count += 1
    if low_value * high_value > 0:
        return None

    for _ in range(128):
        mid = (low + high) / 2
        mid_value = _xnpv(mid, valid)
        if abs(mid_value) < 1e-10:
            return float(mid)
        if low_value * mid_value <= 0:
            high = mid
            high_value = mid_value
        else:
            low = mid
            low_value = mid_value
    result = (low + high) / 2
    return float(result) if isfinite(result) else None


def simulate_benchmark_value_path(
    returns: pd.DataFrame,
    *,
    initial_value: float,
    contribution_by_date: dict[pd.Timestamp, float],
    commission_rate: float,
) -> dict[str, float | None]:
    """Simulate a single-asset benchmark funded by the same external contributions."""

    if returns.empty:
        return {}
    sorted_returns = returns.copy()
    sorted_returns["date"] = pd.to_datetime(sorted_returns["date"])
    sorted_returns = sorted_returns.sort_values("date")
    benchmark_col = _first_existing_column(sorted_returns, ["benchmark_return", "spy_return"])
    if benchmark_col is None:
        return {}

    value = float(initial_value)
    cashflows: list[tuple[date, float]] = []
    total_deposits = 0.0
    total_commissions = 0.0
    twr_returns: list[float] = []

    first_date = sorted_returns["date"].iloc[0].date()
    cashflows.append((first_date, -value))
    total_commissions += value * commission_rate
    value -= value * commission_rate

    for _, row in sorted_returns.iterrows():
        current_date = pd.Timestamp(row["date"])
        contribution = float(contribution_by_date.get(current_date, 0.0))
        if contribution > 0:
            cashflows.append((current_date.date(), -contribution))
            total_deposits += contribution
            commission = contribution * commission_rate
            total_commissions += commission
            value += contribution - commission
        period_return = float(row[benchmark_col])
        value *= 1 + period_return
        twr_returns.append(period_return)

    final_date = sorted_returns["date"].iloc[-1].date()
    cashflows.append((final_date, value))
    total_invested = initial_value + total_deposits
    twr = cumulative_time_weighted_return(twr_returns)
    return {
        "benchmark_ending_value": value,
        "benchmark_total_deposits": total_deposits,
        "benchmark_total_commissions": total_commissions,
        "benchmark_commission_to_deposit_ratio": (
            total_commissions / total_deposits if total_deposits > 0 else None
        ),
        "benchmark_total_return_on_invested_capital": (
            value / total_invested - 1 if total_invested > 0 else None
        ),
        "benchmark_time_weighted_return": twr,
        "benchmark_money_weighted_return": money_weighted_return(cashflows),
    }


def _xnpv(rate: float, cashflows: list[tuple[date, float]]) -> float:
    first_date = cashflows[0][0]
    total = 0.0
    for cashflow_date, amount in cashflows:
        years = (cashflow_date - first_date).days / 365.25
        total += amount / ((1 + rate) ** years)
    return float(total)


def _first_existing_column(frame: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None
