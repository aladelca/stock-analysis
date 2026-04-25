from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_analysis.portfolio.holdings import PortfolioState, align_current_weights


@dataclass(frozen=True)
class RebalanceContext:
    current_weights: pd.Series
    current_market_values: pd.Series
    portfolio_value_before_contribution: float
    contribution_amount: float
    portfolio_value_after_contribution: float
    cash_before_rebalance: float


def build_rebalance_context(
    state: PortfolioState,
    tickers: pd.Index | list[str],
    *,
    contribution_amount: float = 0.0,
    portfolio_value: float | None = None,
) -> RebalanceContext:
    """Build the post-deposit state used by optimization and recommendation logic."""

    if contribution_amount < 0:
        msg = "Contribution amount cannot be negative"
        raise ValueError(msg)

    ticker_index = pd.Index([str(ticker) for ticker in tickers], name="ticker")
    before_value = _resolve_before_value(state, portfolio_value)
    after_value = before_value + float(contribution_amount)
    if after_value <= 0:
        msg = "Portfolio value after contribution must be positive"
        raise ValueError(msg)

    current_values = _current_market_values(state, ticker_index, before_value)
    current_weights = (current_values / after_value).rename("current_weight")
    cash_before_rebalance = max(after_value - float(current_values.sum()), 0.0)

    return RebalanceContext(
        current_weights=current_weights,
        current_market_values=current_values.rename("current_market_value"),
        portfolio_value_before_contribution=before_value,
        contribution_amount=float(contribution_amount),
        portfolio_value_after_contribution=after_value,
        cash_before_rebalance=cash_before_rebalance,
    )


def plan_rebalance_trades(
    target_weights: pd.Series,
    context: RebalanceContext,
    *,
    commission_rate: float,
    min_trade_weight: float,
    no_trade_band: float = 0.0,
) -> pd.DataFrame:
    tickers = pd.Index(target_weights.index.astype(str), name="ticker")
    current_weights = context.current_weights.reindex(tickers).fillna(0.0)
    target_weights = target_weights.reindex(tickers).fillna(0.0).astype(float)
    trade_weights = target_weights - current_weights
    threshold = max(float(min_trade_weight), float(no_trade_band))
    planned = trade_weights.abs().ge(threshold)
    trade_notional = np.where(
        planned,
        trade_weights.to_numpy(dtype=float) * context.portfolio_value_after_contribution,
        0.0,
    )
    commission_amount = np.abs(trade_notional) * float(commission_rate)
    required = np.where(trade_notional > 0, trade_notional + commission_amount, 0.0)
    released = np.where(trade_notional < 0, np.abs(trade_notional) - commission_amount, 0.0)
    cash_after = context.cash_before_rebalance + float(np.sum(released)) - float(np.sum(required))
    deposit_used = _allocate_deposit_to_buys(
        required,
        available_cash=context.cash_before_rebalance,
        contribution_amount=context.contribution_amount,
    )

    return pd.DataFrame(
        {
            "ticker": tickers,
            "current_market_value": current_weights.to_numpy(dtype=float)
            * context.portfolio_value_after_contribution,
            "target_market_value": target_weights.to_numpy(dtype=float)
            * context.portfolio_value_after_contribution,
            "trade_notional": trade_notional,
            "commission_amount": commission_amount,
            "deposit_used_amount": deposit_used,
            "cash_after_trade_amount": cash_after,
            "no_trade_band_applied": trade_weights.abs().lt(threshold)
            & trade_weights.abs().gt(float(min_trade_weight)),
            "portfolio_value_before_contribution": context.portfolio_value_before_contribution,
            "contribution_amount": context.contribution_amount,
            "portfolio_value_after_contribution": context.portfolio_value_after_contribution,
        }
    )


def _resolve_before_value(state: PortfolioState, portfolio_value: float | None) -> float:
    if portfolio_value is not None:
        if portfolio_value <= 0:
            msg = "Portfolio value must be positive when provided"
            raise ValueError(msg)
        return float(portfolio_value)
    resolved = state.resolved_portfolio_value
    if resolved is not None:
        return float(resolved)
    return 1.0


def _current_market_values(
    state: PortfolioState,
    tickers: pd.Index,
    portfolio_value: float,
) -> pd.Series:
    if not state.market_values.empty:
        values = state.market_values.copy()
        values.index = values.index.astype(str)
        return values.reindex(tickers).fillna(0.0).astype(float)
    weights = align_current_weights(state.weights, tickers)
    return (weights * portfolio_value).astype(float).rename("current_market_value")


def _allocate_deposit_to_buys(
    required: np.ndarray,
    *,
    available_cash: float,
    contribution_amount: float,
) -> np.ndarray:
    result = np.zeros(len(required), dtype=float)
    if contribution_amount <= 0:
        return result
    buy_required = np.maximum(required, 0.0)
    total_buy_required = float(buy_required.sum())
    if total_buy_required <= 0:
        return result
    deposit_available_for_buys = min(float(contribution_amount), max(float(available_cash), 0.0))
    if deposit_available_for_buys <= 0:
        return result
    return buy_required / total_buy_required * min(deposit_available_for_buys, total_buy_required)
