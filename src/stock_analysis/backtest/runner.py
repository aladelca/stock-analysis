from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

from stock_analysis.backtest.cashflows import (
    ContributionSchedule,
    contributions_for_rebalance_dates,
    cumulative_time_weighted_return,
    money_weighted_return,
)
from stock_analysis.config import OptimizerConfig
from stock_analysis.optimization.engine import optimize_long_only


class PredictiveModel(Protocol):
    def predict(self, features: pd.DataFrame) -> Sequence[float]:
        """Return forecast scores for the provided feature cross-section."""


@dataclass(frozen=True)
class BacktestConfig:
    horizon_days: int = 5
    training_target_column: str | None = None
    rebalance_step_days: int = 5
    embargo_days: int = 15
    commission_rate: float | None = None
    cost_bps: float = 5.0
    covariance_lookback_days: int = 252
    feature_columns: tuple[str, ...] = ()
    max_rebalances: int | None = None
    max_assets_per_rebalance: int | None = None
    liquidity_column: str = "dollar_volume_21d"
    initial_portfolio_value: float = 1000.0
    monthly_deposit_amount: float = 0.0
    deposit_frequency_days: int = 30
    deposit_start_date: date | None = None
    rebalance_on_deposit_day: bool = True
    no_trade_band: float = 0.0


def run_walk_forward_backtest(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    returns: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], PredictiveModel],
    optimizer_config: OptimizerConfig,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run a walk-forward portfolio backtest in long format, one row per ticker/rebalance."""

    cfg = config or BacktestConfig()
    feature_panel = _prepare_dates(panel)
    label_panel = _prepare_dates(labels)
    returns_frame = _prepare_dates(returns)
    realized_target_col = f"fwd_return_{cfg.horizon_days}d"
    training_target_col = cfg.training_target_column or realized_target_col
    label_columns = ["ticker", "date", realized_target_col]
    if training_target_col != realized_target_col:
        label_columns.append(training_target_col)
    missing_label_columns = [
        column for column in label_columns if column not in label_panel.columns
    ]
    if missing_label_columns:
        msg = f"labels are missing required target columns: {missing_label_columns}"
        raise ValueError(msg)

    merged_panel = feature_panel.merge(
        label_panel[label_columns],
        on=["ticker", "date"],
        how="left",
    )
    rebalance_dates = pd.DatetimeIndex(sorted(merged_panel["date"].dropna().unique()))
    records: list[dict[str, object]] = []
    previous_weights: pd.Series | None = None
    completed_rebalances = 0

    rebalance_step = max(cfg.rebalance_step_days, 1)
    contribution_schedule = ContributionSchedule(
        amount=cfg.monthly_deposit_amount,
        frequency_days=cfg.deposit_frequency_days,
        start_date=cfg.deposit_start_date,
    )
    candidate_rebalance_dates = rebalance_dates[::rebalance_step]
    if cfg.rebalance_on_deposit_day:
        deposit_rebalance_dates = _deposit_rebalance_dates(
            rebalance_dates,
            contribution_schedule,
        )
        if len(deposit_rebalance_dates) > 0:
            candidate_rebalance_dates = pd.DatetimeIndex(
                sorted(
                    {
                        *[pd.Timestamp(value) for value in candidate_rebalance_dates],
                        *[pd.Timestamp(value) for value in deposit_rebalance_dates],
                    }
                )
            )
    if cfg.max_rebalances is not None and len(candidate_rebalance_dates) > cfg.max_rebalances:
        sample_positions = np.linspace(
            0,
            len(candidate_rebalance_dates) - 1,
            cfg.max_rebalances,
        ).round()
        candidate_rebalance_dates = candidate_rebalance_dates[
            np.unique(sample_positions.astype(int))
        ]

    contribution_by_date = contributions_for_rebalance_dates(
        candidate_rebalance_dates,
        contribution_schedule,
    )
    portfolio_value = float(cfg.initial_portfolio_value)
    cashflows: list[tuple[date, float]] = []
    twr_returns: list[float] = []
    total_deposits = 0.0
    total_commissions = 0.0
    first_rebalance_date: date | None = None

    for rebalance_date in candidate_rebalance_dates:
        if cfg.max_rebalances is not None and completed_rebalances >= cfg.max_rebalances:
            break

        train_cutoff = rebalance_date - pd.offsets.BDay(cfg.embargo_days)
        train_df = merged_panel.loc[
            (merged_panel["date"] < train_cutoff) & merged_panel[training_target_col].notna()
        ].copy()
        features_at_rebalance = merged_panel.loc[merged_panel["date"] == rebalance_date].copy()
        features_at_rebalance = features_at_rebalance.dropna(subset=[realized_target_col])
        features_at_rebalance = _limit_features_by_liquidity(features_at_rebalance, cfg)
        if train_df.empty or features_at_rebalance.empty:
            continue

        model = predict_fn(train_df)
        scores = np.asarray(
            model.predict(_select_features(features_at_rebalance, cfg)), dtype=float
        )
        if len(scores) != len(features_at_rebalance):
            msg = "predict_fn model returned a score count that does not match rebalance features"
            raise ValueError(msg)
        features_at_rebalance["forecast_score"] = scores

        optimizer_input = _optimizer_input_from_scores(features_at_rebalance)
        covariance = _estimate_covariance(
            returns_frame,
            optimizer_input["ticker"].tolist(),
            rebalance_date,
            cfg.covariance_lookback_days,
        )
        contribution = float(contribution_by_date.get(pd.Timestamp(rebalance_date), 0.0))
        value_before_contribution = portfolio_value
        value_after_contribution = value_before_contribution + contribution
        if value_after_contribution <= 0:
            msg = "Portfolio value after contribution must be positive"
            raise ValueError(msg)
        optimizer_previous_weights = _post_deposit_previous_weights(
            previous_weights,
            optimizer_input["ticker"].tolist(),
            value_before_contribution=value_before_contribution,
            value_after_contribution=value_after_contribution,
        )
        weights = optimize_long_only(
            optimizer_input,
            covariance,
            optimizer_config,
            w_prev=optimizer_previous_weights,
        )
        aligned_previous = optimizer_previous_weights.reindex(weights.index).fillna(0.0)
        trade_delta = weights - aligned_previous
        executed_trade_delta = _trade_delta_after_no_trade_band(
            trade_delta,
            cfg.no_trade_band,
        )
        executed_weights = (aligned_previous + executed_trade_delta).clip(lower=0.0)
        trade_abs_weight = float(np.abs(executed_trade_delta).sum())
        raw_trade_abs_weight = float(np.abs(trade_delta).sum())
        turnover = 0.5 * trade_abs_weight
        commission_rate = (
            optimizer_config.commission_rate if cfg.commission_rate is None else cfg.commission_rate
        )
        period_cost = (
            commission_rate * trade_abs_weight
            if commission_rate > 0
            else cfg.cost_bps / 10_000 * turnover
        )
        commission_paid = period_cost * value_after_contribution

        realized = features_at_rebalance.set_index("ticker")[realized_target_col].astype(float)
        gross_return = float((executed_weights * realized.reindex(weights.index).fillna(0)).sum())
        net_return = gross_return - period_cost
        portfolio_value_end = value_after_contribution * (1 + net_return)
        buy_notional = float(executed_trade_delta.clip(lower=0).sum() * value_after_contribution)
        sell_notional = float(
            executed_trade_delta.clip(upper=0).abs().sum() * value_after_contribution
        )
        cash_before_rebalance = max(
            value_after_contribution - float(aligned_previous.sum() * value_after_contribution),
            0.0,
        )
        cash_after_rebalance = (
            cash_before_rebalance + sell_notional - buy_notional - commission_paid
        )

        if first_rebalance_date is None:
            first_rebalance_date = rebalance_date.date()
            cashflows.append((first_rebalance_date, -float(cfg.initial_portfolio_value)))
        if contribution > 0:
            cashflows.append((rebalance_date.date(), -contribution))
            total_deposits += contribution
        total_commissions += commission_paid
        twr_returns.append(net_return)
        cumulative_twr = cumulative_time_weighted_return(twr_returns)

        predictions = features_at_rebalance.set_index("ticker")["forecast_score"]
        for ticker, weight in weights.items():
            records.append(
                {
                    "rebalance_date": rebalance_date.date().isoformat(),
                    "ticker": ticker,
                    "target_weight": float(weight),
                    "executed_weight": float(executed_weights.get(ticker, 0.0)),
                    "previous_weight": float(aligned_previous.get(ticker, 0.0)),
                    "forecast_score": float(predictions.get(ticker, np.nan)),
                    "realized_return": float(realized.get(ticker, np.nan)),
                    "portfolio_gross_return": gross_return,
                    "transaction_cost": period_cost,
                    "portfolio_net_return": net_return,
                    "turnover": turnover,
                    "trade_abs_weight": trade_abs_weight,
                    "raw_trade_abs_weight": raw_trade_abs_weight,
                    "commission_rate": commission_rate,
                    "portfolio_value_start": value_before_contribution,
                    "external_contribution": contribution,
                    "portfolio_value_after_contribution": value_after_contribution,
                    "cash_before_rebalance": cash_before_rebalance,
                    "buy_notional": buy_notional,
                    "sell_notional": sell_notional,
                    "commission_paid": commission_paid,
                    "cash_after_rebalance": cash_after_rebalance,
                    "portfolio_value_end": portfolio_value_end,
                    "period_twr_return": net_return,
                    "cumulative_twr_return": cumulative_twr,
                }
            )
        previous_weights = executed_weights
        portfolio_value = portfolio_value_end
        completed_rebalances += 1

    result = pd.DataFrame.from_records(records)
    if result.empty or first_rebalance_date is None:
        return result
    final_date = pd.to_datetime(result["rebalance_date"]).max().date()
    cashflows.append((final_date, portfolio_value))
    return _attach_cashflow_summary(
        result,
        initial_portfolio_value=float(cfg.initial_portfolio_value),
        ending_value=portfolio_value,
        total_deposits=total_deposits,
        total_commissions=total_commissions,
        money_weighted=money_weighted_return(cashflows),
    )


def _prepare_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    return result


def _select_features(frame: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    id_columns = [column for column in ("ticker", "date") if column in frame.columns]
    if config.feature_columns:
        feature_columns = [column for column in config.feature_columns if column not in id_columns]
        return frame.loc[:, [*id_columns, *feature_columns]]
    excluded = {
        "ticker",
        "date",
        "security",
        "gics_sector",
        "gics_sub_industry",
    }
    numeric_columns = [
        column
        for column in frame.columns
        if column not in excluded
        and not column.startswith("fwd_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return frame[[*id_columns, *numeric_columns]]


def _limit_features_by_liquidity(frame: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if config.max_assets_per_rebalance is None or len(frame) <= config.max_assets_per_rebalance:
        return frame
    if config.liquidity_column not in frame.columns:
        msg = (
            "cannot apply max_assets_per_rebalance; "
            f"missing liquidity column {config.liquidity_column}"
        )
        raise ValueError(msg)
    ranked = frame.assign(
        _liquidity=pd.to_numeric(frame[config.liquidity_column], errors="coerce")
    ).sort_values(
        ["_liquidity", "ticker"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ranked.head(config.max_assets_per_rebalance).drop(columns=["_liquidity"])


def _optimizer_input_from_scores(features: pd.DataFrame) -> pd.DataFrame:
    volatility_col = _first_matching_column(features, "volatility_")
    result = pd.DataFrame(
        {
            "ticker": features["ticker"],
            "security": features.get("security", features["ticker"]),
            "gics_sector": features.get("gics_sector", "Unknown"),
            "expected_return": features["forecast_score"].astype(float),
            "volatility": (
                features[volatility_col].astype(float)
                if volatility_col is not None
                else pd.Series(1.0, index=features.index)
            ),
            "eligible_for_optimization": True,
        }
    )
    return result


def _estimate_covariance(
    returns: pd.DataFrame,
    tickers: list[str],
    rebalance_date: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    history = returns.loc[returns["date"] < rebalance_date].copy()
    pivot = (
        history.pivot(index="date", columns="ticker", values="return_1d")
        .sort_index()
        .tail(lookback_days)
    )
    covariance = pivot.reindex(columns=tickers).fillna(0).cov() * 252
    covariance = covariance.fillna(0)
    for ticker in tickers:
        if ticker not in covariance.index:
            covariance.loc[ticker, ticker] = 1e-6
        diagonal_value = float(covariance.loc[[ticker], [ticker]].to_numpy(dtype=float)[0, 0])
        if diagonal_value <= 0:
            covariance.loc[ticker, ticker] = 1e-6
    covariance = covariance.reindex(index=tickers, columns=tickers).fillna(0)
    return (covariance + covariance.T) / 2


def _first_matching_column(frame: pd.DataFrame, prefix: str) -> str | None:
    for column in frame.columns:
        if column.startswith(prefix):
            return column
    return None


def _deposit_rebalance_dates(
    rebalance_dates: pd.DatetimeIndex,
    schedule: ContributionSchedule,
) -> pd.DatetimeIndex:
    mapped = contributions_for_rebalance_dates(rebalance_dates, schedule)
    return pd.DatetimeIndex(
        [rebalance_date for rebalance_date, amount in mapped.items() if amount > 0]
    )


def _post_deposit_previous_weights(
    previous_weights: pd.Series | None,
    tickers: list[str],
    *,
    value_before_contribution: float,
    value_after_contribution: float,
) -> pd.Series:
    if previous_weights is None:
        return pd.Series(0.0, index=pd.Index(tickers, name="ticker"), name="previous_weight")
    dilution = value_before_contribution / value_after_contribution
    return (
        previous_weights.reindex(tickers)
        .fillna(0.0)
        .astype(float)
        .mul(dilution)
        .rename("previous_weight")
    )


def _trade_delta_after_no_trade_band(trade_delta: pd.Series, no_trade_band: float) -> pd.Series:
    if no_trade_band <= 0:
        return trade_delta
    return trade_delta.where(trade_delta.abs().ge(no_trade_band), 0.0)


def _attach_cashflow_summary(
    result: pd.DataFrame,
    *,
    initial_portfolio_value: float,
    ending_value: float,
    total_deposits: float,
    total_commissions: float,
    money_weighted: float | None,
) -> pd.DataFrame:
    enriched = result.copy()
    invested_capital = initial_portfolio_value + total_deposits
    enriched["strategy_ending_value"] = ending_value
    enriched["total_deposits"] = total_deposits
    enriched["total_commissions"] = total_commissions
    enriched["commission_to_deposit_ratio"] = (
        total_commissions / total_deposits if total_deposits > 0 else np.nan
    )
    enriched["total_return_on_invested_capital"] = (
        ending_value / invested_capital - 1 if invested_capital > 0 else np.nan
    )
    enriched["money_weighted_return"] = np.nan if money_weighted is None else money_weighted
    return enriched
