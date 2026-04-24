from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from stock_analysis.config import FeatureConfig
from stock_analysis.domain.schemas import validate_columns

TRADING_DAYS = 252


def compute_asset_daily_features(
    daily_prices: pd.DataFrame,
    constituents: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    prices = daily_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices = prices.dropna(subset=["adj_close"]).sort_values(["ticker", "date"])

    rows: list[dict[str, object]] = []
    for ticker, group in prices.groupby("ticker"):
        group = group.sort_values("date")
        closes = group["adj_close"].astype(float).reset_index(drop=True)
        returns = closes.pct_change().dropna()
        history_days = int(closes.count())
        row: dict[str, object] = {
            "ticker": ticker,
            "as_of_date": str(group["as_of_date"].iloc[-1]),
            "latest_date": group["date"].iloc[-1].date().isoformat(),
            "latest_adj_close": float(closes.iloc[-1]),
            "history_days": history_days,
        }

        for window in config.momentum_windows:
            row[f"momentum_{window}d"] = _momentum(closes, window)
        for window in config.moving_average_windows:
            moving_average = _moving_average(closes, window)
            row[f"moving_average_{window}d"] = moving_average
            row[f"price_to_ma_{window}d"] = (
                float(closes.iloc[-1] / moving_average) if np.isfinite(moving_average) else np.nan
            )

        row[f"volatility_{config.volatility_window}d"] = _volatility(
            returns,
            config.volatility_window,
        )
        row[f"max_drawdown_{config.drawdown_window}d"] = _max_drawdown(
            closes,
            config.drawdown_window,
        )
        row["eligible_for_optimization"] = history_days >= config.min_history_days
        rows.append(row)

    features = pd.DataFrame(rows)
    features = features.merge(
        constituents[["ticker", "security", "gics_sector", "gics_sub_industry"]],
        on="ticker",
        how="left",
    )
    return validate_columns(
        features.sort_values("ticker").reset_index(drop=True), "asset_daily_features"
    )


def _momentum(closes: pd.Series, window: int) -> float:
    if len(closes) <= window:
        return np.nan
    return float(closes.iloc[-1] / closes.iloc[-window - 1] - 1)


def _moving_average(closes: pd.Series, window: int) -> float:
    if len(closes) < window:
        return np.nan
    return float(closes.tail(window).mean())


def _volatility(returns: pd.Series, window: int) -> float:
    if len(returns) < window:
        return np.nan
    return float(returns.tail(window).std(ddof=1) * sqrt(TRADING_DAYS))


def _max_drawdown(closes: pd.Series, window: int) -> float:
    if len(closes) < 2:
        return np.nan
    sample = closes.tail(window)
    running_max = sample.cummax()
    drawdowns = sample / running_max - 1
    return float(drawdowns.min())
