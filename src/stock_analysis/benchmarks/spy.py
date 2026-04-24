from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from stock_analysis.domain.schemas import validate_columns


def build_spy_daily(
    daily_prices: pd.DataFrame,
    *,
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    """Build a daily SPY benchmark series from normalized price bars."""

    if daily_prices.empty:
        return _empty_spy_daily()

    prices = daily_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    if "ticker" not in prices.columns and "provider_ticker" in prices.columns:
        prices["ticker"] = prices["provider_ticker"]

    spy = prices.loc[prices["ticker"].astype(str) == benchmark_ticker].copy()
    if spy.empty:
        return _empty_spy_daily()

    spy = spy.dropna(subset=["adj_close"]).sort_values("date")
    spy["return_1d"] = spy["adj_close"].pct_change(fill_method=None)
    columns = ["date", "adj_close", "return_1d"]
    if "as_of_date" in spy.columns:
        columns.append("as_of_date")
    result = spy[columns].drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    result["date"] = result["date"].dt.date.astype(str)
    return validate_columns(result, "spy_daily")


def build_benchmark_returns(
    spy_daily: pd.DataFrame,
    *,
    horizons: Sequence[int] = (5, 21, 63),
) -> pd.DataFrame:
    """Build forward SPY returns for each requested horizon."""

    if spy_daily.empty:
        return _empty_benchmark_returns()

    spy = spy_daily.copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy["adj_close"] = pd.to_numeric(spy["adj_close"], errors="coerce")
    spy = spy.dropna(subset=["adj_close"]).sort_values("date").reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for horizon in horizons:
        if horizon < 1:
            msg = "benchmark horizons must be positive"
            raise ValueError(msg)
        frame = pd.DataFrame(
            {
                "date": spy["date"],
                "horizon_days": int(horizon),
                "spy_return": spy["adj_close"].shift(-horizon) / spy["adj_close"] - 1,
            }
        )
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True) if frames else _empty_benchmark_returns()
    result["date"] = pd.to_datetime(result["date"]).dt.date.astype(str)
    return validate_columns(result, "benchmark_returns")


def benchmark_return_horizon(
    benchmark_returns: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    horizon_days: int,
) -> float | None:
    """Return a cached SPY forward return for a date/horizon pair."""

    if benchmark_returns.empty:
        return None
    returns = benchmark_returns.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    target_date = pd.Timestamp(as_of_date).normalize()
    match = returns.loc[
        (returns["date"] == target_date) & (returns["horizon_days"] == horizon_days),
        "spy_return",
    ]
    if match.empty or pd.isna(match.iloc[0]):
        return None
    return float(match.iloc[0])


def _empty_spy_daily() -> pd.DataFrame:
    return validate_columns(
        pd.DataFrame(columns=["date", "adj_close", "return_1d", "as_of_date"]),
        "spy_daily",
    )


def _empty_benchmark_returns() -> pd.DataFrame:
    return validate_columns(
        pd.DataFrame(columns=["date", "horizon_days", "spy_return"]),
        "benchmark_returns",
    )
