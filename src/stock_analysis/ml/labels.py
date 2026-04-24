from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from stock_analysis.domain.schemas import validate_columns


def build_forward_return_labels(
    daily_prices: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    benchmark_returns: pd.DataFrame | None = None,
    horizons: Sequence[int] = (5, 21, 63),
) -> pd.DataFrame:
    """Build forward-return labels aligned to a PIT feature panel."""

    panel_keys = _prepare_panel_keys(panel)
    if panel_keys.empty:
        return _empty_labels()

    prices = _prepare_prices(daily_prices)
    labels = panel_keys.copy()

    for horizon in horizons:
        if horizon < 1:
            msg = "label horizons must be positive"
            raise ValueError(msg)
        horizon_returns = _forward_returns(prices, horizon)
        labels = labels.merge(horizon_returns, on=["ticker", "date"], how="left")

        return_col = f"fwd_return_{horizon}d"
        labels = _add_excess_return(labels, benchmark_returns, horizon, return_col)
        labels = _add_rank_and_top_tercile(labels, horizon, return_col)

    labels = labels.sort_values(["ticker", "date"]).reset_index(drop=True)
    labels["date"] = labels["date"].dt.date.astype(str)
    return validate_columns(labels, "labels_panel")


def _prepare_panel_keys(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    keys = panel[["ticker", "date"]].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    return keys.drop_duplicates(subset=["ticker", "date"])


def _prepare_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    prices = daily_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    return (
        prices.dropna(subset=["ticker", "date", "adj_close"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def _forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    result = prices[["ticker", "date", "adj_close"]].copy()
    result[f"fwd_return_{horizon}d"] = (
        prices.groupby("ticker", sort=False)["adj_close"].shift(-horizon) / result["adj_close"] - 1
    )
    return result[["ticker", "date", f"fwd_return_{horizon}d"]]


def _add_excess_return(
    labels: pd.DataFrame,
    benchmark_returns: pd.DataFrame | None,
    horizon: int,
    return_col: str,
) -> pd.DataFrame:
    excess_col = f"fwd_excess_return_{horizon}d"
    if benchmark_returns is None or benchmark_returns.empty:
        labels[excess_col] = np.nan
        return labels

    benchmark = benchmark_returns.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.loc[
        benchmark["horizon_days"].astype(int) == horizon, ["date", "spy_return"]
    ]
    merged = labels.merge(benchmark, on="date", how="left")
    merged[excess_col] = merged[return_col] - merged["spy_return"]
    return merged.drop(columns=["spy_return"])


def _add_rank_and_top_tercile(labels: pd.DataFrame, horizon: int, return_col: str) -> pd.DataFrame:
    rank_col = f"fwd_rank_{horizon}d"
    top_col = f"fwd_is_top_tercile_{horizon}d"
    labels[rank_col] = labels.groupby("date")[return_col].rank(method="average", ascending=True)
    pct_rank = labels.groupby("date")[return_col].rank(method="average", pct=True)
    labels[top_col] = np.where(
        labels[return_col].notna(), (pct_rank >= (2 / 3)).astype(int), np.nan
    )
    return labels


def _empty_labels() -> pd.DataFrame:
    return validate_columns(pd.DataFrame(columns=["ticker", "date"]), "labels_panel")
