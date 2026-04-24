from __future__ import annotations

import pandas as pd

from stock_analysis.domain.schemas import validate_columns
from stock_analysis.io.csv import write_csv
from stock_analysis.io.parquet import write_parquet
from stock_analysis.paths import ProjectPaths


def build_asset_daily_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    prices = daily_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    prices = prices.dropna(subset=["adj_close"]).sort_values(["ticker", "date"])
    prices["return_1d"] = prices.groupby("ticker")["adj_close"].pct_change()
    result = prices[["ticker", "date", "adj_close", "return_1d", "as_of_date"]].copy()
    result["date"] = result["date"].dt.date.astype(str)
    return validate_columns(result, "asset_daily_returns")


def build_asset_universe_snapshot(
    constituents: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    prices = daily_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    stats = (
        prices.groupby("ticker")
        .agg(
            first_price_date=("date", "min"),
            last_price_date=("date", "max"),
            history_days=("date", "nunique"),
        )
        .reset_index()
    )
    snapshot = constituents.merge(stats, on="ticker", how="left")
    snapshot["history_days"] = snapshot["history_days"].fillna(0).astype(int)
    snapshot["first_price_date"] = pd.to_datetime(snapshot["first_price_date"]).dt.date.astype(str)
    snapshot["last_price_date"] = pd.to_datetime(snapshot["last_price_date"]).dt.date.astype(str)
    return snapshot


def write_silver_table(df: pd.DataFrame, name: str, paths: ProjectPaths) -> pd.DataFrame:
    write_parquet(df, paths.silver_path(name))
    write_csv(df, paths.csv_mirror_path("silver", name))
    return df
