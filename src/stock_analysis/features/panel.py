from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from stock_analysis.config import PanelFeatureConfig
from stock_analysis.domain.schemas import validate_columns

TRADING_DAYS = 252


def compute_asset_feature_panel(
    daily_prices: pd.DataFrame,
    constituents: pd.DataFrame,
    config: PanelFeatureConfig,
    *,
    benchmark_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a point-in-time (PIT) panel of features, one row per (ticker, date).

    Every feature at (i, t) uses only prices from [t - lookback, t]. No forward fill,
    no information from t+1 onwards. Rows are truncated until every configured lookback
    and the configured minimum history requirement are satisfied.
    """

    prices = _prepare_prices(daily_prices)
    if prices.empty:
        return _empty_panel(config)

    panel = _compute_per_ticker_features(prices, config)

    if benchmark_returns is not None and not benchmark_returns.empty:
        panel = _add_market_relative_features(panel, benchmark_returns, config)

    panel = _drop_pre_lookback_rows(panel, config)

    if config.compute_cross_sectional_ranks:
        panel = _add_cross_sectional_ranks(panel, config)

    panel = panel.merge(
        constituents[["ticker", "security", "gics_sector", "gics_sub_industry"]].drop_duplicates(
            subset=["ticker"]
        ),
        on="ticker",
        how="left",
    )

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["date"] = panel["date"].dt.date.astype(str)
    return validate_columns(panel, "asset_daily_features_panel")


def _prepare_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    prices = daily_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    if "volume" in prices.columns:
        prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    prices = prices.dropna(subset=["adj_close"]).sort_values(["ticker", "date"])
    return prices.reset_index(drop=True)


def _empty_panel(config: PanelFeatureConfig) -> pd.DataFrame:
    columns = ["ticker", "date", "adj_close", "history_days"]
    for window in config.momentum_windows:
        columns.append(f"momentum_{window}d")
    for window in config.volatility_windows:
        columns.append(f"volatility_{window}d")
    for window in config.drawdown_windows:
        columns.append(f"max_drawdown_{window}d")
    for window in config.moving_average_windows:
        columns.append(f"ma_ratio_{window}d")
    for window in config.return_windows:
        columns.append(f"return_{window}d")
    columns.extend(["dollar_volume_21d", "volume_21d_zscore"])
    return pd.DataFrame(columns=columns)


def _drop_pre_lookback_rows(panel: pd.DataFrame, config: PanelFeatureConfig) -> pd.DataFrame:
    lookback_requirements = [
        config.min_history_days,
        max(config.momentum_windows, default=1) + 1,
        max(config.return_windows, default=1) + 1,
        max(config.volatility_windows, default=1) + 1,
        max(config.drawdown_windows, default=1),
        max(config.moving_average_windows, default=1),
        config.volume_zscore_window,
    ]
    required_history_days = max(lookback_requirements)
    return panel.loc[panel["history_days"] >= required_history_days].copy()


def _compute_per_ticker_features(prices: pd.DataFrame, config: PanelFeatureConfig) -> pd.DataFrame:
    grouped = prices.groupby("ticker", sort=False, group_keys=False)
    panel = prices[["ticker", "date", "adj_close"]].copy()

    panel["history_days"] = grouped.cumcount() + 1

    panel["return_1d"] = grouped["adj_close"].pct_change(fill_method=None)

    for window in config.momentum_windows:
        panel[f"momentum_{window}d"] = grouped["adj_close"].pct_change(
            periods=window, fill_method=None
        )

    for window in config.return_windows:
        if window == 1:
            continue
        panel[f"return_{window}d"] = grouped["adj_close"].pct_change(
            periods=window, fill_method=None
        )

    returns = panel["return_1d"]
    for window in config.volatility_windows:
        panel[f"volatility_{window}d"] = returns.groupby(prices["ticker"], sort=False).rolling(
            window=window, min_periods=window
        ).std(ddof=1).reset_index(level=0, drop=True) * sqrt(TRADING_DAYS)

    for window in config.drawdown_windows:
        running_max = grouped["adj_close"].transform(
            lambda s, w=window: s.rolling(window=w, min_periods=w).max()
        )
        panel[f"max_drawdown_{window}d"] = panel["adj_close"] / running_max - 1

    for window in config.moving_average_windows:
        ma = grouped["adj_close"].transform(
            lambda s, w=window: s.rolling(window=w, min_periods=w).mean()
        )
        panel[f"ma_ratio_{window}d"] = panel["adj_close"] / ma

    _add_volume_features(panel, prices, config)

    return panel


def _add_volume_features(
    panel: pd.DataFrame, prices: pd.DataFrame, config: PanelFeatureConfig
) -> None:
    if "volume" not in prices.columns:
        panel["dollar_volume_21d"] = np.nan
        panel["volume_21d_zscore"] = np.nan
        return

    dollar_volume = prices["adj_close"] * prices["volume"]
    grouped_dollar = dollar_volume.groupby(prices["ticker"], sort=False).rolling(
        window=21, min_periods=21
    )
    panel["dollar_volume_21d"] = grouped_dollar.mean().reset_index(level=0, drop=True)

    w = config.volume_zscore_window
    grouped_volume = (
        prices["volume"].groupby(prices["ticker"], sort=False).rolling(window=w, min_periods=w)
    )
    volume_mean = grouped_volume.mean().reset_index(level=0, drop=True)
    volume_std = grouped_volume.std(ddof=1).reset_index(level=0, drop=True)
    panel["volume_21d_zscore"] = (prices["volume"] - volume_mean) / volume_std


def _add_market_relative_features(
    panel: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    config: PanelFeatureConfig,
) -> pd.DataFrame:
    bench = benchmark_returns.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.sort_values("date").reset_index(drop=True)

    merged = panel.merge(
        bench[["date", "return_1d"]].rename(columns={"return_1d": "benchmark_return_1d"}),
        on="date",
        how="left",
    )

    # Compounded benchmark return over each window aligned to ticker dates.
    bench["log_return"] = np.log1p(bench["return_1d"].astype(float))
    for window in config.return_windows:
        if window == 1:
            continue
        column = f"return_{window}d_excess"
        bench_window = bench["log_return"].rolling(window=window, min_periods=window).sum()
        bench["bench_window_return"] = np.expm1(bench_window)
        aligned = panel[["date"]].merge(
            bench[["date", "bench_window_return"]], on="date", how="left"
        )
        merged[column] = merged[f"return_{window}d"] - aligned["bench_window_return"].to_numpy()
    merged = merged.drop(columns=["benchmark_return_1d"], errors="ignore")
    return merged


def _add_cross_sectional_ranks(panel: pd.DataFrame, config: PanelFeatureConfig) -> pd.DataFrame:
    rank_targets: list[str] = []
    for window in config.momentum_windows:
        rank_targets.append(f"momentum_{window}d")
    for window in config.volatility_windows:
        rank_targets.append(f"volatility_{window}d")

    for column in rank_targets:
        if column not in panel.columns:
            continue
        ranks = panel.groupby("date")[column].rank(method="average", pct=True)
        panel[f"{column}_rank"] = ranks
    return panel
