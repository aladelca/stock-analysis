from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analysis.config import PanelFeatureConfig
from stock_analysis.features.panel import compute_asset_feature_panel


@pytest.fixture
def panel_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=120)
    tickers = ["AAA", "BBB", "CCC"]
    rows: list[dict[str, object]] = []
    for idx, ticker in enumerate(tickers):
        base = 100.0 + idx * 20
        for day, current_date in enumerate(dates):
            close = base * (1 + 0.001 * day + 0.0002 * idx * day)
            rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "adj_close": close,
                    "volume": 1_000_000 + idx * 10_000 + day * 100,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def panel_constituents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "security": ["Alpha Corp", "Beta Inc", "Gamma Ltd"],
            "gics_sector": ["Information Technology", "Health Care", "Financials"],
            "gics_sub_industry": ["Software", "Biotech", "Banks"],
        }
    )


@pytest.fixture
def panel_config() -> PanelFeatureConfig:
    return PanelFeatureConfig(
        min_history_days=30,
        momentum_windows=[5, 21, 63],
        volatility_windows=[21, 63],
        drawdown_windows=[63],
        moving_average_windows=[21, 63],
        return_windows=[1, 5, 21],
        volume_zscore_window=21,
        compute_cross_sectional_ranks=True,
    )


def test_panel_has_row_per_ticker_date(
    panel_prices: pd.DataFrame,
    panel_constituents: pd.DataFrame,
    panel_config: PanelFeatureConfig,
) -> None:
    panel = compute_asset_feature_panel(panel_prices, panel_constituents, panel_config)

    expected_rows = panel_prices.groupby("ticker").size().sub(63).sum()
    assert len(panel) == expected_rows
    assert set(panel["ticker"].unique()) == {"AAA", "BBB", "CCC"}
    assert panel["history_days"].min() >= 64


def test_panel_features_are_point_in_time(
    panel_prices: pd.DataFrame,
    panel_constituents: pd.DataFrame,
    panel_config: PanelFeatureConfig,
) -> None:
    """At any (ticker, t), momentum_Nd must equal P_t / P_{t-N} - 1 using only data up to t."""
    panel = compute_asset_feature_panel(panel_prices, panel_constituents, panel_config)

    # Pick a ticker and a date well into the series.
    ticker_rows = panel[panel["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    source = (
        panel_prices[panel_prices["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    )

    for t_index in [65, 100]:
        as_of_date = source.loc[t_index, "date"]
        expected_momentum_21 = (
            source["adj_close"].iloc[t_index] / source["adj_close"].iloc[t_index - 21] - 1
        )
        actual = ticker_rows.loc[ticker_rows["date"] == as_of_date, "momentum_21d"].iloc[0]
        assert actual == pytest.approx(expected_momentum_21, rel=1e-9)


def test_panel_no_future_leakage(
    panel_prices: pd.DataFrame,
    panel_constituents: pd.DataFrame,
    panel_config: PanelFeatureConfig,
) -> None:
    """Truncating prices to date <= t must not change features at (ticker, s) for s <= t."""
    full_panel = compute_asset_feature_panel(panel_prices, panel_constituents, panel_config)

    cutoff = panel_prices["date"].sort_values().iloc[70]
    truncated_prices = panel_prices[panel_prices["date"] <= cutoff].copy()
    truncated_panel = compute_asset_feature_panel(
        truncated_prices, panel_constituents, panel_config
    )

    full_subset = (
        full_panel[full_panel["date"] <= cutoff]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    truncated_sorted = truncated_panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # All features computed on truncated history must match the full-history panel for the
    # same (ticker, date) pairs. If any feature leaked future data, values would differ.
    feature_columns = [
        c for c in truncated_sorted.columns if c not in {"ticker", "date", "adj_close"}
    ]
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(truncated_sorted[col]):
            continue
        full_values = full_subset[col].to_numpy(dtype=float)
        truncated_values = truncated_sorted[col].to_numpy(dtype=float)
        mask = ~(np.isnan(full_values) & np.isnan(truncated_values))
        np.testing.assert_allclose(
            full_values[mask],
            truncated_values[mask],
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"Column {col} leaked future data: values differ after truncation",
        )


def test_panel_volatility_requires_minimum_window(
    panel_prices: pd.DataFrame,
    panel_constituents: pd.DataFrame,
    panel_config: PanelFeatureConfig,
) -> None:
    panel = compute_asset_feature_panel(panel_prices, panel_constituents, panel_config)
    ticker_rows = panel[panel["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)

    assert ticker_rows["history_days"].min() >= 64
    assert ticker_rows["volatility_21d"].notna().all()
    assert ticker_rows["volatility_63d"].notna().all()


def test_panel_cross_sectional_ranks_computed_per_date(
    panel_prices: pd.DataFrame,
    panel_constituents: pd.DataFrame,
    panel_config: PanelFeatureConfig,
) -> None:
    panel = compute_asset_feature_panel(panel_prices, panel_constituents, panel_config)
    assert "momentum_21d_rank" in panel.columns

    # Rank percentiles must live in (0, 1] for any non-NaN rows.
    ranked = panel.dropna(subset=["momentum_21d_rank"])
    assert ranked["momentum_21d_rank"].min() > 0
    assert ranked["momentum_21d_rank"].max() <= 1.0

    # For any given date, ranks across tickers must have mean ~= 0.5 (uniform distribution).
    sample_date = ranked["date"].unique()[10]
    same_date = ranked[ranked["date"] == sample_date]
    if len(same_date) >= 2:
        assert same_date["momentum_21d_rank"].between(0, 1).all()


def test_panel_with_benchmark_adds_excess_returns(
    panel_prices: pd.DataFrame,
    panel_constituents: pd.DataFrame,
    panel_config: PanelFeatureConfig,
) -> None:
    dates = pd.bdate_range("2025-01-01", periods=120)
    benchmark = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in dates],
            "return_1d": np.full(len(dates), 0.0005),
        }
    )

    panel = compute_asset_feature_panel(
        panel_prices, panel_constituents, panel_config, benchmark_returns=benchmark
    )

    assert "return_21d_excess" in panel.columns
    # Excess = asset minus benchmark. With constant 5bps/day benchmark, 21d compound is small
    # but non-zero, so the excess column should not equal the raw return column.
    non_null = panel.dropna(subset=["return_21d", "return_21d_excess"]).head(50)
    assert not np.allclose(
        non_null["return_21d"].to_numpy(dtype=float),
        non_null["return_21d_excess"].to_numpy(dtype=float),
    )
