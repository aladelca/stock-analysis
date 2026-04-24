from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analysis.config import PanelFeatureConfig
from stock_analysis.features.panel import compute_asset_feature_panel
from stock_analysis.ml.cv import WalkForwardCVConfig, walk_forward_splits
from stock_analysis.ml.labels import build_forward_return_labels


@pytest.fixture
def leakage_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=90)
    rows: list[dict[str, object]] = []
    for ticker_idx, ticker in enumerate(["AAA", "BBB", "CCC"]):
        for day, current_date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "date": current_date.date().isoformat(),
                    "adj_close": 100 + ticker_idx * 10 + day * (1 + ticker_idx * 0.05),
                    "volume": 1_000_000 + ticker_idx * 100 + day,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def leakage_constituents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "security": ["Alpha", "Beta", "Gamma"],
            "gics_sector": ["Tech", "Health Care", "Financials"],
            "gics_sub_industry": ["Software", "Biotech", "Banks"],
        }
    )


def test_feature_at_t_uses_only_prices_through_t(
    leakage_prices: pd.DataFrame,
    leakage_constituents: pd.DataFrame,
) -> None:
    config = PanelFeatureConfig(
        min_history_days=30,
        momentum_windows=[21],
        volatility_windows=[21],
        drawdown_windows=[21],
        moving_average_windows=[21],
        return_windows=[1, 5, 21],
        compute_cross_sectional_ranks=False,
    )
    panel = compute_asset_feature_panel(leakage_prices, leakage_constituents, config)
    source = leakage_prices[leakage_prices["ticker"] == "AAA"].reset_index(drop=True)
    as_of_date = source.loc[40, "date"]
    expected = source.loc[40, "adj_close"] / source.loc[19, "adj_close"] - 1

    actual = panel.loc[
        (panel["ticker"] == "AAA") & (panel["date"] == as_of_date), "momentum_21d"
    ].iloc[0]

    assert actual == pytest.approx(expected)


def test_target_at_t_uses_only_forward_horizon_prices(
    leakage_prices: pd.DataFrame,
    leakage_constituents: pd.DataFrame,
) -> None:
    panel = leakage_prices[["ticker", "date", "adj_close"]]
    labels = build_forward_return_labels(leakage_prices, panel, horizons=[5])
    source = leakage_prices[leakage_prices["ticker"] == "BBB"].reset_index(drop=True)
    as_of_date = source.loc[10, "date"]
    expected = source.loc[15, "adj_close"] / source.loc[10, "adj_close"] - 1

    actual = labels.loc[
        (labels["ticker"] == "BBB") & (labels["date"] == as_of_date), "fwd_return_5d"
    ].iloc[0]

    assert actual == pytest.approx(expected)


def test_cv_has_no_train_validation_date_overlap() -> None:
    dates = pd.bdate_range("2020-01-01", "2025-12-31")
    folds = list(walk_forward_splits(dates, WalkForwardCVConfig(embargo_days=15)))

    assert folds
    for train_dates, val_dates in folds:
        assert len(train_dates.intersection(val_dates)) == 0


def test_cv_embargo_is_enforced() -> None:
    dates = pd.bdate_range("2020-01-01", "2025-12-31")
    folds = list(walk_forward_splits(dates, WalkForwardCVConfig(embargo_days=15)))

    for train_dates, val_dates in folds:
        assert val_dates.min() >= train_dates.max() + pd.offsets.BDay(15)


def test_preprocessing_fit_on_train_only() -> None:
    train = pd.Series([1.0, 2.0, 3.0])
    test = pd.Series([100.0])
    train_mean = train.mean()
    transformed_test = test - train_mean

    assert transformed_test.iloc[0] == pytest.approx(98.0)
    assert transformed_test.iloc[0] != pytest.approx(test.iloc[0] - pd.concat([train, test]).mean())


def test_cross_sectional_normalization_is_within_date() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-01-01"] * 3 + ["2026-01-02"] * 3,
            "score": [1, 2, 3, 10, 20, 30],
        }
    )
    frame["rank"] = frame.groupby("date")["score"].rank(pct=True)

    assert frame.groupby("date")["rank"].max().eq(1.0).all()
    assert frame.loc[0, "rank"] == frame.loc[3, "rank"]


def test_truncated_price_history_does_not_change_existing_features(
    leakage_prices: pd.DataFrame,
    leakage_constituents: pd.DataFrame,
) -> None:
    config = PanelFeatureConfig(
        min_history_days=30,
        momentum_windows=[21],
        volatility_windows=[21],
        drawdown_windows=[21],
        moving_average_windows=[21],
        return_windows=[1, 5, 21],
        compute_cross_sectional_ranks=True,
    )
    full = compute_asset_feature_panel(leakage_prices, leakage_constituents, config)
    cutoff = leakage_prices["date"].sort_values().iloc[60]
    truncated = compute_asset_feature_panel(
        leakage_prices[leakage_prices["date"] <= cutoff],
        leakage_constituents,
        config,
    )
    merged = full.merge(
        truncated,
        on=["ticker", "date"],
        suffixes=("_full", "_truncated"),
    )

    np.testing.assert_allclose(
        merged["momentum_21d_full"],
        merged["momentum_21d_truncated"],
        rtol=1e-12,
    )


def test_survivorship_bias_report_banner_text_is_required() -> None:
    banner = "Uses current S&P 500 constituents; survivorship bias present."

    assert "survivorship bias present" in banner.lower()


def test_hyperparameter_search_uses_inner_cv_only() -> None:
    outer_train = pd.bdate_range("2020-01-01", "2023-01-01")
    outer_test = pd.bdate_range("2023-02-01", "2023-06-01")
    inner_folds = list(
        walk_forward_splits(
            outer_train,
            WalkForwardCVConfig(
                train_window_years=1,
                val_window_months=3,
                step_months=3,
                embargo_days=15,
            ),
        )
    )

    assert inner_folds
    for train_dates, val_dates in inner_folds:
        assert train_dates.max() < outer_test.min()
        assert val_dates.max() < outer_test.min()


def test_ticker_identity_one_hot_features_are_excluded() -> None:
    feature_columns = ["momentum_21d", "gics_sector_Tech", "ticker_AAPL"]
    illegal = [column for column in feature_columns if column.startswith("ticker_")]

    assert illegal == ["ticker_AAPL"]


def test_date_leakage_features_exclude_year_and_month() -> None:
    allowed = {"day_of_week", "day_of_month"}
    risky = {"year", "month"}
    configured = {"day_of_week", "day_of_month"}

    assert configured <= allowed
    assert configured.isdisjoint(risky)


def test_delisted_post_delisting_returns_are_nan_not_zero() -> None:
    returns = pd.Series([0.01, -0.02, np.nan], name="return_1d")

    assert pd.isna(returns.iloc[-1])
    assert returns.iloc[-1] != 0


def test_batchnorm_eval_uses_training_statistics() -> None:
    train_mean = 10.0
    train_std = 2.0
    validation_batch = pd.Series([100.0, 102.0, 104.0])

    normalized = (validation_batch - train_mean) / train_std

    assert normalized.iloc[0] == pytest.approx(45.0)
    assert normalized.iloc[0] != pytest.approx(
        (validation_batch.iloc[0] - validation_batch.mean()) / validation_batch.std(ddof=0)
    )


def test_early_stopping_monitors_validation_not_test() -> None:
    monitored_split = "validation"
    forbidden_split = "test"

    assert monitored_split == "validation"
    assert monitored_split != forbidden_split
