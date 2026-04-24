from __future__ import annotations

from stock_analysis.features.price_features import compute_asset_daily_features
from stock_analysis.ingestion.universe import parse_sp500_constituents


def test_compute_asset_daily_features(sample_html, sample_prices, sample_config) -> None:
    constituents = parse_sp500_constituents(sample_html, sample_config.run.as_of_date)

    features = compute_asset_daily_features(sample_prices, constituents, sample_config.features)

    assert len(features) == 4
    assert features["eligible_for_optimization"].all()
    assert "momentum_10d" in features.columns
    assert "volatility_5d" in features.columns
    assert features["latest_adj_close"].notna().all()
