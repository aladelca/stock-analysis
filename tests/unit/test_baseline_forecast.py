from __future__ import annotations

from stock_analysis.features.price_features import compute_asset_daily_features
from stock_analysis.forecasting.baseline import build_optimizer_inputs
from stock_analysis.ingestion.universe import parse_sp500_constituents
from stock_analysis.medallion.silver import build_asset_daily_returns


def test_build_optimizer_inputs(sample_html, sample_prices, sample_config) -> None:
    constituents = parse_sp500_constituents(sample_html, sample_config.run.as_of_date)
    features = compute_asset_daily_features(sample_prices, constituents, sample_config.features)
    returns = build_asset_daily_returns(sample_prices)

    optimizer_input, covariance = build_optimizer_inputs(features, returns, sample_config.forecast)

    assert optimizer_input["eligible_for_optimization"].all()
    assert set(covariance.columns) == set(optimizer_input["ticker"])
    assert "expected_return" in optimizer_input.columns
