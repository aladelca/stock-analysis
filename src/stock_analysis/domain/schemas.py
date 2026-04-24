from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "sp500_constituents": {
        "ticker",
        "provider_ticker",
        "security",
        "gics_sector",
        "gics_sub_industry",
        "as_of_date",
    },
    "daily_prices": {
        "ticker",
        "provider_ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "as_of_date",
    },
    "asset_daily_returns": {"ticker", "date", "adj_close", "return_1d", "as_of_date"},
    "asset_daily_features": {
        "ticker",
        "as_of_date",
        "latest_adj_close",
        "history_days",
        "eligible_for_optimization",
    },
    "asset_daily_features_panel": {
        "ticker",
        "date",
        "adj_close",
        "history_days",
    },
    "spy_daily": {
        "date",
        "adj_close",
        "return_1d",
    },
    "benchmark_returns": {
        "date",
        "horizon_days",
        "spy_return",
    },
    "labels_panel": {
        "ticker",
        "date",
    },
    "optimizer_input": {
        "ticker",
        "expected_return",
        "volatility",
        "eligible_for_optimization",
        "gics_sector",
    },
    "portfolio_recommendations": {
        "ticker",
        "security",
        "gics_sector",
        "target_weight",
        "action",
        "reason_code",
        "as_of_date",
        "run_id",
    },
    "portfolio_risk_metrics": {"metric", "value", "as_of_date", "run_id"},
    "sector_exposure": {"gics_sector", "target_weight", "as_of_date", "run_id"},
}


def validate_columns(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    required = REQUIRED_COLUMNS[schema_name]
    missing = sorted(required - set(df.columns))
    if missing:
        msg = f"{schema_name} is missing required columns: {missing}"
        raise ValueError(msg)
    return df
