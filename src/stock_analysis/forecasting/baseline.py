from __future__ import annotations

import numpy as np
import pandas as pd

from stock_analysis.config import ForecastConfig
from stock_analysis.domain.schemas import validate_columns


def build_optimizer_inputs(
    features: pd.DataFrame,
    returns: pd.DataFrame,
    config: ForecastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum_column = f"momentum_{config.momentum_window}d"
    volatility_columns = [column for column in features.columns if column.startswith("volatility_")]
    if momentum_column not in features.columns:
        msg = f"Missing required momentum column: {momentum_column}"
        raise ValueError(msg)
    if not volatility_columns:
        msg = "Missing volatility feature columns"
        raise ValueError(msg)

    volatility_column = volatility_columns[0]
    optimizer_input = features.copy()
    optimizer_input["expected_return"] = pd.to_numeric(
        optimizer_input[momentum_column], errors="coerce"
    ).fillna(0) - config.volatility_penalty * pd.to_numeric(
        optimizer_input[volatility_column], errors="coerce"
    ).fillna(0)
    optimizer_input["volatility"] = pd.to_numeric(
        optimizer_input[volatility_column],
        errors="coerce",
    )
    optimizer_input["eligible_for_optimization"] = (
        optimizer_input["eligible_for_optimization"].astype(bool)
        & np.isfinite(optimizer_input["expected_return"])
        & np.isfinite(optimizer_input["volatility"])
        & (optimizer_input["volatility"] > 0)
    )

    returns_matrix = returns.copy()
    returns_matrix["date"] = pd.to_datetime(returns_matrix["date"])
    pivot = (
        returns_matrix.pivot(index="date", columns="ticker", values="return_1d")
        .sort_index()
        .tail(config.covariance_lookback_days)
    )
    eligible = optimizer_input.loc[
        optimizer_input["eligible_for_optimization"],
        "ticker",
    ].tolist()
    covariance = pivot.reindex(columns=eligible).fillna(0).cov() * 252
    covariance = covariance.fillna(0)
    for ticker in covariance.columns:
        diagonal_value = float(covariance.loc[[ticker], [ticker]].to_numpy(dtype=float)[0, 0])
        if diagonal_value <= 0:
            covariance.loc[ticker, ticker] = 1e-8

    columns = [
        "ticker",
        "security",
        "gics_sector",
        *[column for column in ["is_benchmark_candidate"] if column in optimizer_input.columns],
        "expected_return",
        "volatility",
        "eligible_for_optimization",
        momentum_column,
        volatility_column,
        "as_of_date",
    ]
    for column in columns:
        if column not in optimizer_input.columns:
            optimizer_input[column] = pd.NA
    result = optimizer_input[columns].sort_values("ticker").reset_index(drop=True)
    return validate_columns(result, "optimizer_input"), covariance
