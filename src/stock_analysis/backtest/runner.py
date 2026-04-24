from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from stock_analysis.config import OptimizerConfig
from stock_analysis.optimization.engine import optimize_long_only


class PredictiveModel(Protocol):
    def predict(self, features: pd.DataFrame) -> Sequence[float]:
        """Return forecast scores for the provided feature cross-section."""


@dataclass(frozen=True)
class BacktestConfig:
    horizon_days: int = 5
    training_target_column: str | None = None
    rebalance_step_days: int = 5
    embargo_days: int = 15
    cost_bps: float = 5.0
    covariance_lookback_days: int = 252
    feature_columns: tuple[str, ...] = ()
    max_rebalances: int | None = None


def run_walk_forward_backtest(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    returns: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], PredictiveModel],
    optimizer_config: OptimizerConfig,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run a walk-forward portfolio backtest in long format, one row per ticker/rebalance."""

    cfg = config or BacktestConfig()
    feature_panel = _prepare_dates(panel)
    label_panel = _prepare_dates(labels)
    returns_frame = _prepare_dates(returns)
    realized_target_col = f"fwd_return_{cfg.horizon_days}d"
    training_target_col = cfg.training_target_column or realized_target_col
    label_columns = ["ticker", "date", realized_target_col]
    if training_target_col != realized_target_col:
        label_columns.append(training_target_col)
    missing_label_columns = [
        column for column in label_columns if column not in label_panel.columns
    ]
    if missing_label_columns:
        msg = f"labels are missing required target columns: {missing_label_columns}"
        raise ValueError(msg)

    merged_panel = feature_panel.merge(
        label_panel[label_columns],
        on=["ticker", "date"],
        how="left",
    )
    rebalance_dates = pd.DatetimeIndex(sorted(merged_panel["date"].dropna().unique()))
    records: list[dict[str, object]] = []
    previous_weights: pd.Series | None = None
    completed_rebalances = 0

    rebalance_step = max(cfg.rebalance_step_days, 1)
    candidate_rebalance_dates = rebalance_dates[::rebalance_step]
    if cfg.max_rebalances is not None and len(candidate_rebalance_dates) > cfg.max_rebalances:
        sample_positions = np.linspace(
            0,
            len(candidate_rebalance_dates) - 1,
            cfg.max_rebalances,
        ).round()
        candidate_rebalance_dates = candidate_rebalance_dates[
            np.unique(sample_positions.astype(int))
        ]

    for rebalance_date in candidate_rebalance_dates:
        if cfg.max_rebalances is not None and completed_rebalances >= cfg.max_rebalances:
            break

        train_cutoff = rebalance_date - pd.offsets.BDay(cfg.embargo_days)
        train_df = merged_panel.loc[
            (merged_panel["date"] < train_cutoff) & merged_panel[training_target_col].notna()
        ].copy()
        features_at_rebalance = merged_panel.loc[merged_panel["date"] == rebalance_date].copy()
        features_at_rebalance = features_at_rebalance.dropna(subset=[realized_target_col])
        if train_df.empty or features_at_rebalance.empty:
            continue

        model = predict_fn(train_df)
        scores = np.asarray(
            model.predict(_select_features(features_at_rebalance, cfg)), dtype=float
        )
        if len(scores) != len(features_at_rebalance):
            msg = "predict_fn model returned a score count that does not match rebalance features"
            raise ValueError(msg)
        features_at_rebalance["forecast_score"] = scores

        optimizer_input = _optimizer_input_from_scores(features_at_rebalance)
        covariance = _estimate_covariance(
            returns_frame,
            optimizer_input["ticker"].tolist(),
            rebalance_date,
            cfg.covariance_lookback_days,
        )
        weights = optimize_long_only(
            optimizer_input,
            covariance,
            optimizer_config,
            w_prev=previous_weights,
        )
        aligned_previous = (
            pd.Series(0.0, index=weights.index)
            if previous_weights is None
            else previous_weights.reindex(weights.index).fillna(0)
        )
        turnover = float(0.5 * np.abs(weights - aligned_previous).sum())
        period_cost = cfg.cost_bps / 10_000 * turnover

        realized = features_at_rebalance.set_index("ticker")[realized_target_col].astype(float)
        gross_return = float((weights * realized.reindex(weights.index).fillna(0)).sum())
        net_return = gross_return - period_cost

        predictions = features_at_rebalance.set_index("ticker")["forecast_score"]
        for ticker, weight in weights.items():
            records.append(
                {
                    "rebalance_date": rebalance_date.date().isoformat(),
                    "ticker": ticker,
                    "target_weight": float(weight),
                    "previous_weight": float(aligned_previous.get(ticker, 0.0)),
                    "forecast_score": float(predictions.get(ticker, np.nan)),
                    "realized_return": float(realized.get(ticker, np.nan)),
                    "portfolio_gross_return": gross_return,
                    "transaction_cost": period_cost,
                    "portfolio_net_return": net_return,
                    "turnover": turnover,
                }
            )
        previous_weights = weights
        completed_rebalances += 1

    return pd.DataFrame.from_records(records)


def _prepare_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    return result


def _select_features(frame: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if config.feature_columns:
        return frame.loc[:, list(config.feature_columns)]
    excluded = {
        "ticker",
        "date",
        "security",
        "gics_sector",
        "gics_sub_industry",
    }
    numeric_columns = [
        column
        for column in frame.columns
        if column not in excluded
        and not column.startswith("fwd_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return frame[numeric_columns]


def _optimizer_input_from_scores(features: pd.DataFrame) -> pd.DataFrame:
    volatility_col = _first_matching_column(features, "volatility_")
    result = pd.DataFrame(
        {
            "ticker": features["ticker"],
            "security": features.get("security", features["ticker"]),
            "gics_sector": features.get("gics_sector", "Unknown"),
            "expected_return": features["forecast_score"].astype(float),
            "volatility": (
                features[volatility_col].astype(float)
                if volatility_col is not None
                else pd.Series(1.0, index=features.index)
            ),
            "eligible_for_optimization": True,
        }
    )
    return result


def _estimate_covariance(
    returns: pd.DataFrame,
    tickers: list[str],
    rebalance_date: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    history = returns.loc[returns["date"] < rebalance_date].copy()
    pivot = (
        history.pivot(index="date", columns="ticker", values="return_1d")
        .sort_index()
        .tail(lookback_days)
    )
    covariance = pivot.reindex(columns=tickers).fillna(0).cov() * 252
    covariance = covariance.fillna(0)
    for ticker in tickers:
        if ticker not in covariance.index:
            covariance.loc[ticker, ticker] = 1e-6
        diagonal_value = float(covariance.loc[[ticker], [ticker]].to_numpy(dtype=float)[0, 0])
        if diagonal_value <= 0:
            covariance.loc[ticker, ticker] = 1e-6
    covariance = covariance.reindex(index=tickers, columns=tickers).fillna(0)
    return (covariance + covariance.T) / 2


def _first_matching_column(frame: pd.DataFrame, prefix: str) -> str | None:
    for column in frame.columns:
        if column.startswith(prefix):
            return column
    return None
