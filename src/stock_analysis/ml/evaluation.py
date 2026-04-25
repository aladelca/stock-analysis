from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Literal

import numpy as np
import pandas as pd

from stock_analysis.backtest.cashflows import simulate_benchmark_value_path


@dataclass(frozen=True)
class EvaluationConfig:
    prediction_column: str = "prediction"
    target_column: str = "target"
    date_column: str = "date"
    ticker_column: str = "ticker"
    bootstrap_samples: int = 1000
    block_size: int = 5
    random_seed: int = 42
    periods_per_year: int = 252
    number_of_trials: int = 1


def evaluate(
    predictions: pd.DataFrame | pd.Series,
    targets: pd.DataFrame | pd.Series | None = None,
    prices: pd.DataFrame | None = None,
    benchmark_returns: pd.DataFrame | None = None,
    config: EvaluationConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate predictions with predictive, portfolio, and benchmark-relative metrics."""

    del prices
    cfg = _coerce_config(config)
    frame = _merge_predictions_targets(predictions, targets, cfg)
    predictive = _predictive_metrics(frame, cfg)

    portfolio: dict[str, float] = {}
    benchmark: dict[str, float] = {}
    if "portfolio_return" in frame.columns:
        portfolio = portfolio_metrics(frame["portfolio_return"], cfg.periods_per_year)
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark = benchmark_relative_metrics(
                frame[[cfg.date_column, "portfolio_return"]],
                benchmark_returns,
                cfg,
            )

    return {
        "predictive": predictive,
        "portfolio": portfolio,
        "benchmark_relative": benchmark,
    }


def portfolio_metrics(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
) -> dict[str, float]:
    series = pd.Series(returns, dtype=float).dropna()
    if series.empty:
        return {}
    values = series.to_numpy(dtype=float)
    compounded = float(np.prod(1 + values) - 1)
    annualized_return = float((1 + compounded) ** (periods_per_year / len(values)) - 1)
    annualized_volatility = (
        float(np.std(values, ddof=1) * sqrt(periods_per_year)) if len(values) > 1 else 0.0
    )
    sharpe = _safe_divide(annualized_return, annualized_volatility)
    cumulative = pd.Series(np.cumprod(1 + values))
    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = float(drawdown.min())
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": _safe_divide(annualized_return, abs(max_drawdown)),
        "hit_rate": float((series > 0).mean()),
        "deflated_sharpe_ratio": deflated_sharpe_ratio(sharpe),
    }


def benchmark_relative_metrics(
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    config: EvaluationConfig | dict[str, Any] | None = None,
) -> dict[str, float]:
    cfg = _coerce_config(config)
    portfolio = portfolio_returns.copy()
    portfolio[cfg.date_column] = pd.to_datetime(portfolio[cfg.date_column])

    benchmark = benchmark_returns.copy()
    benchmark[cfg.date_column] = pd.to_datetime(benchmark[cfg.date_column])
    benchmark_col = _first_existing_column(
        benchmark, ["benchmark_return", "spy_return", "return_1d"]
    )
    if benchmark_col is None:
        return {}

    merged = portfolio.merge(
        benchmark[[cfg.date_column, benchmark_col]],
        on=cfg.date_column,
        how="inner",
    ).dropna(subset=["portfolio_return", benchmark_col])
    merged = merged.sort_values(cfg.date_column).reset_index(drop=True)
    if len(merged) < 2:
        return {}

    y = merged["portfolio_return"].astype(float).to_numpy()
    x = merged[benchmark_col].astype(float).to_numpy()
    variance = float(np.var(x, ddof=1))
    beta = float(np.cov(x, y, ddof=1)[0, 1] / variance) if variance > 0 else 0.0
    alpha_period = float(y.mean() - beta * x.mean())
    active = y - x
    active_return = float(active.mean() * cfg.periods_per_year)
    tracking_error_period = float(np.std(active, ddof=1))
    tracking_error = float(tracking_error_period * sqrt(cfg.periods_per_year))
    information_ratio = _safe_divide(active_return, tracking_error)
    return {
        "alpha": alpha_period * cfg.periods_per_year,
        "beta": beta,
        "active_return": active_return,
        "active_return_period": float(active.mean()),
        "information_ratio": information_ratio,
        "tracking_error": tracking_error,
        "tracking_error_period": tracking_error_period,
        "portfolio_mean_return_period": float(y.mean()),
        "benchmark_mean_return_period": float(x.mean()),
        "observations": float(len(merged)),
    }


def contribution_cashflow_metrics(
    backtest: pd.DataFrame,
    benchmark_returns: pd.DataFrame | None = None,
) -> dict[str, float]:
    if backtest.empty or "portfolio_value_end" not in backtest.columns:
        return {}

    periods = (
        backtest.drop_duplicates("rebalance_date")
        .sort_values("rebalance_date")
        .reset_index(drop=True)
    )
    last = periods.iloc[-1]
    metrics: dict[str, float] = {
        "strategy_ending_value": float(last["strategy_ending_value"]),
        "strategy_total_deposits": float(last["total_deposits"]),
        "strategy_total_commissions": float(last["total_commissions"]),
        "strategy_time_weighted_return": float(last["cumulative_twr_return"]),
        "strategy_total_return_on_invested_capital": float(
            last["total_return_on_invested_capital"]
        ),
    }
    money_weighted = _finite_or_none(last.get("money_weighted_return"))
    if money_weighted is not None:
        metrics["strategy_money_weighted_return"] = money_weighted
    commission_ratio = _finite_or_none(last.get("commission_to_deposit_ratio"))
    if commission_ratio is not None:
        metrics["strategy_commission_to_deposit_ratio"] = commission_ratio

    if benchmark_returns is not None and not benchmark_returns.empty:
        benchmark = _benchmark_period_frame(periods, benchmark_returns)
        contribution_by_date = {
            pd.Timestamp(row["rebalance_date"]): float(row["external_contribution"])
            for _, row in periods.iterrows()
        }
        benchmark_metrics = simulate_benchmark_value_path(
            benchmark,
            initial_value=float(periods["portfolio_value_start"].iloc[0]),
            contribution_by_date=contribution_by_date,
            commission_rate=float(periods["commission_rate"].iloc[0]),
        )
        for key, value in benchmark_metrics.items():
            if value is not None:
                metrics[key] = float(value)
        if "benchmark_ending_value" in metrics:
            metrics["active_ending_value"] = (
                metrics["strategy_ending_value"] - metrics["benchmark_ending_value"]
            )
        if "benchmark_time_weighted_return" in metrics:
            metrics["active_time_weighted_return"] = (
                metrics["strategy_time_weighted_return"] - metrics["benchmark_time_weighted_return"]
            )
    return {key: value for key, value in metrics.items() if np.isfinite(value)}


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    number_of_trials: int = 1,
) -> float:
    """Lightweight DSR proxy that penalizes multiple trials without extra dependencies."""

    penalty = sqrt(max(number_of_trials, 1))
    return float(sharpe / penalty)


def _predictive_metrics(frame: pd.DataFrame, cfg: EvaluationConfig) -> dict[str, Any]:
    valid = frame.dropna(subset=[cfg.prediction_column, cfg.target_column]).copy()
    if valid.empty:
        return {}

    pred = valid[cfg.prediction_column].astype(float)
    target = valid[cfg.target_column].astype(float)
    error = pred - target
    metrics: dict[str, Any] = {
        "pearson_ic": float(pred.corr(target, method="pearson")),
        "rank_ic": float(pred.corr(target, method="spearman")),
        "hit_rate": float((np.sign(pred) == np.sign(target)).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }
    if _is_binary(target):
        metrics["roc_auc"] = _roc_auc_score(target.to_numpy(), pred.to_numpy())

    metrics["pearson_ic_ci_95"] = _bootstrap_ci(valid, cfg, method="pearson")
    metrics["rank_ic_ci_95"] = _bootstrap_ci(valid, cfg, method="spearman")
    return metrics


def _bootstrap_ci(
    frame: pd.DataFrame,
    cfg: EvaluationConfig,
    *,
    method: Literal["pearson", "spearman"],
) -> tuple[float, float]:
    if cfg.bootstrap_samples <= 0:
        value = float(frame[cfg.prediction_column].corr(frame[cfg.target_column], method=method))
        return value, value

    rng = np.random.default_rng(cfg.random_seed)
    dates = (
        pd.Index(frame[cfg.date_column].unique())
        if cfg.date_column in frame.columns
        else pd.Index(range(len(frame)))
    )
    samples: list[float] = []
    for _ in range(cfg.bootstrap_samples):
        sampled_dates = rng.choice(dates.to_numpy(), size=len(dates), replace=True)
        sampled = frame.set_index(cfg.date_column).loc[sampled_dates].reset_index()
        value = sampled[cfg.prediction_column].corr(sampled[cfg.target_column], method=method)
        if pd.notna(value):
            samples.append(float(value))
    if not samples:
        value = float(frame[cfg.prediction_column].corr(frame[cfg.target_column], method=method))
        return value, value
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _merge_predictions_targets(
    predictions: pd.DataFrame | pd.Series,
    targets: pd.DataFrame | pd.Series | None,
    cfg: EvaluationConfig,
) -> pd.DataFrame:
    pred = _coerce_prediction_frame(predictions, cfg)
    if targets is None:
        if cfg.target_column not in pred.columns:
            msg = "targets must be provided or included in predictions"
            raise ValueError(msg)
        return pred
    target = _coerce_target_frame(targets, cfg)
    join_columns = [
        column
        for column in [cfg.date_column, cfg.ticker_column]
        if column in pred.columns and column in target.columns
    ]
    if not join_columns:
        pred = pred.reset_index(drop=True)
        pred[cfg.target_column] = target[cfg.target_column].reset_index(drop=True)
        return pred
    return pred.merge(target, on=join_columns, how="inner")


def _coerce_prediction_frame(
    predictions: pd.DataFrame | pd.Series,
    cfg: EvaluationConfig,
) -> pd.DataFrame:
    if isinstance(predictions, pd.Series):
        return pd.DataFrame({cfg.prediction_column: predictions})
    frame = predictions.copy()
    if cfg.prediction_column not in frame.columns:
        candidate = _first_existing_column(frame, ["forecast_score", "prediction", "pred"])
        if candidate is None:
            msg = "prediction column not found"
            raise ValueError(msg)
        frame = frame.rename(columns={candidate: cfg.prediction_column})
    return frame


def _coerce_target_frame(
    targets: pd.DataFrame | pd.Series,
    cfg: EvaluationConfig,
) -> pd.DataFrame:
    if isinstance(targets, pd.Series):
        return pd.DataFrame({cfg.target_column: targets})
    frame = targets.copy()
    if cfg.target_column not in frame.columns:
        candidate = _first_existing_column(
            frame,
            ["target", "fwd_return_5d", "fwd_excess_return_5d", "label"],
        )
        if candidate is None:
            msg = "target column not found"
            raise ValueError(msg)
        frame = frame.rename(columns={candidate: cfg.target_column})
    return frame


def _coerce_config(config: EvaluationConfig | dict[str, Any] | None) -> EvaluationConfig:
    if config is None:
        return EvaluationConfig()
    if isinstance(config, EvaluationConfig):
        return config
    return EvaluationConfig(**config)


def _benchmark_period_frame(periods: pd.DataFrame, benchmark_returns: pd.DataFrame) -> pd.DataFrame:
    benchmark = benchmark_returns.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark_col = _first_existing_column(benchmark, ["benchmark_return", "spy_return"])
    if benchmark_col is None:
        return pd.DataFrame()
    period_dates = pd.DataFrame({"date": pd.to_datetime(periods["rebalance_date"])})
    return period_dates.merge(
        benchmark[["date", benchmark_col]].rename(columns={benchmark_col: "benchmark_return"}),
        on="date",
        how="inner",
    ).dropna()


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def _is_binary(target: pd.Series) -> bool:
    values = set(target.dropna().unique().tolist())
    return bool(values) and values <= {0, 1}


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_positive_ranks = float(ranks[positives].sum())
    return (sum_positive_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
