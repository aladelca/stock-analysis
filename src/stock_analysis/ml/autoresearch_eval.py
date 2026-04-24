from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stock_analysis.backtest.runner import BacktestConfig, run_walk_forward_backtest
from stock_analysis.config import OptimizerConfig
from stock_analysis.ml.autoresearch_candidate import (
    build_model_factory,
    get_candidate,
    resolve_feature_columns,
)
from stock_analysis.ml.evaluation import (
    EvaluationConfig,
    benchmark_relative_metrics,
    portfolio_metrics,
)

DecisionStatus = Literal["failed_infrastructure", "rejected", "provisional", "go"]

RESULT_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "iteration_id",
    "git_commit",
    "candidate_id",
    "candidate_description",
    "input_run_root",
    "max_assets",
    "max_rebalances",
    "optimizer_max_weight",
    "commission_rate",
    "cost_bps",
    "candidate_sharpe",
    "spy_sharpe",
    "sharpe_diff",
    "sharpe_diff_ci_low",
    "sharpe_diff_ci_high",
    "annualized_return",
    "active_return",
    "tracking_error",
    "information_ratio",
    "max_drawdown",
    "mean_turnover",
    "ir_observations",
    "status",
    "notes",
)

BASELINE_METRICS: dict[str, float | str] = {
    "candidate_id": "phase2_e8_final",
    "candidate_sharpe": 2.987197,
    "spy_sharpe": 1.130685,
    "sharpe_diff": 1.856512,
    "active_return": 0.636036,
    "information_ratio": 2.212435,
    "sharpe_diff_ci_low": -0.4455413268214286,
    "sharpe_diff_ci_high": 1.5416748795037905,
}


@dataclass(frozen=True)
class AutoresearchEvalConfig:
    candidate_id: str
    input_run_root: Path
    max_assets: int | None = 100
    max_rebalances: int | None = 48
    optimizer_max_weight: float = 0.30
    risk_aversion: float = 10.0
    min_trade_weight: float = 0.005
    lambda_turnover: float = 0.001
    commission_rate: float = 0.02
    horizon_days: int | None = None
    rebalance_step_days: int = 5
    embargo_days: int = 15
    cost_bps: float = 5.0
    covariance_lookback_days: int = 252
    liquidity_column: str = "dollar_volume_21d"
    iteration_id: str | None = None


def evaluate_candidate(config: AutoresearchEvalConfig) -> dict[str, Any]:
    candidate = get_candidate(config.candidate_id)
    horizon_days = config.horizon_days or candidate.horizon_days
    artifacts = _load_phase1_artifacts(config.input_run_root)
    artifacts = _filter_artifacts_by_liquidity(artifacts, config)
    panel = artifacts["panel"]
    feature_columns = resolve_feature_columns(panel, candidate)
    backtest = run_walk_forward_backtest(
        panel,
        artifacts["labels"],
        artifacts["returns"],
        build_model_factory(candidate, feature_columns),
        OptimizerConfig(
            max_weight=config.optimizer_max_weight,
            risk_aversion=config.risk_aversion,
            min_trade_weight=config.min_trade_weight,
            lambda_turnover=config.lambda_turnover,
            commission_rate=config.commission_rate,
        ),
        BacktestConfig(
            horizon_days=horizon_days,
            training_target_column=candidate.training_target_column,
            rebalance_step_days=config.rebalance_step_days,
            embargo_days=config.embargo_days,
            commission_rate=config.commission_rate,
            cost_bps=config.cost_bps,
            covariance_lookback_days=config.covariance_lookback_days,
            feature_columns=feature_columns,
            max_rebalances=config.max_rebalances,
        ),
    )
    if backtest.empty:
        return _failed_result(config, candidate.description, "candidate produced no backtest rows")

    periods_per_year = max(1, round(252 / max(config.rebalance_step_days, 1)))
    portfolio_returns = _portfolio_period_returns(backtest)
    benchmark = _benchmark_for_horizon(artifacts["benchmark"], horizon_days)
    spy_returns = _aligned_spy_returns(portfolio_returns, benchmark)
    if len(spy_returns) < 2:
        return _failed_result(
            config, candidate.description, "SPY benchmark alignment returned <2 rows"
        )

    portfolio = portfolio_metrics(portfolio_returns["portfolio_return"], periods_per_year)
    spy = portfolio_metrics(spy_returns["benchmark_return"], periods_per_year)
    benchmark_relative = benchmark_relative_metrics(
        portfolio_returns,
        benchmark.rename(columns={"spy_return": "benchmark_return"}),
        EvaluationConfig(periods_per_year=periods_per_year),
    )
    ci = sharpe_difference_ci(portfolio_returns, spy_returns)
    comparison = _comparison_metrics(portfolio, spy, benchmark_relative, ci, backtest)
    decision = decide_candidate(comparison)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "iteration_id": config.iteration_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "git_commit": _git_commit(),
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "description": candidate.description,
            "model_kind": candidate.model_kind,
            "feature_columns": list(feature_columns),
            "training_target_column": candidate.training_target_column,
        },
        "config": _config_payload(config, horizon_days),
        "metrics": {
            "portfolio": portfolio,
            "spy": spy,
            "benchmark_relative": benchmark_relative,
            "comparison": comparison,
        },
        "baseline": BASELINE_METRICS,
        "decision": decision,
    }


def decide_candidate(metrics: dict[str, float | None]) -> dict[str, Any]:
    candidate_sharpe = _finite_or_none(metrics.get("candidate_sharpe"))
    spy_sharpe = _finite_or_none(metrics.get("spy_sharpe"))
    active_return = _finite_or_none(metrics.get("active_return"))
    information_ratio = _finite_or_none(metrics.get("information_ratio"))
    ci_low = _finite_or_none(metrics.get("sharpe_diff_ci_low"))

    failures: list[str] = []
    if candidate_sharpe is None or spy_sharpe is None:
        failures.append("missing finite candidate or SPY Sharpe")
    elif candidate_sharpe <= spy_sharpe:
        failures.append("candidate Sharpe does not beat SPY Sharpe")
    if active_return is None or active_return <= 0:
        failures.append("annualized active return is not positive")
    if information_ratio is None or information_ratio <= 0:
        failures.append("SPY-relative information ratio is not positive")

    if failures:
        return {
            "status": "rejected",
            "passed_spy_gate": False,
            "objective_improved": False,
            "notes": "; ".join(failures),
        }

    baseline_sharpe_diff = float(BASELINE_METRICS["sharpe_diff"])
    baseline_ir = float(BASELINE_METRICS["information_ratio"])
    sharpe_diff = _finite_or_none(metrics.get("sharpe_diff"))
    objective_improved = bool(
        sharpe_diff is not None
        and information_ratio is not None
        and sharpe_diff > baseline_sharpe_diff
        and information_ratio >= baseline_ir
    )
    if ci_low is not None and ci_low > 0:
        return {
            "status": "go",
            "passed_spy_gate": True,
            "objective_improved": objective_improved,
            "notes": "candidate beats SPY and Sharpe-difference CI lower bound is positive",
        }
    return {
        "status": "provisional",
        "passed_spy_gate": True,
        "objective_improved": objective_improved,
        "notes": "candidate beats SPY on point estimates but CI lower bound is not positive",
    }


def append_result_tsv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(result_to_tsv_row(result))


def result_to_tsv_row(result: dict[str, Any]) -> dict[str, object]:
    metrics = result.get("metrics", {}).get("comparison", {})
    config = result.get("config", {})
    decision = result.get("decision", {})
    candidate = result.get("candidate", {})
    return {
        "timestamp_utc": result.get("timestamp_utc", ""),
        "iteration_id": result.get("iteration_id", ""),
        "git_commit": result.get("git_commit", ""),
        "candidate_id": candidate.get("candidate_id", ""),
        "candidate_description": candidate.get("description", ""),
        "input_run_root": config.get("input_run_root", ""),
        "max_assets": config.get("max_assets", ""),
        "max_rebalances": config.get("max_rebalances", ""),
        "optimizer_max_weight": config.get("optimizer_max_weight", ""),
        "commission_rate": config.get("commission_rate", ""),
        "cost_bps": config.get("cost_bps", ""),
        "candidate_sharpe": metrics.get("candidate_sharpe", ""),
        "spy_sharpe": metrics.get("spy_sharpe", ""),
        "sharpe_diff": metrics.get("sharpe_diff", ""),
        "sharpe_diff_ci_low": metrics.get("sharpe_diff_ci_low", ""),
        "sharpe_diff_ci_high": metrics.get("sharpe_diff_ci_high", ""),
        "annualized_return": metrics.get("annualized_return", ""),
        "active_return": metrics.get("active_return", ""),
        "tracking_error": metrics.get("tracking_error", ""),
        "information_ratio": metrics.get("information_ratio", ""),
        "max_drawdown": metrics.get("max_drawdown", ""),
        "mean_turnover": metrics.get("mean_turnover", ""),
        "ir_observations": metrics.get("ir_observations", ""),
        "status": decision.get("status", ""),
        "notes": decision.get("notes", ""),
    }


def result_to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True)


def sharpe_difference_ci(
    candidate_returns: pd.DataFrame,
    spy_returns: pd.DataFrame,
    *,
    samples: int = 500,
) -> tuple[float, float] | None:
    merged = candidate_returns.merge(spy_returns, on="date", how="inner").dropna()
    if len(merged) < 5:
        return None
    periods_per_year = _infer_periods_per_year(merged["date"])
    rng = np.random.default_rng(42)
    diffs: list[float] = []
    for _ in range(samples):
        sample = merged.iloc[rng.choice(len(merged), size=len(merged), replace=True)]
        candidate_sharpe = portfolio_metrics(
            sample["portfolio_return"],
            periods_per_year,
        ).get("sharpe", 0.0)
        spy_sharpe = portfolio_metrics(
            sample["benchmark_return"],
            periods_per_year,
        ).get("sharpe", 0.0)
        diffs.append(float(candidate_sharpe) - float(spy_sharpe))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def _load_phase1_artifacts(run_root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "panel": run_root / "silver" / "asset_daily_features_panel.parquet",
        "labels": run_root / "gold" / "labels_panel.parquet",
        "returns": run_root / "silver" / "asset_daily_returns.parquet",
        "benchmark": run_root / "silver" / "benchmark_returns.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        msg = f"missing Phase 1 artifacts for autoresearch evaluation: {missing}"
        raise FileNotFoundError(msg)
    return {name: pd.read_parquet(path) for name, path in paths.items()}


def _filter_artifacts_by_liquidity(
    artifacts: dict[str, pd.DataFrame],
    config: AutoresearchEvalConfig,
) -> dict[str, pd.DataFrame]:
    if config.max_assets is None:
        return artifacts
    panel = artifacts["panel"]
    if config.liquidity_column not in panel.columns:
        msg = f"cannot apply max_assets; missing liquidity column {config.liquidity_column}"
        raise ValueError(msg)
    latest_date = pd.to_datetime(panel["date"]).max()
    dated_panel = panel.assign(_date=pd.to_datetime(panel["date"]))
    top_tickers = (
        dated_panel.loc[dated_panel["_date"] == latest_date]
        .sort_values(config.liquidity_column, ascending=False)
        .head(config.max_assets)["ticker"]
        .astype(str)
        .tolist()
    )
    filtered = artifacts.copy()
    for key in ["panel", "labels", "returns"]:
        filtered[key] = (
            artifacts[key].loc[artifacts[key]["ticker"].astype(str).isin(top_tickers)].copy()
        )
    return filtered


def _benchmark_for_horizon(benchmark: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    result = benchmark.loc[benchmark["horizon_days"].astype(int) == horizon_days].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.date.astype(str)
    return result.dropna(subset=["spy_return"])


def _portfolio_period_returns(backtest: pd.DataFrame) -> pd.DataFrame:
    return backtest.drop_duplicates("rebalance_date")[
        ["rebalance_date", "portfolio_net_return"]
    ].rename(columns={"rebalance_date": "date", "portfolio_net_return": "portfolio_return"})


def _aligned_spy_returns(
    portfolio_returns: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    spy = benchmark[["date", "spy_return"]].rename(columns={"spy_return": "benchmark_return"})
    portfolio_dates = portfolio_returns[["date"]].copy()
    return portfolio_dates.merge(spy, on="date", how="inner").dropna()


def _comparison_metrics(
    portfolio: dict[str, float],
    spy: dict[str, float],
    benchmark_relative: dict[str, float],
    ci: tuple[float, float] | None,
    backtest: pd.DataFrame,
) -> dict[str, float | None]:
    candidate_sharpe = _finite_or_none(portfolio.get("sharpe"))
    spy_sharpe = _finite_or_none(spy.get("sharpe"))
    ci_low = ci[0] if ci else None
    ci_high = ci[1] if ci else None
    return {
        "candidate_sharpe": candidate_sharpe,
        "spy_sharpe": spy_sharpe,
        "sharpe_diff": (
            float(candidate_sharpe - spy_sharpe)
            if candidate_sharpe is not None and spy_sharpe is not None
            else None
        ),
        "sharpe_diff_ci_low": ci_low,
        "sharpe_diff_ci_high": ci_high,
        "annualized_return": _finite_or_none(portfolio.get("annualized_return")),
        "active_return": _finite_or_none(benchmark_relative.get("active_return")),
        "tracking_error": _finite_or_none(benchmark_relative.get("tracking_error")),
        "information_ratio": _finite_or_none(benchmark_relative.get("information_ratio")),
        "max_drawdown": _finite_or_none(portfolio.get("max_drawdown")),
        "mean_turnover": _mean_turnover(backtest),
        "ir_observations": _finite_or_none(benchmark_relative.get("observations")),
    }


def _failed_result(
    config: AutoresearchEvalConfig,
    candidate_description: str,
    note: str,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "iteration_id": config.iteration_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "git_commit": _git_commit(),
        "candidate": {
            "candidate_id": config.candidate_id,
            "description": candidate_description,
        },
        "config": _config_payload(config, config.horizon_days or 5),
        "metrics": {"comparison": {}},
        "baseline": BASELINE_METRICS,
        "decision": {
            "status": "failed_infrastructure",
            "passed_spy_gate": False,
            "objective_improved": False,
            "notes": note,
        },
    }


def _config_payload(config: AutoresearchEvalConfig, horizon_days: int) -> dict[str, Any]:
    return {
        "input_run_root": str(config.input_run_root),
        "max_assets": config.max_assets,
        "max_rebalances": config.max_rebalances,
        "optimizer_max_weight": config.optimizer_max_weight,
        "risk_aversion": config.risk_aversion,
        "min_trade_weight": config.min_trade_weight,
        "lambda_turnover": config.lambda_turnover,
        "commission_rate": config.commission_rate,
        "horizon_days": horizon_days,
        "rebalance_step_days": config.rebalance_step_days,
        "embargo_days": config.embargo_days,
        "cost_bps": config.cost_bps,
        "covariance_lookback_days": config.covariance_lookback_days,
    }


def _finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _mean_turnover(backtest: pd.DataFrame) -> float | None:
    if backtest.empty or "turnover" not in backtest.columns:
        return None
    value = pd.to_numeric(
        backtest.drop_duplicates("rebalance_date")["turnover"],
        errors="coerce",
    ).mean()
    return float(value) if pd.notna(value) else None


def _infer_periods_per_year(dates: pd.Series) -> int:
    parsed = pd.to_datetime(dates).sort_values()
    if len(parsed) < 2:
        return 252
    median_days = max(float(parsed.diff().dt.days.dropna().median()), 1.0)
    return max(1, round(365.25 / median_days))


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()
