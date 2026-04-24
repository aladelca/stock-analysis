from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from stock_analysis.config import OptimizerConfig
from stock_analysis.domain.schemas import validate_columns
from stock_analysis.portfolio.holdings import align_current_weights


def build_recommendations(
    optimizer_input: pd.DataFrame,
    weights: pd.Series,
    config: OptimizerConfig,
    as_of_date: str,
    run_id: str,
    current_weights: pd.Series | dict[str, float] | None = None,
) -> pd.DataFrame:
    result = optimizer_input.copy()
    result["target_weight"] = result["ticker"].map(weights).fillna(0.0).astype(float)
    result["current_weight"] = align_current_weights(
        current_weights,
        pd.Index(result["ticker"].astype(str)),
    ).to_numpy(dtype=float)
    result["_outside_optimizer_universe"] = False

    outside_rows = _current_holdings_outside_optimizer_universe(
        optimizer_input,
        current_weights,
    )
    if not outside_rows.empty:
        result = pd.concat([result, outside_rows], ignore_index=True)

    result["target_weight"] = pd.to_numeric(result["target_weight"], errors="coerce").fillna(0.0)
    result["current_weight"] = pd.to_numeric(result["current_weight"], errors="coerce").fillna(0.0)
    result["trade_weight"] = result["target_weight"] - result["current_weight"]
    result["trade_abs_weight"] = result["trade_weight"].abs()
    result["rebalance_required"] = result["trade_abs_weight"].ge(
        config.min_rebalance_trade_weight
    ) & (result["target_weight"].gt(0) | result["current_weight"].gt(0))
    result["action"] = np.select(
        [
            result["trade_weight"].ge(config.min_rebalance_trade_weight),
            result["trade_weight"].le(-config.min_rebalance_trade_weight),
            result["target_weight"].gt(0) | result["current_weight"].gt(0),
        ],
        ["BUY", "SELL", "HOLD"],
        default="EXCLUDE",
    )
    planned_trade = result["action"].isin(["BUY", "SELL"])
    result["estimated_commission_weight"] = np.where(
        planned_trade,
        config.commission_rate * result["trade_abs_weight"],
        0.0,
    )
    result["net_trade_weight_after_commission"] = np.where(
        planned_trade,
        np.sign(result["trade_weight"])
        * np.maximum(result["trade_abs_weight"] - result["estimated_commission_weight"], 0.0),
        0.0,
    )
    result["cash_required_weight"] = np.where(
        result["action"].eq("BUY"),
        result["trade_weight"] + result["estimated_commission_weight"],
        0.0,
    )
    result["cash_released_weight"] = np.where(
        result["action"].eq("SELL"),
        np.maximum(result["trade_abs_weight"] - result["estimated_commission_weight"], 0.0),
        0.0,
    )
    result["current_weight_label"] = result["current_weight"].map(_percent_label)
    result["target_weight_label"] = result["target_weight"].map(_percent_label)
    result["trade_weight_label"] = result["trade_weight"].map(_signed_percent_label)
    result["estimated_commission_weight_label"] = result["estimated_commission_weight"].map(
        _percent_label
    )
    result["reason_code"] = _reason_codes(result)
    result["as_of_date"] = as_of_date
    result["run_id"] = run_id
    columns = [
        "ticker",
        "security",
        "gics_sector",
        "expected_return",
        "volatility",
        "current_weight",
        "target_weight",
        "trade_weight",
        "trade_abs_weight",
        "current_weight_label",
        "target_weight_label",
        "trade_weight_label",
        "estimated_commission_weight",
        "estimated_commission_weight_label",
        "net_trade_weight_after_commission",
        "cash_required_weight",
        "cash_released_weight",
        "rebalance_required",
        "action",
        "reason_code",
        "as_of_date",
        "run_id",
    ]
    sort_frame = result[columns].copy()
    sort_frame["_action_rank"] = sort_frame["action"].map(
        {"BUY": 0, "SELL": 1, "HOLD": 2, "EXCLUDE": 3}
    )
    return validate_columns(
        sort_frame.sort_values(
            ["_action_rank", "trade_abs_weight", "target_weight"],
            ascending=[True, False, False],
        )
        .drop(columns=["_action_rank"])
        .reset_index(drop=True),
        "portfolio_recommendations",
    )


def build_risk_metrics(
    optimizer_input: pd.DataFrame,
    covariance: pd.DataFrame,
    weights: pd.Series,
    as_of_date: str,
    run_id: str,
) -> pd.DataFrame:
    selected = optimizer_input.set_index("ticker").reindex(weights.index)
    mu = selected["expected_return"].astype(float).to_numpy()
    cov = (
        covariance.reindex(index=weights.index, columns=weights.index)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    w = weights.to_numpy(dtype=float)
    portfolio_return = float(w @ mu)
    portfolio_volatility = float(sqrt(max(w @ cov @ w, 0)))
    rows = [
        ("expected_return", portfolio_return),
        ("expected_volatility", portfolio_volatility),
        ("num_holdings", float((weights > 0).sum())),
        ("max_weight", float(weights.max())),
        ("concentration_hhi", float(np.square(w).sum())),
    ]
    result = pd.DataFrame(rows, columns=["metric", "value"])
    result["as_of_date"] = as_of_date
    result["run_id"] = run_id
    return validate_columns(result, "portfolio_risk_metrics")


def build_sector_exposure(
    optimizer_input: pd.DataFrame,
    weights: pd.Series,
    as_of_date: str,
    run_id: str,
) -> pd.DataFrame:
    joined = optimizer_input[["ticker", "gics_sector"]].copy()
    joined["target_weight"] = joined["ticker"].map(weights).fillna(0.0)
    result = joined.groupby("gics_sector", dropna=False)["target_weight"].sum().reset_index()
    result["as_of_date"] = as_of_date
    result["run_id"] = run_id
    return validate_columns(result.sort_values("target_weight", ascending=False), "sector_exposure")


def _current_holdings_outside_optimizer_universe(
    optimizer_input: pd.DataFrame,
    current_weights: pd.Series | dict[str, float] | None,
) -> pd.DataFrame:
    if current_weights is None:
        return pd.DataFrame()
    series = (
        current_weights if isinstance(current_weights, pd.Series) else pd.Series(current_weights)
    )
    if series.empty:
        return pd.DataFrame()
    series.index = series.index.astype(str)
    universe_tickers = set(optimizer_input["ticker"].astype(str))
    outside = series.loc[~series.index.isin(universe_tickers)].astype(float)
    outside = outside.loc[outside.gt(0)]
    if outside.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "ticker": outside.index,
            "security": outside.index,
            "gics_sector": "Unknown",
            "expected_return": np.nan,
            "volatility": np.nan,
            "eligible_for_optimization": False,
            "target_weight": 0.0,
            "current_weight": outside.to_numpy(dtype=float),
            "_outside_optimizer_universe": True,
        }
    )


def _reason_codes(result: pd.DataFrame) -> pd.Series:
    reason = pd.Series("not_selected_or_below_threshold", index=result.index, dtype="object")
    reason.loc[result["action"].isin(["BUY", "SELL"])] = "rebalance_to_target"
    reason.loc[result["action"].eq("HOLD")] = "within_rebalance_threshold"

    eligible = result.get("eligible_for_optimization", pd.Series(False, index=result.index))
    eligible = eligible.fillna(False).astype(bool)
    reason.loc[result["action"].eq("EXCLUDE") & ~eligible] = "ineligible_data_quality"

    outside = result.get("_outside_optimizer_universe", pd.Series(False, index=result.index))
    outside = outside.fillna(False).astype(bool)
    reason.loc[outside] = "current_holding_outside_optimizer_universe"
    return reason


def _percent_label(value: float) -> str:
    return f"{float(value):.2%}"


def _signed_percent_label(value: float) -> str:
    return f"{float(value):+.2%}"
