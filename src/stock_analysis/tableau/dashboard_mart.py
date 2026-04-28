from __future__ import annotations

from typing import Any, cast

import pandas as pd

from stock_analysis.tableau.account_tracking_marts import latest_performance_fields

DASHBOARD_WEIGHT_EPSILON = 1e-8


def build_dashboard_mart(
    recommendations: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    sector_exposure: pd.DataFrame,
    run_metadata: pd.DataFrame,
    *,
    performance_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    mart = recommendations.copy()
    risk_wide = _risk_metrics_wide(risk_metrics)
    metadata = _single_row_metadata(run_metadata)
    performance = (
        latest_performance_fields(performance_snapshots)
        if performance_snapshots is not None
        else {}
    )

    if "forecast_score" not in mart.columns:
        mart["forecast_score"] = mart["expected_return"]
    if "expected_return_is_calibrated" not in mart.columns:
        mart["expected_return_is_calibrated"] = False
    if "calibrated_expected_return" not in mart.columns:
        mart["calibrated_expected_return"] = pd.NA
    target_weight = pd.to_numeric(mart["target_weight"], errors="coerce").fillna(0.0)
    current_weight = pd.to_numeric(mart["current_weight"], errors="coerce").fillna(0.0)
    trade_weight = pd.to_numeric(mart["trade_weight"], errors="coerce").fillna(0.0)
    mart["is_solver_dust"] = target_weight.abs().gt(0) & target_weight.abs().lt(
        DASHBOARD_WEIGHT_EPSILON
    )
    mart["display_target_weight"] = target_weight.mask(mart["is_solver_dust"], 0.0)
    mart["display_current_weight"] = current_weight.mask(
        current_weight.abs().lt(DASHBOARD_WEIGHT_EPSILON),
        0.0,
    )
    mart["display_trade_weight"] = trade_weight.mask(
        trade_weight.abs().lt(DASHBOARD_WEIGHT_EPSILON),
        0.0,
    )
    mart["selected"] = mart["display_target_weight"].gt(0)
    mart["target_weight_label"] = mart["target_weight"].map(lambda value: f"{float(value):.2%}")
    mart["current_weight_label"] = mart["current_weight"].map(lambda value: f"{float(value):.2%}")
    mart["trade_weight_label"] = mart["trade_weight"].map(lambda value: f"{float(value):+.2%}")
    mart["estimated_commission_weight_label"] = mart["estimated_commission_weight"].map(
        lambda value: f"{float(value):.2%}"
    )
    mart["trade_notional_label"] = mart["trade_notional"].map(_currency_label)
    mart["commission_amount_label"] = mart["commission_amount"].map(_currency_label)
    mart["scatter_size"] = mart["display_target_weight"].where(mart["selected"], 0.001)

    for column, metric_value in risk_wide.items():
        mart[f"portfolio_{column}"] = metric_value
    for column, metadata_value in metadata.items():
        mart[f"run_{column}"] = [metadata_value] * len(mart)  # type: ignore[assignment]
    for column, performance_value in performance.items():
        mart[column] = [performance_value] * len(mart)  # type: ignore[assignment]

    mart["portfolio_return_per_vol"] = _safe_ratio(
        mart.get("portfolio_expected_return"),
        mart.get("portfolio_expected_volatility"),
    )
    mart["run_config_hash_short"] = mart["run_config_hash"].astype(str).str[:8]
    mart["is_data_date_lagged"] = (
        mart["run_requested_as_of_date"].astype(str).ne(mart["run_data_as_of_date"].astype(str))
    )
    mart["data_date_status"] = mart["is_data_date_lagged"].map(
        {True: "Market data date differs from requested run date", False: "Current requested date"}
    )

    sector_weights = sector_exposure.set_index("gics_sector")["target_weight"]
    mart["sector_target_weight"] = mart["gics_sector"].map(sector_weights).fillna(0.0)
    mart = _coerce_date_columns(
        mart,
        [
            "as_of_date",
            "forecast_start_date",
            "forecast_end_date",
            "run_requested_as_of_date",
            "run_data_as_of_date",
            "run_calibration_trained_through_date",
        ],
    )

    column_order = [
        "run_id",
        "as_of_date",
        "ticker",
        "security",
        "gics_sector",
        "forecast_score",
        "expected_return_is_calibrated",
        "calibrated_expected_return",
        "forecast_horizon_days",
        "forecast_start_date",
        "forecast_end_date",
        "realized_return",
        "realized_spy_return",
        "realized_active_return",
        "forecast_error",
        "forecast_hit",
        "outcome_status",
        "volatility",
        "current_weight",
        "display_current_weight",
        "current_weight_label",
        "target_weight",
        "display_target_weight",
        "target_weight_label",
        "trade_weight",
        "display_trade_weight",
        "trade_abs_weight",
        "trade_weight_label",
        "rebalance_required",
        "estimated_commission_weight",
        "estimated_commission_weight_label",
        "net_trade_weight_after_commission",
        "cash_required_weight",
        "cash_released_weight",
        "portfolio_value_before_contribution",
        "contribution_amount",
        "portfolio_value_after_contribution",
        "current_market_value",
        "target_market_value",
        "trade_notional",
        "trade_notional_label",
        "commission_amount",
        "commission_amount_label",
        "deposit_used_amount",
        "cash_after_trade_amount",
        "no_trade_band_applied",
        "is_solver_dust",
        "selected",
        "scatter_size",
        "action",
        "reason_code",
        "sector_target_weight",
        "portfolio_expected_return",
        "portfolio_expected_volatility",
        "portfolio_return_per_vol",
        "portfolio_num_holdings",
        "portfolio_max_weight",
        "portfolio_concentration_hhi",
        "account_total_value",
        "account_initial_value",
        "account_total_deposits",
        "account_invested_capital",
        "account_net_external_cashflow",
        "account_return_on_invested_capital",
        "account_time_weighted_return",
        "account_money_weighted_return",
        "spy_same_cashflow_value",
        "spy_time_weighted_return",
        "spy_money_weighted_return",
        "active_value",
        "active_return",
        "run_requested_as_of_date",
        "run_data_as_of_date",
        "run_live_account_enabled",
        "run_live_account_slug",
        "run_live_cashflow_source",
        "run_live_snapshot_id",
        "run_live_snapshot_date",
        "run_live_unapplied_cashflow_amount",
        "run_live_unapplied_cashflow_count",
        "is_data_date_lagged",
        "data_date_status",
        "run_created_at_utc",
        "run_config_hash",
        "run_config_hash_short",
        "run_expected_return_is_calibrated",
        "run_optimizer_return_unit",
        "run_calibration_enabled",
        "run_calibration_method",
        "run_calibration_target",
        "run_calibration_model_version",
        "run_calibration_status",
        "run_calibration_trained_through_date",
        "run_calibration_observations",
        "run_calibration_mae",
        "run_calibration_rmse",
        "run_calibration_rank_ic",
    ]
    for column in column_order:
        if column not in mart.columns:
            mart[column] = pd.NA
    return (
        mart[column_order]
        .sort_values(["rebalance_required", "trade_abs_weight", "target_weight"], ascending=False)
        .reset_index(drop=True)
    )


def _risk_metrics_wide(risk_metrics: pd.DataFrame) -> dict[str, float]:
    if risk_metrics.empty:
        return {}
    metric_names = risk_metrics["metric"].astype(str).tolist()
    metric_values = risk_metrics["value"].to_numpy(dtype=float)
    return {name: float(metric_values[index]) for index, name in enumerate(metric_names)}


def _single_row_metadata(run_metadata: pd.DataFrame) -> dict[str, object]:
    if run_metadata.empty:
        return {}
    row = run_metadata.iloc[0].to_dict()
    return {
        key: value
        for key, value in row.items()
        if key
        in {
            "requested_as_of_date",
            "data_as_of_date",
            "created_at_utc",
            "config_hash",
            "expected_return_is_calibrated",
            "optimizer_return_unit",
            "calibration_enabled",
            "calibration_method",
            "calibration_target",
            "calibration_model_version",
            "calibration_status",
            "calibration_trained_through_date",
            "calibration_observations",
            "calibration_mae",
            "calibration_rmse",
            "calibration_rank_ic",
            "live_account_enabled",
            "live_account_slug",
            "live_cashflow_source",
            "live_snapshot_id",
            "live_snapshot_date",
            "live_unapplied_cashflow_amount",
            "live_unapplied_cashflow_count",
        }
        and _metadata_value_is_present(value)
    }


def _metadata_value_is_present(value: object) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(cast(Any, value)))
    except (TypeError, ValueError):
        return True


def _safe_ratio(numerator: pd.Series | None, denominator: pd.Series | None) -> pd.Series:
    if numerator is None or denominator is None:
        return pd.Series([pd.NA])
    numeric_denominator = pd.to_numeric(denominator, errors="coerce")
    return pd.to_numeric(numerator, errors="coerce") / numeric_denominator.where(
        numeric_denominator.ne(0)
    )


def _coerce_date_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.date
    return result


def _currency_label(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"${float(numeric):,.2f}"
