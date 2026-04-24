from __future__ import annotations

import pandas as pd


def build_dashboard_mart(
    recommendations: pd.DataFrame,
    risk_metrics: pd.DataFrame,
    sector_exposure: pd.DataFrame,
    run_metadata: pd.DataFrame,
) -> pd.DataFrame:
    mart = recommendations.copy()
    risk_wide = _risk_metrics_wide(risk_metrics)
    metadata = _single_row_metadata(run_metadata)

    mart["selected"] = mart["action"].astype(str).ne("EXCLUDE")
    mart["forecast_score"] = mart["expected_return"]
    mart["target_weight_label"] = mart["target_weight"].map(lambda value: f"{float(value):.2%}")
    mart["scatter_size"] = mart["target_weight"].where(mart["selected"], 0.001)

    for column, metric_value in risk_wide.items():
        mart[f"portfolio_{column}"] = metric_value
    for column, metadata_value in metadata.items():
        mart[f"run_{column}"] = metadata_value

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

    column_order = [
        "run_id",
        "as_of_date",
        "ticker",
        "security",
        "gics_sector",
        "forecast_score",
        "volatility",
        "target_weight",
        "target_weight_label",
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
        "run_requested_as_of_date",
        "run_data_as_of_date",
        "is_data_date_lagged",
        "data_date_status",
        "run_created_at_utc",
        "run_config_hash",
        "run_config_hash_short",
    ]
    for column in column_order:
        if column not in mart.columns:
            mart[column] = pd.NA
    return mart[column_order].sort_values("target_weight", ascending=False).reset_index(drop=True)


def _risk_metrics_wide(risk_metrics: pd.DataFrame) -> dict[str, float]:
    if risk_metrics.empty:
        return {}
    metric_names = risk_metrics["metric"].astype(str).tolist()
    metric_values = risk_metrics["value"].to_numpy(dtype=float)
    return {name: float(metric_values[index]) for index, name in enumerate(metric_names)}


def _single_row_metadata(run_metadata: pd.DataFrame) -> dict[str, str]:
    if run_metadata.empty:
        return {}
    row = run_metadata.iloc[0].to_dict()
    return {
        key: str(value)
        for key, value in row.items()
        if key in {"requested_as_of_date", "data_as_of_date", "created_at_utc", "config_hash"}
    }


def _safe_ratio(numerator: pd.Series | None, denominator: pd.Series | None) -> pd.Series:
    if numerator is None or denominator is None:
        return pd.Series([pd.NA])
    numeric_denominator = pd.to_numeric(denominator, errors="coerce")
    return pd.to_numeric(numerator, errors="coerce") / numeric_denominator.where(
        numeric_denominator.ne(0)
    )
