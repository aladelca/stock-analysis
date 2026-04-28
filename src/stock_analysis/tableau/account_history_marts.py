from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, fields
from datetime import date
from typing import Any, cast

import pandas as pd

from stock_analysis.forecasting.outcomes import attach_forecast_outcomes
from stock_analysis.storage.contracts import (
    AccountTrackingRepository,
    CashflowRecord,
    HoldingSnapshotRecord,
    PerformanceSnapshotRecord,
    PortfolioSnapshotRecord,
    RecommendationLineRecord,
    RecommendationRunRecord,
)

HISTORY_COLUMNS: dict[str, list[str]] = {
    "cashflows_history": [
        *[field.name for field in fields(CashflowRecord)],
        "account_slug",
    ],
    "portfolio_snapshots_history": [
        *[field.name for field in fields(PortfolioSnapshotRecord)],
        "account_slug",
    ],
    "holding_snapshots_history": [
        *[field.name for field in fields(HoldingSnapshotRecord)],
        "account_id",
        "snapshot_date",
        "account_slug",
    ],
    "recommendation_runs_history": [
        *[field.name for field in fields(RecommendationRunRecord)],
        "account_slug",
    ],
    "recommendation_lines_history": [
        *[field.name for field in fields(RecommendationLineRecord)],
        "account_id",
        "run_id",
        "run_requested_as_of_date",
        "run_data_as_of_date",
        "model_version",
        "ml_score_scale",
        "config_hash",
        "run_expected_return_is_calibrated",
        "run_status",
        "account_slug",
    ],
    "performance_snapshots_history": [
        *[field.name for field in fields(PerformanceSnapshotRecord)],
        "account_slug",
    ],
}


def build_account_history_marts(
    *,
    repository: AccountTrackingRepository,
    account_slug: str,
    daily_prices: pd.DataFrame,
    default_horizon_days: int,
    benchmark_ticker: str = "SPY",
) -> dict[str, pd.DataFrame]:
    account = repository.get_account_by_slug(account_slug)
    if account is None or account.id is None:
        return {}

    cashflows = _records_frame(repository.list_cashflows(account.id))
    snapshots = _records_frame(repository.list_portfolio_snapshots(account.id))
    holdings = _holding_history_frame(
        snapshots=repository.list_portfolio_snapshots(account.id),
        repository=repository,
    )
    runs = _records_frame(repository.list_recommendation_runs(account.id))
    lines = _recommendation_lines_history_frame(
        runs=repository.list_recommendation_runs(account.id),
        repository=repository,
        daily_prices=daily_prices,
        default_horizon_days=default_horizon_days,
        benchmark_ticker=benchmark_ticker,
    )
    performance = _records_frame(repository.list_performance_snapshots(account.id))

    tables = {
        "cashflows_history": cashflows,
        "portfolio_snapshots_history": snapshots,
        "holding_snapshots_history": holdings,
        "recommendation_runs_history": runs,
        "recommendation_lines_history": lines,
        "performance_snapshots_history": performance,
    }
    result: dict[str, pd.DataFrame] = {}
    for name, table in tables.items():
        prepared = table.copy()
        prepared["account_slug"] = account.slug
        result[name] = _ensure_columns(prepared, HISTORY_COLUMNS[name])
    return result


def _recommendation_lines_history_frame(
    *,
    runs: list[RecommendationRunRecord],
    repository: AccountTrackingRepository,
    daily_prices: pd.DataFrame,
    default_horizon_days: int,
    benchmark_ticker: str,
) -> pd.DataFrame:
    run_ids = [run.id for run in runs if run.id is not None]
    lines = repository.list_recommendation_lines([str(run_id) for run_id in run_ids])
    line_frame = _records_frame(lines)
    if line_frame.empty:
        return line_frame

    run_frame = _records_frame(runs)
    run_frame = run_frame.rename(
        columns={
            "id": "recommendation_run_id",
            "as_of_date": "run_requested_as_of_date",
            "data_as_of_date": "run_data_as_of_date",
            "expected_return_is_calibrated": "run_expected_return_is_calibrated",
            "status": "run_status",
        }
    )
    merged = line_frame.merge(
        run_frame[
            [
                "recommendation_run_id",
                "account_id",
                "run_id",
                "run_requested_as_of_date",
                "run_data_as_of_date",
                "model_version",
                "ml_score_scale",
                "config_hash",
                "run_expected_return_is_calibrated",
                "run_status",
            ]
        ],
        on="recommendation_run_id",
        how="left",
        suffixes=("", "_run"),
    )
    if "forecast_start_date" not in merged.columns:
        merged["forecast_start_date"] = merged["run_data_as_of_date"]
    else:
        merged["forecast_start_date"] = merged["forecast_start_date"].fillna(
            merged["run_data_as_of_date"]
        )
    if "forecast_score" not in merged.columns:
        merged["forecast_score"] = merged["expected_return"]
    else:
        merged["forecast_score"] = merged["forecast_score"].fillna(merged["expected_return"])
    if "expected_return_is_calibrated" not in merged.columns:
        merged["expected_return_is_calibrated"] = _bool_series(
            merged["run_expected_return_is_calibrated"]
        )
    else:
        merged["expected_return_is_calibrated"] = _bool_series(
            merged["expected_return_is_calibrated"]
            .combine_first(merged["run_expected_return_is_calibrated"])
        )
    return attach_forecast_outcomes(
        merged,
        daily_prices,
        horizon_days=default_horizon_days,
        benchmark_ticker=benchmark_ticker,
    )


def _holding_history_frame(
    *,
    snapshots: list[PortfolioSnapshotRecord],
    repository: AccountTrackingRepository,
) -> pd.DataFrame:
    snapshot_frame = _records_frame(snapshots)
    if snapshot_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    snapshot_lookup = {
        str(snapshot.id): snapshot for snapshot in snapshots if snapshot.id is not None
    }
    for snapshot_id, snapshot in snapshot_lookup.items():
        holdings = repository.list_holding_snapshots(snapshot_id)
        for holding in holdings:
            rows.append(_holding_row(holding, snapshot))
    return pd.DataFrame(rows)


def _holding_row(
    holding: HoldingSnapshotRecord,
    snapshot: PortfolioSnapshotRecord,
) -> dict[str, object]:
    row = _record_dict(holding)
    row["account_id"] = snapshot.account_id
    row["snapshot_date"] = snapshot.snapshot_date.isoformat()
    return row


def _records_frame(records: Sequence[Any]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([_record_dict(record) for record in records])


def _record_dict(record: Any) -> dict[str, object]:
    payload = asdict(record)
    for key, value in payload.items():
        if isinstance(value, date):
            payload[key] = value.isoformat()
    return payload


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result[columns]


def _bool_series(series: pd.Series) -> pd.Series:
    return series.map(_bool_value)


def _bool_value(value: object) -> bool:
    if pd.isna(cast(Any, value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)
