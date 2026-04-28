from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from typing import Any

import pandas as pd

from stock_analysis.forecasting.outcomes import attach_forecast_outcomes
from stock_analysis.storage.contracts import (
    AccountTrackingRepository,
    HoldingSnapshotRecord,
    PortfolioSnapshotRecord,
    RecommendationRunRecord,
)


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
    non_empty: dict[str, pd.DataFrame] = {}
    for name, table in tables.items():
        if table.empty:
            continue
        table["account_slug"] = account.slug
        non_empty[name] = table
    return non_empty


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
