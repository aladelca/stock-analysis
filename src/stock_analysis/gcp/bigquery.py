from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from stock_analysis.config import GcpConfig

RUN_SCOPED_TABLES: frozenset[str] = frozenset(
    {
        "optimizer_input",
        "portfolio_dashboard_mart",
        "tableau_dashboard_mart",
        "portfolio_recommendations",
        "portfolio_risk_metrics",
        "price_coverage",
        "sector_exposure",
        "run_metadata",
        "forecast_calibration_diagnostics",
        "forecast_calibration_predictions",
        "cashflows",
        "portfolio_snapshots",
        "holding_snapshots",
        "recommendation_runs",
        "recommendation_lines",
        "performance_snapshots",
    }
)
FULL_REFRESH_TABLES: frozenset[str] = frozenset(
    {
        "cashflows_history",
        "portfolio_snapshots_history",
        "holding_snapshots_history",
        "recommendation_runs_history",
        "recommendation_lines_history",
        "performance_snapshots_history",
    }
)


def publish_gold_tables_to_bigquery(
    tables: Mapping[str, pd.DataFrame],
    config: GcpConfig,
    *,
    run_id: str,
    bigquery_client: Any | None = None,
) -> dict[str, str]:
    """Publish gold/dashboard tables to managed BigQuery tables."""

    if not config.project_id:
        msg = "Set gcp.project_id before publishing to BigQuery."
        raise ValueError(msg)
    if not config.bigquery_dataset_gold:
        msg = "Set gcp.bigquery_dataset_gold before publishing to BigQuery."
        raise ValueError(msg)

    client = bigquery_client or _bigquery_client(config)
    published: dict[str, str] = {}
    for name, frame in tables.items():
        if frame is None:
            continue
        table_name = _safe_table_name(name)
        table_id = f"{config.project_id}.{config.bigquery_dataset_gold}.{table_name}"
        prepared = _prepare_frame(frame, run_id=run_id)
        if table_name in RUN_SCOPED_TABLES and "run_id" in prepared.columns:
            _publish_run_scoped_table(client, table_id, prepared, run_id)
        elif table_name in FULL_REFRESH_TABLES:
            _publish_full_refresh_table(client, table_id, prepared, run_id)
        else:
            _load_dataframe(client, table_id, prepared, write_disposition=_write_append())
        published[name] = table_id
    return published


def _prepare_frame(frame: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    prepared = frame.copy()
    if "run_id" not in prepared.columns:
        prepared["run_id"] = run_id
    return prepared


def _safe_table_name(name: str) -> str:
    table_name = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    table_name = re.sub(r"_+", "_", table_name).strip("_")
    if not table_name:
        msg = f"Cannot derive a BigQuery table name from {name!r}."
        raise ValueError(msg)
    if table_name[0].isdigit():
        table_name = f"t_{table_name}"
    return table_name


def _publish_run_scoped_table(
    client: Any,
    table_id: str,
    frame: pd.DataFrame,
    run_id: str,
) -> None:
    staging_table_id = _staging_table_id(table_id, run_id)
    _load_dataframe(client, staging_table_id, frame, write_disposition=_write_truncate())
    _ensure_target_table_schema(client, table_id, staging_table_id)
    column_list = _quoted_column_list(frame.columns)
    query = f"""
begin transaction;
delete from `{table_id}` where run_id = @run_id;
insert into `{table_id}` ({column_list})
select {column_list} from `{staging_table_id}`;
commit transaction;
""".strip()
    job_config = _run_id_query_job_config(run_id)
    try:
        client.query(query, job_config=job_config).result()
    finally:
        _drop_table(client, staging_table_id)


def _publish_full_refresh_table(
    client: Any,
    table_id: str,
    frame: pd.DataFrame,
    run_id: str,
) -> None:
    staging_table_id = _staging_table_id(table_id, run_id)
    _load_dataframe(client, staging_table_id, frame, write_disposition=_write_truncate())
    query = f"""
create or replace table `{table_id}` as
select * from `{staging_table_id}`;
""".strip()
    try:
        client.query(query).result()
    finally:
        _drop_table(client, staging_table_id)


def _staging_table_id(table_id: str, run_id: str) -> str:
    project_id, dataset_id, table_name = table_id.split(".", 2)
    suffix = hashlib.sha256(f"{table_id}:{run_id}".encode()).hexdigest()[:12]
    safe_run = _safe_table_name(run_id).lower()[:32]
    return f"{project_id}.{dataset_id}._staging_{table_name}_{safe_run}_{suffix}"


def _ensure_target_table_schema(client: Any, table_id: str, staging_table_id: str) -> None:
    _, not_found = _bigquery_modules()
    try:
        target = client.get_table(table_id)
    except not_found:
        client.query(
            f"create table `{table_id}` as select * from `{staging_table_id}` where false"
        ).result()
        return

    staging = client.get_table(staging_table_id)
    target_columns = {field.name for field in target.schema}
    missing_fields = [field for field in staging.schema if field.name not in target_columns]
    if not missing_fields:
        return

    target.schema = [*target.schema, *missing_fields]
    client.update_table(target, ["schema"])


def _quoted_column_list(columns: pd.Index) -> str:
    return ", ".join(_quote_identifier(str(column)) for column in columns)


def _quote_identifier(identifier: str) -> str:
    if "`" in identifier:
        msg = f"BigQuery identifier contains an unsupported backtick: {identifier!r}"
        raise ValueError(msg)
    return f"`{identifier}`"


def _run_id_query_job_config(run_id: str) -> Any:
    bigquery, _ = _bigquery_modules()
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("run_id", "STRING", run_id),
        ]
    )


def _drop_table(client: Any, table_id: str) -> None:
    query = f"drop table if exists `{table_id}`"
    client.query(query).result()


def _load_dataframe(
    client: Any,
    table_id: str,
    frame: pd.DataFrame,
    *,
    write_disposition: str,
) -> None:
    bigquery, _ = _bigquery_modules()
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
    client.load_table_from_dataframe(frame, table_id, job_config=job_config).result()


def _bigquery_client(config: GcpConfig) -> Any:
    bigquery, _ = _bigquery_modules()
    return bigquery.Client(project=config.project_id, location=config.bigquery_location)


def _bigquery_modules() -> tuple[Any, type[Exception]]:
    try:
        from google.api_core.exceptions import NotFound
        from google.cloud import bigquery
    except ImportError as exc:
        msg = (
            "BigQuery publishing requires the optional gcp extra. "
            "Run `uv sync --extra gcp` or prefix the command with `uv run --extra gcp`."
        )
        raise RuntimeError(msg) from exc
    return bigquery, NotFound


def _write_append() -> str:
    bigquery, _ = _bigquery_modules()
    return str(bigquery.WriteDisposition.WRITE_APPEND)


def _write_truncate() -> str:
    bigquery, _ = _bigquery_modules()
    return str(bigquery.WriteDisposition.WRITE_TRUNCATE)
