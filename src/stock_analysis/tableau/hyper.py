from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def export_hyper_if_available(table: pd.DataFrame, path: Path) -> Path | None:
    try:
        from tableauhyperapi import (
            Connection,
            CreateMode,
            HyperProcess,
            Inserter,
            TableDefinition,
            TableName,
            Telemetry,
        )
    except ImportError:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    with (
        HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper,
        Connection(hyper.endpoint, path, CreateMode.CREATE_AND_REPLACE) as connection,
    ):
        connection.catalog.create_schema("Extract")
        definition = TableDefinition(
            table_name=TableName("Extract", "portfolio_dashboard_mart"),
            columns=[
                TableDefinition.Column(column, _sql_type_for_series(table[column]))
                for column in table.columns
            ],
        )
        connection.catalog.create_table(definition)
        rows = [
            tuple(None if pd.isna(value) else value for value in row)
            for row in table.itertuples(index=False, name=None)
        ]
        with Inserter(connection, definition) as inserter:
            inserter.add_rows(rows)
            inserter.execute()
    return path


def _sql_type_for_series(series: pd.Series):
    from tableauhyperapi import SqlType

    if pd.api.types.is_integer_dtype(series):
        return SqlType.big_int()
    if pd.api.types.is_float_dtype(series):
        return SqlType.double()
    if pd.api.types.is_bool_dtype(series):
        return SqlType.bool()
    non_null = series.dropna()
    if not non_null.empty and non_null.map(lambda value: isinstance(value, date)).all():
        return SqlType.date()
    return SqlType.text()
