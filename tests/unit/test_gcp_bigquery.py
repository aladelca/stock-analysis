from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from stock_analysis.config import GcpConfig
from stock_analysis.gcp import bigquery as bq


def test_publish_gold_tables_to_bigquery_deletes_existing_run_rows_and_loads(monkeypatch) -> None:
    client = FakeBigQueryClient()
    monkeypatch.setattr(bq, "_bigquery_modules", lambda: (FakeBigQueryModule, FakeNotFound))

    published = bq.publish_gold_tables_to_bigquery(
        {
            "portfolio_recommendations": pd.DataFrame(
                {"ticker": ["SPY"], "run_id": ["run-1"], "target_weight": [1.0]}
            ),
            "recommendation_lines_history": pd.DataFrame(
                {"ticker": ["SPY"], "recommendation_run_id": ["rec-run-1"]}
            ),
        },
        GcpConfig(
            enabled=True,
            project_id="project-1",
            bucket="bucket-1",
            bigquery_dataset_gold="gold",
        ),
        run_id="run-1",
        bigquery_client=client,
    )

    assert published == {
        "portfolio_recommendations": "project-1.gold.portfolio_recommendations",
        "recommendation_lines_history": "project-1.gold.recommendation_lines_history",
    }
    recommendations_staging = bq._staging_table_id(
        "project-1.gold.portfolio_recommendations",
        "run-1",
    )
    history_staging = bq._staging_table_id(
        "project-1.gold.recommendation_lines_history",
        "run-1",
    )
    assert client.queries == [
        (
            "create table if not exists `project-1.gold.portfolio_recommendations` as\n"
            f"select * from `{recommendations_staging}` where false;\n\n"
            "begin transaction;\n"
            "delete from `project-1.gold.portfolio_recommendations` where run_id = @run_id;\n"
            "insert into `project-1.gold.portfolio_recommendations` "
            f"select * from `{recommendations_staging}`;\n"
            "commit transaction;"
        ),
        f"drop table if exists `{recommendations_staging}`",
        (
            "create or replace table `project-1.gold.recommendation_lines_history` as\n"
            f"select * from `{history_staging}`;"
        ),
        f"drop table if exists `{history_staging}`",
    ]
    assert client.loaded[0][0] == recommendations_staging
    assert client.loaded[1][0] == history_staging
    assert client.loaded[0][2].write_disposition == "WRITE_TRUNCATE"
    assert client.loaded[1][2].write_disposition == "WRITE_TRUNCATE"
    assert "run_id" in client.loaded[1][1].columns
    assert client.loaded[1][1]["run_id"].iat[0] == "run-1"


def test_publish_gold_tables_to_bigquery_requires_project_id() -> None:
    try:
        bq.publish_gold_tables_to_bigquery(
            {"portfolio_recommendations": pd.DataFrame()},
            GcpConfig(enabled=True, bucket="bucket-1"),
            run_id="run-1",
            bigquery_client=FakeBigQueryClient(),
        )
    except ValueError as exc:
        assert "gcp.project_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


class FakeNotFound(Exception):
    pass


class FakeQueryJob:
    def result(self) -> None:
        return None


class FakeLoadJob:
    def result(self) -> None:
        return None


class FakeBigQueryClient:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.loaded: list[tuple[str, pd.DataFrame, object]] = []

    def query(self, query: str, job_config=None) -> FakeQueryJob:
        del job_config
        self.queries.append(query)
        return FakeQueryJob()

    def load_table_from_dataframe(self, frame: pd.DataFrame, table_id: str, job_config=None):
        self.loaded.append((table_id, frame.copy(), job_config))
        return FakeLoadJob()


class FakeBigQueryModule:
    QueryJobConfig = SimpleNamespace

    class ScalarQueryParameter:
        def __init__(self, name: str, type_: str, value: object) -> None:
            self.name = name
            self.type_ = type_
            self.value = value

    class LoadJobConfig:
        def __init__(self, write_disposition: str) -> None:
            self.write_disposition = write_disposition

    class WriteDisposition:
        WRITE_APPEND = "WRITE_APPEND"
        WRITE_TRUNCATE = "WRITE_TRUNCATE"
