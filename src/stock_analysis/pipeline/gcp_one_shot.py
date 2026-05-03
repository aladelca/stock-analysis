from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from stock_analysis.config import PortfolioConfig
from stock_analysis.domain.models import PipelineResult
from stock_analysis.gcp.bigquery import publish_gold_tables_to_bigquery
from stock_analysis.gcp.gcs_store import GcsArtifactStore
from stock_analysis.ingestion.prices import PriceProvider
from stock_analysis.pipeline.one_shot import OneShotRunOutput, run_one_shot_with_store
from stock_analysis.storage.contracts import AccountTrackingRepository

BIGQUERY_GOLD_TABLES = (
    "optimizer_input",
    "portfolio_recommendations",
    "portfolio_risk_metrics",
    "sector_exposure",
    "run_metadata",
)


@dataclass(frozen=True)
class GcpPipelineResult:
    pipeline: PipelineResult
    gcs_run_root: str
    bigquery_tables: dict[str, str]


def run_gcp_one_shot(
    config: PortfolioConfig,
    *,
    universe_html: str | None = None,
    price_provider: PriceProvider | None = None,
    account_repository: AccountTrackingRepository | None = None,
    storage_client: object | None = None,
    bigquery_client: object | None = None,
) -> GcpPipelineResult:
    """Run the one-shot pipeline with direct Cloud Storage artifacts."""

    if not config.gcp.enabled:
        msg = "Set gcp.enabled=true to run the GCP one-shot pipeline."
        raise ValueError(msg)
    if not config.gcp.bucket:
        msg = "Set gcp.bucket to run the GCP one-shot pipeline."
        raise ValueError(msg)

    run_id = config.run.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    store = GcsArtifactStore(
        bucket=config.gcp.bucket,
        prefix=config.gcp.gcs_prefix,
        run_id=run_id,
        storage_client=storage_client,
    )
    output = run_one_shot_with_store(
        config,
        store=store,
        universe_html=universe_html,
        price_provider=price_provider,
        account_repository=account_repository,
        export_hyper=False,
        log_mlflow=config.mlflow.enabled,
        write_tableau_dashboard_mart=True,
        include_account_history=True,
    )
    bigquery_tables: dict[str, str] = {}
    if config.gcp.publish_bigquery:
        bigquery_tables = publish_gold_tables_to_bigquery(
            _bigquery_publish_tables(output),
            config.gcp,
            run_id=run_id,
            bigquery_client=bigquery_client,
        )
    return GcpPipelineResult(
        pipeline=output.result,
        gcs_run_root=store.run_root_uri,
        bigquery_tables=bigquery_tables,
    )


def _bigquery_publish_tables(output: OneShotRunOutput) -> dict[str, pd.DataFrame]:
    tables = {
        name: output.gold_tables[name]
        for name in BIGQUERY_GOLD_TABLES
        if name in output.gold_tables
    }
    tables.update(output.tableau_tables)
    return tables
