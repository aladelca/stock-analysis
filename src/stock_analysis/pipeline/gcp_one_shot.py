from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

from stock_analysis.config import PortfolioConfig
from stock_analysis.domain.models import PipelineResult
from stock_analysis.forecasting.ml_forecast import (
    MLForecastModelArtifact,
    expected_ml_horizon_days,
    expected_ml_target_column,
)
from stock_analysis.gcp.bigquery import publish_gold_tables_to_bigquery
from stock_analysis.gcp.gcs_store import GcsArtifactStore
from stock_analysis.gcp.model_registry import GcsModelRegistry
from stock_analysis.ingestion.prices import PriceProvider
from stock_analysis.pipeline.one_shot import (
    OneShotMedallionData,
    OneShotRunOutput,
    prepare_one_shot_medallion_data,
    run_one_shot_with_store,
)
from stock_analysis.storage.contracts import AccountTrackingRepository

BIGQUERY_GOLD_TABLES = (
    "optimizer_input",
    "price_coverage",
    "portfolio_recommendations",
    "portfolio_risk_metrics",
    "sector_exposure",
    "run_metadata",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GcpPipelineResult:
    pipeline: PipelineResult
    gcs_run_root: str
    bigquery_tables: dict[str, str]
    model_artifact_uri: str | None = None


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
    medallion_data: OneShotMedallionData | None = None
    ml_model_artifact: MLForecastModelArtifact | None = None
    ml_model_artifact_uri = None
    model_contract_status = "not_checked"
    model_contract_checked_at_utc = ""
    if config.forecast.engine == "ml":
        model_registry = GcsModelRegistry(config.gcp, storage_client=storage_client)
        ml_model_artifact_uri = config.gcp.model_artifact_uri or model_registry.default_model_uri()
        if not model_registry.exists(ml_model_artifact_uri):
            msg = (
                "No GCS ML model artifact found at "
                f"{ml_model_artifact_uri}. Run `stock-analysis train-gcp-model "
                "--config configs/portfolio.gcp.yaml --forecast-engine ml` first."
            )
            raise ValueError(msg)
        medallion_data = prepare_one_shot_medallion_data(
            config,
            store=store,
            universe_html=universe_html,
            price_provider=price_provider,
        )
        logger.info("Loading ML model artifact from %s", ml_model_artifact_uri)
        ml_model_artifact = model_registry.load_artifact(ml_model_artifact_uri)
        model_contract_checked_at_utc = datetime.now(UTC).isoformat()
        _validate_model_artifact_contract(
            config,
            ml_model_artifact_uri,
            ml_model_artifact,
            data_as_of_date=medallion_data.data_as_of_date,
            feature_panel=medallion_data.feature_panel,
        )
        model_contract_status = "passed"

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
        ml_model_artifact=ml_model_artifact,
        ml_model_artifact_uri=ml_model_artifact_uri,
        medallion_data=medallion_data,
        model_contract_status=model_contract_status,
        model_contract_checked_at_utc=model_contract_checked_at_utc,
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
        model_artifact_uri=ml_model_artifact_uri,
    )


def _bigquery_publish_tables(output: OneShotRunOutput) -> dict[str, pd.DataFrame]:
    tables = {
        name: output.gold_tables[name]
        for name in BIGQUERY_GOLD_TABLES
        if name in output.gold_tables
    }
    tables.update(output.tableau_tables)
    return tables


def _validate_model_artifact_contract(
    config: PortfolioConfig,
    model_artifact_uri: str,
    artifact: MLForecastModelArtifact,
    *,
    data_as_of_date: date,
    feature_panel: pd.DataFrame,
) -> None:
    if artifact.model_version != config.forecast.ml_model_version:
        msg = (
            f"Model artifact {model_artifact_uri} was trained for {artifact.model_version!r}, "
            f"but config.forecast.ml_model_version is {config.forecast.ml_model_version!r}."
        )
        raise ValueError(msg)
    expected_horizon = expected_ml_horizon_days(config.forecast)
    if artifact.horizon_days != expected_horizon:
        msg = (
            f"Model artifact {model_artifact_uri} has horizon_days={artifact.horizon_days}, "
            f"but the configured model expects {expected_horizon}."
        )
        raise ValueError(msg)
    expected_target = expected_ml_target_column(config.forecast)
    if artifact.target_column != expected_target:
        msg = (
            f"Model artifact {model_artifact_uri} targets {artifact.target_column!r}, "
            f"but the configured model expects {expected_target!r}."
        )
        raise ValueError(msg)
    if float(artifact.score_scale) != float(config.forecast.ml_score_scale):
        msg = (
            f"Model artifact {model_artifact_uri} uses score_scale={artifact.score_scale}, "
            f"but config.forecast.ml_score_scale is {config.forecast.ml_score_scale}."
        )
        raise ValueError(msg)
    trained_through = _date_from_artifact_field(
        artifact.trained_through_date,
        model_artifact_uri,
        "trained_through_date",
    )
    if trained_through > data_as_of_date and not config.gcp.allow_model_trained_after_data:
        msg = (
            f"Model artifact {model_artifact_uri} was trained through {trained_through}, "
            f"which is after inference data_as_of_date {data_as_of_date}."
        )
        raise ValueError(msg)
    missing_features = [
        column for column in artifact.feature_columns if column not in feature_panel
    ]
    if missing_features:
        msg = (
            f"Model artifact {model_artifact_uri} requires feature columns missing from "
            f"the inference feature panel: {missing_features}"
        )
        raise ValueError(msg)
    if config.forecast.ml_calibration_enabled and config.forecast.ml_use_calibrated_expected_return:
        if not artifact.expected_return_is_calibrated:
            msg = (
                f"Model artifact {model_artifact_uri} is not calibrated, but config requires "
                "calibrated expected returns."
            )
            raise ValueError(msg)
        if artifact.calibration_method != config.forecast.ml_calibration_method:
            msg = (
                f"Model artifact {model_artifact_uri} uses calibration method "
                f"{artifact.calibration_method!r}, but config expects "
                f"{config.forecast.ml_calibration_method!r}."
            )
            raise ValueError(msg)
        if artifact.calibration_target != config.forecast.ml_calibration_target:
            msg = (
                f"Model artifact {model_artifact_uri} calibrates target "
                f"{artifact.calibration_target!r}, but config expects "
                f"{config.forecast.ml_calibration_target!r}."
            )
            raise ValueError(msg)


def _date_from_artifact_field(value: str, model_artifact_uri: str, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        msg = f"Model artifact {model_artifact_uri} has invalid {field_name}: {value!r}"
        raise ValueError(msg) from exc
