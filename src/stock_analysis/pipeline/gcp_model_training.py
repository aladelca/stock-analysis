from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from stock_analysis.artifacts.store import write_table_with_csv
from stock_analysis.config import PortfolioConfig
from stock_analysis.forecasting.ml_forecast import (
    MLForecastModelArtifact,
    train_ml_forecast_model_artifact,
)
from stock_analysis.gcp.gcs_store import GcsArtifactStore
from stock_analysis.gcp.model_registry import GcsModelRegistry
from stock_analysis.ingestion.prices import PriceProvider
from stock_analysis.pipeline.one_shot import prepare_one_shot_medallion_data


@dataclass(frozen=True)
class GcpModelTrainingResult:
    run_id: str
    gcs_run_root: str
    model_uri: str
    production_model_uri: str | None
    artifact_uris: list[str]


def run_gcp_model_training(
    config: PortfolioConfig,
    *,
    universe_html: str | None = None,
    price_provider: PriceProvider | None = None,
    storage_client: object | None = None,
    promote: bool = True,
) -> GcpModelTrainingResult:
    """Train the ML forecast model and write a reusable artifact bundle to GCS."""

    if not config.gcp.enabled:
        msg = "Set gcp.enabled=true to run the GCP model training pipeline."
        raise ValueError(msg)
    if not config.gcp.bucket:
        msg = "Set gcp.bucket to run the GCP model training pipeline."
        raise ValueError(msg)
    if config.forecast.engine != "ml":
        msg = "GCP model training only supports forecast.engine=ml."
        raise ValueError(msg)

    run_id = config.run.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    store = GcsArtifactStore(
        bucket=config.gcp.bucket,
        prefix=config.gcp.gcs_prefix,
        run_id=run_id,
        storage_client=storage_client,
    )
    medallion = prepare_one_shot_medallion_data(
        config,
        store=store,
        universe_html=universe_html,
        price_provider=price_provider,
    )
    training = train_ml_forecast_model_artifact(
        medallion.feature_panel,
        medallion.labels_panel,
        config.forecast,
    )
    config_hash = _config_hash(config)
    registry = GcsModelRegistry(config.gcp, storage_client=storage_client)
    registry_uris = registry.write_artifact(
        training.artifact,
        run_id=run_id,
        config_hash=config_hash,
        promote=promote,
    )

    artifact_uris: list[str] = list(medallion.artifact_uris)
    artifact_uris.extend(registry_uris.values())
    artifact_uris.extend(
        write_table_with_csv(
            store,
            "gold",
            "forecast_calibration_diagnostics",
            training.calibration_diagnostics,
        )
    )
    artifact_uris.extend(
        write_table_with_csv(
            store,
            "gold",
            "forecast_calibration_predictions",
            training.calibration_predictions,
        )
    )
    artifact_uris.extend(
        write_table_with_csv(
            store,
            "gold",
            "model_metadata",
            _model_metadata_frame(
                training.artifact,
                run_id=run_id,
                config_hash=config_hash,
                model_uri=registry_uris["model"],
                production_model_uri=registry_uris.get("production_current"),
            ),
        )
    )

    return GcpModelTrainingResult(
        run_id=run_id,
        gcs_run_root=store.run_root_uri,
        model_uri=registry_uris["model"],
        production_model_uri=registry_uris.get("production_current"),
        artifact_uris=artifact_uris,
    )


def _model_metadata_frame(
    artifact: MLForecastModelArtifact,
    *,
    run_id: str,
    config_hash: str,
    model_uri: str,
    production_model_uri: str | None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model_uri": model_uri,
                "production_model_uri": production_model_uri or "",
                "model_version": artifact.model_version,
                "feature_columns": ",".join(artifact.feature_columns),
                "target_column": artifact.target_column,
                "horizon_days": artifact.horizon_days,
                "score_scale": artifact.score_scale,
                "trained_through_date": artifact.trained_through_date,
                "expected_return_is_calibrated": artifact.expected_return_is_calibrated,
                "calibration_status": artifact.calibration_status,
                "calibration_method": artifact.calibration_method,
                "calibration_target": artifact.calibration_target,
                "calibration_shrinkage": artifact.calibration_shrinkage,
                "created_at_utc": artifact.created_at_utc,
                "config_hash": config_hash,
            }
        ]
    )


def _config_hash(config: PortfolioConfig) -> str:
    config_json = config.model_dump_json(exclude={"run": {"as_of_date"}})
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()
