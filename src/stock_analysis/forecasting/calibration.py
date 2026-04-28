from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

CALIBRATION_PREDICTION_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "forecast_score",
    "realized_return",
    "train_cutoff_date",
    "calibration_fold_start_date",
    "calibrated_expected_return",
    "calibration_sample",
)
CALIBRATION_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "calibration_status",
    "calibration_method",
    "calibration_target",
    "calibration_horizon_days",
    "calibration_observations",
    "calibration_fit_observations",
    "calibration_total_observations",
    "calibration_validation_fraction",
    "calibration_trained_through_date",
    "calibration_mae",
    "calibration_rmse",
    "calibration_rank_ic",
    "calibration_target_mean",
    "calibration_prediction_mean",
    "calibration_score_min",
    "calibration_score_max",
    "calibration_shrinkage",
)


class PredictiveModel(Protocol):
    def predict(self, features: pd.DataFrame) -> list[float]: ...


ModelFactory = Callable[[pd.DataFrame], PredictiveModel]


@dataclass(frozen=True)
class ForecastCalibrationResult:
    calibrated_latest: pd.Series
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame

    @property
    def is_calibrated(self) -> bool:
        if self.diagnostics.empty or "calibration_status" not in self.diagnostics.columns:
            return False
        return str(self.diagnostics["calibration_status"].iat[0]) == "calibrated"


def calibrate_forecast_scores(
    *,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    latest_features: pd.DataFrame,
    model_factory: ModelFactory,
    feature_columns: tuple[str, ...],
    target_column: str,
    horizon_days: int,
    score_scale: float,
    method: str,
    min_observations: int,
    min_validation_observations: int,
    validation_fraction: float,
    min_rank_ic: float,
    max_mae: float | None,
    max_rmse: float | None,
    splits: int,
    embargo_days: int,
    shrinkage: float,
    lookback_days: int | None = None,
    max_assets_per_date: int | None = None,
) -> ForecastCalibrationResult:
    """Fit a leakage-safe score-to-return calibrator and apply it to latest scores."""

    empty_latest = pd.Series(np.nan, index=latest_features.index, name="calibrated_expected_return")
    if method != "isotonic":
        diagnostics = _diagnostics_frame(
            status="unsupported_method",
            method=method,
            target_column=target_column,
            horizon_days=horizon_days,
            observations=0,
            fit_observations=0,
            total_observations=0,
            validation_fraction=validation_fraction,
        )
        return ForecastCalibrationResult(empty_latest, _empty_predictions(), diagnostics)

    prediction_frame = build_oos_prediction_frame(
        panel=panel,
        labels=labels,
        model_factory=model_factory,
        feature_columns=feature_columns,
        target_column=target_column,
        score_scale=score_scale,
        splits=splits,
        embargo_days=embargo_days,
        lookback_days=lookback_days,
        max_assets_per_date=max_assets_per_date,
    )
    usable = prediction_frame.dropna(subset=["forecast_score", "realized_return"]).copy()
    if len(usable) < min_observations:
        diagnostics = _diagnostics_frame(
            status="insufficient_observations",
            method=method,
            target_column=target_column,
            horizon_days=horizon_days,
            observations=len(usable),
            fit_observations=0,
            total_observations=len(usable),
            validation_fraction=validation_fraction,
            trained_through_date=_max_date_text(usable.get("date")),
        )
        return ForecastCalibrationResult(empty_latest, prediction_frame, diagnostics)

    from sklearn.isotonic import IsotonicRegression

    fit_frame, validation_frame = _split_fit_validation(usable, validation_fraction)
    if len(validation_frame) < min_validation_observations:
        diagnostics = _diagnostics_frame(
            status="insufficient_validation_observations",
            method=method,
            target_column=target_column,
            horizon_days=horizon_days,
            observations=len(validation_frame),
            fit_observations=len(fit_frame),
            total_observations=len(usable),
            validation_fraction=validation_fraction,
            trained_through_date=_max_date_text(fit_frame.get("date")),
        )
        return ForecastCalibrationResult(empty_latest, prediction_frame, diagnostics)

    fit_score = fit_frame["forecast_score"].astype(float).to_numpy()
    fit_target = fit_frame["realized_return"].astype(float).to_numpy()
    target_mean = float(np.mean(fit_target))
    validation_calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
    validation_calibrator.fit(fit_score, fit_target)

    validation = validation_frame.copy()
    validation["calibrated_expected_return"] = _shrink(
        np.asarray(
            validation_calibrator.predict(validation["forecast_score"].astype(float).to_numpy()),
            dtype=float,
        ),
        target_mean=target_mean,
        shrinkage=shrinkage,
    )
    validation["calibration_sample"] = "validation"
    fit_artifact = fit_frame.copy()
    fit_artifact["calibrated_expected_return"] = _shrink(
        np.asarray(validation_calibrator.predict(fit_score), dtype=float),
        target_mean=target_mean,
        shrinkage=shrinkage,
    )
    fit_artifact["calibration_sample"] = "fit"
    predictions = pd.concat([fit_artifact, validation], ignore_index=True).sort_values(
        ["date", "ticker"]
    )
    predictions = predictions.reindex(columns=CALIBRATION_PREDICTION_COLUMNS)
    diagnostics = _calibrated_diagnostics(
        validation,
        method=method,
        target_column=target_column,
        horizon_days=horizon_days,
        target_mean=target_mean,
        shrinkage=shrinkage,
        fit_observations=len(fit_frame),
        total_observations=len(usable),
        validation_fraction=validation_fraction,
        trained_through_date=_max_date_text(usable["date"]),
    )
    if not _passes_quality_gates(
        diagnostics,
        min_rank_ic=min_rank_ic,
        max_mae=max_mae,
        max_rmse=max_rmse,
    ):
        diagnostics.loc[:, "calibration_status"] = "failed_quality_gate"
        return ForecastCalibrationResult(empty_latest, predictions, diagnostics)

    final_score = usable["forecast_score"].astype(float).to_numpy()
    final_target = usable["realized_return"].astype(float).to_numpy()
    final_target_mean = float(np.mean(final_target))
    final_calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
    final_calibrator.fit(final_score, final_target)
    latest_scores = pd.to_numeric(latest_features["forecast_score"], errors="coerce").astype(float)
    latest_values = np.full(len(latest_scores), np.nan, dtype=float)
    finite_latest = np.isfinite(latest_scores.to_numpy(dtype=float))
    if finite_latest.any():
        latest_values[finite_latest] = _shrink(
            np.asarray(
                final_calibrator.predict(latest_scores.to_numpy(dtype=float)[finite_latest]),
                dtype=float,
            ),
            target_mean=final_target_mean,
            shrinkage=shrinkage,
        )
    calibrated_latest = pd.Series(
        latest_values,
        index=latest_features.index,
        name="calibrated_expected_return",
    )
    return ForecastCalibrationResult(calibrated_latest, predictions, diagnostics)


def build_oos_prediction_frame(
    *,
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    model_factory: ModelFactory,
    feature_columns: tuple[str, ...],
    target_column: str,
    score_scale: float,
    splits: int,
    embargo_days: int,
    lookback_days: int | None = None,
    max_assets_per_date: int | None = None,
) -> pd.DataFrame:
    train_panel = panel.merge(labels[["ticker", "date", target_column]], on=["ticker", "date"])
    train_panel = train_panel.loc[train_panel[target_column].notna()].copy()
    if lookback_days is not None and not train_panel.empty:
        max_date = pd.Timestamp(train_panel["date"].max())
        start_date = max_date - pd.Timedelta(days=int(lookback_days))
        train_panel = train_panel.loc[train_panel["date"] >= start_date].copy()
    train_panel = _top_assets_by_date(train_panel, max_assets_per_date)
    dates = pd.DatetimeIndex(sorted(train_panel["date"].dropna().unique()))
    if len(dates) < splits + 1:
        return _empty_predictions()

    folds = [fold for fold in np.array_split(dates, splits + 1)[1:] if len(fold) > 0]
    rows: list[pd.DataFrame] = []
    for fold in folds:
        validation_dates = pd.DatetimeIndex(fold)
        fold_start = pd.Timestamp(validation_dates.min())
        train_cutoff = fold_start - pd.offsets.BDay(embargo_days)
        train = train_panel.loc[train_panel["date"] < train_cutoff].copy()
        validate = train_panel.loc[train_panel["date"].isin(validation_dates)].copy()
        if train.empty or validate.empty:
            continue
        model = model_factory(train)
        scores = np.asarray(model.predict(validate), dtype=float) * float(score_scale)
        if len(scores) != len(validate):
            msg = "calibration model returned a score count that does not match validation rows"
            raise ValueError(msg)
        frame = validate[["ticker", "date", target_column]].copy()
        frame["forecast_score"] = scores
        frame["realized_return"] = pd.to_numeric(frame[target_column], errors="coerce")
        frame["train_cutoff_date"] = pd.Timestamp(train_cutoff).date().isoformat()
        frame["calibration_fold_start_date"] = fold_start.date().isoformat()
        frame["calibration_sample"] = pd.NA
        rows.append(frame.drop(columns=[target_column]))

    if not rows:
        return _empty_predictions()
    result = pd.concat(rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.date.astype(str)
    return (
        result.sort_values(["date", "ticker"])
        .reset_index(drop=True)
        .reindex(columns=CALIBRATION_PREDICTION_COLUMNS)
    )


def _shrink(values: np.ndarray, *, target_mean: float, shrinkage: float) -> np.ndarray:
    weight = float(np.clip(shrinkage, 0.0, 1.0))
    return (1 - weight) * values + weight * target_mean


def _split_fit_validation(
    usable: pd.DataFrame,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(usable["date"], errors="coerce").dropna().unique())
    )
    if len(dates) < 2:
        return usable.iloc[0:0].copy(), usable.copy()
    validation_count = int(np.ceil(len(dates) * float(np.clip(validation_fraction, 0.01, 0.99))))
    validation_count = min(max(validation_count, 1), len(dates) - 1)
    validation_dates = set(dates[-validation_count:])
    date_values = pd.to_datetime(usable["date"], errors="coerce")
    validation = usable.loc[date_values.isin(validation_dates)].copy()
    fit = usable.loc[~date_values.isin(validation_dates)].copy()
    return fit, validation


def _top_assets_by_date(frame: pd.DataFrame, max_assets_per_date: int | None) -> pd.DataFrame:
    if max_assets_per_date is None or max_assets_per_date <= 0 or frame.empty:
        return frame
    if "dollar_volume_21d" not in frame.columns:
        return frame
    ranked = frame.assign(
        _calibration_liquidity=pd.to_numeric(frame["dollar_volume_21d"], errors="coerce")
    )
    return (
        ranked.sort_values(["date", "_calibration_liquidity"], ascending=[True, False])
        .groupby("date", group_keys=False)
        .head(max_assets_per_date)
        .drop(columns=["_calibration_liquidity"])
    )


def _passes_quality_gates(
    diagnostics: pd.DataFrame,
    *,
    min_rank_ic: float,
    max_mae: float | None,
    max_rmse: float | None,
) -> bool:
    if diagnostics.empty:
        return False
    row = diagnostics.iloc[0]
    rank_ic = _finite_or_none(row.get("calibration_rank_ic"))
    if rank_ic is None or rank_ic < min_rank_ic:
        return False
    mae = _finite_or_none(row.get("calibration_mae"))
    if max_mae is not None and (mae is None or mae > max_mae):
        return False
    rmse = _finite_or_none(row.get("calibration_rmse"))
    return not (max_rmse is not None and (rmse is None or rmse > max_rmse))


def _calibrated_diagnostics(
    predictions: pd.DataFrame,
    *,
    method: str,
    target_column: str,
    horizon_days: int,
    target_mean: float,
    shrinkage: float,
    fit_observations: int,
    total_observations: int,
    validation_fraction: float,
    trained_through_date: str | None,
) -> pd.DataFrame:
    expected = predictions["calibrated_expected_return"].astype(float)
    realized = predictions["realized_return"].astype(float)
    error = realized - expected
    return _diagnostics_frame(
        status="calibrated",
        method=method,
        target_column=target_column,
        horizon_days=horizon_days,
        observations=len(predictions),
        fit_observations=fit_observations,
        total_observations=total_observations,
        validation_fraction=validation_fraction,
        trained_through_date=trained_through_date,
        mae=float(error.abs().mean()),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        rank_ic=_finite_or_none(predictions["forecast_score"].corr(realized, method="spearman")),
        target_mean=target_mean,
        prediction_mean=float(expected.mean()),
        score_min=float(predictions["forecast_score"].min()),
        score_max=float(predictions["forecast_score"].max()),
        shrinkage=shrinkage,
    )


def _diagnostics_frame(
    *,
    status: str,
    method: str,
    target_column: str,
    horizon_days: int,
    observations: int,
    fit_observations: int | None = None,
    total_observations: int | None = None,
    validation_fraction: float | None = None,
    trained_through_date: str | None = None,
    mae: float | None = None,
    rmse: float | None = None,
    rank_ic: float | None = None,
    target_mean: float | None = None,
    prediction_mean: float | None = None,
    score_min: float | None = None,
    score_max: float | None = None,
    shrinkage: float | None = None,
) -> pd.DataFrame:
    row = {
        "calibration_status": status,
        "calibration_method": method,
        "calibration_target": target_column,
        "calibration_horizon_days": horizon_days,
        "calibration_observations": observations,
        "calibration_fit_observations": fit_observations,
        "calibration_total_observations": total_observations,
        "calibration_validation_fraction": validation_fraction,
        "calibration_trained_through_date": trained_through_date,
        "calibration_mae": mae,
        "calibration_rmse": rmse,
        "calibration_rank_ic": rank_ic,
        "calibration_target_mean": target_mean,
        "calibration_prediction_mean": prediction_mean,
        "calibration_score_min": score_min,
        "calibration_score_max": score_max,
        "calibration_shrinkage": shrinkage,
    }
    return pd.DataFrame([row], columns=list(CALIBRATION_DIAGNOSTIC_COLUMNS))


def _empty_predictions() -> pd.DataFrame:
    return empty_calibration_predictions()


def empty_calibration_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=list(CALIBRATION_PREDICTION_COLUMNS))


def disabled_calibration_diagnostics(
    *,
    method: str,
    target_column: str,
    horizon_days: int,
) -> pd.DataFrame:
    return _diagnostics_frame(
        status="disabled",
        method=method,
        target_column=target_column,
        horizon_days=horizon_days,
        observations=0,
        fit_observations=0,
        total_observations=0,
        validation_fraction=None,
    )


def _max_date_text(values: pd.Series | None) -> str | None:
    if values is None or values.empty:
        return None
    value = pd.to_datetime(values, errors="coerce").max()
    if pd.isna(value):
        return None
    if isinstance(value, date):
        return value.isoformat()
    return pd.Timestamp(value).date().isoformat()


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
