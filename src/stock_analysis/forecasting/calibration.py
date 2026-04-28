from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Protocol

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
)
CALIBRATION_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "calibration_status",
    "calibration_method",
    "calibration_target",
    "calibration_horizon_days",
    "calibration_observations",
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
    splits: int,
    embargo_days: int,
    shrinkage: float,
    lookback_days: int | None = None,
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
    )
    usable = prediction_frame.dropna(subset=["forecast_score", "realized_return"]).copy()
    if len(usable) < min_observations:
        diagnostics = _diagnostics_frame(
            status="insufficient_observations",
            method=method,
            target_column=target_column,
            horizon_days=horizon_days,
            observations=len(usable),
            trained_through_date=_max_date_text(usable.get("date")),
        )
        return ForecastCalibrationResult(empty_latest, prediction_frame, diagnostics)

    from sklearn.isotonic import IsotonicRegression

    score = usable["forecast_score"].astype(float).to_numpy()
    target = usable["realized_return"].astype(float).to_numpy()
    target_mean = float(np.mean(target))
    calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
    calibrator.fit(score, target)

    usable["calibrated_expected_return"] = _shrink(
        np.asarray(calibrator.predict(score), dtype=float),
        target_mean=target_mean,
        shrinkage=shrinkage,
    )
    latest_scores = pd.to_numeric(latest_features["forecast_score"], errors="coerce").astype(float)
    latest_values = np.full(len(latest_scores), np.nan, dtype=float)
    finite_latest = np.isfinite(latest_scores.to_numpy(dtype=float))
    if finite_latest.any():
        latest_values[finite_latest] = _shrink(
            np.asarray(
                calibrator.predict(latest_scores.to_numpy(dtype=float)[finite_latest]),
                dtype=float,
            ),
            target_mean=target_mean,
            shrinkage=shrinkage,
        )
    calibrated_latest = pd.Series(
        latest_values,
        index=latest_features.index,
        name="calibrated_expected_return",
    )
    diagnostics = _calibrated_diagnostics(
        usable,
        method=method,
        target_column=target_column,
        horizon_days=horizon_days,
        target_mean=target_mean,
        shrinkage=shrinkage,
    )
    return ForecastCalibrationResult(calibrated_latest, usable, diagnostics)


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
) -> pd.DataFrame:
    train_panel = panel.merge(labels[["ticker", "date", target_column]], on=["ticker", "date"])
    train_panel = train_panel.loc[train_panel[target_column].notna()].copy()
    if lookback_days is not None and not train_panel.empty:
        max_date = pd.Timestamp(train_panel["date"].max())
        start_date = max_date - pd.Timedelta(days=int(lookback_days))
        train_panel = train_panel.loc[train_panel["date"] >= start_date].copy()
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
        rows.append(frame.drop(columns=[target_column]))

    if not rows:
        return _empty_predictions()
    result = pd.concat(rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.date.astype(str)
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def _shrink(values: np.ndarray, *, target_mean: float, shrinkage: float) -> np.ndarray:
    weight = float(np.clip(shrinkage, 0.0, 1.0))
    return (1 - weight) * values + weight * target_mean


def _calibrated_diagnostics(
    predictions: pd.DataFrame,
    *,
    method: str,
    target_column: str,
    horizon_days: int,
    target_mean: float,
    shrinkage: float,
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
        trained_through_date=_max_date_text(predictions["date"]),
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


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)
