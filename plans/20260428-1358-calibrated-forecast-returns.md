# Calibrated Forecast Returns

## Goal

- Convert the current uncalibrated ML `forecast_score` into a calibrated 5-trading-day expected return that can be shown as a percentage, used by the optimizer in return units, and evaluated with valid `forecast_error` after the horizon closes.
- Preserve `forecast_score` as a ranking/confidence signal, but populate `calibrated_expected_return`, `expected_return`, and `expected_return_is_calibrated=true` when calibration passes minimum data and diagnostics gates.

## Request Snapshot

- User request: "lets do calibrated, what do we need" followed by "can you create a plan to do this?"
- Owner or issue: `stock-analysis-bvf`
- Plan file: `plans/20260428-1358-calibrated-forecast-returns.md`

## Current State

- `src/stock_analysis/forecasting/ml_forecast.py` trains the selected autoresearch candidate on all historical labels before the latest date, predicts the latest cross-section, and currently sets `expected_return = forecast_score`.
- The current live model marks `expected_return_is_calibrated = False`, so Tableau should not display `forecast_score` as an expected percent return.
- `src/stock_analysis/optimization/recommendations.py` already preserves `forecast_score`, `expected_return`, `calibrated_expected_return`, and `expected_return_is_calibrated` when they are present in `optimizer_input`.
- `src/stock_analysis/forecasting/outcomes.py` already computes `forecast_error` only when `calibrated_expected_return` exists or `expected_return_is_calibrated` is true.
- `src/stock_analysis/storage/contracts.py`, `src/stock_analysis/storage/supabase.py`, and `supabase/migrations/202604280003_tableau_forecast_semantics.sql` already support per-line `forecast_score`, `calibrated_expected_return`, and `expected_return_is_calibrated`.
- Run-level metadata currently records `expected_return_is_calibrated`, but it does not record calibration method, target, trained-through date, sample size, MAE, RMSE, or rank IC.
- The optimizer covariance in `src/stock_analysis/forecasting/ml_forecast.py` is annualized while the desired calibrated return is a 5-day return. The implementation must align expected-return and covariance units before promotion.

## Findings

- Calibration must use leakage-safe historical out-of-sample predictions. Fitting a calibrator on in-sample predictions from the same model used to train those predictions would overstate confidence and corrupt `forecast_error`.
- `scikit-learn` is already a core dependency, so `sklearn.isotonic.IsotonicRegression` is available for a monotonic score-to-return calibrator.
- The line-level Tableau/Supabase contract is mostly ready. The missing persistence work is run-level calibration diagnostics plus an optional expected active return field.
- The existing `forecast_score` should remain visible because calibrated expected return can be noisy; Tableau should show both score rank and expected percent return.

## Scope

### In scope

- Add forecast calibration configuration to `ForecastConfig`.
- Add a calibration module that builds leakage-safe historical OOS prediction pairs and fits a monotonic calibrator.
- Integrate calibration into the ML one-shot forecast path.
- Populate calibrated forecast columns in `optimizer_input`, `portfolio_recommendations`, local gold artifacts, Supabase, and Tableau history marts.
- Add run-level calibration diagnostics to metadata and Supabase.
- Align optimizer expected-return and covariance units for the calibrated 5-day horizon.
- Add unit, integration, storage, and Tableau mart tests.
- Update runbooks and methodology docs so Tableau labels calibrated forecasts correctly.

### Out of scope

- Broker integration or automatic trade execution tracking. That is covered by `stock-analysis-2bt`.
- Claiming the strategy statistically beats SPY. Calibration must be validated separately before promotion claims.
- Replacing the selected model family. This calibrates the existing selected candidate first.
- Building new Tableau dashboards. This only ensures the data contract is ready.
- Remote Supabase deployment. The implementation should include migrations, but applying them remotely remains an operator action.

## File Plan

| Path | Action | Details |
| --- | --- | --- |
| `src/stock_analysis/config.py` | modify | Add calibration fields to `ForecastConfig`: `ml_calibration_enabled`, `ml_calibration_method`, `ml_calibration_target`, `ml_calibration_min_observations`, `ml_calibration_lookback_days`, `ml_calibration_splits`, `ml_calibration_shrinkage`, and `ml_use_calibrated_expected_return`. |
| `configs/portfolio.yaml` | modify | Set production defaults explicitly. Recommended rollout: enable calibration only after tests and validation pass; keep the flags visible in config. |
| `src/stock_analysis/forecasting/calibration.py` | create | Implement `ForecastCalibrator`, OOS prediction frame construction, isotonic fit, shrinkage, diagnostics, and safe fallback behavior. |
| `src/stock_analysis/forecasting/ml_forecast.py` | modify | Generate latest `forecast_score`, fit/apply calibrator when enabled, set `calibrated_expected_return`, set calibrated `expected_return`, expose calibration artifacts, and build 5-day covariance when calibrated return units are used. |
| `src/stock_analysis/forecasting/outcomes.py` | modify | Keep current behavior, but add tests or small helpers if needed so forecast errors use `calibrated_expected_return` exactly. |
| `src/stock_analysis/optimization/recommendations.py` | modify | Ensure calibrated columns remain stable in recommendation output and optionally add `calibrated_expected_active_return` if added to the contract. |
| `src/stock_analysis/pipeline/one_shot.py` | modify | Write calibration artifacts, extend `run_metadata`, persist run-level metadata, and pass calibrated optimizer input through the existing recommendation path. |
| `src/stock_analysis/tableau/account_tracking_marts.py` | modify | Add calibration metadata and optional expected active return fields to recommendation run/line marts. |
| `src/stock_analysis/tableau/account_history_marts.py` | modify | Ensure historical recommendation exports include calibration metadata and do not overwrite calibrated fields when reading Supabase history. |
| `src/stock_analysis/tableau/dashboard_mart.py` | modify | Include calibrated expected return and calibration flags in the dashboard mart. |
| `src/stock_analysis/tableau/workbook.py` | modify | Rename or add Tableau field captions so calibrated return is displayed as an expected 5-day return only when calibrated. |
| `src/stock_analysis/storage/contracts.py` | modify | Add run-level calibration fields to `RecommendationRunRecord`; add optional `calibrated_expected_active_return` to `RecommendationLineRecord` if implemented. |
| `src/stock_analysis/storage/supabase.py` | modify | Map new run-level and line-level calibration fields in Supabase row readers/writers. |
| `supabase/migrations/202604280004_forecast_calibration_metadata.sql` | create | Add calibration metadata columns, constraints, and indexes to `recommendation_runs`; add line-level expected active return if selected. |
| `src/stock_analysis/ml/mlflow_tracking.py` | modify | Log calibration params and diagnostics: method, target, trained-through date, observations, MAE, RMSE, rank IC, and enabled flag. |
| `src/stock_analysis/ml/autoresearch_eval.py` | modify | Add optional calibration diagnostics to evaluation output or create a separate calibration validation command if the existing harness should remain stable. |
| `docs/forecasting-optimization-methodology.md` | modify | Document the calibration method, leakage controls, unit alignment, and validation gates. |
| `runbooks/full-execution.md` | modify | Document calibrated forecast outputs and the operator validation command. |
| `runbooks/supabase-account-tracking.md` | modify | Document Supabase fields and how Tableau should display calibrated return versus score. |
| `docs/tableau-dashboard-design.md` | modify | Update field semantics: `forecast_score` is rank/confidence; `calibrated_expected_return` is expected 5-day percentage only when calibrated. |
| `tests/unit/test_forecast_calibration.py` | create | Cover monotonic calibration, shrinkage, missing data fallback, minimum observation gate, diagnostics, and no future-label inclusion. |
| `tests/unit/test_ml_forecast.py` | modify | Cover calibrated and uncalibrated ML optimizer input behavior. |
| `tests/unit/test_forecast_outcomes.py` | modify | Confirm `forecast_error` is populated only from calibrated expected return. |
| `tests/unit/test_account_tracking_marts.py` | modify | Confirm calibration metadata and calibrated fields are included in Tableau-ready marts. |
| `tests/unit/test_supabase_storage.py` | modify | Confirm new Supabase fields round-trip through repository mappings. |
| `tests/integration/test_one_shot_pipeline.py` | modify | Validate one-shot calibrated run writes calibration artifacts, metadata, recommendations, and persisted repository records. |

## Data and Contract Changes

- Add local gold artifacts:
  - `gold/forecast_calibration_predictions.parquet`
  - `gold/forecast_calibration_diagnostics.parquet`
  - CSV mirrors under `gold/csv/`
- Add run metadata fields:
  - `calibration_enabled`
  - `calibration_method`
  - `calibration_target`
  - `calibration_model_version`
  - `calibration_trained_through_date`
  - `calibration_observations`
  - `calibration_mae`
  - `calibration_rmse`
  - `calibration_rank_ic`
  - `calibration_status`
- Keep existing line fields:
  - `forecast_score`
  - `expected_return`
  - `calibrated_expected_return`
  - `expected_return_is_calibrated`
  - `forecast_horizon_days`
  - `forecast_start_date`
  - `forecast_end_date`
  - `forecast_error`
- Add optional line field if the dashboard needs it:
  - `calibrated_expected_active_return`
- Supabase migration should be additive and backward compatible. Existing recommendation rows remain valid with `expected_return_is_calibrated=false` and null calibration metadata.

## Implementation Steps

1. Add calibration config to `ForecastConfig`.
   - Include explicit defaults and validation constraints.
   - Keep `ml_calibration_enabled` default false until validation proves acceptable diagnostics.
   - Add production config values in `configs/portfolio.yaml`.

2. Create `src/stock_analysis/forecasting/calibration.py`.
   - Define dataclasses for `CalibrationResult`, `CalibrationDiagnostics`, and `CalibrationPredictionFrame`.
   - Build historical OOS predictions using expanding or rolling walk-forward splits.
   - Ensure each prediction row at date `t` is produced by a model trained only on dates `< t - embargo`.
   - Fit `IsotonicRegression(out_of_bounds="clip")` from `forecast_score` to realized target.
   - Apply shrinkage toward the historical target mean to avoid overconfident tails.
   - Produce diagnostics: observations, target mean, prediction mean, MAE, RMSE, Spearman rank IC, calibration bucket table, min/max score, and trained-through date.

3. Integrate calibration into `build_ml_optimizer_inputs`.
   - Preserve raw `forecast_score`.
   - When calibration is enabled and diagnostics pass minimum gates, set:
     - `calibrated_expected_return = calibrator.predict(forecast_score)`
     - `expected_return = calibrated_expected_return`
     - `expected_return_is_calibrated = True`
   - When gates fail, leave existing uncalibrated behavior and set `calibration_status` to a clear value such as `insufficient_observations`.
   - Expose calibration artifacts to `one_shot`. Prefer a small result dataclass or a new `build_ml_optimizer_inputs_with_artifacts` wrapper so existing two-value call sites can remain compatible.

4. Align optimizer units.
   - When `expected_return_is_calibrated=true`, estimate covariance on the same 5-trading-day horizon or scale daily covariance to the horizon consistently.
   - Revisit `risk_aversion`, `lambda_turnover`, and `commission_rate` behavior under 5-day return units.
   - Add a config or metadata flag that records `optimizer_return_unit = 5d_return`.

5. Extend one-shot artifact writing.
   - Write calibration predictions and diagnostics to gold plus CSV mirrors.
   - Add calibration fields to `run_metadata`.
   - Log calibration fields to MLflow through `src/stock_analysis/ml/mlflow_tracking.py`.

6. Extend Supabase persistence.
   - Add migration `202604280004_forecast_calibration_metadata.sql`.
   - Update `RecommendationRunRecord` and Supabase mappings.
   - Keep line-level calibrated fields using existing columns; add optional expected active return only if accepted in implementation.

7. Extend Tableau-ready marts.
   - Add calibration run fields to `recommendation_runs`, `recommendation_runs_history`, and dashboard mart.
   - Ensure `recommendation_lines_history` keeps calibrated values from Supabase and can compute realized forecast error after the horizon.

8. Update docs and runbooks.
   - Explain the difference between score, calibrated expected return, and realized return.
   - Add a Tableau labeling rule: only format `calibrated_expected_return` as expected percentage when `expected_return_is_calibrated=true`.
   - Document that forecasts remain pending until `forecast_end_date`.

9. Validate on historical data before promotion.
   - Run a calibration-enabled backtest or autoresearch evaluation.
   - Confirm calibration buckets are monotonic enough, forecast error is centered, and post-cost strategy still clears the SPY gate on point estimates at minimum.
   - Do not mark production as statistically superior unless the existing promotion gate is met.

## Tests

- Unit: `tests/unit/test_forecast_calibration.py`
  - Fit isotonic calibration on synthetic ordered scores.
  - Verify predictions are monotonic and finite.
  - Verify shrinkage pulls predictions toward the target mean.
  - Verify insufficient observations returns disabled/fallback diagnostics.
  - Verify OOS frame builder does not use labels from the prediction date or later.
- Unit: `tests/unit/test_ml_forecast.py`
  - With calibration disabled, current uncalibrated behavior remains unchanged.
  - With calibration enabled and enough history, `expected_return_is_calibrated` is true and `expected_return == calibrated_expected_return`.
  - With calibration enabled but insufficient data, the pipeline keeps score behavior and records a failed calibration status.
- Unit: `tests/unit/test_forecast_outcomes.py`
  - Realized forecast error uses `calibrated_expected_return`.
  - Uncalibrated rows keep `forecast_error` null.
- Unit: `tests/unit/test_account_tracking_marts.py`
  - Recommendation run and line marts include calibration metadata and calibrated fields.
- Unit: `tests/unit/test_supabase_storage.py`
  - New run-level calibration fields round-trip through fake Supabase rows.
- Integration: `tests/integration/test_one_shot_pipeline.py`
  - ML one-shot with calibration writes `forecast_calibration_predictions`, `forecast_calibration_diagnostics`, calibrated recommendations, run metadata, and repository records.
- Regression:
  - Existing uncalibrated one-shot tests continue to pass with default calibration disabled.
  - Existing Tableau export history remains stable when calibration fields are null.

## Validation

- Format: `uv run ruff format --check src tests`
- Lint: `uv run ruff check src tests`
- Types: `uv run mypy src`
- Focused tests:
  - `uv run pytest tests/unit/test_forecast_calibration.py tests/unit/test_ml_forecast.py tests/unit/test_forecast_outcomes.py tests/unit/test_account_tracking_marts.py tests/unit/test_supabase_storage.py tests/integration/test_one_shot_pipeline.py -q`
- Full tests:
  - `OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --extra pytorch pytest -q`
- Local pipeline smoke test:
  - `uv run --extra supabase --extra tableau --extra mlflow stock-analysis run-one-shot --config configs/portfolio.local.yaml --forecast-engine ml`
- Post-run checks:
  - Inspect `gold/optimizer_input.parquet` for `expected_return_is_calibrated=true`.
  - Inspect `gold/portfolio_recommendations.parquet` for non-null `calibrated_expected_return`.
  - Inspect `gold/forecast_calibration_diagnostics.parquet` for finite observations, MAE, RMSE, and rank IC.

## Risks and Mitigations

- Leakage in calibration data -> Build OOS predictions with explicit train cutoff and add tests that fail if future labels are included.
- Overconfident expected returns -> Use shrinkage, winsorization, bucket diagnostics, and a minimum-observation gate.
- Unit mismatch in optimizer -> Record expected-return unit and align covariance to the same 5-day horizon before enabling calibrated optimization.
- Sparse or unstable calibration windows -> Fall back to uncalibrated status rather than emitting false precision.
- Tableau mislabeling -> Add explicit fields and docs so score and calibrated return are not confused.
- Supabase migration drift -> Keep migrations additive and make null metadata valid for old runs.

## Open Questions

- None

## Acceptance Criteria

- A calibration-enabled ML run can emit non-null `calibrated_expected_return` and `expected_return_is_calibrated=true` for eligible recommendation lines.
- `forecast_score` remains available and unchanged as the ranking signal.
- `forecast_error` is populated only after `forecast_end_date` and only when a calibrated expected return exists.
- Run metadata and Supabase recommendation runs include calibration method, target, trained-through date, observations, and diagnostics.
- Tableau exports include stable calibration fields for current and historical recommendation views.
- Existing uncalibrated behavior remains available through config.
- Validation commands pass.

## Definition of Done

- Code implemented for calibration config, calibrator, ML forecast integration, artifact writing, persistence, and Tableau marts.
- Supabase migration added and compatible with existing local data.
- Unit and integration tests added or updated.
- Runbooks and methodology docs updated.
- `uv run ruff format --check src tests`, `uv run ruff check src tests`, `uv run mypy src`, and full pytest pass.
- Plan updated if implementation scope changes.
