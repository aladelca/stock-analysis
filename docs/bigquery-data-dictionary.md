# `stock_analysis_gold` BigQuery Data Dictionary

This document describes only the BigQuery dataset used by the stock-analysis cloud pipeline:

```text
proyectodata-348005.stock_analysis_gold
```

It covers every visible table in `stock_analysis_gold`, including table purpose, granularity,
dashboard use, reason for inclusion, field definitions, and the type of information each table
contains.

Metadata was inspected on May 7, 2026 with `bq ls`, `bq show`, and aggregate `bq query` commands.
This document is based on schemas, row counts, and aggregate counts. It does not rely on row-level
sample data.

## Executive Summary

`stock_analysis_gold` is the Tableau-facing BigQuery dataset produced by the GCP stock-analysis
pipeline. It contains the current cloud outputs for recommendations, optimizer inputs, model
calibration, data coverage, risk metrics, sector exposure, and run audit metadata.

| Dataset | Location | Tables | Main purpose |
| --- | --- | ---: | --- |
| `stock_analysis_gold` | `US` | 9 | Stock-analysis Tableau output layer |

## Important Findings

| Finding | Impact |
| --- | --- |
| `stock_analysis_gold` has partial run coverage across tables. | Some tables have 4 run IDs, others 2 or 3, because schema/publishing changed over time. Use `run_id` filters carefully. |
| Some semantically float or boolean columns are typed as `INTEGER` in BigQuery because current values are null or empty. | Tableau may treat forecast outcome/account fields incorrectly until schemas are stabilized. |
| None of the visible tables are partitioned or clustered. | Fine at current scale, but not ideal as run history grows. |

## Dataset Inventory

Dataset: `proyectodata-348005.stock_analysis_gold`

| Property | Value |
| --- | --- |
| Project | `proyectodata-348005` |
| Dataset | `stock_analysis_gold` |
| Location | `US` |
| Visible tables | 9 |
| Approx rows | 837,700+ |
| Main consumer | Tableau |
| Main grain patterns | run, run-ticker, run-sector, run-metric, run-ticker-date |
| Main information | Recommendations, forecast signals, calibrated returns, optimizer inputs, risk metrics, sector exposure, data coverage, run metadata |

### Stock-Analysis Run Coverage

`stock_analysis_gold` currently contains these run IDs across tables:

| Run ID | Notes |
| --- | --- |
| `20260503T160205Z` | Older inference output; missing newer price-coverage/model-contract fields |
| `20260503T171819Z` | Partial current-run tables |
| `20260503T172406Z` | Partial current-run tables |
| `20260503T174853Z` | Latest complete published run observed in `run_metadata` |

Latest complete run metadata observed:

| Field | Value |
| --- | --- |
| `run_id` | `20260503T174853Z` |
| `requested_as_of_date` | `2026-05-03` |
| `data_as_of_date` | `2026-05-01` |
| `model_version` | `lightgbm_return_zscore` |
| `expected_return_is_calibrated` | `true` |
| `optimizer_return_unit` | `5d_return` |
| `calibration_status` | `calibrated` |
| `price_coverage_ratio` | `0.998015873015873` |
| `benchmark_price_status` | `ok` |
| `model_contract_status` | `passed` |

Latest complete run aggregate checks:

| Table | Aggregate |
| --- | --- |
| `price_coverage` | 503 `ok`, 1 `no_latest_feature` for run `20260503T174853Z` |
| `portfolio_recommendations` | 14 BUY, 106 HOLD, 383 EXCLUDE for run `20260503T174853Z` |
| `portfolio_risk_metrics` | Expected return 0.017884, expected volatility 0.033444, 120 holdings |

## Dataset Purpose

Purpose: current cloud output layer for the stock-analysis pipeline.

Location: `US`

Why it exists:

- It turns Cloud Run output into queryable dashboard tables.
- It separates Tableau serving from Cloud Storage medallion files.
- It provides run-level audit, data quality, forecast, optimizer, recommendation, and risk outputs.

Current tables:

| Table | Rows | Grain | Use | Why included |
| --- | ---: | --- | --- | --- |
| `portfolio_dashboard_mart` | 1,006 | One ticker per dashboard run | Main Tableau wide table | Convenience mart for dashboards |
| `portfolio_recommendations` | 2,012 | One ticker per run | Recommendation and executable trade detail | Canonical recommendation lines |
| `optimizer_input` | 2,012 | One candidate ticker per run | Forecast, eligibility, and feature inspection | Explains model/optimizer inputs |
| `price_coverage` | 1,512 | One requested ticker per run | Data quality/freshness dashboard | Explains missing or stale tickers |
| `run_metadata` | 2 | One row per complete run | Run audit and configuration metadata | Explains how each run was produced |
| `portfolio_risk_metrics` | 20 | One metric per run | Portfolio risk/summary KPIs | Gives portfolio-level risk context |
| `sector_exposure` | 48 | One sector per run | Sector allocation charts | Shows target concentration by sector |
| `forecast_calibration_diagnostics` | 2 | One diagnostic row per run | Forecast calibration status | Validates return forecast semantics |
| `forecast_calibration_predictions` | 831,096 | One ticker-date calibration row per run | Calibration review and model monitoring | Shows how score-to-return calibration behaved historically |

### Recommended Tableau Usage

| Dashboard need | Primary table | Join/filter guidance |
| --- | --- | --- |
| Current recommendation view | `portfolio_dashboard_mart` | Filter to latest `run_id` |
| Trade list | `portfolio_recommendations` | Filter `action in ('BUY', 'SELL', 'HOLD')` and latest `run_id` |
| Model/debug view | `optimizer_input` | Join/filter by `run_id`, `ticker` |
| Data freshness view | `price_coverage` | Filter latest `run_id`; group by `coverage_status` |
| Run selector/status | `run_metadata` | Use `run_id`, `data_as_of_date`, `calibration_status` |
| Calibration dashboard | `forecast_calibration_diagnostics` and `forecast_calibration_predictions` | Filter by `run_id`, date, ticker |
| Sector exposure | `sector_exposure` | Filter latest `run_id` |

### `stock_analysis_gold.portfolio_dashboard_mart`

Granularity: one row per ticker per dashboard run.

Use: primary wide Tableau table for portfolio recommendations. It denormalizes recommendation,
forecast, risk, account, and run metadata fields into one table.

Why included: Tableau is easier to build when the primary dashboard can read one wide table instead
of joining several tables for every view.

Information available:

- Current recommendation per ticker.
- Target/current/trade weights and dollar notionals.
- Forecast score, calibrated expected return, SPY-relative gate status.
- Forecast horizon and pending/realized outcome fields.
- Portfolio-level risk metrics repeated on each row.
- Run metadata and calibration status repeated on each row.
- Account performance fields when live account data exists.

Field groups:

| Fields | Type pattern | Description |
| --- | --- | --- |
| `run_id`, `as_of_date`, `ticker`, `security`, `gics_sector` | STRING/DATE | Row identity, run identity, asset identity, and sector classification |
| `forecast_score`, `expected_return_is_calibrated`, `calibrated_expected_return` | FLOAT/BOOLEAN | Model score and calibrated 5-day expected return semantics |
| `benchmark_expected_return`, `benchmark_expected_return_margin`, `benchmark_return_gate_passed` | FLOAT/BOOLEAN | SPY-relative expected-return gate fields |
| `forecast_horizon_days`, `forecast_start_date`, `forecast_end_date` | INTEGER/DATE | Forecast window definition |
| `realized_return`, `realized_spy_return`, `realized_active_return`, `forecast_error`, `forecast_hit`, `outcome_status` | Semantically FLOAT/BOOLEAN/STRING | Forecast outcome tracking; currently mostly pending/null in cloud output |
| `volatility` | FLOAT | Risk estimate used by optimizer |
| `current_weight`, `display_current_weight`, `current_weight_label` | FLOAT/STRING | Current allocation fields for Tableau display |
| `target_weight`, `executable_target_weight`, `display_target_weight`, labels | FLOAT/STRING | Target allocation fields after optimization and executable trade constraints |
| `trade_weight`, `display_trade_weight`, `trade_abs_weight`, `trade_weight_label` | FLOAT/STRING | Weight-level buy/sell/hold change |
| `rebalance_required`, `no_trade_band_applied`, `is_solver_dust`, `selected` | BOOLEAN | Dashboard flags for filtering and visual cleanup |
| `estimated_commission_weight`, `net_trade_weight_after_commission`, `cash_required_weight`, `cash_released_weight` | FLOAT | Cost and cash impact at weight level |
| `portfolio_value_before_contribution`, `contribution_amount`, `portfolio_value_after_contribution` | FLOAT | Portfolio value before/after configured contribution |
| `current_market_value`, `target_market_value`, `executable_target_market_value` | FLOAT | Dollar market value fields |
| `trade_notional`, `trade_notional_label`, `commission_amount`, `commission_amount_label` | FLOAT/STRING | Dollar trade and commission fields |
| `deposit_used_amount`, `cash_after_trade_amount` | FLOAT | How much deposit/cash is consumed and cash left after planned trades |
| `action`, `reason_code` | STRING | Recommendation action and reason for that action |
| `sector_target_weight` | FLOAT | Total target weight for the ticker's sector |
| `portfolio_expected_return`, `portfolio_expected_volatility`, `portfolio_return_per_vol` | FLOAT | Portfolio-level model/risk metrics repeated on each row |
| `portfolio_num_holdings`, `portfolio_max_weight`, `portfolio_concentration_hhi` | FLOAT | Portfolio concentration and count metrics |
| `account_*`, `spy_*`, `active_*` | Semantically FLOAT | Account performance fields; only meaningful when live account history is present |
| `run_*` fields | STRING/DATE/BOOLEAN/FLOAT | Run-level metadata copied from `run_metadata` |
| `is_data_date_lagged`, `data_date_status` | BOOLEAN/STRING | Whether requested run date differs from true market data date |

Important caveat:

- In the current BigQuery schema, several forecast outcome and account fields are typed as
  `INTEGER` because current rows are null. Semantically, realized returns and account returns should
  be FLOAT, and `forecast_hit` should be BOOLEAN. This should be fixed with explicit BigQuery schema
  loading before those fields become heavily used in Tableau.

### `stock_analysis_gold.portfolio_recommendations`

Granularity: one ticker per run.

Use: canonical recommendation-line table. Use this for trade lists, action counts, and ticker-level
recommendation audit.

Why included: it is narrower and more canonical than `portfolio_dashboard_mart`; it should be the
source for recommendation details.

| Column | Type | Description |
| --- | --- | --- |
| `ticker` | STRING | Portfolio/recommendation ticker |
| `security` | STRING | Security name |
| `gics_sector` | STRING | Sector classification |
| `forecast_score` | FLOAT | Raw model score or ranking signal |
| `expected_return` | FLOAT | Expected return used by optimizer; calibrated return when calibration passed |
| `calibrated_expected_return` | FLOAT | Calibrated forecast return over the model horizon |
| `expected_return_is_calibrated` | BOOLEAN | Whether expected return is in return units |
| `benchmark_expected_return` | FLOAT | SPY expected return used for active-name gate |
| `benchmark_expected_return_margin` | FLOAT | Required active-return margin over SPY |
| `benchmark_return_gate_passed` | BOOLEAN | Whether ticker passed the SPY-relative gate |
| `volatility` | FLOAT | Risk estimate used by optimizer |
| `current_weight` | FLOAT | Current portfolio weight after contribution context |
| `target_weight` | FLOAT | Optimized target weight |
| `trade_weight` | FLOAT | Target weight minus current weight |
| `trade_abs_weight` | FLOAT | Absolute trade weight |
| `current_weight_label` | STRING | Formatted current weight |
| `target_weight_label` | STRING | Formatted target weight |
| `trade_weight_label` | STRING | Formatted signed trade weight |
| `estimated_commission_weight` | FLOAT | Commission estimate as weight |
| `estimated_commission_weight_label` | STRING | Formatted commission weight |
| `net_trade_weight_after_commission` | FLOAT | Trade weight after commission adjustment |
| `cash_required_weight` | FLOAT | Cash required for buys as weight |
| `cash_released_weight` | FLOAT | Cash released by sells as weight |
| `portfolio_value_before_contribution` | FLOAT | Portfolio value before deposit/contribution |
| `contribution_amount` | FLOAT | Deposit/contribution used in the run |
| `portfolio_value_after_contribution` | FLOAT | Portfolio value after contribution |
| `current_market_value` | FLOAT | Current dollar market value |
| `target_market_value` | FLOAT | Target dollar market value |
| `executable_target_weight` | FLOAT | Target weight after cash/no-trade execution constraints |
| `executable_target_market_value` | FLOAT | Executable target dollar value |
| `trade_notional` | FLOAT | Dollar buy/sell notional |
| `commission_amount` | FLOAT | Dollar commission estimate |
| `deposit_used_amount` | FLOAT | Deposit dollars allocated to this buy |
| `cash_after_trade_amount` | FLOAT | Cash left after planned trade set |
| `no_trade_band_applied` | BOOLEAN | Whether trade was suppressed by no-trade band |
| `rebalance_required` | BOOLEAN | Whether trade exceeds rebalance threshold |
| `action` | STRING | BUY, SELL, HOLD, or EXCLUDE |
| `reason_code` | STRING | Explanation for action |
| `as_of_date` | STRING | Market data date for the recommendation |
| `run_id` | STRING | Pipeline run id |
| `forecast_horizon_days` | INTEGER | Forecast horizon in trading days |
| `forecast_start_date` | STRING | Start date for forecast window |
| `forecast_end_date` | STRING | Planned or realized forecast end date |
| `realized_return` | Semantically FLOAT | Realized ticker return after horizon; currently typed INTEGER |
| `realized_spy_return` | Semantically FLOAT | Realized SPY return over same horizon; currently typed INTEGER |
| `realized_active_return` | Semantically FLOAT | Realized return minus SPY return; currently typed INTEGER |
| `forecast_error` | Semantically FLOAT | Realized return minus calibrated expected return; currently typed INTEGER |
| `forecast_hit` | Semantically BOOLEAN | Whether forecast direction matched realized direction; currently typed INTEGER |
| `outcome_status` | STRING | pending, realized, or unavailable |

### `stock_analysis_gold.optimizer_input`

Granularity: one candidate ticker per run.

Use: inspect what the model and optimizer saw before producing recommendations.

Why included: it explains why an asset was eligible or not, whether it passed the benchmark gate,
and which features were available.

| Column | Type | Description |
| --- | --- | --- |
| `ticker` | STRING | Candidate ticker |
| `security` | STRING | Security name |
| `gics_sector` | STRING | Sector classification |
| `is_benchmark_candidate` | BOOLEAN | Whether row is an investable benchmark candidate such as SPY |
| `expected_return` | FLOAT | Optimizer expected return input |
| `forecast_score` | FLOAT | Raw model score/ranking signal |
| `calibrated_expected_return` | FLOAT | Calibrated horizon return |
| `volatility` | FLOAT | Selected volatility estimate |
| `benchmark_expected_return` | FLOAT | SPY expected return comparator |
| `benchmark_expected_return_margin` | FLOAT | Required margin over SPY |
| `benchmark_return_gate_passed` | BOOLEAN | Whether active asset passed SPY gate |
| `eligible_for_optimization` | BOOLEAN | Whether optimizer can allocate to ticker |
| `forecast_engine` | STRING | Forecast engine, currently ML |
| `forecast_model_version` | STRING | Model version used to score ticker |
| `expected_return_is_calibrated` | BOOLEAN | Whether expected return is calibrated |
| `calibration_status` | STRING | Calibration status |
| `as_of_date` | STRING | Latest feature/market date |
| `momentum_21d`, `momentum_63d`, `momentum_126d`, `momentum_252d` | FLOAT | Rolling momentum features |
| `momentum_21d_rank`, `momentum_63d_rank`, `momentum_126d_rank`, `momentum_252d_rank` | FLOAT | Cross-sectional momentum percentile ranks |
| `volatility_21d`, `volatility_63d`, `volatility_126d` | FLOAT | Rolling annualized volatility features |
| `max_drawdown_63d`, `max_drawdown_252d` | FLOAT | Rolling max drawdown features |
| `ma_ratio_50d`, `ma_ratio_200d` | FLOAT | Price-to-moving-average ratio features |
| `return_5d`, `return_21d` | FLOAT | Recent return features |
| `dollar_volume_21d` | FLOAT | 21-day liquidity proxy |
| `volume_21d_zscore` | FLOAT | Volume z-score |
| `return_21d_excess` | FLOAT | 21-day excess return versus benchmark |
| `run_id` | STRING | Pipeline run id |

### `stock_analysis_gold.price_coverage`

Granularity: one requested provider ticker per run.

Use: data-quality and freshness checks. This is the first table to inspect when a ticker disappears
from recommendations.

Why included: the model should not silently score only the tickers that returned data. This table
records requested-versus-usable coverage.

| Column | Type | Description |
| --- | --- | --- |
| `ticker` | STRING | Canonical ticker |
| `provider_ticker` | STRING | Ticker sent to price provider |
| `security` | STRING | Security name |
| `gics_sector` | STRING | Sector classification |
| `is_benchmark_candidate` | BOOLEAN | Whether row is benchmark candidate |
| `is_benchmark_ticker` | BOOLEAN | Whether ticker is a benchmark ticker such as SPY |
| `coverage_status` | STRING | ok, missing, stale, or no_latest_feature |
| `returned_price_rows` | INTEGER | Number of price rows returned |
| `first_price_date` | DATE | First available price date |
| `last_price_date` | DATE | Last available price date |
| `data_as_of_date` | STRING | Latest market data date for the run |
| `stale_calendar_days` | INTEGER | Calendar lag from `data_as_of_date` to last price date |
| `max_stale_calendar_days` | INTEGER | Configured stale threshold |
| `has_price_rows` | BOOLEAN | Whether provider returned any price rows |
| `has_latest_price_date` | BOOLEAN | Whether ticker has price on data date |
| `has_latest_feature_row` | BOOLEAN | Whether ticker has latest feature row |
| `included_in_latest_inference` | BOOLEAN | Whether ticker was usable for latest inference |
| `run_id` | STRING | Pipeline run id |

### `stock_analysis_gold.run_metadata`

Granularity: one row per complete run.

Use: run audit table. Use this for run selectors, freshness badges, model status, and configuration
traceability.

Why included: every recommendation needs a reproducible context: dates, model, calibration, coverage,
account mode, config hash, and model artifact.

| Column | Type | Description |
| --- | --- | --- |
| `run_id` | STRING | Pipeline run id |
| `requested_as_of_date` | STRING | Date requested by operator/config |
| `as_of_date` | STRING | Alias for data date |
| `data_as_of_date` | STRING | Actual latest market data date used |
| `config_hash` | STRING | Hash of runtime config |
| `forecast_engine` | STRING | Forecast engine |
| `model_version` | STRING | Model version |
| `model_family` | STRING | Model family label |
| `model_artifact_uri` | STRING | GCS model artifact or pointer used by inference |
| `model_contract_status` | STRING | Model artifact compatibility validation status |
| `model_contract_checked_at_utc` | STRING | UTC timestamp of contract check |
| `model_contract_failure_reason` | STRING | Failure reason if model contract failed |
| `model_trained_through_date` | STRING | Latest date included in model training |
| `model_created_at_utc` | STRING | Model artifact creation timestamp |
| `forecast_horizon_days` | INTEGER | Forecast horizon in trading days |
| `expected_return_is_calibrated` | BOOLEAN | Whether expected return is in return units |
| `optimizer_return_unit` | STRING | Unit used by optimizer, for example `5d_return` |
| `calibration_enabled` | BOOLEAN | Whether calibration was enabled |
| `ml_max_assets` | INTEGER | Max assets filter, null for full universe |
| `calibration_method` | STRING | Calibration method |
| `calibration_target` | STRING | Calibration target |
| `calibration_model_version` | STRING | Combined model/calibration version label |
| `calibration_status` | STRING | Calibration status |
| `calibration_trained_through_date` | STRING | Latest date used in calibration training |
| `calibration_observations` | INTEGER | Calibration observation count |
| `calibration_mae` | FLOAT | Calibration validation MAE |
| `calibration_rmse` | FLOAT | Calibration validation RMSE |
| `calibration_rank_ic` | FLOAT | Rank information coefficient |
| `min_active_expected_return_vs_benchmark` | FLOAT | Active-name margin over SPY |
| `commission_rate` | FLOAT | Commission assumption |
| `min_rebalance_trade_weight` | FLOAT | Minimum trade threshold |
| `sector_max_weight` | FLOAT | Sector cap |
| `benchmark_candidate_max_weight` | FLOAT | Max weight for benchmark candidate |
| `include_benchmark_tickers_in_universe` | BOOLEAN | Whether SPY was included as candidate |
| `benchmark_candidate_tickers` | STRING | Benchmark candidate tickers |
| `current_holdings_path` | STRING | Scenario holdings path if used |
| `live_account_enabled` | BOOLEAN | Whether live account mode was enabled |
| `live_account_slug` | STRING | Live account slug |
| `live_cashflow_source` | STRING | scenario or actual |
| `live_snapshot_id` | STRING | Live account snapshot id |
| `live_snapshot_date` | STRING | Snapshot date |
| `live_unapplied_cashflow_amount` | FLOAT | Cashflows after latest snapshot |
| `live_unapplied_cashflow_count` | INTEGER | Count of unapplied cashflows |
| `universe_count` | INTEGER | Number of universe rows |
| `price_row_count` | INTEGER | Number of raw/bronze price rows |
| `requested_ticker_count` | INTEGER | Requested ticker count |
| `returned_ticker_count` | INTEGER | Tickers with any returned prices |
| `latest_price_ticker_count` | INTEGER | Tickers with latest date price |
| `latest_feature_ticker_count` | INTEGER | Tickers with latest feature row |
| `usable_ticker_count` | INTEGER | Tickers usable for inference |
| `missing_ticker_count` | INTEGER | Missing ticker count |
| `stale_ticker_count` | INTEGER | Stale ticker count |
| `no_latest_feature_ticker_count` | INTEGER | Tickers with prices but no latest feature |
| `price_coverage_ratio` | FLOAT | Usable/requested ticker ratio |
| `benchmark_price_status` | STRING | Benchmark coverage status |
| `created_at_utc` | STRING | Run metadata creation timestamp |
| `config_json` | STRING | Serialized runtime config |

### `stock_analysis_gold.portfolio_risk_metrics`

Granularity: one metric per run.

Use: portfolio-level KPI tiles.

Why included: risk and concentration should be visible beside recommendation lines.

| Column | Type | Description |
| --- | --- | --- |
| `metric` | STRING | Metric name, for example expected_return, expected_volatility, num_holdings |
| `value` | FLOAT | Metric value |
| `as_of_date` | STRING | Market data date |
| `run_id` | STRING | Pipeline run id |

Current observed metric names:

| Metric | Meaning |
| --- | --- |
| `expected_return` | Portfolio expected return over optimizer horizon |
| `expected_volatility` | Portfolio volatility estimate over optimizer horizon |
| `num_holdings` | Count of positive target weights |
| `max_weight` | Largest target weight |
| `concentration_hhi` | Herfindahl-Hirschman concentration index |

### `stock_analysis_gold.sector_exposure`

Granularity: one sector per run.

Use: sector allocation charts.

Why included: the optimizer has sector constraints and the user needs to see resulting exposure.

| Column | Type | Description |
| --- | --- | --- |
| `gics_sector` | STRING | Sector name |
| `target_weight` | FLOAT | Total target portfolio weight in sector |
| `as_of_date` | STRING | Market data date |
| `run_id` | STRING | Pipeline run id |

### `stock_analysis_gold.forecast_calibration_diagnostics`

Granularity: one row per run calibration result.

Use: validate whether forecast scores can be interpreted as expected returns.

Why included: forecast error and expected return are scientifically meaningful only when calibration
passes.

| Column | Type | Description |
| --- | --- | --- |
| `calibration_status` | STRING | calibrated, disabled, failed, or related status |
| `calibration_method` | STRING | Calibration method, currently isotonic |
| `calibration_target` | STRING | Target calibrated, currently return |
| `calibration_horizon_days` | INTEGER | Horizon calibrated |
| `calibration_observations` | INTEGER | Validation observation count |
| `calibration_fit_observations` | INTEGER | Fit observation count |
| `calibration_total_observations` | INTEGER | Total available calibration observations |
| `calibration_validation_fraction` | FLOAT | Validation fraction used |
| `calibration_trained_through_date` | STRING | Latest date used for calibration training |
| `calibration_mae` | FLOAT | Mean absolute error |
| `calibration_rmse` | FLOAT | Root mean squared error |
| `calibration_rank_ic` | FLOAT | Rank information coefficient |
| `calibration_target_mean` | FLOAT | Mean realized target return |
| `calibration_prediction_mean` | FLOAT | Mean calibrated prediction |
| `calibration_score_min` | FLOAT | Minimum forecast score in calibration set |
| `calibration_score_max` | FLOAT | Maximum forecast score in calibration set |
| `calibration_shrinkage` | FLOAT | Shrinkage applied to calibrated prediction |
| `run_id` | STRING | Pipeline run id |

### `stock_analysis_gold.forecast_calibration_predictions`

Granularity: one historical ticker-date prediction per calibration run.

Use: calibration model monitoring and forecast-quality analysis.

Why included: lets Tableau or SQL inspect score-to-return calibration behavior by ticker/date.

| Column | Type | Description |
| --- | --- | --- |
| `ticker` | STRING | Ticker |
| `date` | STRING | Historical feature/prediction date |
| `forecast_score` | FLOAT | Model score before calibration |
| `realized_return` | FLOAT | Realized forward return used as calibration target |
| `train_cutoff_date` | STRING | Training cutoff date for calibration split |
| `calibration_fold_start_date` | STRING | Start date for calibration fold |
| `calibrated_expected_return` | FLOAT | Calibrated expected return |
| `calibration_sample` | STRING | Sample label, for example fit/validation |
| `run_id` | STRING | Pipeline run id |

## Recommended Cleanup And Improvements

1. Stabilize BigQuery schemas with explicit load schemas. Do not rely on pandas/null inference for
   return, forecast outcome, account, and boolean fields.
2. Partition stock-analysis tables by `as_of_date`, `run_data_as_of_date`, or ingestion date where
   appropriate.
3. Cluster large ticker/date tables by `run_id`, `ticker`, and date.
4. Add table descriptions and column descriptions in BigQuery itself so Tableau users see the
   dictionary in the BI layer.
5. Decide whether BigQuery should keep all historical runs or only production/current runs. If all
   runs stay, add a run status field and a dashboard filter for production runs.
6. Add BigQuery-backed account history tables before using GCP as the full dashboard source of
   truth for deposits, holdings, and performance history.

## Bottom Line

Use `stock_analysis_gold` for the Tableau stock-analysis dashboard. It contains the current cloud
outputs: recommendations, optimizer inputs, calibration diagnostics, risk metrics, sector exposure,
run metadata, and data-quality coverage. The main technical improvement needed inside
`stock_analysis_gold` is explicit BigQuery schema management so future realized forecast and
account-history fields use stable FLOAT/BOOLEAN/DATE types.
