# Portfolio Dashboard Mart Flow Spec

This is the optional Tableau Prep equivalent of the Python-generated dashboard mart. The Python Hyper file is the v1 source of truth, but a Prep flow can reproduce the same shape if visual transformation is required.

## Inputs

Current-run inputs:

- `optimizer_input.csv` keyed by `ticker`
- `portfolio_recommendations.csv` keyed by `ticker`
- `sp500_constituents.csv` keyed by `ticker`
- `asset_daily_features.csv` keyed by `ticker`
- `portfolio_risk_metrics.csv` keyed by `run_id`
- `sector_exposure.csv` keyed by `run_id`, `gics_sector`
- `run_metadata.csv` keyed by `run_id`
- `forecast_calibration_diagnostics.csv` keyed by run, if you want calibration QA views
- `forecast_calibration_predictions.csv` keyed by calibration validation date and ticker, if you
  want score-vs-realized validation views

Live account history inputs, when Supabase-backed account tracking is enabled:

- `recommendation_runs_history.csv` keyed by `id`
- `recommendation_lines_history.csv` keyed by `recommendation_run_id`
- `performance_snapshots_history.csv` keyed by `account_id`, `as_of_date`
- `portfolio_snapshots_history.csv` keyed by `account_id`, `snapshot_date`
- `holding_snapshots_history.csv` keyed by `snapshot_id`, `ticker`
- `cashflows_history.csv` keyed by `id`

## Flow

1. Start from `portfolio_recommendations.csv` so current holdings outside the optimizer universe are
   preserved as SELL rows.
2. LEFT JOIN `optimizer_input.csv` on `ticker`.
3. LEFT JOIN `sp500_constituents.csv` on `ticker`.
4. LEFT JOIN `asset_daily_features.csv` on `ticker`.
5. Join or aggregate portfolio metrics from `portfolio_risk_metrics.csv` so each ticker row has:
   - `portfolio_expected_return`
   - `portfolio_expected_volatility`
   - `portfolio_num_holdings`
   - `portfolio_max_weight`
   - `portfolio_concentration_hhi`
6. Join `run_metadata.csv` so each ticker row has:
   - `run_requested_as_of_date`
   - `run_data_as_of_date`
   - `run_created_at_utc`
   - `run_config_hash`
   - `run_config_hash_short`
   - `run_optimizer_return_unit`
   - `run_calibration_status`
   - `run_calibration_observations`
   - `run_calibration_mae`
   - `run_calibration_rmse`
   - `run_calibration_rank_ic`
7. Join sector exposure by `gics_sector`.
8. Add calculated fields:
   - `current_weight = IFNULL([current_weight], 0)`
   - `target_weight = IFNULL([target_weight], 0)`
   - `executable_target_weight = IFNULL([executable_target_weight], [target_weight])`
   - `trade_weight = IFNULL([trade_weight], 0)`
   - `trade_abs_weight = IFNULL([trade_abs_weight], ABS([trade_weight]))`
   - `is_solver_dust = [executable_target_weight] > 0 AND [executable_target_weight] < 0.00000001`
   - `display_target_weight = IF [is_solver_dust] THEN 0 ELSE [executable_target_weight] END`
   - `display_current_weight = IF ABS([current_weight]) < 0.00000001 THEN 0 ELSE [current_weight] END`
   - `display_trade_weight = IF ABS([trade_weight]) < 0.00000001 THEN 0 ELSE [trade_weight] END`
   - `rebalance_required = IFNULL([rebalance_required], false)`
   - `estimated_commission_weight = IFNULL([estimated_commission_weight], 0)`
   - `action = IFNULL([action], "EXCLUDE")`
   - `reason_code = IFNULL([reason_code], "not_selected_or_below_threshold")`
   - `forecast_score = IFNULL([forecast_score], [expected_return])`
   - `calibrated_expected_return = IF [expected_return_is_calibrated] THEN [expected_return] END`
   - `selected = [display_target_weight] > 0`
   - `current_weight_label = STR(ROUND([current_weight] * 100, 2)) + "%"`
   - `target_weight_label = STR(ROUND([target_weight] * 100, 2)) + "%"`
   - `executable_target_weight_label = STR(ROUND([display_target_weight] * 100, 2)) + "%"`
   - `trade_weight_label = STR(ROUND([trade_weight] * 100, 2)) + "%"`
   - `estimated_commission_weight_label = STR(ROUND([estimated_commission_weight] * 100, 2)) + "%"`
   - `scatter_size = IF [selected] THEN [display_target_weight] ELSE 0.001 END`
   - `portfolio_return_per_vol = [portfolio_expected_return] / [portfolio_expected_volatility]`
   - `is_data_date_lagged = [run_requested_as_of_date] <> [run_data_as_of_date]`
9. Keep the history tables as separate outputs or relationships. Do not join
   `recommendation_lines_history` into the current-run mart unless the view explicitly needs a
   current-vs-history comparison; the grains differ.
10. Output one current-run table named `portfolio_dashboard_mart`.
11. Output account-history tables with their original `_history` names.
12. Output `portfolio_dashboard_mart.hyper` to `tableau_prep_outputs/`.

## Required Output Grain

```text
one row per recommendation ticker per run
```

## Required Output Fields

- `run_id`
- `as_of_date`
- `ticker`
- `security`
- `gics_sector`
- `forecast_score`
- `expected_return_is_calibrated`
- `calibrated_expected_return`
- `forecast_horizon_days`
- `forecast_start_date`
- `forecast_end_date`
- `realized_return`
- `realized_spy_return`
- `realized_active_return`
- `forecast_error`
- `forecast_hit`
- `outcome_status`
- `volatility`
- `current_weight`
- `display_current_weight`
- `current_weight_label`
- `target_weight`
- `executable_target_weight`
- `executable_target_market_value`
- `display_target_weight`
- `target_weight_label`
- `executable_target_weight_label`
- `trade_weight`
- `display_trade_weight`
- `trade_abs_weight`
- `trade_weight_label`
- `rebalance_required`
- `estimated_commission_weight`
- `estimated_commission_weight_label`
- `net_trade_weight_after_commission`
- `cash_required_weight`
- `cash_released_weight`
- `is_solver_dust`
- `selected`
- `scatter_size`
- `action`
- `reason_code`
- `sector_target_weight`
- `portfolio_expected_return`
- `portfolio_expected_volatility`
- `portfolio_return_per_vol`
- `portfolio_num_holdings`
- `portfolio_max_weight`
- `portfolio_concentration_hhi`
- `account_total_value`
- `account_initial_value`
- `account_total_deposits`
- `account_invested_capital`
- `account_return_on_invested_capital`
- `active_value`
- `active_return`
- `run_requested_as_of_date`
- `run_data_as_of_date`
- `is_data_date_lagged`
- `data_date_status`
- `run_created_at_utc`
- `run_config_hash`
- `run_config_hash_short`
- `run_expected_return_is_calibrated`
- `run_optimizer_return_unit`
- `run_calibration_enabled`
- `run_calibration_method`
- `run_calibration_target`
- `run_calibration_model_version`
- `run_calibration_status`
- `run_calibration_trained_through_date`
- `run_calibration_observations`
- `run_calibration_mae`
- `run_calibration_rmse`
- `run_calibration_rank_ic`

## History Dashboard Tables

Use these table grains in Tableau:

- `recommendation_lines_history`: one row per ticker per recommendation run. Use this for
  recommendation history, action history, forecast score trend, realized return, active return
  versus SPY, forecast error, directional hit, and `outcome_status`.
- `recommendation_runs_history`: one row per run. Relate it to
  `recommendation_lines_history.recommendation_run_id = recommendation_runs_history.id`. Use this
  for run-level calibration status, observation count, MAE, RMSE, and rank IC.
- `performance_snapshots_history`: one row per account valuation date. Use this for account value,
  initial value, total deposits, invested capital, return on invested capital, account TWR/MWR,
  SPY same-cashflow value, and active return.
- `cashflows_history`: one row per registered deposit/withdrawal/fee/dividend. Keep it separate
  from recommendation lines unless analyzing deposit timing.

## Validation

- Row count equals `portfolio_recommendations` row count.
- `selected` is true for selected assets and false for excluded assets.
- `SUM(target_weight)` over all rows is approximately `1.0` for the raw optimizer target.
- `SUM(display_target_weight)` where `selected = true` is no higher than `1.0` except for rounding;
  it can be lower when no-trade-band or cash-limit rules leave residual cash.
- Solver-dust rows below the dashboard tolerance have `is_solver_dust = true` and
  `selected = false`.
- `estimated_commission_weight` equals `0.02 * trade_abs_weight` for BUY/SELL rows in the default
  config.
- No selected asset has a negative weight.
- No selected asset exceeds the configured max weight, allowing solver tolerance.
- The Python-generated live-account Hyper contains `portfolio_dashboard_mart`, calibration
  diagnostics tables, current-run account tables, and `_history` tables. The no-live-account Hyper
  contains `portfolio_dashboard_mart` plus calibration diagnostics tables.
