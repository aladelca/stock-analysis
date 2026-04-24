# Portfolio Dashboard Mart Flow Spec

This is the optional Tableau Prep equivalent of the Python-generated dashboard mart. The Python Hyper file is the v1 source of truth, but a Prep flow can reproduce the same shape if visual transformation is required.

## Inputs

- `optimizer_input.csv` keyed by `ticker`
- `portfolio_recommendations.csv` keyed by `ticker`
- `sp500_constituents.csv` keyed by `ticker`
- `asset_daily_features.csv` keyed by `ticker`
- `portfolio_risk_metrics.csv` keyed by `run_id`
- `sector_exposure.csv` keyed by `run_id`, `gics_sector`
- `run_metadata.csv` keyed by `run_id`

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
7. Join sector exposure by `gics_sector`.
8. Add calculated fields:
   - `selected = IFNULL([target_weight], 0) > 0`
   - `current_weight = IFNULL([current_weight], 0)`
   - `target_weight = IFNULL([target_weight], 0)`
   - `trade_weight = IFNULL([trade_weight], 0)`
   - `trade_abs_weight = IFNULL([trade_abs_weight], ABS([trade_weight]))`
   - `rebalance_required = IFNULL([rebalance_required], false)`
   - `estimated_commission_weight = IFNULL([estimated_commission_weight], 0)`
   - `action = IFNULL([action], "EXCLUDE")`
   - `reason_code = IFNULL([reason_code], "not_selected_or_below_threshold")`
   - `forecast_score = [expected_return]`
   - `current_weight_label = STR(ROUND([current_weight] * 100, 2)) + "%"`
   - `target_weight_label = STR(ROUND([target_weight] * 100, 2)) + "%"`
   - `trade_weight_label = STR(ROUND([trade_weight] * 100, 2)) + "%"`
   - `estimated_commission_weight_label = STR(ROUND([estimated_commission_weight] * 100, 2)) + "%"`
   - `scatter_size = IF [selected] THEN [target_weight] ELSE 0.001 END`
   - `portfolio_return_per_vol = [portfolio_expected_return] / [portfolio_expected_volatility]`
   - `is_data_date_lagged = [run_requested_as_of_date] <> [run_data_as_of_date]`
9. Output exactly one table named `portfolio_dashboard_mart`.
10. Output `portfolio_dashboard_mart.hyper` to `tableau_prep_outputs/`.

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
- `volatility`
- `current_weight`
- `current_weight_label`
- `target_weight`
- `target_weight_label`
- `trade_weight`
- `trade_abs_weight`
- `trade_weight_label`
- `rebalance_required`
- `estimated_commission_weight`
- `estimated_commission_weight_label`
- `net_trade_weight_after_commission`
- `cash_required_weight`
- `cash_released_weight`
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
- `run_requested_as_of_date`
- `run_data_as_of_date`
- `is_data_date_lagged`
- `data_date_status`
- `run_created_at_utc`
- `run_config_hash`
- `run_config_hash_short`

## Validation

- Row count equals `portfolio_recommendations` row count.
- `selected` is true for selected assets and false for excluded assets.
- `SUM(target_weight)` over all rows is approximately `1.0`.
- `SUM(target_weight)` where `selected = true` is approximately `1.0`.
- `estimated_commission_weight` equals `0.02 * trade_abs_weight` for BUY/SELL rows in the default
  config.
- No selected asset has a negative weight.
- No selected asset exceeds the configured max weight, allowing solver tolerance.
- The output Hyper contains exactly one table.
