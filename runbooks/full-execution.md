# Full Execution Runbook

This runbook executes the full MVP flow from local setup through Tableau-ready outputs.

## 1. Preconditions

Run all commands from the repository root. If you are not already there, enter the checked-out project directory first:

```bash
cd stock-analysis
```

Required locally:

- Python 3.12
- `uv`
- Network access for Wikipedia and `yfinance`
- Tableau Prep Builder for the visual transformation step
- Tableau Server credentials only if publishing is required

Optional:

- Tableau Hyper API and Tableau Server Client via `uv sync --extra tableau --dev`
- MLflow tracking via `uv sync --extra mlflow --dev`

## 2. Install Dependencies

For the core pipeline:

```bash
uv sync --dev
```

If you need Hyper export or Tableau Server publishing:

```bash
uv sync --extra tableau --dev
```

If you want the one-shot run logged to MLflow:

```bash
uv sync --extra mlflow --dev
```

## 3. Validate The Codebase

Run all quality gates before generating portfolio outputs:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest
```

Expected result:

```text
Ruff format: no files would be reformatted
Ruff check: All checks passed
mypy: Success
pytest: all tests passed
```

## 4. Review Runtime Config

Open `configs/portfolio.yaml` and confirm:

```yaml
run:
  as_of_date: null
  output_root: data
  run_id: null

prices:
  provider: yfinance
  lookback_years: 5

forecast:
  engine: ml
  ml_model_version: e8-scale-0p5-contribution-aware-v1
  ml_horizon_days: 5
  ml_max_assets: 100
  ml_score_scale: 0.5

optimizer:
  max_weight: 0.30
  risk_aversion: 10.0
  commission_rate: 0.02
  min_rebalance_trade_weight: 0.005
  sector_max_weight: 0.35

portfolio_state:
  current_holdings_path: null
  portfolio_value: null

contributions:
  initial_portfolio_value: 1000.0
  monthly_deposit_amount: 0.0
  deposit_frequency_days: 30
  deposit_start_date: null
  rebalance_on_deposit_day: true

execution:
  cash_balance: 0.0
  no_trade_band: 0.0

mlflow:
  enabled: true
  tracking_uri: sqlite:///data/mlflow/mlflow.db
  experiment_name: stock-analysis-portfolio

tableau:
  export_csv: true
  export_hyper: false
  publish_enabled: false
```

Notes:

- `as_of_date: null` means the requested run date is today.
- Gold `as_of_date` is the latest available price date, not necessarily today.
- `run_id: null` lets the pipeline generate a UTC run id.
- `forecast.ml_model_version: e8-scale-0p5-contribution-aware-v1` is the corrected
  contribution-aware E8 model selected in `docs/experiments/e8-scale-0p5-contribution-corrected-20260426.json`.
- `forecast.ml_score_scale: 0.5` applies the score scaling required by the selected `e8_scale_0p5`
  candidate.
- `portfolio_state.current_holdings_path: null` means first-allocation mode.
- `contributions.monthly_deposit_amount: 0.0` disables scenario-mode deposit modeling for the
  one-shot run. Backtests and autoresearch still use the monthly deposit assumption when supplied.
- For live account tracking with arbitrary-date deposits, use the Supabase flow in
  `runbooks/supabase-account-tracking.md` and set `live_account.cashflow_source: actual`.
- `execution.no_trade_band: 0.0` means trades are filtered only by `min_rebalance_trade_weight`.
- `optimizer.commission_rate: 0.02` charges 2% of absolute traded portfolio weight.
- `mlflow.enabled: true` logs parameters, recommendation metrics, risk metrics, and gold artifacts.
- Set `tableau.export_hyper: true` only after installing the Tableau extra.

## 5. Optional Current Holdings Input

If you already have a portfolio, create a holdings file with market values:

```bash
cp -f configs/current_holdings.example.csv configs/current_holdings.local.csv
```

Edit `configs/current_holdings.local.csv`:

```csv
ticker,market_value
AAPL,800
MSFT,700
NVDA,500
```

Then set:

```yaml
portfolio_state:
  current_holdings_path: configs/current_holdings.local.csv
```

Accepted holdings schemas:

```text
ticker,current_weight
ticker,weight
ticker,market_value
```

If `market_value` is supplied, the pipeline normalizes values into weights. If no holdings file is
configured, the run answers "how to allocate new money" rather than "what to trade from my current
book."

To model a scenario-mode one-shot deposit, set the contribution amount:

```yaml
contributions:
  initial_portfolio_value: 1000.0
  monthly_deposit_amount: 100.0
  deposit_frequency_days: 30
  deposit_start_date: null
  rebalance_on_deposit_day: true

execution:
  cash_balance: 0.0
  no_trade_band: 0.02
```

When using a fixed-dollar deposit, provide either `market_value` holdings or an explicit
`portfolio_state.portfolio_value`. The recommendation weights and commission dollars are computed
on the post-deposit portfolio value.

## 6. Run The One-Shot Pipeline

The default production config uses the corrected `e8_scale_0p5` ML forecast flow. To make the
forecast choice explicit in operator runs:

```bash
uv run stock-analysis run-one-shot --config configs/portfolio.yaml --forecast-engine ml
```

Expected output includes:

```text
Completed run <run_id>
Recommendations: data/runs/<run_id>/gold/portfolio_recommendations.parquet
```

Capture the run id:

```bash
export RUN_ID="<run_id>"
```

Example:

```bash
export RUN_ID="20260424T052554Z"
```

## 7. Verify Generated Files

Check the run directory:

```bash
find "data/runs/$RUN_ID" -maxdepth 3 -type f | sort
```

Required outputs:

```text
data/runs/$RUN_ID/raw/sp500_constituents/source.html
data/runs/$RUN_ID/raw/sp500_constituents/metadata.json
data/runs/$RUN_ID/raw/prices/metadata.json
data/runs/$RUN_ID/raw/prices/batch_0000.csv
data/runs/$RUN_ID/bronze/sp500_constituents.parquet
data/runs/$RUN_ID/bronze/daily_prices.parquet
data/runs/$RUN_ID/silver/asset_daily_returns.parquet
data/runs/$RUN_ID/silver/asset_daily_features.parquet
data/runs/$RUN_ID/silver/asset_universe_snapshot.parquet
data/runs/$RUN_ID/gold/optimizer_input.parquet
data/runs/$RUN_ID/gold/covariance_matrix.parquet
data/runs/$RUN_ID/gold/portfolio_recommendations.parquet
data/runs/$RUN_ID/gold/portfolio_risk_metrics.parquet
data/runs/$RUN_ID/gold/sector_exposure.parquet
data/runs/$RUN_ID/gold/run_metadata.parquet
data/runs/$RUN_ID/gold/csv/portfolio_recommendations.csv
```

MLflow outputs, when enabled:

```text
data/mlflow/mlflow.db
data/mlflow/artifacts/
```

Start the local MLflow UI:

```bash
uv run --extra mlflow mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db
```

## 8. Verify Data Date And Run Metadata

```bash
uv run python -c 'import os, pandas as pd
run = os.environ["RUN_ID"]
root = f"data/runs/{run}"
meta = pd.read_parquet(f"{root}/gold/run_metadata.parquet")
features = pd.read_parquet(f"{root}/silver/asset_daily_features.parquet")
recs = pd.read_parquet(f"{root}/gold/portfolio_recommendations.parquet")
print(meta[["requested_as_of_date", "data_as_of_date", "as_of_date", "run_id", "forecast_engine", "model_version"]].to_string(index=False))
print("latest feature date:", features["latest_date"].max())
print("recommendation as_of_date:", recs["as_of_date"].unique())
print("weight sum:", recs["target_weight"].sum())
print("current weight sum:", recs["current_weight"].sum())
print("trade abs weight:", recs.loc[recs["rebalance_required"], "trade_abs_weight"].sum())
print("estimated commission:", recs["estimated_commission_weight"].sum())
print("commission amount:", recs["commission_amount"].sum())
print("contribution:", recs["contribution_amount"].max())
print("post-contribution value:", recs["portfolio_value_after_contribution"].max())
print("max weight:", recs["target_weight"].max())
'
```

Expected:

- `data_as_of_date` equals the latest available market price date.
- Recommendation `as_of_date` equals `data_as_of_date`.
- `forecast_engine` is `ml`.
- `model_version` is `e8-scale-0p5-contribution-aware-v1`.
- `target_weight` sums approximately to `1.0`.
- `estimated_commission_weight` equals `0.02 * trade_abs_weight` for planned BUY/SELL rows.
- `commission_amount` equals `0.02 * abs(trade_notional)` for planned BUY/SELL rows.
- `max weight` is at or below the configured max weight, allowing tiny solver tolerance.

## 9. Inspect Recommendations

Top recommendations:

```bash
uv run python -c 'import os, pandas as pd
run = os.environ["RUN_ID"]
recs = pd.read_parquet(f"data/runs/{run}/gold/portfolio_recommendations.parquet")
cols = ["ticker", "security", "gics_sector", "current_weight", "target_weight", "trade_weight", "trade_notional", "commission_amount", "action", "reason_code"]
print(recs[cols].head(25).to_string(index=False))
'
```

Action semantics:

```text
BUY  = increase the position by trade_weight
SELL = decrease or exit the position by abs(trade_weight)
HOLD = current and target weights differ by less than min_rebalance_trade_weight
EXCLUDE = no current position and no target allocation
```

Risk metrics:

```bash
uv run python -c 'import os, pandas as pd
run = os.environ["RUN_ID"]
risk = pd.read_parquet(f"data/runs/{run}/gold/portfolio_risk_metrics.parquet")
print(risk.to_string(index=False))
'
```

Sector exposure:

```bash
uv run python -c 'import os, pandas as pd
run = os.environ["RUN_ID"]
sectors = pd.read_parquet(f"data/runs/{run}/gold/sector_exposure.parquet")
print(sectors.to_string(index=False))
'
```

## 10. Export Tableau Files From Existing Run

This step must not re-run ingestion. It reads existing Parquet files for the selected run:

```bash
uv run stock-analysis export-tableau --config configs/portfolio.yaml --run-id "$RUN_ID"
```

Expected CSV inputs:

```text
data/runs/$RUN_ID/bronze/csv/sp500_constituents.csv
data/runs/$RUN_ID/silver/csv/asset_daily_features.csv
data/runs/$RUN_ID/gold/csv/portfolio_recommendations.csv
data/runs/$RUN_ID/gold/csv/portfolio_risk_metrics.csv
data/runs/$RUN_ID/gold/csv/sector_exposure.csv
data/runs/$RUN_ID/gold/csv/run_metadata.csv
```

## 11. Optional Hyper Export

Enable Hyper export in a local config copy:

```bash
cp -f configs/portfolio.yaml configs/portfolio.local.yaml
```

Edit `configs/portfolio.local.yaml`:

```yaml
forecast:
  engine: ml
  ml_model_version: e8-scale-0p5-contribution-aware-v1
  ml_horizon_days: 5
  ml_max_assets: 100
  ml_score_scale: 0.5

optimizer:
  max_weight: 0.30
  lambda_turnover: 5.0

tableau:
  export_csv: true
  export_hyper: true
  publish_enabled: false
```

Install Tableau extras and export:

```bash
uv sync --extra tableau --dev
uv run stock-analysis export-tableau --config configs/portfolio.local.yaml --run-id "$RUN_ID"
```

Expected Hyper file:

```text
data/runs/$RUN_ID/gold/tableau_dashboard_mart.hyper
```

This Hyper extract contains the single `portfolio_dashboard_mart` table. With live account tracking enabled, it includes latest account performance fields, but full cashflow and performance history remains in the optional gold CSV/Parquet tables for the run.

## 12. Tableau Prep Transformation

Open Tableau Prep Builder and create or update the flow described in:

```text
tableau/prep/portfolio_dashboard_mart.flow-spec.md
```

Connect these inputs:

```text
data/runs/$RUN_ID/bronze/csv/sp500_constituents.csv
data/runs/$RUN_ID/silver/csv/asset_daily_features.csv
data/runs/$RUN_ID/gold/csv/portfolio_recommendations.csv
data/runs/$RUN_ID/gold/csv/portfolio_risk_metrics.csv
data/runs/$RUN_ID/gold/csv/sector_exposure.csv
data/runs/$RUN_ID/gold/csv/run_metadata.csv
```

Prep responsibilities:

- Join recommendations to constituents on `ticker`.
- Join recommendations to latest features on `ticker`.
- Keep dashboard fields defined in the flow spec.
- Output the dashboard mart.

Recommended output:

```text
tableau_prep_outputs/portfolio_dashboard_mart.hyper
```

Validation inside Tableau Prep:

- `target_weight` sums to approximately `1.0`.
- No selected asset has a negative weight.
- Selected assets do not exceed configured max weight.
- `run_id` is consistent across recommendations, risk metrics, sector exposure, and metadata.

## 13. Optional Tableau Server Publish

Install Tableau extras:

```bash
uv sync --extra tableau --dev
```

Create a local `.env` file. This file is ignored by git:

```bash
cp -f .env.example .env
```

Edit `.env`:

```text
TABLEAU_SERVER_URL=https://us-east-1.online.tableau.com
TABLEAU_SITE_NAME=aladelca
TABLEAU_PAT_NAME=your-token-name
TABLEAU_PAT_VALUE=your-token-secret
```

Create an ignored local config:

```bash
cp -f configs/portfolio.yaml configs/portfolio.local.yaml
```

Edit `configs/portfolio.local.yaml`:

```yaml
forecast:
  engine: ml
  ml_model_version: e8-scale-0p5-contribution-aware-v1
  ml_horizon_days: 5
  ml_max_assets: 100
  ml_score_scale: 0.5

optimizer:
  max_weight: 0.30
  lambda_turnover: 5.0

tableau:
  export_csv: true
  export_hyper: true
  publish_enabled: true
  project_name: Default
  datasource_name: portfolio_dashboard_mart
  workbook_name: portfolio_recommendations
  workbook_output_path: tableau/workbooks/portfolio_recommendations.twb
```

Publishing does not rerun the forecast model. Run `run-one-shot` and `export-tableau`
first, then publish the Python-generated Hyper file from that ML run:

```bash
uv run stock-analysis publish-tableau "data/runs/$RUN_ID/gold/tableau_dashboard_mart.hyper" --config configs/portfolio.local.yaml
```

Publish the Prep-generated mart only after rerunning the Prep flow against the same
ML run outputs:

```bash
uv run stock-analysis publish-tableau tableau_prep_outputs/portfolio_dashboard_mart.hyper --config configs/portfolio.local.yaml
```

## 14. Build Tableau Dashboard

Generate the starter workbook XML:

```bash
uv run stock-analysis generate-tableau-workbook --config configs/portfolio.local.yaml
```

Open the generated workbook:

```text
tableau/workbooks/portfolio_recommendations.twb
```

In Tableau Desktop:

1. Sign in to the configured Tableau Cloud site.
2. Open the generated `.twb`.
3. Confirm it connects to the published datasource `portfolio_dashboard_mart`.
4. Let Tableau Desktop validate or upgrade the workbook XML.
5. Refine formatting, filters, labels, tooltips, and layout.
6. Save the workbook back as `.twb` so it remains diffable in git.

Recommended dashboard sheets:

- KPI Data As Of
- KPI Holdings
- KPI Forecast Score
- KPI Volatility
- KPI Return Vol
- KPI Weight Sum
- Holdings by Weight
- Trade Tickets
- Sector Allocation
- Risk Forecast Scatter
- Freshness Footer

Publish the workbook after validating it in Tableau Desktop:

```bash
uv run stock-analysis publish-tableau-workbook tableau/workbooks/portfolio_recommendations.twb --config configs/portfolio.local.yaml
```

## 15. Troubleshooting

If `yfinance` logs failed tickers:

- Confirm the run still completed.
- Inspect `data/runs/$RUN_ID/raw/prices/metadata.json`.
- Failed or missing tickers should be excluded by data quality filters.

If optimization is infeasible:

- Check the number of eligible assets in `asset_daily_features`.
- Increase `prices.lookback_years`.
- Lower `features.min_history_days`.
- Raise `optimizer.max_weight` only if you intentionally want more concentration.

If `export-tableau` fails:

- Confirm the run exists under `data/runs/$RUN_ID`.
- Confirm required gold Parquet files exist.
- Do not run `run-one-shot` just to export Tableau files unless you want a new run.

If Tableau publish fails:

- Confirm `tableau.publish_enabled: true`.
- Confirm `.env` contains `TABLEAU_SERVER_URL`, `TABLEAU_SITE_NAME`, `TABLEAU_PAT_NAME`, and `TABLEAU_PAT_VALUE`.
- Confirm the Tableau project exists.
- Confirm the datasource path exists.

## 16. End Of Session Checks

Run final validation:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest
```

Check git state:

```bash
git status --short
```

If a git remote is configured:

```bash
git pull --rebase
bd dolt push
git push
git status --branch --short
```

If no remote is configured, `git push` and `bd dolt push` will fail until `origin` is added.
