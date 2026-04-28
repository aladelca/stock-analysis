# Tableau Server Publish Runbook

Publishing requires Tableau credentials and the optional Tableau dependency group.

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

Create an ignored local config file:

```bash
cp -f configs/portfolio.yaml configs/portfolio.local.yaml
```

Enable publishing in `configs/portfolio.local.yaml`:

```yaml
forecast:
  engine: ml
  ml_model_version: lightgbm_return_zscore
  ml_horizon_days: 5
  ml_max_assets: null
  ml_score_scale: 1.0
  ml_min_active_expected_return_vs_benchmark: 0.001

optimizer:
  max_weight: 0.24
  benchmark_candidate_max_weight: 1.0
  lambda_turnover: 5.0
  preserve_outside_holdings: true

tableau:
  export_hyper: true
  publish_enabled: true
```

Regenerate the pipeline outputs before publishing. Publishing only uploads an existing
`.hyper` file; it does not rerun the forecast model.

```bash
uv run stock-analysis run-one-shot --config configs/portfolio.local.yaml --forecast-engine ml
```

Capture the printed run id and export the Tableau Hyper extract:

```bash
export RUN_ID="<run-id>"
uv run stock-analysis export-tableau --config configs/portfolio.local.yaml --run-id "$RUN_ID"
```

Publish the Python-generated Hyper file for the ML run:

```bash
uv run stock-analysis publish-tableau "data/runs/$RUN_ID/gold/tableau_dashboard_mart.hyper" --config configs/portfolio.local.yaml
```

The generated Hyper file always contains `portfolio_dashboard_mart`. When live account tracking is enabled for the run, it also contains the account tracking tables:

- `cashflows`
- `portfolio_snapshots`
- `holding_snapshots`
- `recommendation_runs`
- `recommendation_lines`
- `performance_snapshots`
- `cashflows_history`
- `portfolio_snapshots_history`
- `holding_snapshots_history`
- `recommendation_runs_history`
- `recommendation_lines_history`
- `performance_snapshots_history`

The CSV mirrors under `data/runs/$RUN_ID/gold/csv/` are still written for Tableau Prep, audit, and fallback connections.

If a Tableau Prep flow is used and writes to `tableau_prep_outputs/`, publish that
Prep-generated artifact only after rerunning the flow against the same ML run outputs:

```bash
uv run stock-analysis publish-tableau tableau_prep_outputs/portfolio_dashboard_mart.hyper --config configs/portfolio.local.yaml
```

Generate the starter workbook:

```bash
uv run stock-analysis generate-tableau-workbook --config configs/portfolio.local.yaml
```

After opening and saving the workbook in Tableau Desktop, publish it:

```bash
uv run stock-analysis publish-tableau-workbook tableau/workbooks/portfolio_recommendations.twb --config configs/portfolio.local.yaml
```
