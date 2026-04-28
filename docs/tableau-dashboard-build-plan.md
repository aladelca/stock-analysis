# Tableau Dashboard Build Plan

This plan builds a Tableau Cloud dashboard from the current Python-generated single-table mart:

```text
data/runs/<run_id>/gold/tableau_dashboard_mart.hyper
```

The Hyper file contains exactly one table:

```text
Extract.portfolio_dashboard_mart
```

That shape is intentional. Tableau Cloud can publish it as a datasource without a Tableau Prep step.

## Architecture

```text
Python one-shot pipeline
  -> data/runs/<run_id>/gold/tableau_dashboard_mart.hyper
  -> publish-tableau with PublishMode.Overwrite
  -> Tableau Cloud datasource: portfolio_dashboard_mart
  -> generate-tableau-workbook creates a starter .twb connected to that datasource
  -> Tableau Desktop refinement of the generated workbook
  -> Tableau Cloud workbook refreshes when datasource is overwritten
```

Tableau Prep remains optional. For v1, publish the Python-generated Hyper directly because it is already dashboard-shaped and publishable.

## Prerequisites

- Tableau Desktop installed locally.
- Tableau Cloud site: `https://us-east-1.online.tableau.com`, site `aladelca`.
- `.env` with Tableau PAT credentials.
- Python dependencies installed with Tableau extras.
- A completed run under `data/runs/<run_id>/`.

## Configure Publishing

Create local config:

```bash
cp -f configs/portfolio.yaml configs/portfolio.local.yaml
```

Set:

```yaml
tableau:
  export_csv: true
  export_hyper: true
  publish_enabled: true
  server_url: https://us-east-1.online.tableau.com
  site_name: aladelca
  project_name: Default
  datasource_name: portfolio_dashboard_mart
  workbook_name: portfolio_recommendations
  workbook_output_path: tableau/workbooks/portfolio_recommendations.twb
```

Create `.env`:

```bash
cp -f .env.example .env
```

Set:

```text
TABLEAU_SERVER_URL=https://us-east-1.online.tableau.com
TABLEAU_SITE_NAME=aladelca
TABLEAU_PAT_NAME=<token-name>
TABLEAU_PAT_VALUE=<token-secret>
```

Install Tableau extras:

```bash
uv sync --extra tableau --dev
```

## Seed Datasource

Run a fresh pipeline:

```bash
uv run stock-analysis run-one-shot --config configs/portfolio.local.yaml
export RUN_ID="<printed_run_id>"
```

Or reuse an existing run:

```bash
export RUN_ID="20260424T053617Z"
uv run stock-analysis export-tableau --config configs/portfolio.local.yaml --run-id "$RUN_ID"
```

Verify the generated Hyper has one table:

```bash
uv run python -c 'from pathlib import Path
from tableauhyperapi import Connection, HyperProcess, Telemetry
p = Path(f"data/runs/{__import__(\"os\").environ[\"RUN_ID\"]}/gold/tableau_dashboard_mart.hyper")
with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper, Connection(hyper.endpoint, p) as c:
    tables = c.catalog.get_table_names("Extract")
    print([str(t) for t in tables])
    print("table_count=", len(tables))
'
```

Expected:

```text
table_count= 1
```

Publish:

```bash
uv run stock-analysis publish-tableau "data/runs/$RUN_ID/gold/tableau_dashboard_mart.hyper" --config configs/portfolio.local.yaml
```

Verify in Tableau Cloud that datasource `portfolio_dashboard_mart` exists in project `Default`.

## Current Mart Fields

The workbook should use these fields from `portfolio_dashboard_mart`:

| Field | Purpose |
| --- | --- |
| `run_id` | footer and refresh verification |
| `run_data_as_of_date` | data freshness KPI |
| `run_requested_as_of_date` | compare requested date vs market data date |
| `is_data_date_lagged` | Boolean freshness warning |
| `data_date_status` | human-readable freshness status |
| `ticker` | holdings table and scatter |
| `security` | holdings table and tooltip |
| `gics_sector` | color and sector grouping |
| `forecast_score` | y-axis and holdings signal field |
| `volatility` | x-axis and risk field |
| `current_weight` | prior portfolio allocation |
| `target_weight` | optimizer target allocation |
| `executable_target_weight` | post-band target implied by executable trades |
| `display_target_weight` | dust-cleaned executable allocation for charts |
| `executable_target_weight_label` | formatted executable display |
| `trade_weight` | signed buy/sell percentage |
| `trade_abs_weight` | trade ticket bar size |
| `estimated_commission_weight` | trade cost tooltip/KPI |
| `selected` | filter selected holdings and style scatter |
| `scatter_size` | selected names scale by weight; excluded names stay small |
| `action` | BUY/SELL/HOLD/EXCLUDE signal |
| `reason_code` | explanation tooltip |
| `sector_target_weight` | optional sector tooltip |
| `portfolio_expected_return` | portfolio forecast-score KPI |
| `portfolio_expected_volatility` | portfolio volatility KPI |
| `portfolio_return_per_vol` | informal return/vol KPI |
| `portfolio_num_holdings` | holdings KPI |
| `portfolio_max_weight` | concentration KPI |
| `portfolio_concentration_hhi` | concentration diagnostic |
| `run_config_hash_short` | footer |

## Workbook Generation

Generate the starter workbook XML:

```bash
uv run stock-analysis generate-tableau-workbook --config configs/portfolio.local.yaml
```

Default output:

```text
tableau/workbooks/portfolio_recommendations.twb
```

The generated `.twb` is intentionally a base workbook, not the final polished
artifact. It connects to the published datasource and creates:

- six KPI sheets
- holdings bar sheet
- sector allocation sheet
- risk/forecast scatter sheet
- freshness footer sheet
- fixed-size `Portfolio Recommendations` dashboard

Open the generated workbook in Tableau Desktop, sign in to the `aladelca` site,
and let Tableau validate/upgrade the XML. Then refine formatting, filters,
tooltips, labels, and sheet layout in the GUI.

## Manual Workbook Build

Connect Tableau Desktop to the published datasource:

1. Tableau Desktop -> Connect -> Tableau Server.
2. Server: `https://us-east-1.online.tableau.com`.
3. Site: `aladelca`.
4. Open datasource `portfolio_dashboard_mart`.

If you prefer to build manually instead of using the generator, create sheets:

| Sheet | Mark | Fields |
| --- | --- | --- |
| `kpi_as_of` | Text | `MAX([run_data_as_of_date])` |
| `kpi_holdings` | Text | `MAX([portfolio_num_holdings])` |
| `kpi_forecast_score` | Text | `MAX([portfolio_expected_return])` |
| `kpi_volatility` | Text | `MAX([portfolio_expected_volatility])` |
| `kpi_return_per_vol` | Text | `MAX([portfolio_return_per_vol])` |
| `kpi_weight_sum` | Text | `SUM([display_target_weight])` |
| `holdings_table` | Table/bar | filter `[selected] = true`; sort by `display_target_weight` desc |
| `trade_tickets` | Bar | filter `[rebalance_required] = true`; sort by `trade_abs_weight` desc |
| `sector_treemap` | Square | group by `gics_sector`; size by `SUM([display_target_weight])` |
| `risk_return_scatter` | Circle | x `volatility`; y `forecast_score`; size `scatter_size`; color by `gics_sector`; detail `ticker` |
| `freshness_footer` | Text | `run_id`, `run_data_as_of_date`, `run_requested_as_of_date`, `run_config_hash_short` |

Important aggregation rule:

- Use `MAX()` for repeated portfolio-level columns.
- Use `SUM([display_target_weight])` for executable allocation checks and sector totals; it can be
  below `1.0` when cash remains after no-trade-band or cash-limit rules.
- Use `SUM([target_weight])` only when you intentionally want the raw optimizer target before the
  no-trade band.

## Dashboard Layout

Use the layout from `docs/tableau-dashboard-design.md`:

```text
KPI strip
Holdings table left
Sector treemap top-right
Risk/return scatter bottom-right
Freshness footer
```

Recommended generated fixed size:

```text
1280 x 850
```

## Publish Workbook

Save locally as:

```text
tableau/workbooks/portfolio_recommendations.twb
```

Publish from the CLI after opening and saving once in Tableau Desktop:

```bash
uv run stock-analysis publish-tableau-workbook tableau/workbooks/portfolio_recommendations.twb --config configs/portfolio.local.yaml
```

Or publish from Tableau Desktop:

- Project: `Default`
- Workbook name: `portfolio_recommendations`
- Data source: published `portfolio_dashboard_mart`
- Show sheets as tabs: off

## Refresh Loop

For each new run:

```bash
uv run stock-analysis run-one-shot --config configs/portfolio.local.yaml
export RUN_ID="<printed_run_id>"
uv run stock-analysis publish-tableau "data/runs/$RUN_ID/gold/tableau_dashboard_mart.hyper" --config configs/portfolio.local.yaml
```

Refresh the workbook in Tableau Cloud. If `run_id` changes in the footer, the loop works.

## Known V1 Limitations

- `action` supports BUY, SELL, HOLD, and EXCLUDE, but true sell/hold semantics require a configured
  current holdings file.
- `forecast_score` is a ranking signal. Use `calibrated_expected_return` as expected percentage
  return only when `expected_return_is_calibrated = true`.
- `target_weight` is the optimizer target. Use `display_target_weight` or
  `executable_target_weight` for dashboard holdings, because no-trade-band rows can have a
  nonzero optimizer target and no executable buy.
- No time-series dashboard yet because the mart is one run, one decision.
- Sector exposure is shown, but benchmark-relative sector bands are not implemented.

## Validation Checklist

- [ ] Datasource `portfolio_dashboard_mart` publishes successfully.
- [ ] Hyper file has exactly one table.
- [ ] `generate-tableau-workbook` writes `tableau/workbooks/portfolio_recommendations.twb`.
- [ ] Generated `.twb` opens in Tableau Desktop after signing in to Tableau Cloud.
- [ ] KPI tiles render non-null values.
- [ ] `SUM([display_target_weight])` is nonnegative and does not exceed `1.00` except for rounding.
- [ ] `MAX([portfolio_max_weight])` is at or below configured `optimizer.max_weight`.
- [ ] Holdings table filters to `[selected] = true`.
- [ ] Scatter shows both selected and excluded tickers.
- [ ] Footer `run_id` matches the latest published run.
- [ ] A second run and re-publish updates the footer `run_id` in Tableau Cloud.
