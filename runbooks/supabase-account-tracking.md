# Supabase Account Tracking

This runbook covers Supabase-backed account setup, cashflow registration, portfolio snapshots, and live one-shot recommendations that use actual cashflows.

## Setup

Install the optional Supabase client extra:

```bash
uv sync --extra supabase
```

For local-only development, start the local Supabase stack and apply the local migrations:

```bash
supabase start
supabase db reset
supabase status
```

Do not run `supabase link` or `supabase db push` for local-only work. `supabase db reset` recreates the local database and applies the SQL files in `supabase/migrations/`, so it will wipe local Supabase data.

Set local credentials in `.env` or your shell from `supabase status`:

```bash
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_SERVICE_ROLE_KEY=<service_role_key_from_supabase_status>
```

For a remote Supabase project, apply the SQL migrations in `supabase/migrations/` through your Supabase project, or with the Supabase CLI after linking the project.

Remote credentials look like this:

```bash
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
```

Use the service role key only from local/server jobs. Do not expose it in Tableau, browser code, or committed config files.

Row level security is owner-scoped through `accounts.owner_id`. Service-role jobs bypass RLS; browser, mobile, or Tableau user-credential access requires account rows whose `owner_id` matches `auth.uid()`.

Enable the database-backed account flow in a local config:

```yaml
live_account:
  enabled: true
  account_slug: main
  cashflow_source: actual

supabase:
  enabled: true
  url_env: SUPABASE_URL
  key_env: SUPABASE_SERVICE_ROLE_KEY
```

For a local config copy:

```bash
cp -f configs/portfolio.yaml configs/portfolio.local.yaml
```

## Register The Account

```bash
uv run --extra supabase stock-analysis upsert-account \
  --config configs/portfolio.local.yaml \
  --account-slug main \
  --display-name "Main Brokerage" \
  --owner-id "<supabase-auth-user-uuid>" \
  --benchmark-ticker SPY
```

Omit `--owner-id` for service-role-only local automation. Add it before exposing account data through authenticated Supabase clients.

## Register A Deposit

Deposits can be registered on any calendar date. The command stores deposits as positive cashflows.

```bash
uv run --extra supabase stock-analysis register-cashflow \
  --config configs/portfolio.local.yaml \
  --date 2026-04-24 \
  --amount 500 \
  --type deposit
```

For account performance versus SPY, cashflows entered on weekends or market holidays are applied to the next available SPY trading date in the benchmark simulation. The original cashflow date remains unchanged in Supabase and in the cashflow mart.

Withdrawals, fees, and taxes are stored as negative cashflows even if you pass a positive amount:

```bash
uv run --extra supabase stock-analysis register-cashflow \
  --config configs/portfolio.local.yaml \
  --date 2026-04-24 \
  --amount 25 \
  --type fee
```

## List Cashflows

```bash
uv run --extra supabase stock-analysis list-cashflows \
  --config configs/portfolio.local.yaml \
  --from 2026-04-01 \
  --to 2026-04-30
```

## Import A Portfolio Snapshot

Holdings files can be CSV or Parquet and must include `ticker` and `market_value`. Optional columns are `quantity` and `price`.

Example CSV:

```csv
ticker,market_value,quantity,price
AAPL,400,2,200
MSFT,350,1,350
```

Import the snapshot:

```bash
uv run --extra supabase stock-analysis import-portfolio-snapshot \
  --config configs/portfolio.local.yaml \
  --date 2026-04-24 \
  --cash-balance 250 \
  --holdings data/manual/current_holdings.csv
```

If no holdings file is available, provide the account-level market value directly:

```bash
uv run --extra supabase stock-analysis import-portfolio-snapshot \
  --config configs/portfolio.local.yaml \
  --date 2026-04-24 \
  --market-value 1000 \
  --cash-balance 250
```

Account-level market value snapshots without holdings are useful for performance-only history, but not for live recommendations. A live recommendation run with `market_value > 0` requires ticker-level holdings so the optimizer can tell invested value apart from available cash. For an all-cash account, set `--market-value 0` and pass the cash balance.

## Check Latest Snapshot

```bash
uv run --extra supabase stock-analysis show-latest-portfolio-snapshot \
  --config configs/portfolio.local.yaml \
  --as-of-date 2026-04-24
```

## Run Live Recommendations

When `live_account.cashflow_source: actual` is set, `run-one-shot` loads the latest portfolio snapshot on or before the market data date, applies settled cashflows after that snapshot, and uses that amount as the rebalance contribution. The monthly deposit assumption remains for backtesting and scenario mode.

```bash
uv run --extra supabase stock-analysis run-one-shot \
  --config configs/portfolio.local.yaml \
  --forecast-engine ml
```

If withdrawals, fees, or taxes create a negative net cashflow after the latest snapshot, import a fresh snapshot before running recommendations. This prevents the live flow from estimating post-withdrawal holdings incorrectly.

Recommended cadence:

- Register deposits as they happen.
- Import a fresh holdings snapshot after market close when you want measured return tracking.
- Run recommendations after the fresh snapshot and after any deposit you want considered in the next rebalance.

The live run writes the standard recommendation outputs plus these Tableau-ready gold tables:

- `cashflows`
- `portfolio_snapshots`
- `holding_snapshots`
- `recommendation_runs`
- `recommendation_lines`
- `performance_snapshots`

CSV mirrors are written under `data/runs/<run_id>/gold/csv/`. `export-tableau` also mirrors these tables when they exist in an existing run.

For live account runs, `recommendation_runs`, `recommendation_lines`, and `performance_snapshots` are also persisted back to Supabase after the local gold artifacts are written.

Recommendation lines now include forecast outcome fields:

- `forecast_horizon_days`
- `forecast_start_date`
- `forecast_end_date`
- `realized_return`
- `realized_spy_return`
- `realized_active_return`
- `forecast_error`
- `forecast_hit`
- `outcome_status`

For the latest run, these fields are usually `pending` because the 5-trading-day horizon has not elapsed. When `export-tableau` is run later with enough price history, the historical recommendation export recalculates realized horizon outcomes from the available adjusted-close data.

For Supabase-backed accounts, `export-tableau` also emits account-level history tables when credentials are available:

- `cashflows_history`
- `portfolio_snapshots_history`
- `holding_snapshots_history`
- `recommendation_runs_history`
- `recommendation_lines_history`
- `performance_snapshots_history`

Use `recommendation_lines_history` for Tableau views that need recommendation history by ticker, action, model version, run date, forecast score, realized return, active return versus SPY, and hit/miss status. Use the single-run `recommendation_lines` table only for the current run.

`performance_snapshots` uses actual imported portfolio snapshots as valuation points. If you want return tracking to be meaningful, import a fresh snapshot after market close before running the recommendation flow. Cashflows after the latest snapshot are still used for recommendations, but they are not a substitute for an updated valuation snapshot.

## Operational Notes

Supabase is the source for accounts, cashflows, portfolio snapshots, and holdings, and it stores live recommendation and performance history after each successful live run.

The generated Hyper extract contains `portfolio_dashboard_mart` plus account tracking tables when they exist for the run. CSV and Parquet mirrors remain available under `data/runs/<run_id>/gold/`.

Snapshot import inserts the portfolio snapshot and holding rows in separate Supabase calls, with compensating cleanup if holding insertion fails. If both holding insertion and cleanup fail, inspect the snapshot before using it for live recommendations.
