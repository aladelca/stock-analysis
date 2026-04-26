# Supabase Account Tracking

This runbook covers Supabase-backed account setup, cashflow registration, portfolio snapshots, and live one-shot recommendations that use actual cashflows.

## Setup

Install the optional Supabase client extra:

```bash
uv sync --extra supabase
```

Apply the SQL migration in `supabase/migrations/202604260001_account_tracking.sql` through your Supabase project, or with the Supabase CLI if you manage migrations there.

Set credentials in `.env` or your shell:

```bash
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
```

Use the service role key only from local/server jobs. Do not expose it in Tableau, browser code, or committed config files.

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

## Register The Account

```bash
uv run --extra supabase stock-analysis upsert-account \
  --config configs/portfolio.local.yaml \
  --account-slug main \
  --display-name "Main Brokerage" \
  --benchmark-ticker SPY
```

## Register A Deposit

Deposits can be registered on any calendar date. The command stores deposits as positive cashflows.

```bash
uv run --extra supabase stock-analysis register-cashflow \
  --config configs/portfolio.local.yaml \
  --date 2026-04-24 \
  --amount 500 \
  --type deposit
```

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

The live run writes the standard recommendation outputs plus these Tableau-ready gold tables:

- `cashflows`
- `portfolio_snapshots`
- `holding_snapshots`
- `recommendation_runs`
- `recommendation_lines`
- `performance_snapshots`

CSV mirrors are written under `data/runs/<run_id>/gold/csv/`. `export-tableau` also mirrors these tables when they exist in an existing run.

`performance_snapshots` uses actual imported portfolio snapshots as valuation points. If you want return tracking to be meaningful, import a fresh snapshot after market close before running the recommendation flow. Cashflows after the latest snapshot are still used for recommendations, but they are not a substitute for an updated valuation snapshot.
