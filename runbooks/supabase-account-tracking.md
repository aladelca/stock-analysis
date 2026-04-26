# Supabase Account Tracking

This runbook covers the implemented account-tracking foundation. Live one-shot recommendations are not wired to actual Supabase cashflows yet; that work is tracked by `stock-analysis-1kn`.

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

## Current Limitation

The database and registration commands are ready, but `run-one-shot` still uses the current scenario-style contribution setting. The next implementation step is to load the latest Supabase snapshot plus unapplied settled cashflows and pass that live state into the recommendation flow.
