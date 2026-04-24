# One-Shot S&P 500 Runbook

## Setup

```bash
uv sync --dev
```

## Run

```bash
uv run stock-analysis run-one-shot --config configs/portfolio.yaml
```

## Expected Outputs

- `data/runs/<run_id>/raw/`
- `data/runs/<run_id>/raw/prices/*.csv`
- `data/runs/<run_id>/bronze/sp500_constituents.parquet`
- `data/runs/<run_id>/bronze/daily_prices.parquet`
- `data/runs/<run_id>/silver/asset_daily_returns.parquet`
- `data/runs/<run_id>/silver/asset_daily_features.parquet`
- `data/runs/<run_id>/gold/portfolio_recommendations.parquet`
- `data/runs/<run_id>/gold/portfolio_risk_metrics.parquet`
- `data/runs/<run_id>/gold/sector_exposure.parquet`
- `data/runs/<run_id>/gold/csv/*.csv`

`as_of_date` in gold outputs is the latest available market data date, not the wall-clock run date.

## Troubleshooting

- If optimization is infeasible, lower `optimizer.max_weight` only if there are enough eligible assets, or increase data quality by extending `prices.lookback_years`.
- If price download fails, rerun later or reduce the universe for local testing.
- If Tableau Hyper export is skipped, install the optional Tableau extra and enable `tableau.export_hyper`.
