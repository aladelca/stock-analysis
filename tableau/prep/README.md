# Tableau Prep Transformation Path

Tableau Prep consumes Python-generated CSV files and creates the dashboard mart.

Required inputs:

- `data/runs/<run_id>/bronze/csv/sp500_constituents.csv`
- `data/runs/<run_id>/silver/csv/asset_daily_features.csv`
- `data/runs/<run_id>/gold/csv/portfolio_recommendations.csv`
- `data/runs/<run_id>/gold/csv/portfolio_risk_metrics.csv`
- `data/runs/<run_id>/gold/csv/sector_exposure.csv`

Expected output:

- `tableau_prep_outputs/portfolio_dashboard_mart.hyper`

The optimizer, forecast model, price ingestion, and medallion logic stay in Python. Prep owns only final dashboard shaping: joins, renames, type checks, and dashboard-facing output.
