# Stock Analysis

One-shot, end-of-day S&P 500 portfolio assistant.

The MVP retrieves a current S&P 500 universe, loads free historical end-of-day prices, builds medallion data outputs, forecasts portfolio scores, optimizes a long-only trade-aware portfolio in Python, logs enabled runs to MLflow, and emits Tableau Prep-ready dashboard marts.

## Quickstart

```bash
uv sync --extra mlflow --dev
uv run stock-analysis run-one-shot --config configs/portfolio.yaml
```

Generated data is written under `data/runs/<run_id>/` and ignored by git.

## Validation

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest
```

## Tableau

The Python pipeline writes CSV mirrors for Tableau Prep under each run directory. Tableau Hyper and Tableau Server publishing are optional because they require Tableau-specific SDKs and credentials.

Generate a starter Tableau workbook connected to the published dashboard mart:

```bash
uv run stock-analysis generate-tableau-workbook --config configs/portfolio.local.yaml
```

Default output is `tableau/workbooks/portfolio_recommendations.twb`.

See:

- `runbooks/full-execution.md`
- `runbooks/tableau-prep.md`
- `runbooks/tableau-server-publish.md`
- `tableau/prep/portfolio_dashboard_mart.flow-spec.md`
- `docs/forecasting-optimization-methodology.md`
- `docs/tableau-dashboard-design.md`
- `docs/tableau-dashboard-build-plan.md`
- `tableau/workbooks/README.md`
- `docs/ml-upgrade-plan.md`

## Data Caveat

The default price provider uses `yfinance`, which is convenient and free for prototyping but unofficial. For production use, replace it with a licensed market data provider.
