# One-Shot S&P 500 Portfolio Assistant

## Goal

- Implement a one-shot, end-of-day, long-only S&P 500 portfolio assistant that retrieves free market data, writes auditable medallion data outputs, computes baseline forecast/risk signals, optimizes target weights in Python, and prepares Tableau Server-facing dashboard marts through Tableau Prep transformations.

## Request Snapshot

- User request: "Based on the final idea, create a plan and include everything needed to implement it."
- Owner or issue: `stock-analysis-94v`
- Plan file: `plans/20260423-2357-one-shot-sp500-portfolio-assistant.md`

## Current State

- The repo is effectively empty for application code. Current files are `AGENTS.md`, `CLAUDE.md`, and `docs/portfolio-optimizer-final-idea.md`.
- There is no `pyproject.toml`, package layout, test suite, Ruff config, mypy config, CLI entry point, or Tableau artifact directory.
- `docs/portfolio-optimizer-final-idea.md` defines the agreed architecture: one-shot run, current S&P 500 universe, long-only portfolio, free data sources, Python optimization, Tableau Prep only for transformation/dashboard marts, and Tableau Server as the visualization target.
- `uv` is available at `/Users/adrianalarcon/.local/bin/uv`; local Python is `Python 3.12.0`.
- No git remote is configured, so session-level push requirements cannot be satisfied until a remote is added.
- Beads is active and must be used for task tracking.

## Findings

- The implementation needs project bootstrap work before feature code can exist: packaging, dependency management, tooling, tests, and runtime config.
- Python should be the canonical path for ingestion, medallion writes, feature engineering, forecasting, and optimization because those pieces need unit tests and type checks.
- Tableau Prep should operate on exported CSV/Hyper files from the Python pipeline and should not own optimization, forecast logic, backtesting, or market data retrieval.
- The first implementation should default to a fully invested long-only portfolio with max single-name weight of `5%`, using all S&P 500 tickers that pass data quality filters.
- Free price ingestion should be abstracted behind a provider interface because `yfinance` is convenient but unofficial; Stooq can be added as a fallback without changing downstream contracts.
- Tableau Prep flow generation can be handled in two layers: a checked-in flow specification/runbook for human-editable Prep work and an optional `cwprep`-based generator if the library proves stable in local testing.

## Scope

### In scope

- Bootstrap a Python 3.12 project with `uv`, Ruff, mypy, pytest, and typed package code.
- Implement a one-shot CLI command: `stock-analysis run-one-shot --config configs/portfolio.yaml`.
- Ingest the current S&P 500 universe from the Wikipedia constituents table and snapshot it with `as_of_date`.
- Ingest historical end-of-day adjusted prices for the S&P 500 universe from a free provider, starting with `yfinance`.
- Persist medallion outputs under `data/raw`, `data/bronze`, `data/silver`, and `data/gold`.
- Produce CSV exports for Tableau Prep inputs.
- Compute baseline price-derived features: returns, volatility, momentum, drawdown, moving averages, and data quality flags.
- Estimate simple expected returns and risk from momentum and covariance.
- Optimize a fully invested, long-only portfolio in Python with CVXPY.
- Generate gold recommendation outputs: optimizer input, target weights, buy/sell recommendations, risk metrics, sector exposure, run metadata, and Tableau Prep input CSVs.
- Add Tableau Hyper export for Python-generated gold outputs.
- Add Tableau Prep flow specification and runbook for dashboard mart transformations.
- Add a Tableau Server publishing module and command surface that can be configured later with credentials.
- Add unit and integration tests for ingestion parsing, medallion contracts, feature engineering, optimizer constraints, and one-shot orchestration with mocked providers.
- Add runbooks for one-shot execution, Tableau Prep CLI execution, and Tableau Server publishing.

### Out of scope

- Broker integration or real trade execution.
- Intraday data, real-time prices, or streaming.
- Paid APIs, X/Grok social ingestion, or Alpha Vantage full S&P 500 ingestion.
- Kaggle competition replication.
- Deep learning or advanced NLP signals.
- Production scheduling, cron, Airflow, Dagster, or cloud deployment.
- Tableau dashboard design in Desktop beyond generated data sources, dashboard mart contracts, and build guidance.
- Short selling, leverage, options, derivatives, and cash allocation.

## File Plan

| Path | Action | Details |
| --- | --- | --- |
| `.python-version` | create | Pin local Python to `3.12` for `uv` consistency. |
| `.gitignore` | modify | Ensure generated data, local Tableau outputs, virtualenvs, caches, logs, and credentials are ignored while keeping source configs and plan docs tracked. |
| `pyproject.toml` | create | Define package metadata, Python version, dependencies, optional Tableau extras, Ruff config, mypy config, pytest config, and console script `stock-analysis`. |
| `README.md` | create | Document project purpose, one-shot MVP flow, install commands, quickstart, and data/source caveats. |
| `configs/portfolio.yaml` | create | Runtime config for universe source, price provider, lookback window, max weight, risk aversion, output paths, Tableau settings, and publishing toggles. |
| `configs/tableau.example.yaml` | create | Non-secret example config for Tableau Server URL, site, project, datasource names, and Prep CLI path. |
| `.env.example` | create | Document optional env vars such as `TABLEAU_SERVER_URL`, `TABLEAU_SITE_NAME`, `TABLEAU_PAT_NAME`, `TABLEAU_PAT_VALUE`, and `SEC_USER_AGENT`. |
| `src/stock_analysis/__init__.py` | create | Package marker and version export. |
| `src/stock_analysis/cli.py` | create | Typer CLI with `run-one-shot`, `export-tableau`, and `publish-tableau` commands. |
| `src/stock_analysis/config.py` | create | Pydantic settings models for portfolio, data, optimizer, output, and Tableau configuration. |
| `src/stock_analysis/logging.py` | create | Structured logging setup for one-shot runs and data freshness diagnostics. |
| `src/stock_analysis/paths.py` | create | Central helpers for resolving raw/bronze/silver/gold/tableau paths by `as_of_date` and `run_id`. |
| `src/stock_analysis/domain/models.py` | create | Typed dataclasses or Pydantic models for `UniverseMember`, `PriceBar`, `PortfolioRecommendation`, `RiskMetric`, and `RunMetadata`. |
| `src/stock_analysis/domain/schemas.py` | create | Pandera or explicit schema definitions for medallion tables and gold contracts. |
| `src/stock_analysis/ingestion/universe.py` | create | Load and normalize current S&P 500 constituents from Wikipedia, including ticker normalization for provider compatibility. |
| `src/stock_analysis/ingestion/prices.py` | create | Provider interface and `YFinancePriceProvider` implementation for EOD adjusted price retrieval. |
| `src/stock_analysis/ingestion/raw_store.py` | create | Persist raw source payloads and metadata without mutation. |
| `src/stock_analysis/medallion/bronze.py` | create | Convert raw universe and price payloads into typed bronze Parquet/CSV tables. |
| `src/stock_analysis/medallion/silver.py` | create | Build cleaned asset/day price, return, and feature base tables with data quality filters. |
| `src/stock_analysis/features/price_features.py` | create | Compute momentum, volatility, drawdown, moving averages, beta proxy, and sufficient-history flags. |
| `src/stock_analysis/forecasting/baseline.py` | create | Generate baseline expected return and risk inputs from price-derived features. |
| `src/stock_analysis/optimization/constraints.py` | create | Define optimizer constraints: long-only, fully invested, max single-name weight, and eligible assets only. |
| `src/stock_analysis/optimization/engine.py` | create | CVXPY optimizer that maximizes expected return minus risk penalty and returns target weights. |
| `src/stock_analysis/optimization/recommendations.py` | create | Convert weights into recommendations, reason codes, risk summaries, and sector exposure. |
| `src/stock_analysis/pipeline/one_shot.py` | create | Orchestrate the full one-shot flow from ingestion through gold outputs and Tableau exports. |
| `src/stock_analysis/io/parquet.py` | create | Read/write helpers for Parquet with stable schemas and deterministic column ordering. |
| `src/stock_analysis/io/csv.py` | create | CSV export helpers for Tableau Prep inputs. |
| `src/stock_analysis/tableau/hyper.py` | create | Create Tableau `.hyper` extracts from gold tables using Tableau Hyper API when dependency is installed. |
| `src/stock_analysis/tableau/publish.py` | create | Tableau Server publishing wrapper using Tableau Server Client or REST API, initially guarded by config flags. |
| `src/stock_analysis/tableau/prep_contract.py` | create | Define Prep input/output file names, required columns, field display names, and dashboard mart contracts. |
| `src/stock_analysis/tableau/prep_flow_generator.py` | create | Optional `cwprep`-based generator for a starter `.tfl` flow if `cwprep` is available; fail gracefully when not installed. |
| `tableau/prep/README.md` | create | Explain the Prep transformation path, input files, output mart, CLI execution, and boundaries. |
| `tableau/prep/portfolio_dashboard_mart.flow-spec.md` | create | Human-readable flow spec: inputs, joins, calculated fields, renamed fields, outputs, and validation checklist. |
| `tableau/prep/portfolio_dashboard_mart.tfl` | create | Starter Tableau Prep flow artifact if generated successfully with `cwprep`; otherwise tracked in a follow-up once validated in Prep Builder. |
| `tableau/server/README.md` | create | Document Tableau Server publishing setup, PAT authentication, datasource names, and refresh workflow. |
| `runbooks/one-shot-sp500.md` | create | Operator runbook for local setup, one-shot execution, expected outputs, and troubleshooting. |
| `runbooks/tableau-prep.md` | create | Operator runbook for opening/running the Prep flow manually and via Prep CLI. |
| `runbooks/tableau-server-publish.md` | create | Operator runbook for publishing Hyper extracts/data sources to Tableau Server. |
| `tests/conftest.py` | create | Shared fixtures for temp directories, sample universe, sample prices, and config overrides. |
| `tests/fixtures/sp500_sample.html` | create | Minimal Wikipedia-like S&P 500 table fixture. |
| `tests/fixtures/prices_sample.csv` | create | Deterministic adjusted close/OHLCV fixture for multiple tickers. |
| `tests/unit/test_config.py` | create | Validate config defaults, path resolution, and invalid config failures. |
| `tests/unit/test_universe_ingestion.py` | create | Validate S&P 500 table parsing, ticker normalization, and raw snapshot metadata. |
| `tests/unit/test_price_features.py` | create | Validate return, momentum, volatility, drawdown, moving average, and missing-data behavior. |
| `tests/unit/test_baseline_forecast.py` | create | Validate expected return and risk input generation from features. |
| `tests/unit/test_optimizer.py` | create | Validate CVXPY constraints: weights sum to one, weights are non-negative, and max weight is respected. |
| `tests/unit/test_recommendations.py` | create | Validate recommendation rows, reason codes, risk metrics, and sector exposure outputs. |
| `tests/unit/test_tableau_contract.py` | create | Validate Tableau Prep input/export contract columns and field names. |
| `tests/integration/test_one_shot_pipeline.py` | create | Run the full one-shot pipeline with mocked/sample providers and assert all medallion/gold files exist with valid schemas. |
| `docs/portfolio-optimizer-final-idea.md` | keep | Source architecture decision already created; update only if implementation decisions diverge. |
| `plans/20260423-2357-one-shot-sp500-portfolio-assistant.md` | create | This implementation plan. |

## Data and Contract Changes

- CLI contract:
  - `stock-analysis run-one-shot --config configs/portfolio.yaml`
  - `stock-analysis export-tableau --config configs/portfolio.yaml --run-id <run_id>`
  - `stock-analysis publish-tableau --config configs/portfolio.yaml --run-id <run_id>`
- Config contract in `configs/portfolio.yaml`:
  - `run.as_of_date`: optional date; default is current local date.
  - `run.output_root`: default `data`.
  - `universe.provider`: default `wikipedia_sp500`.
  - `prices.provider`: default `yfinance`.
  - `prices.lookback_years`: default `5`.
  - `features.min_history_days`: default `252`.
  - `optimizer.max_weight`: default `0.05`.
  - `optimizer.risk_aversion`: default configurable numeric value.
  - `optimizer.fully_invested`: default `true`.
  - `tableau.export_csv`: default `true`.
  - `tableau.export_hyper`: default `false` unless Hyper dependency is installed.
  - `tableau.publish_enabled`: default `false`.
- Raw outputs:
  - `data/raw/sp500_constituents/as_of_date=<YYYY-MM-DD>/source.html`
  - `data/raw/sp500_constituents/as_of_date=<YYYY-MM-DD>/metadata.json`
  - `data/raw/prices/as_of_date=<YYYY-MM-DD>/<ticker>.json` or provider-native serialized payload
  - `data/raw/prices/as_of_date=<YYYY-MM-DD>/metadata.json`
- Bronze outputs:
  - `data/bronze/sp500_constituents.parquet`
  - `data/bronze/daily_prices.parquet`
  - CSV mirrors under `data/bronze/csv/` for Tableau Prep.
- Silver outputs:
  - `data/silver/asset_daily_returns.parquet`
  - `data/silver/asset_daily_features.parquet`
  - `data/silver/asset_universe_snapshot.parquet`
  - CSV mirrors under `data/silver/csv/` for Tableau Prep.
- Gold outputs:
  - `data/gold/optimizer_input.parquet`
  - `data/gold/portfolio_recommendations.parquet`
  - `data/gold/portfolio_risk_metrics.parquet`
  - `data/gold/sector_exposure.parquet`
  - `data/gold/run_metadata.parquet`
  - CSV mirrors under `data/gold/csv/` for Tableau Prep.
- Tableau Prep input contract:
  - `data/bronze/csv/sp500_constituents.csv`
  - `data/silver/csv/asset_daily_features.csv`
  - `data/gold/csv/portfolio_recommendations.csv`
  - `data/gold/csv/portfolio_risk_metrics.csv`
  - `data/gold/csv/sector_exposure.csv`
- Tableau Prep output contract:
  - `tableau_prep_outputs/portfolio_dashboard_mart.hyper`
  - Optional CSV debug output: `tableau_prep_outputs/portfolio_dashboard_mart.csv`.
- Environment variables:
  - `TABLEAU_SERVER_URL`, `TABLEAU_SITE_NAME`, `TABLEAU_PAT_NAME`, `TABLEAU_PAT_VALUE` for publishing.
  - `SEC_USER_AGENT` reserved for future SEC EDGAR work, not required for MVP.
- Credentials:
  - No credentials should be committed.
  - Tableau PAT values must be read from environment variables or ignored local config only.

## Implementation Steps

1. Bootstrap the Python project:
   - Add `.python-version`, `pyproject.toml`, `README.md`, `.env.example`, and `.gitignore` updates.
   - Configure runtime dependencies: `pandas`, `numpy`, `pyarrow`, `pydantic`, `pydantic-settings`, `PyYAML`, `typer`, `rich`, `yfinance`, `cvxpy`, and `scikit-learn` only if needed for covariance utilities.
   - Configure dev dependencies: `pytest`, `pytest-cov`, `ruff`, `mypy`, `types-PyYAML`, `pandas-stubs`, and `responses` or `requests-mock` if HTTP mocking is needed.
   - Add optional Tableau extras: `tableauhyperapi`, `tableauserverclient`, and `cwprep`.
2. Implement configuration and project infrastructure:
   - Create `src/stock_analysis/config.py` with strict config models and defaults.
   - Create path helpers in `src/stock_analysis/paths.py` so all layers write to deterministic locations.
   - Add logging setup in `src/stock_analysis/logging.py`.
3. Implement domain schemas:
   - Define the required columns and types for universe, prices, features, optimizer input, recommendations, risk metrics, sector exposure, and run metadata.
   - Add schema validation on every medallion boundary before writing files.
4. Implement S&P 500 universe ingestion:
   - Fetch or read the Wikipedia constituents table.
   - Store the raw HTML and source metadata.
   - Normalize columns to `ticker`, `provider_ticker`, `security`, `gics_sector`, `gics_sub_industry`, `headquarters_location`, `date_added`, `cik`, `founded`, and `as_of_date`.
   - Convert tickers such as `BRK.B` and `BF.B` to provider-compatible `BRK-B` and `BF-B` for `yfinance`, while preserving original ticker symbols.
5. Implement price ingestion:
   - Define `PriceProvider` protocol in `src/stock_analysis/ingestion/prices.py`.
   - Implement `YFinancePriceProvider` with batched downloads and retry/error capture.
   - Store raw provider payload metadata and normalized daily OHLCV/adjusted-close bars.
   - Filter out assets with insufficient history or unusable adjusted close data.
6. Implement bronze and silver medallion transforms:
   - Convert raw universe and price data into bronze Parquet and CSV mirror outputs.
   - Build silver returns and feature base tables with deterministic sorting and column ordering.
   - Attach data quality flags such as `missing_price_ratio`, `history_days`, and `eligible_for_optimization`.
7. Implement price-derived features:
   - Compute daily simple returns.
   - Compute trailing volatility over configurable windows.
   - Compute trailing momentum over configurable windows.
   - Compute max drawdown over the lookback period.
   - Compute moving average features and price-to-moving-average ratios.
8. Implement baseline forecasting:
   - Convert feature scores into `expected_return`.
   - Build historical return matrix and covariance matrix from eligible assets.
   - Save `gold/optimizer_input.parquet` with `ticker`, `expected_return`, `volatility`, `eligible_for_optimization`, `gics_sector`, and supporting signal fields.
9. Implement Python optimizer:
   - Use CVXPY to solve the long-only fully invested objective.
   - Enforce max single-name weight from config.
   - Exclude ineligible assets before optimization.
   - Raise a clear error if too few eligible assets remain or if the problem is infeasible.
10. Implement recommendation outputs:
   - Produce target weight rows for all eligible assets.
   - Generate action labels such as `BUY`, `HOLD`, and `EXCLUDE` for MVP, with `SELL` reserved for future current-portfolio inputs.
   - Produce sector exposure from target weights joined to universe metadata.
   - Produce risk metrics such as expected portfolio return, expected volatility, concentration, number of holdings, and max weight.
   - Produce run metadata with source timestamps, config hash, row counts, and warning counts.
11. Implement CSV and Hyper exports:
   - Write CSV mirrors for Tableau Prep inputs.
   - Add `src/stock_analysis/tableau/hyper.py` to export `.hyper` files when `tableauhyperapi` is available.
   - Keep Hyper export optional so local development can pass without Tableau dependencies.
12. Implement Tableau Prep artifacts:
   - Add `tableau/prep/portfolio_dashboard_mart.flow-spec.md` describing the exact Prep flow.
   - Add `tableau/prep/README.md` with required input files and output locations.
   - Add optional `src/stock_analysis/tableau/prep_flow_generator.py` to generate a starter `.tfl` with `cwprep`; if generation is unreliable, document manual Prep creation steps and do not block Python MVP.
13. Implement Tableau Server publishing:
   - Add `src/stock_analysis/tableau/publish.py` with a disabled-by-default publishing function.
   - Read Tableau credentials from environment variables.
   - Support publishing the Python-generated Hyper extract and the Prep-generated dashboard mart separately.
14. Implement CLI orchestration:
   - `run-one-shot` should execute universe ingestion, price ingestion, medallion transforms, features, forecast, optimization, recommendation outputs, CSV export, and optional Hyper export.
   - `export-tableau` should regenerate Tableau-facing CSV/Hyper outputs from an existing run.
   - `publish-tableau` should publish configured extracts only when `tableau.publish_enabled=true`.
15. Add tests and fixtures:
   - Build deterministic sample input fixtures.
   - Unit test every pure transformation and optimizer constraint.
   - Integration test the one-shot pipeline with mocked providers and temp directories.
16. Add runbooks:
   - Document setup, one-shot execution, expected outputs, Tableau Prep CLI execution, and Tableau Server publishing.
   - Include troubleshooting for missing price data, infeasible optimization, missing Tableau dependencies, and Prep CLI path issues.
17. Update the plan if implementation changes scope:
   - If `cwprep` cannot generate a usable `.tfl`, document the limitation and keep the Prep flow spec/runbook as the implementation artifact until a GUI-validated `.tfl` is checked in.

## Tests

- Unit: `tests/unit/test_config.py` should cover default config, invalid weights, missing output root, and Tableau publish disabled by default.
- Unit: `tests/unit/test_universe_ingestion.py` should cover parsing a Wikipedia-like table, preserving original tickers, and generating provider-compatible tickers.
- Unit: `tests/unit/test_price_features.py` should cover return math, rolling momentum, rolling volatility, drawdown, moving average features, and insufficient-history flags.
- Unit: `tests/unit/test_baseline_forecast.py` should cover expected return calculation, eligible universe filtering, and covariance input shape.
- Unit: `tests/unit/test_optimizer.py` should cover long-only weights, `sum(weights) = 1`, `weight <= max_weight`, ineligible asset exclusion, and infeasible problem handling.
- Unit: `tests/unit/test_recommendations.py` should cover recommendation schema, action labels, sector exposure aggregation, and risk metrics.
- Unit: `tests/unit/test_tableau_contract.py` should cover required Tableau Prep input/output columns and field display mappings.
- Integration: `tests/integration/test_one_shot_pipeline.py` should run the full pipeline using fixture data and assert raw, bronze, silver, gold, and Tableau CSV outputs are created.
- Regression: add a fixture where one asset has missing prices and verify it is excluded without failing the full run.
- Regression: add a fixture where all assets are ineligible and verify the optimizer raises a clear domain error.

## Validation

- Install/sync: `uv sync --all-extras --dev`
- Format: `uv run ruff format --check src tests`
- Lint: `uv run ruff check src tests`
- Types: `uv run mypy src`
- Tests: `uv run pytest`
- One-shot smoke test after implementation: `uv run stock-analysis run-one-shot --config configs/portfolio.yaml`
- Tableau export smoke test after implementation: `uv run stock-analysis export-tableau --config configs/portfolio.yaml --run-id <run_id>`
- Optional Tableau publish smoke test only with credentials configured: `uv run stock-analysis publish-tableau --config configs/portfolio.yaml --run-id <run_id>`

## Risks and Mitigations

- `yfinance` is unofficial and can break or throttle -> isolate it behind `PriceProvider`, persist raw metadata, and add Stooq as a future fallback.
- Full S&P 500 price retrieval can be slow or partially missing -> batch requests, capture per-ticker errors, and optimize only eligible assets.
- CVXPY solver availability can differ by machine -> declare solver dependencies clearly and add fallback solver configuration in `configs/portfolio.yaml`.
- `max_weight=0.05` requires at least 20 eligible assets to fully invest -> validate eligibility count before solving and emit a clear error.
- Tableau Hyper API may be unavailable on some machines -> make Hyper export optional and keep CSV outputs as the stable Tableau Prep input.
- `cwprep` is beta and reverse-engineered -> treat it as optional; keep a human-readable Prep flow spec and runbook as the reliable path.
- Tableau Server credentials are sensitive -> read them only from environment variables or ignored local config.
- Generated data can be large -> ignore `data/`, `tableau_prep_outputs/`, `.hyper`, and local logs in `.gitignore`.

## Open Questions

- None

## Acceptance Criteria

- `uv run stock-analysis run-one-shot --config configs/portfolio.yaml` can run end-to-end with fixture or live free data and create raw, bronze, silver, gold, and Tableau Prep input outputs.
- The optimizer produces a long-only fully invested portfolio where all weights are non-negative, sum to one, and respect the configured max single-name weight.
- The pipeline writes `portfolio_recommendations`, `portfolio_risk_metrics`, `sector_exposure`, and `run_metadata` in gold Parquet plus CSV mirrors.
- Tableau Prep has a documented transformation contract and runbook that consumes Python outputs and emits `portfolio_dashboard_mart.hyper`.
- Tableau Server publishing is implemented behind an explicit disabled-by-default config switch.
- Unit and integration tests cover parsing, transformations, forecasting inputs, optimization constraints, recommendation outputs, and one-shot orchestration.
- Ruff format, Ruff lint, mypy, and pytest commands are defined and pass.

## Definition of Done

- Python package, CLI, config, medallion pipeline, baseline forecast, optimizer, Tableau exports, Prep contracts, and runbooks are implemented.
- Generated data and secrets are ignored from git.
- Tests are added for all critical transformations and optimizer constraints.
- `uv run ruff format --check src tests`, `uv run ruff check src tests`, `uv run mypy src`, and `uv run pytest` pass.
- The implementation plan is updated if any major scope decision changes during coding.
