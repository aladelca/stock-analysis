# Portfolio Optimizer Final Idea

## Tema o pregunta central

Build an agentic, one-shot, end-of-day portfolio assistant for the S&P 500 that retrieves free market data, prepares it with a medallion data model, optimizes a long-only portfolio in Python, and serves Tableau Server dashboards through Tableau Prep-transformed marts.

## Objetivo

Create a practical MVP that recommends a long-only S&P 500 allocation after market close. The first version should produce target weights, buy/sell signals, risk metrics, and Tableau-ready outputs without broker execution, intraday data, or paid APIs.

## Lo que ya esta claro

- The first version is a one-shot run, not a scheduled live service.
- The asset universe is the current S&P 500 constituent list.
- The portfolio is long-only.
- Optimization stays outside Tableau Prep and is implemented in Python.
- Tableau Prep is used only for transformation and dashboard mart shaping.
- Tableau Server is available as the dashboard publishing and consumption layer.
- Free data sources are required for the MVP.

## Restricciones y criterios

- Data must be auditable and reproducible enough to support portfolio recommendations.
- Raw source payloads should be stored unchanged before transformation.
- The Python path should remain the canonical path for tested logic, forecasting, and optimization.
- Tableau Prep should consume already parsed bronze/silver files and emit dashboard-facing marts.
- The MVP should avoid X/Grok data because reliable social ingestion is not free.
- The MVP should avoid real-time prices because free real-time market data is limited, unofficial, or license-sensitive.
- Outputs should be simple enough to inspect manually in Tableau Desktop or Tableau Server.

## Hallazgos de research

- S&P Global describes the S&P 500 as a large-cap U.S. equity index covering about 80% of available market capitalization, with 503 constituents reported on its official index page during research on April 24, 2026.
- Wikipedia exposes a practical free S&P 500 constituents table that can be ingested with `pandas.read_html`, but S&P Global remains the authoritative source.
- Alpha Vantage offers broad market and news APIs but its free tier is limited to 25 requests per day, which is too restrictive for full S&P 500 daily ingestion.
- SEC EDGAR APIs are free, require no API key for public company submissions and XBRL data, and require fair-access behavior including a declared User-Agent and a 10 requests/second maximum.
- Tableau Prep can call Python through TabPy script steps, but this should be reserved for Prep-specific transformations, not optimization.
- Tableau Prep Builder supports running flows from the command line, which makes one-shot refreshes possible on a machine with Prep Builder installed.
- Tableau MCP is official and useful for integrating agents with Tableau Server or Cloud, but it should not replace the deterministic data pipeline.
- Tableau Hyper API can create `.hyper` extracts for Tableau consumption and is the right publishing bridge from Python outputs to Tableau.

## Direccion recomendada

### Arquitectura general

Use a layered pipeline with clear responsibility boundaries:

```text
extract_agent
  -> raw source files

python_transform_agent
  -> bronze parsed tables
  -> silver analytic tables
  -> canonical gold optimizer tables

tableau_prep_transform_agent
  -> Tableau Prep flow over bronze/silver/gold CSV or Hyper inputs
  -> dashboard marts

forecast_agent
  -> expected return and risk estimates

optimizer_agent
  -> long-only target weights and trade recommendations

tableau_agent
  -> Hyper extract creation
  -> Tableau Server publishing
  -> dashboard refresh support
```

### Data medallion model

The first version should keep the medallion model small:

```text
data/raw/
  runs/<run_id>/raw/sp500_constituents/
  runs/<run_id>/raw/prices/

data/bronze/
  runs/<run_id>/bronze/sp500_constituents.parquet
  runs/<run_id>/bronze/daily_prices.parquet

data/silver/
  runs/<run_id>/silver/asset_daily_returns.parquet
  runs/<run_id>/silver/asset_daily_features.parquet
  runs/<run_id>/silver/asset_universe_snapshot.parquet

data/gold/
  runs/<run_id>/gold/optimizer_input.parquet
  runs/<run_id>/gold/portfolio_recommendations.parquet
  runs/<run_id>/gold/portfolio_risk_metrics.parquet
  runs/<run_id>/gold/sector_exposure.parquet
  runs/<run_id>/gold/tableau_dashboard_mart.hyper

tableau_prep_outputs/
  portfolio_dashboard_mart.hyper
```

### Free MVP data sources

Use this priority order:

1. S&P 500 constituents from Wikipedia for MVP convenience, snapshotted with `as_of_date`.
2. Historical end-of-day prices from `yfinance` or Stooq for one-shot prototyping.
3. SEC EDGAR later for filing event signals.
4. FRED later for macro regime features.
5. Alpha Vantage only as optional enrichment for a small subset because of the 25 requests/day limit.

### Python optimization path

The Python optimizer should consume `gold/optimizer_input.parquet` and produce recommendation tables.

Initial constraints:

```text
0 <= weight_i <= 0.05
sum(weight_i) = 1
long-only
fully invested
exclude assets with insufficient history
```

Initial objective:

```text
maximize expected_return - risk_aversion * portfolio_variance
```

Initial forecast can be intentionally simple:

```text
expected_return = momentum_score - volatility_penalty
risk = covariance matrix from daily returns
```

Use CVXPY for optimization because it can support future constraints such as sector caps, turnover, transaction costs, and minimum trade sizes.

### Tableau Prep path

Tableau Prep should be a dashboard mart transformation path, not the source of portfolio logic.

Inputs:

```text
data/runs/<run_id>/bronze/csv/sp500_constituents.csv
data/runs/<run_id>/silver/csv/asset_daily_features.csv
data/runs/<run_id>/gold/csv/portfolio_recommendations.csv
data/runs/<run_id>/gold/csv/portfolio_risk_metrics.csv
data/runs/<run_id>/gold/csv/sector_exposure.csv
```

Prep responsibilities:

```text
rename dashboard-facing fields
standardize data types
join recommendations with company metadata
join sector exposure with portfolio metrics
create clean Tableau dashboard mart
output portfolio_dashboard_mart.hyper
```

Prep should not own:

```text
price retrieval
feature backtesting
forecast model logic
optimization
portfolio constraint solving
model evaluation
```

TabPy can be added later for light transformation functions inside Prep, but the MVP should avoid using TabPy unless Prep cannot express a required transformation.

### Tableau Server layer

Use Tableau Server for dashboard consumption and published data sources.

Initial dashboards:

```text
portfolio target weights
top buy and sell recommendations
sector allocation
risk and return summary
asset-level signal table
data freshness and run metadata
```

The Python pipeline should generate `.hyper` extracts using Tableau Hyper API. Tableau Prep can also output `.hyper` marts. Tableau Server publishing can be automated later with Tableau REST API or Tableau MCP.

## Recomendacion provisional

Start with a deterministic Python-first pipeline and add Tableau Prep as a visual transformation lane for dashboard marts. This keeps the quant logic testable and reproducible while still satisfying the requirement to use Tableau Prep for transformations.

The critical assumption is that free EOD price ingestion through `yfinance` or Stooq is acceptable for an MVP. If this becomes a production portfolio assistant, price data licensing and reliability need to be revisited.

## Preguntas abiertas

- Should the first optimizer use all S&P 500 stocks or only the top N by market cap/liquidity to reduce data quality issues?
- Should the MVP include a cash allocation, or force 100% investment?
- Should max single-name weight remain 5%, or match a benchmark-aware cap?
- Should Tableau Prep output be considered authoritative for dashboards, or should dashboards also be able to connect directly to Python-generated gold tables?

## Proximos pasos

1. Scaffold the Python project with modules for ingestion, medallion transforms, features, optimization, and Tableau export.
2. Create a one-shot CLI command such as `stock-analysis run-one-shot --universe sp500`.
3. Implement S&P 500 universe snapshot ingestion.
4. Implement free EOD price ingestion and raw/bronze/silver/gold outputs.
5. Implement baseline momentum-risk optimizer with CVXPY.
6. Export gold CSV files for Tableau Prep.
7. Create the first Tableau Prep flow that emits `portfolio_dashboard_mart.hyper`.
8. Add a runbook for executing Python, running Tableau Prep CLI, and publishing to Tableau Server.

## Fuentes

- [S&P 500 official index page](https://www.spglobal.com/spdji/en/indices/equity/sp-500/)
- [Wikipedia S&P 500 constituents table](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
- [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)
- [Alpha Vantage premium and free limit information](https://www.alphavantage.co/premium/)
- [SEC EDGAR data access and fair access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [Tableau Prep Python and TabPy script steps](https://help.tableau.com/current/prep/en-us/prep_scripts_TabPy.htm)
- [Tableau Prep command line flow refresh](https://help.tableau.com/current/prep/en-us/prep_run_commandline.htm)
- [Tableau MCP server](https://github.com/tableau/tableau-mcp)
- [Tableau Hyper API](https://www.tableau.com/developer/tools/hyper-api)
