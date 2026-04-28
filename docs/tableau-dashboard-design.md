# Tableau Dashboard Design

## Purpose

The dashboard should answer three questions from a single screen:

1. What should I hold?
2. Why these names?
3. Is this portfolio risky or concentrated?

The current data model is one run, one decision. It is a recommendation dashboard, not a backtest dashboard.

## Data Source

Use the published Tableau Cloud datasource:

```text
portfolio_dashboard_mart
```

The starter workbook can be generated with:

```bash
uv run stock-analysis generate-tableau-workbook --config configs/portfolio.local.yaml
```

Use that `.twb` as the base, then refine in Tableau Desktop.

This datasource is generated from:

```text
data/runs/<run_id>/gold/tableau_dashboard_mart.hyper
```

Current grain:

```text
one row per S&P 500 ticker for one run
```

Selected holdings have:

```text
selected = true
action = BUY
target_weight > 0
```

Excluded tickers have:

```text
selected = false
action = EXCLUDE
target_weight = 0 or below display threshold
```

## Layout

Use a single dashboard, fixed at `1280 x 850` for the generated base workbook.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ KPI STRIP: as-of · holdings · forecast score · vol · return/vol · Σw    │
├───────────────────────────────────────────┬─────────────────────────────┤
│                                           │                             │
│ HOLDINGS TABLE                            │ SECTOR TREEMAP              │
│ ticker · security · sector · weight bar   │ sector exposure             │
│ forecast_score · volatility · action      │                             │
│ sorted by target_weight desc              ├─────────────────────────────┤
│                                           │                             │
│                                           │ RISK / FORECAST SCATTER     │
│                                           │ x=volatility                │
│                                           │ y=forecast_score            │
│                                           │ size=scatter_size           │
│                                           │ color=sector/selection      │
├───────────────────────────────────────────┴─────────────────────────────┤
│ FOOTER: run_id · requested date · data date · date status · config hash  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Design Principles

- Use one screen. Avoid tabs until there is historical data.
- Treat `forecast_score` as a ranking signal. Treat `calibrated_expected_return` as an expected
  5-trading-day return only when `expected_return_is_calibrated = true`.
- Make concentration visible. The treemap should make dominant sectors obvious.
- Keep selected and excluded assets in the same scatter so the recommendation context is visible.
- Use the footer to make run freshness auditable.

## KPI Strip

Create six single-value sheets.

| Tile | Formula |
| --- | --- |
| Data as-of | `MAX([run_data_as_of_date])` |
| # holdings | `MAX([portfolio_num_holdings])` |
| Portfolio forecast score | `MAX([portfolio_expected_return])` |
| Portfolio volatility | `MAX([portfolio_expected_volatility])` |
| Return/vol | `MAX([portfolio_return_per_vol])` |
| Weight sum | `SUM([target_weight])` |

Formatting:

- Use two decimals for score and volatility.
- Use percentage format for target weights.
- Color weight sum green when it is within `[0.999, 1.001]`; otherwise red.
- If `is_data_date_lagged` is true, show a small warning marker near the data-as-of tile.

## Holdings Table

Filter:

```text
selected = true
```

Rows:

- `ticker`
- `security`
- `gics_sector`

Displayed measures:

- `target_weight` as a horizontal bar.
- `target_weight_label` as text.
- `forecast_score`, formatted to two decimals, plus calibrated expected return in percent when
  available.
- `volatility`, formatted to two decimals.
- `action`.
- `reason_code` in tooltip.

Sort:

```text
target_weight desc
```

Default visible rows:

```text
Top 20, scrollable
```

## Sector Treemap

Use the same table, not a separate sector table.

Dimension:

```text
gics_sector
```

Size:

```text
SUM([target_weight])
```

Color:

```text
SUM([target_weight])
```

Tooltip:

- `gics_sector`
- `SUM(target_weight)`
- `MAX(sector_target_weight)`

Do not use `SUM([sector_target_weight])` because that field is repeated on each ticker row.

## Risk / Forecast Scatter

Use all rows in the mart.

Columns:

```text
volatility
```

Rows:

```text
forecast_score
```

Marks:

- Detail: `ticker`
- Tooltip: `ticker`, `security`, `gics_sector`, `target_weight`, `action`, `reason_code`
- Size: `scatter_size`
- Color: `gics_sector`
- Shape or opacity: use `selected` to distinguish selected vs excluded names

Recommended styling:

- Selected assets: full opacity.
- Excluded assets: 20-30% opacity or gray.
- Reference line on median `volatility`.
- Reference line on median `forecast_score`.

This scatter is now feasible without a future mart extension because the current Python mart contains all 503 S&P 500 rows and a `selected` field.

## Freshness Footer

Use a compact text sheet with:

- `run_id`
- `run_requested_as_of_date`
- `run_data_as_of_date`
- `data_date_status`
- `run_config_hash_short`

If `is_data_date_lagged` is true, the footer should make clear that the latest market data date differs from the requested run date.

## Tooltips

For holdings and scatter marks, include:

- Ticker and company name.
- Sector.
- Current weight.
- Target weight.
- Trade weight.
- Estimated commission.
- Forecast score.
- Volatility.
- Action.
- Reason code.
- Run id.
- Data as-of date.

## Current V1 Caveats

- BUY, SELL, and HOLD are portfolio-weight rebalance recommendations, not broker-submitted orders.
- Current holdings must be supplied for true sell/hold semantics. Without a holdings file, the run
  is first-allocation mode and most selected assets appear as BUY.
- EXCLUDE means no current position and no target allocation.
- `forecast_score` should not be labeled as a predicted percentage return. Use
  `calibrated_expected_return` for expected-return labels only when the run calibration status is
  `calibrated`.
- No time-series chart should be added until multi-run history is intentionally modeled.

## Future Enhancements

- Add sector cap reference lines.
- Add benchmark-relative diagnostics.
- Add a historical run table for trend charts.

## Validation Checklist

- [ ] KPI strip renders six non-null tiles.
- [ ] `SUM([target_weight])` reads approximately `1.00`.
- [ ] Holdings table only shows `selected = true`.
- [ ] Treemap sector weights sum to approximately `1.00`.
- [ ] Scatter includes selected and excluded tickers.
- [ ] Selected scatter marks are visually distinct from excluded marks.
- [ ] Footer `run_id` matches the published run.
- [ ] Footer shows both requested and actual market data dates.
- [ ] Dashboard loads in under three seconds.
