# Tableau Workbooks

This directory contains generated Tableau workbook XML files.

Generate the starter workbook:

```bash
uv run stock-analysis generate-tableau-workbook --config configs/portfolio.local.yaml
```

Default output:

```text
tableau/workbooks/portfolio_recommendations.twb
```

The generated workbook connects to the published Tableau datasource named
`portfolio_dashboard_mart`. Open the `.twb` in Tableau Desktop, sign in to the
configured Tableau Cloud site, and refine the sheets/dashboard visually.

The generator creates:

- KPI sheets for market data date, holdings, forecast score, volatility,
  return/vol, and weight sum.
- A holdings bar sheet.
- A sector allocation sheet.
- A risk/forecast scatter sheet.
- A freshness footer.
- A fixed-size `Portfolio Recommendations` dashboard.

Publish after local validation:

```bash
uv run stock-analysis publish-tableau-workbook tableau/workbooks/portfolio_recommendations.twb --config configs/portfolio.local.yaml
```

Keep this file under version control as the base template. After Tableau Desktop
refinements, save the workbook back as `.twb` so the XML remains diffable.
