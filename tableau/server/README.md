# Tableau Server Publishing

Publishing is disabled by default.

Set these environment variables before enabling publishing:

```bash
export TABLEAU_SERVER_URL="https://tableau.example.com"
export TABLEAU_SITE_NAME="example-site"
export TABLEAU_PAT_NAME="token-name"
export TABLEAU_PAT_VALUE="token-value"
```

Then set `tableau.publish_enabled: true` in a local ignored config file and run:

```bash
uv run stock-analysis publish-tableau data/runs/<run_id>/gold/tableau_dashboard_mart.hyper --config configs/portfolio.yaml
```

To publish the generated workbook:

```bash
uv run stock-analysis publish-tableau-workbook tableau/workbooks/portfolio_recommendations.twb --config configs/portfolio.yaml
```
