# Tableau Server Publish Runbook

Publishing requires Tableau credentials and the optional Tableau dependency group.

```bash
uv sync --extra tableau --dev
```

Create a local `.env` file. This file is ignored by git:

```bash
cp -f .env.example .env
```

Edit `.env`:

```text
TABLEAU_SERVER_URL=https://us-east-1.online.tableau.com
TABLEAU_SITE_NAME=aladelca
TABLEAU_PAT_NAME=your-token-name
TABLEAU_PAT_VALUE=your-token-secret
```

Create an ignored local config file:

```bash
cp -f configs/portfolio.yaml configs/portfolio.local.yaml
```

Enable publishing in `configs/portfolio.local.yaml`:

```yaml
tableau:
  export_hyper: true
  publish_enabled: true
```

Then publish:

```bash
uv run stock-analysis publish-tableau tableau_prep_outputs/portfolio_dashboard_mart.hyper --config configs/portfolio.local.yaml
```

Generate the starter workbook:

```bash
uv run stock-analysis generate-tableau-workbook --config configs/portfolio.local.yaml
```

After opening and saving the workbook in Tableau Desktop, publish it:

```bash
uv run stock-analysis publish-tableau-workbook tableau/workbooks/portfolio_recommendations.twb --config configs/portfolio.local.yaml
```
