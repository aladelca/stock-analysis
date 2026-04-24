# Tableau Prep Runbook

1. Run the Python pipeline.

```bash
uv run stock-analysis run-one-shot --config configs/portfolio.yaml
```

2. Open Tableau Prep Builder.
3. Build or open the flow described in `tableau/prep/portfolio_dashboard_mart.flow-spec.md`.
4. Connect the flow to the CSV files under:

```text
data/runs/<run_id>/bronze/csv/
data/runs/<run_id>/silver/csv/
data/runs/<run_id>/gold/csv/
```

5. Output the mart to:

```text
tableau_prep_outputs/portfolio_dashboard_mart.hyper
```

For command-line execution, use the Tableau Prep Builder CLI path for your local Tableau version and pass the `.tfl` file once it has been validated in the GUI.
