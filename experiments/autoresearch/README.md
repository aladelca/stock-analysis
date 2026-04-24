# Autoresearch Experiments

This directory contains the local adaptation of the `karpathy/autoresearch` workflow for the
stock-analysis model search loop. The repo does not vendor or import upstream autoresearch; it uses
the same operating pattern: one editable candidate surface, one fixed evaluator, and an append-only
results ledger.

## Manual Evaluation

Run the fast evaluator from the repository root:

```bash
uv run python scripts/autoresearch_eval.py \
  --candidate e8_baseline \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 48 \
  --optimizer-max-weight 0.30 \
  --results-tsv experiments/autoresearch/results.tsv
```

The same evaluator is also exposed through the app CLI:

```bash
uv run stock-analysis autoresearch-eval \
  --candidate e8_baseline \
  --input-run-root data/runs/phase2-source-20260424 \
  --results-tsv experiments/autoresearch/results.tsv
```

Successful infrastructure runs exit `0` even when the candidate is rejected. The JSON `decision`
block records `rejected`, `provisional`, or `go`.

## MLflow Tracking

MLflow is available as an optional tracking sink. It is aligned with this harness when used for
browsing runs, comparing params, and collecting artifacts; it does not replace the fixed evaluator
or append-only TSV ledger.

```bash
uv run --extra mlflow python scripts/autoresearch_eval.py \
  --candidate e8_baseline \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 48 \
  --optimizer-max-weight 0.30 \
  --results-tsv experiments/autoresearch/results.tsv \
  --mlflow \
  --mlflow-tracking-uri sqlite:///data/mlflow/mlflow.db
```

Open the local UI with:

```bash
uv run --extra mlflow mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db
```

## Promotion Rule

A candidate must beat SPY on Sharpe, annualized active return, and SPY-relative information ratio.
It is only a `go` when the Sharpe-difference bootstrap confidence interval lower bound is positive.
Candidates that beat SPY on point estimates but fail the CI requirement remain `provisional`.

The current documented baseline is the Phase 2 E8 Ridge plus LightGBM blend:

| Metric | Value |
| --- | ---: |
| Candidate Sharpe | 2.987197 |
| SPY Sharpe | 1.130685 |
| Annualized active return | 0.636036 |
| SPY-relative information ratio | 2.212435 |
| Sharpe-difference 95% CI | [-0.446, 1.542] |
| Status | provisional |

## Editable Surface

Autonomous experiments should only edit `src/stock_analysis/ml/autoresearch_candidate.py` unless a
human explicitly broadens the scope. The evaluator, labels, SPY benchmark alignment, and IR formula
are fixed.
