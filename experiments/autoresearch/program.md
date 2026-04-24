# Autoresearch Program

## Objective

Improve the production ML forecasting candidate while preserving the fixed SPY-relative evaluation
contract. The target is not merely to beat SPY on point estimates. The target is a `go` decision:
candidate Sharpe beats SPY Sharpe, active return is positive, SPY-relative information ratio is
positive, and the Sharpe-difference confidence interval lower bound is above zero.

## Allowed Edits

- `src/stock_analysis/ml/autoresearch_candidate.py`
- New candidate-only helper modules under `src/stock_analysis/ml/` when they are imported only by
  `autoresearch_candidate.py`
- Candidate documentation under `experiments/autoresearch/`

## Forbidden Edits

- `src/stock_analysis/ml/autoresearch_eval.py`
- `src/stock_analysis/ml/evaluation.py`
- `src/stock_analysis/backtest/runner.py`
- SPY benchmark generation, label generation, benchmark alignment, or information-ratio formula
- Phase 2 gating semantics unless the user explicitly requests an evaluator change

## Loop

1. Inspect `experiments/autoresearch/results.tsv` and choose one candidate change.
2. Edit only the allowed candidate surface.
3. Run the fast evaluator:

   ```bash
   uv run python scripts/autoresearch_eval.py \
     --candidate e8_baseline \
     --input-run-root data/runs/phase2-source-20260424 \
     --max-assets 100 \
     --max-rebalances 48 \
     --optimizer-max-weight 0.30 \
     --results-tsv experiments/autoresearch/results.tsv
   ```

   Add `--mlflow --mlflow-tracking-uri sqlite:///data/mlflow/mlflow.db` when MLflow comparison is
   useful. MLflow is an optional mirror of the fixed result object, not a separate scoring
   authority.

4. Keep a candidate only when it improves the objective without weakening SPY-relative IR.
5. Before promotion, run a confirmation evaluation with broader rebalance coverage and record the
   result in `docs/experiments/autoresearch-summary.md`.

## Decision Contract

- `rejected`: candidate fails Sharpe versus SPY, active return, or SPY-relative IR.
- `provisional`: candidate beats SPY on point estimates, but the Sharpe-difference CI includes zero.
- `go`: candidate beats SPY and the Sharpe-difference CI lower bound is positive.
- `failed_infrastructure`: artifacts, benchmark rows, candidate wiring, or optimizer execution
  failed before a valid comparison could be made.

## Result Ledger

`experiments/autoresearch/results.tsv` is append-only. Do not rewrite older rows. Add narrative
context in `docs/experiments/autoresearch-summary.md` instead of editing historical ledger values.
