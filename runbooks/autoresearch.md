# Autoresearch Runbook

## Purpose

Use the autoresearch harness to iterate on ML forecasting candidates while preserving the current
SPY-relative metric contract. The harness is designed for fast local loops and a separate
confirmation run before promotion.

## Prerequisites

- Phase 1 artifacts exist under `data/runs/phase2-source-20260424`.
- Dependencies are installed through `uv`.
- The working tree is clean enough that candidate changes can be reviewed separately from evaluator
  or documentation changes.

## Fast Evaluation

```bash
uv run python scripts/autoresearch_eval.py \
  --candidate e8_baseline \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 48 \
  --optimizer-max-weight 0.30 \
  --results-tsv experiments/autoresearch/results.tsv
```

The command prints JSON and optionally appends one row to the TSV ledger. A rejected candidate is a
valid evaluator outcome and still exits successfully.

## Interpreting Results

Primary fields:

- `candidate_sharpe`: annualized portfolio Sharpe for the candidate optimizer output.
- `spy_sharpe`: annualized SPY Sharpe on the same rebalance dates and horizon.
- `active_return`: annualized mean of candidate period return minus SPY period return.
- `tracking_error`: annualized standard deviation of active returns.
- `information_ratio`: `active_return / tracking_error`, calculated after exact date alignment.
- `ir_observations`: matched candidate/SPY observations used in the IR calculation.
- `status`: `rejected`, `provisional`, `go`, or `failed_infrastructure`.

## Confirmation Run

Before promotion, rerun the candidate with the broadest feasible rebalance coverage. Record the
command, config, ledger row, and interpretation in `docs/experiments/autoresearch-summary.md`.

## Promotion

Promote only a confirmed `go` result. Copy the stable candidate definition into the production ML
forecasting path or convert production inference to load the candidate by id. Regenerate the Phase 2
experiment docs and Tableau outputs after promotion.
