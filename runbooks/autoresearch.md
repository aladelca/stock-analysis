# Autoresearch Runbook

## Purpose

Use the autoresearch harness to iterate on ML forecasting candidates while preserving the current
SPY-relative metric contract. The harness is designed for fast local loops and a separate
confirmation run before promotion.

## Prerequisites

- Phase 1 artifacts exist under `data/runs/phase2-source-20260424`.
- Dependencies are installed through `uv`.
- Use `uv run --extra mlflow ...` or `uv sync --extra mlflow` before using MLflow tracking.
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
  --commission-rate 0.02 \
  --results-tsv experiments/autoresearch/results.tsv
```

The command prints JSON and optionally appends one row to the TSV ledger. A rejected candidate is a
valid evaluator outcome and still exits successfully.

## Turnover Penalty Tuning

When commission is large, tune `lambda_turnover` instead of guessing a single penalty. The tuning
command retrains the candidate for each penalty, backtests it, and selects the best value by the
configured objective metric. The default objective is SPY-relative `information_ratio`.

```bash
uv run stock-analysis tune-turnover \
  --candidate e8_baseline \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 48 \
  --optimizer-max-weight 0.30 \
  --commission-rate 0.02 \
  --turnover-penalties 0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5 \
  --objective-metric information_ratio \
  --json-output docs/experiments/e8-turnover-sweep-2pct-20260424.json
```

Read `best.lambda_turnover` from the output and copy it into `configs/portfolio.yaml` only after
reviewing the full sweep table. In the current 2% commission sweep, the best tested value is `5.0`.
If all rows remain rejected, treat the result as evidence that the current model/cadence is not
viable under the configured commission, not as a production promotion.

## MLflow Tracking

MLflow logging is opt-in:

```bash
uv run --extra mlflow python scripts/autoresearch_eval.py \
  --candidate e8_baseline \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 48 \
  --optimizer-max-weight 0.30 \
  --commission-rate 0.02 \
  --results-tsv experiments/autoresearch/results.tsv \
  --mlflow \
  --mlflow-tracking-uri sqlite:///data/mlflow/mlflow.db
```

The MLflow run logs evaluator params, portfolio/SPY/comparison metrics, decision tags, the complete
`result.json`, and the TSV ledger artifact when emitted. Keep using
`experiments/autoresearch/results.tsv` as the append-only audit ledger; MLflow is for interactive
comparison and local artifact browsing.

Start the local UI with:

```bash
uv run --extra mlflow mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db
```

## Contribution-Aware Evaluation

Use this when you want the backtest to model paycheck-style deposits and compare the same cash-flow
schedule against SPY:

```bash
uv run --extra mlflow stock-analysis autoresearch-eval \
  --candidate e8_baseline \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 48 \
  --optimizer-max-weight 0.30 \
  --risk-aversion 10 \
  --min-trade-weight 0.005 \
  --lambda-turnover 5.0 \
  --commission-rate 0.02 \
  --initial-portfolio-value 1000 \
  --monthly-deposit-amount 100 \
  --deposit-frequency-days 30 \
  --no-trade-band 0.02 \
  --horizon-days 5 \
  --rebalance-step-days 5 \
  --embargo-days 15 \
  --covariance-lookback-days 252 \
  --iteration-id e8-contribution-aware-20260424 \
  --mlflow \
  --json-output docs/experiments/e8-contribution-aware-20260424.json
```

Interpret cash-flow fields separately:

- `strategy_time_weighted_return` measures strategy skill independent of deposit timing.
- `strategy_money_weighted_return` measures the investor-specific return after deposits.
- `strategy_ending_value` is the dollar result after initial capital, deposits, returns, and commissions.
- `benchmark_ending_value` applies the same initial capital and deposits to SPY.
- `active_ending_value` is the strategy ending value minus the SPY same-deposit ending value.

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
