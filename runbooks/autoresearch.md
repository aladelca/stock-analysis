# Autoresearch Runbook

## Purpose

Use the autoresearch harness to iterate on ML forecasting candidates while preserving the current
SPY-relative metric contract. The harness is designed for fast local loops and a separate
confirmation run before promotion.

## Prerequisites

- Phase 1 artifacts exist under `data/runs/phase2-source-20260424`.
- Dependencies are installed through `uv`.
- Use `uv run --extra mlflow ...` or `uv sync --extra mlflow` before using MLflow tracking.
- Use `uv run --extra pytorch ...` before evaluating `torch_mlp_*`, `torch_lstm_*`, or
  `torch_transformer_*` candidates. On Apple Silicon, PyTorch candidates automatically use `mps`
  when available and fall back to CPU otherwise.
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

`--max-assets` is applied point-in-time at each rebalance date using the configured liquidity column.
Do not use latest-date universe filtering for promotion evidence.

## PyTorch Sequence Models

LSTM and Transformer candidates are available for research, but they are not the current production
recommendation. Sequence models need `ticker` and `date` in the prediction frame so the backtest can
assemble point-in-time trailing windows per ticker; existing tabular models ignore those identifier
columns.

Run the fast sequence screen with the same production-economics settings used for the current best
LightGBM candidate:

```bash
uv run --extra pytorch python scripts/autoresearch_eval.py \
  --candidate torch_lstm_momentum_return_zscore \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 48 \
  --optimizer-max-weight 0.24 \
  --risk-aversion 10 \
  --min-trade-weight 0.005 \
  --lambda-turnover 5.0 \
  --commission-rate 0.02 \
  --initial-portfolio-value 1000 \
  --monthly-deposit-amount 100 \
  --deposit-frequency-days 30 \
  --no-trade-band 0.04 \
  --horizon-days 5 \
  --rebalance-step-days 5 \
  --embargo-days 15 \
  --covariance-lookback-days 252 \
  --iteration-id torch-lstm-momentum-return-zscore-pit-fast48-$(date +%Y%m%d) \
  --results-tsv experiments/autoresearch/results.tsv \
  --json-output docs/experiments/torch-lstm-momentum-return-zscore-pit-fast48-$(date +%Y%m%d).json
```

The 2026-04-28 screen rejected the Transformer variants and found the best sequence candidate to be
`torch_lstm_momentum_return_zscore`. Its broad check was still weaker than LightGBM: Sharpe 1.376
versus SPY 1.140, IR 0.852, and CI low -0.334. Do not promote it over
`lightgbm_return_zscore`.

Run optional PyTorch unit coverage separately from the LightGBM-heavy suite to avoid native OpenMP
library collisions in one pytest process on macOS:

```bash
OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 STOCK_ANALYSIS_TEST_PYTORCH=1 \
  uv run --extra pytorch pytest -q tests/unit/test_autoresearch_candidate.py -k torch
```

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
  --candidate e8_scale_0p5 \
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
  --rebalance-on-deposit-day \
  --no-trade-band 0.02 \
  --horizon-days 5 \
  --rebalance-step-days 5 \
  --embargo-days 15 \
  --covariance-lookback-days 252 \
  --iteration-id e8-scale-0p5-contribution-corrected-20260426 \
  --mlflow \
  --json-output docs/experiments/e8-scale-0p5-contribution-corrected-20260426.json
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

Before promotion, rerun the candidate with the broadest feasible rebalance coverage and the
point-in-time liquidity screen. Record the command, config, ledger row, and interpretation in
`docs/experiments/autoresearch-summary.md`.

The current broad production-economics validation command for the best researched candidate is:

```bash
uv run --extra mlflow stock-analysis autoresearch-eval \
  --candidate lightgbm_return_zscore \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 241 \
  --optimizer-max-weight 0.24 \
  --risk-aversion 10 \
  --min-trade-weight 0.005 \
  --lambda-turnover 5.0 \
  --commission-rate 0.02 \
  --initial-portfolio-value 1000 \
  --monthly-deposit-amount 100 \
  --deposit-frequency-days 30 \
  --rebalance-on-deposit-day \
  --no-trade-band 0.04 \
  --horizon-days 5 \
  --rebalance-step-days 5 \
  --embargo-days 15 \
  --covariance-lookback-days 252 \
  --iteration-id lightgbm-return-zscore-production-pit-broad-$(date +%Y%m%d) \
  --results-tsv experiments/autoresearch/results.tsv \
  --json-output docs/experiments/lightgbm-return-zscore-production-pit-broad-$(date +%Y%m%d).json
```

For an account-size sensitivity that starts with $300 and then applies the same monthly deposits,
change only `--initial-portfolio-value 300` and use an iteration id that includes `user300`.

The 2026-04-28 corrected PIT broad search found `lightgbm_return_zscore` with max weight 0.24 and
band 0.04 as the best researched candidate on point estimates, but only as `provisional`. It beat SPY
on ending value and Sharpe point estimates, not on the statistically confirmed Sharpe-difference
gate.

## Promotion

Promote only a confirmed `go` result. Copy the stable candidate definition into the production ML
forecasting path or convert production inference to load the candidate by id. Regenerate the Phase 2
experiment docs and Tableau outputs after promotion.
