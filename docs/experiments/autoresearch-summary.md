# Autoresearch Experimentation Summary

## Experiment Design

The autoresearch harness adapts the upstream autoresearch pattern to this repository without adding
a runtime dependency. Candidate model logic lives in `src/stock_analysis/ml/autoresearch_candidate.py`.
The evaluator in `src/stock_analysis/ml/autoresearch_eval.py` is fixed and uses the same
walk-forward backtest, optimizer, portfolio metrics, and aligned SPY-relative IR calculation as the
Phase 2 reporting path.

MLflow is added as an optional tracking sink for this harness. When enabled, it logs the exact
evaluator result object, flattened params, metrics, decision tags, and emitted artifacts. The
append-only TSV ledger remains the audit source for promotion decisions.

Fast-loop evaluations use the Phase 2 source artifacts, the latest top 100 assets by
`dollar_volume_21d`, weekly 5-trading-day targets, 5 bps transaction costs, a 0.30 max-weight cap,
and the turnover-aware optimizer.

## Baseline

The starting point is the documented Phase 2 E8 Ridge plus LightGBM blend:

| Metric | Value |
| --- | ---: |
| Candidate Sharpe | 2.987197 |
| SPY Sharpe | 1.130685 |
| Sharpe difference | 1.856512 |
| Annualized return | 1.086616 |
| Annualized active return | 0.636036 |
| Tracking error | 0.287482 |
| SPY-relative information ratio | 2.212435 |
| Max drawdown | -0.202734 |
| Mean turnover | 0.686668 |
| Aligned SPY observations | 46 |
| Sharpe-difference 95% CI | [-0.446, 1.542] |
| Decision | provisional |

## ML Model Result

The seeded `e8_baseline` candidate was evaluated through the new harness against
`data/runs/phase2-source-20260424` with top 100 assets, 48 requested rebalances, and a 0.30
optimizer max-weight cap. The measured metrics matched the documented Phase 2 E8 baseline:

| Candidate | Sharpe | SPY Sharpe | Sharpe Diff | Active Return | SPY-Relative IR | CI Low | CI High | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `e8_baseline` | 2.987197 | 1.130685 | 1.856511 | 0.636036 | 2.212435 | -0.445541 | 1.541675 | provisional |

The first MLflow-tracked search batch evaluated score-calibration, feature-subset, Ridge, LightGBM
regression, and LightGBM LambdaRank candidates. The best candidate was `e8_scale_0p85`, which keeps
the same Ridge plus LightGBM model family as E8 and scales forecast scores by `0.85` before
optimization.

| Candidate | Sharpe | SPY Sharpe | Sharpe Diff | Active Return | SPY-Relative IR | CI Low | Mean Turnover | Status | Objective Improved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `e8_scale_0p85` | 3.026434 | 1.130685 | 1.895749 | 0.603803 | 2.278831 | -0.399057 | 0.691957 | provisional | yes |
| `e8_scale_0p9` | 3.023593 | 1.130685 | 1.892907 | 0.616570 | 2.263636 | -0.418502 | 0.689014 | provisional | yes |
| `e8_scale_0p8` | 2.999766 | 1.130685 | 1.869081 | 0.586625 | 2.278141 | -0.397268 | 0.695324 | provisional | yes |
| `e8_scale_4p0` | 2.991126 | 1.130685 | 1.860440 | 0.843023 | 2.105149 | -0.421310 | 0.671771 | provisional | no |
| `e8_scale_0p5` | 2.984744 | 1.130685 | 1.854058 | 0.477388 | 2.417264 | -0.261428 | 0.718203 | provisional | no |

The winner improves the primary objective over baseline: Sharpe difference increases from
`1.856511` to `1.895749`, and SPY-relative IR increases from `2.212435` to `2.278831`. It remains
`provisional` because the Sharpe-difference bootstrap CI lower bound is still negative.

## Optimization Model Result

All candidate forecasts are evaluated through the same long-only optimizer:
`mu^T w - gamma * w^T Sigma w - lambda_turnover * ||w - w_prev||_1`, subject to full investment,
non-negative weights, and a max-weight cap. The fast-loop default keeps `gamma=10`,
`lambda_turnover=0.001`, and `max_weight=0.30`.

## SPY-Relative IR Audit

SPY-relative IR is calculated only after exact date alignment between candidate period returns and
SPY forward returns for the same horizon. The evaluator computes:

`active_return = mean(candidate_return - spy_return) * periods_per_year`

`tracking_error = std(candidate_return - spy_return) * sqrt(periods_per_year)`

`information_ratio = active_return / tracking_error`

The `ir_observations` ledger column records the aligned row count used for this calculation.

## Current Outcome

The experiment improved the model objective but did not reach statistical `go`. The current
champion is `e8_scale_0p85`, a calibrated E8 blend. Promotion should wait for a confirmation run
with broader rebalance coverage and a production inference change that applies the same score
scaling in `src/stock_analysis/forecasting/ml_forecast.py`.
