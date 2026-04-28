# Autoresearch Experimentation Summary

## Experiment Design

The autoresearch harness adapts the upstream autoresearch pattern to this repository without adding
a runtime dependency on the autoresearch project. Candidate model logic lives in
`src/stock_analysis/ml/autoresearch_candidate.py`. The evaluator in
`src/stock_analysis/ml/autoresearch_eval.py` is fixed and uses the same walk-forward backtest,
optimizer, portfolio metrics, and aligned SPY-relative IR calculation as the Phase 2 reporting path.

MLflow is enabled as an optional tracking sink. Each tracked run logs the evaluator result object,
flattened parameters, portfolio metrics, decision tags, and the append-only TSV ledger artifact.
The ledger at `experiments/autoresearch/results.tsv` remains the audit source for promotion
decisions.

Fast-loop evaluations use `data/runs/phase2-source-20260424`, the point-in-time top 100 assets by
`dollar_volume_21d` at each rebalance, 5-trading-day forward targets, 48 requested rebalances,
5 bps transaction costs, a 0.30 optimizer max-weight cap, and the turnover-aware optimizer.

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

## Search Results

The strongest model found in the latest MLflow-backed search is
`e8_weight_ridge_1p2_lgbm_0p8_scale_1p20`. It keeps the E8 model family, reweights the internal
z-scored model legs toward Ridge, and scales the final forecast score by 1.20 before optimization.

| Candidate | Sharpe | SPY Sharpe | Sharpe Diff | Active Return | SPY-Relative IR | CI Low | Mean Turnover | Status | Objective Improved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `e8_weight_ridge_1p2_lgbm_0p8_scale_1p20` | 3.255864 | 1.130685 | 2.125178 | 0.729631 | 2.313480 | -0.335028 | 0.682642 | provisional | yes |
| `e8_weight_ridge_1p2_lgbm_0p8_scale_1p10` | 3.255394 | 1.130685 | 2.124709 | 0.706076 | 2.336423 | -0.351793 | 0.679460 | provisional | yes |
| `e8_weight_ridge_1p2_lgbm_0p8_scale_1p30` | 3.253273 | 1.130685 | 2.122588 | 0.750389 | 2.291243 | -0.342935 | 0.683211 | provisional | yes |
| `e8_weight_ridge_1p2_lgbm_0p8_scale_1p15` | 3.253133 | 1.130685 | 2.122448 | 0.717022 | 2.323494 | -0.343275 | 0.681167 | provisional | yes |
| `e8_weight_ridge_1p2_lgbm_0p8_scale_1p05` | 3.241327 | 1.130685 | 2.110641 | 0.691341 | 2.341578 | -0.366302 | 0.678875 | provisional | yes |

The winner improves the baseline Sharpe difference from `1.856511` to `2.125178` and keeps
SPY-relative IR above the baseline threshold at `2.313480`. It remains `provisional` because the
Sharpe-difference bootstrap CI lower bound is still negative. MLflow run:
`79481fba8545431d8a71a1ed699bc571`.

## CatBoost Experiments

CatBoost was added as a first-class candidate family in `src/stock_analysis/ml/phase2.py` and
`src/stock_analysis/ml/autoresearch_candidate.py`. The batch covered standalone CatBoost return
regression, a deeper CatBoost parameter profile, rank-label regression, top-tercile classification,
momentum feature subsets, Ridge plus CatBoost blends, and a three-model Ridge plus LightGBM plus
CatBoost ensemble.

| CatBoost Candidate | Sharpe | SPY Sharpe | Sharpe Diff | Active Return | SPY-Relative IR | CI Low | Mean Turnover | Objective Improved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ridge_catboost_1p5_0p5_scale_1p80` | 3.250392 | 1.130685 | 2.119706 | 0.932179 | 2.172948 | -0.397155 | 0.679841 | no |
| `e8_catboost_1p3_0p7_0p2_scale_1p20` | 3.224760 | 1.130685 | 2.094075 | 0.747811 | 2.276817 | -0.397780 | 0.663750 | yes |
| `ridge_catboost_1p5_0p5_scale_1p30` | 3.138071 | 1.130685 | 2.007386 | 0.774161 | 2.243235 | -0.401370 | 0.682144 | yes |
| `ridge_catboost_1p5_0p5_scale_1p20` | 3.070051 | 1.130685 | 1.939366 | 0.735213 | 2.230238 | -0.410118 | 0.685305 | yes |
| `catboost_momentum_return_zscore` | 2.582096 | 1.130685 | 1.451410 | 0.556259 | 2.001439 | -0.433351 | 0.723880 | no |

The best CatBoost-involved candidate that passes the objective gate is
`e8_catboost_1p3_0p7_0p2_scale_1p20`, with MLflow run
`7a0fdab08fa64eaeb06c555c61317c50`. It improves over the original baseline on point estimates but
does not beat the best Ridge plus LightGBM candidate. The highest CatBoost Sharpe-difference point
estimate is `ridge_catboost_1p5_0p5_scale_1p80`, but its SPY-relative IR falls below the baseline
gate, so it is not a promotion candidate.

## Optimization Model Result

All candidate forecasts are evaluated through the same long-only optimizer:
`mu^T w - gamma * w^T Sigma w - lambda_turnover * ||w - w_prev||_1`, subject to full investment,
non-negative weights, and a max-weight cap. The fast-loop default keeps `gamma=10`,
`lambda_turnover=0.001`, and `max_weight=0.30`.

The optimizer sweep around the earlier `e8_scale_0p85` candidate confirmed that `max_weight=0.30`
remains the best local cap among `0.20`, `0.25`, `0.28`, `0.32`, `0.35`, and `0.40` under the same
evaluation contract. The new winning search direction came from model-leg weighting and score-scale
calibration, not from changing the optimizer cap.

## SPY-Relative IR Audit

SPY-relative IR is calculated only after exact date alignment between candidate period returns and
SPY forward returns for the same horizon. The evaluator computes:

`active_return = mean(candidate_return - spy_return) * periods_per_year`

`tracking_error = std(candidate_return - spy_return) * sqrt(periods_per_year)`

`information_ratio = active_return / tracking_error`

The `ir_observations` ledger column records the aligned row count used for this calculation.

## Contribution-Aware Production Check

The corrected contribution-aware production check uses 2% commissions, `lambda_turnover=5.0`,
`monthly_deposit_amount=100`, `rebalance_on_deposit_day=true`, and the same-deposit SPY benchmark
with benchmark TWR net of commissions. Under that contract, the strongest sampled candidate is
`e8_scale_0p5`, not the earlier fast-loop champion.

| Candidate | Status | Ending Value | SPY Ending Value | Sharpe | SPY Sharpe | SPY-Relative IR | CI Low | Mean Turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e8_scale_0p5` | go | $8,610.66 | $6,115.11 | 4.002253 | 1.320164 | 3.134023 | 0.170819 | 0.026897 |

Artifact:
`docs/experiments/e8-scale-0p5-contribution-corrected-20260426.json`.

The one-shot ML flow is configured as `e8-scale-0p5-contribution-aware-v1`, which keeps the E8 Ridge
plus LightGBM blend and applies `ml_score_scale=0.5` before optimization.

### 2026-04-28 PIT Broad Validation

The 2026-04-28 validation removes the latest-liquidity lookahead from `max_assets`: each rebalance is
limited to the top 100 assets by `dollar_volume_21d` known on that rebalance date. This broader
validation requested 241 rebalance/deposit dates and produced 236 aligned SPY observations.

| Candidate | Status | Ending Value | SPY Ending Value | Sharpe | SPY Sharpe | SPY-Relative IR | CI Low | Total Commissions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e8_baseline` | rejected | $15,454.58 | $10,045.59 | 0.939092 | 1.139604 | 0.670123 | -1.084678 | $2,210.09 |
| `e8_scale_0p5` | provisional | $14,575.13 | $10,045.59 | 1.212379 | 1.139604 | 0.763414 | -0.675658 | $387.52 |
| `e8_weight_ridge_1p2_lgbm_0p8_scale_1p20` | rejected | $16,763.22 | $10,045.59 | 0.991536 | 1.139604 | 0.739829 | -1.140419 | $3,040.39 |

Artifacts:
`docs/experiments/e8-baseline-production-pit-broad-20260428.json`,
`docs/experiments/e8-scale-0p5-production-pit-broad-20260428.json`, and
`docs/experiments/e8-weight-ridge-1p2-lgbm-0p8-scale-1p20-production-pit-broad-20260428.json`.

Account-size sensitivity with `initial_portfolio_value=300` keeps the same conclusion. The
`e8_scale_0p5` strategy ended at $11,288.79 versus the same-deposit SPY value of $8,490.87, but its
Sharpe was 1.138959 versus SPY at 1.139604, so the evaluator rejected it for the strict SPY Sharpe
gate. Artifact: `docs/experiments/e8-scale-0p5-user300-pit-broad-20260428.json`.

Decision: keep `e8_scale_0p5` as the best current production candidate because it has the best
risk-adjusted profile among the tested production-economics candidates and much lower commission drag
than the alternatives. Do not claim a scientifically confirmed SPY beat yet. The corrected evidence
supports "beats SPY on ending-value point estimates under the tested deposit schedule"; it does not
support production copy or dashboards that state the model has conclusively beaten SPY.

### 2026-04-28 Cost-Aware PIT Model Search

Follow-up search under the same PIT production-economics contract tested E8 scale/weight variants,
standalone LightGBM, CatBoost, and PyTorch MLP, LSTM, and Transformer candidates. PyTorch was run
with `mps` available on the local Apple Silicon environment via the optional `pytorch` extra.

The strongest broad-confirmed candidate was `lightgbm_return_zscore` with `optimizer_max_weight=0.24`,
`lambda_turnover=5.0`, and `no_trade_band=0.04`.

| Candidate / Setting | Status | Ending Value | SPY Ending Value | Sharpe | SPY Sharpe | SPY-Relative IR | CI Low | Total Commissions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lightgbm_return_zscore`, max weight 0.24, band 0.04 | provisional | $19,607.92 | $10,045.59 | 1.686490 | 1.139604 | 1.361028 | -0.078446 | $416.58 |
| `lightgbm_return_zscore`, max weight 0.25, band 0.04 | provisional | $19,590.57 | $10,045.59 | 1.681303 | 1.139604 | 1.350918 | -0.081831 | $416.17 |
| `lightgbm_return_zscore`, max weight 0.30, band 0.02 | provisional | $19,514.99 | $10,045.59 | 1.607539 | 1.139604 | 1.310368 | -0.164219 | $371.67 |
| `catboost_return_zscore` | provisional | $16,997.49 | $10,045.59 | 1.296346 | 1.139604 | 0.883926 | -0.638166 | $498.03 |
| `catboost_momentum_return_zscore` | provisional | $15,514.11 | $10,045.59 | 1.275287 | 1.139604 | 0.831108 | -0.666697 | $330.03 |

PyTorch MLP fast-screen results did not justify broad confirmation. The best MLP screen was
`torch_mlp_return_deep_zscore` at 48 requested rebalances, with Sharpe 1.559 versus SPY 1.320 and
CI low -0.792.

PyTorch LSTM and Transformer candidates were then added as first-class autoresearch candidates. They
train per-ticker trailing feature windows point-in-time at each rebalance, using `ticker` and `date`
only to assemble historical windows and not as numeric alpha features. The 48-rebalance screen used
the current best production-economics optimizer settings: `optimizer_max_weight=0.24`,
`lambda_turnover=5.0`, 2% commission, $100 monthly deposits, and `no_trade_band=0.04`.

| Neural Sequence Candidate / Setting | Scope | Status | Ending Value | SPY Ending Value | Sharpe | SPY Sharpe | SPY-Relative IR | CI Low |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `torch_lstm_momentum_return_zscore`, max weight 0.24, band 0.04 | broad 241 | provisional | $14,371.04 | $10,045.59 | 1.375979 | 1.139604 | 0.851981 | -0.333648 |
| `torch_lstm_momentum_return_zscore`, max weight 0.24, band 0.04 | fast 48 | provisional | $6,982.64 | $6,115.11 | 1.635387 | 1.320164 | 0.801520 | -0.709346 |
| `torch_lstm_return_zscore`, max weight 0.24, band 0.04 | fast 48 | provisional | $6,729.69 | $6,115.11 | 1.407835 | 1.320164 | 0.663765 | -0.795821 |
| `torch_lstm_return_deep_zscore`, max weight 0.24, band 0.04 | fast 48 | provisional | $6,766.21 | $6,115.11 | 1.374672 | 1.320164 | 0.738616 | -0.724888 |
| `torch_transformer_return_zscore`, max weight 0.24, band 0.04 | fast 48 | rejected | $6,740.75 | $6,115.11 | 1.240219 | 1.320164 | 0.201597 | -0.976523 |
| `torch_transformer_return_deep_zscore`, max weight 0.24, band 0.04 | fast 48 | rejected | $6,434.21 | $6,115.11 | 1.005355 | 1.320164 | -0.107549 | -0.988385 |
| `torch_transformer_momentum_return_zscore`, max weight 0.24, band 0.04 | fast 48 | rejected | $6,521.72 | $6,115.11 | 1.023103 | 1.320164 | -0.034211 | -0.963996 |

The best neural sequence model beats SPY on broad point estimates, but it does not beat the current
LightGBM candidate. The apples-to-apples 48-rebalance LightGBM check with max weight 0.24 and band
0.04 produced Sharpe 2.097 versus SPY 1.320 and SPY-relative IR 1.594. The broad LightGBM candidate
also remains stronger: Sharpe 1.686, IR 1.361, ending value $19,607.92, and CI low -0.078.

Decision: `lightgbm_return_zscore` with max weight 0.24 and no-trade band 0.04 is the new best
researched candidate on point estimates. It still does not clear the strict bootstrap gate, so the
status remains `provisional`. The product should not claim "we beat SPY" as a statistically confirmed
statement yet. Do not promote the LSTM or Transformer models to production. They are useful research
baselines, but the evidence says the better model is still the simpler LightGBM return regressor under
the cost-aware PIT contract.

The next highest-value model work is a stronger validation design, including historical constituent
membership and a longer out-of-sample window. After that, revisit model complexity with sequence
ensembles only if they add incremental rank information over LightGBM after transaction costs.
