# Phase 2 Experimentation and Backtesting Summary

> Uses current S&P 500 constituents; survivorship bias present.

## Experiment Design

Phase 2 compares the current heuristic forecast against market, equal-weight, linear, LightGBM, ranking, classification, and blended ML models. All model experiments feed forecast scores into the same frozen long-only optimizer so model differences are attributable to `mu` quality. The continuation objective for this report is SPY-relative: find an ML portfolio with higher Sharpe than SPY and positive information ratio.
When `max_assets` is configured, experiments run on the most liquid assets by latest `dollar_volume_21d`; this keeps optimizer runtimes manageable and is explicitly part of the experiment scope.

## Run Configuration

- Source artifacts: `data/runs/phase2-source-20260424`.
- Feature panel window: 2022-04-25 to 2026-04-23.
- Label panel window: 2022-04-25 to 2026-04-23.
- Experiment universe: top 100 assets by latest `dollar_volume_21d`; 100 tickers in scope.
- Completed rebalance observations: 46 dates from 2022-05-23 to 2026-03-23.
- Backtest setup: 5-trading-day target, 5-business-day rebalance step, 48 requested rebalance samples, 5 bps transaction cost.
- Optimizer setup: max_weight=0.3, risk_aversion=10, lambda_turnover=0.001.
- LightGBM execution: fixed grid, no nested CV with random_seed=42.

## Experimentation Outcome

- Best ML model by Pearson IC: E4 (Ridge on rank-normalized features), IC 0.0472719.
- Best ML model by rank IC: E4 (Ridge on rank-normalized features), rank IC 0.0517037.
- Best optimized ML portfolio: E8 (Linear blend of ridge + LightGBM regression), Sharpe 2.9872, annualized return 1.08662, SPY-relative IR 2.21243, annualized active return 0.636036, mean turnover 0.686668.
- Interpretation: predictive IC and optimized portfolio quality can diverge; the promotion decision is therefore based on the portfolio backtest versus SPY, not on predictive IC alone.

## SPY-Relative IR Calculation

Information ratio is calculated only on exact rebalance-date matches between portfolio period returns and SPY forward returns for the same horizon. `active_return = mean(portfolio_return - spy_return) * periods_per_year`, `tracking_error = std(portfolio_return - spy_return) * sqrt(periods_per_year)`, and `IR = active_return / tracking_error`. The results table includes the annualized active return, annualized tracking error, and aligned observation count used in that calculation.

## Optimization Model

The optimizer maximizes `mu^T w - gamma * w^T Sigma w - lambda_turnover * ||w - w_prev||_1`, with long-only weights, a max-weight constraint, and configurable transaction costs applied during backtesting. This is the turnover-aware configuration required for weekly rebalancing.

## ML Models

E3/E4 use train-only median imputation and standardization. E5 uses LightGBM regression, E6 uses LightGBM LambdaRank, E7 uses a LightGBM top-tercile classifier, and E8 blends Ridge and LightGBM regression after z-scoring predictions.

## Results

| experiment_id | model_name | status | pearson_ic | rank_ic | sharpe | annualized_return | max_drawdown | active_return | tracking_error | information_ratio | ir_observations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Current heuristic | completed | 0.00427917 | 0.0631344 | 1.72428 | 0.576873 | -0.186728 | 0.342199 | 0.292978 | 1.168 | 46 |
| E1 | SPY buy-and-hold | completed |  |  | 1.13069 | 0.170089 | -0.119402 |  |  |  |  |
| E2 | Equal-weight S&P 500 | completed |  |  | 1.78208 | 0.325484 | -0.114071 | 0.130506 | 0.0595574 | 2.19127 | 46 |
| E3 | Ridge regression | completed | -0.00821206 | 0.0484747 | 0.463982 | 0.0527093 | -0.132423 | -0.110772 | 0.119079 | -0.930243 | 46 |
| E4 | Ridge on rank-normalized features | completed | 0.0472719 | 0.0517037 | 1.40722 | 0.586331 | -0.246053 | 0.37534 | 0.350187 | 1.07183 | 46 |
| E5 | LightGBM regression | completed | -0.0356454 | 0.0331105 | 0.348051 | 0.0388197 | -0.139606 | -0.124292 | 0.121099 | -1.02637 | 46 |
| E6 | LightGBM LambdaRank | completed | 0.0378929 | 0.044545 | 1.28736 | 0.280273 | -0.158473 | 0.10199 | 0.170363 | 0.598661 | 46 |
| E7 | LightGBM top-tercile classifier | completed | 0.0296645 | 0.0237128 | 0.186138 | 0.0248662 | -0.156729 | -0.135184 | 0.113261 | -1.19356 | 46 |
| E8 | Linear blend of ridge + LightGBM regression | completed | 0.0419753 | -0.000435306 | 2.9872 | 1.08662 | -0.202734 | 0.636036 | 0.287482 | 2.21243 | 46 |

## Winner Sweeps

_No sweeps were run._

## Gating Decision

PROVISIONAL: E8 beats SPY on Sharpe and has positive IR, but the Sharpe-difference CI includes zero. Sharpe-difference 95% CI vs SPY: [-0.446, 1.542].

## Caveats

- Current-constituent S&P 500 universe creates survivorship bias.
- Transaction-cost model is flat and excludes market impact.
- LightGBM tuning grid is intentionally small for repeatable local execution.
