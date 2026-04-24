# Phase 2 Report

> Uses current S&P 500 constituents; survivorship bias present.

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
