# Phase 2 Experimentation and Backtesting Summary

> Uses current S&P 500 constituents; survivorship bias present.

## Experiment Design

Phase 2 compares the current heuristic forecast against market, equal-weight, linear, LightGBM, ranking, classification, and blended ML models. All model experiments feed forecast scores into the same frozen long-only optimizer so model differences are attributable to `mu` quality.
When `max_assets` is configured, experiments run on the most liquid assets by latest `dollar_volume_21d`; this keeps optimizer runtimes manageable and is explicitly part of the experiment scope.

## Run Configuration

- Source artifacts: `data/runs/phase2-source-20260424`, rebuilt from the local 5-year price run and a fresh SPY benchmark series.
- Price window available to the panel: 2021-04-26 to 2026-04-23.
- Feature/label panel window after lookback truncation: 2022-04-25 to 2026-04-23.
- Experiment universe: top 80 assets by latest `dollar_volume_21d`; optimized portfolios had 79 eligible names on the sampled rebalance dates.
- Backtest sample: 10 completed weekly rebalance observations spread from 2022-09-01 to 2025-12-09. The command requested 12 samples, but two candidate dates were skipped by label/eligibility filters.
- Primary target: 5-trading-day forward return. Horizon and cadence sweeps also evaluate 21-day and 63-day targets.
- LightGBM execution mode: fixed compact parameter grid for the local report run (`--no-nested-cv`) to keep the full E0-E8 plus sweep run interactive. The code path supports nested inner-CV tuning when the flag is omitted.

## Optimization Model

The optimizer maximizes `mu^T w - gamma * w^T Sigma w - lambda_turnover * ||w - w_prev||_1`, with long-only weights, a max-weight constraint, and configurable transaction costs applied during backtesting. This is the turnover-aware configuration required for weekly rebalancing.

In this run, the best optimized ML portfolio was E8, the Ridge + LightGBM regression blend. Its average weekly turnover was about 64%, so the portfolio result is sensitive to transaction-cost assumptions even though the 0-20 bps cost sweep remained positive in this bounded sample.

## ML Models

E3/E4 use train-only median imputation and standardization. E5 uses LightGBM regression, E6 uses LightGBM LambdaRank, E7 uses a LightGBM top-tercile classifier, and E8 blends Ridge and LightGBM regression after z-scoring predictions.

The best pure predictive model by Pearson IC was E7 (LightGBM top-tercile classifier, IC 0.2418), and the best by rank IC was E5 (LightGBM regression, rank IC 0.2716). Those predictive wins did not translate into the best optimized portfolio because their SPY-relative information ratios were negative in this sample. E8 had the strongest optimized portfolio metrics, but its IC was below the heuristic E0, so it fails the Phase 2 gate despite strong Sharpe.

## Results

| experiment_id | model_name | status | pearson_ic | rank_ic | sharpe | annualized_return | max_drawdown | information_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Current heuristic | completed | 0.21439 | 0.097774 | 5.71021 | 1.77607 | -0.0205935 | 1.85284 |
| E1 | SPY buy-and-hold | completed |  |  | 0.834423 | 0.127319 | -0.244964 |  |
| E2 | Equal-weight S&P 500 | completed |  |  | 1.52584 | 0.324658 | -0.204605 | 2.2388 |
| E3 | Ridge regression | completed | 0.178572 | 0.202702 | 3.92635 | 0.428979 | -0.0250826 | -5.06244 |
| E4 | Ridge on rank-normalized features | completed | 0.0954995 | 0.0982941 | 6.81645 | 2.33426 | -0.0436069 | 2.50864 |
| E5 | LightGBM regression | completed | 0.215524 | 0.271646 | 3.90604 | 0.428367 | -0.0260109 | -5.09158 |
| E6 | LightGBM LambdaRank | completed | 0.190058 | 0.136924 | 7.68824 | 1.39573 | -0.0189188 | 2.13385 |
| E7 | LightGBM top-tercile classifier | completed | 0.241788 | 0.241213 | 5.19636 | 0.511602 | -0.022529 | -5.18842 |
| E8 | Linear blend of ridge + LightGBM regression | completed | 0.0872461 | 0.0982995 | 8.89106 | 1.97664 | -0.0210469 | 3.75379 |

## Winner Sweeps

| sweep | value | winner_model | sharpe | annualized_return | max_drawdown | information_ratio | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| horizon | h=5 | E8 | 8.89106 | 1.97664 | -0.0210469 | 3.75379 | completed |
| horizon | h=21 | E8 | 0.185788 | 0.0551789 | -0.102206 | 0.77798 | completed |
| horizon | h=63 | E8 | 1.68401 | 0.375405 | -0.0657124 | 1.78224 | completed |
| cost_bps | 0 | E8 | 9.11038 | 2.02388 | -0.0206029 | 3.89262 | completed |
| cost_bps | 5 | E8 | 8.89106 | 1.97664 | -0.0210469 | 3.75379 | completed |
| cost_bps | 10 | E8 | 8.67529 | 1.93011 | -0.0214909 | 3.61518 | completed |
| cost_bps | 20 | E8 | 8.25419 | 1.8392 | -0.022379 | 3.33866 | completed |
| lambda_turnover | 0 | E8 | 8.89322 | 1.97744 | -0.0210342 | 3.75524 | completed |
| lambda_turnover | 0.0005 | E8 | 8.89173 | 1.97676 | -0.0210431 | 3.75414 | completed |
| lambda_turnover | 0.001 | E8 | 8.89106 | 1.97664 | -0.0210469 | 3.75379 | completed |
| lambda_turnover | 0.002 | E8 | 8.89057 | 1.97651 | -0.0210469 | 3.75356 | completed |
| lambda_turnover | 0.005 | E8 | 8.88914 | 1.97615 | -0.021047 | 3.75289 | completed |
| max_weight | 0.02 | E8 | 8.32588 | 1.42419 | -0.0150568 | 3.31759 | completed |
| max_weight | 0.05 | E8 | 8.89106 | 1.97664 | -0.0210469 | 3.75379 | completed |
| max_weight | 0.1 | E8 | 10.4581 | 2.78095 | -0.032622 | 3.89745 | completed |
| cadence | weekly | E8 | 8.89106 | 1.97664 | -0.0210469 | 3.75379 | completed |
| cadence | monthly | E8 | 0.185788 | 0.0551789 | -0.102206 | 0.77798 | completed |
| cadence | quarterly | E8 | 1.68401 | 0.375405 | -0.0657124 | 1.78224 | completed |

## Gating Decision

NO-GO: no Phase 2 ML model currently beats E0 on OOS IC and SPY-relative IR. Do not start Phase 3.

Interpretation: E8 is the best portfolio backtest in the bounded local run, but Phase 2's promotion rule is stricter than "highest Sharpe." A model must beat the heuristic on out-of-sample IC and also produce positive SPY-relative IR. E5 and E7 beat E0 on IC but have negative IR; E8 has positive IR but does not beat E0 on IC. The correct decision is to keep the heuristic as the production baseline and continue Phase 2 diagnostics before considering Phase 3.

## Caveats

- Current-constituent S&P 500 universe creates survivorship bias.
- Transaction-cost model is flat and excludes market impact.
- LightGBM tuning grid is intentionally small for repeatable local execution.
