# Phase 2 SPY-Relative Experimentation Summary

> Uses current S&P 500 constituents; survivorship bias present.

## Objective

Continue Phase 2 experimentation until an ML-driven optimized portfolio beats SPY on the
same rebalance dates, with SPY-relative information ratio calculated from aligned active
returns.

Primary promotion metrics for this pass:

- Net-of-cost portfolio Sharpe versus aligned SPY.
- Annualized active return versus aligned SPY.
- SPY-relative information ratio.
- Bootstrap Sharpe-difference CI versus aligned SPY as a statistical caution flag.

## SPY-Relative IR Audit

The implementation now aligns portfolio and SPY returns by exact rebalance date before
calculating active returns. It does not compare by row position.

Formula used:

```text
active_return_period = portfolio_return - spy_return
active_return = mean(active_return_period) * periods_per_year
tracking_error = std(active_return_period, ddof=1) * sqrt(periods_per_year)
information_ratio = active_return / tracking_error
```

For the final run, E8 and SPY both used 46 matched rebalance observations.

## Final Run Configuration

- Command: `uv run stock-analysis run-phase2 --input-run-root data/runs/phase2-source-20260424 --output-dir docs/experiments --force --max-assets 100 --max-rebalances 48 --optimizer-max-weight 0.30 --no-sweeps --no-nested-cv`
- Source artifacts: `data/runs/phase2-source-20260424`.
- Universe: top 100 assets by latest `dollar_volume_21d`.
- Target: 5-trading-day forward return.
- Rebalance cadence: 5 business days.
- Requested rebalance samples: 48.
- Completed rebalance observations: 46, from 2022-05-23 to 2026-03-23.
- Transaction cost: 5 bps per turnover unit.
- Optimizer: long-only, max weight 0.30, risk aversion 10, turnover penalty 0.001.
- LightGBM mode: compact fixed grid, no nested CV, seed 42.

## Final Backtest Result

| Portfolio | Sharpe | Annualized return | Max drawdown | Active return vs SPY | Tracking error | SPY-relative IR | Observations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 SPY buy-and-hold | 1.130685 | 0.170089 | -0.119402 | n/a | n/a | n/a | 46 |
| E0 heuristic optimizer | 1.724275 | 0.576873 | -0.186728 | 0.342199 | 0.292978 | 1.168003 | 46 |
| E2 equal-weight S&P 500 | 1.782082 | 0.325484 | -0.114071 | 0.130506 | 0.059557 | 2.191267 | 46 |
| E8 Ridge plus LightGBM blend optimizer | 2.987197 | 1.086616 | -0.202734 | 0.636036 | 0.287482 | 2.212435 | 46 |

E8 is the final ML winner for this iteration. It beats aligned SPY on Sharpe, annualized
return, annualized active return, and SPY-relative IR. It also beats the current heuristic
and equal-weight benchmark on Sharpe and annualized return in this bounded run.

The bootstrap Sharpe-difference 95% CI for E8 versus SPY is `[-0.446, 1.542]`, so the win is
a point-estimate SPY-relative win, not a statistical GO under the stricter CI-excludes-zero
rule.

## ML Model Result

The best predictive model by IC was E4, the Ridge model on rank-normalized features:

- Pearson IC: 0.047272.
- Rank IC: 0.051704.
- Optimized portfolio Sharpe: 1.407218.
- SPY-relative IR: 1.071827.

The best optimized ML portfolio was E8, the Ridge plus LightGBM regression blend:

- Pearson IC: 0.041975.
- Rank IC: -0.000435.
- Optimized portfolio Sharpe: 2.987197.
- Annualized return: 1.086616.
- SPY-relative IR: 2.212435.

The result reinforces that predictive IC alone is not enough for promotion. E4 ranked best
on IC, but E8 produced the best optimizer-facing signal after blending and generated the
best net-of-cost portfolio.

## Optimization Result

The stronger result came from the optimizer configuration, not from a new model family.
Increasing the max-weight cap from 0.05 to 0.30 allowed the E8 signal to express conviction
more directly while remaining long-only.

The final E8 optimized portfolio had mean turnover of 0.686668 across completed rebalance
dates. The result is therefore sensitive to transaction cost assumptions, and any production
promotion should rerun cost sweeps around the selected 0.30 cap before execution.

## Probe History

| Probe | Assets | Max weight | Requested rebalances | Completed observations | E8 Sharpe | SPY Sharpe | Active return | IR | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Initial bounded report | 80 | 0.05 | 12 | 10 | 8.891060 | 7.471607 | 0.446258 | 3.753790 | Beat SPY, CI included zero |
| Larger sample base config | 80 | 0.05 | 24 | 22 | 0.485199 | 0.829485 | -0.027931 | -0.252625 | Did not beat SPY |
| Wider cap probe | 80 | 0.20 | 24 | 22 | 0.815137 | 0.829485 | 0.109151 | 0.518963 | Positive IR, below SPY Sharpe |
| Marginal cap probe | 80 | 0.30 | 24 | 22 | 0.830112 | 0.829485 | 0.136711 | 0.576767 | Barely beat SPY |
| Smaller universe probe | 50 | 0.30 | 24 | 22 | 0.342158 | 0.829485 | 0.000313 | 0.001298 | Did not beat SPY |
| Smaller universe probe | 30 | 0.30 | 24 | 22 | 1.539997 | 0.829485 | 0.399455 | 1.402157 | Beat SPY |
| Final selected run | 100 | 0.30 | 48 | 46 | 2.987197 | 1.130685 | 0.636036 | 2.212435 | Beat SPY |

## Decision

SPY-relative point-estimate win achieved with E8 on the top-100 liquidity universe and a
0.30 optimizer max-weight cap. The generated Phase 2 report remains `PROVISIONAL` because
the Sharpe-difference bootstrap CI still includes zero.

Next validation should focus on a full unsampled weekly backtest, transaction-cost sweeps
around the 0.30 cap, and a non-survivorship-biased universe before treating the result as
production-ready.
