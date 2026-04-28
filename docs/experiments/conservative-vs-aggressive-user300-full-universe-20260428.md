# Conservative vs Aggressive Full-Universe Backtest - 2026-04-28

## Setup

- Source artifacts: `data/runs/20260428T205216Z`
- Candidate: `lightgbm_return_zscore`
- Universe: full eligible universe, including SPY in the asset panel
- Horizon: 5 trading days
- Rebalance step: 5 trading days plus deposit-mapped rebalance dates
- Requested max rebalances: 241
- Matched SPY-relative observations: 235
- Initial portfolio value: $300
- Monthly deposit amount: $100
- Deposit frequency: 30 calendar days
- Total deposits: $4,800

## Profile Settings

| Profile | Risk aversion | Lambda turnover | Commission rate | No-trade band |
| --- | ---: | ---: | ---: | ---: |
| Conservative | 10.0 | 5.0 | 2.0% | 4.0% |
| Aggressive | 4.0 | 0.005 | 0.2% | 1.0% |

## Results

| Metric | Conservative | Aggressive |
| --- | ---: | ---: |
| Decision | provisional | provisional |
| Candidate Sharpe | 1.469 | 1.331 |
| SPY Sharpe | 1.142 | 1.142 |
| Sharpe difference | 0.328 | 0.190 |
| Sharpe diff CI low | -0.513 | -0.822 |
| Annualized return | 40.58% | 50.06% |
| Active return vs SPY | 20.41% | 30.22% |
| Information ratio | 1.034 | 1.006 |
| Max drawdown | -23.40% | -40.99% |
| Mean turnover | 4.27% | 57.73% |
| Strategy ending value | $16,650.02 | $18,662.87 |
| Same-cashflow SPY ending value | $8,148.97 | $8,298.64 |
| Active ending value | $8,501.05 | $10,364.23 |
| Strategy TWR | 395.78% | 573.71% |
| Strategy MWR | 63.49% | 70.43% |
| Strategy total commissions | $2,355.07 | $3,195.16 |
| Commissions / deposits | 49.06% | 66.57% |

## Interpretation

Aggressive produced a higher ending value and higher annualized return, but it did so by taking
substantially more risk and turnover. Its mean turnover was about 13.5x the conservative profile,
its drawdown was materially worse, and its total commissions were higher despite the lower
commission-rate assumption.

Conservative had the better risk-adjusted profile: higher Sharpe, higher information ratio, less
drawdown, and much lower turnover. Neither profile is a statistically confirmed GO because both
Sharpe-difference confidence intervals include zero.

## Recommendation

Do not promote the aggressive profile as the default based on this backtest. It can be kept as an
explicit high-risk mode, but the conservative profile remains the better risk-adjusted default under
the current walk-forward evidence.

## Methodology Caveat

This uses the existing autoresearch backtest harness. It compares optimizer profiles using raw model
forecast scores. It does not yet replay the live calibrated expected-return workflow or the
SPY-relative calibrated-return gate. Follow-up bead `stock-analysis-d5g` tracks adding that live-flow
backtest mode.

