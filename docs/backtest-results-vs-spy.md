# Backtest Results vs SPY

## Summary

The E8 Ridge + LightGBM optimizer was evaluated with the production trade-aware cost model:

```text
commission = abs(target_weight - previous_weight) * portfolio_value * 0.02
```

Example:

```text
portfolio_value = 1000
recommended buy = 50%
commission = 0.50 * 1000 * 0.02 = 10
```

In portfolio-weight terms:

```text
estimated_commission_weight = 0.50 * 0.02 = 0.01
```

This cost applies to both buys and sells because both are transactions.

The initial weekly E8 model with `lambda_turnover=0.001` failed under this cost model. I then swept
larger `lambda_turnover` values so the optimizer could select the best tested turnover penalty by
SPY-relative information ratio. The best tested value was:

```text
lambda_turnover = 5.0
```

That setting reduced mean turnover from `68.24%` to `5.68%` and reduced estimated commission drag
from `2.73%` to `0.23%` per rebalance. It still did **not** beat SPY, so the strategy remains
rejected.

## Test Setup

| Item | Value |
| --- | --- |
| Source artifacts | `data/runs/phase2-source-20260424` |
| Initial result artifact | `docs/experiments/e8-commission-2pct-20260424.json` |
| Turnover sweep artifact | `docs/experiments/e8-turnover-sweep-2pct-20260424.json` |
| Backtest window | 2022-05-23 to 2026-03-23 |
| Completed rebalance observations | 46 |
| Rebalance cadence | 5 business days |
| Forecast horizon | 5 trading days |
| Transaction cost | 2% of absolute traded notional |
| Turnover penalty sweep | 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5 |
| Sweep objective | Maximize information ratio |
| Universe | Top 100 assets by latest `dollar_volume_21d` |
| Portfolio constraints | Long-only, fully invested |
| Optimizer max weight | 30% |
| Risk aversion | 10 |
| Benchmark | SPY aligned to the same rebalance dates and horizon |

Turnover tuning command:

```bash
uv run stock-analysis tune-turnover \
  --candidate e8_baseline \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 48 \
  --optimizer-max-weight 0.30 \
  --risk-aversion 10 \
  --min-trade-weight 0.005 \
  --commission-rate 0.02 \
  --turnover-penalties 0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5 \
  --objective-metric information_ratio \
  --horizon-days 5 \
  --rebalance-step-days 5 \
  --embargo-days 15 \
  --covariance-lookback-days 252 \
  --iteration-id e8-turnover-sweep-2pct-20260424 \
  --json-output docs/experiments/e8-turnover-sweep-2pct-20260424.json
```

## Results

| Strategy | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe | Max Drawdown | Cumulative Active Return vs SPY | Annualized Active Return vs SPY | Information Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY buy-and-hold | 15.55% | 17.01% | 15.04% | 1.131 | -11.94% | 0.00% | n/a | n/a |
| E8, 2% commission, `lambda_turnover=0.001` | -43.39% | -46.12% | 36.51% | -1.263 | -44.44% | -51.00% | -71.73% | -2.539 |
| E8, 2% commission, selected `lambda_turnover=5.0` | 8.80% | 9.60% | 34.66% | 0.277 | -18.86% | -5.85% | -1.85% | -0.067 |

## Contribution-Aware Run

After adding contribution-aware backtesting, I reran E8 with the same selected turnover penalty,
2% commission, and a realistic deposit schedule:

```text
initial_portfolio_value = 1000
monthly_deposit_amount = 100
deposit_frequency_days = 30
no_trade_band = 0.02
lambda_turnover = 5.0
```

Artifact:

```text
docs/experiments/e8-contribution-aware-20260424.json
```

MLflow:

```text
experiment = stock-analysis-autoresearch
run_id = 9389faa1ef314962ba3df487b0900d09
tracking_uri = sqlite:///data/mlflow/mlflow.db
```

Same-deposit outcome:

| Metric | E8 Contribution-Aware | SPY Same Deposits |
| --- | ---: | ---: |
| Ending value | $6,671.40 | $6,151.76 |
| Active ending value | $519.64 | n/a |
| Total deposits | $4,700.00 | $4,700.00 |
| Total commissions | $316.18 | $114.00 |
| Commission / deposits | 6.73% | 2.43% |
| Time-weighted return | 25.29% | 15.55% |
| Money-weighted return | 7.11% | 3.43% |
| Return on invested capital | 17.04% | 7.93% |

Risk/skill metrics on the same rebalance dates:

| Strategy | Annualized Return | Annualized Volatility | Sharpe | Max Drawdown | Annualized Active Return vs SPY | Information Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY same dates | 17.01% | 15.04% | 1.131 | -11.94% | n/a | n/a |
| E8 contribution-aware | 27.77% | 35.79% | 0.776 | -17.45% | 13.86% | 0.521 |

Interpretation:

```text
The same-deposit investor outcome beat SPY in ending dollars.
The strategy still failed the existing promotion gate because candidate Sharpe did not beat SPY Sharpe.
```

This is the correct distinction. Deposits improve investor wealth and reduce forced selling, but the
model gate should still use time-weighted, benchmark-relative metrics because deposits are external
cash flows.

## Turnover Penalty Sweep

| Lambda Turnover | Cumulative Return | Annualized Return | Sharpe | Information Ratio | Mean Turnover | Commission Drag / Rebalance | Max Drawdown |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | -43.39% | -46.12% | -1.263 | -2.539 | 68.24% | 2.73% | -44.44% |
| 0.005 | -43.36% | -46.09% | -1.262 | -2.537 | 68.14% | 2.73% | -44.42% |
| 0.010 | -43.33% | -46.06% | -1.260 | -2.533 | 68.02% | 2.72% | -44.38% |
| 0.020 | -43.27% | -46.00% | -1.258 | -2.527 | 67.77% | 2.71% | -44.32% |
| 0.050 | -43.09% | -45.81% | -1.249 | -2.509 | 66.98% | 2.68% | -44.15% |
| 0.100 | -42.81% | -45.52% | -1.239 | -2.489 | 65.60% | 2.62% | -43.87% |
| 0.200 | -39.46% | -42.05% | -1.153 | -2.307 | 61.09% | 2.44% | -40.69% |
| 0.500 | -33.84% | -36.18% | -0.997 | -1.999 | 51.37% | 2.05% | -38.19% |
| 1.000 | -21.99% | -23.66% | -0.609 | -1.218 | 39.70% | 1.59% | -32.66% |
| 2.000 | 0.58% | 0.63% | 0.015 | -0.232 | 24.50% | 0.98% | -24.06% |
| 5.000 | 8.80% | 9.60% | 0.277 | -0.067 | 5.68% | 0.23% | -18.86% |

The selected penalty cuts turnover materially:

```text
initial_abs_trade_weight = 2 * 0.6824 = 1.3648
initial_commission_drag = 1.3648 * 0.02 = 0.0273

selected_abs_trade_weight = 2 * 0.0568 = 0.1136
selected_commission_drag = 0.1136 * 0.02 = 0.0023
```

## Comparison To Previous 5 bps Baseline

The earlier research baseline used a much smaller cost assumption: 5 bps per turnover unit. Under
that assumption, E8 had a strong point estimate:

| Strategy | Cumulative Return | Annualized Return | Sharpe | Max Drawdown | Information Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| E8 Ridge + LightGBM, old 5 bps baseline | 96.74% | 108.66% | 2.987 | -20.27% | 2.212 |

That result is no longer the relevant production result if the intended commission is 2% of traded
notional. The 2% assumption changes the optimizer/backtest economics materially.

## Optimization Model In The Selected Run

The objective was:

```text
maximize_w  mu^T w
          - gamma * w^T Sigma w
          - lambda_turnover * ||w - w_prev||_1
          - commission_rate * ||w - w_prev||_1
```

Subject to:

```text
sum(w) = 1
w_i >= 0
w_i <= max_weight
```

Selected parameters:

```text
gamma = 10
lambda_turnover = 5.0
commission_rate = 0.02
max_weight = 0.30
```

## Leakage Controls

The backtest is designed to avoid target leakage inside each rebalance prediction.

At a rebalance date `t`, the runner uses this sequence:

1. Build point-in-time features for every asset at date `t`.
2. Define `train_cutoff = t - embargo_days`.
3. Train the model only on rows with `feature_date < train_cutoff`.
4. Predict only on the cross-section where `feature_date = t`.
5. Optimize weights from those predictions and covariance estimated from returns before `t`.
6. Score the portfolio with the realized forward return from `t` to `t + horizon`.

For this run:

```text
horizon = 5 business days
embargo = 15 business days
```

The 15-business-day embargo is deliberately longer than the 5-day target horizon. That means the
latest training label used for a rebalance at `t` ends before the prediction date. This avoids a
training sample whose forward-return label overlaps the prediction period.

The model receives only numeric feature columns. Columns beginning with `fwd_` are explicitly
excluded from the prediction feature frame, so forward returns are not passed as predictors.

## Interpretation

The correct interpretation is:

```text
No obvious target leakage in the walk-forward prediction loop.
A larger turnover penalty materially improves the strategy under 2% commission.
The selected weekly E8 strategy still does not beat SPY.
```

This does not prove the forecast model has no signal. It proves that the weekly rebalance strategy
is not strong enough after a realistic turnover penalty is selected under a 2% commission
assumption.

## Limitations

- Current S&P 500 constituents create survivorship bias.
- Latest liquidity selection can introduce look-ahead bias in universe filtering.
- E8 was selected after research comparisons on the same historical window.
- The universe is restricted to the top 100 names by latest liquidity.
- The backtest has only 46 completed rebalance observations.
- Transaction costs are simplified and do not include bid/ask spread or market impact.
- The optimizer allows 30% max single-name weight, which is aggressive for a diversified portfolio.

## Recommended Next Steps

1. Test a longer rebalance cadence, such as 21 trading days, to reduce transaction frequency.
2. Add a no-trade band so small signal changes do not trigger expensive rebalances.
3. Lower max single-name weight and add sector caps for a more realistic production portfolio.
4. Sweep `lambda_turnover` beyond `5.0` only if paired with lower rebalance frequency.
5. Replace current-constituent membership and latest-liquidity selection with point-in-time inputs.
