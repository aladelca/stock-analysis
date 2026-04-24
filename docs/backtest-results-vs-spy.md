# Backtest Results vs SPY

## Summary

The E8 Ridge + LightGBM optimizer was rerun with the production trade-aware cost model:

```text
commission = abs(target_weight - previous_weight) * portfolio_value * 0.02
```

Example:

```text
portfolio_value = 1000
recommended buy = 50%
commission = 0.50 * 1000 * 0.02 = 10
```

In portfolio-weight terms, the same trade has:

```text
estimated_commission_weight = 0.50 * 0.02 = 0.01
```

This cost applies to both buys and sells because both are transactions. With this assumption, the
current high-turnover E8 model does **not** beat SPY. The strategy is rejected on point estimates:
negative return, negative active return, and negative information ratio.

## Test Setup

| Item | Value |
| --- | --- |
| Source artifacts | `data/runs/phase2-source-20260424` |
| Result artifact | `docs/experiments/e8-commission-2pct-20260424.json` |
| Backtest window | 2022-05-23 to 2026-03-23 |
| Completed rebalance observations | 46 |
| Rebalance cadence | 5 business days |
| Forecast horizon | 5 trading days |
| Transaction cost | 2% of absolute traded notional |
| Universe | Top 100 assets by latest `dollar_volume_21d` |
| Portfolio constraints | Long-only, fully invested |
| Optimizer max weight | 30% |
| Risk aversion | 10 |
| Turnover penalty | 0.001 |
| Benchmark | SPY aligned to the same rebalance dates and horizon |

Command:

```bash
uv run stock-analysis autoresearch-eval \
  --candidate e8_baseline \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 48 \
  --optimizer-max-weight 0.30 \
  --risk-aversion 10 \
  --min-trade-weight 0.005 \
  --lambda-turnover 0.001 \
  --commission-rate 0.02 \
  --horizon-days 5 \
  --rebalance-step-days 5 \
  --embargo-days 15 \
  --covariance-lookback-days 252 \
  --iteration-id e8-commission-2pct-20260424 \
  --json-output docs/experiments/e8-commission-2pct-20260424.json
```

## Results

| Strategy | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe | Max Drawdown | Cumulative Active Return vs SPY | Annualized Active Return vs SPY | Information Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY buy-and-hold | 15.55% | 17.01% | 15.04% | 1.131 | -11.94% | 0.00% | n/a | n/a |
| E8 Ridge + LightGBM, 2% commission | -43.39% | -46.12% | 36.51% | -1.263 | -44.44% | -51.00% | -71.73% | -2.539 |

Additional trade-cost diagnostics:

| Metric | Value |
| --- | ---: |
| Mean turnover per rebalance | 68.24% |
| Mean absolute traded weight per rebalance | 136.48% |
| Approx. commission drag per rebalance | 2.73% |
| Sharpe difference vs SPY | -2.394 |
| Sharpe difference 95% bootstrap CI | [-2.349, -0.560] |
| Decision | rejected |

The commission drag is large because turnover is high:

```text
mean_abs_trade_weight = 2 * mean_turnover = 2 * 0.6824 = 1.3648
mean_commission_drag = 1.3648 * 0.02 = 0.0273
```

That means the model pays roughly 2.73% of portfolio value in commission every 5-business-day
rebalance on average. This overwhelms the forecast edge in the current E8 configuration.

## Comparison To Previous 5 bps Baseline

The earlier research baseline used a much smaller cost assumption: 5 bps per turnover unit. Under
that assumption, E8 had a strong point estimate:

| Strategy | Cumulative Return | Annualized Return | Sharpe | Max Drawdown | Information Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| E8 Ridge + LightGBM, old 5 bps baseline | 96.74% | 108.66% | 2.987 | -20.27% | 2.212 |

That result is no longer the relevant production result if the intended commission is 2% of traded
notional. The 2% assumption changes the optimizer/backtest economics materially.

## Optimization Model In This Run

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

Parameters:

```text
gamma = 10
lambda_turnover = 0.001
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
The E8 model is not viable under a 2% commission-on-traded-notional assumption.
The primary failure mode is excessive turnover relative to transaction cost.
```

This does not prove the forecast model has no signal. It proves that the current optimizer settings
and rebalance cadence are not compatible with a 2% commission assumption.

## Limitations

- Current S&P 500 constituents create survivorship bias.
- Latest liquidity selection can introduce look-ahead bias in universe filtering.
- E8 was selected after research comparisons on the same historical window.
- The universe is restricted to the top 100 names by latest liquidity.
- The backtest has only 46 completed rebalance observations.
- Transaction costs are simplified and do not include bid/ask spread or market impact.
- The optimizer allows 30% max single-name weight, which is aggressive for a diversified portfolio.

## Recommended Next Steps

1. Increase the turnover penalty substantially and rerun the same SPY-relative backtest.
2. Test a longer rebalance cadence, such as 21 trading days, to reduce transaction frequency.
3. Add a no-trade band so small signal changes do not trigger expensive rebalances.
4. Lower max single-name weight and add sector caps for a more realistic production portfolio.
5. Replace current-constituent membership and latest-liquidity selection with point-in-time inputs.
