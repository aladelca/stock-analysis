# Backtest Results vs SPY

## Summary

The current research backtest shows the ML optimizer outperforming SPY on point estimates over the
tested window.

These reported figures are the pre-trade-aware research baseline using the earlier 5 bps cost
assumption. The production optimizer now supports a 2% commission-rate assumption and current
holdings. A new research run should be generated before comparing the trade-aware production model
against SPY.

This is not yet production proof. The run uses current S&P 500 constituents, so survivorship bias is
present, and the Sharpe-difference confidence interval still includes zero.

## Test Setup

| Item | Value |
| --- | --- |
| Source artifacts | `data/runs/phase2-source-20260424` |
| Backtest window | 2022-05-23 to 2026-03-23 |
| Completed rebalance observations | 46 |
| Rebalance cadence | 5 business days |
| Forecast horizon | 5 trading days |
| Transaction cost | Historical baseline: 5 bps per turnover unit |
| Universe | Top 100 assets by latest `dollar_volume_21d` |
| Portfolio constraints | Long-only, fully invested |
| Optimizer max weight | 30% |
| Risk aversion | 10 |
| Turnover penalty | 0.001 |
| Benchmark | SPY aligned to the same rebalance dates and horizon |

## Results

| Strategy | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe | Max Drawdown | Cumulative Active Return vs SPY | Annualized Active Return vs SPY | Information Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SPY buy-and-hold | 15.55% | 17.01% | 15.04% | 1.131 | -11.94% | 0.00% | n/a | n/a |
| Current heuristic optimizer | 52.05% | 57.69% | 33.46% | 1.724 | -18.67% | 31.59% | 34.22% | 1.168 |
| Equal-weight S&P 500 | 29.59% | 32.55% | 18.26% | 1.782 | -11.41% | 12.16% | 13.05% | 2.191 |
| E8 Ridge + LightGBM blend optimizer | 96.74% | 108.66% | 36.38% | 2.987 | -20.27% | 70.27% | 63.60% | 2.212 |

## Current Production Candidate

The current configured model is:

```text
phase2-e8-ridge-lightgbm-blend-v1
```

It blends Ridge and LightGBM regression forecasts, then sends the scores into the same long-only
mean-variance optimizer used by the heuristic baseline.

The historical optimizer objective for these results was:

```text
maximize mu^T w - gamma * w^T Sigma w - lambda_turnover * ||w - w_prev||_1
```

The current production objective adds explicit commission drag:

```text
maximize mu^T w
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

In the reported run:

```text
gamma = 10
lambda_turnover = 0.001
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

The relevant implementation is:

```text
train_cutoff = rebalance_date - BDay(embargo_days)
train rows = date < train_cutoff and target is known
prediction rows = date == rebalance_date
covariance rows = return date < rebalance_date
```

The model receives only numeric feature columns. Columns beginning with `fwd_` are explicitly
excluded from the prediction feature frame, so forward returns are not passed as predictors.

## Training Time vs Prediction Time

Example for a rebalance on `2026-03-23`:

| Stage | Date Logic | What Is Available |
| --- | --- | --- |
| Training data | rows with feature dates before `2026-03-02` | Historical features and labels that would already be realized before the prediction date |
| Prediction data | rows with feature date `2026-03-23` | Same-day point-in-time features only |
| Risk estimate | daily returns before `2026-03-23` | Historical covariance only |
| Realized scoring | forward return after `2026-03-23` | Used only after the prediction to evaluate the backtest |

The model is refit at each rebalance using only that rebalance's allowed training window. It does
not train once on the full dataset and then replay historical predictions.

The one-shot live pipeline is similar but not identical. It trains on rows before the latest feature
date where the target label is non-null, then predicts the latest cross-section. Because labels near
the latest date are null until future prices exist, the latest prediction date is not used as a
training label. However, the live one-shot training path does not currently apply the same explicit
15-business-day embargo used by the research backtest. Adding that embargo to the live path would
make production inference more conservative and more consistent with the backtest contract.

## Model Validation Methodology

Validation has three layers:

1. Feature/label leakage tests. Unit tests verify that features at `(ticker, t)` use prices through
   `t`, while labels use only the configured forward horizon after `t`. Tests also verify embargoed
   walk-forward splits and train-only preprocessing statistics.
2. Walk-forward portfolio validation. Each candidate model is evaluated through the same rebalance
   loop, optimizer, transaction-cost assumption, and SPY alignment. Portfolio returns are calculated
   only from predictions made at each rebalance date.
3. SPY-relative comparison. Portfolio period returns and SPY forward returns are aligned by exact
   rebalance date before calculating active return, tracking error, and information ratio.

The validation metrics reported are:

| Metric | Meaning |
| --- | --- |
| Pearson IC | Linear correlation between prediction and realized forward return |
| Rank IC | Rank correlation between prediction and realized forward return |
| Sharpe | Annualized return divided by annualized volatility |
| Max drawdown | Worst cumulative drawdown over the tested period |
| Active return | Mean portfolio return minus SPY return, annualized |
| Information ratio | Annualized active return divided by annualized tracking error |

## What Is Still Not Solved

The current result does not have forward-return target leakage inside the rebalance loop, but it is
still not a final production validation.

Remaining research risks:

- Survivorship bias remains because the universe uses current S&P 500 constituents.
- Model-selection overfitting is possible because E8 was selected after comparing multiple candidates
  on the same historical research window.
- The top-100 liquidity filter uses latest liquidity, which is convenient for research but should be
  replaced with point-in-time liquidity selection.
- The result is based on 46 completed rebalances, which is a small sample.
- The Sharpe-difference confidence interval against SPY includes zero.

So the correct interpretation is:

```text
No obvious target leakage in the walk-forward prediction loop.
Still provisional because survivorship bias, latest-liquidity look-ahead risk,
and model-selection bias remain.
```

## Interpretation

The E8 optimizer produced the strongest result in this run:

- It compounded to 96.74% over the tested rebalance periods.
- It beat aligned SPY by 70.27% cumulatively.
- It produced a 2.987 Sharpe versus 1.131 for SPY.
- It generated 63.60% annualized active return versus SPY.

The result is still provisional because the bootstrap Sharpe-difference 95% confidence interval
against SPY was:

```text
[-0.446, 1.542]
```

That interval includes zero, so the point estimate is strong but not statistically conclusive under
the stricter promotion rule.

## Limitations

- Current S&P 500 constituents create survivorship bias.
- Latest liquidity selection can introduce look-ahead bias in universe filtering.
- E8 was selected after research comparisons on the same historical window.
- The universe is restricted to the top 100 names by latest liquidity.
- The backtest has only 46 completed rebalance observations.
- Transaction costs are simplified and do not include market impact.
- Turnover is high for the ML optimizer, approximately 68.67% per rebalance.
- The optimizer allows 30% max single-name weight, which is aggressive for a diversified portfolio.

## Recommended Next Steps

1. Replace current-constituent membership with point-in-time S&P 500 membership.
2. Run a full unsampled weekly backtest across all available dates.
3. Add benchmark-relative exposure constraints and rerun with the production sector cap.
4. Sweep transaction costs and turnover penalties around the E8 model.
5. Promote only after out-of-sample validation where the Sharpe-difference confidence interval
   excludes zero.
