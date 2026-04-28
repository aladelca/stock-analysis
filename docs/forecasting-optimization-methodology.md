# Forecasting And Optimization Methodology

This document describes the scientific methodology implemented in the current MVP. It is intentionally explicit about what is a forecast, what is only a score, how risk is estimated, and how the portfolio optimization problem is formulated.

## Scope

The current system is a one-shot, end-of-day, long-only S&P 500 portfolio assistant. It does not execute trades. It produces target portfolio weights and Tableau-ready diagnostics from historical adjusted prices.

The implemented methodology has three stages:

1. Price-derived feature construction.
2. Baseline forecast-score and covariance estimation.
3. Long-only mean-variance-style portfolio optimization.

## Data Inputs

For each eligible asset `i`, the pipeline uses daily adjusted close prices:

```text
P_{i,t}
```

where:

- `i` indexes an asset in the S&P 500 universe.
- `t` indexes a trading day.
- `P_{i,t}` is the adjusted close price.
- The canonical `as_of_date` is the latest available market data date, not the wall-clock run date.

Daily simple returns are computed as:

```text
r_{i,t} = P_{i,t} / P_{i,t-1} - 1
```

An asset is eligible for optimization only if it has at least `features.min_history_days` valid adjusted close observations and finite forecast/risk inputs.

Default config:

```yaml
features:
  min_history_days: 252
  momentum_windows: [63, 126, 252]
  volatility_window: 63
  drawdown_window: 252
  moving_average_windows: [50, 200]

forecast:
  momentum_window: 252
  volatility_penalty: 0.25
  covariance_lookback_days: 252

optimizer:
  max_weight: 0.05
  risk_aversion: 10.0
```

## Feature Engineering

### Momentum

For a configured lookback window `L`, momentum is:

```text
M_{i,L} = P_{i,T} / P_{i,T-L} - 1
```

where `T` is the latest available price date for the asset.

The default windows are:

```text
L in {63, 126, 252}
```

These approximate 3-month, 6-month, and 12-month trading horizons.

### Annualized Volatility

For a volatility window `V`, annualized volatility is:

```text
sigma_i = std(r_{i,T-V+1:T}) * sqrt(252)
```

The MVP default uses:

```text
V = 63
```

This is a short-horizon volatility estimate, roughly 3 trading months.

### Maximum Drawdown

For a drawdown window `D`, maximum drawdown is:

```text
DD_i = min_{t in T-D+1:T} (P_{i,t} / max_{s <= t} P_{i,s} - 1)
```

This is currently computed for diagnostics and future model work. It is not yet used directly in the optimizer objective.

### Moving Average Ratios

For a moving average window `A`:

```text
MA_{i,A} = mean(P_{i,T-A+1:T})
```

```text
PriceToMA_{i,A} = P_{i,T} / MA_{i,A}
```

These are currently emitted as features for diagnostics and future forecast improvements. They are not yet used directly in the MVP forecast score.

## Forecast Methodology

### Current MVP Forecast Is A Score

The current field is named `expected_return` in the optimizer input, but scientifically it should be interpreted as a forecast score rather than a calibrated expected return.

For each eligible asset `i`, the MVP computes:

```text
mu_i = M_{i,F} - lambda_vol * sigma_i
```

where:

- `mu_i` is the forecast score used by the optimizer.
- `M_{i,F}` is momentum over the configured forecast momentum window.
- `F = forecast.momentum_window`, default `252`.
- `sigma_i` is annualized volatility over the configured volatility window.
- `lambda_vol = forecast.volatility_penalty`, default `0.25`.

With defaults:

```text
mu_i = momentum_252d_i - 0.25 * volatility_63d_i
```

Interpretation:

- Positive momentum increases the asset score.
- High volatility reduces the asset score.
- The score is cross-sectional and heuristic.
- It is not yet probability-calibrated.
- It is not yet a statistically validated return forecast.

### Eligibility Filter

An asset remains eligible only if:

```text
history_days_i >= min_history_days
```

```text
mu_i is finite
```

```text
sigma_i is finite and sigma_i > 0
```

Assets that fail these filters receive zero portfolio weight.

## Risk Model

The risk model uses the empirical covariance matrix of daily returns over the latest `K` trading days:

```text
K = forecast.covariance_lookback_days
```

Default:

```text
K = 252
```

Let:

```text
R = matrix of daily returns for eligible assets over the latest K days
```

The annualized covariance matrix is:

```text
Sigma = cov(R) * 252
```

Implementation details:

- Missing returns in the pivoted return matrix are filled with `0`.
- The covariance matrix is symmetrized before optimization.
- A small diagonal jitter is added in the optimizer to improve numerical stability:

```text
Sigma_stable = (Sigma + Sigma^T) / 2 + epsilon * I
```

where:

```text
epsilon = 1e-8
```

## Optimization Model

### Sets And Indices

Let:

```text
N = number of eligible assets
i = 1, ..., N
```

### Parameters

```text
mu_i
```

Forecast score for asset `i`.

```text
Sigma
```

Annualized covariance matrix of eligible asset returns.

```text
gamma
```

Risk aversion parameter:

```text
gamma = optimizer.risk_aversion
```

Default:

```text
gamma = 10.0
```

```text
w_max
```

Maximum allowed allocation per asset:

```text
w_max = optimizer.max_weight
```

Default:

```text
w_max = 0.05
```

### Decision Variables

The optimizer solves for portfolio weights:

```text
w_i for i = 1, ..., N
```

where:

```text
w_i = fraction of portfolio capital allocated to asset i
```

In vector notation:

```text
w = [w_1, w_2, ..., w_N]^T
```

The trade-aware layer also uses the current portfolio weights:

```text
w_prev_i = current portfolio weight for asset i before rebalance
```

Current weights come from `portfolio_state.current_holdings_path`. If a contribution is configured,
the optimizer uses post-contribution current weights:

```text
w_prev_post_i = current_market_value_i / (portfolio_value_before_contribution + contribution)
```

If no holdings file is configured, the run is treated as a first allocation from cash:

```text
w_prev_i = 0 for all optimizer assets
```

### Objective Function

The optimizer solves a long-only, trade-aware mean-variance-style quadratic optimization problem:

```text
maximize_w  mu^T w
          - gamma * w^T Sigma w
          - lambda_turnover * ||w - w_prev||_1
          - commission_rate * ||w - w_prev||_1
```

Interpretation:

- `mu^T w` rewards assets with higher forecast scores.
- `w^T Sigma w` penalizes portfolio variance.
- `gamma` controls the return-score versus risk tradeoff.
- `||w - w_prev||_1` measures absolute traded portfolio weight.
- `lambda_turnover` discourages unnecessary rebalance churn.
- `commission_rate` models direct transaction cost. The default production assumption is `0.02`,
  meaning 2% of the absolute traded notional.
- The current production config uses `lambda_turnover = 5.0`, selected from a turnover-penalty
  sweep under the 2% commission model.
- Larger `gamma` produces a more risk-averse allocation.
- Smaller `gamma` allows more concentration in high-score assets, subject to constraints.

This is implemented with CVXPY as:

```text
Maximize(
    mu @ weights
    - risk_aversion * quad_form(weights, Sigma)
    - lambda_turnover * norm1(weights - previous_weights)
    - commission_rate * norm1(weights - previous_weights)
)
```

### Constraints

#### Fully Invested Constraint

```text
sum_i w_i = 1
```

The portfolio allocates 100% of available capital.

#### Long-Only Constraint

```text
w_i >= 0 for all i
```

Short positions are not allowed.

#### Maximum Single-Asset Weight

```text
w_i <= w_max for all i
```

Default:

```text
w_i <= 0.05
```

No single asset can receive more than 5% of portfolio weight.

#### Optional Sector Cap

If `optimizer.sector_max_weight` is configured, each sector receives an aggregate upper bound:

```text
sum_{i in sector s} w_i <= sector_max_weight
```

The default production config uses:

```text
sector_max_weight = 0.35
```

This prevents the optimizer from allocating too much of the portfolio to one GICS sector.

#### Optional Trade Budget

If `optimizer.max_trade_abs_weight` is configured, the optimizer limits total absolute trade size:

```text
||w - w_prev||_1 <= max_trade_abs_weight
```

This is a convex turnover budget. It is separate from `lambda_turnover`, which is a penalty in the
objective rather than a hard limit.

#### Eligibility Constraint

Only assets passing the data-quality and forecast-quality filters are included in the optimization universe.

Equivalently, for ineligible assets `j`:

```text
w_j = 0
```

The implementation enforces this by excluding ineligible assets from the optimization problem and assigning them zero weight in recommendations.

#### Minimum Feasible Universe Check

Before solving, the system checks:

```text
N >= ceil(1 / w_max)
```

This is required because a fully invested portfolio cannot satisfy `w_i <= w_max` if there are too few eligible assets.

With default `w_max = 0.05`:

```text
N >= 20
```

## Solver Behavior

The optimizer attempts available solvers in this order unless a solver is configured:

```text
CLARABEL
OSQP
SCS
CVXPY default
```

The problem is convex when `Sigma` is positive semidefinite. The implementation uses `cp.psd_wrap(Sigma)` after symmetrizing and adding diagonal jitter.

After solving, tiny negative numerical weights are clipped to zero and weights are renormalized to sum to one.

## Output Interpretation

### `portfolio_recommendations`

Each row contains:

- `ticker`
- `security`
- `gics_sector`
- `expected_return`
- `volatility`
- `current_weight`
- `target_weight`
- `trade_weight`
- `trade_abs_weight`
- `estimated_commission_weight`
- `net_trade_weight_after_commission`
- `cash_required_weight`
- `cash_released_weight`
- `rebalance_required`
- `action`
- `reason_code`
- `as_of_date`
- `run_id`

Trade math:

```text
trade_weight_i = target_weight_i - current_weight_i
trade_abs_weight_i = abs(trade_weight_i)
estimated_commission_weight_i = commission_rate * trade_abs_weight_i
```

Action logic:

```text
BUY     if trade_weight_i >= min_rebalance_trade_weight
SELL    if trade_weight_i <= -min_rebalance_trade_weight
HOLD    if abs(trade_weight_i) < min_rebalance_trade_weight
        and (target_weight_i > 0 or current_weight_i > 0)
EXCLUDE if target_weight_i = 0 and current_weight_i = 0
```

Current holdings that are not in the optimizer universe are preserved in the recommendation output
as `SELL` rows with `target_weight = 0`. This keeps unsupported positions visible instead of
dropping them silently.

Cash-flow helper fields:

```text
cash_required_weight_i = trade_weight_i + estimated_commission_weight_i for BUY rows
cash_released_weight_i = trade_abs_weight_i - estimated_commission_weight_i for SELL rows
```

When a portfolio value or market-value holdings are available, recommendations also include dollar
fields:

```text
portfolio_value_before_contribution
contribution_amount
portfolio_value_after_contribution
current_market_value
target_market_value
trade_notional
commission_amount
deposit_used_amount
cash_after_trade_amount
no_trade_band_applied
```

Commission dollars are computed from traded notional:

```text
commission_amount_i = abs(trade_notional_i) * commission_rate
```

Example with a deposit:

```text
portfolio value before contribution = 1000
contribution = 100
portfolio value after contribution = 1100
BUY trade = 50% of post-contribution value = 550
commission = 550 * 0.02 = 11
```

The no-trade band is an execution rule, not a mixed-integer optimizer constraint:

```text
if abs(trade_weight_i) < execution.no_trade_band:
    action_i = HOLD
```

This keeps the optimization problem convex while suppressing small trades in the recommendation
layer.

## Contribution-Aware Backtesting

The walk-forward backtest can model fixed external contributions every configured number of calendar
days. Calendar deposit dates are mapped to the next available rebalance date.

At each rebalance date `t`, the sequence is:

1. Train on point-in-time rows before `t - embargo_days`.
2. Predict scores at `t`.
3. Add the scheduled external contribution.
4. Dilute existing weights onto the post-contribution portfolio base.
5. Optimize target weights using post-contribution `w_prev`.
6. Charge commission on absolute executed trade notional.
7. Hold the portfolio for the forecast horizon.

The backtest reports two different performance concepts:

```text
TWR = time-weighted return
```

TWR measures strategy skill independent of deposit timing. It is the primary metric for model and
SPY-relative validation.

```text
MWR / XIRR = money-weighted return
```

MWR measures the investor-specific result after external deposits. It depends on when cash was added
and is reported as an outcome metric, not as the primary model-skill gate.

Same-deposit SPY comparison uses the same initial value, same deposit dates, same deposit amounts,
and the same commission assumption.

### `portfolio_risk_metrics`

The pipeline reports:

```text
expected_return = mu^T w
```

```text
expected_volatility = sqrt(w^T Sigma w)
```

```text
num_holdings = count(w_i > 0)
```

```text
max_weight = max_i w_i
```

```text
concentration_hhi = sum_i w_i^2
```

Important: `expected_return` is currently based on the heuristic forecast score `mu`, not a calibrated statistical expected return.

### `sector_exposure`

Sector exposure is computed as:

```text
sector_weight_s = sum_{i in sector s} w_i
```

When `optimizer.sector_max_weight` is configured, the same sector exposure calculation is also used
as an optimization constraint.

## Scientific Assumptions

The current implementation assumes:

- Historical adjusted prices are sufficient for a first baseline model.
- Medium-term momentum contains useful cross-sectional information.
- Recent volatility is a meaningful penalty for unstable assets.
- Empirical covariance over the latest 252 trading days is an acceptable first-pass portfolio risk model.
- Long-only mean-variance optimization is a reasonable starting point for portfolio construction.
- A max-weight cap reduces single-name concentration risk.
- A configurable sector cap reduces sector concentration risk.
- Current holdings are supplied as portfolio weights or market values, not share lots.

## Known Limitations

- Calibrated expected returns are available only when the calibration gate passes. If calibration is
  disabled or fails minimum-observation checks, `forecast_score` remains a ranking signal rather
  than a predicted percentage return.
- Missing returns are filled with zero in the covariance matrix, which can understate risk for sparse assets.
- Transaction costs are modeled as a flat percentage commission, not market impact or spread cost.
- Current portfolio input is weight-based; share-count order generation is not implemented.
- Fixed-dollar one-shot contributions require either market-value holdings or an explicit portfolio
  value so dollars can be converted into post-contribution weights.
- No-trade bands are applied after optimization, so residual cash can remain when small trades are
  suppressed.
- There are no industry, beta, or liquidity constraints yet.
- The covariance matrix uses a simple empirical estimator rather than shrinkage or factor risk modeling.
- The optimizer can produce tiny numerical weights due to solver tolerance.

## Recommended Next Methodology Improvements

1. Add a stronger walk-forward validation gate for calibrated expected returns, including bucket
   calibration plots and post-cost SPY comparison.
2. Add walk-forward backtesting with out-of-sample evaluation.
3. Add covariance shrinkage, such as Ledoit-Wolf, to improve stability.
4. Add transaction costs and turnover constraints.
5. Add current holdings input to generate true buy/sell orders.
6. Add sector and industry caps.
7. Add benchmark-relative diagnostics versus SPY or equal-weight S&P 500.
8. Add robust forecast scaling, winsorization, and feature standardization.
9. Add confidence intervals or forecast uncertainty bands.
10. Add model versioning to every forecast and optimization output.
11. Add share-count order generation and tax-lot-aware execution after broker/account integration is
    in scope.
