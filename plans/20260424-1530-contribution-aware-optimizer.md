# Contribution-Aware Portfolio Optimization

## Goal

- Add monthly deposit modeling to the research and recommendation workflow so the system can answer:
  - How should new cash be allocated?
  - Which assets should be bought or sold, in percentages and dollars?
  - How much commission is paid when buys and sells occur?
  - How does the strategy perform versus SPY when both receive the same deposits?
- Preserve the existing forecast validation discipline by reporting time-weighted return separately
  from investor cash-flow outcomes such as ending value and money-weighted return.

## Request Snapshot

- User request: "create a plan to include monthly paycheck-style deposits every 30 natural days, model their effect on results, and improve the model ideas around deposit-aware rebalancing."
- Owner or issue: `stock-analysis-sdm`
- Plan file: `plans/20260424-1530-contribution-aware-optimizer.md`

## Current State

- `src/stock_analysis/backtest/runner.py` implements `run_walk_forward_backtest` as a long-format
  one row per ticker/rebalance DataFrame.
- `BacktestConfig` currently controls `horizon_days`, `rebalance_step_days`, `embargo_days`,
  `commission_rate`, `cost_bps`, `covariance_lookback_days`, selected features, and
  `max_rebalances`.
- Backtest state is currently weight-only:
  - first rebalance starts from `previous_weights = None`;
  - each later rebalance uses prior target weights as `w_prev`;
  - transaction cost is subtracted from period return;
  - no explicit cash account, deposit schedule, portfolio value path, or money-weighted return exists.
- `src/stock_analysis/optimization/engine.py` optimizes fully invested long-only weights:

```text
maximize_w  mu^T w
          - gamma * w^T Sigma w
          - lambda_turnover * ||w - w_prev||_1
          - commission_rate * ||w - w_prev||_1

subject to sum(w) = 1
           0 <= w_i <= max_weight
           optional sector caps
```

- `src/stock_analysis/optimization/recommendations.py` emits `BUY`, `SELL`, `HOLD`, `EXCLUDE`,
  target/current/trade weights, and commission estimates.
- `src/stock_analysis/portfolio/holdings.py` can load current holdings from `current_weight`,
  `weight`, or `market_value`, but it returns only normalized weights. It does not preserve dollar
  market value, cash balance, or contribution amount.
- `src/stock_analysis/ml/autoresearch_eval.py` compares candidate net portfolio returns to SPY and
  can log results to MLflow. It does not include deposit-aware metrics.
- `docs/backtest-results-vs-spy.md` documents the current 2% commission result and correctly
  interprets commission as:

```text
commission = abs(target_weight - previous_weight) * portfolio_value * 0.02
```

## Findings

- Monthly deposits should not be mixed directly into model skill metrics. Deposits change ending
  wealth and money-weighted return, but they are external cash flows, not alpha.
- The current weight-only backtest can still report forecast skill via time-weighted net returns.
  A contribution-aware backtest must add a separate value ledger.
- Deposits can materially reduce commission drag because buys can be funded with new cash before
  selling existing positions.
- The current optimizer already accepts `w_prev`, so deposit-aware behavior can be introduced by
  shrinking pre-existing asset weights against post-deposit portfolio value before solving.
- Live one-shot deposit recommendations need a dollar denominator. If the holdings file contains
  only weights, a fixed dollar deposit cannot be converted to post-deposit weights without an
  external `portfolio_value`.
- A no-trade band is not naturally convex as a hard optimizer constraint without integer variables.
  For v1 it should be implemented as an execution/recommendation rule, while a convex turnover
  budget can be implemented directly in CVXPY.

## Scope

### In scope

- Add contribution/cashflow configuration for backtests and one-shot recommendations.
- Model fixed dollar deposits every 30 calendar days, mapped to the next available trading
  rebalance date.
- Use post-deposit portfolio value as the base for trade weights and commission estimates.
- Add cash-aware rebalance planning:
  - apply contribution before optimization;
  - compute `w_prev_post_deposit`;
  - optimize target weights against post-deposit current weights;
  - generate buy/sell/hold rows and cash usage diagnostics.
- Add a no-trade band as a post-optimization execution rule for small trades.
- Add an optional convex turnover budget constraint:

```text
||w - w_prev||_1 <= max_trade_abs_weight
```

- Extend backtest output with portfolio value, deposits, commissions, cash, TWR, and MWR metrics.
- Compare the strategy to SPY with the same initial capital and same deposit schedule.
- Extend autoresearch output and MLflow logging with contribution-aware metrics.
- Update docs with the distinction between prediction time, trade time, deposits, TWR, MWR, and
  SPY comparison.

### Out of scope

- Broker execution or real orders.
- Tax-lot optimization.
- Share-count rounding.
- Margin, shorts, options, or leverage.
- Salary/payroll integration. The system will accept a fixed contribution amount or a precomputed
  monthly amount.
- Intraday deposit timing. Deposits are applied at rebalance time using end-of-day research
  assumptions.
- Integer programming for exact no-trade-band constraints.
- Paid market data or paid APIs.

## File Plan

| Path | Action | Details |
| --- | --- | --- |
| `src/stock_analysis/config.py` | modify | Add `ContributionConfig` and `ExecutionConfig`, wire them into `PortfolioConfig`; add fields for `initial_portfolio_value`, `monthly_deposit_amount`, `deposit_frequency_days`, `deposit_start_date`, `cash_balance`, `no_trade_band`, `max_trade_abs_weight`, and `rebalance_on_deposit_day`. |
| `configs/portfolio.yaml` | modify | Add default contribution/execution sections with disabled deposits by default: `monthly_deposit_amount: 0.0`, `deposit_frequency_days: 30`, `no_trade_band: 0.0`, `max_trade_abs_weight: null`. |
| `configs/current_holdings.example.csv` | modify | Add comments or example rows showing `market_value`-based holdings for live deposit workflows. |
| `src/stock_analysis/portfolio/holdings.py` | modify | Introduce a `PortfolioState` dataclass that preserves `weights`, `market_values`, `cash_balance`, and `portfolio_value`; keep `load_current_weights` as a backward-compatible wrapper. |
| `src/stock_analysis/portfolio/rebalance.py` | create | Implement deposit-aware rebalance context and trade planning: contribution application, post-deposit weights, buy/sell notional, commission dollars, cash after trades, and no-trade-band filtering. |
| `src/stock_analysis/optimization/engine.py` | modify | Add optional `max_trade_abs_weight` constraint to `optimize_long_only`; keep the existing objective and solver behavior. |
| `src/stock_analysis/optimization/recommendations.py` | modify | Accept optional rebalance plan output and emit dollar fields: `current_market_value`, `target_market_value`, `trade_notional`, `commission_amount`, `deposit_used_amount`, `cash_after_trade_amount`, and post-deposit base metadata. |
| `src/stock_analysis/domain/schemas.py` | modify | Extend `portfolio_recommendations` required/optional validation to include contribution-aware columns without breaking existing runs. |
| `src/stock_analysis/pipeline/one_shot.py` | modify | Load `PortfolioState`, apply configured one-shot contribution before optimization, pass post-deposit weights as `w_prev`, and pass rebalance plan details to recommendations. |
| `src/stock_analysis/backtest/runner.py` | modify | Add optional contribution-aware simulation while preserving the current zero-deposit behavior; output per-period value/cashflow columns duplicated on each ticker row for compatibility. |
| `src/stock_analysis/backtest/cashflows.py` | create | Implement contribution schedule generation, period ledger helpers, TWR calculation, and pure-Python XIRR/MWR calculation without adding dependencies. |
| `src/stock_analysis/ml/evaluation.py` | modify | Add helper metrics for cash-flow simulations: ending value, total deposits, total commissions, commission/deposit ratio, MWR, TWR, SPY ending value, active ending value, and active TWR. |
| `src/stock_analysis/ml/autoresearch_eval.py` | modify | Add contribution fields to `AutoresearchEvalConfig`, include them in `BacktestConfig`, append contribution-aware metrics to JSON/TSV result payloads, and keep existing SPY gates based on TWR unless explicitly changed. |
| `src/stock_analysis/ml/mlflow_tracking.py` | modify | Log contribution config and cashflow metrics for autoresearch and one-shot runs. |
| `src/stock_analysis/cli.py` | modify | Add CLI flags to `autoresearch-eval` and `tune-turnover`: `--initial-portfolio-value`, `--monthly-deposit-amount`, `--deposit-frequency-days`, `--no-trade-band`, and `--max-trade-abs-weight`. |
| `scripts/autoresearch_eval.py` | modify | Mirror the new CLI flags for non-Typer execution. |
| `docs/forecasting-optimization-methodology.md` | modify | Document deposit timing, post-deposit weights, decision variables, turnover budget constraint, no-trade band, and TWR/MWR interpretation. |
| `docs/backtest-results-vs-spy.md` | modify | Add a contribution-aware result section after implementation/rerun, including same-deposit SPY comparison. |
| `runbooks/full-execution.md` | modify | Add commands and required holdings/deposit inputs for one-shot and contribution-aware backtests. |
| `tests/unit/test_holdings.py` | modify | Cover `PortfolioState` loading from `market_value`, weights-only backward compatibility, cash balance, and missing portfolio value validation. |
| `tests/unit/test_optimizer.py` | modify | Cover `max_trade_abs_weight` feasibility and failure behavior. |
| `tests/unit/test_recommendations.py` | modify | Cover dollar trade/commission/deposit columns and no-trade-band `HOLD` behavior. |
| `tests/unit/test_backtest_runner.py` | modify | Cover deposit schedule, first deposit, post-deposit weights, commission dollars, and zero-deposit backward compatibility. |
| `tests/unit/test_autoresearch_eval.py` | modify | Cover result/TSV propagation of contribution-aware metrics. |
| `tests/integration/test_autoresearch_harness.py` | modify | Run the synthetic candidate with a nonzero monthly deposit and assert cashflow metrics are present. |
| `tests/integration/test_one_shot_pipeline.py` | modify | Assert one-shot recommendations reconcile target weights, trade notionals, commission dollars, and deposit metadata. |

## Data and Contract Changes

- New config section:

```yaml
contributions:
  initial_portfolio_value: 1000.0
  monthly_deposit_amount: 0.0
  deposit_frequency_days: 30
  deposit_start_date: null
  rebalance_on_deposit_day: true

execution:
  cash_balance: 0.0
  no_trade_band: 0.0
  max_trade_abs_weight: null
```

- Live one-shot fixed-dollar deposits require one of:
  - holdings with `market_value`, so `portfolio_value = sum(market_value) + cash_balance`;
  - an explicit `initial_portfolio_value`/`portfolio_value` override in config.
- Existing weight-only holdings remain supported when `monthly_deposit_amount = 0.0`.
- Backtest default behavior remains unchanged when monthly deposits are zero.
- New period-level backtest columns:

```text
portfolio_value_start
external_contribution
portfolio_value_after_contribution
cash_before_rebalance
buy_notional
sell_notional
commission_paid
cash_after_rebalance
portfolio_value_end
period_twr_return
cumulative_twr_return
strategy_ending_value
spy_ending_value
```

- New recommendation columns:

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

- Commission semantics:

```text
buy_commission_i = max(trade_notional_i, 0) * commission_rate
sell_commission_i = abs(min(trade_notional_i, 0)) * commission_rate
```

- Example:

```text
pre-deposit portfolio value = 1000
deposit = 100
post-deposit value = 1100
BUY trade = 50% of post-deposit value = 550
commission = 550 * 0.02 = 11
```

## Scientific Methodology

### Timing

For rebalance date `t`:

1. Train only on rows where feature date `< t - embargo_days`.
2. Predict cross-sectional forecast scores using features at `t`.
3. Apply any scheduled external deposit at `t`.
4. Convert existing holdings into post-deposit weights:

```text
w_prev_post_i = current_market_value_i / (portfolio_value_before_deposit + deposit_t)
```

5. Optimize target weights with `w_prev_post` as prior weights.
6. Generate trades from target weights and current dollar holdings.
7. Charge commission on absolute buy and sell notional.
8. Hold the executed portfolio for the configured horizon.
9. Record realized return, ending value, TWR, MWR cash flows, and SPY comparison.

### Optimization Function

Decision variable:

```text
w_i = target post-rebalance weight for asset i
```

Parameters:

```text
mu_i          forecast score for asset i
Sigma        covariance matrix
gamma        risk aversion
lambda_turnover
commission_rate
w_prev_i     post-deposit current asset weight
u_max        optional max absolute trade weight budget
```

Objective:

```text
maximize_w  mu^T w
          - gamma * w^T Sigma w
          - lambda_turnover * ||w - w_prev||_1
          - commission_rate * ||w - w_prev||_1
```

Constraints:

```text
sum_i w_i = 1
0 <= w_i <= max_weight
sum_{i in sector s} w_i <= sector_max_weight, if configured
||w - w_prev||_1 <= max_trade_abs_weight, if configured
```

Execution filter:

```text
if abs(target_weight_i - current_weight_i) < no_trade_band:
    action_i = HOLD
```

The no-trade band is intentionally outside the convex optimizer in v1. It avoids mixed-integer
optimization and keeps the solver reliable.

### Performance Metrics

- TWR answers: "Did the strategy produce good returns independent of deposit timing?"
- MWR/XIRR answers: "What did this investor earn after adding money over time?"
- Ending value answers: "How many dollars would I have under this deposit schedule?"
- SPY comparison uses the same initial value and same deposits on the same dates.
- Candidate acceptance remains based on TWR/SPY-relative metrics by default because deposits are
  external cash flows. MWR and ending value are reported, not used as the primary model-skill gate.

## Implementation Steps

1. Add config models for contribution and execution behavior in `src/stock_analysis/config.py`.
2. Extend `PortfolioState` handling in `src/stock_analysis/portfolio/holdings.py` while keeping
   `load_current_weights` backward-compatible for existing code.
3. Create `src/stock_analysis/backtest/cashflows.py` with:
   - `scheduled_contribution_for_date`;
   - `build_contribution_schedule`;
   - `time_weighted_return`;
   - `money_weighted_return`;
   - period ledger helpers.
4. Create `src/stock_analysis/portfolio/rebalance.py` with:
   - `build_rebalance_context`;
   - `post_deposit_current_weights`;
   - `plan_rebalance_trades`;
   - no-trade-band handling.
5. Add `max_trade_abs_weight` to `OptimizerConfig` and `optimize_long_only`.
6. Update `run_walk_forward_backtest` to:
   - initialize portfolio value;
   - apply deposits every 30 natural days;
   - convert current values to post-deposit weights;
   - call the optimizer;
   - compute trade dollars and commissions;
   - compute realized period TWR and ending value.
7. Update SPY benchmarking so the same deposit ledger is applied to SPY.
8. Extend `evaluate_candidate`, `result_to_tsv_row`, and `RESULT_COLUMNS` with contribution-aware
   metrics.
9. Add CLI flags to Typer and script entry points.
10. Update one-shot pipeline to generate deposit-aware live recommendations when a contribution is
    configured.
11. Update MLflow logging for contribution config and metrics.
12. Update docs/runbooks with execution examples and interpretation.
13. Rerun an E8 contribution-aware comparison and write results to `docs/backtest-results-vs-spy.md`.

## Tests

- Unit: `tests/unit/test_holdings.py` validates `PortfolioState`, market-value normalization,
  cash balance, and backward compatibility.
- Unit: `tests/unit/test_backtest_runner.py` validates scheduled deposits, post-deposit weights,
  commission dollars, ending value, and zero-deposit unchanged behavior.
- Unit: `tests/unit/test_optimizer.py` validates `max_trade_abs_weight` constraints.
- Unit: `tests/unit/test_recommendations.py` validates no-trade-band and dollar recommendation
  fields.
- Unit: `tests/unit/test_autoresearch_eval.py` validates JSON/TSV contribution metric propagation.
- Integration: `tests/integration/test_autoresearch_harness.py` validates an end-to-end synthetic
  contribution-aware backtest.
- Integration: `tests/integration/test_one_shot_pipeline.py` validates live recommendation outputs
  with contribution metadata.
- Regression: current zero-deposit autoresearch and one-shot tests must continue to pass without
  changing expected public behavior.

## Validation

- Format: `uv run ruff format --check src tests scripts`
- Lint: `uv run ruff check src tests scripts`
- Types: `uv run mypy src`
- Tests: `uv run pytest`
- Optional MLflow smoke test after implementation:

```bash
uv run --extra mlflow stock-analysis autoresearch-eval \
  --candidate e8_baseline \
  --input-run-root data/runs/phase2-source-20260424 \
  --max-assets 100 \
  --max-rebalances 48 \
  --commission-rate 0.02 \
  --lambda-turnover 5.0 \
  --initial-portfolio-value 1000 \
  --monthly-deposit-amount 100 \
  --deposit-frequency-days 30 \
  --mlflow \
  --json-output docs/experiments/e8-contribution-aware-20260424.json
```

## Risks and Mitigations

- Risk: Deposits distort performance interpretation -> Mitigation: report TWR as strategy skill and
  MWR/ending value as investor outcome.
- Risk: Weight-only live holdings cannot support fixed dollar deposits -> Mitigation: require
  `market_value` holdings or explicit portfolio value when deposits are enabled.
- Risk: No-trade band can make final weights sum below 100% -> Mitigation: report residual cash and
  keep target weights distinct from executed weights.
- Risk: 2% commission can make most active strategies unviable -> Mitigation: include no-trade band,
  turnover budget, and sweepable `lambda_turnover`.
- Risk: Adding contribution logic breaks existing research outputs -> Mitigation: defaults keep
  deposits disabled and preserve existing columns/semantics.
- Risk: XIRR can fail with invalid cash-flow signs -> Mitigation: return `None` with a clear reason
  when cash flows do not include at least one outflow and one inflow.

## Open Questions

- None

## Acceptance Criteria

- A backtest can be run with `initial_portfolio_value`, `monthly_deposit_amount`, and
  `deposit_frequency_days`.
- The strategy and SPY are compared using the same deposit schedule.
- Outputs include TWR, MWR/XIRR, ending value, total deposits, total commissions, commission as a
  percent of deposits, and active ending value versus SPY.
- One-shot recommendations can show how much new cash to allocate and how much commission each
  trade costs.
- Existing zero-deposit behavior remains backward-compatible.
- MLflow logs contribution parameters and cashflow metrics when enabled.

## Definition of Done

- Code implemented for contribution-aware backtests and one-shot recommendations.
- Tests added or updated for holdings, rebalance planning, backtest, optimizer, recommendations,
  autoresearch, and integration flows.
- `docs/forecasting-optimization-methodology.md`, `docs/backtest-results-vs-spy.md`, and
  `runbooks/full-execution.md` updated.
- Ruff format, Ruff lint, mypy, and pytest pass.
- Beads issue is closed and changes are committed and pushed.
