# Trade-Aware Optimizer And Rebalance Recommendations

## Goal

- Convert the optimizer output from target allocations only into actionable rebalance
  recommendations: what to buy, what to sell, and by what portfolio percentage.
- Include a 2% commission assumption every time a transaction changes a position.
- Improve the optimization model so turnover, commission drag, sector concentration, and practical
  trade thresholds are modeled explicitly instead of only documented as future work.
- Log one-shot portfolio run parameters, metrics, and gold artifacts to MLflow when enabled.

## Request Snapshot

- User request: "plan for that, still assumes how to allocate money but now tell me when and what to buy (at least in percentages), assume a 2% of comission every time you do a transaction, improve the optimization model as well."
- Owner or issue: `stock-analysis-2xy`
- Plan file: `plans/20260424-1407-trade-aware-optimizer.md`

## Current State

- `src/stock_analysis/optimization/engine.py` implements `optimize_long_only`.
- The objective is currently:

```text
mu^T w - risk_aversion * w^T Sigma w - lambda_turnover * ||w - w_prev||_1
```

- Constraints are currently:

```text
sum(w) = 1
w_i >= 0
w_i <= max_weight
```

- `src/stock_analysis/optimization/recommendations.py` maps target weights into `BUY` or
  `EXCLUDE`. It does not know current holdings, so it cannot emit true `SELL` or `HOLD`.
- `src/stock_analysis/backtest/runner.py` passes `w_prev` into the optimizer and charges a flat
  period cost:

```text
period_cost = cost_bps / 10_000 * turnover
turnover = 0.5 * sum(abs(w - w_prev))
```

- The current default research transaction cost is 5 bps. The new request requires 2% commission per
  transaction. In portfolio-weight terms this should be modeled as:

```text
commission_rate = 0.02
trade_cost = commission_rate * sum(abs(w - w_prev))
```

- `configs/portfolio.yaml` currently sets:

```yaml
optimizer:
  max_weight: 0.30
  risk_aversion: 10.0
  min_trade_weight: 0.005
  lambda_turnover: 0.001
```

- `src/stock_analysis/pipeline/one_shot.py` calls `optimize_long_only(...)` without previous
  holdings and calls `build_recommendations(...)` without current portfolio weights.
- `src/stock_analysis/domain/schemas.py` requires `portfolio_recommendations` columns but does not
  require trade deltas, commission cost, or prior weights.
- `docs/forecasting-optimization-methodology.md` explicitly states that true sell orders require
  current holdings and that transaction costs are not implemented in the one-shot recommendation
  model.

## Findings

- Current output answers "how should capital be allocated after rebalance" via `target_weight`.
- Current output does not answer "what should I buy/sell now" because there is no current portfolio
  input.
- The optimizer already supports previous weights through `w_prev`, which is the correct integration
  point for trade-aware optimization.
- Existing backtesting already records `previous_weight`, `target_weight`, `turnover`, and
  `transaction_cost`, so extending the production one-shot pipeline to emit trade deltas is
  compatible with the current research design.
- A 2% commission is large compared with normal equity commissions, but it can be implemented as a
  configurable rate because the user explicitly requested it.

## Scope

### In scope

- Add a current holdings input contract based on portfolio weights.
- Add config fields for current holdings path, commission rate, min rebalance trade size, optional
  cash handling, and sector caps.
- Update one-shot pipeline to read current holdings, align them to optimizer tickers, and pass them
  as `w_prev`.
- Improve optimizer objective to include explicit commission cost on dollar turnover:

```text
maximize mu^T w
       - gamma * w^T Sigma w
       - lambda_turnover * ||w - w_prev||_1
       - commission_rate * ||w - w_prev||_1
```

- Add optional sector cap constraints:

```text
sum_{i in sector s} w_i <= sector_max_weight
```

- Generate trade recommendations with:

```text
current_weight
target_weight
trade_weight = target_weight - current_weight
trade_abs_weight = abs(trade_weight)
estimated_commission_weight = commission_rate * trade_abs_weight
net_target_weight_after_commission
action in {BUY, SELL, HOLD, EXCLUDE}
```

- Update Tableau mart and workbook contract so dashboards can show trade deltas and commission drag.
- Update backtest runner to use the same commission rate semantics as the optimizer and reports.
- Add tests for buy/sell/hold classification and commission-aware optimization behavior.

### Out of scope

- Share-count order generation.
- Broker integration or order execution.
- Tax-lot optimization.
- Intraday execution timing.
- Market impact model beyond the configured 2% commission assumption.
- Survivorship-bias fix and point-in-time universe membership.
- Recalibrating forecast scores into true expected percentage returns.

## File Plan

| Path | Action | Details |
| --- | --- | --- |
| `src/stock_analysis/config.py` | modify | Add `PortfolioStateConfig` or optimizer fields for `current_holdings_path`, `commission_rate`, `min_rebalance_trade_weight`, `sector_max_weight`, and `cash_ticker` if needed. |
| `configs/portfolio.yaml` | modify | Set `commission_rate: 0.02`, define default `current_holdings_path`, and add conservative sector cap defaults. |
| `configs/tableau.example.yaml` | modify | Include trade-aware Tableau fields only if config examples duplicate optimizer settings. |
| `src/stock_analysis/portfolio/holdings.py` | create | Load current holdings CSV/Parquet, validate columns, normalize to weights, and align to optimizer universe. |
| `src/stock_analysis/optimization/engine.py` | modify | Add commission-aware objective term and optional sector cap constraints while preserving existing long-only behavior. |
| `src/stock_analysis/optimization/recommendations.py` | modify | Emit current weight, target weight, delta trade weight, action, commission estimate, and rebalance reason codes. |
| `src/stock_analysis/domain/schemas.py` | modify | Extend `portfolio_recommendations` schema with new trade and commission columns. |
| `src/stock_analysis/pipeline/one_shot.py` | modify | Read current holdings, pass `w_prev` to `optimize_long_only`, and pass current weights into `build_recommendations`. |
| `src/stock_analysis/ml/mlflow_tracking.py` | modify | Add one-shot portfolio MLflow logging for optimizer params, risk metrics, action counts, trade cost metrics, and artifacts. |
| `src/stock_analysis/backtest/runner.py` | modify | Replace `cost_bps` primary semantics with `commission_rate`, keeping `cost_bps` compatibility if needed. |
| `src/stock_analysis/tableau/dashboard_mart.py` | modify | Add trade fields to the single Tableau mart and selected/action logic. |
| `src/stock_analysis/tableau/workbook.py` | modify | Add generated workbook fields for `trade_weight`, `trade_weight_label`, `estimated_commission_weight`, and true action. |
| `docs/forecasting-optimization-methodology.md` | modify | Document the new objective, decision variables, constraints, and trade classification rules. |
| `docs/backtest-results-vs-spy.md` | modify | Note whether the 2% commission model is active in the rerun backtest results. |
| `runbooks/full-execution.md` | modify | Add instructions for creating current holdings input and reading BUY/SELL/HOLD outputs. |
| `tests/unit/test_optimizer.py` | modify | Cover commission-aware objective and sector cap feasibility. |
| `tests/unit/test_recommendations.py` | modify | Cover BUY/SELL/HOLD/EXCLUDE classification from current vs target weights. |
| `tests/unit/test_holdings.py` | create | Validate holdings loading, normalization, duplicate ticker handling, and missing current holdings behavior. |
| `tests/integration/test_one_shot_pipeline.py` | modify | Assert new recommendation columns exist and weights/deltas reconcile. |
| `tests/unit/test_backtest_runner.py` | modify | Assert 2% commission cost is applied to turnover consistently. |
| `tests/unit/test_tableau_workbook.py` | modify | Assert generated workbook includes trade-aware fields. |

## Data and Contract Changes

- Add optional current holdings input:

```csv
ticker,current_weight
AAPL,0.08
MSFT,0.07
SPY,0.00
```

- Alternative accepted columns:

```text
ticker,weight
ticker,current_weight
ticker,market_value
```

- If `market_value` is provided, weights are normalized by total positive market value.
- If no holdings file is provided, behavior remains first-allocation mode:

```text
current_weight_i = 0 for all assets
action = BUY for selected target weights
```

- New `portfolio_recommendations` columns:

```text
current_weight
target_weight
trade_weight
trade_abs_weight
trade_weight_label
estimated_commission_weight
net_trade_weight_after_commission
action
reason_code
rebalance_required
```

- Action semantics:

```text
BUY  if trade_weight >= min_rebalance_trade_weight
SELL if trade_weight <= -min_rebalance_trade_weight
HOLD if abs(trade_weight) < min_rebalance_trade_weight and target_weight > 0
EXCLUDE if target_weight == 0 and current_weight == 0
```

- Commission assumption:

```text
commission_rate = 0.02
estimated_commission_weight_i = commission_rate * abs(trade_weight_i)
```

- Portfolio-level estimated commission:

```text
sum_i estimated_commission_weight_i
```

## Implementation Steps

1. Add config model fields:
   - `optimizer.commission_rate: float = 0.02`
   - `optimizer.min_rebalance_trade_weight: float = 0.005`
   - `optimizer.sector_max_weight: float | None = 0.35`
   - `portfolio.current_holdings_path: Path | None = None` or equivalent root-level config.
2. Create `src/stock_analysis/portfolio/holdings.py`.
   - Implement `load_current_weights(path: Path | None) -> pd.Series`.
   - Implement accepted schema handling for `current_weight`, `weight`, or `market_value`.
   - Validate no negative weights unless explicit short support is introduced later.
3. Update `optimize_long_only`.
   - Keep existing signature compatible.
   - Add `sector_map: pd.Series | None = None` or derive sector constraints from optimizer input.
   - Add objective penalty for `commission_rate * cp.norm1(weights - previous_weights)`.
   - Add sector cap constraints when configured.
   - Keep solver fallback behavior.
4. Update recommendation generation.
   - Accept `current_weights`.
   - Compute trade deltas and commission estimates.
   - Classify `BUY`, `SELL`, `HOLD`, `EXCLUDE`.
   - Preserve current target allocation columns for backwards compatibility.
5. Wire one-shot pipeline.
   - Load current weights before optimization.
   - Pass `w_prev=current_weights`.
   - Pass current weights into recommendations.
   - Include commission and holdings-path metadata in `run_metadata`.
6. Add MLflow logging.
   - Respect `mlflow.enabled`, `mlflow.tracking_uri`, and `mlflow.experiment_name`.
   - Log config params, run metadata, risk metrics, action counts, trade notional, estimated
     commission, and gold artifacts.
7. Update backtest runner.
   - Use `commission_rate` from `OptimizerConfig` or `BacktestConfig`.
   - Keep `cost_bps` deprecated compatibility if existing tests rely on it.
   - Make cost formula consistent with production recommendation estimate.
8. Update Tableau output.
   - Add trade fields to `dashboard_mart`.
   - Update generated workbook field definitions and holdings table/tooltips.
9. Update docs and runbooks.
   - Explain target allocation vs trade deltas.
   - Show a sample current holdings file.
   - Explain 2% commission math.
   - Explain where MLflow runs and artifacts are written.
10. Run tests and validation.

## Tests

- Unit: `tests/unit/test_holdings.py`
  - Loads `current_weight` CSV and returns a normalized `pd.Series`.
  - Loads `market_value` CSV and normalizes weights.
  - Rejects negative weights.
  - Rejects duplicate tickers.
- Unit: `tests/unit/test_optimizer.py`
  - Commission-aware optimizer changes less than no-commission optimizer when previous weights exist.
  - Sector cap constraint limits aggregate sector weight.
  - Existing long-only and max-weight tests still pass.
- Unit: `tests/unit/test_recommendations.py`
  - BUY when target exceeds current by threshold.
  - SELL when target is below current by threshold.
  - HOLD when delta is below threshold.
  - EXCLUDE when both target and current are zero.
  - Commission estimate equals `0.02 * abs(trade_weight)`.
- Unit: `tests/unit/test_backtest_runner.py`
  - Period cost uses `commission_rate * sum(abs(w - w_prev))`.
  - `portfolio_net_return = gross_return - transaction_cost`.
- Integration: `tests/integration/test_one_shot_pipeline.py`
  - Pipeline runs without holdings file and emits first-allocation BUY recommendations.
  - Pipeline runs with a holdings fixture and emits at least one BUY/SELL/HOLD action.
  - New recommendation columns exist in Parquet and CSV mirrors.
- Unit: `tests/unit/test_tableau_workbook.py`
  - Generated `.twb` includes trade-aware fields.

## Validation

- Format: `uv run ruff format --check src tests`
- Lint: `uv run ruff check src tests`
- Types: `uv run mypy src`
- Tests: `uv run pytest`
- Optional pipeline smoke:

```bash
uv run stock-analysis run-one-shot --config configs/portfolio.yaml
```

## Risks and Mitigations

- 2% commission may cause the optimizer to avoid almost all trades.
  - Mitigation: keep first-allocation mode clear; expose `commission_rate` as config; show estimated commission in outputs.
- Forecast scores are not calibrated returns, so comparing 2% commission directly to `mu` scale can distort optimization.
  - Mitigation: document `expected_return_is_calibrated = false`; consider score scaling or calibrated returns as a follow-up.
- Sector caps can make optimization infeasible with high max-weight constraints and small eligible universes.
  - Mitigation: validate minimum feasible assets and provide clear `OptimizationError` messages.
- Current holdings may include tickers outside the optimizer universe.
  - Mitigation: carry them into recommendation output as `SELL` if unsupported, or report them in a separate validation warning. Recommended v1: include out-of-universe holdings as forced SELL candidates with target weight 0.
- Backtest result comparability changes when moving from 5 bps to 2% commission.
  - Mitigation: rerun and clearly label new backtest results.

## Open Questions

- None

## Acceptance Criteria

- A current holdings file can be supplied and the pipeline emits `BUY`, `SELL`, `HOLD`, and `EXCLUDE` actions based on percentage deltas.
- Recommendations include target weight, current weight, trade weight, absolute trade weight, and estimated commission weight.
- The optimizer objective includes explicit commission drag based on turnover from current weights.
- Backtest transaction cost math uses the same commission-rate convention as production recommendations.
- Tableau mart includes the new trade fields.
- MLflow logs one-shot portfolio params, metrics, and artifacts when enabled.
- Documentation explains allocation vs trade instructions and the 2% commission assumption.

## Definition of Done

- Code implemented for holdings ingestion, commission-aware optimization, trade recommendations,
  backtest cost alignment, Tableau mart updates, MLflow logging, and docs.
- Unit and integration tests added or updated.
- `uv run ruff format --check src tests` passes.
- `uv run ruff check src tests` passes.
- `uv run mypy src` passes.
- `uv run pytest` passes.
- Plan updated if implementation scope changes.
