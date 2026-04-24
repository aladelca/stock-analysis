# ML Upgrade Plan — From Heuristic Score to Evaluated Forecasting Model

Detailed plan to replace the current `mu_i = momentum_252d - 0.25 * volatility_63d` heuristic score with a real, cross-validated, backtested forecasting model. Explicitly covers data-leakage prevention, walk-forward CV, a SPY benchmark, and a multi-experiment catalog (linear, gradient boosting, deep learning, Prophet).

Companion to `docs/forecasting-optimization-methodology.md`, which documents the *current* MVP methodology and lists the limitations this plan addresses.

## Goals and non-goals

### In scope

- Build a proper point-in-time (PIT) feature panel.
- Define supervised-learning targets (forward returns) without look-ahead bias.
- Run walk-forward cross-validation with embargo gaps.
- Train and compare multiple model families on the same evaluation harness.
- Backtest the resulting portfolios against SPY and equal-weight benchmarks with transaction costs.
- Replace the heuristic in `src/stock_analysis/forecasting/baseline.py` with the winning model when and only when it beats all baselines after costs, out-of-sample, by a statistically meaningful margin.

### Not in scope (explicit non-goals)

- Intraday data, real-time forecasting.
- Alternative data (news, filings, options, social).
- Live trading integration.
- Factor risk model / covariance shrinkage (tracked separately in the methodology doc).
- Multi-horizon model ensembling beyond a simple linear blend.
- Neural-network architectures more complex than a modest MLP or small TCN.

## Guiding principles (non-negotiable)

1. **PIT-correct always.** A feature at time `t` may only use data from `[t - k, t]`. A target at time `t` uses returns over `[t, t + h]`. Between any train and test fold, an embargo gap of **at least `h` trading days** separates the end of train labels from the start of test features so overlapping labels never leak.
2. **Time-ordered splits only.** Never random-shuffle the panel. All CV folds are walk-forward along the date axis.
3. **Fit preprocessing on train only.** Scalers, winsorizers, rank normalizers, imputers — fit on the train fold, apply to val/test. A single `sklearn.pipeline.Pipeline` should wrap both transforms and estimator so nothing leaks by accident.
4. **Benchmark everything.** Every experiment reports: predictive metrics (IC, rank IC), portfolio metrics (Sharpe, MaxDD, Calmar, turnover), and benchmark-relative metrics (alpha vs SPY, information ratio, tracking error). Every experiment is compared to the current heuristic, equal-weight S&P 500, and SPY buy-and-hold.
5. **Statistical rigor over storytelling.** Report confidence intervals on IC and Sharpe. Use Deflated Sharpe Ratio (López de Prado) or bootstrap CI to avoid declaring noise as alpha. Single-fold wins are not results.
6. **Reproducibility.** Every experiment pins seeds, config, code hash, data hash. Experiment results land under `data/experiments/<experiment_id>/` alongside the run directory.
7. **Survivorship bias is acknowledged.** The current Wikipedia S&P 500 snapshot is today's constituents only. We flag this limitation in every backtest report and gate the "replace the heuristic" decision on whether survivorship-free data is available.

## Locked-in decisions (2026-04-24 review)

These resolve the open questions at the bottom of this doc. Recorded here so later experiments do not re-litigate them.

1. **Survivorship bias.** Phase 1 and Phase 2 ship using current Wikipedia S&P 500 constituents with a loud caveat banner on every backtest report. In parallel, `stock-analysis-z4a` reconstructs point-in-time constituents from Wikipedia revision history (free, ~1-2 weeks imprecision at index-change boundaries). Phase 4 production switch is gated on either (a) PIT constituents in place or (b) applying a conservative 0.2 Sharpe haircut before comparing against target hurdles.
2. **Rebalance cadence: weekly (h = 5 trading days).** Primary target horizon for all Phase 2 and Phase 3 experiments is 5-day forward return. Monthly (21d) and quarterly (63d) are secondary sweeps. CV embargo defaults to 15 days for the primary h=5 setup and scales up for 21d/63d sweeps. Cost sensitivity sweeps become more important, not less, since weekly cadence amplifies transaction-cost drag.
3. **Optimizer lockdown, with one foundational addition.** All Phase 2 and Phase 3 experiments run against a **single frozen optimizer configuration**, so model-vs-model differences are attributable to `mu` quality only. Because weekly rebalance makes the cost-unaware optimizer unviable (~5-10% annual drag at 5bps), the frozen configuration includes an L1 turnover penalty from day one: `max_w  μᵀw − γ·wᵀΣw − λ_turnover·‖w − w_prev‖₁`. Implemented under `stock-analysis-suy`. Further optimizer variants (sector caps, Ledoit-Wolf shrinkage, etc.) become their own post-winner experiment track (O1, O2, …), not mixed into the ML experiments.
4. **Experiment tracking: filesystem.** One directory per experiment under `data/experiments/<experiment_id>/`, with an `experiments_index.parquet` roll-up. No MLflow until experiment count exceeds ~30.

## Current-state gaps this plan closes

Mapped to `docs/forecasting-optimization-methodology.md :: Recommended Next Methodology Improvements`:

| Improvement                                                       | Addressed by phase |
| ----------------------------------------------------------------- | ------------------ |
| 1. Rename `expected_return` → `forecast_score`, add calibrated    | Phase 1, Phase 4   |
| 2. Walk-forward backtesting with out-of-sample evaluation          | Phase 1, Phase 2  |
| 3. Covariance shrinkage (Ledoit-Wolf)                              | Deferred (separate issue) |
| 4. Transaction costs and turnover constraints                      | Phase 2           |
| 5. Current holdings input for real BUY/SELL/HOLD                   | Existing issue `stock-analysis-ed0` |
| 6. Sector and industry caps                                        | Deferred (separate issue) |
| 7. Benchmark-relative diagnostics vs SPY                           | Phase 1, all phases |
| 8. Robust forecast scaling, winsorization, standardization         | Phase 2           |
| 9. Confidence intervals / forecast uncertainty                     | Phase 3           |
| 10. Model versioning on every output                               | Phase 1           |

---

## Phase 1 — Evaluation infrastructure (must land before any model)

**Objective:** Build the harness that every experiment plugs into. No modeling until this exists, otherwise experiments will be incomparable.

### 1A. Point-in-time feature panel

Current silver `asset_daily_features.parquet` has **one row per ticker** (the latest). ML needs **one row per (ticker, date)**.

- New silver table: `asset_daily_features_panel.parquet` with columns `[ticker, date, <features>]`, indexed on `(ticker, date)`.
- For each historical date `t`, compute every feature as if `t` were "today":
  - `momentum_{63,126,252}d` = `P_t / P_{t-w} - 1`
  - `volatility_{21,63,126}d` = annualized std of returns in `[t-w+1, t]`
  - `max_drawdown_{63,126,252}d` = rolling
  - `ma_ratio_{50,200}d` = `P_t / mean(P_{t-w+1:t})`
  - `return_1d, return_5d, return_21d` (lagged features are fine as inputs; see 1B on target definition)
  - `volume_21d_zscore`, `dollar_volume_21d`
  - Cross-sectional ranks computed per-date: `momentum_252d_rank`, etc.
  - Market-relative: `return_21d - return_21d_spy`
- Use pandas `rolling` with `min_periods` set, and truncate to dates where all required lookbacks are satisfied. No forward-fill.
- Validate no look-ahead by asserting `panel.groupby("ticker")["feature"].shift(0)` never uses `t+1`.

**Deliverables:** new module `src/stock_analysis/features/panel.py`, new silver output, schema validation.

### 1B. Target (label) generation

Three canonical targets, all computed per `(ticker, t)`:

```text
fwd_return_h_i_t = P_{i, t+h} / P_{i, t} - 1
```

Horizons: `h ∈ {5, 21, 63}` trading days (weekly, monthly, quarterly). **Default model horizon is 5d** (matches the locked-in weekly rebalance cadence). 21d and 63d are secondary sweeps.

Also:

- `fwd_excess_return_h = fwd_return_h - fwd_spy_return_h` (alpha target)
- `fwd_rank_h` = cross-sectional rank of `fwd_return_h` within that date (rank-target variant)
- `fwd_is_top_tercile_h` = binary label, 1 if in top tercile of that date's returns (classification variant)

**Leakage gate:** Targets are computed with `shift(-h)` on the `adj_close` series per ticker and merged to the panel at time `t`. No target appears in any feature column. Any row where `t + h > latest_available_date` is set to NaN and excluded.

**Deliverables:** `src/stock_analysis/ml/labels.py`, new gold table `labels_panel.parquet`.

### 1C. SPY benchmark ingestion

- Add SPY (and optionally `^GSPC`) to the price ingestion list.
- Build `silver/spy_daily.parquet` with `[date, adj_close, return_1d]`.
- Compute benchmark return series for any horizon `h`: `spy_return_h_t = P_{spy, t+h} / P_{spy, t} - 1`.
- Cache in `silver/benchmark_returns.parquet`.

**Deliverables:** new ingestion flow, silver table, benchmark utility module `src/stock_analysis/benchmarks/spy.py`.

### 1D. Walk-forward CV splitter

- Module `src/stock_analysis/ml/cv.py` exposing an iterator of `(train_idx, val_idx)` splits along the date axis.
- Parameters: `train_window_years`, `val_window_months`, `step_months`, `embargo_days`, `expanding` (bool).
- **Embargo rule:** `test_start >= train_end + embargo_days` where `embargo_days >= max(target_horizon_days) + safety_margin`. Default safety margin: 5 days.
- Support both rolling and expanding windows.
- Support **purged CV** option: drop train rows whose target horizon overlaps the test window.
- Yield pandas `DatetimeIndex` slices, not positional indices (works directly with the panel).

Default configuration for experiments (h=5 primary):

```yaml
cv:
  train_window_years: 3
  val_window_months: 6
  step_months: 3
  embargo_days: 15     # covers 5-day target horizon + safety margin
  expanding: false
```

For secondary horizon sweeps, embargo scales with `h`:

```yaml
# h = 21 sweep
embargo_days: 30
# h = 63 sweep
embargo_days: 75
```

The splitter asserts `embargo_days >= max_target_horizon + safety_margin` at configuration time.

**Deliverables:** `cv.py` plus unit tests that assert no date appears in both train and test for any fold, and that embargo is respected.

### 1E. Evaluation harness

One function — `evaluate(predictions, targets, prices, benchmark_returns, config)` — that takes date-indexed predictions and produces a uniform results dict:

**Predictive metrics (per fold, then aggregated):**

- Pearson IC = `corr(pred, target)`
- Rank IC (Spearman)
- IC t-statistic and 95% CI via bootstrap (1000 resamples over dates)
- Hit rate (sign accuracy)
- RMSE, MAE on returns
- ROC-AUC for classification variants

**Portfolio metrics (from walk-forward backtest in 1F):**

- Annualized return, annualized volatility, Sharpe (ann. return / ann. vol)
- Max drawdown, Calmar (ann. return / |MaxDD|)
- Hit rate (% positive months)
- Turnover (average `0.5 * sum |w_t - w_{t-1}|` per rebalance)
- Deflated Sharpe Ratio (accounts for number of trials)

**Benchmark-relative metrics (vs SPY and vs equal-weight S&P 500):**

- Alpha, beta (regress excess returns on benchmark excess returns)
- Information Ratio = mean(r_portfolio - r_benchmark) / std(r_portfolio - r_benchmark)
- Tracking error
- Active share (when a benchmark weight vector exists — for SPY we proxy with equal-weight)

All metrics reported **with confidence intervals** from block bootstrap (block size ≈ target horizon).

**Deliverables:** `src/stock_analysis/ml/evaluation.py`, unit tests on synthetic data with known IC.

### 1F. Walk-forward backtest runner

The backtest loop:

```text
for each rebalance date r in [start, end]:
  1. Train model on panel[t < r - embargo].
  2. Predict forecast_score for every eligible ticker at date r.
  3. Run the existing CVXPY optimizer with predicted mu and covariance estimated on
     returns[t < r]. Covariance estimator is held constant across experiments so only
     mu changes drive differences.
  4. Hold the resulting portfolio for h trading days.
  5. Compute realized portfolio return, turnover, and transaction cost.
  6. Record weights, predictions, realized returns to data/experiments/<id>/backtest.parquet.
```

Transaction-cost model (simple and explicit): `cost = 0.0005 * turnover` applied to the period return (5 bps one-way, configurable). Record both gross and net series.

Rebalance cadence: **weekly (`h=5`) for v1** (locked-in decision). Monthly and quarterly are secondary sweeps. Weekly cadence amplifies cost sensitivity, so every backtest reports both gross and net-of-cost series side by side.

The runner passes the previous rebalance's weights (`w_prev`) into the optimizer so the turnover penalty in the frozen optimizer configuration can take effect. For the first rebalance of a backtest window, `w_prev` is zero (equivalent to starting from cash).

**Deliverables:** `src/stock_analysis/backtest/runner.py`, config schema, unit tests on synthetic deterministic data.

### 1G. Experiment tracking

- One directory per experiment: `data/experiments/<experiment_id>/`.
- Contents: `config.yaml`, `predictions.parquet`, `backtest.parquet`, `metrics.json`, `code_hash.txt`, `seed.txt`, `feature_importance.parquet` (when applicable).
- Index file `data/experiments/experiments_index.parquet` with one row per run for dashboard consumption.
- Optional: integrate with MLflow later. Not required for v1 — filesystem + parquet is enough.

**Deliverables:** `src/stock_analysis/ml/tracking.py`, CLI command `stock-analysis run-experiment --config experiments/<name>.yaml`.

### Phase 1 exit criteria

- Panel features computed and schema-validated.
- SPY benchmark series available.
- CV splitter unit-tested with no date overlap and correct embargo.
- Evaluation harness produces a results dict on a dummy `predictor = lambda x: 0`.
- Walk-forward backtest runner produces a dated portfolio series for the current heuristic, matched against manual calculation.

---

## Phase 2 — Baselines and classic ML

**Objective:** Establish the bar. Nothing from Phase 3 is worth reporting until beaten against these.

### Experiments (each run through the Phase 1 harness)

| ID   | Model                                        | Target                 | Framing          | Why run it                                      |
| ---- | -------------------------------------------- | ---------------------- | ---------------- | ----------------------------------------------- |
| E0   | Current heuristic (`mu = mom_252 - 0.25*vol`) | 5d forward return      | Cross-sectional  | The bar                                         |
| E1   | SPY buy-and-hold                              | n/a                    | n/a              | Market benchmark                                |
| E2   | Equal-weight S&P 500 (weekly rebalance)      | n/a                    | n/a              | Naive diversified benchmark                     |
| E3   | Ridge regression on standardized features     | 5d forward return      | Cross-sectional  | Simplest learnable baseline                     |
| E4   | Ridge on **rank-normalized** features         | 5d forward rank        | Cross-sectional  | Robust to outliers; rank-target stabilizes      |
| E5   | LightGBM regression                           | 5d forward return      | Cross-sectional  | Non-linear interactions without DL              |
| E6   | LightGBM LambdaRank                           | 5d forward rank        | Listwise rank    | Directly optimizes the portfolio-relevant metric |
| E7   | LightGBM binary classifier                    | fwd_is_top_tercile_5d  | Cross-sectional  | Calibrated probability → rank assets by P(top)  |
| E8   | Linear blend of E3 + E5 (z-scored predictions) | 5d forward return     | Ensemble         | Diversification often beats best single model   |

All experiments run against the same frozen optimizer configuration (including the L1 turnover penalty), so model-vs-model differences are attributable to `mu` quality alone.

**Feature set for Phase 2 experiments** (identical across E3–E8 for fair comparison):

- Momentum: `mom_{21, 63, 126, 252}d` and their cross-sectional ranks
- Volatility: `vol_{21, 63, 126}d`
- Drawdown: `mdd_{63, 252}d`
- MA ratio: `ma_ratio_{50, 200}d`
- Reversal: `return_5d` (short-term reversal signal)
- Volume: `dollar_volume_21d`, `volume_21d_zscore`
- Market-relative: `return_21d - return_21d_spy`
- GICS sector one-hot or target-encoded (target-encoded fit on train only)
- Rolling 252d idiosyncratic volatility = `std(return_i - beta_i * return_spy)` where `beta_i` is computed on train window only

**Hyperparameter tuning (E5, E6, E7):** **Nested CV.** Outer loop is the walk-forward split (produces the OOS metrics we report). Inner loop is a smaller walk-forward split within the outer train window to pick hyperparameters. Nested CV is expensive but is the only honest way to report generalization.

Search space:

- LightGBM: `num_leaves ∈ {15, 31, 63}`, `learning_rate ∈ {0.01, 0.05}`, `n_estimators ∈ {100, 300, 1000}` with early stopping, `feature_fraction ∈ {0.7, 0.9}`, `bagging_fraction ∈ {0.7, 0.9}`.
- Ridge: `alpha ∈ {0.01, 0.1, 1, 10, 100}`.

### Phase 2 sweeps (on the winning Phase 2 model only)

- **Horizon sensitivity:** retrain at `h ∈ {5, 21, 63}` with matching rebalance cadences. Report IC and backtest for each. Weekly (h=5) is primary; sweep confirms it is the right choice.
- **Cost sensitivity:** rerun backtest at 0, 5, 10, 20 bps one-way. At weekly cadence this is the most important sweep — the model's edge must survive realistic costs, not only 0-bps fantasies.
- **Turnover-penalty sensitivity:** sweep `λ_turnover ∈ {0, 0.0005, 0.001, 0.002, 0.005}`. Zero is the cost-unaware baseline; the sweep quantifies how much of the weekly edge comes from the L1 penalty.
- **Max-weight sensitivity:** rerun optimizer at `w_max ∈ {0.02, 0.05, 0.10}`.
- **Rebalance cadence comparison:** weekly (default) vs monthly vs quarterly using the same model. Confirms the h=5 locked-in choice empirically. If monthly or quarterly is within 0.1 Sharpe net-of-cost, consider revisiting the cadence decision.

### Phase 2 exit criteria

- All E0–E8 results reported as a side-by-side markdown table under `docs/experiments/phase2-report.md`.
- At least one ML model beats the heuristic E0 on OOS IC and net-of-cost Sharpe **and** beats SPY on information ratio over the full backtest window, with the difference statistically non-trivial (95% bootstrap CI on Sharpe difference excludes zero).
- If no model clears that bar: stop, diagnose, do not advance to Phase 3.

---

## Phase 3 — Deep learning and time-series models

Only run if Phase 2 produces a clear winner over the heuristic. Otherwise the DL complexity is not justified by the data.

### Experiments

| ID   | Model                                                     | Framing                        | Notes                                             |
| ---- | --------------------------------------------------------- | ------------------------------ | ------------------------------------------------- |
| E9   | MLP (2-3 hidden layers, dropout, batch norm)              | Cross-sectional, 21d target    | Simplest DL; works on tabular panel               |
| E10  | TCN (temporal convolutional network) on per-ticker sequences | Per-ticker time-series        | Captures autoregressive structure                 |
| E11  | Small Transformer (4 heads, 2 layers) on per-ticker sequences | Per-ticker time-series     | Only if E10 shows signal                          |
| E12  | Prophet per-ticker (on log-adjusted close)                | Per-ticker point forecast      | Off-the-shelf trend + seasonality                 |
| E13  | Prophet per-sector composite index                        | Sector-level, then allocate within | Reduces per-ticker noise              |
| E14  | LSTM encoder + MLP head on panel (multi-asset)            | Panel, 21d target              | Deep model with cross-asset context               |

**DL-specific leakage risks to hard-guard:**

- Per-ticker sequence models: train on sequences ending at `t`, target from `[t, t+h]`. Never include target-period prices in the sequence input.
- Batch normalization: use running stats from train only at eval time. (PyTorch default handles this, but verify.)
- Early stopping: monitor on the walk-forward validation fold, **not on test**.

**Prophet-specific notes:**

- Prophet works per-ticker. For 500+ tickers, parallelize with `joblib`. Expect training time to dominate.
- Prophet outputs `yhat` which is a log-price forecast. Convert to expected return at horizon `h` by `exp(yhat_{t+h} - yhat_t) - 1`.
- Prophet's implicit assumption (smooth trends + seasonality) may fit some names and not others. Report per-sector IC to understand where it helps.
- **No stock-picking magic**: Prophet forecasts levels, not cross-sectional rankings. Expect it to underperform on portfolio metrics unless paired with a rank-aware post-processing step.

**DL evaluation:** same harness as Phase 2. DL predictions go through the same `evaluate()` function. No special treatment.

### Phase 3 exit criteria

- Report under `docs/experiments/phase3-report.md` comparing every DL experiment to the Phase 2 winner.
- Advance any DL model to Phase 4 only if it beats the Phase 2 winner on net-of-cost Sharpe by ≥0.2 absolute and IR vs SPY by ≥0.1.

---

## Phase 4 — Productionization

**Objective:** Replace the heuristic in the live pipeline with the winner from Phase 2 or Phase 3.

### Tasks

- **Model artifact storage.** Trained model pickle (or joblib) saved under `data/models/<model_id>/`. Every artifact has `model_version`, `feature_hash`, `training_window`, `code_hash`, `as_of_date`.
- **Inference path.** New module `src/stock_analysis/forecasting/ml_forecast.py` that loads the latest trained model, computes features for the current as-of-date only (cross-section), produces `forecast_score`, plugs into the existing `build_optimizer_inputs` signature as a drop-in for the heuristic.
- **Calibrated `expected_return`.** Either a separate calibration head (isotonic regression on OOS predictions) or a clearly documented caveat. For v1, keep the field as `forecast_score` in the mart and only rename to `expected_return` once calibration is proven.
- **Config switch.** `forecast.engine: heuristic | ml` in `configs/portfolio.yaml`. Defaults to `heuristic` until the ML engine has accumulated sufficient live track record (at least three months of recorded predictions for monitoring).
- **Monitoring.** Every live run records `predicted_score` alongside `realized_return` after the horizon passes, enabling live IC monitoring without any rebuild.
- **Dashboard additions.** `run_metadata` gains `model_version`, `model_family`, `expected_return_is_calibrated` (bool). The Tableau dashboard footer surfaces model version alongside `config_hash`.

### Production integration status — 2026-04-24

- `forecast.engine` now supports `heuristic` and `ml`; `configs/portfolio.yaml` is set to `ml`.
- The live ML path uses the Phase 2 E8 Ridge + LightGBM regression blend, retrained on available labels at run time and inferred only on the latest cross-section.
- The production optimizer uses the selected SPY-relative configuration: top 100 assets by latest `dollar_volume_21d` and `optimizer.max_weight: 0.30`.
- `run_metadata` records `forecast_engine`, `model_version`, `model_family`, and `expected_return_is_calibrated`.
- The ML score remains uncalibrated; `expected_return` is the optimizer-facing forecast score, not a calibrated return estimate.

### Phase 4 exit criteria

- `stock-analysis run-one-shot --config configs/portfolio.yaml --forecast-engine ml` runs end-to-end.
- The Tableau dashboard shows model metadata.
- A shadow run mode exists: run both heuristic and ML, store both predictions, but portfolio uses heuristic until operator flips the switch.

---

## Data leakage safeguards (consolidated checklist)

Every experiment must pass these checks, encoded as assertions in the evaluation harness:

- [ ] Feature at `(i, t)` uses only prices from `[t - lookback, t]`. Assertion: `panel.groupby('ticker').apply(lambda g: g['feature'].iloc[k]).compare(feature_from_full_series.iloc[k])` within tolerance.
- [ ] Target at `(i, t)` uses only prices from `[t, t + h]`. Assertion: `target.shift(h) ≈ (P.shift(-h) / P - 1).shift(h)`.
- [ ] No overlap between any (train_fold_dates ∪ train_target_horizon) and test_fold_dates. Unit test per fold.
- [ ] Embargo ≥ target horizon. Assertion raised if violated.
- [ ] All preprocessing (scaling, ranking, target encoding, imputation) fit on train only. Enforced by using `sklearn.pipeline.Pipeline`.
- [ ] No cross-sectional normalization uses pooled-across-date statistics. Rank within date, z-score within date — never across dates.
- [ ] No future adjusted-close values polluting features. yfinance adjusts retroactively; we snapshot the adjusted series at ingestion time, cache it, and feature-compute from the snapshot so re-adjustment does not alter history.
- [ ] Survivorship bias documented. Every backtest report banner must state: "Uses current S&P 500 constituents; survivorship bias present." Replace with PIT constituent history when available.
- [ ] Hyperparameter search on inner CV only. Final test fold is untouched by tuning. Nested CV enforced.
- [ ] Target leakage through identifier: no ticker-ID one-hot. Sector is allowed; ticker identity is not (would memorize).
- [ ] No leakage through date features: day-of-week, day-of-month allowed; year/month are risky in walk-forward. Exclude them for Phase 2.
- [ ] Delisted/renamed tickers: yfinance handles splits and dividends via `adj_close`; record delisting dates when known and treat post-delisting returns as NaN, not zero.

---

## Evaluation report template

Every experiment produces a one-page markdown report under `docs/experiments/<experiment_id>.md` with this structure:

```markdown
# <Experiment ID> — <Model name>

**Dates:** 20XX-XX-XX to 20XX-XX-XX
**Target:** <e.g., 21d forward return>
**CV:** walk-forward, train 3y rolling, embargo 30d, 12 folds
**Config hash:** abc1234
**Code hash:** def5678

## Predictive metrics
| Metric   | Value | 95% CI          |
| -------- | ----- | --------------- |
| IC       | 0.0XX | [0.0YY, 0.0ZZ]  |
| Rank IC  | 0.0XX | [0.0YY, 0.0ZZ]  |
| Hit rate | 0.XX  | [0.YY, 0.ZZ]    |

## Portfolio metrics (net of 5bps one-way cost)
| Metric         | Portfolio | Heuristic | SPY  | EW-S&P500 |
| -------------- | --------- | --------- | ---- | --------- |
| Ann. return    | X.XX%     | X.XX%     | X.XX%| X.XX%     |
| Ann. vol       | X.XX%     | X.XX%     | X.XX%| X.XX%     |
| Sharpe         | X.XX      | X.XX      | X.XX | X.XX      |
| MaxDD          | -X.XX%    | -X.XX%    | ...  | ...       |
| Calmar         | X.XX      | X.XX      | ...  | ...       |
| Turnover       | X.XX      | X.XX      | n/a  | X.XX      |

## Benchmark-relative (vs SPY)
| Metric  | Value | 95% CI          |
| ------- | ----- | --------------- |
| Alpha   | X.XX% | [Y.YY, Z.ZZ]    |
| Beta    | X.XX  | [Y.YY, Z.ZZ]    |
| IR      | X.XX  | [Y.YY, Z.ZZ]    |

## Feature importance (top 10)
...

## Caveats
- Survivorship bias present (current constituents only).
- Transaction cost model is flat, no market impact.
- ...
```

---

## Experimentation timeline (rough)

Estimates are elapsed time with the harness built, not calendar time.

| Phase | Scope                                | Engineering effort | Compute  |
| ----- | ------------------------------------ | ------------------ | -------- |
| 1     | Panel, labels, CV, harness, backtest | 3-4 days           | minutes  |
| 2     | E0-E8 + sweeps                       | 2-3 days           | hours    |
| 3     | E9-E14 (Prophet is the slow one)     | 4-5 days           | hours-day |
| 4     | Productionization                    | 2 days             | minutes  |

## Related beads issues

Filed as part of this plan:

| ID                     | Phase | Title                                                                |
| ---------------------- | ----- | -------------------------------------------------------------------- |
| `stock-analysis-0qr`   | 1     | Build point-in-time feature panel (`asset_daily_features_panel`)     |
| `stock-analysis-wzo`   | 1     | Forward-return labels (h=5/21/63) plus excess-vs-SPY and rank targets |
| `stock-analysis-xlv`   | 1     | Ingest SPY as benchmark; silver `benchmark_returns` table            |
| `stock-analysis-0w0`   | 1     | Walk-forward CV splitter with embargo and purging                    |
| `stock-analysis-2o2`   | 1     | Evaluation harness (IC, Sharpe, IR, bootstrap CIs)                   |
| `stock-analysis-txi`   | 1     | Walk-forward backtest runner with transaction costs                  |
| `stock-analysis-6xs`   | 1     | Experiment tracking: directory layout, index parquet, CLI command    |
| `stock-analysis-suy`   | 1     | Turnover-aware optimizer with L1 weight-change penalty (foundational for weekly) |
| `stock-analysis-072`   | 2     | Phase 2 baselines + classic ML (E0-E8) and sweep report              |
| `stock-analysis-t5i`   | 3     | Phase 3 DL + Prophet experiments (E9-E14)                            |
| `stock-analysis-5ph`   | 4     | Production integration: ML inference path + `forecast.engine` switch |
| `stock-analysis-jp9`   | —     | Data-leakage regression test suite (cross-cutting)                   |
| `stock-analysis-z4a`   | —     | Reconstruct PIT S&P 500 constituents from Wikipedia revision history (parallel; gates Phase 4) |

Dependencies:
- `wzo` depends on `0qr` (labels need the panel).
- `0w0`, `2o2`, `txi` depend on `0qr` and `wzo`.
- `xlv` is independent but needed for `2o2` benchmark metrics.
- `0z8` depends on nothing new (optimizer change); blocks `txi` and `072` (backtest needs turnover-aware optimizer since weekly is the locked-in cadence).
- `072` depends on all of Phase 1.
- `t5i` depends on `072` (only run DL if Phase 2 shows signal).
- `5ph` depends on `072` (and optionally `t5i`), and is gated on `5vf` or a documented Sharpe haircut.
- `jp9` blocks `072` — no ML experiments run until leakage tests pass.
- `5vf` is a parallel track; does not block Phase 1 or Phase 2 but gates the Phase 4 production switch.

## Risks and mitigations

| Risk                                                                       | Mitigation                                                          |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Overfitting via hyperparameter search                                      | Nested CV; report DSR; small search grids                           |
| Regime change (2020-era data doesn't predict 2026)                         | Rolling train window; regime-aware sweeps in Phase 2                |
| Transaction costs eat the edge                                             | Net-of-cost Sharpe is the gating metric; cost sensitivity sweep     |
| Survivorship bias inflates backtest returns                                | Every report labels this caveat; gate production switch on PIT data |
| DL hyperparameters drive results more than architecture                    | Report MLP with fixed hyperparameters; only advance if robust       |
| Prophet is slow to fit for 500 tickers                                     | Parallelize with joblib; subset to top-200 by liquidity for v1      |
| CVXPY solver instability with learned `mu` that has extreme values         | Winsorize `forecast_score` at cross-sectional 1%/99% before optimizer |
| Model-drift: live IC degrades                                              | Phase 4 monitoring captures realized returns; alert if IC < threshold over last 6 months |
| "Best model" picked by cherry-picking single fold                          | Require IC and Sharpe improvements across ≥80% of folds, not overall only |

## Definition of done for the overall ML upgrade

- Walk-forward backtest spanning ≥5 years of data, monthly rebalance, with full Phase 1 harness.
- At least one model in Phase 2 or Phase 3 beats E0 (heuristic) and E1 (SPY) on net-of-cost Sharpe **and** vs-SPY Information Ratio, with ≥80% of walk-forward folds positive.
- `docs/experiments/` contains the individual experiment reports plus a synthesis report.
- The live pipeline can run with `forecast.engine: ml` and produces recommendations. The heuristic remains available as a fallback.
- The Tableau dashboard footer shows model version and `expected_return_is_calibrated`.
- Every leakage-safeguard test passes in CI.
