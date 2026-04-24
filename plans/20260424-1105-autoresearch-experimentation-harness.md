# Autoresearch Experimentation Harness

## Goal

- Implement an autoresearch-inspired experimentation harness that can iterate on candidate
  forecasting models for this repository while preserving a fixed, auditable SPY-relative
  evaluation contract.
- The harness should make it easy for an agent to edit one candidate model file, run a
  deterministic evaluator, append results to a ledger, and only promote candidates that improve
  the current Phase 2 E8 baseline on SPY-relative portfolio metrics.

## Request Snapshot

- User request: "i want to use autoresearch to do an experimentation to improve the model ... check how can we implement it" followed by "ok create the plan for doing this $plan-feature-implementation"
- Owner or issue: `stock-analysis-o9m`
- Plan file: `plans/20260424-1105-autoresearch-experimentation-harness.md`

## Current State

- The project already uses `uv`; no install bootstrap is needed for `uv`.
- `src/stock_analysis/ml/phase2.py` owns the Phase 2 experiment harness:
  - `Phase2Config`
  - `run_phase2`
  - `RidgeForecastModel`
  - `LightGBMForecastModel`
  - `BlendedForecastModel`
  - `_gating_decision`
  - `_sharpe_difference_ci`
- `src/stock_analysis/ml/evaluation.py` owns aligned benchmark-relative metrics:
  - `benchmark_relative_metrics`
  - `information_ratio = active_return / tracking_error`
  - exact date alignment before active-return calculation
- `src/stock_analysis/forecasting/ml_forecast.py` owns the production ML inference path and
  currently uses the Phase 2 E8 Ridge plus LightGBM blend.
- `configs/portfolio.yaml` is already configured for the current production model:
  - `forecast.engine: ml`
  - `forecast.ml_model_version: phase2-e8-ridge-lightgbm-blend-v1`
  - `forecast.ml_max_assets: 100`
  - `optimizer.max_weight: 0.30`
- `docs/experiments/phase2-spy-relative-experimentation.md` records the current benchmark:
  - E8 Sharpe `2.987197`
  - SPY Sharpe `1.130685`
  - Active return `0.636036`
  - SPY-relative IR `2.212435`
  - 46 aligned observations
  - Sharpe-difference 95% CI `[-0.446, 1.542]`
- `docs/experiments/phase2-detailed-summary.md` and `phase2-report.md` still mark the result
  `PROVISIONAL` because the Sharpe-difference CI includes zero.
- `src/stock_analysis/ml/tracking.py` provides filesystem experiment tracking under
  `data/experiments/`, but it is not yet an autoresearch-style ledger.
- `src/stock_analysis/ml/experiments.py` supports YAML-driven heuristic experiments only; it
  does not yet support arbitrary candidate model modules.

## Findings

- `karpathy/autoresearch` is best treated as a workflow pattern, not as an importable runtime
  dependency for this repo. The upstream pattern uses a fixed evaluator, one editable training
  file, a results ledger, and an agent loop that keeps metric-improving commits.
- This repository already has the more important pieces for a safe adaptation: fixed data
  artifacts, aligned SPY-relative metrics, and a Phase 2 backtest runner.
- The highest-risk failure mode is allowing the autonomous loop to edit the evaluator,
  benchmark alignment, labels, or SPY-relative IR calculation. The implementation must restrict
  the editable surface to candidate model logic and config-like candidate choices.
- The right primary objective is not just "beat SPY" on point estimates. The current model
  already does that. The next meaningful objective is to improve the result enough that
  `_gating_decision` becomes `GO`, ideally by moving the bootstrap Sharpe-difference CI lower
  bound above zero.

## Scope

### In scope

- Add an autoresearch candidate module with a small, explicit interface for model factories and
  optimizer/search-space variants.
- Add a fixed evaluator script that executes candidate models against the current Phase 2 data
  and emits machine-readable metrics.
- Add a results ledger under `experiments/autoresearch/` for repeatable experiment comparison.
- Add a runbook/program file that tells an agent exactly what it may edit and how to decide
  whether a candidate improves.
- Add unit and integration tests for evaluator contracts, metric parsing, and candidate model
  wiring.
- Add CLI or script-level commands that can be run from the repo root with `uv run`.

### Out of scope

- Running the full autonomous search loop in this plan step.
- Installing or vendoring `karpathy/autoresearch`.
- Allowing autonomous edits to `src/stock_analysis/ml/evaluation.py`,
  `src/stock_analysis/ml/phase2.py` gating logic, SPY benchmark generation, or labels.
- Promoting a new production model before a confirmation backtest and documentation update.
- Replacing Beads, `ExperimentTracker`, or the existing Phase 2 reports.

## File Plan

| Path | Action | Details |
| --- | --- | --- |
| `src/stock_analysis/ml/autoresearch_candidate.py` | create | Define the editable candidate surface. Expose a `CandidateSpec` dataclass, candidate registry, feature column selection, model factory builders, and score transforms. This is the main file an autoresearch agent may edit. |
| `src/stock_analysis/ml/autoresearch_eval.py` | create | Implement reusable evaluator functions that load Phase 1 artifacts, run candidate backtests, align SPY benchmark rows, compute metrics, compare against the current E8 baseline, and return structured results. |
| `scripts/autoresearch_eval.py` | create | Thin CLI wrapper around `stock_analysis.ml.autoresearch_eval`. Accept candidate id, input run root, output root, max assets, max rebalances, optimizer max weight, and JSON/TSV output options. |
| `experiments/autoresearch/program.md` | create | Agent-facing instructions adapted from `karpathy/autoresearch`: editable files, forbidden files, objective, loop, keep/revert criteria, commit convention, and confirmation-run protocol. |
| `experiments/autoresearch/results.tsv` | create | Seed ledger with baseline rows for current production E8 and any initial candidate. Columns should be stable and append-only. |
| `experiments/autoresearch/README.md` | create | Human-facing explanation of how to run the evaluator manually and how to start/stop an autoresearch loop. |
| `runbooks/autoresearch.md` | create | Operator runbook: prerequisites, baseline evaluation, fast loop, confirmation run, interpreting results, and promotion checklist. |
| `src/stock_analysis/cli.py` | modify | Optional but recommended: add `stock-analysis autoresearch-eval` command if the script-only interface is too detached from the existing CLI. Keep `scripts/autoresearch_eval.py` even if CLI is added, for agent simplicity. |
| `src/stock_analysis/ml/phase2.py` | modify | Add extension seams only if needed: a public helper to run a custom `ExperimentSpec` or `ModelFactory` without changing private functions. Do not alter SPY-relative metric formulas or gating semantics. |
| `src/stock_analysis/ml/tracking.py` | modify | Optional: add metadata fields for autoresearch iteration id, parent commit, candidate id, and objective value if the evaluator writes through `ExperimentTracker`. |
| `tests/unit/test_autoresearch_candidate.py` | create | Validate candidate registry contracts, feature-column validation, score transform behavior, and model factory construction on synthetic data. |
| `tests/unit/test_autoresearch_eval.py` | create | Validate objective comparison, TSV row serialization, failure handling, and refusal to pass when SPY-relative active return or IR is non-positive. |
| `tests/integration/test_autoresearch_eval.py` | create | Run the evaluator on small synthetic Phase 1 artifacts and assert JSON/TSV outputs include candidate and SPY metrics with aligned observation counts. |
| `docs/experiments/autoresearch-summary.md` | create | Summary template for completed autoresearch runs. Include baseline, best candidate, rejected candidates, metric deltas, and confirmation result. |
| `pyproject.toml` | modify | Only if needed to expose a script entry point. No new dependency is expected for the initial harness. |

## Data and Contract Changes

- New append-only TSV contract: `experiments/autoresearch/results.tsv`.
- Proposed columns:
  - `timestamp_utc`
  - `iteration_id`
  - `git_commit`
  - `candidate_id`
  - `candidate_description`
  - `input_run_root`
  - `max_assets`
  - `max_rebalances`
  - `optimizer_max_weight`
  - `cost_bps`
  - `candidate_sharpe`
  - `spy_sharpe`
  - `sharpe_diff`
  - `sharpe_diff_ci_low`
  - `sharpe_diff_ci_high`
  - `annualized_return`
  - `active_return`
  - `tracking_error`
  - `information_ratio`
  - `max_drawdown`
  - `mean_turnover`
  - `ir_observations`
  - `status`
  - `notes`
- New JSON output contract from `scripts/autoresearch_eval.py`:
  - top-level `candidate`
  - top-level `config`
  - top-level `metrics`
  - top-level `baseline`
  - top-level `decision`
- No database migrations.
- No new environment variables.
- No new runtime dependency is expected.

## Implementation Steps

1. Create `experiments/autoresearch/` and document the autoresearch adaptation.
   - Add `program.md` with strict edit boundaries.
   - Add `README.md` with manual commands.
   - Add a `results.tsv` header row.

2. Create `src/stock_analysis/ml/autoresearch_candidate.py`.
   - Define `CandidateSpec`.
   - Provide `candidate_id`, `description`, `feature_columns`, `model_factory`,
     `training_target_column`, and optional `score_transform`.
   - Seed it with the current E8 behavior so baseline parity can be verified.
   - Keep the surface small enough for autonomous edits without touching evaluators.

3. Add evaluator core in `src/stock_analysis/ml/autoresearch_eval.py`.
   - Load artifacts from `data/runs/phase2-source-20260424` by default.
   - Apply `max_assets` using the same liquidity logic as Phase 2.
   - Run the candidate model through the same walk-forward backtest and optimizer.
   - Compute portfolio metrics using existing `portfolio_metrics`.
   - Compute SPY-relative metrics using existing `benchmark_relative_metrics`.
   - Compute Sharpe-difference CI using the existing Phase 2 CI helper or a public equivalent.
   - Compare against stored baseline thresholds from
     `docs/experiments/phase2-spy-relative-experimentation.md` or a structured constant.

4. Add `scripts/autoresearch_eval.py`.
   - Parse arguments with stdlib `argparse` or use the existing Typer app if a CLI command is
     preferred.
   - Print compact JSON to stdout for agents.
   - Append one TSV row when `--results-tsv` is provided.
   - Exit `0` for successful evaluation even if the candidate does not improve; encode pass/fail
     in JSON. Reserve non-zero exit for infrastructure failures.

5. Add optional `stock-analysis autoresearch-eval` CLI command.
   - Mirror script arguments.
   - Keep the script as the canonical command for autoresearch agents because it is easy to call
     and easy to sandbox.

6. Add tests.
   - Unit-test candidate registry and score transforms.
   - Unit-test evaluator decision logic with hand-built metric dictionaries.
   - Integration-test a small synthetic backtest that includes SPY and verifies aligned
     `ir_observations`.

7. Add runbook and summary template.
   - `runbooks/autoresearch.md` documents the human workflow.
   - `docs/experiments/autoresearch-summary.md` documents how to summarize completed loops.

8. Run validation.
   - Execute formatting, linting, typing, focused tests, then the full suite.
   - Run one evaluator smoke test against `data/runs/phase2-source-20260424` if that ignored
     artifact exists locally.

9. Promotion workflow for later implementation.
   - If a candidate reaches `GO`, copy its stable model definition into
     `src/stock_analysis/forecasting/ml_forecast.py` or convert the production model to import the
     candidate registry by id.
   - Regenerate `docs/experiments/phase2-detailed-summary.md`,
     `docs/experiments/phase2-spy-relative-experimentation.md`, and Tableau outputs.
   - Keep a rollback path by preserving the current E8 candidate id.

## Tests

- Unit: `tests/unit/test_autoresearch_candidate.py`
  - Candidate registry contains a baseline candidate.
  - Candidate feature columns must exist or fail with a clear error.
  - Candidate model factory returns objects with `predict`.
  - Score transforms preserve row counts and finite values.

- Unit: `tests/unit/test_autoresearch_eval.py`
  - Decision is `reject` when Sharpe does not beat SPY.
  - Decision is `reject` when active return is non-positive.
  - Decision is `reject` when information ratio is non-positive.
  - Decision is `provisional` when point estimate beats SPY but CI lower bound is <= 0.
  - Decision is `go` when point estimate beats SPY and CI lower bound is > 0.
  - TSV serialization writes stable column order.

- Integration: `tests/integration/test_autoresearch_eval.py`
  - Build small synthetic Phase 1 artifacts under `tmp_path`.
  - Run the evaluator on a seeded candidate.
  - Assert output JSON includes candidate, SPY, active-return, tracking-error, IR, and aligned
    observation count.
  - Assert evaluator appends one TSV row.

- Regression:
  - Reuse `tests/unit/test_evaluation.py::test_benchmark_relative_information_ratio_uses_aligned_active_returns`.
  - Reuse `tests/unit/test_phase2.py::test_run_phase2_writes_reports_and_tracking_artifacts`.

## Validation

- Format: `uv run ruff format --check src tests scripts`
- Lint: `uv run ruff check src tests scripts`
- Types: `uv run mypy src`
- Focused tests:
  - `uv run pytest tests/unit/test_autoresearch_candidate.py tests/unit/test_autoresearch_eval.py tests/integration/test_autoresearch_eval.py`
- Full tests:
  - `uv run pytest`
- Manual smoke, if local Phase 2 source artifacts exist:
  - `uv run python scripts/autoresearch_eval.py --candidate e8_baseline --input-run-root data/runs/phase2-source-20260424 --max-assets 100 --max-rebalances 48 --optimizer-max-weight 0.30 --results-tsv experiments/autoresearch/results.tsv`

## Risks and Mitigations

- Risk: The autonomous loop overfits to the 46-observation sampled run.
  - Mitigation: require a fast-loop metric and a separate confirmation run with larger/full
    rebalance coverage before promotion.

- Risk: The agent improves results by changing evaluation code.
  - Mitigation: `program.md` must mark evaluator, SPY metrics, labels, and backtest sampling as
    forbidden edit surfaces; code review rejects changes there unless explicitly requested.

- Risk: Candidate models become too slow for iterative research.
  - Mitigation: default fast-loop config uses fixed LightGBM params, no nested CV, and bounded
    `max_rebalances`; confirmation runs can be slower.

- Risk: Candidate score scale breaks the optimizer.
  - Mitigation: candidate module must expose score transforms and finite-value checks; evaluator
    rejects non-finite forecasts and records failure status instead of crashing late.

- Risk: Production model and candidate model drift.
  - Mitigation: seed candidate registry with the current production E8 candidate and add a parity
    smoke test before trying new candidates.

- Risk: Results ledger is polluted by failed infrastructure runs.
  - Mitigation: distinguish `status=failed_infrastructure`, `status=rejected`, `status=provisional`,
    and `status=go`.

## Open Questions

- None

## Acceptance Criteria

- `experiments/autoresearch/program.md` defines a safe autonomous loop with explicit editable and
  forbidden files.
- `scripts/autoresearch_eval.py` can evaluate at least the seeded E8 baseline candidate and print
  JSON metrics.
- `experiments/autoresearch/results.tsv` receives stable append-only rows from evaluator runs.
- Evaluator decisions distinguish `reject`, `provisional`, and `go` using SPY-relative Sharpe,
  active return, information ratio, and CI lower bound.
- Tests cover candidate registry, evaluator decision logic, TSV output, and an integration smoke
  run on synthetic artifacts.
- The harness does not change SPY-relative IR formula, SPY date alignment, labels, or Phase 2
  gating semantics.

## Definition of Done

- Plan implemented in code and docs.
- Tests added or updated for new candidate/evaluator behavior.
- `uv run ruff format --check src tests scripts` passes.
- `uv run ruff check src tests scripts` passes.
- `uv run mypy src` passes.
- `uv run pytest` passes.
- Manual evaluator smoke command succeeds when local Phase 2 source artifacts exist.
- Plan updated if implementation scope changes.
