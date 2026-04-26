from __future__ import annotations

import csv

import pandas as pd
import pytest

from stock_analysis.ml.autoresearch_eval import (
    RESULT_COLUMNS,
    append_result_tsv,
    decide_candidate,
    result_to_tsv_row,
    select_best_turnover_row,
)


def test_decide_candidate_rejects_when_sharpe_does_not_beat_spy() -> None:
    decision = decide_candidate(
        {
            "candidate_sharpe": 1.0,
            "spy_sharpe": 1.1,
            "active_return": 0.1,
            "information_ratio": 0.5,
            "sharpe_diff_ci_low": 0.1,
        }
    )

    assert decision["status"] == "rejected"
    assert not decision["passed_spy_gate"]


def test_decide_candidate_rejects_when_active_return_is_non_positive() -> None:
    decision = decide_candidate(
        {
            "candidate_sharpe": 2.0,
            "spy_sharpe": 1.1,
            "active_return": 0.0,
            "information_ratio": 0.5,
            "sharpe_diff_ci_low": 0.1,
        }
    )

    assert decision["status"] == "rejected"
    assert "active return" in decision["notes"]


def test_decide_candidate_rejects_when_information_ratio_is_non_positive() -> None:
    decision = decide_candidate(
        {
            "candidate_sharpe": 2.0,
            "spy_sharpe": 1.1,
            "active_return": 0.1,
            "information_ratio": -0.1,
            "sharpe_diff_ci_low": 0.1,
        }
    )

    assert decision["status"] == "rejected"
    assert "information ratio" in decision["notes"]


def test_decide_candidate_marks_provisional_when_ci_includes_zero() -> None:
    decision = decide_candidate(
        {
            "candidate_sharpe": 3.0,
            "spy_sharpe": 1.0,
            "sharpe_diff": 2.0,
            "active_return": 0.5,
            "information_ratio": 2.3,
            "sharpe_diff_ci_low": -0.01,
        }
    )

    assert decision["status"] == "provisional"
    assert decision["passed_spy_gate"]


def test_decide_candidate_marks_go_when_ci_lower_bound_is_positive() -> None:
    decision = decide_candidate(
        {
            "candidate_sharpe": 3.0,
            "spy_sharpe": 1.0,
            "sharpe_diff": 2.0,
            "active_return": 0.5,
            "information_ratio": 2.3,
            "sharpe_diff_ci_low": 0.01,
        }
    )

    assert decision["status"] == "go"
    assert decision["objective_improved"]


def test_result_to_tsv_row_uses_stable_columns() -> None:
    result = {
        "timestamp_utc": "2026-04-24T00:00:00+00:00",
        "iteration_id": "iter-1",
        "git_commit": "abc123",
        "candidate": {"candidate_id": "candidate", "description": "Candidate"},
        "config": {
            "input_run_root": "data/runs/source",
            "max_assets": 100,
            "max_rebalances": 48,
            "optimizer_max_weight": 0.3,
            "commission_rate": 0.02,
            "cost_bps": 5.0,
        },
        "metrics": {
            "comparison": {
                "candidate_sharpe": 2.0,
                "spy_sharpe": 1.0,
                "sharpe_diff": 1.0,
                "active_return": 0.2,
                "tracking_error": 0.1,
                "information_ratio": 2.0,
                "ir_observations": 12.0,
            }
        },
        "decision": {"status": "provisional", "notes": "beats SPY on point estimates"},
    }

    row = result_to_tsv_row(result)

    assert tuple(row) == RESULT_COLUMNS
    assert row["candidate_id"] == "candidate"
    assert row["commission_rate"] == 0.02
    assert row["information_ratio"] == 2.0


def test_select_best_turnover_row_prefers_objective_then_lower_turnover() -> None:
    summary = pd.DataFrame(
        {
            "lambda_turnover": [0.01, 0.1, 1.0],
            "information_ratio": [0.5, 0.7, 0.7],
            "annualized_return": [0.10, 0.08, 0.08],
            "candidate_sharpe": [1.0, 1.1, 1.1],
            "mean_turnover": [0.6, 0.3, 0.2],
        }
    )

    best = select_best_turnover_row(summary, "information_ratio")

    assert best["lambda_turnover"] == 1.0
    assert best["information_ratio"] == 0.7


def test_append_result_tsv_writes_header_and_row(tmp_path) -> None:
    path = tmp_path / "results.tsv"
    result = {
        "timestamp_utc": "2026-04-24T00:00:00+00:00",
        "iteration_id": "iter-1",
        "git_commit": "abc123",
        "candidate": {"candidate_id": "candidate", "description": "Candidate"},
        "config": {"input_run_root": "data/runs/source"},
        "metrics": {"comparison": {"candidate_sharpe": 2.0}},
        "decision": {"status": "provisional", "notes": "note"},
    }

    append_result_tsv(path, result)

    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "candidate"
    assert float(rows[0]["candidate_sharpe"]) == pytest.approx(2.0)


def test_append_result_tsv_migrates_older_header(tmp_path) -> None:
    path = tmp_path / "results.tsv"
    old_columns = [
        "timestamp_utc",
        "iteration_id",
        "git_commit",
        "candidate_id",
        "candidate_description",
        "input_run_root",
        "max_assets",
        "max_rebalances",
        "optimizer_max_weight",
        "cost_bps",
        "candidate_sharpe",
        "spy_sharpe",
        "sharpe_diff",
        "sharpe_diff_ci_low",
        "sharpe_diff_ci_high",
        "annualized_return",
        "active_return",
        "tracking_error",
        "information_ratio",
        "max_drawdown",
        "mean_turnover",
        "ir_observations",
        "status",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_columns, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": "2026-04-24T00:00:00+00:00",
                "candidate_id": "old-candidate",
                "candidate_sharpe": 1.5,
                "status": "provisional",
            }
        )

    append_result_tsv(
        path,
        {
            "timestamp_utc": "2026-04-25T00:00:00+00:00",
            "candidate": {"candidate_id": "new-candidate", "description": "Candidate"},
            "config": {"commission_rate": 0.02},
            "metrics": {
                "comparison": {"candidate_sharpe": 2.0},
                "cashflow": {"strategy_ending_value": 1200.0},
            },
            "decision": {"status": "provisional"},
        },
    )

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == RESULT_COLUMNS
    assert [row["candidate_id"] for row in rows] == ["old-candidate", "new-candidate"]
    assert rows[0]["commission_rate"] == ""
    assert rows[1]["commission_rate"] == "0.02"
    assert rows[1]["strategy_ending_value"] == "1200.0"
