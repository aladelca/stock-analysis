from __future__ import annotations

import csv

import pytest

from stock_analysis.ml.autoresearch_eval import (
    RESULT_COLUMNS,
    append_result_tsv,
    decide_candidate,
    result_to_tsv_row,
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
    assert row["information_ratio"] == 2.0


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
