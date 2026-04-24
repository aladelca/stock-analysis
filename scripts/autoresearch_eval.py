from __future__ import annotations

import argparse
from pathlib import Path

from stock_analysis.ml.autoresearch_eval import (
    AutoresearchEvalConfig,
    append_result_tsv,
    evaluate_candidate,
    result_to_json,
)
from stock_analysis.ml.mlflow_tracking import (
    DEFAULT_MLFLOW_EXPERIMENT_NAME,
    DEFAULT_MLFLOW_TRACKING_URI,
    log_autoresearch_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an autoresearch candidate against the fixed SPY-relative contract."
    )
    parser.add_argument("--candidate", default="e8_baseline", help="Candidate id to evaluate.")
    parser.add_argument(
        "--input-run-root",
        type=Path,
        default=Path("data/runs/phase2-source-20260424"),
        help="Phase 1 run root containing silver/gold artifacts.",
    )
    parser.add_argument("--max-assets", type=int, default=100)
    parser.add_argument("--max-rebalances", type=int, default=48)
    parser.add_argument("--optimizer-max-weight", type=float, default=0.30)
    parser.add_argument("--risk-aversion", type=float, default=10.0)
    parser.add_argument("--min-trade-weight", type=float, default=0.005)
    parser.add_argument("--lambda-turnover", type=float, default=0.001)
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.02,
        help="Commission rate applied to absolute traded notional.",
    )
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--rebalance-step-days", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=15)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--covariance-lookback-days", type=int, default=252)
    parser.add_argument("--liquidity-column", default="dollar_volume_21d")
    parser.add_argument("--iteration-id", default=None)
    parser.add_argument(
        "--results-tsv",
        type=Path,
        default=None,
        help="Optional append-only TSV ledger path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the full JSON result.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log the evaluator result to MLflow.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help=f"MLflow tracking URI. Defaults to {DEFAULT_MLFLOW_TRACKING_URI}.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=DEFAULT_MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name used when --mlflow is set.",
    )
    args = parser.parse_args()

    result = evaluate_candidate(
        AutoresearchEvalConfig(
            candidate_id=args.candidate,
            input_run_root=args.input_run_root,
            max_assets=args.max_assets,
            max_rebalances=args.max_rebalances,
            optimizer_max_weight=args.optimizer_max_weight,
            risk_aversion=args.risk_aversion,
            min_trade_weight=args.min_trade_weight,
            lambda_turnover=args.lambda_turnover,
            commission_rate=args.commission_rate,
            horizon_days=args.horizon_days,
            rebalance_step_days=args.rebalance_step_days,
            embargo_days=args.embargo_days,
            cost_bps=args.cost_bps,
            covariance_lookback_days=args.covariance_lookback_days,
            liquidity_column=args.liquidity_column,
            iteration_id=args.iteration_id,
        )
    )
    if args.results_tsv is not None:
        append_result_tsv(args.results_tsv, result)
    if args.mlflow:
        artifacts = [args.results_tsv] if args.results_tsv is not None else []
        run_id = log_autoresearch_result(
            result,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment_name,
            artifacts=artifacts,
        )
        result["mlflow"] = {
            "experiment_name": args.mlflow_experiment_name,
            "run_id": run_id,
            "tracking_uri": args.mlflow_tracking_uri or DEFAULT_MLFLOW_TRACKING_URI,
        }
    output = result_to_json(result)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(f"{output}\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
