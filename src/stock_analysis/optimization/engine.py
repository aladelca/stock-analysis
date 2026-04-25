from __future__ import annotations

import math

import cvxpy as cp
import numpy as np
import pandas as pd

from stock_analysis.config import OptimizerConfig


class OptimizationError(RuntimeError):
    """Raised when portfolio optimization cannot produce a valid solution."""


def optimize_long_only(
    optimizer_input: pd.DataFrame,
    covariance: pd.DataFrame,
    config: OptimizerConfig,
    *,
    w_prev: pd.Series | dict[str, float] | None = None,
) -> pd.Series:
    eligible = optimizer_input.loc[optimizer_input["eligible_for_optimization"].astype(bool)].copy()
    min_assets = math.ceil(1 / config.max_weight)
    if len(eligible) < min_assets:
        msg = (
            f"Need at least {min_assets} eligible assets for max_weight={config.max_weight}; "
            f"found {len(eligible)}"
        )
        raise OptimizationError(msg)

    eligible = eligible.set_index("ticker").sort_index()
    tickers = eligible.index.tolist()
    mu = eligible["expected_return"].astype(float).to_numpy()
    cov = covariance.reindex(index=tickers, columns=tickers).fillna(0).to_numpy(dtype=float)
    cov = (cov + cov.T) / 2
    cov = cov + np.eye(len(tickers)) * 1e-8
    previous_weights = _align_previous_weights(w_prev, tickers)

    weights = cp.Variable(len(tickers))
    utility = mu @ weights - config.risk_aversion * cp.quad_form(weights, cp.psd_wrap(cov))
    if config.lambda_turnover > 0:
        utility -= config.lambda_turnover * cp.norm1(weights - previous_weights)
    if config.commission_rate > 0:
        utility -= config.commission_rate * cp.norm1(weights - previous_weights)
    objective = cp.Maximize(utility)
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= config.max_weight,
    ]
    if config.max_trade_abs_weight is not None:
        constraints.append(cp.norm1(weights - previous_weights) <= config.max_trade_abs_weight)
    constraints.extend(_sector_constraints(eligible, weights, config))
    problem = cp.Problem(objective, constraints)

    solvers = [config.solver] if config.solver else ["CLARABEL", "OSQP", "SCS", None]
    last_error: Exception | None = None
    for solver in solvers:
        try:
            if solver is None:
                problem.solve()
            elif solver in cp.installed_solvers():
                problem.solve(solver=solver)
            else:
                continue
        except Exception as exc:  # pragma: no cover - solver-specific fallback
            last_error = exc
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and weights.value is not None:
            raw_weights = np.maximum(np.asarray(weights.value, dtype=float), 0)
            if raw_weights.sum() <= 0:
                break
            normalized = raw_weights / raw_weights.sum()
            return pd.Series(normalized, index=tickers, name="target_weight")

    msg = f"Optimization failed with status={problem.status}"
    if last_error is not None:
        msg = f"{msg}; last solver error: {last_error}"
    raise OptimizationError(msg)


def _align_previous_weights(
    w_prev: pd.Series | dict[str, float] | None,
    tickers: list[str],
) -> np.ndarray:
    if w_prev is None:
        return np.zeros(len(tickers))
    if isinstance(w_prev, pd.Series):
        aligned = w_prev.reindex(tickers).fillna(0).astype(float)
    else:
        aligned = pd.Series(w_prev, dtype=float).reindex(tickers).fillna(0)
    return aligned.to_numpy(dtype=float)


def _sector_constraints(
    eligible: pd.DataFrame,
    weights: cp.Variable,
    config: OptimizerConfig,
) -> list[cp.Constraint]:
    if config.sector_max_weight is None or "gics_sector" not in eligible.columns:
        return []
    sectors = eligible["gics_sector"].fillna("Unknown").astype(str)
    constraints: list[cp.Constraint] = []
    for sector in sorted(sectors.unique()):
        mask = (sectors == sector).astype(float).to_numpy()
        constraints.append(cp.sum(cp.multiply(mask, weights)) <= config.sector_max_weight)
    return constraints
