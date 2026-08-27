"""Immutable request-level cost tail guard primitives."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math

import numpy as np


_ORDER_EPSILON = 1.0 + 1e-12


def _finite_tuple(values: tuple[float, ...], size: int, label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain exactly {size} finite values")
    return result


@dataclass(frozen=True)
class TailGuard:
    ax31_edges: tuple[float, float, float]
    think_edges: tuple[float, float, float]
    ax31_log_guards: tuple[float, float, float, float]
    think_log_guards: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for name in ("ax31_edges", "think_edges"):
            edges = _finite_tuple(getattr(self, name), 3, name)
            if tuple(sorted(edges)) != edges:
                raise ValueError(f"{name} must be nondecreasing")
            object.__setattr__(self, name, edges)
        for name in ("ax31_log_guards", "think_log_guards"):
            guards = _finite_tuple(getattr(self, name), 4, name)
            if any(value < 0.0 for value in guards):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, guards)


def bucket_index(value: float, edges: tuple[float, float, float]) -> int:
    numeric = float(value)
    checked_edges = _finite_tuple(edges, 3, "edges")
    if not math.isfinite(numeric) or tuple(sorted(checked_edges)) != checked_edges:
        raise ValueError("bucket values and edges must be finite and nondecreasing")
    return bisect_right(checked_edges, numeric)


def _cost_matrix(costs: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(costs, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result):
        raise ValueError(f"{label} must have shape (rows, 3)")
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{label} must be finite and positive")
    return result


def _log_cost_matrix(log_costs: np.ndarray) -> np.ndarray:
    result = np.asarray(log_costs, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result):
        raise ValueError("base_log_costs must have shape (rows, 3)")
    if not np.isfinite(result).all():
        raise ValueError("base_log_costs must be finite")
    return result


def apply_ordering_clamp(costs: np.ndarray) -> np.ndarray:
    result = _cost_matrix(costs, "costs").copy()
    with np.errstate(over="ignore", invalid="ignore"):
        result[:, 1] = np.maximum(result[:, 1], result[:, 0] * _ORDER_EPSILON)
        result[:, 2] = np.maximum(result[:, 2], result[:, 1] * _ORDER_EPSILON)
    if not np.isfinite(result).all():
        raise ValueError("ordered costs must be finite")
    return result


def apply_tail_guard(
    base_costs: np.ndarray, base_log_costs: np.ndarray, guard: TailGuard
) -> np.ndarray:
    costs = _cost_matrix(base_costs, "base_costs")
    log_costs = _log_cost_matrix(base_log_costs)
    if costs.shape != log_costs.shape:
        raise ValueError("base costs and log costs must have matching shapes")
    result = costs.copy()
    for column, edges, guards in (
        (1, guard.ax31_edges, guard.ax31_log_guards),
        (2, guard.think_edges, guard.think_log_guards),
    ):
        selected = np.asarray(
            [guards[bucket_index(value, edges)] for value in log_costs[:, column]]
        )
        with np.errstate(over="ignore", invalid="ignore"):
            factors = np.exp(selected)
            result[:, column] *= factors
        if not np.isfinite(factors).all() or not np.isfinite(result[:, column]).all():
            raise ValueError("tail guard multiplication must be finite")
    return apply_ordering_clamp(result)
