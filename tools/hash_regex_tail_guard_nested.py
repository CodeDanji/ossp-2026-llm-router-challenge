"""Immutable request-level cost tail guard primitives."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
import math
from numbers import Integral

import numpy as np

import hash_regex_cost_stabilization_nested as v32
from promptbudget.safety import grouped_folds


_ORDER_EPSILON = 1.0 + 1e-12
_TAIL_QUANTILE = 0.90
_HETEROGENEITY_MIN_SPREAD = math.log(1.10)
_HETEROGENEITY_REQUIRED_SEEDS = 2


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

    @classmethod
    def from_oof_predictions(
        cls, predictions: np.ndarray, residuals: np.ndarray
    ) -> "TailGuard":
        predicted = _log_cost_matrix(predictions)
        residual = _log_cost_matrix(residuals)
        if predicted.shape != residual.shape:
            raise ValueError("OOF predictions and residuals must align")
        ax31 = bucket_residual_summary(predicted[:, 1], residual[:, 1])
        think = bucket_residual_summary(predicted[:, 2], residual[:, 2])
        if not all(ax31["counts"]) or not all(think["counts"]):
            raise ValueError("every tail bucket must contain OOF residuals")
        return cls(
            ax31_edges=ax31["edges"],
            think_edges=think["edges"],
            ax31_log_guards=tuple(max(0.0, value) for value in ax31["p90"]),
            think_log_guards=tuple(max(0.0, value) for value in think["p90"]),
        )


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


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    ordered = np.sort(values)
    return float(ordered[math.ceil(quantile * len(ordered)) - 1])


def bucket_residual_summary(
    predictions: np.ndarray, residuals: np.ndarray
) -> dict[str, object]:
    predicted = np.asarray(predictions, dtype=float)
    observed = np.asarray(residuals, dtype=float)
    if (
        predicted.ndim != 1
        or observed.ndim != 1
        or len(predicted) < 4
        or predicted.shape != observed.shape
        or not np.isfinite(predicted).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("bucket inputs must be aligned finite vectors of at least four rows")
    edges = tuple(_higher_quantile(predicted, quantile) for quantile in (0.25, 0.50, 0.75))
    buckets = tuple(
        np.sort(observed[[bucket_index(value, edges) == index for value in predicted]])
        for index in range(4)
    )
    def summarize(bucket: np.ndarray, quantile: float) -> float | None:
        return _higher_quantile(bucket, quantile) if len(bucket) else None

    return {
        "edges": edges,
        "counts": tuple(len(bucket) for bucket in buckets),
        "p50": tuple(summarize(bucket, 0.50) for bucket in buckets),
        "p90": tuple(summarize(bucket, _TAIL_QUANTILE) for bucket in buckets),
        "p95": tuple(summarize(bucket, 0.95) for bucket in buckets),
        "max": tuple(float(bucket[-1]) if len(bucket) else None for bucket in buckets),
        "factors": tuple(
            math.exp(value) if value is not None else None
            for value in (summarize(bucket, _TAIL_QUANTILE) for bucket in buckets)
        ),
    }


def _partition_indices(data: v32.EvaluationData, indices: tuple[int, ...]) -> tuple[int, ...]:
    selected = tuple(indices)
    if len(selected) < 4 or len(set(selected)) != len(selected):
        raise ValueError("tail guard partition must contain unique rows")
    if any(isinstance(index, bool) or not isinstance(index, Integral) for index in selected):
        raise ValueError("tail guard partition indices must be integers")
    if any(index < 0 or index >= len(data.groups) for index in selected):
        raise ValueError("tail guard partition index is out of range")
    if len({data.groups[index] for index in selected}) < 4:
        raise ValueError("tail guard needs at least four content groups")
    return tuple(int(index) for index in selected)


def fit_tail_guard(
    data: v32.EvaluationData, partition_indices: tuple[int, ...], seed: int
) -> TailGuard:
    selected = _partition_indices(data, partition_indices)
    folds = grouped_folds(tuple(data.groups[index] for index in selected), folds=4, seed=seed)
    oof = np.empty((len(selected), 3), dtype=float)
    covered = np.zeros(len(selected), dtype=bool)
    for fold in folds:
        train_indices = tuple(selected[index] for index in fold.train_indices)
        validation_indices = tuple(selected[index] for index in fold.validation_indices)
        if set(data.groups[index] for index in train_indices) & set(
            data.groups[index] for index in validation_indices
        ):
            raise ValueError("grouped OOF fold leaks a content group")
        heads = v32.fit_log_cost_heads(
            data.matrix[list(train_indices)], data.log_costs[list(train_indices)]
        )
        local_validation = tuple(fold.validation_indices)
        if covered[list(local_validation)].any():
            raise ValueError("grouped OOF coverage overlaps")
        oof[list(local_validation)] = v32.predict_heads(
            heads, data.matrix[list(validation_indices)]
        )
        covered[list(local_validation)] = True
    if not covered.all() or not np.isfinite(oof).all():
        raise ValueError("grouped OOF coverage is incomplete")
    residuals = data.log_costs[list(selected)] - oof
    return TailGuard.from_oof_predictions(oof, residuals)


def diagnostic_status(reports: dict[int, dict[str, dict[str, object]]]) -> str:
    for model in ("ax31", "think"):
        passing = 0
        for report in reports.values():
            values = report.get(model, {}).get("p90")
            if values is None:
                continue
            numeric = tuple(float(value) for value in values)
            if len(numeric) == 4 and all(math.isfinite(value) for value in numeric):
                passing += max(numeric) - min(numeric) >= _HETEROGENEITY_MIN_SPREAD
        if passing >= _HETEROGENEITY_REQUIRED_SEEDS:
            return "tail-signal-present"
    return "tail-no-signal"


def score_guarded_batch_policy(
    data: v32.EvaluationData,
    indices: tuple[int, ...],
    scores: np.ndarray,
    base_log_costs: np.ndarray,
    guard: TailGuard,
) -> dict[str, object]:
    logs = _log_cost_matrix(base_log_costs)
    with np.errstate(over="ignore", invalid="ignore"):
        base_costs = np.exp(logs)
    if not np.isfinite(base_costs).all() or np.any(base_costs <= 0.0):
        raise ValueError("base log costs must exponentiate to finite positive costs")
    guarded = apply_tail_guard(apply_ordering_clamp(base_costs), logs, guard)
    report = v32.score_batch_policy(
        data, indices, scores, np.log(guarded), (1.0, 1.0)
    )
    report["guard_metadata"] = {
        "light_multiplier": 1.0,
        "ax31_edges": guard.ax31_edges,
        "think_edges": guard.think_edges,
        "ax31_log_guards": guard.ax31_log_guards,
        "think_log_guards": guard.think_log_guards,
    }
    return report


def evaluate_inner_guard(
    data: v32.EvaluationData, outer_train: tuple[int, ...], seed: int = 137
) -> dict[str, object]:
    selected = _partition_indices(data, tuple(outer_train))
    reports = []
    inner_folds = []
    for fold in grouped_folds(tuple(data.groups[index] for index in selected), folds=4, seed=seed):
        train_indices = tuple(selected[index] for index in fold.train_indices)
        validation_indices = tuple(selected[index] for index in fold.validation_indices)
        guard = fit_tail_guard(data, train_indices, seed)
        quality = v32.fit_raw_quality_heads(
            data.matrix[list(train_indices)], data.scores[list(train_indices)]
        )
        costs = v32.fit_log_cost_heads(
            data.matrix[list(train_indices)], data.log_costs[list(train_indices)]
        )
        report = score_guarded_batch_policy(
            data,
            validation_indices,
            v32.predict_heads(quality, data.matrix[list(validation_indices)]),
            v32.predict_heads(costs, data.matrix[list(validation_indices)]),
            guard,
        )
        reports.append(report)
        inner_folds.append(
            {
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "guard": guard,
                "report": report,
            }
        )
    return {
        "inner_folds": tuple(inner_folds),
        "pooled_for_routing": False,
        "admission": v32.admit_inner_candidate(reports),
    }


def select_inner_guard(
    data: v32.EvaluationData, outer_train: tuple[int, ...], seed: int
) -> dict[str, object]:
    evaluation = evaluate_inner_guard(data, outer_train, seed)
    if not evaluation["admission"]["admitted"]:
        return {"status": "no-admitted-tail-guard", "route": "all-light", "guard": None}
    return {
        "status": "admitted-tail-guard",
        "route": "tail-guarded",
        "admission": evaluation["admission"],
        "inner_folds": evaluation["inner_folds"],
        "pooled_for_routing": False,
    }


def evaluate_outer_guard_fold(
    data: v32.EvaluationData,
    outer_train: tuple[int, ...],
    outer_test: tuple[int, ...],
    seed: int = 137,
    fold: int = 0,
) -> dict[str, object]:
    train_indices = _partition_indices(data, tuple(outer_train))
    test_indices = _partition_indices(data, tuple(outer_test))
    if set(train_indices) & set(test_indices) or set(data.groups[index] for index in train_indices) & set(
        data.groups[index] for index in test_indices
    ):
        raise ValueError("outer train and test partitions must be group-disjoint")
    selection = select_inner_guard(data, train_indices, seed)
    quality = v32.fit_raw_quality_heads(data.matrix[list(train_indices)], data.scores[list(train_indices)])
    costs = v32.fit_log_cost_heads(data.matrix[list(train_indices)], data.log_costs[list(train_indices)])
    score_prediction = v32.predict_heads(quality, data.matrix[list(test_indices)])
    cost_prediction = v32.predict_heads(costs, data.matrix[list(test_indices)])
    raw_report = v32.score_batch_policy(data, test_indices, score_prediction, cost_prediction, (1.0, 1.0))
    fallback = selection["route"] == "all-light"
    guard = None
    if fallback:
        selected_report = v32._score_all_light_policy(data, test_indices)
    else:
        guard = fit_tail_guard(data, train_indices, seed)
        selected_report = score_guarded_batch_policy(
            data, test_indices, score_prediction, cost_prediction, guard
        )
    selected_non_light = v32._final_non_light(selected_report)
    raw_non_light = v32._final_non_light(raw_report)
    return {
        "seed": seed,
        "fold": fold,
        "outer_train_indices": train_indices,
        "outer_test_indices": test_indices,
        "inner_selection": selection,
        "route": selection["route"],
        "fallback_all_light": fallback,
        "guard": guard,
        "selected_report": selected_report,
        "outer_test_evaluations": 1,
        "selected_final_non_light": selected_non_light,
        "tier_retention": selected_non_light["tiers"],
        "total_retention": selected_non_light["total"],
        "raw_comparator": {
            "pair": (1.0, 1.0),
            "report": raw_report,
            "final_non_light": raw_non_light,
        },
    }


def aggregate_outer_guard_folds(folds: list[dict[str, object]]) -> dict[str, object]:
    aggregate = v32.aggregate_outer_folds(folds)
    retention = aggregate["retention"]["non_light_retention"]
    if not aggregate["promotion"]["outer_45_of_45_pass"]:
        status = "cost-calibration-no-go"
    elif aggregate["fallback_all_light_folds"]:
        status = "safe-but-collapse"
    elif retention["not_applicable"]:
        status = "safe-but-collapse"
    elif retention["value"] < Decimal("0.20"):
        status = "safe-but-collapse"
    else:
        status = "safe-candidate"
    return {**aggregate, "status": status}
