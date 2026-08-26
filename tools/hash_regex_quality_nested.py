# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Pure Train-only helpers for the fixed hash-regex quality experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


HASH_BINS = 256
RIDGE_ALPHA = 100.0
QUALITY_CANDIDATES = ("raw", "weighted-uplift", "regret-weighted-raw")
GAMMA_GRID = (0.0, 1.0, 2.0, 4.0)
ETA_GRID = (0.0, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class LinearScoreHeads:
    """Three standardized linear heads with their fit-partition transform."""

    mean: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray


def _matrix(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a non-empty finite matrix")
    return result


def _targets(value: np.ndarray, rows: int, label: str) -> np.ndarray:
    result = _matrix(value, label)
    if result.shape != (rows, 3):
        raise ValueError(f"{label} must have shape (rows, 3)")
    return result


def _strength(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("strength must be finite and non-negative")
    return result


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (matrix - mean) / scale, mean, scale


def _ridge(standardized: np.ndarray, targets: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    if weights is not None:
        root = np.sqrt(weights).reshape(-1, 1)
        design = design * root
        targets = targets * root
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    fitted = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return fitted[0], fitted[1:]


def positive_uplift_weights(uplift: np.ndarray, gamma: float) -> np.ndarray:
    """Return fit-row-only, mean-one positive weights for one uplift head."""

    values = np.asarray(uplift, dtype=float)
    strength = _strength(gamma)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("uplift must be a non-empty finite vector")
    weights = 1.0 + strength * np.maximum(values, 0.0)
    return weights / weights.mean()


def fit_quality_heads(
    matrix: np.ndarray,
    scores: np.ndarray,
    *,
    kind: str,
    strength: float = 0.0,
    regret: np.ndarray | None = None,
) -> LinearScoreHeads:
    """Fit only the pre-registered raw or uplift quality objectives."""

    features = _matrix(matrix, "matrix")
    target = _targets(scores, features.shape[0], "scores")
    strength = _strength(strength)
    if kind not in {"raw", "direct-uplift", "weighted-uplift", "regret-weighted-raw"}:
        raise ValueError("unknown quality objective")
    standardized, mean, scale = _standardize(features)
    if kind == "raw":
        intercept, coefficients = _ridge(standardized, target)
    elif kind == "direct-uplift":
        light_intercept, light_coefficients = _ridge(standardized, target[:, [0]])
        delta_intercept, delta_coefficients = _ridge(
            standardized, target[:, 1:] - target[:, [0]]
        )
        intercept = np.concatenate((light_intercept, light_intercept + delta_intercept))
        coefficients = np.column_stack(
            (light_coefficients[:, 0], light_coefficients + delta_coefficients)
        )
    elif kind == "weighted-uplift":
        light_intercept, light_coefficients = _ridge(standardized, target[:, [0]])
        delta = target[:, 1:] - target[:, [0]]
        upgrade_intercepts = []
        upgrade_coefficients = []
        for column in range(delta.shape[1]):
            intercept, coefficients = _ridge(
                standardized,
                delta[:, [column]],
                positive_uplift_weights(delta[:, column], strength),
            )
            upgrade_intercepts.append(intercept[0])
            upgrade_coefficients.append(coefficients[:, 0])
        intercept = np.asarray([light_intercept[0], *(light_intercept[0] + item for item in upgrade_intercepts)])
        coefficients = np.column_stack(
            (light_coefficients[:, 0], *(light_coefficients[:, 0] + item for item in upgrade_coefficients))
        )
    else:
        if regret is None:
            raise ValueError("regret is required for regret-weighted-raw")
        regret_rows = _targets(regret, features.shape[0], "regret")
        means = regret_rows.mean(axis=0)
        weights = np.ones_like(regret_rows) if np.all(means == 0.0) else 1.0 + strength * regret_rows / np.where(means > 0.0, means, 1.0)
        weights /= weights.mean(axis=0)
        intercepts = []
        coefficients_by_head = []
        for column in range(3):
            intercept, coefficients = _ridge(standardized, target[:, [column]], weights[:, column])
            intercepts.append(intercept[0])
            coefficients_by_head.append(coefficients[:, 0])
        intercept = np.asarray(intercepts)
        coefficients = np.column_stack(coefficients_by_head)
    return LinearScoreHeads(mean, scale, intercept, coefficients)


def fit_log_cost_heads(matrix: np.ndarray, log_costs: np.ndarray) -> LinearScoreHeads:
    """Fit the unchanged ordinary ridge log-cost family."""

    features = _matrix(matrix, "matrix")
    targets = _targets(log_costs, features.shape[0], "log_costs")
    standardized, mean, scale = _standardize(features)
    intercept, coefficients = _ridge(standardized, targets)
    return LinearScoreHeads(mean, scale, intercept, coefficients)


def predict_heads(heads: LinearScoreHeads, matrix: np.ndarray) -> np.ndarray:
    features = _matrix(matrix, "matrix")
    if features.shape[1] != heads.mean.shape[0]:
        raise ValueError("matrix feature count does not match fitted heads")
    return ((features - heads.mean) / heads.scale) @ heads.coefficients + heads.intercept
