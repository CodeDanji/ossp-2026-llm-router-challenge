"""Train-only linear heads and cost calibration for the nested hash-regex evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from numbers import Integral
from typing import Sequence

import numpy as np

import hash_regex
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    InputBatch,
    Outcome,
    OutcomeBatch,
    RoutingPolicy,
    Submission,
)
from ossp_router.scoring import score_submissions
from promptbudget.input_adapter import to_prompt_record
from promptbudget.safety import canonical_content_group, grouped_folds


RIDGE_ALPHA = 100.0
HASH_BINS = 256


def _immutable(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LinearHeads:
    mean: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mean", "scale", "intercept", "coefficients"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


@dataclass(frozen=True)
class EvaluationData:
    matrix: np.ndarray
    groups: tuple[str, ...]
    scores: np.ndarray
    log_costs: np.ndarray
    costs: np.ndarray
    policy: RoutingPolicy
    inputs: InputBatch
    outcomes: OutcomeBatch

    def __post_init__(self) -> None:
        for name in ("matrix", "scores", "log_costs", "costs"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


def _matrix(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or not result.shape[0] or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a non-empty finite matrix")
    return result


def _targets(value: np.ndarray, rows: int, label: str) -> np.ndarray:
    result = _matrix(value, label)
    if result.shape != (rows, 3):
        raise ValueError(f"{label} must have shape (rows, 3)")
    return result


def _fit(matrix: np.ndarray, targets: np.ndarray) -> LinearHeads:
    mean = matrix.mean(axis=0)
    scale = np.where(matrix.std(axis=0) > 0.0, matrix.std(axis=0), 1.0)
    standardized = (matrix - mean) / scale
    design = np.column_stack((np.ones(len(matrix)), standardized))
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    fitted = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return LinearHeads(mean, scale, fitted[0], fitted[1:])


def fit_raw_quality_heads(matrix: np.ndarray, scores: np.ndarray) -> LinearHeads:
    features = _matrix(matrix, "matrix")
    return _fit(features, _targets(scores, len(features), "scores"))


def fit_log_cost_heads(matrix: np.ndarray, log_costs: np.ndarray) -> LinearHeads:
    features = _matrix(matrix, "matrix")
    return _fit(features, _targets(log_costs, len(features), "log_costs"))


def predict_heads(heads: LinearHeads, matrix: np.ndarray) -> np.ndarray:
    features = _matrix(matrix, "matrix")
    if features.shape[1] != len(heads.mean):
        raise ValueError("matrix feature count does not match fitted heads")
    return ((features - heads.mean) / heads.scale) @ heads.coefficients + heads.intercept


def _outcome_cost(outcome: Outcome, policy: RoutingPolicy) -> float:
    rates = policy.models[outcome.model_id]
    value = rates.fixed_cost + (
        Decimal(outcome.input_tokens) * rates.input_token_rate
        + Decimal(outcome.output_tokens) * rates.output_token_rate
    ) / Decimal(policy.token_unit)
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("outcome costs must be finite and positive")
    return result


def make_evaluation_data(inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy) -> EvaluationData:
    if inputs.split != "train" or outcomes.split != "train":
        raise ValueError("nested evaluation accepts Train batches only")
    if (inputs.schema_version, inputs.challenge_id, inputs.split) != (outcomes.schema_version, outcomes.challenge_id, outcomes.split):
        raise ValueError("input and outcome metadata must match")
    index = {(row.episode_id, row.model_id): row for row in outcomes.outcomes}
    expected = {(episode.episode_id, model_id) for episode in inputs.episodes for model_id in MODEL_IDS}
    if set(index) != expected:
        raise ValueError("outcome matrix must cover every Train episode and model")
    matrix = np.asarray([hash_regex.raw_feature_vector(episode, HASH_BINS) for episode in inputs.episodes], dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 270:
        raise ValueError("raw feature vector must have 270 dimensions")
    costs = np.asarray([[_outcome_cost(index[(episode.episode_id, model)], policy) for model in MODEL_IDS] for episode in inputs.episodes])
    scores = np.asarray([[float(index[(episode.episode_id, model)].score) for model in MODEL_IDS] for episode in inputs.episodes])
    return EvaluationData(matrix, tuple(canonical_content_group(to_prompt_record(e).text) for e in inputs.episodes), scores, np.log(costs), costs, policy, inputs, outcomes)


def apply_cost_multipliers(costs: np.ndarray, *, ax31: float, think: float) -> np.ndarray:
    result = _targets(costs, np.asarray(costs).shape[0] if np.asarray(costs).ndim == 2 else 0, "costs").copy()
    if np.any(result <= 0.0):
        raise ValueError("costs must be finite and positive")
    for name, value in (("ax31", ax31), ("think", think)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} multiplier must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        result[:, 1] *= float(ax31)
        result[:, 2] *= float(think)
    if not np.isfinite(result).all():
        raise ValueError("scaled costs must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        result[:, 1] = np.maximum(result[:, 1], result[:, 0] * (1.0 + 1e-12))
        result[:, 2] = np.maximum(result[:, 2], result[:, 1] * (1.0 + 1e-12))
    if not np.isfinite(result).all():
        raise ValueError("ordered costs must be finite")
    return result


def _slice_batches(
    data: EvaluationData, indices: tuple[int, ...]
) -> tuple[InputBatch, OutcomeBatch]:
    episodes = tuple(data.inputs.episodes[index] for index in indices)
    identifiers = {episode.episode_id for episode in episodes}
    outcomes = tuple(
        outcome
        for outcome in data.outcomes.outcomes
        if outcome.episode_id in identifiers
    )
    return (
        InputBatch(
            data.inputs.schema_version,
            data.inputs.challenge_id,
            data.inputs.split,
            episodes,
        ),
        OutcomeBatch(
            data.outcomes.schema_version,
            data.outcomes.challenge_id,
            data.outcomes.split,
            outcomes,
        ),
    )


def _validated_indices(
    data: EvaluationData, indices: tuple[int, ...]
) -> tuple[int, ...]:
    selected = tuple(indices)
    if not selected:
        raise ValueError("indices must be non-empty")
    if any(isinstance(index, bool) or not isinstance(index, Integral) for index in selected):
        raise ValueError("indices must be integers")
    selected = tuple(int(index) for index in selected)
    if len(set(selected)) != len(selected):
        raise ValueError("indices must be unique")
    if any(index < 0 or index >= len(data.inputs.episodes) for index in selected):
        raise ValueError("indices must reference evaluation episodes")
    return selected


def _prediction_rows(
    scores: np.ndarray, log_costs: np.ndarray, upgrade_multipliers: tuple[float, float]
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    score_array = _targets(scores, len(scores), "scores")
    log_cost_array = _targets(log_costs, len(score_array), "log_costs")
    costs = np.exp(np.clip(log_cost_array, -50.0, 50.0))
    costs[:, 1] = np.maximum(costs[:, 1], costs[:, 0] * (1.0 + 1e-12))
    costs[:, 2] = np.maximum(costs[:, 2], costs[:, 1] * (1.0 + 1e-12))
    costs = apply_cost_multipliers(
        costs, ax31=upgrade_multipliers[0], think=upgrade_multipliers[1]
    )
    return (
        [
            {
                model_id: min(1.0, max(0.0, float(score[index])))
                for index, model_id in enumerate(MODEL_IDS)
            }
            for score in score_array
        ],
        [
            {model_id: float(cost[index]) for index, model_id in enumerate(MODEL_IDS)}
            for cost in costs
        ],
    )


def score_batch_policy(
    data: EvaluationData,
    indices: tuple[int, ...],
    scores: np.ndarray,
    log_costs: np.ndarray,
    upgrade_multipliers: tuple[float, float],
) -> dict[str, object]:
    """Route every tier and score the final submissions through the official scorer."""

    selected_indices = _validated_indices(data, indices)
    score_rows, cost_rows = _prediction_rows(
        scores, log_costs, upgrade_multipliers
    )
    if len(selected_indices) != len(score_rows):
        raise ValueError("prediction rows and indices must align")

    selected: dict[str, tuple[str, ...]] = {}
    predicted_ratios: dict[str, float] = {}
    for tier in TIERS:
        choices, predicted_ratio = hash_regex.select_models(
            score_rows,
            cost_rows,
            budget_multiplier=float(data.policy.tiers[tier].budget_multiplier),
            safety_ratio=1.0,
        )
        if tier == "premium":
            filled, predicted_ratio = hash_regex.fill_ax31_upgrades(
                choices,
                score_rows,
                cost_rows,
                budget_multiplier=float(data.policy.tiers[tier].budget_multiplier),
                safety_ratio=hash_regex.PREMIUM_AX31_FILL_SAFETY_RATIO,
            )
            choices = filled
        selected[tier] = choices
        predicted_ratios[tier] = predicted_ratio

    inputs, outcomes = _slice_batches(data, selected_indices)
    submissions = tuple(
        Submission(
            schema_version=inputs.schema_version,
            challenge_id=inputs.challenge_id,
            policy_id=data.policy.policy_id,
            split=inputs.split,
            tier=tier,
            decisions=tuple(
                Decision(episode.episode_id, model_id)
                for episode, model_id in zip(inputs.episodes, selected[tier])
            ),
        )
        for tier in TIERS
    )
    report = score_submissions(inputs, outcomes, submissions, data.policy)
    report["tiers"] = {
        tier: {
            **report["tiers"][tier],
            "actual_ratio": report["tiers"][tier]["budget_ratio"],
            "predicted_ratio": predicted_ratios[tier],
            "includes_premium_fill": tier == "premium",
        }
        for tier in TIERS
    }
    return report


def admit_inner_candidate(reports: Sequence[dict[str, object]]) -> dict[str, object]:
    """Require every official tier check across the four routed inner batches."""

    if len(reports) != 4:
        raise ValueError("inner admission requires exactly four batch reports")
    checks = []
    points = 0.0
    rows = 0
    for fold, report in enumerate(reports):
        tiers = report["tiers"]
        fold_rows = int(tiers["fast"]["num_episodes"])
        if fold_rows <= 0 or any(
            int(tiers[tier]["num_episodes"]) != fold_rows for tier in TIERS
        ):
            raise ValueError("inner reports require aligned positive tier row counts")
        points += float(report["final_weighted_points_total"])
        rows += fold_rows
        for tier in TIERS:
            metrics = tiers[tier]
            checks.append(
                {
                    "fold": fold,
                    "tier": tier,
                    "budget_passed": bool(metrics["budget_passed"]),
                    "actual_ratio": float(metrics["actual_ratio"]),
                }
            )
    passed_checks = sum(check["budget_passed"] for check in checks)
    return {
        "admitted": passed_checks == len(checks),
        "passed_checks": passed_checks,
        "required_checks": len(checks),
        "checks": tuple(checks),
        "maximum_actual_ratio": max(check["actual_ratio"] for check in checks),
        "points": points,
        "rows": rows,
        "official_score": points / rows,
    }


def evaluate_inner_pair(
    data: EvaluationData,
    outer_train: Sequence[int],
    seed: int = 137,
    pair: tuple[float, float] = (1.10, 1.25),
) -> dict[str, object]:
    """Cross-fit one multiplier pair and route each grouped validation batch once."""

    selected = _validated_indices(data, tuple(outer_train))
    local_groups = tuple(data.groups[index] for index in selected)
    reports = []
    inner_folds = []
    for fold in grouped_folds(local_groups, folds=4, seed=seed):
        train_indices = tuple(selected[index] for index in fold.train_indices)
        validation_indices = tuple(selected[index] for index in fold.validation_indices)
        quality = fit_raw_quality_heads(
            data.matrix[list(train_indices)], data.scores[list(train_indices)]
        )
        costs = fit_log_cost_heads(
            data.matrix[list(train_indices)], data.log_costs[list(train_indices)]
        )
        report = score_batch_policy(
            data,
            validation_indices,
            predict_heads(quality, data.matrix[list(validation_indices)]),
            predict_heads(costs, data.matrix[list(validation_indices)]),
            pair,
        )
        reports.append(report)
        inner_folds.append(
            {
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "report": report,
            }
        )
    return {
        "pair": pair,
        "inner_folds": tuple(inner_folds),
        "pooled_for_routing": False,
        "admission": admit_inner_candidate(reports),
    }


def select_inner_multiplier(
    data: EvaluationData, outer_train: Sequence[int], seed: int
) -> dict[str, object]:
    """Pick the admitted multiplier pair using only grouped outer-train folds."""

    candidates = []
    for ax31 in (1.00, 1.10, 1.25):
        for think in (1.00, 1.25, 1.50):
            evaluation = evaluate_inner_pair(data, outer_train, seed, (ax31, think))
            admission = evaluation["admission"]
            if admission["admitted"]:
                candidates.append((
                    -float(admission["official_score"]),
                    float(admission["maximum_actual_ratio"]),
                    ax31,
                    think,
                    evaluation,
                ))
    if not candidates:
        return {"status": "no-admitted-multiplier", "pair": None, "route": "all-light"}
    chosen = min(candidates)
    return {
        "status": "admitted-multiplier",
        "pair": (chosen[2], chosen[3]),
        "route": "cost-multiplied",
        "admission": chosen[4]["admission"],
        "inner_folds": chosen[4]["inner_folds"],
        "pooled_for_routing": False,
    }
