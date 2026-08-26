# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Pure Train-only helpers for the fixed hash-regex quality experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from decimal import Decimal
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


@dataclass(frozen=True)
class EvaluationData:
    """The single Train-only materialization used by every fixed candidate."""

    matrix: np.ndarray
    groups: tuple[str, ...]
    scores: np.ndarray
    log_costs: np.ndarray
    costs: np.ndarray
    policy: RoutingPolicy
    inputs: InputBatch
    outcomes: OutcomeBatch


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


def _outcome_cost(outcome: Outcome, policy: RoutingPolicy) -> float:
    rates = policy.models[outcome.model_id]
    cost = rates.fixed_cost + (
        Decimal(outcome.input_tokens) * rates.input_token_rate
        + Decimal(outcome.output_tokens) * rates.output_token_rate
    ) / Decimal(policy.token_unit)
    result = float(cost)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("outcome costs must be finite and positive")
    return result


def make_evaluation_data(
    inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy
) -> EvaluationData:
    """Materialize only the current Train inputs and complete outcome matrix once."""

    if inputs.split != "train" or outcomes.split != "train":
        raise ValueError("nested evaluation accepts Train batches only")
    if (
        inputs.schema_version != outcomes.schema_version
        or inputs.challenge_id != outcomes.challenge_id
        or inputs.split != outcomes.split
    ):
        raise ValueError("input and outcome metadata must match")
    outcome_index = {(row.episode_id, row.model_id): row for row in outcomes.outcomes}
    expected = {(episode.episode_id, model_id) for episode in inputs.episodes for model_id in MODEL_IDS}
    if set(outcome_index) != expected:
        raise ValueError("outcome matrix must cover every Train episode and model")
    matrix = np.asarray(
        [hash_regex.raw_feature_vector(episode, HASH_BINS) for episode in inputs.episodes],
        dtype=float,
    )
    costs = np.asarray(
        [
            [_outcome_cost(outcome_index[(episode.episode_id, model_id)], policy) for model_id in MODEL_IDS]
            for episode in inputs.episodes
        ],
        dtype=float,
    )
    scores = np.asarray(
        [
            [float(outcome_index[(episode.episode_id, model_id)].score) for model_id in MODEL_IDS]
            for episode in inputs.episodes
        ],
        dtype=float,
    )
    return EvaluationData(
        matrix=matrix,
        groups=tuple(canonical_content_group(to_prompt_record(episode).text) for episode in inputs.episodes),
        scores=scores,
        log_costs=np.log(costs),
        costs=costs,
        policy=policy,
        inputs=inputs,
        outcomes=outcomes,
    )


def require_group_disjoint(
    train_indices: Sequence[int], test_indices: Sequence[int], groups: Sequence[str]
) -> None:
    if not train_indices or not test_indices:
        raise ValueError("train and test indices must be non-empty")
    if any(index < 0 or index >= len(groups) for index in (*train_indices, *test_indices)):
        raise ValueError("split index is out of bounds")
    if set(train_indices) & set(test_indices):
        raise ValueError("split indices overlap")
    if {groups[index] for index in train_indices} & {groups[index] for index in test_indices}:
        raise ValueError("content group crosses train and test")


def grouped_inner_folds(groups: Sequence[str], seed: int):
    return grouped_folds(tuple(groups), folds=4, seed=seed)


def lagrange_selection_and_penalty(
    scores: np.ndarray, costs: np.ndarray, multiplier: float
) -> tuple[tuple[int, ...], float]:
    """Mirror the runtime selector and expose its final batch-unit penalty."""

    score_array = _targets(scores, _matrix(scores, "scores").shape[0], "scores")
    cost_array = _targets(costs, score_array.shape[0], "costs")
    if not math.isfinite(multiplier) or multiplier <= 0.0 or np.any(cost_array <= 0.0):
        raise ValueError("multiplier and costs must be positive and finite")
    light_total = float(cost_array[:, 0].sum())
    cap = light_total * max(1.0, float(multiplier))

    def choose(penalty: float) -> tuple[tuple[int, ...], float]:
        utility = score_array - penalty * cost_array / light_total
        selected = tuple(int(np.argmax(row)) for row in utility)
        return selected, float(sum(cost_array[index, model] for index, model in enumerate(selected)))

    selected, total = choose(0.0)
    penalty = 0.0
    if total > cap:
        low, high = 0.0, 1.0
        selected, total = choose(high)
        while total > cap and high < 2**60:
            low, high = high, high * 2.0
            selected, total = choose(high)
        for _ in range(80):
            middle = (low + high) / 2.0
            candidate, candidate_total = choose(middle)
            if candidate_total <= cap:
                high, selected, total = middle, candidate, candidate_total
            else:
                low = middle
        penalty = high
    if total > cap:
        selected = tuple(0 for _ in range(len(score_array)))
    return selected, penalty


def lagrange_selections_for_safety_grid(
    scores: np.ndarray,
    costs: np.ndarray,
    multiplier: float,
    safety_ratios: Sequence[float],
) -> tuple[tuple[int, ...], ...]:
    """Vectorized evaluator-only form of the runtime's fixed 80-step selector."""

    score_array = _targets(scores, _matrix(scores, "scores").shape[0], "scores")
    cost_array = _targets(costs, score_array.shape[0], "costs")
    safety = np.asarray(safety_ratios, dtype=float)
    if not len(safety) or not np.isfinite(safety).all() or np.any((safety <= 0.0) | (safety > 1.0)):
        raise ValueError("safety ratios must be finite values in (0, 1]")
    if not math.isfinite(multiplier) or multiplier <= 0.0 or np.any(cost_array <= 0.0):
        raise ValueError("multiplier and costs must be positive and finite")
    light_total = float(cost_array[:, 0].sum())
    caps = light_total * np.maximum(1.0, float(multiplier) * safety)

    def choose(penalties: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        utility = score_array[None, :, :] - penalties[:, None, None] * cost_array[None, :, :] / light_total
        selected = utility.argmax(axis=2)
        totals = np.take_along_axis(cost_array[None, :, :], selected[:, :, None], axis=2).sum(axis=(1, 2))
        return selected, totals

    selected, totals = choose(np.zeros(len(safety)))
    unresolved = totals > caps
    low, high = np.zeros(len(safety)), np.ones(len(safety))
    selected_high, totals_high = choose(high)
    while np.any(unresolved & (totals_high > caps) & (high < 2**60)):
        needs_more = unresolved & (totals_high > caps) & (high < 2**60)
        low[needs_more] = high[needs_more]
        high[needs_more] *= 2.0
        selected_high, totals_high = choose(high)
    selected[unresolved] = selected_high[unresolved]
    totals[unresolved] = totals_high[unresolved]
    for _ in range(80):
        middle = (low + high) / 2.0
        candidate, candidate_totals = choose(middle)
        feasible = unresolved & (candidate_totals <= caps)
        high[feasible] = middle[feasible]
        selected[feasible] = candidate[feasible]
        totals[feasible] = candidate_totals[feasible]
        infeasible = unresolved & ~feasible
        low[infeasible] = middle[infeasible]
    selected[totals > caps] = 0
    return tuple(tuple(int(model) for model in row) for row in selected)


def tier_regrets(
    scores: np.ndarray, costs: np.ndarray, policy: RoutingPolicy
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute weighted per-action regret from Train outcomes in runtime units."""

    score_array = _matrix(scores, "scores")
    cost_array = _targets(costs, score_array.shape[0], "costs")
    if score_array.shape[1] != 3:
        raise ValueError("scores must have shape (rows, 3)")
    light_total = float(cost_array[:, 0].sum())
    regrets = np.zeros_like(score_array)
    penalties: dict[str, float] = {}
    for tier, tier_policy in policy.tiers.items():
        _selected, penalty = lagrange_selection_and_penalty(
            score_array, cost_array, float(tier_policy.budget_multiplier)
        )
        penalties[tier] = penalty
        utility = score_array - penalty * cost_array / light_total
        regrets += float(tier_policy.weight) * (utility.max(axis=1, keepdims=True) - utility)
    return np.maximum(regrets, 0.0), penalties


def _prediction_rows(scores: np.ndarray, log_costs: np.ndarray) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    score_rows, cost_rows = [], []
    for score, log_cost in zip(scores, log_costs):
        score_row = {model_id: min(1.0, max(0.0, float(score[index]))) for index, model_id in enumerate(MODEL_IDS)}
        cost_row = {model_id: math.exp(min(50.0, max(-50.0, float(log_cost[index])))) for index, model_id in enumerate(MODEL_IDS)}
        cost_row[MODEL_IDS[1]] = max(cost_row[MODEL_IDS[1]], cost_row[MODEL_IDS[0]] * (1.0 + 1e-12))
        cost_row[MODEL_IDS[2]] = max(cost_row[MODEL_IDS[2]], cost_row[MODEL_IDS[1]] * (1.0 + 1e-12))
        score_rows.append(score_row)
        cost_rows.append(cost_row)
    return score_rows, cost_rows


def route_complete_policy(
    score_rows: Sequence[dict[str, float]],
    cost_rows: Sequence[dict[str, float]],
    policy: RoutingPolicy,
    tier: str,
    safety_ratio: float,
) -> tuple[tuple[str, ...], float]:
    """Route through the unchanged runtime allocator, including Premium fill."""

    selected, ratio = hash_regex.select_models(
        score_rows,
        cost_rows,
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=safety_ratio,
    )
    if tier == "premium":
        selected, ratio = hash_regex.fill_ax31_upgrades(
            selected,
            score_rows,
            cost_rows,
            budget_multiplier=float(policy.tiers[tier].budget_multiplier),
            safety_ratio=hash_regex.PREMIUM_AX31_FILL_SAFETY_RATIO,
        )
    return selected, ratio


def _slice_batches(data: EvaluationData, indices: Sequence[int]) -> tuple[InputBatch, OutcomeBatch]:
    episodes = tuple(data.inputs.episodes[index] for index in indices)
    identifiers = {episode.episode_id for episode in episodes}
    outcomes = tuple(row for row in data.outcomes.outcomes if row.episode_id in identifiers)
    return (
        InputBatch(data.inputs.schema_version, data.inputs.challenge_id, data.inputs.split, episodes),
        OutcomeBatch(data.outcomes.schema_version, data.outcomes.challenge_id, data.outcomes.split, outcomes),
    )


def _score_complete_policy(
    data: EvaluationData,
    indices: Sequence[int],
    scores: np.ndarray,
    log_costs: np.ndarray,
    safety_ratios: dict[str, float],
) -> dict[str, object]:
    score_rows, cost_rows = _prediction_rows(scores, log_costs)
    selected = {}
    predicted_ratios = {}
    for tier in TIERS:
        selected[tier], predicted_ratios[tier] = route_complete_policy(
            score_rows, cost_rows, data.policy, tier, safety_ratios[tier]
        )
    inputs, outcomes = _slice_batches(data, indices)
    submissions = tuple(
        Submission(
            schema_version=inputs.schema_version,
            challenge_id=inputs.challenge_id,
            policy_id=data.policy.policy_id,
            split=inputs.split,
            tier=tier,
            decisions=tuple(Decision(episode.episode_id, model_id) for episode, model_id in zip(inputs.episodes, selected[tier])),
        )
        for tier in TIERS
    )
    report = score_submissions(inputs, outcomes, submissions, data.policy)
    return {"score": report, "selected": selected, "predicted_ratios": predicted_ratios}


def _score_one_tier(
    data: EvaluationData,
    indices: Sequence[int],
    score_rows: Sequence[dict[str, float]],
    cost_rows: Sequence[dict[str, float]],
    tier: str,
    safety: float,
) -> dict[str, object]:
    selected, predicted_ratio = route_complete_policy(score_rows, cost_rows, data.policy, tier, safety)
    return {"metrics": _score_selected_tier(data, indices, tier, selected), "predicted_ratio": predicted_ratio}


def _score_selected_tier(
    data: EvaluationData,
    indices: Sequence[int],
    tier: str,
    selected: Sequence[str],
) -> dict[str, object]:
    inputs, outcomes = _slice_batches(data, indices)
    all_light = tuple(MODEL_IDS[0] for _episode in inputs.episodes)
    submissions = tuple(
        Submission(
            inputs.schema_version,
            inputs.challenge_id,
            data.policy.policy_id,
            inputs.split,
            candidate,
            tuple(Decision(episode.episode_id, model_id) for episode, model_id in zip(inputs.episodes, selected if candidate == tier else all_light)),
        )
        for candidate in TIERS
    )
    return score_submissions(inputs, outcomes, submissions, data.policy)["tiers"][tier]


def _safety_grid(policy: RoutingPolicy, tier: str) -> tuple[float, ...]:
    minimum = 1.0 / float(policy.tiers[tier].budget_multiplier)
    return tuple(minimum + (1.0 - minimum) * index / 120.0 for index in range(121))


def select_provisional_safety(
    score_rows: np.ndarray,
    log_cost_rows: np.ndarray,
    data: EvaluationData,
    tier: str,
    indices: Sequence[int] | None = None,
) -> dict[str, object]:
    """Select one Train-only safety value using the actual official tier result."""

    selected_indices = tuple(range(len(score_rows))) if indices is None else tuple(indices)
    if len(score_rows) != len(selected_indices) or len(log_cost_rows) != len(selected_indices):
        raise ValueError("prediction rows and scoring indices must align")
    score_mappings, cost_mappings = _prediction_rows(score_rows, log_cost_rows)
    safety_grid = _safety_grid(data.policy, tier)
    numeric_scores = np.asarray([[row[model_id] for model_id in MODEL_IDS] for row in score_mappings])
    numeric_costs = np.asarray([[row[model_id] for model_id in MODEL_IDS] for row in cost_mappings])
    batched = lagrange_selections_for_safety_grid(
        numeric_scores,
        numeric_costs,
        float(data.policy.tiers[tier].budget_multiplier),
        safety_grid,
    )
    best = None
    metric_cache: dict[tuple[str, ...], dict[str, object]] = {}
    for safety, selected_indices_by_row in zip(safety_grid, batched):
        selected = tuple(MODEL_IDS[index] for index in selected_indices_by_row)
        if tier == "premium":
            selected, _ratio = hash_regex.fill_ax31_upgrades(
                selected,
                score_mappings,
                cost_mappings,
                budget_multiplier=float(data.policy.tiers[tier].budget_multiplier),
                safety_ratio=hash_regex.PREMIUM_AX31_FILL_SAFETY_RATIO,
            )
        metrics = metric_cache.get(selected)
        if metrics is None:
            metrics = _score_selected_tier(data, selected_indices, tier, selected)
            metric_cache[selected] = metrics
        rank = (
            bool(metrics["budget_passed"]),
            float(metrics["tier_score"]),
            -float(metrics["budget_ratio"]),
            -safety,
        )
        if best is None or rank > best[0]:
            best = (rank, safety, metrics)
    assert best is not None
    return {
        "safety_ratio": best[1],
        "metrics": best[2],
        "includes_premium_fill": tier == "premium",
    }


def _inner_prediction_bundle(data: EvaluationData, outer_train: Sequence[int], seed: int) -> dict[str, object]:
    """Cross-fit immutable cost predictions once and all fixed quality choices."""

    outer_train = tuple(outer_train)
    local_groups = tuple(data.groups[index] for index in outer_train)
    cost_oof = np.empty((len(outer_train), 3), dtype=float)
    score_oof = {("raw", 0.0): np.empty((len(outer_train), 3), dtype=float)}
    for strength in GAMMA_GRID:
        score_oof[("weighted-uplift", strength)] = np.empty((len(outer_train), 3), dtype=float)
    for strength in ETA_GRID:
        score_oof[("regret-weighted-raw", strength)] = np.empty((len(outer_train), 3), dtype=float)
    for fold in grouped_inner_folds(local_groups, seed):
        local_train, local_validation = fold.train_indices, fold.validation_indices
        train_indices = tuple(outer_train[index] for index in local_train)
        validation_indices = tuple(outer_train[index] for index in local_validation)
        cost_heads = fit_log_cost_heads(data.matrix[list(train_indices)], data.log_costs[list(train_indices)])
        cost_oof[list(local_validation)] = predict_heads(cost_heads, data.matrix[list(validation_indices)])
        regrets, _penalties = tier_regrets(data.scores[list(train_indices)], data.costs[list(train_indices)], data.policy)
        for (kind, strength), prediction in score_oof.items():
            heads = fit_quality_heads(
                data.matrix[list(train_indices)],
                data.scores[list(train_indices)],
                kind=kind,
                strength=strength,
                regret=regrets if kind == "regret-weighted-raw" else None,
            )
            prediction[list(local_validation)] = predict_heads(heads, data.matrix[list(validation_indices)])
    _regret, penalties = tier_regrets(data.scores[list(outer_train)], data.costs[list(outer_train)], data.policy)
    return {"cost_oof": cost_oof, "score_oof": score_oof, "lambdas": penalties}


def inner_oof_bundle(data: EvaluationData, outer_train: Sequence[int], seed: int) -> dict[str, object]:
    require_group_disjoint(outer_train, tuple(index for index in range(len(data.groups)) if index not in set(outer_train)), data.groups)
    return _inner_prediction_bundle(data, outer_train, seed)


def _light_policy_report(data: EvaluationData, indices: Sequence[int]) -> dict[str, object]:
    inputs, outcomes = _slice_batches(data, indices)
    submissions = tuple(
        Submission(inputs.schema_version, inputs.challenge_id, data.policy.policy_id, inputs.split, tier, tuple(Decision(episode.episode_id, MODEL_IDS[0]) for episode in inputs.episodes))
        for tier in TIERS
    )
    return score_submissions(inputs, outcomes, submissions, data.policy)


def select_inner_configuration(data: EvaluationData, outer_train: Sequence[int], seed: int) -> dict[str, dict[str, object]]:
    """Choose each candidate only from its grouped inner OOF predictions."""

    bundle = inner_oof_bundle(data, outer_train, seed)
    result: dict[str, dict[str, object]] = {}
    grids = {"raw": (0.0,), "weighted-uplift": GAMMA_GRID, "regret-weighted-raw": ETA_GRID}
    for candidate in QUALITY_CANDIDATES:
        choices = []
        for strength in grids[candidate]:
            scores = bundle["score_oof"][(candidate, strength)]
            safety = {
                tier: select_provisional_safety(scores, bundle["cost_oof"], data, tier, outer_train)["safety_ratio"]
                for tier in TIERS
            }
            policy_report = _score_complete_policy(data, outer_train, scores, bundle["cost_oof"], safety)["score"]
            tiers = policy_report["tiers"]
            admitted = all(tiers[tier]["budget_passed"] for tier in TIERS)
            choices.append((admitted, float(policy_report["final_score"]), max(float(tiers[tier]["budget_ratio"]) for tier in TIERS), strength, safety, policy_report))
        admitted = [choice for choice in choices if choice[0]]
        if admitted:
            chosen = min(admitted, key=lambda choice: (-choice[1], choice[2], choice[3]))
            result[candidate] = {"admitted": True, "strength": chosen[3], "safety": chosen[4], "metrics": chosen[5], "lambdas": bundle["lambdas"]}
        else:
            result[candidate] = {"admitted": False, "strength": 0.0, "safety": {tier: 1.0 for tier in TIERS}, "metrics": _light_policy_report(data, outer_train), "lambdas": bundle["lambdas"], "fallback": "all-light"}
    return result


def _fit_outer_heads(data: EvaluationData, indices: Sequence[int], kind: str, strength: float) -> tuple[LinearScoreHeads, LinearScoreHeads]:
    regrets, _lambdas = tier_regrets(data.scores[list(indices)], data.costs[list(indices)], data.policy)
    quality = fit_quality_heads(data.matrix[list(indices)], data.scores[list(indices)], kind=kind, strength=strength, regret=regrets if kind == "regret-weighted-raw" else None)
    return quality, fit_log_cost_heads(data.matrix[list(indices)], data.log_costs[list(indices)])


def evaluate_outer_fold(
    data: EvaluationData, outer_train: Sequence[int], outer_test: Sequence[int], seed: int, fold: int
) -> dict[str, object]:
    require_group_disjoint(outer_train, outer_test, data.groups)
    configurations = select_inner_configuration(data, outer_train, seed)
    candidates: dict[str, dict[str, object]] = {}
    raw_predictions = None
    for candidate in QUALITY_CANDIDATES:
        configuration = configurations[candidate]
        quality, costs = _fit_outer_heads(data, outer_train, candidate, float(configuration["strength"]))
        score_prediction = predict_heads(quality, data.matrix[list(outer_test)])
        cost_prediction = predict_heads(costs, data.matrix[list(outer_test)])
        full = _score_complete_policy(data, outer_test, score_prediction, cost_prediction, configuration["safety"])["score"]
        candidates[candidate] = {"strength": configuration["strength"], "safety": configuration["safety"], "lambdas": configuration["lambdas"], "score": full, "tiers": full["tiers"], "groups": tuple(sorted({data.groups[index] for index in outer_test}))}
        if candidate == "raw":
            raw_predictions = (score_prediction, cost_prediction)
    assert raw_predictions is not None
    direct, direct_costs = _fit_outer_heads(data, outer_train, "direct-uplift", 0.0)
    direct_prediction = predict_heads(direct, data.matrix[list(outer_test)])
    if not np.allclose(direct_prediction, raw_predictions[0], atol=1e-10):
        raise AssertionError("direct uplift control diverged from raw predictions")
    raw_score = float(candidates["raw"]["score"]["final_score"])
    for candidate in QUALITY_CANDIDATES:
        candidates[candidate]["paired_difference_from_raw"] = float(candidates[candidate]["score"]["final_score"]) - raw_score
    return {"seed": seed, "fold": fold, "candidates": candidates, "controls": {"direct_uplift_matches_raw": True}}


def choose_winner(folds: Sequence[dict[str, object]]) -> dict[str, object]:
    """Apply the pre-registered gates; Raw is the fixed fallback."""

    if not folds:
        raise ValueError("at least one outer-fold record is required")
    candidates = []
    for candidate in QUALITY_CANDIDATES[1:]:
        records = [fold["candidates"][candidate] for fold in folds]
        differences = [float(record["paired_difference_from_raw"]) for record in records]
        all_caps = all(record["tiers"][tier]["budget_passed"] for record in records for tier in TIERS)
        seed_differences = []
        for seed in sorted({int(fold["seed"]) for fold in folds}):
            values = [float(fold["candidates"][candidate]["paired_difference_from_raw"]) for fold in folds if int(fold["seed"]) == seed]
            seed_differences.append(sum(values) / len(values))
        gates = {
            "all_tier_caps_pass": all_caps,
            "two_seed_means_positive": sum(value > 0.0 for value in seed_differences) >= 2,
            "eight_fold_differences_positive": sum(value > 0.0 for value in differences) >= 8,
            "mean_improvement_at_least_0_002": sum(differences) / len(differences) >= 0.002,
        }
        candidates.append((candidate, gates, differences, seed_differences, records))
    admitted = [item for item in candidates if all(item[1].values())]
    if not admitted:
        return {"name": "raw", "winner": "raw", "all_gates_pass": False, "candidates": {item[0]: item[1] for item in candidates}, "status": "raw-retained", "fallback_artifact": "baselines/hash-regex-public.v1.json"}
    winner = min(
        admitted,
        key=lambda item: (
            -sum(item[2]) / len(item[2]),
            -min(item[3]),
            max(float(record["tiers"][tier]["budget_ratio"]) for record in item[4] for tier in TIERS),
            QUALITY_CANDIDATES.index(item[0]),
        ),
    )
    return {"name": winner[0], "winner": winner[0], "all_gates_pass": True, "candidates": {item[0]: item[1] for item in candidates}, "status": "admitted"}
