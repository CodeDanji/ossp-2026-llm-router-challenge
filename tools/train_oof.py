# SPDX-License-Identifier: Apache-2.0
"""Train compact PromptBudget ridge heads from public Train data only."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - CLI environment dependent
    np = None

from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    ProtocolError,
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_policy,
    policy_sha256,
)
from promptbudget.artifact import PromptBudgetArtifact, TierSettings, write_artifact
from promptbudget.input_adapter import to_prompt_record
from promptbudget.linear import LinearHead
from promptbudget.safety import OUTER_SEEDS, canonical_content_group, grouped_folds, monetary_cost_multipliers, repeated_outer_folds
from promptbudget.text_features import DENSE_FEATURE_NAMES, extract_features
from validate_data import file_sha256, validate_batches


RESIDUAL_QUANTILE = 0.99
TIER_UPPER_COST_LIMITS = {"fast": 1.15, "balanced": 2.0, "premium": 4.0}


class TierPolicyCandidate:
    """Train-only validation outcomes for one pre-registered tier policy."""

    def __init__(self, settings: TierSettings, fold_scores: Sequence[float], fold_upper_cost_ratios: Sequence[float], fold_upgrade_fractions: Sequence[float], grid_index: int) -> None:
        self.settings = settings
        self.fold_scores = tuple(fold_scores)
        self.fold_upper_cost_ratios = tuple(fold_upper_cost_ratios)
        self.fold_upgrade_fractions = tuple(fold_upgrade_fractions)
        self.grid_index = grid_index
        if not self.fold_scores or not (len(self.fold_scores) == len(self.fold_upper_cost_ratios) == len(self.fold_upgrade_fractions)):
            raise ValueError("tier policy folds must be non-empty and aligned")
        if not all(math.isfinite(value) for values in (self.fold_scores, self.fold_upper_cost_ratios, self.fold_upgrade_fractions) for value in values):
            raise ValueError("tier policy folds must be finite")

    @property
    def average_score(self) -> float:
        return float(mean(self.fold_scores))

    @property
    def average_upgrade_fraction(self) -> float:
        return float(mean(self.fold_upgrade_fractions))


def _tier_gate(tier: str, ratios: Sequence[float]) -> bool:
    if tier not in TIER_UPPER_COST_LIMITS or not ratios:
        raise ValueError("unknown tier or empty cost ratios")
    limit = TIER_UPPER_COST_LIMITS[tier]
    return all(ratio <= limit for ratio in ratios) if tier == "fast" else all(ratio < limit for ratio in ratios)


def choose_tier_policy(tier: str, candidates: Sequence[TierPolicyCandidate]) -> TierPolicyCandidate | None:
    """Choose an admitted policy by score, then the fixed conservative one-SE order."""

    admitted = [candidate for candidate in candidates if _tier_gate(tier, candidate.fold_upper_cost_ratios)]
    if not admitted:
        return None
    best = max(admitted, key=lambda candidate: (candidate.average_score, -candidate.grid_index))
    standard_error = 0.0 if len(best.fold_scores) == 1 else stdev(best.fold_scores) / math.sqrt(len(best.fold_scores))
    eligible = [candidate for candidate in admitted if candidate.average_score >= best.average_score - standard_error]
    return min(
        eligible,
        key=lambda candidate: (
            candidate.average_upgrade_fraction,
            candidate.settings.max_relative_cost,
            -candidate.settings.min_gain_ax31,
            -candidate.settings.lambda_cost,
            candidate.grid_index,
        ),
    )


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required; install tools/requirements-train.txt")


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.hash_dimension not in (2 ** 16, 2 ** 18, 2 ** 20):
        raise ValueError("--hash-dimension must be an allowed feature dimension")
    if args.max_sparse_features <= 0:
        raise ValueError("--max-sparse-features must be positive")
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        raise ValueError("--alpha must be a positive finite number")
    if args.folds != 5:
        raise ValueError("--folds is fixed at 5 for leakage-safe outer CV")


def v21_output_paths(*paths: Path) -> Tuple[Path, ...]:
    """Require every mutable v2.1 artifact path below one dedicated build root."""

    resolved = tuple(Path(path).resolve() for path in paths)
    roots = []
    for path in resolved:
        root = next((parent for parent in (path.parent, *path.parents) if parent.name == "promptbudget-v2.1" and parent.parent.name == "build"), None)
        if root is None:
            raise ValueError("v2.1 outputs must be below build/promptbudget-v2.1")
        roots.append(root)
    if len(set(roots)) != 1:
        raise ValueError("v2.1 outputs must share one build root")
    return resolved


def _feature_rows(inputs: Any, hash_dimension: int) -> Tuple[Any, List[Mapping[int, float]]]:
    dense_rows = []
    sparse_rows = []
    for episode in inputs.episodes:
        vector = extract_features(to_prompt_record(episode).text, hash_dimension)
        dense_rows.append(vector.dense)
        sparse_rows.append(vector.sparse)
    return np.asarray(dense_rows, dtype=np.float64), sparse_rows


def _targets(inputs: Any, outcomes: Any) -> Tuple[Any, Any]:
    index = {(item.episode_id, item.model_id): item for item in outcomes.outcomes}
    rows = []
    actual_outputs = []
    for episode in inputs.episodes:
        model_rows = [index[(episode.episode_id, model_id)] for model_id in MODEL_IDS]
        rows.append(
            [float(item.score) for item in model_rows]
            + [math.log1p(item.output_tokens) for item in model_rows]
            + [math.log1p(model_rows[0].input_tokens)]
        )
        actual_outputs.append([float(item.output_tokens) for item in model_rows])
    return np.asarray(rows, dtype=np.float64), np.asarray(actual_outputs, dtype=np.float64)


def _select_sparse(sparse_rows: Sequence[Mapping[int, float]], target: Any, limit: int) -> List[int]:
    """Choose sparse indices by absolute centered correlation, then index."""

    row_count = len(sparse_rows)
    target_centered = target - target.mean()
    target_norm = float(np.sqrt(np.dot(target_centered, target_centered)))
    stats: Dict[int, List[float]] = {}
    for row, value in zip(sparse_rows, target_centered):
        for index, feature in row.items():
            item = stats.setdefault(index, [0.0, 0.0, 0.0])
            item[0] += feature
            item[1] += feature * feature
            item[2] += feature * float(value)
    ranked = []
    for index, (feature_sum, square_sum, cross_sum) in stats.items():
        variance = square_sum - feature_sum * feature_sum / row_count
        covariance = cross_sum  # target_centered sums to zero.
        denominator = math.sqrt(max(0.0, variance)) * target_norm
        correlation = 0.0 if denominator <= 0.0 else covariance / denominator
        ranked.append((abs(correlation), index))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [index for _score, index in ranked[:limit]]


def _design_matrix(dense: Any, sparse_rows: Sequence[Mapping[int, float]], selected: Sequence[int]) -> Any:
    matrix = np.empty((dense.shape[0], dense.shape[1] + len(selected)), dtype=np.float64)
    matrix[:, : dense.shape[1]] = dense
    sparse_index = {value: position for position, value in enumerate(selected)}
    matrix[:, dense.shape[1] :] = 0.0
    for row_index, row in enumerate(sparse_rows):
        for index, value in row.items():
            position = sparse_index.get(index)
            if position is not None:
                matrix[row_index, dense.shape[1] + position] = value
    return matrix


def _fit_ridge(matrix: Any, targets: Any, alpha: float) -> Tuple[Any, Any]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    target_mean = targets.mean(axis=0)
    centered = targets - target_mean
    rows, columns = standardized.shape
    if rows <= columns:
        coefficients = standardized.T @ np.linalg.solve(
            standardized @ standardized.T + alpha * np.eye(rows), centered
        )
    else:
        coefficients = np.linalg.solve(
            standardized.T @ standardized + alpha * np.eye(columns),
            standardized.T @ centered,
        )
    raw_coefficients = coefficients / scale[:, None]
    intercept = target_mean - mean @ raw_coefficients
    return intercept, raw_coefficients


def _predict(matrix: Any, intercept: Any, coefficients: Any) -> Any:
    return matrix @ coefficients + intercept


def _fit_predict_indices(
    dense: Any,
    sparse_rows: Sequence[Mapping[int, float]],
    targets: Any,
    train_indices: Sequence[int],
    predict_indices: Sequence[int],
    max_sparse_features: int,
    alpha: float,
) -> Tuple[Any, List[int], Any, Any]:
    """Fit all data-derived state on train indices and predict a disjoint set."""

    selected = _select_sparse(
        [sparse_rows[index] for index in train_indices],
        targets[list(train_indices), :3].mean(axis=1),
        max_sparse_features,
    )
    train_matrix = _design_matrix(dense[list(train_indices)], [sparse_rows[index] for index in train_indices], selected)
    predict_matrix = _design_matrix(dense[list(predict_indices)], [sparse_rows[index] for index in predict_indices], selected)
    intercept, coefficients = _fit_ridge(train_matrix, targets[list(train_indices)], alpha)
    return _predict(predict_matrix, intercept, coefficients), selected, intercept, coefficients


def _prepare_fold_matrices(
    dense: Any,
    sparse_rows: Sequence[Mapping[int, float]],
    targets: Any,
    train_indices: Sequence[int],
    predict_indices: Sequence[int],
    feature_counts: Sequence[int],
) -> Mapping[int, Tuple[Any, Any, List[int]]]:
    """Select sparse features once per fold/feature count, independent of alpha."""

    train_sparse = [sparse_rows[index] for index in train_indices]
    predict_sparse = [sparse_rows[index] for index in predict_indices]
    target = targets[list(train_indices), :3].mean(axis=1)
    prepared = {}
    for feature_count in sorted(set(feature_counts)):
        selected = _select_sparse(train_sparse, target, feature_count)
        prepared[feature_count] = (
            _design_matrix(dense[list(train_indices)], train_sparse, selected),
            _design_matrix(dense[list(predict_indices)], predict_sparse, selected),
            selected,
        )
    return prepared


def _candidate_grid(args: argparse.Namespace) -> Tuple[Tuple[int, float], ...]:
    """The pre-registered v2.1 common absolute-head grid."""

    return tuple((features, alpha) for features in (64, 256) for alpha in (1.0, 10.0, 100.0))


def _policy_grid(tier: str) -> Tuple[TierSettings, ...]:
    values = {
        "fast": ((0.5, 1.0, 2.0), (0.05, 0.10), (1.0, 1.25)),
        "balanced": ((0.05, 0.10, 0.20), (0.0, 0.05), (2.0, 4.0)),
        "premium": ((0.0, 0.01, 0.05), (0.0, 0.05), (4.0, 10.0)),
    }
    try:
        lambdas, gains, costs = values[tier]
    except KeyError as error:
        raise ValueError("unknown tier") from error
    return tuple(TierSettings(lambda_cost, gain, gain, 1.0, maximum) for lambda_cost in lambdas for gain in gains for maximum in costs)


def _fallback_settings() -> TierSettings:
    return TierSettings(100.0, 1.0, 1.0, 1.0, 1.0)


def _actual_costs(inputs: Any, outcomes: Any, policy: Any) -> Any:
    index = {(item.episode_id, item.model_id): item for item in outcomes.outcomes}
    rows = []
    for episode in inputs.episodes:
        values = []
        for model_id in MODEL_IDS:
            outcome, rates = index[(episode.episode_id, model_id)], policy.models[model_id]
            values.append(float(rates.fixed_cost) + (float(rates.input_token_rate) * outcome.input_tokens + float(rates.output_token_rate) * outcome.output_tokens) / policy.token_unit)
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def _cost_calibration(predictions: Any, actual_costs: Any, character_counts: Sequence[int], policy: Any) -> Mapping[str, Mapping[str, float]]:
    input_tokens = np.maximum(1.0, np.expm1(np.clip(predictions[:, 6], -50.0, 50.0)))
    result = {}
    for position, model_id in enumerate(MODEL_IDS):
        output_tokens = np.maximum(1.0, np.expm1(np.clip(predictions[:, 3 + position], -50.0, 50.0)))
        rates = policy.models[model_id]
        predicted_costs = float(rates.fixed_cost) + (float(rates.input_token_rate) * input_tokens + float(rates.output_token_rate) * output_tokens) / policy.token_unit
        values, _fallback = monetary_cost_multipliers(predicted=predicted_costs, actual=actual_costs[:, position], character_counts=character_counts, minimum_samples=100, quantile=RESIDUAL_QUANTILE)
        result[model_id] = values
    return result


def _predicted_costs(predictions: Any, character_counts: Sequence[int], calibration: Mapping[str, Mapping[str, float]], policy: Any) -> Any:
    input_tokens = np.maximum(1.0, np.expm1(np.clip(predictions[:, 6], -50.0, 50.0)))
    costs = np.zeros((len(predictions), len(MODEL_IDS)), dtype=np.float64)
    for position, model_id in enumerate(MODEL_IDS):
        output_tokens = np.maximum(1.0, np.expm1(np.clip(predictions[:, 3 + position], -50.0, 50.0)))
        rates = policy.models[model_id]
        base = float(rates.fixed_cost) + (float(rates.input_token_rate) * input_tokens + float(rates.output_token_rate) * output_tokens) / policy.token_unit
        multipliers = np.asarray([calibration[model_id]["short" if count <= 512 else "medium" if count <= 2048 else "long"] for count in character_counts], dtype=np.float64)
        costs[:, position] = base * multipliers
    return costs


def _decisions(predictions: Any, costs: Any, settings: TierSettings) -> Tuple[int, ...]:
    result = []
    for quality_row, cost_row in zip(np.clip(predictions[:, :3], 0.0, 1.0), costs):
        light_utility = quality_row[0] - settings.lambda_cost * cost_row[0]
        candidates = [0]
        for position, threshold in ((1, settings.min_gain_ax31), (2, settings.min_gain_think)):
            if quality_row[position] - quality_row[0] >= threshold and cost_row[position] / cost_row[0] <= settings.max_relative_cost and quality_row[position] - settings.lambda_cost * cost_row[position] >= light_utility:
                candidates.append(position)
        result.append(min(candidates, key=lambda position: (-(quality_row[position] - settings.lambda_cost * cost_row[position]), cost_row[position], position)))
    return tuple(result)


def _tier_score(decisions: Sequence[int], targets: Any, actual_costs: Any, tier: str, policy: Any) -> float:
    selected_cost = sum(actual_costs[row, decision] for row, decision in enumerate(decisions))
    baseline_cost = float(actual_costs[:, 0].sum())
    if baseline_cost <= 0.0 or selected_cost > baseline_cost * float(policy.tiers[tier].budget_multiplier):
        return 0.0
    return float(np.mean([targets[row, decision] for row, decision in enumerate(decisions)]))


def _cross_fit_predictions(dense: Any, sparse_rows: Sequence[Mapping[int, float]], targets: Any, groups: Sequence[str], indices: Sequence[int], spec: Tuple[int, float], seed: int) -> Any | None:
    local_groups = [groups[index] for index in indices]
    if len(set(local_groups)) < 4:
        return None
    total = np.zeros((len(indices), targets.shape[1]), dtype=np.float64)
    counts = np.zeros((len(indices), 1), dtype=np.int64)
    for fold in grouped_folds(local_groups, folds=4, seed=seed):
        fit_indices = [indices[index] for index in fold.train_indices]
        validation_indices = [indices[index] for index in fold.validation_indices]
        prediction, _selected, _intercept, _coefficients = _fit_predict_indices(dense, sparse_rows, targets, fit_indices, validation_indices, spec[0], spec[1])
        total[list(fold.validation_indices)] += prediction
        counts[list(fold.validation_indices)] += 1
    return None if (counts == 0).any() else total / counts


def _select_candidate(
    dense: Any,
    sparse_rows: Sequence[Mapping[int, float]],
    targets: Any,
    groups: Sequence[str],
    available_indices: Sequence[int],
    args: argparse.Namespace,
    seed: int,
) -> Tuple[Tuple[int, float], Mapping[str, TierSettings], Mapping[str, object]]:
    """Use inner grouped validation only; callers never pass outer-test indices here."""

    local_groups = [groups[index] for index in available_indices]
    inner = grouped_folds(local_groups, folds=4, seed=seed)
    grid = _candidate_grid(args)
    policy_grid = {tier: _policy_grid(tier) for tier in TIERS}
    states = {(spec, tier, index): [[], [], []] for spec in grid for tier in TIERS for index in range(len(policy_grid[tier]))}
    for fold in inner:
        train_indices = [available_indices[index] for index in fold.train_indices]
        validation_indices = [available_indices[index] for index in fold.validation_indices]
        for spec in grid:
            calibration_predictions = _cross_fit_predictions(dense, sparse_rows, targets, groups, train_indices, spec, seed)
            if calibration_predictions is None:
                continue
            validation_predictions, _selected, _intercept, _coefficients = _fit_predict_indices(dense, sparse_rows, targets, train_indices, validation_indices, spec[0], spec[1])
            # The caller attaches policy and actual costs to the targets namespace for this train-only helper.
            actual_costs, character_counts, policy = args._actual_costs, args._character_counts, args._policy
            calibration = _cost_calibration(calibration_predictions, actual_costs[train_indices], [character_counts[index] for index in train_indices], policy)
            costs = _predicted_costs(validation_predictions, [character_counts[index] for index in validation_indices], calibration, policy)
            for tier in TIERS:
                for index, settings in enumerate(policy_grid[tier]):
                    decisions = _decisions(validation_predictions, costs, settings)
                    state = states[(spec, tier, index)]
                    state[0].append(_tier_score(decisions, targets[validation_indices, :3], actual_costs[validation_indices], tier, policy))
                    state[1].append(float(costs[range(len(decisions)), list(decisions)].sum() / actual_costs[validation_indices, 0].sum()))
                    state[2].append(float(sum(decision != 0 for decision in decisions) / len(decisions)))
    selected = {}
    selection = {}
    head_scores = []
    for spec in grid:
        total = 0.0
        selection[spec] = {}
        for tier in TIERS:
            candidates = tuple(TierPolicyCandidate(policy_grid[tier][index], tuple(state[0]), tuple(state[1]), tuple(state[2]), index) for index in range(len(policy_grid[tier])) if (state := states[(spec, tier, index)])[0])
            chosen = choose_tier_policy(tier, candidates)
            selection[spec][tier] = chosen
            total += 0.0 if chosen is None else chosen.average_score * float(args._policy.tiers[tier].weight)
        head_scores.append((total, spec))
    _score, spec = max(head_scores, key=lambda item: (item[0], -grid.index(item[1])))
    for tier in TIERS:
        selected[tier] = _fallback_settings() if selection[spec][tier] is None else selection[spec][tier].settings
    return spec, selected, {"head_scores": [{"spec": list(item), "weighted_score": score} for score, item in head_scores], "tiers": {tier: None if selection[spec][tier] is None else {"fold_scores": list(selection[spec][tier].fold_scores), "fold_upper_cost_ratios": list(selection[spec][tier].fold_upper_cost_ratios), "fold_upgrade_fractions": list(selection[spec][tier].fold_upgrade_fractions), "grid_index": selection[spec][tier].grid_index} for tier in TIERS}}


def _heads(intercept: Any, coefficients: Any, selected: Sequence[int]) -> Tuple[Mapping[str, LinearHead], Mapping[str, LinearHead], LinearHead]:
    dense_count = len(DENSE_FEATURE_NAMES)

    def head(column: int) -> LinearHead:
        return LinearHead(
            float(intercept[column]),
            tuple(float(value) for value in coefficients[:dense_count, column]),
            {index: float(coefficients[dense_count + position, column]) for position, index in enumerate(selected)},
        )

    quality = {model_id: head(position) for position, model_id in enumerate(MODEL_IDS)}
    output = {model_id: head(3 + position) for position, model_id in enumerate(MODEL_IDS)}
    return quality, output, head(6)


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    data = (json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{0}.tmp-{1}".format(path.name, os.getpid()))
    try:
        temporary.write_bytes(data)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def train(args: argparse.Namespace) -> Mapping[str, object]:
    _require_numpy()
    _validate_arguments(args)
    v21_output_paths(args.artifact, args.manifest, args.report)
    inputs = load_input(args.input)
    outcomes = load_outcomes(args.outcomes)
    rows, _outcome_rows = validate_batches(inputs, outcomes)
    policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
    if policy.schema_version != inputs.schema_version:
        raise ProtocolError("policy schema_version does not match Train input")
    dense, sparse_rows = _feature_rows(inputs, args.hash_dimension)
    targets, actual_outputs = _targets(inputs, outcomes)
    groups = tuple(canonical_content_group(to_prompt_record(episode).text) for episode in inputs.episodes)
    character_counts = tuple(len(to_prompt_record(episode).text) for episode in inputs.episodes)
    actual_costs = _actual_costs(inputs, outcomes, policy)
    args._actual_costs, args._character_counts, args._policy = actual_costs, character_counts, policy
    if len(set(groups)) < 5:
        raise ValueError("Train requires at least five distinct content groups for outer CV")
    outer_prediction_sum = np.zeros_like(targets)
    outer_prediction_count = np.zeros((targets.shape[0], 1), dtype=np.int64)
    outer_report = []
    for seed, outer_fold in repeated_outer_folds(groups):
        spec, tier_settings, selection = _select_candidate(
            dense, sparse_rows, targets, groups, outer_fold.train_indices, args, seed
        )
        predictions, _selected, _intercept, _coefficients = _fit_predict_indices(
            dense, sparse_rows, targets, outer_fold.train_indices, outer_fold.validation_indices, spec[0], spec[1]
        )
        outer_prediction_sum[list(outer_fold.validation_indices)] += predictions
        outer_prediction_count[list(outer_fold.validation_indices)] += 1
        outer_report.append({
            "seed": seed,
            "outer_train_rows": len(outer_fold.train_indices),
            "outer_test_rows": len(outer_fold.validation_indices),
            "outer_train_groups": len({groups[index] for index in outer_fold.train_indices}),
            "outer_test_groups": len({groups[index] for index in outer_fold.validation_indices}),
            "selected_candidate": {"sparse_feature_count": spec[0], "alpha": spec[1]},
            "selected_sparse_feature_count": spec[0],
            "selected_alpha": spec[1],
            "inner_selection": selection,
            "selected_tiers": {tier: {"lambda_cost": tier_settings[tier].lambda_cost, "min_gain": tier_settings[tier].min_gain_ax31, "max_relative_cost": tier_settings[tier].max_relative_cost} for tier in TIERS},
        })
    if (outer_prediction_count == 0).any():
        raise RuntimeError("nested outer CV did not predict every Train row")
    oof_predictions = outer_prediction_sum / outer_prediction_count
    final_spec, tiers, final_selection = _select_candidate(
        dense, sparse_rows, targets, groups, tuple(range(rows)), args, OUTER_SEEDS[0]
    )
    _full_predictions, selected, intercept, coefficients = _fit_predict_indices(
        dense, sparse_rows, targets, tuple(range(rows)), tuple(range(rows)), final_spec[0], final_spec[1]
    )
    quality_heads, output_heads, input_head = _heads(intercept, coefficients, selected)
    calibration_predictions = _cross_fit_predictions(dense, sparse_rows, targets, groups, tuple(range(rows)), final_spec, OUTER_SEEDS[0])
    if calibration_predictions is None:
        raise RuntimeError("final cost calibration requires four content groups")
    cost_calibration = _cost_calibration(calibration_predictions, actual_costs, character_counts, policy)
    residuals = {model_id: cost_calibration[model_id]["global"] for model_id in MODEL_IDS}
    provenance = {
        "train_input_sha256": file_sha256(args.input),
        "train_outcomes_sha256": file_sha256(args.outcomes),
        "row_count": rows,
        "outer_folds": 5,
        "inner_folds": 4,
        "outer_seeds": list(OUTER_SEEDS),
        "alpha": float(final_spec[1]),
        "max_sparse_features": final_spec[0],
        "selected_sparse_feature_count": len(selected),
        "residual_quantile": RESIDUAL_QUANTILE,
        "bucket_definition": "character_count: short<=512, medium<=2048, long>2048",
        "bucket_minimum_samples": 100,
        "global_fallback_rule": "use global one-sided multiplier when a bucket is undersized",
    }
    artifact = PromptBudgetArtifact(
        args.hash_dimension, tuple(DENSE_FEATURE_NAMES), policy.policy_id, policy_sha256(policy),
        quality_heads, output_heads, input_head, residuals, tiers, "absolute-linear", "train-nested-grouped-cv-v2.1", provenance, cost_calibration,
    )
    write_artifact(args.artifact, args.manifest, artifact)
    mse = (oof_predictions - targets) ** 2
    report = {
        "report_type": "promptbudget-train-nested-grouped-cv-v2",
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "training_provenance": provenance,
        "hash_dimension": args.hash_dimension,
        "selected_sparse_feature_count": len(selected),
        "selected_candidate": {
            "name": "absolute-linear-v2.1",
            "family": "absolute-linear",
            "alpha": final_spec[1],
            "inner_selection": final_selection,
        },
        "split_provenance": {
            "content_group_count": len(set(groups)),
            "outer_folds": outer_report,
            "outer_test_use": "comparison reporting only; never passed to candidate selection or final fitting",
        },
        "oof_mse": {
            "quality": {model_id: float(mse[:, index].mean()) for index, model_id in enumerate(MODEL_IDS)},
            "log1p_output_tokens": {model_id: float(mse[:, 3 + index].mean()) for index, model_id in enumerate(MODEL_IDS)},
            "log1p_input_tokens_ax31_light": float(mse[:, 6].mean()),
        },
        "output_residual_multipliers": residuals,
        "comparison": {
            "reference": "fixed-v1",
            "resampling_unit": "content-group",
            "bootstrap_seed": 20260825,
            "bootstrap_repetitions": 1000,
            "primary_endpoint": "quality_points_total",
            "note": "outer predictions are report-only and are not reused for final selection",
        },
    }
    _write_report(args.report, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PromptBudget OOF ridge heads from public Train data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hash-dimension", type=int, default=2 ** 16)
    parser.add_argument("--max-sparse-features", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = train(args)
    except (OSError, ProtocolError, RuntimeError, ValueError, np.linalg.LinAlgError if np is not None else ValueError) as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("NumPy is required"):
            print("ERROR: NumPy is required; install tools/requirements-train.txt", file=sys.stderr)
        else:
            print("ERROR: training failed", file=sys.stderr)
        return 2
    print("OK: train_rows={0} selected_sparse_features={1}".format(report["training_provenance"]["row_count"], report["selected_sparse_feature_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
