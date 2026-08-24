# SPDX-License-Identifier: Apache-2.0
"""Train compact PromptBudget ridge heads from public Train data only."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
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
from promptbudget.safety import CandidateResult, OUTER_SEEDS, canonical_content_group, choose_one_standard_error, grouped_folds, repeated_outer_folds
from promptbudget.text_features import DENSE_FEATURE_NAMES, extract_features
from validate_data import file_sha256, validate_batches


RESIDUAL_QUANTILE = 0.99


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


def _candidate_grid(args: argparse.Namespace) -> Tuple[Tuple[int, float], ...]:
    """A small pre-registered grid; the runtime supports the absolute-linear family only."""

    feature_counts = tuple(sorted({min(64, args.max_sparse_features), args.max_sparse_features}))
    alphas = tuple(sorted({args.alpha, max(1.0, args.alpha / 10.0), args.alpha * 10.0}))
    return tuple((features, alpha) for features in feature_counts for alpha in alphas)


def _select_candidate(
    dense: Any,
    sparse_rows: Sequence[Mapping[int, float]],
    targets: Any,
    groups: Sequence[str],
    available_indices: Sequence[int],
    args: argparse.Namespace,
    seed: int,
) -> Tuple[Tuple[int, float], CandidateResult]:
    """Use inner grouped validation only; callers never pass outer-test indices here."""

    local_groups = [groups[index] for index in available_indices]
    inner = grouped_folds(local_groups, folds=4, seed=seed)
    results = []
    grid = _candidate_grid(args)
    for grid_index, (feature_count, alpha) in enumerate(grid):
        losses = []
        for fold in inner:
            train_indices = [available_indices[index] for index in fold.train_indices]
            validation_indices = [available_indices[index] for index in fold.validation_indices]
            predictions, _selected, _intercept, _coefficients = _fit_predict_indices(
                dense, sparse_rows, targets, train_indices, validation_indices, feature_count, alpha
            )
            losses.append(float(((predictions - targets[validation_indices]) ** 2).mean()))
        results.append(CandidateResult(
            "absolute-linear-{0}".format(grid_index), tuple(losses), feature_count, 0, 1.0, grid_index
        ))
    chosen = choose_one_standard_error(results)
    return grid[chosen.grid_index], chosen


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
    inputs = load_input(args.input)
    outcomes = load_outcomes(args.outcomes)
    rows, _outcome_rows = validate_batches(inputs, outcomes)
    policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
    if policy.schema_version != inputs.schema_version:
        raise ProtocolError("policy schema_version does not match Train input")
    dense, sparse_rows = _feature_rows(inputs, args.hash_dimension)
    targets, actual_outputs = _targets(inputs, outcomes)
    groups = tuple(canonical_content_group(to_prompt_record(episode).text) for episode in inputs.episodes)
    if len(set(groups)) < 5:
        raise ValueError("Train requires at least five distinct content groups for outer CV")
    outer_prediction_sum = np.zeros_like(targets)
    outer_prediction_count = np.zeros((targets.shape[0], 1), dtype=np.int64)
    outer_report = []
    for seed, outer_fold in repeated_outer_folds(groups):
        spec, candidate = _select_candidate(
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
            "selected_candidate": candidate.name,
            "selected_sparse_feature_count": spec[0],
            "selected_alpha": spec[1],
            "inner_validation_losses": list(candidate.fold_losses),
        })
    if (outer_prediction_count == 0).any():
        raise RuntimeError("nested outer CV did not predict every Train row")
    oof_predictions = outer_prediction_sum / outer_prediction_count
    final_spec, final_candidate = _select_candidate(
        dense, sparse_rows, targets, groups, tuple(range(rows)), args, OUTER_SEEDS[0]
    )
    _full_predictions, selected, intercept, coefficients = _fit_predict_indices(
        dense, sparse_rows, targets, tuple(range(rows)), tuple(range(rows)), final_spec[0], final_spec[1]
    )
    quality_heads, output_heads, input_head = _heads(intercept, coefficients, selected)
    predicted_outputs = np.maximum(1.0, np.expm1(np.clip(oof_predictions[:, 3:6], -50.0, 50.0)))
    residuals = {
        model_id: max(1.0, float(np.quantile(actual_outputs[:, index] / predicted_outputs[:, index], RESIDUAL_QUANTILE, method="higher")))
        for index, model_id in enumerate(MODEL_IDS)
    }
    tiers = {tier: TierSettings(0.0, 0.0, 0.0, 1.0, 100.0) for tier in TIERS}
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
        "bucket_definition": "global-only-v2; small conditional buckets use global fallback",
        "bucket_minimum_samples": 1,
        "global_fallback_rule": "use global one-sided multiplier when a bucket is undersized",
    }
    artifact = PromptBudgetArtifact(
        args.hash_dimension, tuple(DENSE_FEATURE_NAMES), policy.policy_id, policy_sha256(policy),
        quality_heads, output_heads, input_head, residuals, tiers, "absolute-linear", "train-nested-grouped-cv-v2", provenance,
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
            "name": final_candidate.name,
            "family": "absolute-linear",
            "alpha": final_spec[1],
            "inner_validation_losses": list(final_candidate.fold_losses),
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
