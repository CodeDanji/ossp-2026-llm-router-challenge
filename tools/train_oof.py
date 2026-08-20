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
    if args.folds <= 1:
        raise ValueError("--folds must be at least 2 for OOF training")


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


def _oof(matrix: Any, targets: Any, folds: int, alpha: float) -> Any:
    row_count = matrix.shape[0]
    predictions = np.empty_like(targets)
    fold_ids = np.arange(row_count) % folds
    for fold in range(folds):
        held_out = fold_ids == fold
        intercept, coefficients = _fit_ridge(matrix[~held_out], targets[~held_out], alpha)
        predictions[held_out] = _predict(matrix[held_out], intercept, coefficients)
    return predictions


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
    if args.folds > rows:
        raise ValueError("--folds must not exceed Train row count")
    policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
    if policy.schema_version != inputs.schema_version:
        raise ProtocolError("policy schema_version does not match Train input")
    dense, sparse_rows = _feature_rows(inputs, args.hash_dimension)
    targets, actual_outputs = _targets(inputs, outcomes)
    selected = _select_sparse(sparse_rows, targets[:, :3].mean(axis=1), args.max_sparse_features)
    matrix = _design_matrix(dense, sparse_rows, selected)
    oof_predictions = _oof(matrix, targets, args.folds, args.alpha)
    intercept, coefficients = _fit_ridge(matrix, targets, args.alpha)
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
        "folds": args.folds,
        "alpha": float(args.alpha),
        "max_sparse_features": args.max_sparse_features,
        "residual_quantile": RESIDUAL_QUANTILE,
    }
    artifact = PromptBudgetArtifact(
        args.hash_dimension, tuple(DENSE_FEATURE_NAMES), policy.policy_id, policy_sha256(policy),
        quality_heads, output_heads, input_head, residuals, tiers, "absolute-linear", "train-oof-v1", provenance,
    )
    write_artifact(args.artifact, args.manifest, artifact)
    mse = (oof_predictions - targets) ** 2
    report = {
        "report_type": "promptbudget-train-oof-v1",
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "training_provenance": provenance,
        "hash_dimension": args.hash_dimension,
        "selected_sparse_feature_count": len(selected),
        "oof_mse": {
            "quality": {model_id: float(mse[:, index].mean()) for index, model_id in enumerate(MODEL_IDS)},
            "log1p_output_tokens": {model_id: float(mse[:, 3 + index].mean()) for index, model_id in enumerate(MODEL_IDS)},
            "log1p_input_tokens_ax31_light": float(mse[:, 6].mean()),
        },
        "output_residual_multipliers": residuals,
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
