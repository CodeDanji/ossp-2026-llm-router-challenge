# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Guarded Train-only CLI for the hash-regex cost-stabilization study."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import hash_regex_cost_stabilization_nested as nested
from ossp_router.protocol import ProtocolError, load_bundled_policy, load_input, load_outcomes
from promptbudget.safety import repeated_outer_folds
from validate_data import validate_batches


TRAIN_ROWS = 1760
MULTIPLIER_GRID = tuple((ax31, think) for ax31 in (1.00, 1.10, 1.25) for think in (1.00, 1.25, 1.50))


def reject_dev_path(path: Path) -> None:
    if any(part.lower() == "dev" for part in path.parts):
        raise ValueError("nested evaluation does not accept a Dev path")


def require_output_path(path: Path) -> Path:
    resolved = path.resolve()
    root = next(
        (
            parent
            for parent in (resolved.parent, *resolved.parents)
            if parent.name == "hash-regex-cost-stabilization" and parent.parent.name == "build"
        ),
        None,
    )
    if root is None or root not in resolved.parents:
        raise ValueError("report output must be below build/hash-regex-cost-stabilization")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_normalize(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(_json_normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validated_grid_selection(data: nested.EvaluationData, outer_train: Sequence[int], seed: int) -> dict[str, object]:
    grid = []
    candidates = []
    for pair in MULTIPLIER_GRID:
        evaluation = nested.evaluate_inner_pair(data, outer_train, seed, pair)
        admission = evaluation["admission"]
        if (
            len(evaluation["inner_folds"]) != 4
            or int(admission["required_checks"]) != 12
            or len(admission["checks"]) != 12
        ):
            raise ValueError("each multiplier pair must contain four inner reports and twelve checks")
        grid.append(evaluation)
        if admission["admitted"]:
            candidates.append((
                nested._exact_decimal(admission["official_score"], "official score").copy_negate(),
                nested._exact_decimal(admission["maximum_actual_ratio"], "maximum actual ratio"),
                pair,
                evaluation,
            ))
    if not candidates:
        return {"status": "no-admitted-multiplier", "pair": None, "route": "all-light", "candidate_grid": grid}
    _score, _ratio, pair, evaluation = min(candidates)
    return {
        "status": "admitted-multiplier",
        "pair": pair,
        "route": "cost-multiplied",
        "admission": evaluation["admission"],
        "inner_folds": evaluation["inner_folds"],
        "pooled_for_routing": False,
        "candidate_grid": grid,
    }


def _evaluate_outer_fold(data: nested.EvaluationData, train: Sequence[int], test: Sequence[int], seed: int, number: int) -> dict[str, object]:
    original = nested.select_inner_multiplier
    try:
        nested.select_inner_multiplier = _validated_grid_selection
        return nested.evaluate_outer_fold(data, train, test, seed, number)
    finally:
        nested.select_inner_multiplier = original


def _smoke_diagnostics(fold: Mapping[str, object]) -> Mapping[str, object]:
    checks = [
        {"seed": fold["seed"], "fold": fold["fold"], "tier": tier, "budget_passed": fold["selected_report"]["tiers"][tier]["budget_passed"], "actual_ratio": fold["selected_report"]["tiers"][tier]["actual_ratio"]}
        for tier in ("fast", "balanced", "premium")
    ]
    return {
        "actual_checks": checks,
        "actual_checks_passed": sum(bool(check["budget_passed"]) for check in checks),
        "actual_checks_required": 3,
        "status": "smoke-not-promotable",
    }


def evaluate(args: argparse.Namespace) -> Mapping[str, object]:
    for path in (args.input, args.outcomes):
        reject_dev_path(path)
    report_path = require_output_path(args.report)
    if args.one_outer_fold == args.full and args.execute:
        raise ValueError("--execute requires exactly one of --one-outer-fold or --full")
    if not args.execute and (args.one_outer_fold or args.full):
        raise ValueError("run selectors require --execute")
    inputs, outcomes = load_input(args.input), load_outcomes(args.outcomes)
    rows, _ = validate_batches(inputs, outcomes)
    if inputs.split != "train" or outcomes.split != "train" or rows != TRAIN_ROWS:
        raise ValueError("nested evaluation requires exactly 1,760 Train rows")
    policy = load_bundled_policy()
    data = nested.make_evaluation_data(inputs, outcomes, policy)
    schedule = repeated_outer_folds(data.groups)
    provenance = {
        "train_input_sha256": _sha256(args.input),
        "train_outcomes_sha256": _sha256(args.outcomes),
        "row_count": rows,
        "content_group_count": len(set(data.groups)),
    }
    grid = {"ax31": [1.00, 1.10, 1.25], "think": [1.00, 1.25, 1.50], "pairs": MULTIPLIER_GRID}
    schedule_report = [{"seed": seed, "fold": number} for number, (seed, _split) in enumerate(schedule)]
    if not args.execute:
        return _json_normalize({
            "report_type": "hash-regex-cost-stabilization-nested-evaluation-v1",
            "mode": "dry-run",
            "training_provenance": provenance,
            "outer_schedule": schedule_report,
            "multiplier_grid": grid,
            "terminal_status": "not-executed",
        })
    selected = schedule if args.full else schedule[:1]
    folds = [
        _evaluate_outer_fold(data, split.train_indices, split.validation_indices, seed, number)
        for number, (seed, split) in enumerate(selected)
    ]
    diagnostics = nested.aggregate_outer_folds(folds) if args.full else _smoke_diagnostics(folds[0])
    report = _json_normalize({
        "report_type": "hash-regex-cost-stabilization-nested-evaluation-v1",
        "mode": "full" if args.full else "one-outer-fold-smoke",
        "training_provenance": provenance,
        "outer_schedule": schedule_report,
        "multiplier_grid": grid,
        "folds": folds,
        "outer_diagnostics": diagnostics,
        "terminal_status": diagnostics["status"],
    })
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate guarded cost-stabilized hash-regex routing on Train folds only.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--one-outer-fold", action="store_true")
    parser.add_argument("--full", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = evaluate(_parser().parse_args(argv))
    except (OSError, ProtocolError, ValueError, ArithmeticError) as error:
        print(f"ERROR: nested evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"mode": report["mode"], "terminal_status": report["terminal_status"]}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
