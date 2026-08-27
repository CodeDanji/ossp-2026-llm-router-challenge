# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Safe CLI for Train-only hash-regex quality-objective evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import hash_regex_quality_nested as nested
from ossp_router.protocol import ProtocolError, load_bundled_policy, load_input, load_outcomes
from promptbudget.safety import repeated_outer_folds
from validate_data import validate_batches


TRAIN_ROWS = 1760


def reject_dev_path(path: Path) -> None:
    if any(part.lower() == "dev" for part in path.parts):
        raise ValueError("nested evaluation does not accept a Dev path")


def require_output_path(path: Path) -> Path:
    resolved = path.resolve()
    root = next(
        (
            parent
            for parent in (resolved.parent, *resolved.parents)
            if parent.name == "hash-regex-quality" and parent.parent.name == "build"
        ),
        None,
    )
    if root is None or root not in resolved.parents:
        raise ValueError("report output must be below build/hash-regex-quality")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
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


def evaluate(args: argparse.Namespace) -> Mapping[str, object]:
    for path in (args.input, args.outcomes):
        reject_dev_path(path)
    report_path = require_output_path(args.report)
    if args.one_outer_fold and args.full:
        raise ValueError("choose exactly one of --one-outer-fold or --full")
    if args.execute and not (args.one_outer_fold or args.full):
        raise ValueError("--execute requires --one-outer-fold or --full")
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
        "outer_seeds": [137, 271, 811],
        "outer_folds": 5,
        "inner_folds": 4,
    }
    if not args.execute:
        return {
            "report_type": "hash-regex-quality-nested-evaluation-v1",
            "mode": "dry-run",
            "training_provenance": provenance,
            "outer_schedule": [{"seed": seed, "fold": index} for index, (seed, _split) in enumerate(schedule)],
        }
    selected = schedule if args.full else schedule[:1]
    folds = [
        nested.evaluate_outer_fold(data, split.train_indices, split.validation_indices, seed, number)
        for number, (seed, split) in enumerate(selected)
    ]
    report: dict[str, object] = {
        "report_type": "hash-regex-quality-nested-evaluation-v1",
        "mode": "full" if args.full else "one-outer-fold-smoke",
        "training_provenance": provenance,
        "folds": folds,
    }
    if args.full:
        report["winner"] = nested.choose_winner(folds)
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate approved hash-regex quality candidates on Train folds only.")
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
    print(json.dumps({"mode": report["mode"], "winner": report.get("winner", {}).get("name")}, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
