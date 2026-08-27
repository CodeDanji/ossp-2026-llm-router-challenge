"""Train-only nested evaluator for the fixed v3.3 tail guard."""

from __future__ import annotations

import argparse
import dataclasses
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import hash_regex_cost_stabilization_nested as v32
import hash_regex_tail_guard_nested as nested
from ossp_router.protocol import ProtocolError, load_bundled_policy, load_input, load_outcomes
from promptbudget.safety import repeated_outer_folds
from validate_data import validate_batches


TRAIN_ROWS = 1760


def reject_dev_path(path: Path) -> None:
    if any(part.lower() == "dev" for part in path.parts):
        raise ValueError("tail evaluator does not accept a Dev path")


def require_report_path(path: Path) -> Path:
    resolved = path.resolve()
    if not any(parent.name == "hash-regex-tail-guard" and parent.parent.name == "build" for parent in resolved.parents):
        raise ValueError("report output must be below build/hash-regex-tail-guard")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_diagnostic(path: Path, input_hash: str, outcome_hash: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("diagnostic must be valid JSON") from error
    if value.get("terminal_status") != "tail-signal-present":
        raise ValueError("evaluator requires a tail-signal-present diagnostic")
    provenance = value.get("training_provenance", {})
    if provenance.get("train_input_sha256") != input_hash or provenance.get("train_outcomes_sha256") != outcome_hash:
        raise ValueError("diagnostic input/outcome hashes do not match evaluator paths")
    return value


def _normalize(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
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


def _smoke(fold: Mapping[str, object]) -> Mapping[str, object]:
    checks = [
        {"seed": fold["seed"], "fold": fold["fold"], "tier": tier,
         "budget_passed": fold["selected_report"]["tiers"][tier]["budget_passed"],
         "actual_ratio": fold["selected_report"]["tiers"][tier]["actual_ratio"]}
        for tier in ("fast", "balanced", "premium")
    ]
    return {"actual_checks": checks, "actual_checks_passed": sum(bool(item["budget_passed"]) for item in checks), "actual_checks_required": 3, "status": "smoke-not-promotable"}


def evaluate(args: argparse.Namespace) -> Mapping[str, object]:
    for path in (args.input, args.outcomes, args.diagnostic):
        reject_dev_path(path)
    report_path = require_report_path(args.report)
    if args.execute and args.one_outer_fold == args.full:
        raise ValueError("--execute requires exactly one of --one-outer-fold or --full")
    if not args.execute and (args.one_outer_fold or args.full):
        raise ValueError("run selectors require --execute")
    input_hash, outcome_hash = _sha256(args.input), _sha256(args.outcomes)
    diagnostic = _load_diagnostic(args.diagnostic, input_hash, outcome_hash)
    inputs, outcomes = load_input(args.input), load_outcomes(args.outcomes)
    rows, _ = validate_batches(inputs, outcomes)
    if inputs.split != "train" or outcomes.split != "train" or rows != TRAIN_ROWS:
        raise ValueError("tail evaluator requires exactly 1,760 Train rows")
    provenance = {"train_input_sha256": input_hash, "train_outcomes_sha256": outcome_hash, "row_count": rows, "diagnostic_sha256": _sha256(args.diagnostic)}
    if not args.execute:
        return {"report_type": "hash-regex-tail-guard-nested-evaluation-v1", "mode": "dry-run", "training_provenance": provenance, "terminal_status": "not-executed"}
    data = v32.make_evaluation_data(inputs, outcomes, load_bundled_policy())
    schedule = repeated_outer_folds(data.groups)
    selected = schedule if args.full else schedule[:1]
    folds = [
        nested.evaluate_outer_guard_fold(data, split.train_indices, split.validation_indices, seed, number)
        for number, (seed, split) in enumerate(selected)
    ]
    diagnostics = nested.aggregate_outer_guard_folds(folds) if args.full else _smoke(folds[0])
    report = {
        "report_type": "hash-regex-tail-guard-nested-evaluation-v1",
        "mode": "full" if args.full else "one-outer-fold-smoke",
        "training_provenance": provenance,
        "diagnostic_provenance": diagnostic,
        "outer_schedule": [{"seed": seed, "fold": number} for number, (seed, _split) in enumerate(schedule)],
        "folds": folds,
        "outer_diagnostics": diagnostics,
        "terminal_status": diagnostics["status"],
        "forbidden_artifact_assertion": True,
    }
    _atomic_json(report_path, report)
    return _normalize(report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the fixed v3.3 Train-only tail guard.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--one-outer-fold", action="store_true")
    parser.add_argument("--full", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = evaluate(_parser().parse_args(argv))
    except (OSError, ProtocolError, ValueError, ArithmeticError) as error:
        print(f"ERROR: tail evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"mode": report["mode"], "terminal_status": report["terminal_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
