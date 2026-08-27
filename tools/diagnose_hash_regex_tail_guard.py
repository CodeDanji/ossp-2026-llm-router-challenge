"""Train-only diagnostic for v3.3 cost-tail heterogeneity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np

import hash_regex_cost_stabilization_nested as v32
import hash_regex_tail_guard_nested as nested
from ossp_router.protocol import ProtocolError, load_bundled_policy, load_input, load_outcomes
from promptbudget.safety import grouped_folds
from validate_data import validate_batches


TRAIN_ROWS = 1760
SEEDS = (137, 271, 811)


def reject_dev_path(path: Path) -> None:
    if any(part.lower() == "dev" for part in path.parts):
        raise ValueError("tail diagnostic does not accept a Dev path")


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


def _oof_report(data: v32.EvaluationData, seed: int) -> dict[str, object]:
    rows = tuple(range(len(data.groups)))
    oof = np.empty((len(rows), 3), dtype=float)
    covered = np.zeros(len(rows), dtype=bool)
    for fold in grouped_folds(data.groups, folds=4, seed=seed):
        train = tuple(rows[index] for index in fold.train_indices)
        valid = tuple(rows[index] for index in fold.validation_indices)
        head = v32.fit_log_cost_heads(data.matrix[list(train)], data.log_costs[list(train)])
        oof[list(fold.validation_indices)] = v32.predict_heads(head, data.matrix[list(valid)])
        covered[list(fold.validation_indices)] = True
    if not covered.all() or not np.isfinite(oof).all():
        raise ValueError("grouped OOF coverage is incomplete")
    residuals = data.log_costs - oof
    return {
        "ax31": nested.bucket_residual_summary(oof[:, 1], residuals[:, 1]),
        "think": nested.bucket_residual_summary(oof[:, 2], residuals[:, 2]),
    }


def _normalize(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
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


def diagnose(args: argparse.Namespace) -> Mapping[str, object]:
    for path in (args.input, args.outcomes):
        reject_dev_path(path)
    report_path = require_report_path(args.report)
    inputs, outcomes = load_input(args.input), load_outcomes(args.outcomes)
    rows, _ = validate_batches(inputs, outcomes)
    if inputs.split != "train" or outcomes.split != "train" or rows != TRAIN_ROWS:
        raise ValueError("tail diagnostic requires exactly 1,760 Train rows")
    provenance = {"train_input_sha256": _sha256(args.input), "train_outcomes_sha256": _sha256(args.outcomes), "row_count": rows}
    if not args.execute:
        return {"report_type": "hash-regex-tail-guard-diagnostic-v1", "mode": "dry-run", "training_provenance": provenance, "terminal_status": "not-executed"}
    data = v32.make_evaluation_data(inputs, outcomes, load_bundled_policy())
    reports = {seed: _oof_report(data, seed) for seed in SEEDS}
    status = nested.diagnostic_status(reports)
    report = {
        "report_type": "hash-regex-tail-guard-diagnostic-v1",
        "mode": "execute",
        "training_provenance": provenance,
        "contract": {"bucket_count": 4, "residual_quantile": 0.90, "seeds": SEEDS},
        "grouped_oof": reports,
        "terminal_status": status,
        "forbidden_artifact_assertion": True,
    }
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Train-only request-level cost tails.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = diagnose(_parser().parse_args(argv))
    except (OSError, ProtocolError, ValueError, ArithmeticError) as error:
        print(f"ERROR: tail diagnostic failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"mode": report["mode"], "terminal_status": report["terminal_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
