"""Lock the v3.3 safe candidate into a Train-only runtime artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import hash_regex
import hash_regex_cost_stabilization_nested as v32
import hash_regex_tail_guard_nested as nested
from ossp_router.protocol import ProtocolError, load_bundled_policy, load_input, load_outcomes, policy_sha256
from validate_data import validate_batches


TRAIN_ROWS = 1760
SEED = 137


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
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


def locked_configuration(report: Mapping[str, object]) -> Mapping[str, object]:
    if report.get("terminal_status") != "safe-candidate":
        raise ValueError("finalization requires a safe-candidate nested report")
    return {"seed": SEED, "hash_bins": 256, "ridge_alpha": 100.0, "tier_safety_ratios": {"fast": 1.0, "balanced": 1.0, "premium": 1.0}}


def _head(head: v32.LinearHeads, model: int) -> Mapping[str, object]:
    return {"intercept": float(head.intercept[model]), "coefficients": [float(value) for value in head.coefficients[:, model]]}


def finalize(args: argparse.Namespace) -> Mapping[str, object]:
    report = json.loads(args.nested_report.read_text(encoding="utf-8"))
    config = locked_configuration(report)
    input_hash, outcome_hash = _sha256(args.input), _sha256(args.outcomes)
    provenance = report.get("training_provenance", {})
    if provenance.get("train_input_sha256") != input_hash or provenance.get("train_outcomes_sha256") != outcome_hash:
        raise ValueError("nested report hashes do not match finalizer input")
    diagnostic = report.get("diagnostic_provenance", {})
    if diagnostic.get("terminal_status") != "tail-signal-present":
        raise ValueError("nested report must retain a signal-present diagnostic")
    inputs, outcomes = load_input(args.input), load_outcomes(args.outcomes)
    rows, _ = validate_batches(inputs, outcomes)
    if inputs.split != "train" or outcomes.split != "train" or rows != TRAIN_ROWS:
        raise ValueError("finalization requires exactly 1,760 Train rows")
    policy = load_bundled_policy()
    data = v32.make_evaluation_data(inputs, outcomes, policy)
    guard = nested.fit_tail_guard(data, tuple(range(rows)), SEED)
    quality = v32.fit_raw_quality_heads(data.matrix, data.scores)
    costs = v32.fit_log_cost_heads(data.matrix, data.log_costs)
    artifact = {
        "artifact_type": hash_regex.TAIL_GUARD_ARTIFACT_TYPE,
        "schema_version": 1,
        "feature_version": hash_regex.FEATURE_VERSION,
        "hash_algorithm": "fnv1a64-signed-word-1-2",
        "hash_bins": 256,
        "dense_feature_names": list(hash_regex.DENSE_FEATURE_NAMES),
        "model_ids": list(policy.models),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "feature_mean": [float(value) for value in quality.mean],
        "feature_scale": [float(value) for value in quality.scale],
        "score_heads": {model: _head(quality, index) for index, model in enumerate(policy.models)},
        "log_cost_heads": {model: _head(costs, index) for index, model in enumerate(policy.models)},
        "tier_safety_ratios": config["tier_safety_ratios"],
        "training_summary": {"input_sha256": input_hash, "outcomes_sha256": outcome_hash, "nested_report_sha256": _sha256(args.nested_report), "seed": SEED, "optimizer": "grouped-oof-tail-guard-v1"},
        "tail_cost_guard": {
            "bucket_count": 4,
            "quantile": 0.90,
            "models": {
                "ax31": {"edges": list(guard.ax31_edges), "log_guards": list(guard.ax31_log_guards)},
                "axk1-think": {"edges": list(guard.think_edges), "log_guards": list(guard.think_log_guards)},
            },
        },
    }
    hash_regex.parse_artifact(artifact)
    _atomic_json(args.artifact, artifact)
    result = {"report_type": "hash-regex-tail-guard-finalization-v1", "terminal_status": "finalized", "training_provenance": artifact["training_summary"], "artifact_sha256": _sha256(args.artifact), "guard": artifact["tail_cost_guard"], "forbidden_dev_assertion": True}
    _atomic_json(args.report, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize a safe v3.3 tail guard from Train only.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--nested-report", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = finalize(_parser().parse_args(argv))
    except (OSError, ProtocolError, ValueError, ArithmeticError) as error:
        print(f"ERROR: tail finalization failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"terminal_status": result["terminal_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
