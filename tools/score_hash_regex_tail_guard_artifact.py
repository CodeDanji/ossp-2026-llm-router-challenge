"""One-pass Dev sanity scorer for a frozen v3.3 artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import hash_regex
from ossp_router.protocol import ProtocolError, load_bundled_policy, load_input, load_outcomes
from ossp_router.scoring import score_submissions


def require_dev_path(path: Path) -> Path:
    if not any(part.lower() == "dev" for part in path.parts):
        raise ValueError("Dev sanity requires a Dev input or outcome path")
    return path


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


def score(args: argparse.Namespace) -> Mapping[str, object]:
    require_dev_path(args.input)
    require_dev_path(args.outcomes)
    inputs, outcomes = load_input(args.input), load_outcomes(args.outcomes)
    if inputs.split != "dev" or outcomes.split != "dev":
        raise ValueError("Dev sanity accepts only Dev batches")
    artifact = hash_regex.load_artifact(args.artifact)
    policy = load_bundled_policy()
    submissions = [hash_regex.make_hash_regex_submission(inputs, policy, artifact, tier).submission for tier in ("fast", "balanced", "premium")]
    report = score_submissions(inputs, outcomes, submissions, policy)
    status = "dev-sanity-passed" if all(report["tiers"][tier]["budget_passed"] for tier in report["tiers"]) else "fallback-required"
    result = {"report_type": "hash-regex-tail-guard-dev-sanity-v1", "terminal_status": status, "artifact": str(args.artifact), "score": report, "frozen_artifact_assertion": True}
    _atomic_json(args.report, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a frozen v3.3 tail guard once on Dev.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = score(_parser().parse_args(argv))
    except (OSError, ProtocolError, ValueError, ArithmeticError) as error:
        print(f"ERROR: Dev sanity failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"terminal_status": result["terminal_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
