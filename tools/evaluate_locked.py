# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""The only CLI permitted to score a PromptBudget locked holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping, Optional, Sequence

from ossp_router.protocol import TIERS, load_bundled_policy, load_input, load_outcomes, load_policy
from ossp_router.scoring import score_submissions
from promptbudget.artifact import load_artifact
from promptbudget.input_adapter import to_prompt_record, to_submission
from promptbudget.locked_eval import AppendOnlyLedger, LockedEvaluationError, holdout_digest, require_reservation
from promptbudget.policy import select_model
from promptbudget.safety import canonical_content_group


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".{0}.".format(path.name))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_provenance() -> Mapping[str, object]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", True
    return {
        "git_commit": commit,
        "worktree_state": "dirty" if dirty else "clean",
        "evaluator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpreter": sys.version,
        "sqlite_version": __import__("sqlite3").sqlite_version,
    }


def _group_manifest(inputs) -> Sequence[tuple[str, int]]:
    counts = {}
    for episode in inputs.episodes:
        group = canonical_content_group(to_prompt_record(episode).text)
        counts[group] = counts.get(group, 0) + 1
    return tuple(sorted(counts.items()))


def _score(inputs, outcomes, artifact_path: Path, manifest_path: Path, policy_path: Optional[Path]) -> Mapping[str, object]:
    artifact = load_artifact(artifact_path, manifest_path)
    policy = load_policy(policy_path) if policy_path is not None else load_bundled_policy()
    submissions = []
    for tier in TIERS:
        decisions = (
            select_model(to_prompt_record(episode).text, tier, artifact, policy)
            for episode in inputs.episodes
        )
        submissions.append(to_submission(inputs, tier, decisions, policy.policy_id))
    return score_submissions(inputs, outcomes, submissions, policy)


def evaluate_locked(args: argparse.Namespace) -> Mapping[str, object]:
    input_bytes, outcome_bytes = args.input.read_bytes(), args.outcomes.read_bytes()
    inputs = load_input(args.input)
    digest = holdout_digest(input_bytes, outcome_bytes, inputs.schema_version, _group_manifest(inputs))
    tracked = [args.artifact, args.manifest]
    if args.reference_artifact is not None:
        tracked.extend((args.reference_artifact, args.reference_manifest))
    before = {str(path): _sha256(path) for path in tracked}
    metadata = {
        "holdout_input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "holdout_outcomes_sha256": hashlib.sha256(outcome_bytes).hexdigest(),
        "schema_version": inputs.schema_version,
        "group_manifest": _group_manifest(inputs),
        "artifact_hashes": before,
        "operator": args.operator,
        "cli_arguments": [str(item) for item in args.argv],
        **_git_provenance(),
    }
    ledger = AppendOnlyLedger(args.ledger)
    reservation = ledger.reserve(digest, metadata)
    try:
        require_reservation(ledger, digest)
        outcomes = load_outcomes(args.outcomes)
        candidate = _score(inputs, outcomes, args.artifact, args.manifest, args.policy)
        reference = None
        if args.reference_artifact is not None:
            reference = _score(inputs, outcomes, args.reference_artifact, args.reference_manifest, args.policy)
        after = {str(path): _sha256(path) for path in tracked}
        if before != after:
            raise LockedEvaluationError("artifact or manifest changed during locked evaluation")
        report = {
            "report_type": "promptbudget-locked-evaluation-v2",
            "holdout_digest": digest,
            "reservation_utc": reservation.reserved_at_utc,
            "candidate": candidate,
            "reference": reference,
            "provenance": metadata,
        }
        _atomic_json(args.report, report)
        ledger.append(digest, "completed", {"report_sha256": _sha256(args.report), "artifact_hashes": after})
        return report
    except Exception as error:
        ledger.append(digest, "failed", {"error_type": type(error).__name__, "message": str(error)})
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reserve and evaluate a locked PromptBudget holdout once.")
    for name in ("input", "outcomes", "artifact", "manifest", "report", "ledger"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--reference-artifact", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    args.argv = tuple(argv if argv is not None else sys.argv[1:])
    if bool(args.reference_artifact) != bool(args.reference_manifest):
        parser.error("--reference-artifact and --reference-manifest must be provided together")
    try:
        evaluate_locked(args)
    except (OSError, ValueError, LockedEvaluationError) as error:
        print("ERROR: locked evaluation failed: {0}".format(error), file=sys.stderr)
        return 2
    print("OK: locked holdout reserved and evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
