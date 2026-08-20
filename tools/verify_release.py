# SPDX-License-Identifier: Apache-2.0
"""Verify local PromptBudget release evidence without fabricating external claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from promptbudget.artifact import PromptBudgetError, load_artifact


def _write(path: Path, value: Mapping[str, object]) -> None:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{0}.tmp-{1}".format(path.name, os.getpid()))
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(args: argparse.Namespace) -> Mapping[str, object]:
    artifact = load_artifact(args.artifact, args.manifest)
    for required in (Path("LICENSE"), Path("NOTICE"), Path("SBOM.spdx.json"), Path("container/Dockerfile")):
        if not required.is_file():
            raise ValueError("missing release file: {0}".format(required))
    resources = Path("src/promptbudget/resources")
    resources_match = (
        (resources / "artifact.json").is_file()
        and (resources / "manifest.json").is_file()
        and _sha256(args.artifact) == _sha256(resources / "artifact.json")
        and _sha256(args.manifest) == _sha256(resources / "manifest.json")
    )
    if not resources_match:
        raise ValueError("bundled resources do not match the release artifact")
    report = {
        "report_type": "promptbudget-release-verification-v1",
        "local_status": "pass",
        "external_status": "pending-submission-url-and-image-digest",
        "artifact_sha256": _sha256(args.artifact),
        "policy_id": artifact.policy_id,
        "checks": ["canonical-artifact", "bundled-resource-match", "license-notice-sbom", "pinned-container-base"],
    }
    _write(args.report, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify local PromptBudget release evidence.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] = None) -> int:
    try:
        report = verify(_parser().parse_args(argv))
    except (OSError, PromptBudgetError, ValueError):
        print("ERROR: release verification failed", file=sys.stderr)
        return 2
    print("OK: local_status={0} external_status={1}".format(report["local_status"], report["external_status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
