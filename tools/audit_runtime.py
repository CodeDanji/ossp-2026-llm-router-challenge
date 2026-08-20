# SPDX-License-Identifier: Apache-2.0
"""Check PromptBudget's content-only runtime boundary and determinism."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from ossp_router.protocol import ProtocolError, load_bundled_policy, load_input
from promptbudget.artifact import PromptBudgetError, load_artifact
from promptbudget.input_adapter import to_prompt_record
from promptbudget.policy import select_model


FORBIDDEN = ("episode_id", "output_key", "outcome", "requests", "urllib", "socket", "http.client")


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


def audit(args: argparse.Namespace) -> Mapping[str, object]:
    source_root = Path(args.source_root)
    for filename in ("policy.py", "text_features.py"):
        text = (source_root / filename).read_text(encoding="utf-8")
        found = [token for token in FORBIDDEN if token in text]
        if found:
            raise PromptBudgetError("forbidden runtime signal in {0}: {1}".format(filename, ", ".join(found)))
    artifact = load_artifact(args.artifact, args.manifest)
    policy = load_bundled_policy()
    inputs = load_input(args.input)
    first = {episode.episode_id: {tier: select_model(to_prompt_record(episode).text, tier, artifact, policy) for tier in policy.tiers} for episode in inputs.episodes}
    reversed_results = {episode.episode_id: {tier: select_model(to_prompt_record(episode).text, tier, artifact, policy) for tier in policy.tiers} for episode in reversed(inputs.episodes)}
    if first != reversed_results:
        raise PromptBudgetError("input-order permutation changed a decision")
    report = {
        "report_type": "promptbudget-runtime-audit-v1",
        "status": "pass",
        "episode_count": len(inputs.episodes),
        "checks": ["content-only-source", "no-network-import", "repeatable-selection", "input-order-permutation"],
    }
    _write(args.report, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit PromptBudget runtime boundaries.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("src/promptbudget"))
    return parser


def main(argv: Sequence[str] = None) -> int:
    try:
        report = audit(_parser().parse_args(argv))
    except (OSError, ProtocolError, PromptBudgetError, ValueError) as exc:
        print("ERROR: runtime audit failed: {0}".format(exc), file=sys.stderr)
        return 2
    print("OK: status={0}".format(report["status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
