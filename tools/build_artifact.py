# SPDX-License-Identifier: Apache-2.0
"""Materialize one verified PromptBudget artifact without synthesis."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from promptbudget.artifact import PromptBudgetError, load_artifact


def _copy_atomic(source: Path, target: Path) -> None:
    data = source.read_bytes(); target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    if hashlib.sha256(data).digest() != hashlib.sha256(target.read_bytes()).digest(): raise OSError("copied bytes do not match")


def build(args: argparse.Namespace) -> None:
    load_artifact(args.artifact, args.manifest)
    targets = (Path(args.output_dir) / "artifact.json", Path(args.output_dir) / "manifest.json", Path(args.package_resources) / "artifact.json", Path(args.package_resources) / "manifest.json")
    for source, target in zip((args.artifact, args.manifest, args.artifact, args.manifest), targets): _copy_atomic(source, target)
    for artifact, manifest in ((targets[0], targets[1]), (targets[2], targets[3])): load_artifact(artifact, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy a verified PromptBudget artifact to release locations.")
    parser.add_argument("--artifact", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/promptbudget-v1")); parser.add_argument("--package-resources", type=Path, default=Path("src/promptbudget/resources")); return parser


def main(argv: Sequence[str] = None) -> int:
    try: build(_parser().parse_args(argv))
    except (OSError, PromptBudgetError, ValueError): print("ERROR: build failed", file=sys.stderr); return 2
    print("OK: artifact materialized"); return 0


if __name__ == "__main__": raise SystemExit(main())
