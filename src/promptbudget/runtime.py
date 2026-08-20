# SPDX-License-Identifier: Apache-2.0
"""Production PromptBudget router entry point."""
from __future__ import annotations

import argparse
import os
import tempfile
from importlib import resources
from pathlib import Path

from ossp_router.protocol import ProtocolError, load_input, load_policy, load_bundled_policy, submission_to_dict, dumps_json
from .artifact import load_artifact
from .input_adapter import to_prompt_record, to_submission
from .policy import select_model
from .schema import PromptBudgetError, TIERS


def _write_atomic(path: Path, text: str) -> None:
    if not path.parent.is_dir():
        raise PromptBudgetError(f"output directory does not exist: {path.parent}")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _resource_paths():
    try:
        artifact = resources.files("promptbudget.resources").joinpath("artifact.json")
        manifest = resources.files("promptbudget.resources").joinpath("manifest.json")
        if not artifact.is_file() or not manifest.is_file():
            raise FileNotFoundError("bundled artifact.json/manifest.json")
        return artifact, manifest
    except (OSError, FileNotFoundError) as exc:
        raise PromptBudgetError("PromptBudget trained artifact is missing; provide --artifact and --manifest") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="router-run")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tier", required=True, choices=TIERS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact")
    parser.add_argument("--manifest")
    parser.add_argument("--policy")
    args = parser.parse_args(argv)
    try:
        if bool(args.artifact) != bool(args.manifest):
            raise PromptBudgetError("--artifact and --manifest must be provided together")
        if args.artifact:
            artifact = load_artifact(Path(args.artifact), Path(args.manifest))
        else:
            artifact_resource, manifest_resource = _resource_paths()
            with resources.as_file(artifact_resource) as artifact_path, resources.as_file(manifest_resource) as manifest_path:
                artifact = load_artifact(Path(artifact_path), Path(manifest_path))
        policy = load_policy(Path(args.policy)) if args.policy else load_bundled_policy()
        inputs = load_input(Path(args.input))
        models = [select_model(to_prompt_record(episode).text, args.tier, artifact, policy) for episode in inputs.episodes]
        submission = to_submission(inputs, args.tier, models, policy.policy_id)
        _write_atomic(Path(args.output), dumps_json(submission_to_dict(submission)))
        return 0
    except (OSError, ProtocolError, PromptBudgetError) as exc:
        print(f"router-run: {exc}", file=__import__("sys").stderr)
        return 2
