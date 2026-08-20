# SPDX-License-Identifier: Apache-2.0
"""Validate a complete public Train input/outcome matrix without disclosing rows."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Sequence, Tuple

from ossp_router.protocol import MODEL_IDS, InputBatch, OutcomeBatch, ProtocolError, load_input, load_outcomes


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_batches(inputs: InputBatch, outcomes: OutcomeBatch) -> Tuple[int, int]:
    """Return aggregate dimensions after enforcing the exact Train outcome matrix."""

    if inputs.schema_version != outcomes.schema_version:
        raise ProtocolError("input and outcomes schema_version do not match")
    if inputs.challenge_id != outcomes.challenge_id:
        raise ProtocolError("input and outcomes challenge_id do not match")
    if inputs.split != outcomes.split:
        raise ProtocolError("input and outcomes split do not match")
    if inputs.split != "train":
        raise ProtocolError("Train-only tooling requires split='train'")
    expected = {
        (episode.episode_id, model_id)
        for episode in inputs.episodes
        for model_id in MODEL_IDS
    }
    actual = {(outcome.episode_id, outcome.model_id) for outcome in outcomes.outcomes}
    if len(outcomes.outcomes) != len(expected) or actual != expected:
        raise ProtocolError("outcomes must contain exactly every input episode and model")
    if any(
        outcome.num_generations <= 0
        or outcome.input_tokens <= 0
        or outcome.output_tokens <= 0
        for outcome in outcomes.outcomes
    ):
        raise ProtocolError("outcome generations and token counts must be positive")
    return len(inputs.episodes), len(outcomes.outcomes)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate public Train data aggregates.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        outcomes = load_outcomes(args.outcomes)
        rows, outcome_rows = validate_batches(inputs, outcomes)
    except (OSError, ProtocolError, ValueError) as exc:
        del exc
        print("ERROR: validation failed", file=sys.stderr)
        return 2
    print("OK: train_episodes={0} outcome_rows={1} models={2}".format(rows, outcome_rows, len(MODEL_IDS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
