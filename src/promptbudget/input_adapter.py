# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Adapters between the frozen router protocol and PromptBudget values."""

from __future__ import annotations

from typing import Iterable

from ossp_router.protocol import (
    Episode,
    InputBatch,
    ProtocolError,
    Submission,
    parse_submission,
)

from .schema import MODEL_IDS, TIERS, PromptBudgetError, PromptRecord


def to_prompt_record(episode: Episode) -> PromptRecord:
    """Return an episode's policy text and its output-only reassembly key."""

    has_prompt = episode.prompt is not None
    has_messages = episode.messages is not None
    if has_prompt == has_messages:
        raise PromptBudgetError(
            "episode must contain exactly one of prompt or messages."
        )
    if has_prompt:
        return PromptRecord(text=episode.prompt, output_key=episode.episode_id)

    return PromptRecord(
        text="\n".join(
            f"<role>{message.role}</role>\n{message.content}"
            for message in episode.messages
        ),
        output_key=episode.episode_id,
    )


def to_submission(
    input_batch: InputBatch,
    tier: str,
    model_ids: Iterable[str],
    policy_id: str,
) -> Submission:
    """Build a protocol-valid submission from ordered model selections."""

    if tier not in TIERS:
        raise PromptBudgetError(f"unknown tier: {tier!r}")
    try:
        selected_models = tuple(model_ids)
    except TypeError as exc:
        raise PromptBudgetError("model_ids must be an iterable of model IDs.") from exc
    if len(selected_models) != len(input_batch.episodes):
        raise PromptBudgetError(
            "model_ids count must match the number of input episodes."
        )
    for model_id in selected_models:
        if model_id not in MODEL_IDS:
            raise PromptBudgetError(f"unknown model_id: {model_id!r}")

    value = {
        "schema_version": input_batch.schema_version,
        "challenge_id": input_batch.challenge_id,
        "policy_id": policy_id,
        "split": input_batch.split,
        "tier": tier,
        "decisions": [
            {"episode_id": episode.episode_id, "model_id": model_id}
            for episode, model_id in zip(input_batch.episodes, selected_models)
        ],
    }
    try:
        return parse_submission(value)
    except ProtocolError as exc:
        raise PromptBudgetError(f"invalid submission: {exc}") from exc
