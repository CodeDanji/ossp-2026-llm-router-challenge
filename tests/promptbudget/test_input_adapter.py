# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PromptBudget input and submission adapters."""

from __future__ import annotations

import unittest

import ossp_router.protocol
import promptbudget.schema
from ossp_router.protocol import Episode, InputBatch, Message
from promptbudget.input_adapter import to_prompt_record, to_submission
from promptbudget.schema import PromptBudgetError, PromptRecord


class InputAdapterTest(unittest.TestCase):
    def test_schema_reexports_protocol_constants_by_identity(self) -> None:
        self.assertIs(
            promptbudget.schema.MODEL_IDS,
            ossp_router.protocol.MODEL_IDS,
        )
        self.assertIs(
            promptbudget.schema.TIERS,
            ossp_router.protocol.TIERS,
        )

    def test_prompt_record_preserves_prompt_text_and_episode_id(self) -> None:
        episode = Episode(
            episode_id="opaque-episode-id",
            prompt="  preserve this text exactly\n",
        )

        self.assertEqual(
            PromptRecord("  preserve this text exactly\n", "opaque-episode-id"),
            to_prompt_record(episode),
        )

    def test_prompt_record_flattens_messages_in_original_order(self) -> None:
        episode = Episode(
            episode_id="opaque-episode-id",
            messages=(
                Message("system", "Follow the instructions."),
                Message("user", "Explain\nthis."),
            ),
        )

        self.assertEqual(
            PromptRecord(
                "<role>system</role>\nFollow the instructions.\n"
                "<role>user</role>\nExplain\nthis.",
                "opaque-episode-id",
            ),
            to_prompt_record(episode),
        )

    def test_output_keys_do_not_change_policy_relevant_text(self) -> None:
        prompt_record = to_prompt_record(
            Episode("prompt-key", prompt="<role>user</role>\nSame body")
        )
        messages_record = to_prompt_record(
            Episode(
                "messages-key",
                messages=(Message("user", "Same body"),),
            )
        )

        self.assertNotEqual(prompt_record.output_key, messages_record.output_key)
        self.assertEqual(prompt_record.text, messages_record.text)

    def test_submission_rejects_unknown_model_id(self) -> None:
        with self.assertRaisesRegex(PromptBudgetError, "model_id"):
            to_submission(self._batch(), "fast", ("unknown-model",), "test-policy")

    def test_submission_rejects_model_count_that_differs_from_episodes(self) -> None:
        with self.assertRaisesRegex(PromptBudgetError, "count"):
            to_submission(self._batch(), "fast", (), "test-policy")

    def test_submission_rejects_unknown_tier(self) -> None:
        with self.assertRaisesRegex(PromptBudgetError, "tier"):
            to_submission(
                self._batch(), "not-a-tier", ("ax31-light",), "test-policy"
            )

    def test_submission_uses_input_header_and_original_episode_order(self) -> None:
        batch = InputBatch(
            schema_version=1,
            challenge_id="challenge-id",
            split="dev",
            episodes=(
                Episode("second-in-source", prompt="second"),
                Episode("first-in-source", prompt="first"),
            ),
        )

        submission = to_submission(
            batch,
            "balanced",
            ("ax31-light", "axk1-think"),
            "test-policy",
        )

        self.assertEqual(1, submission.schema_version)
        self.assertEqual("challenge-id", submission.challenge_id)
        self.assertEqual("test-policy", submission.policy_id)
        self.assertEqual("dev", submission.split)
        self.assertEqual("balanced", submission.tier)
        self.assertEqual(
            (("second-in-source", "ax31-light"), ("first-in-source", "axk1-think")),
            tuple((decision.episode_id, decision.model_id) for decision in submission.decisions),
        )

    @staticmethod
    def _batch() -> InputBatch:
        return InputBatch(
            schema_version=1,
            challenge_id="challenge-id",
            split="dev",
            episodes=(Episode("opaque-episode-id", prompt="text"),),
        )


if __name__ == "__main__":
    unittest.main()
