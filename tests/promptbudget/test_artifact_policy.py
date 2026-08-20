# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for PromptBudget artifacts and text-only selection."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from ossp_router.protocol import load_bundled_policy, policy_sha256
from promptbudget.artifact import (
    LinearHead,
    PromptBudgetArtifact,
    TierSettings,
    load_artifact,
    write_artifact,
)
from promptbudget.policy import predict_models, select_model
from promptbudget.schema import PromptBudgetError
from promptbudget.text_features import DENSE_FEATURE_NAMES, extract_features


def _head(intercept: float) -> LinearHead:
    return LinearHead(intercept, (0.0,) * len(DENSE_FEATURE_NAMES), {})


def _artifact(
    *,
    qualities: tuple[float, float, float] = (0.5, 0.6, 0.7),
    tiers: dict[str, TierSettings] | None = None,
) -> PromptBudgetArtifact:
    policy = load_bundled_policy()
    model_ids = tuple(policy.models)
    return PromptBudgetArtifact(
        hash_dimension=2**16,
        dense_feature_names=DENSE_FEATURE_NAMES,
        policy_id=policy.policy_id,
        policy_sha256=policy_sha256(policy),
        quality_heads={
            model_id: _head(quality)
            for model_id, quality in zip(model_ids, qualities)
        },
        output_heads={model_id: _head(math.log1p(4.0)) for model_id in model_ids},
        input_head=_head(math.log1p(8.0)),
        cost_residual_multipliers={model_id: 1.0 for model_id in model_ids},
        tiers=tiers
        or {
            tier: TierSettings(0.0, 0.0, 0.0, 1.0, 10.0)
            for tier in policy.tiers
        },
        family="absolute-linear",
        code_version="test",
        training_provenance={"source": "test"},
    )


class ArtifactPolicyTest(unittest.TestCase):
    def test_round_trip_uses_bundled_policy_and_actual_text_features(self) -> None:
        routing_policy = load_bundled_policy()
        artifact = _artifact()
        vector = extract_features("Explain 2 + 2 in one sentence.", artifact.hash_dimension)

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "router.json"
            manifest_path = Path(temporary_directory) / "router.manifest.json"
            write_artifact(artifact_path, manifest_path, artifact)
            loaded = load_artifact(artifact_path, manifest_path)

        prediction = predict_models(vector.text, "balanced", loaded, routing_policy)[
            "ax31"
        ]
        self.assertEqual(artifact, loaded)
        self.assertEqual(len(DENSE_FEATURE_NAMES), len(vector.dense))
        self.assertAlmostEqual(0.6, prediction.quality)
        self.assertGreater(prediction.output_tokens, 0.0)
        self.assertGreater(prediction.c_upper, 0.0)

    def test_manifest_tampering_is_rejected(self) -> None:
        artifact = _artifact()
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "router.json"
            manifest_path = Path(temporary_directory) / "router.manifest.json"
            write_artifact(artifact_path, manifest_path, artifact)
            manifest_path.write_text(
                '{"artifact_file":"router.json","artifact_sha256":"'
                + "0" * 64
                + '","format_version":1}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PromptBudgetError, "manifest"):
                load_artifact(artifact_path, manifest_path)

    def test_selection_excludes_low_gain_and_breaks_equal_ties_by_protocol_order(self) -> None:
        artifact = _artifact(
            qualities=(0.2, 0.8, 0.8),
            tiers={
                "fast": TierSettings(0.0, 0.7, 0.7, 1.0, 10.0),
                "balanced": TierSettings(0.0, 0.0, 0.0, 1.0, 10.0),
                "premium": TierSettings(0.0, 0.0, 0.0, 1.0, 10.0),
            },
        )
        routing_policy = load_bundled_policy()

        self.assertEqual(
            "ax31-light", select_model("same prompt", "fast", artifact, routing_policy)
        )
        self.assertEqual(
            "ax31", select_model("same prompt", "balanced", artifact, routing_policy)
        )


if __name__ == "__main__":
    unittest.main()
