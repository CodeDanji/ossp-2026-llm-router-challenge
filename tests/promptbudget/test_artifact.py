# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Artifact persistence contracts."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from ossp_router.protocol import load_bundled_policy, policy_sha256
from promptbudget.artifact import (
    LinearHead,
    PromptBudgetArtifact,
    TierSettings,
    load_artifact,
    write_artifact,
)
from promptbudget.schema import PromptBudgetError
from promptbudget.text_features import DENSE_FEATURE_NAMES


class ArtifactTest(unittest.TestCase):
    def test_writes_loads_and_rejects_a_tampered_manifest(self) -> None:
        policy = load_bundled_policy()
        head = LinearHead(0.0, (0.0,) * len(DENSE_FEATURE_NAMES), {})
        artifact = PromptBudgetArtifact(
            hash_dimension=2**16,
            dense_feature_names=DENSE_FEATURE_NAMES,
            policy_id=policy.policy_id,
            policy_sha256=policy_sha256(policy),
            quality_heads={model_id: head for model_id in policy.models},
            output_heads={model_id: head for model_id in policy.models},
            input_head=head,
            cost_residual_multipliers={model_id: 1.0 for model_id in policy.models},
            tiers={
                tier: TierSettings(0.0, 0.0, 0.0, 1.0, 1.0)
                for tier in policy.tiers
            },
            family="absolute-linear",
            code_version="test",
            training_provenance={"source": "test"},
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / "router.json"
            manifest_path = root / "router.manifest.json"
            write_artifact(artifact_path, manifest_path, artifact)
            self.assertEqual(artifact, load_artifact(artifact_path, manifest_path))
            valid_bytes = artifact_path.read_bytes()
            artifact_path.write_bytes(valid_bytes + b" ")
            with self.assertRaisesRegex(PromptBudgetError, "manifest"):
                load_artifact(artifact_path, manifest_path)
            artifact_path.write_bytes(valid_bytes)
            with self.assertRaises(PromptBudgetError):
                write_artifact(artifact_path, manifest_path, replace(artifact, training_provenance={"bad": object()}))
            manifest_path.write_text(
                '{"artifact_file":"router.json","artifact_sha256":"'
                + "0" * 64
                + '","format_version":1}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PromptBudgetError, "manifest"):
                load_artifact(artifact_path, manifest_path)


if __name__ == "__main__":
    unittest.main()
