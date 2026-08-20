from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from ossp_router.protocol import load_bundled_policy, load_submission, policy_sha256
from promptbudget.artifact import LinearHead, PromptBudgetArtifact, TierSettings, write_artifact
from promptbudget.runtime import main
from promptbudget.text_features import DENSE_FEATURE_NAMES


class RuntimeIntegrationTest(unittest.TestCase):
    def test_reproducible_id_keyed_submission_and_atomic_output(self):
        policy = load_bundled_policy()
        head = lambda value: LinearHead(value, (0.0,) * len(DENSE_FEATURE_NAMES), {})
        artifact = PromptBudgetArtifact(
            2**16, DENSE_FEATURE_NAMES, policy.policy_id, policy_sha256(policy),
            {m: head(0.5 if m == "ax31-light" else 0.6) for m in policy.models},
            {m: head(1.0) for m in policy.models}, head(1.0),
            {m: 1.0 for m in policy.models},
            {t: TierSettings(0.0, 0.0, 0.0, 1.0, 10.0) for t in policy.tiers},
            "absolute-linear", "test", {"source": "integration"},
        )
        root = Path(__file__).parents[2]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            artifact_path, manifest_path = target / "artifact.json", target / "manifest.json"
            write_artifact(artifact_path, manifest_path, artifact)
            input_path = target / "inputs.json"
            input_path.write_bytes((root / "data/toy/inputs.json").read_bytes())
            one, two = target / "one.json", target / "two.json"
            explicit = ["--input", str(input_path), "--tier", "balanced", "--output", str(one), "--artifact", str(artifact_path), "--manifest", str(manifest_path)]
            self.assertEqual(0, main(explicit))
            if os.name != "nt":
                self.assertEqual(0o644, stat.S_IMODE(one.stat().st_mode))
            self.assertEqual(0, main(explicit[:-1] + [str(manifest_path), "--output", str(two)]))
            self.assertEqual(load_submission(one), load_submission(two))
            permuted_input = target / "permuted-inputs.json"
            permuted = json.loads(input_path.read_text(encoding="utf-8"))
            permuted["episodes"] = list(reversed(permuted["episodes"]))
            permuted_input.write_text(json.dumps(permuted), encoding="utf-8")
            permuted_output = target / "permuted.json"
            permuted_args = ["--input", str(permuted_input), "--tier", "balanced", "--output", str(permuted_output), "--artifact", str(artifact_path), "--manifest", str(manifest_path)]
            self.assertEqual(0, main(permuted_args))
            original_by_id = {d.episode_id: d.model_id for d in load_submission(one).decisions}
            permuted_by_id = {d.episode_id: d.model_id for d in load_submission(permuted_output).decisions}
            self.assertEqual(original_by_id, permuted_by_id)
            self.assertFalse(any(p.name.startswith(".") for p in target.iterdir()))


if __name__ == "__main__":
    unittest.main()
