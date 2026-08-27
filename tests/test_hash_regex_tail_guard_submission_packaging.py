"""Submission-package checks for the frozen PromptBudget v3.3 router."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines import hash_regex
from ossp_router.protocol import load_bundled_policy, load_input, submission_to_dict


ROOT = Path(__file__).parents[1]
RELEASE_DIR = ROOT / "build" / "hash-regex-tail-guard"
ARTIFACT = RELEASE_DIR / "final-artifact.json"
MANIFEST = RELEASE_DIR / "manifest.json"
FROZEN_SHA256 = "c60d38ce2df670e6206689392389b66d89f810936969bcf47c2a0f29b86b88ce"


class TailGuardSubmissionPackagingTest(unittest.TestCase):
    def test_release_manifest_binds_the_frozen_v33_artifact(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(
            {
                "artifact_file": "final-artifact.json",
                "artifact_sha256": FROZEN_SHA256,
                "artifact_type": "ossp-hash-regex-tail-guard-v1",
                "format_version": 1,
            },
            manifest,
        )
        self.assertEqual(FROZEN_SHA256, hashlib.sha256(ARTIFACT.read_bytes()).hexdigest())

    def test_docker_copies_the_v33_router_and_release_bundle(self) -> None:
        dockerfile = (ROOT / "container" / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("COPY --chown=65532:65532 baselines /opt/router/baselines", dockerfile)
        self.assertIn(
            "COPY --chown=65532:65532 build/hash-regex-tail-guard/final-artifact.json "
            "build/hash-regex-tail-guard/manifest.json "
            "/opt/router/build/hash-regex-tail-guard/",
            dockerfile,
        )

    def test_docker_context_includes_only_the_v33_release_bundle(self) -> None:
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

        self.assertIn("!build/", dockerignore)
        self.assertIn("build/**", dockerignore)
        self.assertIn("!build/hash-regex-tail-guard/", dockerignore)
        self.assertIn("build/hash-regex-tail-guard/**", dockerignore)
        self.assertIn(
            "!build/hash-regex-tail-guard/final-artifact.json", dockerignore
        )
        self.assertIn("!build/hash-regex-tail-guard/manifest.json", dockerignore)

    def test_container_entrypoint_uses_hash_regex_release_bundle_for_premium(self) -> None:
        from container import entrypoint

        inputs = ROOT / "data" / "toy" / "inputs.json"
        expected = hash_regex.make_hash_regex_submission(
            load_input(inputs),
            load_bundled_policy(),
            hash_regex.load_artifact(ARTIFACT),
            "premium",
        ).submission
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "submission.json"
            with mock.patch.object(
                hash_regex,
                "fill_ax31_upgrades",
                wraps=hash_regex.fill_ax31_upgrades,
            ) as fill:
                result = entrypoint.main(
                    ["--input", str(inputs), "--tier", "premium", "--output", str(output)]
                )

            self.assertEqual(0, result)
            fill.assert_called_once()
            self.assertEqual(
                submission_to_dict(expected),
                json.loads(output.read_text(encoding="utf-8")),
            )

    def test_entrypoint_resolves_the_runtime_root_after_docker_copy(self) -> None:
        from container import entrypoint

        self.assertEqual(
            Path("/opt/router/entrypoint.py").parent,
            entrypoint._runtime_root(Path("/opt/router/entrypoint.py")),
        )
        self.assertEqual(
            ROOT,
            entrypoint._runtime_root(ROOT / "container" / "entrypoint.py"),
        )

    def test_entrypoint_script_finds_the_bundled_hash_regex_router(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "submission.json"
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "container" / "entrypoint.py"),
                    "--input",
                    str(ROOT / "data" / "toy" / "inputs.json"),
                    "--tier",
                    "fast",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )

        self.assertEqual(0, completed.returncode, completed.stderr)


if __name__ == "__main__":
    unittest.main()
