# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Boundary contracts for the hash-regex nested evaluator command."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_hash_regex_quality_nested", TOOLS / "evaluate_hash_regex_quality_nested.py"
)
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
assert SPEC.loader is not None
SPEC.loader.exec_module(cli)


class EvaluateHashRegexQualityNestedTest(unittest.TestCase):
    def test_evaluator_rejects_dev_path_before_loading(self) -> None:
        with self.assertRaises(ValueError):
            cli.reject_dev_path(Path("data/materialized/dev/inputs.json"))

    def test_default_dry_run_writes_no_report(self) -> None:
        args = cli._parser().parse_args((
            "--input", "data/materialized/train/inputs.json",
            "--outcomes", "data/train/outcomes.json",
            "--report", "build/hash-regex-quality/test-report.json",
        ))
        self.assertFalse(args.execute)
        self.assertFalse(args.full)

    def test_report_root_is_isolated(self) -> None:
        with self.assertRaises(ValueError):
            cli.require_output_path(Path("build/promptbudget-v3/report.json"))


if __name__ == "__main__":
    unittest.main()
