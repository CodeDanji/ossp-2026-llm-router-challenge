# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Boundary contracts for the guarded cost-stabilization evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


TOOLS = Path(__file__).parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_hash_regex_cost_stabilization_nested",
    TOOLS / "evaluate_hash_regex_cost_stabilization_nested.py",
)
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
assert SPEC.loader is not None
SPEC.loader.exec_module(cli)


class EvaluateHashRegexCostStabilizationNestedTest(unittest.TestCase):
    def test_dev_path_is_rejected_before_loading(self) -> None:
        args = cli._parser().parse_args((
            "--input", "data/dev/inputs.json",
            "--outcomes", "data/train/outcomes.json",
            "--report", "build/hash-regex-cost-stabilization/report.json",
        ))
        with mock.patch.object(cli, "load_input", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "Dev"):
                cli.evaluate(args)

    def test_dry_run_writes_no_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "build" / "hash-regex-cost-stabilization" / "dry-run.json"
            args = cli._parser().parse_args((
                "--input", "data/materialized/train/inputs.json",
                "--outcomes", "data/train/outcomes.json",
                "--report", str(report),
            ))
            result = cli.evaluate(args)
            self.assertEqual("dry-run", result["mode"])
            self.assertFalse(report.exists())

    def test_report_root_is_isolated(self) -> None:
        with self.assertRaises(ValueError):
            cli.require_output_path(Path("build/promptbudget-v3/report.json"))

    def test_grid_selection_rejects_a_malformed_check_list(self) -> None:
        malformed = {
            "inner_folds": ({}, {}, {}, {}),
            "admission": {
                "required_checks": 12,
                "checks": tuple({} for _ in range(11)),
                "admitted": False,
            },
        }
        with mock.patch.object(cli.nested, "evaluate_inner_pair", return_value=malformed):
            with self.assertRaisesRegex(ValueError, "twelve checks"):
                cli._validated_grid_selection(object(), (), 137)


if __name__ == "__main__":
    unittest.main()
