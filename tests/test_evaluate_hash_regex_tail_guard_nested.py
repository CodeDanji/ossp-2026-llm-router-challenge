import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


TOOLS = Path(__file__).parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_hash_regex_tail_guard_nested", TOOLS / "evaluate_hash_regex_tail_guard_nested.py"
)
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
assert SPEC.loader is not None
SPEC.loader.exec_module(cli)


class EvaluateHashRegexTailGuardNestedTest(unittest.TestCase):
    def test_evaluator_rejects_diagnostic_without_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            diagnostic = Path(directory) / "diagnostic.json"
            diagnostic.write_text(json.dumps({"terminal_status": "tail-no-signal"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                cli._load_diagnostic(diagnostic, "input", "outcomes")

    def test_output_root_is_isolated(self):
        with self.assertRaises(ValueError):
            cli.require_report_path(Path("build/hash-regex-cost-stabilization/report.json"))

    def test_dry_run_does_not_write_full_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "build" / "hash-regex-tail-guard" / "report.json"
            args = cli._parser().parse_args((
                "--input", "data/materialized/train/inputs.json",
                "--outcomes", "data/train/outcomes.json",
                "--diagnostic", "build/hash-regex-tail-guard/tail-diagnostic.json",
                "--report", str(report),
            ))
            batch = type("Batch", (), {"split": "train"})()
            with mock.patch.object(cli, "_sha256", return_value="hash"), mock.patch.object(
                cli, "_load_diagnostic", return_value={}
            ), mock.patch.object(cli, "load_input", return_value=batch), mock.patch.object(
                cli, "load_outcomes", return_value=batch
            ), mock.patch.object(cli, "validate_batches", return_value=(1760, None)):
                result = cli.evaluate(args)

        self.assertEqual("dry-run", result["mode"])
        self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
