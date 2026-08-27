import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


TOOLS = Path(__file__).parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "diagnose_hash_regex_tail_guard", TOOLS / "diagnose_hash_regex_tail_guard.py"
)
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
assert SPEC.loader is not None
SPEC.loader.exec_module(cli)


class DiagnoseHashRegexTailGuardTest(unittest.TestCase):
    def test_dev_path_is_rejected_before_load(self):
        args = cli._parser().parse_args((
            "--input", "data/materialized/dev/inputs.json",
            "--outcomes", "data/train/outcomes.json",
            "--report", "build/hash-regex-tail-guard/tail-diagnostic.json",
        ))
        with mock.patch.object(cli, "load_input", side_effect=AssertionError("must not load")):
            with self.assertRaises(ValueError):
                cli.diagnose(args)

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "build" / "hash-regex-tail-guard" / "report.json"
            args = cli._parser().parse_args((
                "--input", "data/materialized/train/inputs.json",
                "--outcomes", "data/train/outcomes.json",
                "--report", str(report),
            ))
            batch = type("Batch", (), {"split": "train"})()
            with mock.patch.object(cli, "load_input", return_value=batch), mock.patch.object(
                cli, "load_outcomes", return_value=batch
            ), mock.patch.object(cli, "validate_batches", return_value=(1760, None)), mock.patch.object(
                cli, "_sha256", return_value="hash"
            ):
                result = cli.diagnose(args)

        self.assertEqual("dry-run", result["mode"])
        self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
