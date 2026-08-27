import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

import numpy as np

import screen_hash_regex_v4_fill as cli
from ossp_router.protocol import load_bundled_policy


class ScreenHashRegexV4FillTest(unittest.TestCase):
    def test_dry_run_is_train_only_and_does_not_write_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "build" / "hash-regex-v4-deadline-triage" / "screening.json"
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
                result = cli.screen(args)
            self.assertFalse(report.exists())

        self.assertEqual("not-executed", result["terminal_status"])

    def test_cli_rejects_dev_and_nonisolated_output(self):
        args = cli._parser().parse_args((
            "--input", "data/materialized/dev/inputs.json",
            "--outcomes", "data/train/outcomes.json",
            "--report", "build/out.json",
        ))

        with self.assertRaises(ValueError):
            cli.screen(args)

        nonisolated = cli._parser().parse_args((
            "--input", "data/materialized/train/inputs.json",
            "--outcomes", "data/train/outcomes.json",
            "--report", "build/out.json",
        ))
        with self.assertRaises(ValueError):
            cli.screen(nonisolated)

    def test_no_recovery_signal_skips_candidate_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "build" / "hash-regex-v4-deadline-triage" / "screening.json"
            args = cli._parser().parse_args((
                "--input", "data/materialized/train/inputs.json",
                "--outcomes", "data/train/outcomes.json",
                "--report", str(report),
                "--execute",
            ))
            batch = type("Batch", (), {"split": "train"})()
            log_costs = np.column_stack((
                np.zeros(12), np.linspace(0.1, 0.9, 12), np.linspace(1.0, 1.8, 12),
            ))
            data = cli.v32.EvaluationData(
                np.arange(12 * 270, dtype=float).reshape(12, 270) / 100.0,
                tuple(f"group-{index}" for index in range(12)), np.zeros((12, 3)),
                log_costs, np.exp(log_costs), load_bundled_policy(),
                type("Inputs", (), {"episodes": tuple(range(12))})(), object(),
            )
            with mock.patch.object(cli, "load_input", return_value=batch), mock.patch.object(
                cli, "load_outcomes", return_value=batch
            ), mock.patch.object(cli, "validate_batches", return_value=(1760, None)), mock.patch.object(
                cli, "_sha256", return_value="hash"
            ), mock.patch.object(cli.v32, "make_evaluation_data", return_value=data), mock.patch.object(
                cli.v4, "_score_routes", return_value={"tiers": {
                    tier: {"total_cost": "1.00", "budget_limit": "2.00"}
                    for tier in ("fast", "balanced", "premium")
                }}
            ), mock.patch.object(
                cli, "screen_candidate", side_effect=AssertionError("must not screen")
            ):
                result = cli.screen(args)

            self.assertTrue(report.exists())

        self.assertEqual("no-recovery-signal", result["terminal_status"])

    def test_winner_uses_weighted_changed_tier_delta(self):
        fast = {
            "candidate": "fast-ax31-fill",
            "changed_tier_pooled_cap_neutral_quality_deltas": {"fast": Decimal("0.08")},
            "actual_checks": ({"actual_ratio": Decimal("1.01")},),
        }
        combined = {
            "candidate": "fast-balanced-ax31-fill",
            "changed_tier_pooled_cap_neutral_quality_deltas": {
                "fast": Decimal("0.07"), "balanced": Decimal("0.07"),
            },
            "actual_checks": ({"actual_ratio": Decimal("1.01")},),
        }

        self.assertEqual(combined, min((fast, combined), key=cli._winner_key))

    def test_recoverability_uses_official_decimal_cost_slack(self):
        data = type("Data", (), {"scores": np.asarray([[0.0, 1.0, 0.0]]), "costs": np.asarray([[1.0, 999.0, 999.0]])})()
        baseline = {tier: cli.v4.Route(("axk1-low",), 0.0) for tier in ("fast", "balanced", "premium")}
        metrics = {"total_cost": "9.00", "budget_limit": "10.00"}
        with mock.patch.object(cli.v4, "_score_routes", return_value={"tiers": {"fast": {"total_cost": "9.50"}}}):
            slack, count = cli._official_recoverable_ax31_pool(data, (0,), baseline, "fast", metrics)

        self.assertEqual(Decimal("1.00"), slack)
        self.assertEqual(1, count)


if __name__ == "__main__":
    unittest.main()
