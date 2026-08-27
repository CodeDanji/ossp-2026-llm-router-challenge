import math
import unittest
import dataclasses

import numpy as np

import hash_regex_cost_stabilization_nested as v32
import hash_regex_tail_guard_nested as nested


class HashRegexTailGuardNestedTests(unittest.TestCase):
    def test_guard_uses_model_specific_predicted_log_cost_buckets(self):
        guard = nested.TailGuard(
            ax31_edges=(-2.0, 0.0, 2.0),
            think_edges=(-1.0, 1.0, 3.0),
            ax31_log_guards=(0.0, 0.1, 0.2, 0.3),
            think_log_guards=(0.0, 0.4, 0.5, 0.6),
        )

        result = nested.apply_tail_guard(
            base_costs=np.asarray([[1.0, 2.0, 3.0]]),
            base_log_costs=np.asarray([[0.0, 0.2, 1.2]]),
            guard=guard,
        )

        self.assertEqual(1.0, result[0, 0])
        self.assertAlmostEqual(2.0 * math.exp(0.2), result[0, 1])
        self.assertAlmostEqual(3.0 * math.exp(0.5), result[0, 2])

    def test_guard_is_nonnegative_and_reapplies_cost_order(self):
        guard = nested.TailGuard(
            ax31_edges=(0.0, 1.0, 2.0),
            think_edges=(0.0, 1.0, 2.0),
            ax31_log_guards=(0.0, 0.0, 0.0, 0.0),
            think_log_guards=(0.0, 0.0, 0.0, 0.0),
        )

        with self.assertRaises(ValueError):
            nested.TailGuard(
                (0.0, 1.0, 2.0),
                (0.0, 1.0, 2.0),
                (-0.1, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            )

        guarded = nested.apply_tail_guard(
            np.asarray([[2.0, 1.0, 1.5]]), np.zeros((1, 3)), guard
        )
        self.assertGreater(guarded[0, 1], guarded[0, 0])
        self.assertGreater(guarded[0, 2], guarded[0, 1])

    def test_fit_tail_guard_uses_only_partition_oof_residuals(self):
        data = self._data(12)
        partition = tuple(range(8))
        first = nested.fit_tail_guard(data, partition, seed=137)
        changed_logs = data.log_costs.copy()
        changed_logs[8:] += 9.0
        changed = dataclasses.replace(data, log_costs=changed_logs)

        self.assertEqual(first, nested.fit_tail_guard(changed, partition, seed=137))

    def test_equal_count_edges_use_bisect_right_and_report_tied_counts(self):
        summary = nested.bucket_residual_summary(
            np.asarray([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]),
            np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        )

        self.assertEqual(4, len(summary["counts"]))
        self.assertEqual(6, sum(summary["counts"]))
        self.assertEqual(2, nested.bucket_index(1.0, (0.0, 1.0, 2.0)))

    def test_signal_requires_spread_for_two_grouped_oof_seeds(self):
        reports = {
            137: {"ax31": {"p90": (0.0, 0.0, math.log(1.11), math.log(1.11))}},
            271: {"ax31": {"p90": (0.0, 0.0, math.log(1.11), math.log(1.11))}},
            811: {"ax31": {"p90": (0.0, 0.0, 0.0, 0.0)}},
        }

        self.assertEqual("tail-signal-present", nested.diagnostic_status(reports))

    @staticmethod
    def _data(rows):
        matrix = np.arange(rows * 270, dtype=float).reshape(rows, 270) / 100.0
        log_costs = np.column_stack(
            (
                np.linspace(0.0, 0.1, rows),
                np.linspace(0.1, 1.1, rows),
                np.linspace(0.2, 1.2, rows),
            )
        )
        return v32.EvaluationData(
            matrix=matrix,
            groups=tuple(f"group-{index}" for index in range(rows)),
            scores=np.zeros((rows, 3)),
            log_costs=log_costs,
            costs=np.exp(log_costs),
            policy=None,
            inputs=None,
            outcomes=None,
        )


if __name__ == "__main__":
    unittest.main()
