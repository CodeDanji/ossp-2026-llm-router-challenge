import math
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
