import dataclasses
import unittest

import numpy as np

import hash_regex_cost_stabilization_nested as nested


class HashRegexCostStabilizationNestedTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.arange(8 * 270, dtype=float).reshape(8, 270) / 100.0
        self.scores = np.column_stack(
            (
                np.linspace(0.2, 0.8, 8),
                np.linspace(0.3, 0.9, 8),
                np.linspace(0.4, 1.0, 8),
            )
        )

    def test_multiplier_only_changes_upgrade_costs_and_restores_order(self):
        result = nested.apply_cost_multipliers(
            np.asarray([[2.0, 1.0, 1.5]]), ax31=1.25, think=1.50
        )
        self.assertEqual(2.0, result[0, 0])
        self.assertGreater(result[0, 1], result[0, 0])
        self.assertGreater(result[0, 2], result[0, 1])

    def test_raw_fit_cannot_read_held_out_labels(self):
        first = nested.fit_raw_quality_heads(self.matrix[:4], self.scores[:4])
        changed = self.scores.copy()
        changed[4:] = 99.0
        second = nested.fit_raw_quality_heads(self.matrix[:4], changed[:4])
        np.testing.assert_allclose(
            nested.predict_heads(first, self.matrix[4:]),
            nested.predict_heads(second, self.matrix[4:]),
        )

    def test_log_cost_fit_predict_and_shapes(self):
        costs = np.asarray(
            [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [2.5, 3.5, 4.5]]
        )
        heads = nested.fit_log_cost_heads(self.matrix[:4], np.log(costs))
        prediction = nested.predict_heads(heads, self.matrix[4:])
        self.assertEqual((4, 3), prediction.shape)
        self.assertIsInstance(heads, nested.LinearHeads)
        self.assertTrue(dataclasses.is_dataclass(heads))

    def test_fit_rejects_bad_targets_and_costs(self):
        with self.assertRaises(ValueError):
            nested.fit_raw_quality_heads(self.matrix[:4], self.scores[:3])
        for bad in (np.zeros((1, 3)), np.asarray([[1.0, -1.0, 2.0]]), np.asarray([[1.0, np.nan, 2.0]])):
            with self.assertRaises(ValueError):
                nested.apply_cost_multipliers(bad, ax31=1.25, think=1.5)

    def test_fitted_arrays_are_read_only(self):
        heads = nested.fit_raw_quality_heads(self.matrix[:4], self.scores[:4])
        with self.assertRaises((ValueError, TypeError)):
            heads.mean[0] = 0.0


if __name__ == "__main__":
    unittest.main()
