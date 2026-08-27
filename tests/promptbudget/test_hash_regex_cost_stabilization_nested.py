import dataclasses
from decimal import Decimal
import unittest

import numpy as np

import hash_regex_cost_stabilization_nested as nested
from ossp_router.protocol import (
    MODEL_IDS,
    Episode,
    InputBatch,
    Outcome,
    OutcomeBatch,
    load_bundled_policy,
)


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

    def test_premium_admission_scores_after_ax31_fill(self):
        data, scores, log_costs = self._policy_data()
        report = nested.score_batch_policy(
            data, (0, 1), scores, log_costs, (1.0, 1.0)
        )

        self.assertTrue(report["tiers"]["premium"]["includes_premium_fill"])

    def test_premium_report_includes_fill_when_it_keeps_all_light(self):
        data, scores, log_costs = self._policy_data()
        report = nested.score_batch_policy(
            data, (0, 1), np.zeros_like(scores), log_costs, (1.0, 1.0)
        )

        self.assertTrue(report["tiers"]["premium"]["includes_premium_fill"])

    def test_actual_scorer_not_predicted_ratio_decides_admission(self):
        data, scores, log_costs = self._policy_data()
        admission_scores = np.asarray([[0.5, 0.5, 0.0], [0.5, 0.9, 0.0]])
        admission_log_costs = np.log(np.asarray([[1.0, 1.1, 100.0]] * 2))
        report = nested.score_batch_policy(
            data, (0, 1), admission_scores, admission_log_costs, (1.10, 1.25)
        )

        self.assertTrue(report["tiers"]["fast"]["budget_passed"])
        self.assertEqual("0.765875", report["tiers"]["fast"]["actual_ratio"])
        self.assertAlmostEqual(1.105, report["tiers"]["fast"]["predicted_ratio"])

    def test_complete_policy_is_permutation_invariant(self):
        data, scores, log_costs = self._policy_data()
        original = nested.score_batch_policy(
            data, (0, 1), scores, log_costs, (1.10, 1.25)
        )
        permuted = nested.score_batch_policy(
            data, (1, 0), scores[::-1], log_costs[::-1], (1.10, 1.25)
        )

        self.assertEqual(
            {
                tier: original["tiers"][tier]["model_counts"]
                for tier in ("fast", "balanced", "premium")
            },
            {
                tier: permuted["tiers"][tier]["model_counts"]
                for tier in ("fast", "balanced", "premium")
            },
        )

    @staticmethod
    def _policy_data():
        policy = load_bundled_policy()
        inputs = InputBatch(
            schema_version=policy.schema_version,
            challenge_id="cost-stabilization-test",
            split="train",
            episodes=(Episode("first", prompt="first"), Episode("second", prompt="second")),
        )
        outcomes = OutcomeBatch(
            schema_version=inputs.schema_version,
            challenge_id=inputs.challenge_id,
            split=inputs.split,
            outcomes=tuple(
                Outcome(
                    episode.episode_id,
                    model_id,
                    Decimal("0.5") + Decimal(model_index) / Decimal("10"),
                    1,
                    0 if model_id == MODEL_IDS[0] else 1_000_000,
                    1_000_000 if model_id == MODEL_IDS[0] else 0,
                )
                for episode in inputs.episodes
                for model_index, model_id in enumerate(MODEL_IDS)
            ),
        )
        scores = np.asarray(
            [
                [0.4785362410675672, 0.12932670599894536, 0.14178118263552142],
                [0.6157451953355798, 0.6463632531114818, 0.9426475000532337],
            ]
        )
        costs = np.asarray(
            [
                [1.0, 2.62584337539718, 2.9322764565349932],
                [1.0, 3.319587604437482, 9.600077008387892],
            ]
        )
        log_costs = np.log(costs)
        return (
            nested.EvaluationData(
                np.zeros((2, 270)),
                ("first", "second"),
                scores,
                log_costs,
                costs,
                policy,
                inputs,
                outcomes,
            ),
            scores,
            log_costs,
        )


if __name__ == "__main__":
    unittest.main()
