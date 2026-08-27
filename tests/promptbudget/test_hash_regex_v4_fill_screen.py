import dataclasses
import unittest
from decimal import Decimal

import numpy as np

import hash_regex_tail_guard_nested as tail_guard
import hash_regex_v4_fill_screen as screen
from ossp_router.protocol import MODEL_IDS, Episode, InputBatch, Outcome, OutcomeBatch, load_bundled_policy
from promptbudget.safety import grouped_folds


class HashRegexV4FillScreenTests(unittest.TestCase):
    def test_candidate_ids_are_fixed(self):
        self.assertEqual(
            (
                "fast-ax31-fill",
                "balanced-ax31-fill",
                "fast-balanced-ax31-fill",
            ),
            screen.CANDIDATES,
        )

    def test_fast_fill_only_promotes_light_to_ax31(self):
        data = self._data(4)
        guard = self._guard()
        scores = np.asarray(
            [[0.1, 0.9, 0.0], [0.1, 0.8, 0.0], [0.1, 0.7, 0.0], [0.1, 0.6, 0.0]]
        )
        costs = np.log(np.asarray([[1.0, 1.1, 100.0]] * 4))

        routes = screen.route_guarded_candidate(
            data, (0, 1, 2, 3), scores, costs, guard, "fast-ax31-fill"
        )

        self.assertEqual(routes["balanced"].choices, routes["baseline"]["balanced"].choices)
        self.assertEqual(routes["premium"].choices, routes["baseline"]["premium"].choices)
        self.assertTrue(all(model == MODEL_IDS[1] for model in routes["fast"].added_models))
        self.assertTrue(
            all(
                before == after or (before == MODEL_IDS[0] and after == MODEL_IDS[1])
                for before, after in zip(routes["baseline"]["fast"].choices, routes["fast"].choices)
            )
        )

    def test_screen_requires_twelve_actual_cap_checks(self):
        data = self._data(12)
        folds = grouped_folds(data.groups, folds=4, seed=137)

        result = screen.screen_candidate(data, folds, "balanced-ax31-fill")

        self.assertEqual(12, result["actual_checks_required"])
        self.assertEqual(12, len(result["actual_checks"]))
        self.assertEqual(4, len(result["fold_deltas"]["balanced"]))

    def test_screen_rejects_duplicate_validation_group(self):
        data = self._data(12)
        folds = grouped_folds(data.groups, folds=4, seed=137)
        malformed = (folds[0], folds[0], folds[2], folds[3])

        with self.assertRaises(ValueError):
            screen.screen_candidate(data, malformed, "balanced-ax31-fill")

    def test_public_entry_points_reject_dev_data(self):
        train = self._data(12)
        dev = dataclasses.replace(
            train,
            inputs=dataclasses.replace(train.inputs, split="dev"),
            outcomes=dataclasses.replace(train.outcomes, split="dev"),
        )
        scores = train.scores[:4]
        costs = train.log_costs[:4]

        with self.assertRaises(ValueError):
            screen.route_guarded_candidate(
                dev, (0, 1, 2, 3), scores, costs, self._guard(), "fast-ax31-fill"
            )
        with self.assertRaises(ValueError):
            screen.screen_candidate(dev, grouped_folds(dev.groups, folds=4, seed=137), "fast-ax31-fill")

    @staticmethod
    def _guard():
        return tail_guard.TailGuard(
            (-1.0, 0.0, 1.0),
            (-1.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        )

    @staticmethod
    def _data(rows):
        policy = load_bundled_policy()
        inputs = InputBatch(
            schema_version=policy.schema_version,
            challenge_id="v4-fill-screen-test",
            split="train",
            episodes=tuple(Episode(f"episode-{index}", prompt=f"prompt-{index}") for index in range(rows)),
        )
        outcomes = OutcomeBatch(
            schema_version=inputs.schema_version,
            challenge_id=inputs.challenge_id,
            split=inputs.split,
            outcomes=tuple(
                Outcome(
                    episode.episode_id,
                    model_id,
                    Decimal("0.40") + Decimal(model_index) / Decimal("10"),
                    1,
                    0 if model_id == MODEL_IDS[0] else 1_000_000,
                    1_000_000 if model_id == MODEL_IDS[0] else 0,
                )
                for episode in inputs.episodes
                for model_index, model_id in enumerate(MODEL_IDS)
            ),
        )
        matrix = np.arange(rows * 270, dtype=float).reshape(rows, 270) / 100.0
        scores = np.column_stack(
            (np.full(rows, 0.2), np.full(rows, 0.7), np.full(rows, 0.1))
        )
        log_costs = np.column_stack(
            (np.zeros(rows), np.linspace(0.1, 0.9, rows), np.linspace(2.0, 2.8, rows))
        )
        costs = np.exp(log_costs)
        return screen.v32.EvaluationData(
            matrix, tuple(f"group-{index}" for index in range(rows)), scores, log_costs, costs,
            policy, inputs, outcomes,
        )


if __name__ == "__main__":
    unittest.main()
