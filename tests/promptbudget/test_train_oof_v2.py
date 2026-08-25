# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Nested-CV training helpers must not redo feature selection per alpha."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from promptbudget.artifact import TierSettings


TOOLS = Path(__file__).parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("train_oof", TOOLS / "train_oof.py")
train_oof = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_oof)


class TrainOofV2Test(unittest.TestCase):
    def test_ridge_alpha_batch_matches_individual_ridge_predictions(self) -> None:
        matrix = np.asarray(((0.0, 1.0), (1.0, 0.0), (2.0, 1.0), (3.0, 0.0)), dtype=np.float64)
        targets = np.asarray(((0.0,), (1.0,), (2.0,), (3.0,)), dtype=np.float64)
        for alpha, (intercept, coefficients) in train_oof.fit_ridge_alphas(matrix, targets, (1.0, 10.0)).items():
            expected_intercept, expected_coefficients = train_oof._fit_ridge(matrix, targets, alpha)
            self.assertTrue(np.allclose(intercept, expected_intercept))
            self.assertTrue(np.allclose(coefficients, expected_coefficients))

    def test_batch_fit_predicts_each_alpha_from_one_feature_preparation(self) -> None:
        dense = np.zeros((5, 2), dtype=np.float64)
        sparse = ({0: 1.0}, {1: 1.0}, {0: 1.0}, {1: 1.0}, {0: 1.0})
        targets = np.ones((5, 7), dtype=np.float64)
        results = train_oof.fit_predict_specs(
            dense, sparse, targets, (0, 1, 2, 3), (4,), ((1, 1.0), (1, 10.0), (2, 1.0))
        )
        self.assertEqual({(1, 1.0), (1, 10.0), (2, 1.0)}, set(results))
        self.assertEqual((1, 7), results[(1, 1.0)][0].shape)

    def test_v21_training_rejects_artifact_outputs_outside_its_build_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            allowed = root / "build" / "promptbudget-v2.1"
            self.assertEqual(
                (allowed / "artifact.json", allowed / "manifest.json", allowed / "report.json"),
                train_oof.v21_output_paths(allowed / "artifact.json", allowed / "manifest.json", allowed / "report.json"),
            )
            with self.assertRaises(ValueError):
                train_oof.v21_output_paths(root / "artifact.json", allowed / "manifest.json", allowed / "report.json")

    def test_tier_policy_selection_requires_every_fold_gate_and_prefers_conservative_one_se_candidate(self) -> None:
        safe = train_oof.TierPolicyCandidate(
            TierSettings(1.0, 0.10, 0.10, 1.0, 1.0),
            (10.0, 10.0, 10.0, 10.0), (1.10, 1.10, 1.10, 1.10),
            (0.10, 0.10, 0.10, 0.10), 1,
        )
        aggressive = train_oof.TierPolicyCandidate(
            TierSettings(0.5, 0.05, 0.05, 1.0, 1.25),
            (10.1, 10.5, 9.7, 10.1), (1.10, 1.14, 1.10, 1.10),
            (0.90, 0.90, 0.90, 0.90), 0,
        )
        rejected = train_oof.TierPolicyCandidate(
            TierSettings(0.5, 0.05, 0.05, 1.0, 1.25),
            (10.1, 10.1, 10.1, 10.1), (4.0, 4.0, 4.0, 4.0),
            (0.90, 0.90, 0.90, 0.90), 2,
        )
        self.assertEqual(safe, train_oof.choose_tier_policy("fast", (safe, aggressive)))
        self.assertIsNone(train_oof.choose_tier_policy("premium", (rejected,)))

    def test_all_light_predicted_route_has_one_relative_cost_even_when_calibrated(self) -> None:
        costs = np.asarray(((9.0, 18.0, 27.0), (12.0, 24.0, 36.0)), dtype=np.float64)
        self.assertEqual(1.0, train_oof._relative_predicted_cost_ratio(costs, (0, 0)))
        self.assertEqual(2.0, train_oof._relative_predicted_cost_ratio(costs, (1, 1)))

    def test_prepares_selected_features_once_for_each_feature_count(self) -> None:
        dense = np.zeros((4, 2), dtype=np.float64)
        sparse = ({0: 1.0}, {1: 1.0}, {0: 1.0}, {1: 1.0})
        targets = np.ones((4, 7), dtype=np.float64)
        prepared = train_oof._prepare_fold_matrices(
            dense, sparse, targets, (0, 1, 2), (3,), (1, 2)
        )
        self.assertEqual({1, 2}, set(prepared))
        self.assertEqual((3, 3), prepared[1][0].shape)
        self.assertEqual((1, 4), prepared[2][1].shape)


if __name__ == "__main__":
    unittest.main()
