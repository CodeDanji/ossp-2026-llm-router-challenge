# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Focused mathematical contracts for the hash-regex quality evaluator."""

from __future__ import annotations

import unittest

import numpy as np

import hash_regex
import hash_regex_quality_nested as nested
from ossp_router.protocol import MODEL_IDS


class QualityFitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        self.scores = np.asarray(
            [[0.2, 0.3, 0.5], [0.4, 0.6, 0.7], [0.1, 0.2, 0.1], [0.9, 0.8, 0.9]]
        )

    def test_direct_uplift_is_raw_prediction_invariant(self) -> None:
        raw = nested.fit_quality_heads(self.matrix, self.scores, kind="raw")
        direct = nested.fit_quality_heads(self.matrix, self.scores, kind="direct-uplift")
        np.testing.assert_allclose(
            nested.predict_heads(raw, self.matrix),
            nested.predict_heads(direct, self.matrix),
            atol=1e-10,
        )

    def test_uplift_weights_are_positive_and_mean_one(self) -> None:
        weights = nested.positive_uplift_weights(np.asarray([-0.5, 0.0, 0.5, 1.0]), gamma=4.0)
        self.assertTrue(np.all(weights > 0.0))
        self.assertAlmostEqual(1.0, float(weights.mean()))

    def test_zero_regret_strength_is_raw_invariant(self) -> None:
        raw = nested.fit_quality_heads(self.matrix, self.scores, kind="raw")
        weighted = nested.fit_quality_heads(
            self.matrix,
            self.scores,
            kind="regret-weighted-raw",
            strength=0.0,
            regret=np.zeros((4, 3)),
        )
        np.testing.assert_allclose(
            nested.predict_heads(raw, self.matrix),
            nested.predict_heads(weighted, self.matrix),
            atol=1e-10,
        )

    def test_weighted_fit_rejects_invalid_strength_or_regret_shape(self) -> None:
        with self.assertRaises(ValueError):
            nested.fit_quality_heads(self.matrix, self.scores, kind="weighted-uplift", strength=-1.0)
        with self.assertRaises(ValueError):
            nested.fit_quality_heads(
                self.matrix,
                self.scores,
                kind="regret-weighted-raw",
                strength=1.0,
                regret=np.zeros((4, 2)),
            )

    def test_group_disjointness_rejects_shared_content(self) -> None:
        with self.assertRaises(ValueError):
            nested.require_group_disjoint((0, 1), (2,), ("a", "b", "a"))

    def test_lagrange_matches_runtime_batch_light_unit(self) -> None:
        scores = np.asarray([[0.1, 0.8, 0.9], [0.3, 0.4, 0.8]])
        costs = np.asarray([[1.0, 2.0, 4.0], [2.0, 3.0, 8.0]])
        selected, penalty = nested.lagrange_selection_and_penalty(scores, costs, multiplier=1.25)
        score_rows = [dict(zip(MODEL_IDS, row)) for row in scores]
        cost_rows = [dict(zip(MODEL_IDS, row)) for row in costs]
        expected, _ratio = hash_regex.select_models(
            score_rows, cost_rows, budget_multiplier=1.25, safety_ratio=1.0
        )
        self.assertEqual(tuple(MODEL_IDS[index] for index in selected), expected)
        self.assertGreaterEqual(penalty, 0.0)

    def test_batched_safety_selection_matches_runtime_allocator(self) -> None:
        scores = np.asarray([[0.1, 0.8, 0.9], [0.3, 0.4, 0.8]])
        costs = np.asarray([[1.0, 2.0, 4.0], [2.0, 3.0, 8.0]])
        selections = nested.lagrange_selections_for_safety_grid(scores, costs, 1.25, (0.8, 1.0))
        score_rows = [dict(zip(MODEL_IDS, row)) for row in scores]
        cost_rows = [dict(zip(MODEL_IDS, row)) for row in costs]
        for safety, selected in zip((0.8, 1.0), selections):
            expected, _ratio = hash_regex.select_models(
                score_rows, cost_rows, budget_multiplier=1.25, safety_ratio=safety
            )
            self.assertEqual(tuple(MODEL_IDS[index] for index in selected), expected)
