# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Focused mathematical contracts for the hash-regex quality evaluator."""

from __future__ import annotations

import unittest

import numpy as np

import hash_regex_quality_nested as nested


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

