# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Tests for real sparse linear PromptBudget inference."""

from __future__ import annotations

import unittest
from fractions import Fraction

from promptbudget.schema import PromptBudgetError
from promptbudget.text_features import FeatureVector, extract_features
from promptbudget.linear import LinearHead, predict_head


class LinearTest(unittest.TestCase):
    def test_predicts_real_dense_and_sparse_dot_and_rejects_dimension_mismatch(self) -> None:
        vector = extract_features("Please choose A) or B?", 2**16)
        sparse_index = next(iter(vector.sparse))
        head = LinearHead(
            intercept=1.25,
            dense_coefficients=(2.0,) + (0.0,) * (len(vector.dense) - 1),
            sparse_coefficients={sparse_index: -0.5},
        )
        expected = 1.25 + 2.0 * vector.dense[0] - 0.5 * vector.sparse[sparse_index]
        self.assertEqual(expected, predict_head(head, vector))
        with self.assertRaises(PromptBudgetError):
            predict_head(
                LinearHead(1.0, (1.0,), {}),
                vector,
            )
        malformed = FeatureVector(vector.text, vector.dense, {-1: 1.0})
        with self.assertRaises(PromptBudgetError):
            predict_head(head, malformed)
        with self.assertRaises(PromptBudgetError):
            LinearHead(Fraction(10**10000, 1), head.dense_coefficients, {})


if __name__ == "__main__":
    unittest.main()
