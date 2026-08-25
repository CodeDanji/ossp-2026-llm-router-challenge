# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Safety contracts for PromptBudget v2 training and calibration."""

from __future__ import annotations

from decimal import Decimal
import unittest

from promptbudget.safety import (
    CandidateResult,
    bucketed_multipliers,
    choose_one_standard_error,
    grouped_folds,
    is_fast_admissible,
    monetary_cost_multipliers,
)


class SafetyTest(unittest.TestCase):
    def test_monetary_cost_calibration_uses_fixed_boundaries_and_global_fallback(self) -> None:
        predicted = (1.0,) * 103
        actual = (1.2,) * 100 + (2.0, 2.0, 3.0)
        counts = (512,) * 100 + (513, 2048, 2049)
        multipliers, fallback = monetary_cost_multipliers(
            predicted=predicted,
            actual=actual,
            character_counts=counts,
            minimum_samples=100,
            quantile=0.99,
        )
        self.assertEqual(1.2, multipliers["short"])
        self.assertEqual(multipliers["global"], multipliers["medium"])
        self.assertEqual(multipliers["global"], multipliers["long"])
        self.assertEqual(("long", "medium"), fallback)
        with self.assertRaises(ValueError):
            monetary_cost_multipliers(
                predicted=(0.0,), actual=(1.0,), character_counts=(1,),
                minimum_samples=100, quantile=0.99,
            )

    def test_grouped_folds_never_split_a_content_group(self) -> None:
        groups = ("a", "b", "a", "c", "d", "e", "f", "g")
        for fold in grouped_folds(groups, folds=4, seed=17):
            train_groups = {groups[index] for index in fold.train_indices}
            validation_groups = {groups[index] for index in fold.validation_indices}
            self.assertFalse(train_groups & validation_groups)

    def test_fast_requires_full_point_ten_margin_on_every_fold(self) -> None:
        cap = Decimal("1.000000")
        self.assertFalse(
            is_fast_admissible(
                (Decimal("0.900000"), Decimal("0.904208")), cap
            )
        )
        self.assertTrue(
            is_fast_admissible(
                (Decimal("0.900000"), Decimal("0.899999")), cap
            )
        )

    def test_one_standard_error_prefers_simpler_conservative_candidate(self) -> None:
        chosen = choose_one_standard_error(
            (
                CandidateResult("best", (0.10, 0.12, 0.11), 100, 1, 1.0, 9),
                CandidateResult("safe", (0.105, 0.115, 0.11), 10, 0, 1.25, 1),
            )
        )
        self.assertEqual("safe", chosen.name)

    def test_small_buckets_use_the_global_multiplier_deterministically(self) -> None:
        multipliers, fallback = bucketed_multipliers(
            predicted=(10.0, 10.0, 10.0, 10.0),
            actual=(10.0, 20.0, 30.0, 40.0),
            buckets=("korean", "code", "long", "global"),
            minimum_samples=2,
            quantile=0.75,
        )
        self.assertEqual(multipliers["korean"], multipliers["global"])
        self.assertEqual(multipliers["code"], multipliers["global"])
        self.assertEqual(multipliers["long"], multipliers["global"])
        self.assertEqual(("code", "korean", "long"), fallback)


if __name__ == "__main__":
    unittest.main()
