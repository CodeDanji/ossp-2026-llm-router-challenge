# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Dev calibration must not weaken PromptBudget v2 Fast admission."""

from __future__ import annotations

from decimal import Decimal
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "calibrate_policy", Path(__file__).parents[2] / "tools" / "calibrate_policy.py"
)
calibrate_policy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(calibrate_policy)


class CalibratePolicyV2Test(unittest.TestCase):
    def test_fast_rejects_a_0_095792_margin_but_other_tiers_remain_strict(self) -> None:
        cap = Decimal("1.000000")
        self.assertFalse(calibrate_policy._is_tier_admissible("fast", Decimal("0.904208"), cap))
        self.assertTrue(calibrate_policy._is_tier_admissible("fast", Decimal("0.900000"), cap))
        self.assertTrue(calibrate_policy._is_tier_admissible("balanced", Decimal("0.999999"), cap))
        self.assertFalse(calibrate_policy._is_tier_admissible("balanced", cap, cap))

    def test_missing_candidate_uses_the_explicit_all_light_v1_fallback(self) -> None:
        fallback = calibrate_policy._v1_all_light_fallback()
        self.assertEqual(100.0, fallback.lambda_cost)
        self.assertEqual(1.0, fallback.min_gain_ax31)
        self.assertEqual(1.0, fallback.min_gain_think)
        self.assertEqual(1.0, fallback.max_relative_cost)


if __name__ == "__main__":
    unittest.main()
