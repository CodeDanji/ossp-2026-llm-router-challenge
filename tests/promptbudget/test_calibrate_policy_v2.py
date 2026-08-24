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


if __name__ == "__main__":
    unittest.main()
