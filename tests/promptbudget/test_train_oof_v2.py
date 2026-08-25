# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Nested-CV training helpers must not redo feature selection per alpha."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


TOOLS = Path(__file__).parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("train_oof", TOOLS / "train_oof.py")
train_oof = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_oof)


class TrainOofV2Test(unittest.TestCase):
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
