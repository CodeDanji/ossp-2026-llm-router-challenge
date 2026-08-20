# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic, content-only PromptBudget text features."""

from __future__ import annotations

from dataclasses import fields
import unittest

from promptbudget.schema import PromptBudgetError
from promptbudget.text_features import (
    DENSE_FEATURE_NAMES,
    FeatureVector,
    extract_features,
    normalize_text,
)


class TextFeaturesTest(unittest.TestCase):
    def test_normalize_text_unifies_equivalent_whitespace_and_rejects_blank(self) -> None:
        self.assertEqual("Café next", normalize_text("Cafe\u0301\r\n\u2003next"))
        self.assertEqual(
            extract_features("Cafe\u0301", 2**16),
            extract_features("Café", 2**16),
        )

        for value in ("", " \t\u2003\r\n", 3):
            with self.subTest(value=value):
                with self.assertRaises(PromptBudgetError):
                    normalize_text(value)  # type: ignore[arg-type]

    def test_sparse_hashing_uses_fixed_blake2b_bins_and_validates_dimension(self) -> None:
        vector = extract_features("x", 2**16)

        self.assertEqual({11116: -1.0, 7031: 1.0}, dict(vector.sparse))
        self.assertEqual(vector, extract_features("x", 2**16))
        with self.assertRaises(PromptBudgetError):
            extract_features("x", 2**17)

    def test_extract_features_detects_rich_structured_content(self) -> None:
        text = (
            "<role>system</role>\r\n"
            "> quoted line\r\n"
            "\r\n"
            "<role>user</role>\r\n"
            "Please choose A) or B)? Why?\r\n"
            "```python\r\n"
            "raise ValueError('error')\r\n"
            "```\r\n"
            "Use `inline` at https://example.test on 2026-08-20 for $12 and 5kg. "
            "\\frac{a}{b} failed. (nested [brackets]) 한글 123"
        )

        vector = extract_features(text, 2**16)
        values = dict(zip(DENSE_FEATURE_NAMES, vector.dense))

        self.assertEqual(9.0, values["line_count"])
        self.assertEqual(2.0, values["paragraph_count"])
        self.assertEqual(2.0, values["role_turn_count"])
        self.assertEqual(2.0, values["code_fence_count"])
        self.assertEqual(1.0, values["inline_code_count"])
        self.assertEqual(1.0, values["quote_line_count"])
        self.assertGreaterEqual(values["max_bracket_depth"], 2.0)
        for name in (
            "choice_signal_ratio",
            "question_signal_ratio",
            "imperative_signal_ratio",
            "hangul_ratio",
            "latin_ratio",
            "digit_ratio",
            "symbol_ratio",
            "whitespace_ratio",
        ):
            self.assertGreater(values[name], 0.0, name)
        for name in (
            "url_count",
            "date_count",
            "currency_count",
            "unit_count",
            "latex_count",
        ):
            self.assertEqual(1.0, values[name], name)
        self.assertEqual(2.0, values["error_marker_count"])

    def test_vector_is_immutable_content_only_and_repeatable(self) -> None:
        first = extract_features("<role>user</role> Same content", 2**16)
        second = extract_features("<role>user</role> Same content", 2**16)

        self.assertEqual(("text", "dense", "sparse"), tuple(field.name for field in fields(FeatureVector)))
        self.assertNotIn("output_key", FeatureVector.__dataclass_fields__)
        self.assertEqual(first, second)
        with self.assertRaises(TypeError):
            first.sparse[0] = 1.0  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
