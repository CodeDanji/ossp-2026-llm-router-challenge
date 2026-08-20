# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, content-only text features for PromptBudget models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re
from types import MappingProxyType
from typing import Final, Mapping
import unicodedata

from .schema import PromptBudgetError


_ALLOWED_HASH_DIMENSIONS: Final = (2**16, 2**18, 2**20)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_ROLE = re.compile(r"<role>", re.IGNORECASE)
_INLINE_CODE = re.compile(r"(?<!`)`[^`\n]+`(?!`)")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
_DATE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
)
_CURRENCY = re.compile(
    r"(?:[$€£₩¥]\s*\d+(?:[.,]\d+)*|\b(?:USD|KRW|EUR|JPY)\s*\d+(?:[.,]\d+)*)",
    re.IGNORECASE,
)
_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|ms|s|min|h|kg|g|km|m|cm|mm|MB|GB|TB|tokens?|words?)(?!\w)",
    re.IGNORECASE,
)
_LATEX = re.compile(r"\\(?:[A-Za-z]+|[()[\]{}])|\${1,2}[^$\n]+\${1,2}")
_ERROR = re.compile(
    r"\b(?:error|exception|traceback|failed|failure|stack\s+trace|오류|예외|실패)\b",
    re.IGNORECASE,
)
_CHOICE = re.compile(
    r"\b(?:choose|choice|select|option|either|alternatives?|or|선택|옵션)\b|\b[A-D]\)",
    re.IGNORECASE,
)
_IMPERATIVE = re.compile(
    r"\b(?:please|must|should|choose|select|write|explain|return|provide|list|create|calculate|answer)\b|(?:해주세요|하라|해라|하십시오|하세요)",
    re.IGNORECASE,
)

DENSE_FEATURE_NAMES: Final = (
    "character_count",
    "word_count",
    "line_count",
    "paragraph_count",
    "role_turn_count",
    "code_fence_count",
    "inline_code_count",
    "quote_line_count",
    "max_bracket_depth",
    "choice_signal_ratio",
    "question_signal_ratio",
    "imperative_signal_ratio",
    "hangul_ratio",
    "latin_ratio",
    "digit_ratio",
    "symbol_ratio",
    "whitespace_ratio",
    "url_count",
    "date_count",
    "currency_count",
    "unit_count",
    "latex_count",
    "error_marker_count",
)
"""Names for the dense feature values, in exactly their vector order."""


@dataclass(frozen=True)
class FeatureVector:
    """Normalized content and fixed dense/sparse features for one prompt."""

    text: str
    dense: tuple[float, ...]
    sparse: Mapping[int, float]


def normalize_text(text: str) -> str:
    """Return canonical prompt text, rejecting missing or blank content."""

    if not isinstance(text, str):
        raise PromptBudgetError("text must be a string.")
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise PromptBudgetError("text must not be blank.")
    return normalized


def _source_text(text: str) -> str:
    """Canonicalize text while retaining line boundaries for structure features."""

    if not isinstance(text, str):
        raise PromptBudgetError("text must be a string.")
    source = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    if not source.strip():
        raise PromptBudgetError("text must not be blank.")
    return source


def _paragraph_count(source: str) -> int:
    return sum(
        bool(paragraph.strip())
        for paragraph in re.split(r"\n[^\S\r\n]*\n+", source)
    )


def _max_bracket_depth(text: str) -> int:
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closing = set(pairs.values())
    depth = maximum = 0
    for character in text:
        if character in pairs:
            depth += 1
            maximum = max(maximum, depth)
        elif character in closing:
            depth = max(0, depth - 1)
    return maximum


def _latin(character: str) -> bool:
    return unicodedata.name(character, "").startswith("LATIN ")


def _ratio(count: int, denominator: int) -> float:
    return float(min(count, denominator)) / max(1, denominator)


def _hash_index_and_sign(value: str, hash_dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, person=b"PBRouter"
    ).digest()
    number = int.from_bytes(digest, "big")
    return number & (hash_dimension - 1), -1.0 if number & (1 << 63) else 1.0


def _sparse_features(text: str, hash_dimension: int) -> Mapping[int, float]:
    values: defaultdict[int, float] = defaultdict(float)

    bounded = f" {text.casefold()} "
    for size in range(3, 6):
        for start in range(len(bounded) - size + 1):
            index, sign = _hash_index_and_sign(
                f"c{size}:{bounded[start : start + size]}", hash_dimension
            )
            values[index] += sign

    tokens = tuple(token.casefold() for token in _WORD.findall(text))
    for token in tokens:
        index, sign = _hash_index_and_sign(f"w1:{token}", hash_dimension)
        values[index] += sign
    for left, right in zip(tokens, tokens[1:]):
        index, sign = _hash_index_and_sign(f"w2:{left}\x1f{right}", hash_dimension)
        values[index] += sign

    return MappingProxyType({index: value for index, value in values.items() if value})


def extract_features(text: str, hash_dimension: int) -> FeatureVector:
    """Extract deterministic text features without IDs, order, or process state."""

    if (
        isinstance(hash_dimension, bool)
        or not isinstance(hash_dimension, int)
        or hash_dimension not in _ALLOWED_HASH_DIMENSIONS
    ):
        raise PromptBudgetError("hash_dimension must be an allowed power of two.")

    source = _source_text(text)
    normalized = normalize_text(text)
    words = _WORD.findall(normalized)
    character_count = len(normalized)
    lines = source.split("\n")
    denominator = max(1, character_count)
    hangul = sum("\uac00" <= character <= "\ud7a3" for character in normalized)
    latin = sum(_latin(character) for character in normalized)
    digits = sum(character.isdigit() for character in normalized)
    whitespace = sum(character.isspace() for character in normalized)
    symbols = sum(
        not character.isspace()
        and not character.isalnum()
        and not ("\uac00" <= character <= "\ud7a3")
        for character in normalized
    )
    word_count = len(words)

    dense = (
        float(character_count),
        float(word_count),
        float(len(lines)),
        float(_paragraph_count(source)),
        float(len(_ROLE.findall(normalized))),
        float(source.count("```")),
        float(len(_INLINE_CODE.findall(source))),
        float(sum(line.lstrip().startswith(">") for line in lines)),
        float(_max_bracket_depth(source)),
        _ratio(len(_CHOICE.findall(normalized)), word_count),
        _ratio(len(re.findall(r"[?？]", source)), word_count),
        _ratio(len(_IMPERATIVE.findall(normalized)), word_count),
        float(hangul) / denominator,
        float(latin) / denominator,
        float(digits) / denominator,
        float(symbols) / denominator,
        float(whitespace) / denominator,
        float(len(_URL.findall(normalized))),
        float(len(_DATE.findall(normalized))),
        float(len(_CURRENCY.findall(normalized))),
        float(len(_UNIT.findall(normalized))),
        float(len(_LATEX.findall(normalized))),
        float(len(_ERROR.findall(normalized))),
    )
    return FeatureVector(normalized, dense, _sparse_features(normalized, hash_dimension))
