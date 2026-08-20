# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Small immutable values and frozen v1 constants for PromptBudget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ossp_router.protocol import MODEL_IDS, TIERS

MODEL_ORDER: Final = MODEL_IDS
"""Ascending PromptBudget model order used by downstream policies."""


class PromptBudgetError(ValueError):
    """Raised when a PromptBudget adapter input is invalid."""


@dataclass(frozen=True)
class PromptRecord:
    """Normalized policy text and an opaque key for output reassembly."""

    text: str
    output_key: str
