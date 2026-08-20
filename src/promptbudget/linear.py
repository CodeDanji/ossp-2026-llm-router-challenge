# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only sparse linear inference for PromptBudget feature vectors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

from .schema import PromptBudgetError
from .text_features import FeatureVector


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PromptBudgetError(f"{name} must be a finite number.")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PromptBudgetError(f"{name} must be a finite number.") from error
    if not math.isfinite(converted):
        raise PromptBudgetError(f"{name} must be a finite number.")
    return converted


def validate_head(head: "LinearHead") -> None:
    """Validate values in a constructed linear head."""

    _finite_number(head.intercept, "intercept")
    for index, coefficient in enumerate(head.dense_coefficients):
        _finite_number(coefficient, f"dense_coefficients[{index}]")
    if not isinstance(head.sparse_coefficients, Mapping):
        raise PromptBudgetError("sparse_coefficients must be a mapping.")
    for index, coefficient in head.sparse_coefficients.items():
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise PromptBudgetError("sparse coefficient indices must be non-negative integers.")
        _finite_number(coefficient, f"sparse_coefficients[{index}]")


@dataclass(frozen=True)
class LinearHead:
    intercept: float
    dense_coefficients: tuple[float, ...]
    sparse_coefficients: Mapping[int, float]

    def __post_init__(self) -> None:
        validate_head(self)


def predict_head(head: LinearHead, vector: FeatureVector) -> float:
    """Return the linear prediction for one content-only feature vector."""

    if len(head.dense_coefficients) != len(vector.dense):
        raise PromptBudgetError("dense feature dimension does not match the linear head.")
    for index, value in enumerate(vector.dense):
        _finite_number(value, f"vector.dense[{index}]")
    for index, value in vector.sparse.items():
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise PromptBudgetError("vector sparse indices must be non-negative integers.")
        _finite_number(value, f"vector.sparse[{index}]")
    terms = [head.intercept]
    terms.extend(
        coefficient * value
        for coefficient, value in zip(head.dense_coefficients, vector.dense)
    )
    terms.extend(
        head.sparse_coefficients[index] * value
        for index, value in vector.sparse.items()
        if index in head.sparse_coefficients
    )
    output = math.fsum(terms)
    if not math.isfinite(output):
        raise PromptBudgetError("linear prediction must be finite.")
    return output
