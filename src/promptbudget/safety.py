# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, leakage-safe selection helpers used only by training tools."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import math
from statistics import mean, stdev
from typing import Iterable, Mapping, Sequence, Tuple
import unicodedata


FAST_MARGIN = Decimal("0.10")
OUTER_SEEDS = (137, 271, 811)


@dataclass(frozen=True)
class GroupedFold:
    """An index split whose train and validation groups are disjoint."""

    train_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]


@dataclass(frozen=True)
class CandidateResult:
    """Fold losses and the pre-registered complexity ordering for one candidate."""

    name: str
    fold_losses: Tuple[float, ...]
    active_sparse_features: int
    family_rank: int
    residual_multiplier: float
    grid_index: int

    def __post_init__(self) -> None:
        if not self.fold_losses:
            raise ValueError("candidate requires at least one fold loss")
        if not all(math.isfinite(value) for value in self.fold_losses):
            raise ValueError("candidate fold losses must be finite")

    @property
    def average_loss(self) -> float:
        return float(mean(self.fold_losses))

    @property
    def complexity_key(self) -> Tuple[int, int, float, int]:
        return (
            self.active_sparse_features,
            self.family_rank,
            self.residual_multiplier,
            self.grid_index,
        )


def canonical_content_group(text: str) -> str:
    """Return a stable content-only group key for an episode's prompt text."""

    if not isinstance(text, str):
        raise TypeError("content text must be a string")
    canonical = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _group_order(groups: Iterable[str], seed: int) -> Tuple[str, ...]:
    unique = set(groups)
    if not unique:
        raise ValueError("at least one content group is required")
    return tuple(
        sorted(
            unique,
            key=lambda group: hashlib.sha256(
                str(seed).encode("ascii") + b"\0" + group.encode("utf-8")
            ).digest(),
        )
    )


def grouped_folds(groups: Sequence[str], *, folds: int, seed: int) -> Tuple[GroupedFold, ...]:
    """Create deterministic folds without allowing a content group to cross a boundary."""

    if folds < 2:
        raise ValueError("folds must be at least two")
    ordered_groups = _group_order(groups, seed)
    if len(ordered_groups) < folds:
        raise ValueError("folds must not exceed the number of content groups")
    assignments = {group: index % folds for index, group in enumerate(ordered_groups)}
    result = []
    for fold in range(folds):
        validation = tuple(index for index, group in enumerate(groups) if assignments[group] == fold)
        training = tuple(index for index, group in enumerate(groups) if assignments[group] != fold)
        if not validation or not training:
            raise ValueError("each grouped fold must contain train and validation rows")
        result.append(GroupedFold(training, validation))
    return tuple(result)


def repeated_outer_folds(groups: Sequence[str]) -> Tuple[Tuple[int, GroupedFold], ...]:
    """Return the fixed three-seed, five-fold outer CV schedule."""

    return tuple(
        (seed, fold)
        for seed in OUTER_SEEDS
        for fold in grouped_folds(groups, folds=5, seed=seed)
    )


def is_fast_admissible(upper_cost_ratios: Sequence[Decimal], tier_cap: Decimal) -> bool:
    """Require every validation fold to retain the Fast tier's 0.10 budget room."""

    if not upper_cost_ratios:
        return False
    threshold = Decimal(tier_cap) - FAST_MARGIN
    return all(Decimal(value) <= threshold for value in upper_cost_ratios)


def aggregate_upper_cost_ratio(
    request_upper_costs: Sequence[Decimal],
    baseline_costs: Sequence[Decimal],
    groups: Sequence[str],
    *,
    z_score: Decimal = Decimal("2.326348"),
) -> Tuple[Decimal, Decimal, Decimal]:
    """Return request, aggregate and controlling one-sided cost-ratio upper bounds.

    The first bound comes from conformal per-request cost uppers.  The second is
    a group-clustered one-sided normal upper bound over aggregate group ratios.
    The caller must use the returned maximum as its admission predicate.
    """

    if not (len(request_upper_costs) == len(baseline_costs) == len(groups)) or not groups:
        raise ValueError("costs and groups must be non-empty and aligned")
    totals = {}
    for upper, baseline, group in zip(request_upper_costs, baseline_costs, groups):
        upper_value, baseline_value = Decimal(upper), Decimal(baseline)
        if baseline_value <= 0:
            raise ValueError("baseline costs must be positive")
        item = totals.setdefault(group, [Decimal("0"), Decimal("0")])
        item[0] += upper_value
        item[1] += baseline_value
    request_ratio = sum(Decimal(value) for value in request_upper_costs) / sum(Decimal(value) for value in baseline_costs)
    group_ratios = [upper / baseline for upper, baseline in totals.values()]
    group_mean = sum(group_ratios) / Decimal(len(group_ratios))
    if len(group_ratios) == 1:
        aggregate_ratio = group_mean
    else:
        variance = sum((ratio - group_mean) ** 2 for ratio in group_ratios) / Decimal(len(group_ratios) - 1)
        aggregate_ratio = group_mean + Decimal(z_score) * variance.sqrt() / Decimal(len(group_ratios)).sqrt()
    return request_ratio, aggregate_ratio, max(request_ratio, aggregate_ratio)


def choose_one_standard_error(candidates: Sequence[CandidateResult]) -> CandidateResult:
    """Choose the simplest conservative candidate within the best candidate's one-SE band."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    best = min(candidates, key=lambda candidate: (candidate.average_loss, candidate.grid_index))
    if len(best.fold_losses) == 1:
        standard_error = 0.0
    else:
        standard_error = stdev(best.fold_losses) / math.sqrt(len(best.fold_losses))
    eligible = [candidate for candidate in candidates if candidate.average_loss <= best.average_loss + standard_error]
    return min(eligible, key=lambda candidate: (candidate.complexity_key, candidate.average_loss, candidate.name))


def _quantile_higher(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[position]


def prompt_length_bucket(character_count: int) -> str:
    if isinstance(character_count, bool) or not isinstance(character_count, int) or character_count < 0:
        raise ValueError("character count must be a non-negative integer")
    return "short" if character_count <= 512 else "medium" if character_count <= 2048 else "long"


def monetary_cost_multipliers(
    *,
    predicted: Sequence[float],
    actual: Sequence[float],
    character_counts: Sequence[int],
    minimum_samples: int,
    quantile: float,
) -> Tuple[Mapping[str, float], Tuple[str, ...]]:
    """Return fixed length-bucket monetary-cost multipliers and fallbacks."""

    if not (len(predicted) == len(actual) == len(character_counts)) or not predicted:
        raise ValueError("calibration rows must be non-empty and aligned")
    if minimum_samples < 1 or not 0.0 < quantile <= 1.0:
        raise ValueError("invalid calibration parameters")
    buckets = {"short": [], "medium": [], "long": []}
    ratios = []
    for estimate, observed, character_count in zip(predicted, actual, character_counts):
        estimate, observed = float(estimate), float(observed)
        if not math.isfinite(estimate) or not math.isfinite(observed) or estimate <= 0.0 or observed <= 0.0:
            raise ValueError("monetary costs must be finite and positive")
        ratio = observed / estimate
        ratios.append(ratio)
        buckets[prompt_length_bucket(character_count)].append(ratio)
    global_multiplier = _quantile_higher(ratios, quantile)
    fallback = tuple(bucket for bucket in ("long", "medium", "short") if len(buckets[bucket]) < minimum_samples)
    multipliers = {"global": global_multiplier}
    multipliers.update({
        bucket: global_multiplier if bucket in fallback else _quantile_higher(values, quantile)
        for bucket, values in buckets.items()
    })
    return multipliers, fallback


def bucketed_multipliers(
    *,
    predicted: Sequence[float],
    actual: Sequence[float],
    buckets: Sequence[str],
    minimum_samples: int,
    quantile: float,
) -> Tuple[Mapping[str, float], Tuple[str, ...]]:
    """Calculate one-sided residual multipliers with an explicit global fallback."""

    if not (len(predicted) == len(actual) == len(buckets)) or not predicted:
        raise ValueError("calibration rows must be non-empty and aligned")
    if minimum_samples < 1 or not 0.0 < quantile <= 1.0:
        raise ValueError("invalid calibration parameters")
    ratios = [max(1.0, float(observed) / max(1.0, float(estimate))) for estimate, observed in zip(predicted, actual)]
    global_multiplier = _quantile_higher(ratios, quantile)
    per_bucket = {}
    for bucket, ratio in zip(buckets, ratios):
        per_bucket.setdefault(bucket, []).append(ratio)
    fallback = tuple(
        sorted(
            bucket
            for bucket, values in per_bucket.items()
            if bucket != "global" and len(values) < minimum_samples
        )
    )
    multipliers = {"global": global_multiplier}
    for bucket, values in per_bucket.items():
        if bucket == "global":
            continue
        multipliers[bucket] = global_multiplier if bucket in fallback else _quantile_higher(values, quantile)
    return multipliers, fallback
