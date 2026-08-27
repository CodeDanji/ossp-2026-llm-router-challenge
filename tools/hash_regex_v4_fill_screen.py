"""Pure Train-only v4 AX31-fill screening primitives."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from statistics import median
from typing import Sequence

import numpy as np

import hash_regex
import hash_regex_cost_stabilization_nested as v32
import hash_regex_tail_guard_nested as nested
from ossp_router.protocol import MODEL_IDS, TIERS, Decision, Submission
from ossp_router.scoring import score_submissions


CANDIDATES = ("fast-ax31-fill", "balanced-ax31-fill", "fast-balanced-ax31-fill")
_CANDIDATE_TIERS = {
    "fast-ax31-fill": ("fast",),
    "balanced-ax31-fill": ("balanced",),
    "fast-balanced-ax31-fill": ("fast", "balanced"),
}


@dataclass(frozen=True)
class Route:
    choices: tuple[str, ...]
    predicted_ratio: float
    added_models: tuple[str, ...] = ()


def _require_candidate(candidate: str) -> tuple[str, ...]:
    try:
        return _CANDIDATE_TIERS[candidate]
    except KeyError as error:
        raise ValueError(f"unknown v4 candidate: {candidate}") from error


def _prediction_rows(
    scores: np.ndarray, log_costs: np.ndarray, guard: nested.TailGuard | None
) -> tuple[list[dict[str, float]], list[dict[str, float]], tuple[tuple[int, int], ...]]:
    score_array = np.asarray(scores, dtype=float)
    logs = np.asarray(log_costs, dtype=float)
    if score_array.ndim != 2 or score_array.shape[1] != len(MODEL_IDS) or score_array.shape != logs.shape:
        raise ValueError("score and log-cost predictions must be aligned (rows, 3) matrices")
    if not len(score_array) or not np.isfinite(score_array).all() or not np.isfinite(logs).all():
        raise ValueError("score and log-cost predictions must be non-empty and finite")
    with np.errstate(over="ignore", invalid="ignore"):
        costs = np.exp(logs)
    guarded = nested.apply_ordering_clamp(costs)
    if guard is not None:
        guarded = nested.apply_tail_guard(guarded, logs, guard)
    buckets = tuple(
        (
            nested.bucket_index(float(log_row[1]), guard.ax31_edges) if guard else -1,
            nested.bucket_index(float(log_row[2]), guard.think_edges) if guard else -1,
        )
        for log_row in logs
    )
    return (
        [
            {model_id: min(1.0, max(0.0, float(row[column]))) for column, model_id in enumerate(MODEL_IDS)}
            for row in score_array
        ],
        [
            {model_id: float(row[column]) for column, model_id in enumerate(MODEL_IDS)}
            for row in guarded
        ],
        buckets,
    )


def _baseline_routes(
    score_rows: Sequence[dict[str, float]], cost_rows: Sequence[dict[str, float]], data: v32.EvaluationData
) -> dict[str, Route]:
    routes = {}
    for tier in TIERS:
        choices, predicted_ratio = hash_regex.select_models(
            score_rows,
            cost_rows,
            budget_multiplier=float(data.policy.tiers[tier].budget_multiplier),
            safety_ratio=1.0,
        )
        if tier == "premium":
            choices, predicted_ratio = hash_regex.fill_ax31_upgrades(
                choices,
                score_rows,
                cost_rows,
                budget_multiplier=float(data.policy.tiers[tier].budget_multiplier),
                safety_ratio=hash_regex.PREMIUM_AX31_FILL_SAFETY_RATIO,
            )
        routes[tier] = Route(tuple(choices), float(predicted_ratio))
    return routes


def _score_routes(data: v32.EvaluationData, indices: tuple[int, ...], routes: dict[str, Route]) -> dict[str, object]:
    inputs, outcomes = v32._slice_batches(data, indices)
    report = score_submissions(
        inputs,
        outcomes,
        tuple(
            Submission(
                inputs.schema_version,
                inputs.challenge_id,
                data.policy.policy_id,
                inputs.split,
                tier,
                tuple(Decision(episode.episode_id, model_id) for episode, model_id in zip(inputs.episodes, routes[tier].choices)),
            )
            for tier in TIERS
        ),
        data.policy,
    )
    report["tiers"] = {
        tier: {
            **report["tiers"][tier],
            "actual_ratio": report["tiers"][tier]["budget_ratio"],
            "predicted_ratio": routes[tier].predicted_ratio,
            "includes_premium_fill": tier == "premium",
        }
        for tier in TIERS
    }
    return report


def route_guarded_candidate(
    data: v32.EvaluationData,
    indices: Sequence[int],
    score_prediction: np.ndarray,
    cost_prediction: np.ndarray,
    guard: nested.TailGuard,
    candidate: str,
) -> dict[str, object]:
    """Reconstruct the guarded v3.3 route and apply only the allowed AX31 fills."""

    candidate_tiers = _require_candidate(candidate)
    selected_indices = v32._validated_indices(data, tuple(indices))
    score_rows, guarded_cost_rows, guard_buckets = _prediction_rows(score_prediction, cost_prediction, guard)
    raw_score_rows, raw_cost_rows, _unused = _prediction_rows(score_prediction, cost_prediction, None)
    if len(selected_indices) != len(score_rows):
        raise ValueError("prediction rows and indices must align")
    baseline = _baseline_routes(score_rows, guarded_cost_rows, data)
    raw_baseline = _baseline_routes(raw_score_rows, raw_cost_rows, data)
    routes = dict(baseline)
    for tier in candidate_tiers:
        before = baseline[tier]
        choices, predicted_ratio = hash_regex.fill_ax31_upgrades(
            before.choices,
            score_rows,
            guarded_cost_rows,
            budget_multiplier=float(data.policy.tiers[tier].budget_multiplier),
            safety_ratio=1.0,
        )
        added = tuple(after for before_model, after in zip(before.choices, choices) if before_model == MODEL_IDS[0] and after != before_model)
        if any(model != MODEL_IDS[1] for model in added) or any(
            before_model != after and not (before_model == MODEL_IDS[0] and after == MODEL_IDS[1])
            for before_model, after in zip(before.choices, choices)
        ):
            raise AssertionError("candidate fill may only promote Light to AX31")
        routes[tier] = Route(tuple(choices), float(predicted_ratio), added)
    report = _score_routes(data, selected_indices, routes)
    baseline_report = _score_routes(data, selected_indices, baseline)
    return {
        "candidate": candidate,
        "indices": selected_indices,
        "baseline": baseline,
        "raw_baseline": raw_baseline,
        "fast": routes["fast"],
        "balanced": routes["balanced"],
        "premium": routes["premium"],
        "guard_buckets": guard_buckets,
        "report": report,
        "baseline_report": baseline_report,
    }


def _choices(routes: object, tier: str) -> tuple[str, ...]:
    route = routes[tier]  # type: ignore[index]
    return tuple(route.choices if isinstance(route, Route) else route)


def blocked_upgrade_diagnostic(
    data: v32.EvaluationData,
    indices: Sequence[int],
    raw_routes: dict[str, object],
    guarded_routes: dict[str, object],
) -> dict[str, object]:
    """Summarize held-out Raw non-Light -> guarded Light AX31 recovery opportunities."""

    selected = v32._validated_indices(data, tuple(indices))
    buckets = tuple(guarded_routes.get("guard_buckets", ()))
    if buckets and len(buckets) != len(selected):
        raise ValueError("guard buckets and selected indices must align")
    result = {}
    for tier in TIERS:
        raw = _choices(raw_routes, tier)
        guarded = _choices(guarded_routes, tier)
        rows = []
        current_cost = sum(float(data.costs[index, MODEL_IDS.index(model)]) for index, model in zip(selected, guarded))
        light_total = sum(float(data.costs[index, 0]) for index in selected)
        cap = light_total * float(data.policy.tiers[tier].budget_multiplier)
        for local, (index, raw_model, guarded_model) in enumerate(zip(selected, raw, guarded)):
            if raw_model == MODEL_IDS[0] or guarded_model != MODEL_IDS[0]:
                continue
            gain = float(data.scores[index, 1] - data.scores[index, 0])
            incremental = float(data.costs[index, 1] - data.costs[index, 0])
            rows.append({
                "index": index,
                "observed_score_gain_vs_light": gain,
                "actual_incremental_cost": incremental,
                "guard_bucket": None if not buckets else buckets[local][0],
            })
        slack = max(0.0, cap - current_cost)
        remaining = slack
        recovered = 0.0
        for row in sorted(rows, key=lambda value: (value["observed_score_gain_vs_light"] / value["actual_incremental_cost"] if value["actual_incremental_cost"] > 0 else float("inf")), reverse=True):
            if row["observed_score_gain_vs_light"] > 0 and 0 < row["actual_incremental_cost"] <= remaining:
                remaining -= row["actual_incremental_cost"]
                recovered += row["observed_score_gain_vs_light"]
        result[tier] = {
            "blocked_row_count": len(rows),
            "rows": tuple(rows),
            "actual_slack": slack,
            "oracle_greedy_actual_slack_recovery_upper_bound": recovered,
        }
    return result


def _tier_quality(report: dict[str, object], tier: str) -> Decimal:
    metrics = report["tiers"][tier]
    rows = int(metrics["num_episodes"])
    if rows <= 0:
        raise ValueError("tier quality requires positive row count")
    return Decimal(str(metrics["quality_points_total"])) / Decimal(rows)


def screen_candidate(data: v32.EvaluationData, folds: Sequence[object], candidate: str) -> dict[str, object]:
    """Cross-fit a fixed four-fold Train screen without reading held-out labels while fitting."""

    changed_tiers = _require_candidate(candidate)
    if len(folds) != 4:
        raise ValueError("v4 screen requires exactly four grouped folds")
    records = []
    for number, fold in enumerate(folds):
        train_indices = tuple(fold.train_indices)
        validation_indices = tuple(fold.validation_indices)
        if set(train_indices) & set(validation_indices) or set(data.groups[index] for index in train_indices) & set(data.groups[index] for index in validation_indices):
            raise ValueError("screen folds must be group-disjoint")
        guard = nested.fit_tail_guard(data, train_indices, seed=137)
        quality = v32.fit_raw_quality_heads(data.matrix[list(train_indices)], data.scores[list(train_indices)])
        costs = v32.fit_log_cost_heads(data.matrix[list(train_indices)], data.log_costs[list(train_indices)])
        routes = route_guarded_candidate(
            data,
            validation_indices,
            v32.predict_heads(quality, data.matrix[list(validation_indices)]),
            v32.predict_heads(costs, data.matrix[list(validation_indices)]),
            guard,
            candidate,
        )
        records.append({"fold": number, "routes": routes, "diagnostic": blocked_upgrade_diagnostic(data, validation_indices, routes["raw_baseline"], routes)})
    checks = tuple(
        {
            "fold": record["fold"],
            "tier": tier,
            "budget_passed": bool(record["routes"]["report"]["tiers"][tier]["budget_passed"]),
            "actual_ratio": Decimal(str(record["routes"]["report"]["tiers"][tier]["actual_ratio"])),
        }
        for record in records
        for tier in TIERS
    )
    fold_deltas = {
        tier: tuple(_tier_quality(record["routes"]["report"], tier) - _tier_quality(record["routes"]["baseline_report"], tier) for record in records)
        for tier in changed_tiers
    }
    pooled = {
        tier: (
            sum(
                (
                    Decimal(str(record["routes"]["report"]["tiers"][tier]["quality_points_total"]))
                    - Decimal(str(record["routes"]["baseline_report"]["tiers"][tier]["quality_points_total"]))
                    for record in records
                ),
                Decimal("0"),
            )
            / Decimal(sum(int(record["routes"]["report"]["tiers"][tier]["num_episodes"]) for record in records))
        )
        for tier in changed_tiers
    }
    new_think = sum(
        before != MODEL_IDS[2] and after == MODEL_IDS[2]
        for record in records
        for tier in changed_tiers
        for before, after in zip(record["routes"]["baseline"][tier].choices, record["routes"][tier].choices)
    )
    fallback_count = sum(
        all(model == MODEL_IDS[0] for tier in TIERS for model in record["routes"][tier].choices)
        for record in records
    )
    passed = sum(check["budget_passed"] for check in checks)
    return {
        "candidate": candidate,
        "folds": tuple(records),
        "actual_checks": checks,
        "actual_checks_required": 12,
        "actual_checks_passed": passed,
        "fallback_count": fallback_count,
        "changed_tier_pooled_cap_neutral_quality_deltas": pooled,
        "fold_deltas": fold_deltas,
        "new_think_decisions": new_think,
        "screen_passed": passed == 12 and fallback_count == 0 and new_think == 0 and all(
            pooled[tier] > 0 and median(fold_deltas[tier]) > 0 for tier in changed_tiers
        ),
    }
