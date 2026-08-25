# SPDX-License-Identifier: Apache-2.0
"""Calibrate PromptBudget tier settings on public Dev data only."""

from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, Mapping, Sequence

from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    ProtocolError,
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_policy,
)
from ossp_router.scoring import ScoringError, score_submissions
from promptbudget.artifact import TierSettings, load_artifact, write_artifact
from promptbudget.input_adapter import to_prompt_record, to_submission
from promptbudget.policy import ModelPrediction, predict_models
from promptbudget.schema import PromptBudgetError
from promptbudget.safety import FAST_MARGIN, aggregate_upper_cost_ratio, canonical_content_group, is_fast_admissible


LAMBDAS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
MIN_GAINS = (0.0, 0.01, 0.03, 0.05)
SAFETY_MULTIPLIERS = (0.25, 0.5, 0.75, 1.0)
MAX_RELATIVE_COSTS = (2.0, 4.0, 10.0, 20.0)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{0}.tmp-{1}".format(path.name, os.getpid()))
    try:
        temporary.write_bytes(data)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_matrix(inputs, outcomes) -> None:
    if inputs.schema_version != outcomes.schema_version or inputs.challenge_id != outcomes.challenge_id or inputs.split != outcomes.split:
        raise ProtocolError("Dev input and outcomes metadata do not match")
    expected = {(episode.episode_id, model_id) for episode in inputs.episodes for model_id in MODEL_IDS}
    actual = {(outcome.episode_id, outcome.model_id) for outcome in outcomes.outcomes}
    if len(outcomes.outcomes) != len(expected) or actual != expected:
        raise ProtocolError("Dev outcomes must contain exactly every input episode and model")


def _select(predictions: Mapping[str, ModelPrediction], settings: TierSettings, light_model_id: str) -> str:
    light = predictions[light_model_id]
    light_cost = light.c_upper * settings.safety_multiplier
    light_utility = light.quality - settings.lambda_cost * light_cost
    candidates = [ModelPrediction(light.model_id, light.quality, light.output_tokens, light_cost, light_utility)]
    for model_id, threshold in (("ax31", settings.min_gain_ax31), ("axk1-think", settings.min_gain_think)):
        prediction = predictions[model_id]
        cost = prediction.c_upper * settings.safety_multiplier
        utility = prediction.quality - settings.lambda_cost * cost
        if prediction.quality - light.quality < threshold:
            continue
        if cost / (light.c_upper * settings.safety_multiplier) > settings.max_relative_cost:
            continue
        if utility >= light_utility:
            candidates.append(ModelPrediction(model_id, prediction.quality, prediction.output_tokens, cost, utility))
    return min(candidates, key=lambda item: (-item.utility, item.c_upper, MODEL_IDS.index(item.model_id))).model_id


def _policy_settings(tier: str) -> Sequence[TierSettings]:
    """Return the public-Dev search grid for every tier policy parameter."""

    if tier not in TIERS:
        raise ValueError("unknown tier")
    return (
        *(
            TierSettings(lambda_cost, min_gain, min_gain, safety_multiplier, max_relative_cost)
            for lambda_cost in LAMBDAS
            for min_gain in MIN_GAINS
            for max_relative_cost in MAX_RELATIVE_COSTS
            for safety_multiplier in SAFETY_MULTIPLIERS
        ),
        _v1_all_light_fallback(),
    )


def _v1_all_light_fallback() -> TierSettings:
    """Return the explicit safe fallback when no v2 candidate is admissible."""

    return TierSettings(100.0, 1.0, 1.0, 1.25, 1.0)


def _is_tier_admissible(tier: str, upper_cost_ratio: Decimal, tier_cap: Decimal) -> bool:
    """Apply the pre-registered Fast margin and strict limits for other tiers."""

    if tier == "fast":
        return is_fast_admissible((upper_cost_ratio,), tier_cap)
    return upper_cost_ratio < tier_cap


def _actual_cost(outcome, policy) -> Decimal:
    rates = policy.models[outcome.model_id]
    return rates.fixed_cost + (
        rates.input_token_rate * Decimal(outcome.input_tokens)
        + rates.output_token_rate * Decimal(outcome.output_tokens)
    ) / Decimal(policy.token_unit)


def _upper_cost_bounds(inputs, outcomes, predictions, decisions, settings, policy):
    """Keep the conformal request upper and clustered aggregate upper independent."""

    outcome_index = {(item.episode_id, item.model_id): item for item in outcomes.outcomes}
    upper_costs, baseline_costs, groups = [], [], []
    for episode, row, model_id in zip(inputs.episodes, predictions, decisions):
        upper_costs.append(Decimal(str(row[model_id].c_upper)) * Decimal(str(settings.safety_multiplier)))
        baseline_costs.append(_actual_cost(outcome_index[(episode.episode_id, policy.light_model_id)], policy))
        groups.append(canonical_content_group(to_prompt_record(episode).text))
    return aggregate_upper_cost_ratio(upper_costs, baseline_costs, groups)


def calibrate(args: argparse.Namespace) -> Mapping[str, object]:
    inputs = load_input(args.input)
    outcomes = load_outcomes(args.outcomes)
    if inputs.split != "dev":
        raise ProtocolError("calibration requires split='dev'")
    _validate_matrix(inputs, outcomes)
    policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
    draft = load_artifact(args.draft_artifact, args.draft_manifest)
    base = replace(
        draft,
        tiers={tier: replace(settings, safety_multiplier=1.0) for tier, settings in draft.tiers.items()},
    )
    base_predictions = tuple(
        predict_models(to_prompt_record(episode).text, TIERS[0], base, policy)
        for episode in inputs.episodes
    )
    baseline = {tier: to_submission(inputs, tier, (policy.light_model_id for _ in inputs.episodes), policy.policy_id) for tier in TIERS}
    selected: Dict[str, TierSettings] = {}
    tier_reports: Dict[str, Mapping[str, object]] = {}
    for tier in TIERS:
        best = None
        grid = _policy_settings(tier)
        for settings in grid:
            decisions = tuple(_select(predictions, settings, policy.light_model_id) for predictions in base_predictions)
            candidate = dict(baseline)
            candidate[tier] = to_submission(inputs, tier, decisions, policy.policy_id)
            scored = score_submissions(inputs, outcomes, tuple(candidate[name] for name in TIERS), policy)["tiers"][tier]
            conformal_ratio, aggregate_ratio, upper_ratio = _upper_cost_bounds(
                inputs, outcomes, base_predictions, decisions, settings, policy
            )
            actual_ratio = Decimal(str(scored["budget_ratio"]))
            if not _is_tier_admissible(tier, actual_ratio, policy.tiers[tier].budget_multiplier):
                continue
            rank = (
                Decimal(str(scored["tier_points_total"])),
                Decimal(str(settings.safety_multiplier)),
                -actual_ratio,
            )
            if best is None or rank > best[0]:
                best = (rank, settings, {
                    **scored,
                    "conformal_upper_cost_ratio": str(conformal_ratio),
                    "aggregate_upper_cost_ratio": str(aggregate_ratio),
                    "upper_cost_ratio": str(upper_ratio),
                })
        if best is None:
            selected[tier] = _v1_all_light_fallback()
            tier_reports[tier] = {"fallback": "v1-all-light", "budget_ratio": None}
            continue
        selected[tier] = best[1]
        tier_reports[tier] = best[2]
    artifact = replace(draft, tiers=selected, code_version="train-heads-dev-policy-calibrated-v2.1")
    write_artifact(args.artifact, args.manifest, artifact)
    report = {
        "report_type": "promptbudget-public-dev-policy-calibration-v2.1",
        "certificate_kind": "public validation/tuning; not an independent generalization evaluation",
        "train_frozen_structure": {
            "family": draft.family,
            "hash_dimension": draft.hash_dimension,
            "selected_sparse_features": draft.training_provenance.get("selected_sparse_feature_count"),
            "alpha": draft.training_provenance.get("alpha"),
            "residual_family": "one-sided conformal upper multiplier",
        },
        "input_sha256": _file_sha256(args.input),
        "outcomes_sha256": _file_sha256(args.outcomes),
        "draft_artifact_sha256": _file_sha256(args.draft_artifact),
        "candidate_count_per_tier": len(_policy_settings(TIERS[0])),
        "policy_id": policy.policy_id,
        "tiers": {
            tier: {
                "candidate_count": len(_policy_settings(tier)),
                "settings": {
                    "lambda_cost": selected[tier].lambda_cost,
                    "min_gain_ax31": selected[tier].min_gain_ax31,
                    "min_gain_think": selected[tier].min_gain_think,
                    "safety_multiplier": selected[tier].safety_multiplier,
                    "max_relative_cost": selected[tier].max_relative_cost,
                },
                "actual_cost_ratio": tier_reports[tier]["budget_ratio"],
                "aggregate_upper_cost_ratio": tier_reports[tier].get("aggregate_upper_cost_ratio"),
                "conformal_upper_cost_ratio": tier_reports[tier].get("conformal_upper_cost_ratio"),
                "upper_cost_ratio": tier_reports[tier].get("upper_cost_ratio"),
                "budget_margin": None if tier_reports[tier]["budget_ratio"] is None else str(policy.tiers[tier].budget_multiplier - Decimal(str(tier_reports[tier]["budget_ratio"]))),
                "tier_score": tier_reports[tier].get("tier_score"),
                "model_distribution": tier_reports[tier].get("model_counts"),
                "budget_pass": tier_reports[tier]["budget_ratio"] is not None,
                "fast_margin_required": str(FAST_MARGIN) if tier == "fast" else None,
                "fallback": tier_reports[tier].get("fallback"),
            }
            for tier in TIERS
        },
    }
    _write_json(args.report, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate PromptBudget settings on public Dev data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--draft-artifact", type=Path, required=True)
    parser.add_argument("--draft-manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = calibrate(args)
    except (OSError, ProtocolError, PromptBudgetError, ScoringError, RuntimeError, ValueError, ArithmeticError) as exc:
        print("ERROR: Dev calibration failed: {0}".format(exc), file=sys.stderr)
        return 2
    print("OK: calibrated_tiers={0}".format(len(report["tiers"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
