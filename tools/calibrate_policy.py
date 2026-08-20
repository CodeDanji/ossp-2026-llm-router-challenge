# SPDX-License-Identifier: Apache-2.0
"""Calibrate PromptBudget tier settings on public Dev data only."""

from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal
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


LAMBDAS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
MIN_GAINS = (0.0, 0.01, 0.03, 0.05)
SAFETY_MULTIPLIERS = (1.0, 1.1, 1.25)
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


def _settings() -> Sequence[TierSettings]:
    grid = tuple(
        TierSettings(lambda_cost, min_gain, min_gain, safety_multiplier, max_relative_cost)
        for lambda_cost in LAMBDAS
        for min_gain in MIN_GAINS
        for safety_multiplier in SAFETY_MULTIPLIERS
        for max_relative_cost in MAX_RELATIVE_COSTS
    )
    return grid + (TierSettings(100.0, 1.0, 1.0, 1.25, 1.0),)


def _rank(report: Mapping[str, object], settings: TierSettings):
    return (
        Decimal(str(report["quality_points_total"])),
        -Decimal(str(report["total_cost"])),
        -Decimal(str(settings.lambda_cost)),
        Decimal(str(settings.min_gain_ax31)),
        -Decimal(str(settings.safety_multiplier)),
        -Decimal(str(settings.max_relative_cost)),
    )


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
    grid = _settings()
    for tier in TIERS:
        best = None
        for settings in grid:
            decisions = (_select(predictions, settings, policy.light_model_id) for predictions in base_predictions)
            candidate = dict(baseline)
            candidate[tier] = to_submission(inputs, tier, decisions, policy.policy_id)
            scored = score_submissions(inputs, outcomes, tuple(candidate[name] for name in TIERS), policy)["tiers"][tier]
            if Decimal(str(scored["budget_ratio"])) >= policy.tiers[tier].budget_multiplier:
                continue
            rank = _rank(scored, settings)
            if best is None or rank > best[0]:
                best = (rank, settings, scored)
        if best is None:
            raise RuntimeError("no strictly in-budget calibration candidate for tier {0}".format(tier))
        selected[tier] = best[1]
        tier_reports[tier] = best[2]
    artifact = replace(draft, tiers=selected, code_version="train-oof-dev-calibrated-v1")
    write_artifact(args.artifact, args.manifest, artifact)
    report = {
        "report_type": "promptbudget-dev-calibration-v1",
        "candidate_count_per_tier": len(grid),
        "policy_id": policy.policy_id,
        "tiers": {
            tier: {
                "candidate_count": len(grid),
                "settings": {
                    "lambda_cost": selected[tier].lambda_cost,
                    "min_gain_ax31": selected[tier].min_gain_ax31,
                    "min_gain_think": selected[tier].min_gain_think,
                    "safety_multiplier": selected[tier].safety_multiplier,
                    "max_relative_cost": selected[tier].max_relative_cost,
                },
                "actual_cost_ratio": tier_reports[tier]["budget_ratio"],
                "budget_margin": str(policy.tiers[tier].budget_multiplier - Decimal(str(tier_reports[tier]["budget_ratio"]))),
                "tier_score": tier_reports[tier]["tier_score"],
                "model_distribution": tier_reports[tier]["model_counts"],
                "budget_pass": Decimal(str(tier_reports[tier]["budget_ratio"])) < policy.tiers[tier].budget_multiplier,
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
