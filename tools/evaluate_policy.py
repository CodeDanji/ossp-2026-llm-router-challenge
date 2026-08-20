# SPDX-License-Identifier: Apache-2.0
"""Evaluate a PromptBudget artifact on Dev data with aggregate-only output."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

from ossp_router.protocol import MODEL_IDS, TIERS, ProtocolError, load_bundled_policy, load_input, load_outcomes, load_policy
from ossp_router.scoring import ScoringError, score_submissions
from ossp_router.heuristic import make_submission
from promptbudget.artifact import PromptBudgetError, load_artifact
from promptbudget.input_adapter import to_prompt_record, to_submission
from promptbudget.policy import ModelPrediction, predict_models


def _write(path: Path, value: Mapping[str, object]) -> None:
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True); fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _select(predictions, settings, light_model_id):
    light = predictions[light_model_id]
    light_cost = light.c_upper * settings.safety_multiplier
    light_utility = light.quality - settings.lambda_cost * light_cost
    candidates = [ModelPrediction(light.model_id, light.quality, light.output_tokens, light_cost, light_utility)]
    for model_id, threshold in (("ax31", settings.min_gain_ax31), ("axk1-think", settings.min_gain_think)):
        prediction = predictions[model_id]
        cost = prediction.c_upper * settings.safety_multiplier
        utility = prediction.quality - settings.lambda_cost * cost
        if prediction.quality - light.quality >= threshold and cost / light_cost <= settings.max_relative_cost and utility >= light_utility:
            candidates.append(ModelPrediction(model_id, prediction.quality, prediction.output_tokens, cost, utility))
    return min(candidates, key=lambda item: (-item.utility, item.c_upper, MODEL_IDS.index(item.model_id))).model_id


def _candidate_report(scored, policy):
    tiers = {}
    strict_pass = True
    for tier in TIERS:
        item = scored["tiers"][tier]; ratio, cap = Decimal(item["budget_ratio"]), policy.tiers[tier].budget_multiplier
        passed = ratio < cap; strict_pass = strict_pass and passed
        tiers[tier] = {"actual_cost_ratio": item["budget_ratio"], "budget_multiplier": item["budget_multiplier"], "budget_margin": str(cap - ratio), "tier_score": item["tier_score"], "budget_pass": passed, "model_distribution": item["model_counts"]}
    return {"status": "pass" if strict_pass else "fail", "final_score": scored["final_score"], "tiers": tiers}


def _markdown(report):
    rows = ["# PromptBudget Dev policy comparison", "", "All values are aggregate public-Dev measurements; no prompt, ID, or outcome row is included.", "", "| Candidate | Fast cost / margin / score / distribution | Balanced cost / margin / score / distribution | Premium cost / margin / score / distribution | Weighted score | Runtime | Status |", "| --- | --- | --- | --- | ---: | --- | --- |"]
    for name, candidate in report["candidates"].items():
        if candidate["status"] == "deferred":
            rows.append("| {0} | — | — | — | — | — | deferred |".format(name)); continue
        cells = []
        for tier in TIERS:
            item = candidate["tiers"][tier]
            cells.append("{0} / {1} / {2} / {3}".format(item["actual_cost_ratio"], item["budget_margin"], item["tier_score"], json.dumps(item["model_distribution"], sort_keys=True, separators=(",", ":"))))
        rows.append("| {0} | {1} | {2} | {3} | {4} | stdlib | {5} |".format(name, cells[0], cells[1], cells[2], candidate["final_score"], candidate["status"]))
    return "\n".join(rows) + "\n"


def evaluate(args: argparse.Namespace) -> Mapping[str, object]:
    inputs, outcomes = load_input(args.input), load_outcomes(args.outcomes)
    if inputs.split != "dev" or outcomes.split != "dev": raise ValueError("evaluation requires Dev input and outcomes")
    policy = load_policy(args.policy) if args.policy else load_bundled_policy()
    artifact = load_artifact(args.artifact, args.manifest)
    base = replace(artifact, tiers={tier: replace(settings, safety_multiplier=1.0) for tier, settings in artifact.tiers.items()})
    predictions = tuple(predict_models(to_prompt_record(episode).text, TIERS[0], base, policy) for episode in inputs.episodes)
    absolute = score_submissions(inputs, outcomes, [to_submission(inputs, tier, (_select(row, artifact.tiers[tier], policy.light_model_id) for row in predictions), policy.policy_id) for tier in TIERS], policy)
    all_light = score_submissions(inputs, outcomes, [to_submission(inputs, tier, (policy.light_model_id for _ in inputs.episodes), policy.policy_id) for tier in TIERS], policy)
    heuristic = score_submissions(inputs, outcomes, [make_submission(inputs, policy, tier, strategy="prompt-heuristic") for tier in TIERS], policy)
    candidates = {"all-light": _candidate_report(all_light, policy), "official-prompt-heuristic": _candidate_report(heuristic, policy), "absolute-linear": _candidate_report(absolute, policy), "delta-linear": {"status": "deferred"}, "sparse-knn": {"status": "deferred"}}
    report = {"report_type": "promptbudget-dev-evaluation-v1", "policy_id": policy.policy_id, "policy_sha256": absolute["policy_sha256"], "selected": "absolute-linear", "status": candidates["absolute-linear"]["status"], "final_score": candidates["absolute-linear"]["final_score"], "tiers": candidates["absolute-linear"]["tiers"], "candidates": candidates}
    _write(args.report, report)
    if args.markdown_report is not None:
        args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_report.write_text(_markdown(report), encoding="utf-8", newline="\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PromptBudget on Dev data.")
    for name in ("input", "outcomes", "artifact", "manifest", "report"): parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policy", type=Path); parser.add_argument("--markdown-report", type=Path); return parser


def main(argv: Sequence[str] = None) -> int:
    try: report = evaluate(_parser().parse_args(argv))
    except (OSError, ProtocolError, PromptBudgetError, ScoringError, ValueError, ArithmeticError): print("ERROR: evaluation failed", file=sys.stderr); return 2
    print("OK: status={0}".format(report["status"])); return 0


if __name__ == "__main__": raise SystemExit(main())
