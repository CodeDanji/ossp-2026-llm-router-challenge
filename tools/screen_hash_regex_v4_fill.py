"""Run the fixed Train-only v4 AX31-fill triage screen."""

from __future__ import annotations

import argparse
import dataclasses
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np

import hash_regex_cost_stabilization_nested as v32
import hash_regex_tail_guard_nested as nested
import hash_regex_v4_fill_screen as v4
from hash_regex_v4_fill_screen import (
    CANDIDATES,
    blocked_upgrade_diagnostic,
    screen_candidate,
)
from ossp_router.protocol import MODEL_IDS, ProtocolError, load_bundled_policy, load_input, load_outcomes
from promptbudget.safety import grouped_folds
from validate_data import validate_batches


TRAIN_ROWS = 1760
SEED = 137
FOLDS = 4
REPORT_DIRECTORY = "hash-regex-v4-deadline-triage"


def reject_dev_path(path: Path) -> None:
    if "dev" in str(path).lower():
        raise ValueError("v4 fill screening does not accept a Dev path")


def require_v4_report_path(path: Path) -> Path:
    resolved = path.resolve()
    if not any(parent.name == REPORT_DIRECTORY and parent.parent.name == "build" for parent in resolved.parents):
        raise ValueError(f"report output must be below build/{REPORT_DIRECTORY}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _official_recoverable_ax31_pool(
    data: v32.EvaluationData,
    indices: Sequence[int],
    baseline: Mapping[str, v4.Route],
    tier: str,
    metrics: Mapping[str, object],
) -> tuple[Decimal, int]:
    """Return Decimal scorer slack and AX31 upgrades that individually fit it."""

    baseline_cost = Decimal(str(metrics["total_cost"]))
    slack = Decimal(str(metrics["budget_limit"])) - baseline_cost
    if slack <= 0:
        return slack, 0
    recoverable = 0
    for local, (index, choice) in enumerate(zip(indices, baseline[tier].choices)):
        if choice != MODEL_IDS[0] or data.scores[index, 1] <= data.scores[index, 0]:
            continue
        choices = list(baseline[tier].choices)
        choices[local] = MODEL_IDS[1]
        candidate = dict(baseline)
        candidate[tier] = v4.Route(tuple(choices), baseline[tier].predicted_ratio)
        upgraded_cost = Decimal(str(v4._score_routes(data, tuple(indices), candidate)["tiers"][tier]["total_cost"]))
        if Decimal("0") < upgraded_cost - baseline_cost <= slack:
            recoverable += 1
    return slack, recoverable


def _run_diagnostic(data: v32.EvaluationData, folds: Sequence[object]) -> dict[str, object]:
    """Use the frozen guarded route to find held-out Fast/Balance recovery signal."""

    tier_totals = {
        tier: {"blocked_observed_score_gain_total": 0.0, "slack_recoverable_ax31_pool_count": 0}
        for tier in ("fast", "balanced")
    }
    records = []
    for number, fold in enumerate(folds):
        train_indices = tuple(fold.train_indices)
        validation_indices = tuple(fold.validation_indices)
        guard = nested.fit_tail_guard(data, train_indices, seed=SEED)
        quality = v32.fit_raw_quality_heads(data.matrix[list(train_indices)], data.scores[list(train_indices)])
        costs = v32.fit_log_cost_heads(data.matrix[list(train_indices)], data.log_costs[list(train_indices)])
        score_prediction = v32.predict_heads(quality, data.matrix[list(validation_indices)])
        cost_prediction = v32.predict_heads(costs, data.matrix[list(validation_indices)])
        score_rows, guarded_cost_rows, guard_buckets = v4._prediction_rows(
            score_prediction, cost_prediction, guard
        )
        _raw_scores, raw_cost_rows, _raw_buckets = v4._prediction_rows(
            score_prediction, cost_prediction, None
        )
        baseline = v4._baseline_routes(score_rows, guarded_cost_rows, data)
        raw_baseline = v4._baseline_routes(score_rows, raw_cost_rows, data)
        guarded = {**baseline, "guard_buckets": guard_buckets}
        diagnostic = blocked_upgrade_diagnostic(data, validation_indices, raw_baseline, guarded)
        baseline_report = v4._score_routes(data, validation_indices, baseline)
        tier_record = {}
        for tier in ("fast", "balanced"):
            item = diagnostic[tier]
            blocked_gain = sum(float(row["observed_score_gain_vs_light"]) for row in item["rows"])
            slack, recoverable = _official_recoverable_ax31_pool(
                data, validation_indices, baseline, tier, baseline_report["tiers"][tier]
            )
            tier_totals[tier]["blocked_observed_score_gain_total"] += blocked_gain
            tier_totals[tier]["slack_recoverable_ax31_pool_count"] += recoverable
            tier_record[tier] = {
                "blocked_row_count": item["blocked_row_count"],
                "blocked_observed_score_gain_total": blocked_gain,
                "official_actual_budget_slack": slack,
                "slack_recoverable_ax31_pool_count": recoverable,
            }
        records.append({"fold": number, "tiers": tier_record})
    for tier, total in tier_totals.items():
        total["recovery_signal"] = (
            total["blocked_observed_score_gain_total"] > 0
            and total["slack_recoverable_ax31_pool_count"] > 0
        )
    return {
        "folds": tuple(records),
        "tiers": tier_totals,
        "target_tier_recovery_signal": any(total["recovery_signal"] for total in tier_totals.values()),
    }


def _candidate_summary(result: Mapping[str, object]) -> dict[str, object]:
    return {
        key: result[key]
        for key in (
            "candidate",
            "actual_checks",
            "actual_checks_required",
            "actual_checks_passed",
            "fallback_count",
            "changed_tier_pooled_cap_neutral_quality_deltas",
            "fold_deltas",
            "new_think_decisions",
            "screen_passed",
        )
    }


def _winner_key(result: Mapping[str, object]) -> tuple[Decimal, Decimal, str]:
    deltas = result["changed_tier_pooled_cap_neutral_quality_deltas"]
    weights = {"fast": Decimal("0.4"), "balanced": Decimal("0.3"), "premium": Decimal("0.3")}
    weighted_delta = sum((weights[tier] * Decimal(str(value)) for tier, value in deltas.items()), Decimal("0"))
    maximum_ratio = max(Decimal(str(check["actual_ratio"])) for check in result["actual_checks"])
    return (-weighted_delta, maximum_ratio, str(result["candidate"]))


def screen(args: argparse.Namespace) -> Mapping[str, object]:
    for path in (args.input, args.outcomes, args.report):
        reject_dev_path(path)
    report_path = require_v4_report_path(args.report)
    inputs, outcomes = load_input(args.input), load_outcomes(args.outcomes)
    rows, _ = validate_batches(inputs, outcomes)
    if inputs.split != "train" or outcomes.split != "train" or rows != TRAIN_ROWS:
        raise ValueError("v4 fill screening requires exactly 1,760 Train rows")
    provenance = {
        "input_sha256": _sha256(args.input),
        "outcomes_sha256": _sha256(args.outcomes),
        "row_count": rows,
    }
    if not args.execute:
        return {
            "report_type": "hash-regex-v4-fill-screen-v1",
            "mode": "dry-run",
            "training_provenance": provenance,
            "terminal_status": "not-executed",
            "forbidden_artifact_assertion": True,
        }
    if report_path.exists():
        raise ValueError("v4 fill screen report already exists; reruns are forbidden")

    data = v32.make_evaluation_data(inputs, outcomes, load_bundled_policy())
    folds = grouped_folds(data.groups, folds=FOLDS, seed=SEED)
    diagnostic = _run_diagnostic(data, folds)
    candidates: list[Mapping[str, object]] = []
    if not diagnostic["target_tier_recovery_signal"]:
        terminal_status = "no-recovery-signal"
        winner = None
    else:
        for candidate in CANDIDATES[:2]:
            candidates.append(screen_candidate(data, folds, candidate))
        if any(bool(candidate["screen_passed"]) for candidate in candidates):
            candidates.append(screen_candidate(data, folds, CANDIDATES[2]))
        winners = [candidate for candidate in candidates if candidate["screen_passed"]]
        if winners:
            winner = min(winners, key=_winner_key)
            terminal_status = "screen-winner"
        else:
            winner = None
            terminal_status = "no-safe-recovery-candidate"
    report = {
        "report_type": "hash-regex-v4-fill-screen-v1",
        "mode": "execute",
        "training_provenance": provenance,
        "v3_3_comparator": "tail-guarded",
        "seed": SEED,
        "fold_count": FOLDS,
        "diagnostic": diagnostic,
        "candidates": tuple(_candidate_summary(candidate) for candidate in candidates),
        "terminal_status": terminal_status,
        "winner_candidate": None if winner is None else winner["candidate"],
        "forbidden_artifact_assertion": True,
    }
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed Train-only v4 fill screen.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = screen(_parser().parse_args(argv))
    except (OSError, ProtocolError, ValueError, ArithmeticError) as error:
        print(f"ERROR: v4 fill screening failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"mode": report["mode"], "terminal_status": report["terminal_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
