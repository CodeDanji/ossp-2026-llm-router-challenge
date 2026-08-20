"""Production selection from a persisted, content-only PromptBudget artifact."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from typing import Mapping

from ossp_router.protocol import MODEL_IDS, RoutingPolicy, policy_sha256

from .artifact import PromptBudgetArtifact
from .linear import predict_head
from .schema import PromptBudgetError
from .text_features import extract_features


@dataclass(frozen=True)
class ModelPrediction:
    model_id: str
    quality: float
    output_tokens: float
    c_upper: float
    utility: float


def _tokens(log_value: float) -> float:
    if not math.isfinite(log_value):
        raise PromptBudgetError("token prediction must be finite.")
    return max(0.0, math.expm1(max(-50.0, min(50.0, log_value))))


def _validate(text: str, tier: str, artifact: PromptBudgetArtifact, policy: RoutingPolicy):
    if tier not in policy.tiers or tier not in artifact.tiers:
        raise PromptBudgetError(f"unknown tier: {tier!r}")
    if artifact.family != "absolute-linear":
        raise PromptBudgetError("delta-linear artifacts are not supported for production selection")
    if artifact.policy_id != policy.policy_id or artifact.policy_sha256 != policy_sha256(policy):
        raise PromptBudgetError("artifact policy does not match routing policy")
    return artifact.tiers[tier]


def predict_models(text: str, tier: str, artifact: PromptBudgetArtifact, routing_policy: RoutingPolicy) -> Mapping[str, ModelPrediction]:
    settings = _validate(text, tier, artifact, routing_policy)
    vector = extract_features(text, artifact.hash_dimension)
    input_tokens = _tokens(predict_head(artifact.input_head, vector))
    result = {}
    for model_id in MODEL_IDS:
        quality = max(0.0, min(1.0, predict_head(artifact.quality_heads[model_id], vector)))
        output_tokens = _tokens(predict_head(artifact.output_heads[model_id], vector))
        rates = routing_policy.models[model_id]
        residual = Decimal(str(artifact.cost_residual_multipliers[model_id]))
        safety = Decimal(str(settings.safety_multiplier))
        unit = Decimal(routing_policy.token_unit)
        cost = (rates.fixed_cost + (rates.input_token_rate * Decimal(str(input_tokens)) + rates.output_token_rate * Decimal(str(output_tokens))) / unit) * residual * safety
        c_upper = float(cost)
        if not math.isfinite(c_upper) or c_upper <= 0:
            raise PromptBudgetError("predicted cost upper must be finite and positive")
        utility = quality - settings.lambda_cost * c_upper
        result[model_id] = ModelPrediction(model_id, quality, output_tokens, c_upper, utility)
    return result


def select_model(text: str, tier: str, artifact: PromptBudgetArtifact, routing_policy: RoutingPolicy) -> str:
    predictions = predict_models(text, tier, artifact, routing_policy)
    settings = artifact.tiers[tier]
    light = predictions[routing_policy.light_model_id]
    candidates = [light]
    for model_id, gain in (("ax31", settings.min_gain_ax31), ("axk1-think", settings.min_gain_think)):
        prediction = predictions[model_id]
        if prediction.quality - light.quality < gain:
            continue
        if prediction.c_upper / light.c_upper > settings.max_relative_cost:
            continue
        if prediction.utility >= light.utility:
            candidates.append(prediction)
    return min(candidates, key=lambda p: (-p.utility, p.c_upper, MODEL_IDS.index(p.model_id))).model_id
