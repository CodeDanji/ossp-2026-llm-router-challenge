# SPDX-License-Identifier: Apache-2.0
"""Strict, deterministic persistence for PromptBudget linear artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json, math, os, tempfile
from numbers import Real
from pathlib import Path
from typing import Mapping

from .linear import LinearHead, validate_head
from .schema import PromptBudgetError
from .text_features import DENSE_FEATURE_NAMES
from ossp_router.protocol import MODEL_IDS, TIERS

ARTIFACT_TYPE = "promptbudget-router-v1"
FORMAT_VERSION = 1
_HASH_DIMENSIONS = (2**16, 2**18, 2**20)

def _num(v, name, *, positive=False, minimum=None):
    if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(float(v)):
        raise PromptBudgetError(f"{name} must be a finite number.")
    x = float(v)
    if positive and x <= 0: raise PromptBudgetError(f"{name} must be positive.")
    if minimum is not None and x < minimum: raise PromptBudgetError(f"{name} is out of range.")
    return x

def _validate_json(value, name="training_provenance"):
    if value is None or isinstance(value, (str, bool, int)): return
    if isinstance(value, float):
        if not math.isfinite(value): raise PromptBudgetError(f"{name} must be JSON-safe.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str): raise PromptBudgetError(f"{name} keys must be strings.")
            _validate_json(item, name)
        return
    if isinstance(value, (list, tuple)):
        for item in value: _validate_json(item, name)
        return
    raise PromptBudgetError(f"{name} must be JSON-safe.")

@dataclass(frozen=True)
class TierSettings:
    lambda_cost: float
    min_gain_ax31: float
    min_gain_think: float
    safety_multiplier: float
    max_relative_cost: float
    def __post_init__(self):
        _num(self.lambda_cost, "lambda_cost", minimum=0)
        _num(self.min_gain_ax31, "min_gain_ax31", minimum=0)
        _num(self.min_gain_think, "min_gain_think", minimum=0)
        _num(self.safety_multiplier, "safety_multiplier", positive=True)
        _num(self.max_relative_cost, "max_relative_cost", minimum=1)

@dataclass(frozen=True)
class PromptBudgetArtifact:
    hash_dimension: int
    dense_feature_names: tuple[str, ...]
    policy_id: str
    policy_sha256: str
    quality_heads: Mapping[str, LinearHead]
    output_heads: Mapping[str, LinearHead]
    input_head: LinearHead
    cost_residual_multipliers: Mapping[str, float]
    tiers: Mapping[str, TierSettings]
    family: str
    code_version: str
    training_provenance: Mapping[str, object]
    def __post_init__(self):
        if isinstance(self.hash_dimension, bool) or self.hash_dimension not in _HASH_DIMENSIONS: raise PromptBudgetError("hash_dimension is unsupported.")
        if tuple(self.dense_feature_names) != tuple(DENSE_FEATURE_NAMES): raise PromptBudgetError("dense_feature_names must match the feature schema.")
        if not isinstance(self.policy_id, str) or not self.policy_id.strip(): raise PromptBudgetError("policy_id must be nonblank.")
        if not isinstance(self.policy_sha256, str) or not __import__('re').fullmatch(r"[0-9a-f]{64}", self.policy_sha256): raise PromptBudgetError("policy_sha256 must be lowercase hexadecimal.")
        if self.family not in ("absolute-linear", "delta-linear"): raise PromptBudgetError("family is invalid.")
        if not isinstance(self.code_version, str) or not self.code_version.strip(): raise PromptBudgetError("code_version must be nonblank.")
        if not isinstance(self.training_provenance, Mapping): raise PromptBudgetError("training_provenance must be a mapping.")
        _validate_json(self.training_provenance)
        for heads in (self.quality_heads, self.output_heads):
            if set(heads) != set(MODEL_IDS): raise PromptBudgetError("head keys must match model IDs.")
            for head in heads.values(): validate_head(head)
        validate_head(self.input_head)
        if set(self.cost_residual_multipliers) != set(MODEL_IDS): raise PromptBudgetError("residual multiplier keys are invalid.")
        for v in self.cost_residual_multipliers.values(): _num(v, "cost_residual_multiplier", positive=True)
        if set(self.tiers) != set(TIERS): raise PromptBudgetError("tier keys are invalid.")

def _head_dict(h, hash_dimension=2**20):
    validate_head(h)
    pairs = sorted((int(i), _num(v, "sparse coefficient")) for i,v in h.sparse_coefficients.items())
    if any(i >= hash_dimension for i,_ in pairs): raise PromptBudgetError("sparse coefficient index is out of range.")
    return {"intercept": _num(h.intercept, "intercept"), "dense_coefficients": [_num(v, "dense coefficient") for v in h.dense_coefficients], "sparse_coefficients": [[i,v] for i,v in pairs]}

def artifact_to_dict(a):
    if not isinstance(a, PromptBudgetArtifact): raise PromptBudgetError("artifact must be PromptBudgetArtifact.")
    headmap=lambda m:{k:_head_dict(m[k], a.hash_dimension) for k in MODEL_IDS}
    return {"type":ARTIFACT_TYPE,"version":1,"hash_dimension":a.hash_dimension,"dense_feature_names":list(a.dense_feature_names),"policy_id":a.policy_id,"policy_sha256":a.policy_sha256,"quality_heads":headmap(a.quality_heads),"output_heads":headmap(a.output_heads),"input_head":_head_dict(a.input_head,a.hash_dimension),"cost_residual_multipliers":{k:_num(a.cost_residual_multipliers[k],k,positive=True) for k in MODEL_IDS},"tiers":{k:{"lambda_cost":a.tiers[k].lambda_cost,"min_gain_ax31":a.tiers[k].min_gain_ax31,"min_gain_think":a.tiers[k].min_gain_think,"safety_multiplier":a.tiers[k].safety_multiplier,"max_relative_cost":a.tiers[k].max_relative_cost} for k in TIERS},"family":a.family,"code_version":a.code_version,"training_provenance":dict(a.training_provenance)}

def _head(raw, dense_len, hash_dimension):
    if not isinstance(raw, Mapping) or set(raw) != {"intercept","dense_coefficients","sparse_coefficients"}: raise PromptBudgetError("invalid head fields")
    sparse=raw["sparse_coefficients"]
    if not isinstance(sparse,list): raise PromptBudgetError("sparse_coefficients must be a list")
    d={}
    for p in sparse:
        if not isinstance(p,list) or len(p)!=2 or isinstance(p[0],bool) or not isinstance(p[0],int) or p[0]<0 or p[0]>=hash_dimension or p[0] in d: raise PromptBudgetError("invalid sparse coefficient")
        d[p[0]]=_num(p[1],"sparse coefficient")
    dense=raw["dense_coefficients"]
    if not isinstance(dense,list): raise PromptBudgetError("dense_coefficients must be a list")
    if len(dense) != dense_len: raise PromptBudgetError("dense coefficient dimension is invalid")
    if any(sparse[i][0] >= sparse[i+1][0] for i in range(len(sparse)-1)): raise PromptBudgetError("sparse coefficient indexes must be ascending")
    return LinearHead(_num(raw["intercept"],"intercept"),tuple(_num(v,"dense coefficient") for v in dense),d)

def parse_artifact(raw):
    if not isinstance(raw,Mapping): raise PromptBudgetError("artifact must be an object")
    expected={"type","version","hash_dimension","dense_feature_names","policy_id","policy_sha256","quality_heads","output_heads","input_head","cost_residual_multipliers","tiers","family","code_version","training_provenance"}
    if set(raw)!=expected: raise PromptBudgetError("artifact fields are invalid")
    if raw["type"] != ARTIFACT_TYPE or raw["version"] != FORMAT_VERSION: raise PromptBudgetError("artifact type or version is invalid")
    try:
        dim = raw["hash_dimension"]; n = len(DENSE_FEATURE_NAMES)
        return PromptBudgetArtifact(dim,tuple(raw["dense_feature_names"]),raw["policy_id"],raw["policy_sha256"],{k:_head(raw["quality_heads"][k],n,dim) for k in MODEL_IDS},{k:_head(raw["output_heads"][k],n,dim) for k in MODEL_IDS},_head(raw["input_head"],n,dim),{k:_num(raw["cost_residual_multipliers"][k],k,positive=True) for k in MODEL_IDS},{k:TierSettings(**raw["tiers"][k]) for k in TIERS},raw["family"],raw["code_version"],raw["training_provenance"])
    except (KeyError,TypeError,ValueError) as e: raise PromptBudgetError("invalid artifact") from e

def _bytes(a): return (json.dumps(artifact_to_dict(a),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode()
def write_artifact(path, manifest, artifact):
    data=_bytes(artifact); path,manifest=Path(path),Path(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, data)
    digest=hashlib.sha256(data).hexdigest()
    manifest_data=(json.dumps({"artifact_file":path.name,"artifact_sha256":digest,"format_version":1},sort_keys=True,separators=(",",":"))+"\n").encode()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(manifest, manifest_data)

def _atomic_write(path, data):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
def load_artifact(path, manifest):
    path,manifest=Path(path),Path(manifest)
    try: m=json.loads(manifest.read_text(encoding="utf-8")); data=path.read_bytes()
    except Exception as e: raise PromptBudgetError("manifest or artifact could not be read") from e
    if not isinstance(m,dict) or set(m)!={"artifact_file","artifact_sha256","format_version"} or m["artifact_file"]!=path.name or m["format_version"]!=1 or m["artifact_sha256"]!=hashlib.sha256(data).hexdigest(): raise PromptBudgetError("manifest does not match artifact")
    try:
        artifact = parse_artifact(json.loads(data.decode("utf-8")))
        if _bytes(artifact) != data: raise PromptBudgetError("artifact is not canonical")
        return artifact
    except Exception as e: raise PromptBudgetError("artifact is invalid") from e
