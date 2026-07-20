from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# v2 re-execution manifest (architecture.md sections 4.3, 5). Authored by the challenge runner;
# any miner-written manifest is ignored. The prequential bits-per-byte scoring fields are filled
# in by the scoring recast; the runner records the forced-init re-execution provenance.
RUN_MANIFEST_V2_FILENAME = "prism_run_manifest.v2.json"
RUN_MANIFEST_V2_SCHEMA_VERSION = "prism_run_manifest.v2"
# Typed, observability-only `compute` block recorded on the v2 run manifest.
COMPUTE_BLOCK_SCHEMA = "prism_compute.v1"
# Challenge-owned train time-series (telemetry-rt / VAL-TELE-*). Side-car artifact written by
# the in-container runner alongside online_loss; miner-authored copies are ignored for grade.
TRAIN_SERIES_V1_SCHEMA = "prism_train_series.v1"
TRAIN_SERIES_V1_FILENAME = "prism_train_series.v1.json"
TRAIN_SERIES_V1_JSONL_FILENAME = "prism_train_series.v1.jsonl"


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComputeBlock(SchemaModel):
    """Typed, observability-only compute record for the v2 run manifest.

    Records the GPUs actually LEASED for the scored re-execution (``gpu_count``; ``== 1`` for the
    scored single-GPU ``nproc=1`` path) plus the launch shape already known to the harness
    (``world_size`` / ``nproc_per_node`` / ``device``). It is a RECORDED field for observability and
    so VAL-GPU-005 can be asserted by field: the prequential bits-per-byte ``final_score`` derives
    SOLELY from compute-normalized learning metrics and never reads ``gpu_count`` (there is no
    GPU-count reward and no multi-GPU scaling bonus). For the single-node scored run
    ``gpu_count == world_size == nproc_per_node`` and equals the DB ``eval_jobs.actual_gpu_count``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Aliased to the JSON key "schema" (matching the score block convention) without shadowing the
    # deprecated ``BaseModel.schema`` attribute.
    compute_schema: str = Field(default=COMPUTE_BLOCK_SCHEMA, min_length=1, alias="schema")
    gpu_count: int = Field(ge=0)
    world_size: int = Field(ge=1)
    nproc_per_node: int = Field(ge=1)
    device: str = Field(min_length=1)
    max_gpu_count: int | None = Field(default=None, ge=1)
    # Realized parameter count of the model the runner ACTUALLY trained/scored (the model handed to
    # the challenge online-loss instrument), recorded so the cap can be shown to bind the scored
    # model, not just ``build_model`` in isolation (architecture.md 4.1, 6; VAL-CHEAT-022).
    model_params: int | None = Field(default=None, ge=0)
    # Dual ladder stage labels (VAL-RESLAB-003). Observability-only; never enters final_score.
    param_ladder_stage: str | None = Field(default=None, min_length=1)
    param_ladder_cap: int | None = Field(default=None, ge=0)
    # Observability-only scientific compute telemetry, all NULLABLE so manifests that predate the
    # instrumentation (or runs without a CUDA device) simply omit them. The runner measures
    # ``peak_vram_bytes`` / ``peak_rss_bytes`` / ``wall_clock_seconds`` (rank-0, exact for scored
    # ``nproc=1`` path); the host derives ``estimated_flops`` (6ND) during reconciliation. NONE of
    # these are ever read by ``score_prequential_bpb`` (they never affect ``final_score``).
    peak_vram_bytes: int | None = Field(default=None, ge=0)
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    wall_clock_seconds: float | None = Field(default=None, ge=0.0)
    estimated_flops: float | None = Field(default=None, ge=0.0)


class ExecutionMode(StrEnum):
    GPU_PROXY_EVAL = "gpu_proxy_eval"
    FULL_SCALE_EVAL = "full_scale_eval"


class DeterministicEvidence(SchemaModel):
    rule_id: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    line: int | None = Field(default=None, ge=1)
    ast_node: str | None = Field(default=None, min_length=1)
    snippet_hash: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_location(self) -> DeterministicEvidence:
        if self.line is None and self.ast_node is None:
            raise ValueError("deterministic evidence requires line or ast_node")
        return self


class ArchitectureGraph(SchemaModel):
    schema_version: str = Field(default="architecture_graph.v1", min_length=1)
    modules: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)
    parameterized_blocks: list[dict[str, Any]] = Field(default_factory=list)
    tokenizer_constraints: dict[str, Any] = Field(default_factory=dict)
    dynamic_routing: dict[str, Any] = Field(default_factory=dict)
    interface: dict[str, Any] = Field(default_factory=dict)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class DuplicateReport(SchemaModel):
    schema_version: str = Field(default="duplicate_report.v1", min_length=1)
    submission_id: str = Field(min_length=1)
    architecture_graph_hash: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    candidate_architecture_id: str | None = Field(default=None, min_length=1)
    candidate_architecture_graph_hash: str | None = Field(
        default=None, min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$"
    )
    source_similarity: float = Field(ge=0.0, le=1.0)
    graph_similarity: float = Field(ge=0.0, le=1.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    outcome: Literal["reject", "attach", "quarantine", "allow"]
    evidence: list[DeterministicEvidence] = Field(default_factory=list)
    reason: str = Field(min_length=1)
