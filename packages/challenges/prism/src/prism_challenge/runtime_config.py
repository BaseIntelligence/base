from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .config import PrismSettings
from .db import loads
from .evaluator.dataset import FINEWEB_EDU_SUBSETS
from .evaluator.modes import (
    FULL_SCALE_PHASE_1_TOKEN_TARGET,
    FULL_SCALE_PHASE_2_TOKEN_TARGET,
    GPU_PROXY_TOKEN_TARGET,
)
from .evaluator.schemas import RUN_MANIFEST_V2_FILENAME


class RuntimeConfigError(RuntimeError):
    pass


def _require_sum_one(values: Mapping[str, float], label: str) -> Mapping[str, float]:
    total = sum(float(value) for value in values.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{label} must sum to 1.0, got {total:.6f}")
    return values


def _require_benchmark_sanity_secondary(value: Mapping[str, float]) -> Mapping[str, float]:
    benchmark_weight = float(value.get("benchmark_sanity", 0.0))
    other_weights = [float(weight) for key, weight in value.items() if key != "benchmark_sanity"]
    if benchmark_weight > 0.15:
        raise ValueError("benchmark_sanity cannot exceed the capped 0.15 score share")
    if other_weights and benchmark_weight >= max(other_weights):
        raise ValueError("benchmark_sanity cannot be the primary architecture/training signal")
    return value


class WeightPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: float = Field(ge=0, le=1)
    training: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> WeightPair:
        _require_sum_one(
            {"architecture": self.architecture, "training": self.training}, "reward_pools"
        )
        return self


class ScoreWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_architecture_weight: float = Field(ge=0, le=1)
    final_recipe_weight: float = Field(ge=0, le=1)
    architecture_formula: dict[str, float]
    training_formula: dict[str, float]

    @field_validator("architecture_formula", "training_formula")
    @classmethod
    def formula_sums_to_one(cls, value: dict[str, float]) -> dict[str, float]:
        summed = _require_sum_one(value, "score formula weights")
        return dict(_require_benchmark_sanity_secondary(summed))

    @model_validator(mode="after")
    def final_weights_sum_to_one(self) -> ScoreWeights:
        _require_sum_one(
            {
                "final_architecture_weight": self.final_architecture_weight,
                "final_recipe_weight": self.final_recipe_weight,
            },
            "final_score weights",
        )
        return self


class BenchmarkWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mmlu: float = Field(default=0.20, ge=0, le=1)
    gsm8k: float = Field(default=0.15, ge=0, le=1)
    math: float = Field(default=0.15, ge=0, le=1)
    humaneval: float = Field(default=0.15, ge=0, le=1)
    arc_challenge: float = Field(default=0.10, ge=0, le=1)
    needle: float = Field(default=0.10, ge=0, le=1)
    ifeval: float = Field(default=0.10, ge=0, le=1)
    truthfulqa: float = Field(default=0.05, ge=0, le=1)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> BenchmarkWeights:
        _require_sum_one(self.model_dump(), "benchmark_weights")
        return self


class DuplicateThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact_source_similarity: float = Field(default=0.98, ge=0, le=1)
    quarantine_source_similarity: float = Field(default=0.85, ge=0, le=1)
    same_architecture_similarity: float = Field(default=0.82, ge=0, le=1)
    static_reject_similarity: float = Field(default=0.96, ge=0, le=1)


class GpuPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_gpu_count: int = Field(default=8, ge=1, le=8)
    actual_gpu_count: int = Field(default=1, ge=1, le=8)
    gpu_type: str | None = None
    official_fixed_profile: bool = True
    allocation_policy: str = "fixed_official_profile"

    @model_validator(mode="after")
    def actual_count_within_declared_max(self) -> GpuPolicy:
        if self.actual_gpu_count > self.max_gpu_count:
            raise ValueError("actual_gpu_count must be <= max_gpu_count")
        return self


class DatasetConfigs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fineweb_sample_count: int = Field(default=128, ge=1)
    frozen_revision: str = "bootstrap-default"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    network_fallback_allowed: bool = False


class ExecutionModeTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_proxy_eval: dict[str, Any]
    full_scale_eval: dict[str, Any]


class ArtifactLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_code_bytes: int = Field(default=7_500_000, ge=1)
    max_files: int = Field(default=200, ge=1)
    max_bytes: int = Field(default=2_000_000, ge=1)
    required_manifest_name: str = RUN_MANIFEST_V2_FILENAME


class SandboxLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docker_enabled: bool = False
    cpus: float = Field(default=1.0, gt=0)
    memory: str = "512m"
    pids_limit: int = Field(default=128, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)
    network: str = "none"
    read_only: bool = True


class DiagnosticsThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activation_norm_spike_ratio: float = Field(default=3.0, gt=0)
    gradient_noise_max: float = Field(default=10.0, gt=0)
    min_attention_diversity: float = Field(default=0.05, ge=0, le=1)
    representation_collapse_max: float = Field(default=0.95, ge=0, le=1)


class LossComparabilityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main_track_requires_comparable_loss: bool = True
    allow_redistribution_for_non_main_track: bool = True
    redistribution_policy: str = "redistribute_to_defined_components"
    byte_normalized_fallback: bool = True


class RuntimePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward_pools: WeightPair
    score_weights: ScoreWeights
    benchmark_weights: BenchmarkWeights
    duplicate_thresholds: DuplicateThresholds
    gpu_policy: GpuPolicy
    dataset_configs: DatasetConfigs
    execution_mode_targets: ExecutionModeTargets
    artifact_limits: ArtifactLimits
    sandbox_limits: SandboxLimits
    diagnostics_thresholds: DiagnosticsThresholds
    loss_comparability_policy: LossComparabilityPolicy


RUNTIME_CONFIG_KEYS = frozenset(RuntimePolicy.model_fields)


def runtime_policy_defaults(settings: PrismSettings) -> dict[str, Any]:
    # VAL-RESLAB-008: architecture/training emission pools default 0.50/0.50 consistently
    # with get_weights + PrismSettings.architecture_reward_weight / training_reward_weight.
    return {
        "reward_pools": {
            "architecture": 0.50,
            "training": 0.50,
        },
        "score_weights": {
            "final_architecture_weight": settings.arch_weight,
            "final_recipe_weight": settings.recipe_weight,
            "architecture_formula": {
                "learning_scaling_dynamics": 0.35,
                "standardized_lm_quality": 0.20,
                "compute_efficiency": 0.15,
                "parameter_efficiency": 0.10,
                "diagnostics_health": 0.10,
                "robustness_stability": 0.05,
                "benchmark_sanity": 0.05,
            },
            "training_formula": {
                "architecture_normalized_heldout_improvement": 0.30,
                "learning_stability_dynamics": 0.25,
                "benchmark_sanity": 0.15,
                "compute_efficiency": 0.10,
                "reproducibility_stability": 0.10,
                "robustness_failure_behavior": 0.05,
                "artifact_completeness": 0.05,
            },
        },
        "benchmark_weights": BenchmarkWeights().model_dump(),
        "duplicate_thresholds": {
            "exact_source_similarity": 0.98,
            "quarantine_source_similarity": 0.85,
            # Deterministic source-similarity default (former component-agent knob removed).
            "same_architecture_similarity": 0.82,
            "static_reject_similarity": settings.plagiarism_static_reject_threshold,
        },
        "gpu_policy": {
            "max_gpu_count": settings.base_eval_max_gpu_count,
            "actual_gpu_count": settings.base_eval_gpu_count,
            "gpu_type": settings.base_eval_gpu_type,
            "official_fixed_profile": True,
            "allocation_policy": "fixed_official_profile",
        },
        "dataset_configs": {
            "fineweb_sample_count": settings.fineweb_sample_count,
            "frozen_revision": "bootstrap-default",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "network_fallback_allowed": False,
        },
        "execution_mode_targets": {
            "gpu_proxy_eval": {
                "official_score": True,
                "max_tokens": GPU_PROXY_TOKEN_TARGET,
                "gpu_count": 1,
                "dataset_subset": "sample-10BT",
                "dataset_tokens": int(FINEWEB_EDU_SUBSETS["sample-10BT"]["token_count"]),
            },
            "full_scale_eval": {
                "official_score": True,
                "max_tokens": FULL_SCALE_PHASE_1_TOKEN_TARGET,
                "phase_1_max_tokens": FULL_SCALE_PHASE_1_TOKEN_TARGET,
                "phase_2_max_tokens": FULL_SCALE_PHASE_2_TOKEN_TARGET,
                "gpu_count": 1,
                "phase_1_dataset_subset": "sample-10BT",
                "phase_2_dataset_subset": "sample-100BT",
                "phase_1_dataset_tokens": int(FINEWEB_EDU_SUBSETS["sample-10BT"]["token_count"]),
                "phase_2_dataset_tokens": int(FINEWEB_EDU_SUBSETS["sample-100BT"]["token_count"]),
            },
        },
        "artifact_limits": {
            "max_code_bytes": settings.max_code_bytes,
            "max_files": settings.plagiarism_storage_max_files,
            "max_bytes": settings.plagiarism_storage_max_bytes,
            "required_manifest_name": RUN_MANIFEST_V2_FILENAME,
        },
        "sandbox_limits": {
            "docker_enabled": settings.docker_enabled,
            "cpus": settings.docker_cpus,
            "memory": settings.docker_memory,
            "pids_limit": settings.docker_pids_limit,
            "timeout_seconds": settings.plagiarism_sandbox_timeout_seconds,
            "network": settings.docker_network,
            "read_only": settings.docker_read_only,
        },
        "diagnostics_thresholds": DiagnosticsThresholds().model_dump(),
        "loss_comparability_policy": LossComparabilityPolicy().model_dump(),
    }


def resolve_runtime_policy(
    settings: PrismSettings,
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_sql_fallback: bool = False,
) -> RuntimePolicy:
    defaults = runtime_policy_defaults(settings)
    payload = _merge_sql_values(defaults, rows)
    try:
        return RuntimePolicy.model_validate(payload)
    except ValidationError as exc:
        if allow_sql_fallback:
            return RuntimePolicy.model_validate(defaults)
        raise RuntimeConfigError(f"invalid SQL runtime config: {exc}") from exc


def _merge_sql_values(
    defaults: dict[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    payload = {key: _copy_value(value) for key, value in defaults.items()}
    for row in rows:
        key = str(row["config_key"])
        if key not in RUNTIME_CONFIG_KEYS:
            continue
        value = loads(str(row["value_json"]))
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return payload


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_copy_value(inner) for inner in value]
    return value
