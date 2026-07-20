from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from base.challenge_sdk.config import ChallengeSettings
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import SettingsConfigDict

_DATA_TMP_ARTIFACT_ROOT = Path("/data/tmp/prism-eval-artifacts")
_TMP_ARTIFACT_ROOT = Path("/tmp/prism-eval-artifacts")


def _path_parent_is_creatable(path: Path) -> bool:
    """True when ``path`` can be mkdir'd without permission errors (probe, rollback)."""
    try:
        if path.exists():
            return path.is_dir() and os.access(path, os.W_OK | os.X_OK)
        # Find nearest existing ancestor that would host the remaining parents.
        existing = path.parent
        created: list[Path] = []
        while not existing.exists() and existing != existing.parent:
            created.append(existing)
            existing = existing.parent
        if not existing.exists() or not existing.is_dir():
            return False
        if not os.access(existing, os.W_OK | os.X_OK):
            return False
        # Real probe: mkdir intermediates + leaf, cleanup what we created.
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
        for created_path in reversed(created):
            try:
                created_path.rmdir()
            except OSError:
                break
        return True
    except OSError:
        return False


def _default_base_eval_artifact_root() -> Path:
    """Compose prefers ``/data/tmp``; CI/dev hosts without ``/data`` use temp/tmp."""
    if _path_parent_is_creatable(_DATA_TMP_ARTIFACT_ROOT):
        return _DATA_TMP_ARTIFACT_ROOT
    if _path_parent_is_creatable(_TMP_ARTIFACT_ROOT):
        return _TMP_ARTIFACT_ROOT
    return Path(tempfile.gettempdir()) / "prism-eval-artifacts"


_PROOF_RUNTIME_ENVIRONMENT_NAMES = frozenset(
    {
        "PRISM_ATTESTATION",
        "PRISM_EXECUTOR_ID",
        "PRISM_IMAGE_DIGEST",
        "PRISM_MINER_HOTKEY",
        "PRISM_POD_ID",
        "PRISM_PROVIDER_NAME",
    }
)


class WorkerPlaneConfig(BaseModel):
    """Prism worker-plane feature block (architecture.md 3.4/3.5).

    OFF by default: with ``enabled`` false prism behaves exactly as before the compute plane (no
    ExecutionProof emission, no admission gate, legacy audit-free finalization). Env overrides use
    the nested delimiter, e.g. ``PRISM_WORKER_PLANE__ENABLED=true``.
    """

    model_config = SettingsConfigDict(extra="forbid")

    enabled: bool = False
    admission_requires_worker: bool = False
    # Base master coordination base URL the admission rule queries for >=1 active worker bound to
    # the submitting hotkey (``GET {master_base_url}/v1/workers/active?hotkey=``). Auth reuses the
    # existing prism<->master bridge shared token as the bearer. Unset while
    # ``admission_requires_worker`` is on => admission fails closed (no submission accepted), which
    # is the same deterministic rejection as an explicit zero-worker answer (architecture.md 3.5).
    master_base_url: str | None = None
    # Bounded admission-check latency (seconds). A master that is unreachable/slow/5xx must never
    # hang a submission: the query is capped here and any failure folds into the fail-closed
    # NO_ACTIVE_WORKER rejection (architecture.md 3.5; VAL-PRISM-020).
    admission_timeout_seconds: float = Field(default=5.0, gt=0.0)
    audit_rate_tier0: float = Field(default=0.10, ge=0.0, le=1.0)
    audit_rate_tier1: float = Field(default=0.05, ge=0.0, le=1.0)
    audit_rate_tier2: float = Field(default=0.02, ge=0.0, le=1.0)
    # Per-audit claim lease (seconds). The validator audit cycle claims each pending audit under
    # this lease before replaying it, so in a MULTI-validator deployment each pending audit is
    # replayed by at most one validator (idempotent-but-wasteful redundant GPU/CPU replays are
    # avoided). A crashed claimant's lease expires and the audit becomes reclaimable; the default is
    # generous enough to exceed a normal replay wall-time. Single-validator behaviour is unchanged.
    audit_claim_lease_seconds: float = Field(default=1800.0, gt=0.0)
    # Server-side SECRET salt mixed into the audit sampler seed so audit selection cannot be
    # predicted from the public ``submission_id`` alone (a different salt selects a different set),
    # while staying reproducible for a fixed salt and preserving the per-tier rates (architecture.md
    # 3.4; VAL-FINAL-006). Kept out of the config repr like every other secret.
    audit_salt: str | None = Field(default=None, repr=False)
    # sr25519 signing key (URI ``//Name`` / mnemonic / seed) for the worker that emits
    # ExecutionProofs. This is the worker's OWN key, injected by the worker agent -- NEVER a
    # master-side secret. Unset -> prism emits no signed proof (the base worker plane may still
    # stamp a tier-0 proof from the manifest hash).
    signing_key: str | None = Field(default=None, repr=False)
    # Pinned evaluator/worker image digest (``sha256:<64hex>``) a claimed tier-1 proof is checked
    # against at ingestion: a tier-1 claim whose ``image_digest`` does not match this value is not
    # verifiable, so its EFFECTIVE tier is downgraded to 0 for audit sampling (architecture.md 3.4;
    # VAL-PRISM-019). Unset -> no tier-1 claim is verifiable, so every tier-1 claim downgrades to 0.
    pinned_image_digest: str | None = Field(default=None)
    # EXPLICIT test-mode config that swaps the docker/broker executor for the repo's OWN CPU
    # re-exec seam (``evaluator.mock_reexec.cpu_reexec_run``): a real, deterministic
    # ``prism_run_manifest.v2`` is produced on CPU with no GPU/Docker/broker. This is opt-in and
    # OFF by default (production always uses the real broker). It exists so a local mission harness
    # can stand up a faithful worker/audit-replay path on a CPU-only host. When
    # ``cpu_reexec_train_data_dir`` is unset a tiny locked train shard is staged under the eval
    # artifact root; the tiny vocab/seq/step budget keep a scored run fast + deterministic.
    cpu_reexec_test_mode: bool = False
    cpu_reexec_train_data_dir: str | None = None
    cpu_reexec_vocab_size: int = Field(default=64, ge=2)
    cpu_reexec_sequence_length: int = Field(default=16, ge=2)
    cpu_reexec_seed: int = 1234
    cpu_reexec_step_budget: int = Field(default=24, ge=1)
    cpu_reexec_train_lines: int = Field(default=64, ge=1)


REMOVED_LLM_SETTING_NAMES = frozenset(
    {
        "llm_review_enabled",
        "llm_review_required",
        "llm_gateway_url",
        "llm_gateway_token",
        "llm_gateway_token_file",
        "llm_review_timeout_seconds",
        "held_review_timeout_seconds",
        "llm_review_temperature",
        "llm_review_max_tokens",
        "llm_review_max_retries",
        "llm_review_max_source_chars",
        # Residual nondeterministic component-agent knobs (VAL-GATE-007).
        "component_agent_enabled",
        "component_agent_required",
        "component_agent_model",
        "component_agent_min_confidence",
        "component_agent_transfer_confidence",
        "component_agent_same_threshold",
        "component_agent_hold_threshold",
        "component_agent_candidate_top_k",
        "component_agent_mermaid_enabled",
        "component_hold_low_confidence",
    }
)
REMOVED_LLM_ENV_NAMES = frozenset(
    {
        "PRISM_LLM_REVIEW_ENABLED",
        "PRISM_LLM_REVIEW_REQUIRED",
        "PRISM_LLM_GATEWAY_URL",
        "PRISM_GATEWAY_TOKEN",
        "PRISM_GATEWAY_TOKEN_FILE",
        "BASE_LLM_GATEWAY_URL",
        "BASE_GATEWAY_TOKEN",
        "BASE_GATEWAY_TOKEN_FILE",
        "PRISM_LLM_REVIEW_TIMEOUT_SECONDS",
        "PRISM_HELD_REVIEW_TIMEOUT_SECONDS",
        "PRISM_LLM_REVIEW_TEMPERATURE",
        "PRISM_LLM_REVIEW_MAX_TOKENS",
        "PRISM_LLM_REVIEW_MAX_RETRIES",
        "PRISM_LLM_REVIEW_MAX_SOURCE_CHARS",
        "PRISM_COMPONENT_AGENT_ENABLED",
        "PRISM_COMPONENT_AGENT_REQUIRED",
        "PRISM_COMPONENT_AGENT_MODEL",
        "PRISM_COMPONENT_AGENT_MIN_CONFIDENCE",
        "PRISM_COMPONENT_AGENT_TRANSFER_CONFIDENCE",
        "PRISM_COMPONENT_AGENT_SAME_THRESHOLD",
        "PRISM_COMPONENT_AGENT_HOLD_THRESHOLD",
        "PRISM_COMPONENT_AGENT_CANDIDATE_TOP_K",
        "PRISM_COMPONENT_AGENT_MERMAID_ENABLED",
        "PRISM_COMPONENT_HOLD_LOW_CONFIDENCE",
    }
)


class PrismSettings(ChallengeSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRISM_",
        env_file=".env",
        extra="forbid",
        populate_by_name=True,
        env_nested_delimiter="__",
    )

    def __init__(self, **values: Any) -> None:
        removed = sorted(set(values) & REMOVED_LLM_SETTING_NAMES)
        if removed:
            raise ValueError(
                "Unsupported removed Prism LLM configuration keys: "
                + ", ".join(removed)
                + ". Deterministic admission no longer accepts gateway/review or "
                "component-agent settings."
            )
        env_hits = sorted(
            name
            for name in os.environ
            if name in REMOVED_LLM_ENV_NAMES
            or (
                name.startswith("PRISM_")
                and (
                    ("LLM" in name and "GATEWAY" in name)
                    or name.startswith("PRISM_COMPONENT_AGENT_")
                    or name == "PRISM_COMPONENT_HOLD_LOW_CONFIDENCE"
                )
            )
            or name
            in {
                "BASE_LLM_GATEWAY_URL",
                "BASE_GATEWAY_TOKEN",
                "BASE_GATEWAY_TOKEN_FILE",
            }
        )
        if env_hits:
            raise ValueError(
                "Unsupported removed Prism LLM environment keys: "
                + ", ".join(env_hits)
                + ". Deterministic admission no longer accepts gateway/review or "
                "component-agent settings."
            )
        known = {
            *self._known_environment_names(),
            *_PROOF_RUNTIME_ENVIRONMENT_NAMES,
            "PRISM_ENV_FILE",
        }
        unknown = sorted(
            name for name in os.environ if name.startswith("PRISM_") and name not in known
        )
        if unknown:
            raise ValueError(f"Unknown Prism configuration key: {unknown[0]}")
        super().__init__(**values)

    @classmethod
    def _known_environment_names(cls) -> set[str]:
        names: set[str] = set()
        for field_name, field in cls.model_fields.items():
            aliases = field.validation_alias
            if isinstance(aliases, AliasChoices):
                names.update(
                    alias
                    for alias in aliases.choices
                    if isinstance(alias, str) and alias.startswith("PRISM_")
                )
            elif isinstance(aliases, str) and aliases.startswith("PRISM_"):
                names.add(aliases)
            names.add(f"PRISM_{field_name.upper()}")
            if field_name == "worker_plane":
                for nested_name in WorkerPlaneConfig.model_fields:
                    names.add(f"PRISM_WORKER_PLANE__{nested_name.upper()}")
        return names

    worker_plane: WorkerPlaneConfig = Field(default_factory=WorkerPlaneConfig)

    database_url: str = Field(
        default="sqlite+aiosqlite:////data/prism.sqlite3",
        validation_alias=AliasChoices("PRISM_DATABASE_URL", "CHALLENGE_DATABASE_URL"),
    )
    slug: str = "prism"
    name: str = "Prism"
    version: str = "0.1.0"
    api_version: str = "1.0"
    sdk_version: str = "1.0.0"
    raw_weight_push_enabled: bool = True
    # Master coordination base URL for authenticated raw-weight push. When set
    # together with the challenge shared token, Prism starts the push loop.
    master_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PRISM_MASTER_BASE_URL",
            "CHALLENGE_MASTER_BASE_URL",
            "MASTER_BASE_URL",
        ),
    )
    raw_weight_push_interval_seconds: float = Field(default=30.0, ge=0.1)
    raw_weight_push_freshness_seconds: int = Field(default=300, ge=30)
    raw_weight_push_timeout_seconds: float = Field(default=10.0, gt=0.0)
    capabilities: tuple[str, ...] = Field(
        default_factory=lambda: (
            "challenge.scoring",
            "challenge.ordinary_proof",
            "challenge.state",
            "challenge.raw_weight_push",
        )
    )
    port: int = 8080
    database_path: Path = Path("/tmp/prism.sqlite3")
    shared_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("PRISM_SHARED_TOKEN", "CHALLENGE_SHARED_TOKEN"),
    )
    shared_token_file: str | None = Field(
        default="/run/secrets/base/challenge_token",
        repr=False,
        validation_alias=AliasChoices("PRISM_SHARED_TOKEN_FILE", "CHALLENGE_SHARED_TOKEN_FILE"),
    )
    allow_insecure_signatures: bool = False
    signature_ttl_seconds: int = 300
    epoch_seconds: int = 21_600
    max_code_bytes: int = 7_500_000
    # Dual param ladder (VAL-RESLAB-003): explore/provisional default = 124M continuous admission.
    # Promote-stage runs use ``promote_max_parameters`` (350M) with ``param_ladder_stage=promote``.
    # Legacy single 150M emission default is gone from the dual-ladder path.
    max_parameters: int = 124_000_000
    explore_max_parameters: int = 124_000_000
    promote_max_parameters: int = 350_000_000
    # Default emission admission stage label (explore | promote). Promote path is opt-in.
    param_ladder_stage: Literal["explore", "promote"] = "explore"
    max_layers: int = 96
    # Sequence geometry for worker-plane PrismContext (VAL-SCALE-006).
    # Default remains short-ctx 128 for legacy emission continuity; operators raise
    # sequence_length (and pin.seq_len) to ≥256 / target 512 for scale-eval P1 cups.
    # max_sequence_length is the hard ceiling (default 512); sequence_length may not exceed it.
    max_sequence_length: int = 512
    sequence_length: int = 128
    # Eval token budget pass-through onto PrismContext.token_budget (None = unset / runner default).
    # Scale-eval P1 uses ≥1_000_000 under a matched ProtocolPin; emission path unchanged when None.
    token_budget: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("PRISM_TOKEN_BUDGET", "CHALLENGE_TOKEN_BUDGET"),
    )
    # Static build_model instantiation gate (architecture.md section 4.1): the param-count phase
    # instantiates build_model under the forced seed in a bounded child process before any GPU
    # work, so hostile construction is time/memory-bounded at the static phase.
    static_instantiation_timeout_seconds: float = 30.0
    static_instantiation_memory_headroom_bytes: int = 8_589_934_592
    fineweb_sample_count: int = 128
    execution_backend: str = "base_gpu"
    prism_role: Literal["challenge"] = "challenge"
    # Root stdlib log level applied by ``configure_logging`` on the deploy entrypoints (the uvicorn
    # ``prism_challenge.app:app`` API/combined process and the standalone ``prism-worker`` CLI).
    # Uvicorn configures only its own ``uvicorn.*`` loggers, so without this application INFO
    # (queue drain, worker iterations, persisted eval-log paths) propagates to an unconfigured root
    # and is swallowed. Default INFO; override with ``PRISM_LOG_LEVEL`` (e.g. DEBUG/WARNING).
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("PRISM_LOG_LEVEL", "CHALLENGE_LOG_LEVEL"),
    )
    public_submissions_enabled: bool = True
    worker_claim_timeout_seconds: int = 900
    # Combined mode (single-service deploy): when True the uvicorn API process ALSO runs the
    # evaluation worker loop as a background asyncio task, so ONE ``challenge-prism`` service both
    # serves the API and drains the eval queue. Enabled by the exact env var ``PRISM_COMBINED_MODE``
    # (env_prefix PRISM_ + field name). Default OFF preserves the separate ``prism-worker`` deploy
    # and every existing test. The worker only ORCHESTRATES GPU work via the broker
    # (docker_backend="broker"), so the combined service needs no local GPU; it does need the
    # broker URL + token env (PRISM_DOCKER_BROKER_URL + PRISM_DOCKER_BROKER_TOKEN[_FILE]) or
    # nothing drains.
    combined_mode: bool = False
    combined_worker_interval_seconds: float = Field(default=5.0, ge=0.0)
    l2_top_k: int = 200
    l3_top_k: int = 20
    kendall_tau_min: float = 0.4
    arch_weight: float = Field(default=0.7, ge=0, le=1)
    recipe_weight: float = Field(default=0.3, ge=0, le=1)
    component_rewards_enabled: bool = True
    # VAL-RESLAB-008: two-tier ownership defaults 0.50/0.50 (kill 0.65/0.35 drift).
    architecture_reward_weight: float = Field(default=0.50, ge=0, le=1)
    training_reward_weight: float = Field(default=0.50, ge=0, le=1)
    architecture_improvement_min_delta_abs: float = Field(default=0.01, ge=0)
    architecture_improvement_min_delta_rel: float = Field(default=0.005, ge=0)
    architecture_transfer_min_delta_abs: float = Field(default=0.08, ge=0)
    architecture_transfer_min_delta_rel: float = Field(default=0.05, ge=0)
    training_improvement_min_delta_abs: float = Field(default=0.02, ge=0)
    training_improvement_min_delta_rel: float = Field(default=0.005, ge=0)
    training_transfer_min_delta_abs: float = Field(default=0.05, ge=0)
    training_transfer_min_delta_rel: float = Field(default=0.03, ge=0)
    training_improvement_z_score: float = Field(default=1.0, ge=0)
    training_metric_default_std: float = Field(default=0.0, ge=0)
    component_eval_seed_count: int = Field(default=1, ge=1)
    component_eval_repeat_count: int = Field(default=1, ge=1)
    hf_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("PRISM_HF_TOKEN", "HF_TOKEN"),
    )
    hf_token_file: Path | None = Field(
        default=Path("/run/secrets/hf_token"),
        validation_alias=AliasChoices("PRISM_HF_TOKEN_FILE", "HF_TOKEN_FILE"),
    )
    checkpoint_cadence_seconds: int = Field(
        default=3600,
        ge=1,
        validation_alias=AliasChoices(
            "PRISM_CHECKPOINT_CADENCE_SECONDS", "PRISM_HF_CHECKPOINT_CADENCE_SECONDS"
        ),
    )
    checkpoint_repo_id: str = Field(
        default="baseintelligence/prism-checkpoints",
        validation_alias=AliasChoices("PRISM_CHECKPOINT_REPO_ID", "PRISM_HF_CHECKPOINT_REPO_ID"),
    )
    subnet_rules_json: str | None = None
    subnet_rules_file: Path | None = None
    plagiarism_enabled: bool = True
    plagiarism_min_similarity: float = 0.65
    plagiarism_static_reject_threshold: float = 0.96
    plagiarism_top_k: int = 2
    plagiarism_sandbox_enabled: bool = False
    plagiarism_sandbox_image: str = "python:3.12-alpine"
    plagiarism_sandbox_timeout_seconds: int = 30
    plagiarism_storage_max_files: int = 200
    plagiarism_storage_max_bytes: int = 2_000_000
    docker_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("PRISM_DOCKER_ENABLED", "CHALLENGE_DOCKER_ENABLED"),
    )
    docker_bin: str = Field(
        default="docker",
        validation_alias=AliasChoices("PRISM_DOCKER_BIN", "CHALLENGE_DOCKER_BIN"),
    )
    docker_backend: str = Field(
        default="broker",
        validation_alias=AliasChoices("PRISM_DOCKER_BACKEND", "CHALLENGE_DOCKER_BACKEND"),
    )
    docker_broker_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PRISM_DOCKER_BROKER_URL", "CHALLENGE_DOCKER_BROKER_URL"),
    )
    docker_broker_token: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("PRISM_DOCKER_BROKER_TOKEN", "CHALLENGE_DOCKER_BROKER_TOKEN"),
    )
    docker_broker_token_file: str | None = Field(
        # Mirror shared_token_file: production mounts the challenge/broker token under
        # /run/secrets/base/. Defaulting the path keeps broker-backend construction valid at
        # import/collection time without requiring live secrets in the packaging environment
        # (the path does not need to exist for ChallengeSettings executor validation).
        default="/run/secrets/base/challenge_token",
        repr=False,
        validation_alias=AliasChoices(
            "PRISM_DOCKER_BROKER_TOKEN_FILE", "CHALLENGE_DOCKER_BROKER_TOKEN_FILE"
        ),
    )
    docker_allowed_images: tuple[str, ...] = Field(
        default=("baseintelligence/", "ghcr.io/baseintelligence/"),
        validation_alias=AliasChoices(
            "PRISM_DOCKER_ALLOWED_IMAGES", "CHALLENGE_DOCKER_ALLOWED_IMAGES"
        ),
    )
    docker_network: str = Field(
        default="none",
        validation_alias=AliasChoices("PRISM_DOCKER_NETWORK", "CHALLENGE_DOCKER_NETWORK"),
    )
    docker_cpus: float = Field(
        default=1.0,
        validation_alias=AliasChoices("PRISM_DOCKER_CPUS", "CHALLENGE_DOCKER_CPUS"),
    )
    docker_memory: str = Field(
        default="512m",
        validation_alias=AliasChoices("PRISM_DOCKER_MEMORY", "CHALLENGE_DOCKER_MEMORY"),
    )
    docker_memory_swap: str | None = Field(
        default="512m",
        validation_alias=AliasChoices("PRISM_DOCKER_MEMORY_SWAP", "CHALLENGE_DOCKER_MEMORY_SWAP"),
    )
    docker_pids_limit: int = Field(
        default=128,
        validation_alias=AliasChoices("PRISM_DOCKER_PIDS_LIMIT", "CHALLENGE_DOCKER_PIDS_LIMIT"),
    )
    docker_read_only: bool = Field(
        default=True,
        validation_alias=AliasChoices("PRISM_DOCKER_READ_ONLY", "CHALLENGE_DOCKER_READ_ONLY"),
    )
    docker_user: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PRISM_DOCKER_USER", "CHALLENGE_DOCKER_USER"),
    )
    base_eval_image: str = "ghcr.io/baseintelligence/prism-evaluator:latest"
    # Wall-clock budget hardening (architecture.md sections 4.3, 9). The score is
    # compute-normalized (tokens/FLOPs), so wall-clock is ONLY a safety cap, not part of the
    # score. Three layers, smallest first:
    #   1. ``base_eval_budget_seconds`` (graceful, 10-30 min): the challenge runner stops the
    #      single-pass loop at this point and scores on the PARTIAL captured stream.
    #   2. ``base_eval_budget_seconds + base_eval_watchdog_grace_seconds`` (hard): a
    #      runner watchdog thread terminates a loop that hangs OUTSIDE the instrumented iterator
    #      (so a non-iterating hang is still bounded), landing the run failed with a budget reason.
    #   3. ``base_eval_timeout_seconds`` (outer docker/broker cap): the absolute backstop, set
    #      strictly above budget+grace so the runner gets a chance to stop gracefully first.
    base_eval_budget_seconds: int = 1200
    base_eval_watchdog_grace_seconds: int = 120
    # Bound on the only writable path (``ctx.artifacts_dir``): a runner watchdog fails the run if
    # the artifacts dir grows past this quota so an artifacts disk-fill cannot take down the host
    # (architecture.md section 9; VAL-HARNESS-026).
    base_eval_artifacts_quota_bytes: int = 2_147_483_648
    base_eval_timeout_seconds: int = 1800
    # Orchestration-level HARD wall-time cap (architecture.md sections 4.3, 9). The inner docker /
    # broker timeout (``base_eval_hard_timeout_seconds``) should normally fire first, but a hung
    # broker / un-cancellable worker thread could otherwise hold the single GPU forever; this is
    # the absolute backstop the worker enforces around ``evaluator.evaluate`` so an over-time eval
    # is KILLED (its container reaped) and its GPU lease RELEASED. ``0`` auto-derives it as
    # ``base_eval_hard_timeout_seconds + base_eval_orchestration_grace_seconds`` so it always sits
    # strictly above the inner cap; a positive value overrides it (used to force a tiny cap).
    base_eval_orchestration_timeout_seconds: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_ORCHESTRATION_TIMEOUT_SECONDS",
            "CHALLENGE_BASE_EVAL_ORCHESTRATION_TIMEOUT_SECONDS",
        ),
    )
    base_eval_orchestration_grace_seconds: int = Field(default=300, ge=0)
    base_eval_cpus: float = 2.0
    base_eval_memory: str = "8g"
    base_eval_memory_swap: str | None = "8g"
    base_eval_pids_limit: int = 512
    base_eval_read_only: bool = True
    base_eval_max_gpu_count: int = Field(default=8, ge=1, le=8)
    base_eval_gpu_count: int = 1
    # Per-eval GPU VRAM cap in MiB (architecture.md section 9). Docker has no native per-container
    # VRAM cgroup, so the cap is propagated to the container env (``PRISM_GPU_VRAM_CAP_MIB``) and
    # the challenge runner clamps the torch CUDA allocator via
    # ``torch.cuda.set_per_process_memory_fraction`` BEFORE any miner code runs, so an oversized
    # model (``max_code_bytes`` up to 7.5MB) cannot exhaust GPU memory and wedge the single worker.
    # ``0`` disables the cap (deploys set a concrete value with headroom in config.example.yaml).
    base_eval_gpu_vram_mib: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_GPU_VRAM_MIB", "CHALLENGE_BASE_EVAL_GPU_VRAM_MIB"
        ),
    )
    # Multi-GPU static contract policy (architecture.md section 8). Gate A statically verifies the
    # miner training.py uses the distributed primitives + a rank-0 write guard and rejects a
    # gpu_count > 8 / multi-node request before any GPU work. ``reject`` (default) hard-rejects a
    # non-distributed script; ``flag`` advances but logs; ``off`` skips the check.
    distributed_contract_policy: Literal["reject", "flag", "off"] = Field(
        default="reject",
        validation_alias=AliasChoices(
            "PRISM_DISTRIBUTED_CONTRACT_POLICY", "CHALLENGE_DISTRIBUTED_CONTRACT_POLICY"
        ),
    )
    base_eval_gpu_type: str | None = None
    base_gpu_targets: str | None = None
    base_eval_gpu_server: str | None = None
    base_eval_gpu_device_ids: tuple[str, ...] = ()
    base_eval_task: str = "architecture"
    # Prefer the composepersistent `/data/tmp` volume when writable (locked-down
    # container `/tmp` breaks hardcoding `/tmp/prism-eval-artifacts` as uid 1000).
    # On bare hosts / CI without `/data`, fall back to the process temp dir so
    # unit tests and local pytest never PermissionError on mkdir.
    base_eval_artifact_root: Path = Field(
        default_factory=lambda: _default_base_eval_artifact_root(),
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_ARTIFACT_ROOT",
            "CHALLENGE_BASE_EVAL_ARTIFACT_ROOT",
        ),
    )
    # Persistent dir for the COMPLETE evaluated-agent (GPU training-run) stdout/stderr, written per
    # attempt so the full run stream survives the destroyed eval container (the broker returns up
    # to ~5MB per stream, which prism otherwise only parses for metrics / failure detail and then
    # discards). Defaults (see ``resolved_eval_log_dir``) to an ``eval-logs`` subdir on the SAME
    # persistent ``/data`` volume as the sqlite DB; override with ``PRISM_EVAL_LOG_DIR``.
    eval_log_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("PRISM_EVAL_LOG_DIR", "CHALLENGE_EVAL_LOG_DIR"),
    )
    # Read-only locked FineWeb-Edu train split mount (architecture.md section 3). The broker
    # bind-mounts the staged train shards here (RO); the challenge runner resolves ctx.data_dir to
    # this path and fails fast when it is missing/empty (no random-token fallback).
    base_eval_data_dir: str = Field(
        default="/data/fineweb-edu/train",
        validation_alias=AliasChoices("PRISM_BASE_EVAL_DATA_DIR", "PRISM_EVAL_DATA_DIR"),
    )
    # Secret held-out val split (architecture.md sections 5, 6). It is NEVER bind-mounted into the
    # eval container (VAL-HARNESS-015 / VAL-CHEAT-007) and never exposed via PrismContext; only the
    # CHALLENGE SCORER reads it (host-side) to compute the held-out delta-over-random-init
    # tie-breaker and the train-vs-held-out anti-memorization gap. An unset/empty path simply
    # skips the held-out delta (the run still scores on prequential bpb).
    base_eval_val_data_dir: str = Field(
        default="/data/fineweb-edu/val",
        validation_alias=AliasChoices("PRISM_BASE_EVAL_VAL_DATA_DIR", "PRISM_EVAL_VAL_DATA_DIR"),
    )
    # Host-readable train split for the anti-memorization gap (architecture.md section 6.2). The
    # held-out scorer re-evaluates the trained model byte-level over a fixed prefix of the EXPOSED
    # train split to obtain the CONVERGED (final-checkpoint) train bpb, used as the train side of
    # the train-vs-held-out gap. Measuring the gap against the converged model (not the
    # curve-averaged prequential AUC, which is inflated by early high-loss steps and shrinks the
    # gap) reliably flags a genuine memorizer while leaving a benign learner unflagged. The train
    # split is NOT secret, so the deploy may mount it into the scorer container; when this path is
    # unset/unavailable the gap gracefully falls back to the (basis-gated) prequential reference.
    base_eval_train_data_dir: str = Field(
        default="/data/fineweb-edu/train",
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_TRAIN_DATA_DIR", "PRISM_EVAL_TRAIN_DATA_DIR"
        ),
    )
    # Host-side held-out compute budget (architecture.md sections 4, 5; m4-heldout-live-budget-
    # tuning). The held-out delta + anti-memorization gap are computed on the worker host (CPU)
    # AFTER the container eval, evaluating a random-init twin + the trained model over the SECRET
    # val split. The full single-threaded eval overruns a tight timeout, so the scorer caps the
    # held-out eval to a FIXED, DETERMINISTIC val byte budget (a stable prefix, identical for both
    # models so the delta stays comparable) and uses a raised, configurable timeout. The byte
    # denominator keeps the delta tokenizer-agnostic; the fixed prefix keeps it deterministic. A
    # byte budget <= 0 scores the entire val split.
    base_eval_heldout_val_byte_budget: int = Field(
        default=65536,
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_HELDOUT_VAL_BYTE_BUDGET", "PRISM_EVAL_HELDOUT_VAL_BYTE_BUDGET"
        ),
    )
    base_eval_heldout_timeout_seconds: float = Field(
        default=600.0,
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_HELDOUT_TIMEOUT_SECONDS", "PRISM_EVAL_HELDOUT_TIMEOUT_SECONDS"
        ),
    )
    base_eval_reference_tokenizer_dir: str = Field(
        default="/opt/reference-tokenizers",
        validation_alias=AliasChoices(
            "PRISM_BASE_EVAL_REFERENCE_TOKENIZER_DIR", "PRISM_REFERENCE_TOKENIZER_DIR"
        ),
    )
    validator_hotkeys: tuple[str, ...] = ()
    validator_assignment_timeout_seconds: int = 900
    validator_assignment_max_attempts: int = 3

    def internal_token(self) -> str:
        if self.shared_token:
            return self.shared_token
        if self.shared_token_file and Path(self.shared_token_file).exists():
            return Path(self.shared_token_file).read_text(encoding="utf-8").strip()
        raise RuntimeError("PRISM_SHARED_TOKEN or PRISM_SHARED_TOKEN_FILE is required")

    @property
    def base_eval_hard_timeout_seconds(self) -> int:
        """Outer docker/broker timeout, forced strictly above the graceful budget + watchdog grace.

        The runner's graceful budget and hard watchdog must both fire BEFORE this absolute backstop
        so an over-budget loop is stopped gracefully (or failed with a budget reason) rather than
        bluntly killed by the broker; a slack margin gives the runner time to author its manifest.
        """
        floor = self.base_eval_budget_seconds + self.base_eval_watchdog_grace_seconds + 60
        return max(self.base_eval_timeout_seconds, floor)

    @property
    def resolved_orchestration_timeout_seconds(self) -> float:
        if self.base_eval_orchestration_timeout_seconds > 0:
            return self.base_eval_orchestration_timeout_seconds
        return float(
            self.base_eval_hard_timeout_seconds + self.base_eval_orchestration_grace_seconds
        )

    @property
    def resolved_database_path(self) -> Path:
        if self.database_url.startswith("sqlite+aiosqlite:///"):
            return Path(self.database_url.removeprefix("sqlite+aiosqlite:///"))
        return self.database_path

    @property
    def resolved_eval_log_dir(self) -> Path:
        """Persistent dir for full evaluated-agent stdout/stderr (same volume as the sqlite DB).

        Defaults to an ``eval-logs`` subdir next to the DB on the persistent ``/data`` volume so the
        complete GPU training-run stream outlives the (destroyed) eval container; overridable via
        ``PRISM_EVAL_LOG_DIR``.
        """
        if self.eval_log_dir is not None:
            return self.eval_log_dir
        return self.resolved_database_path.parent / "eval-logs"

    def hf_token_value(self) -> str | None:
        if self.hf_token:
            return self.hf_token
        if self.hf_token_file and self.hf_token_file.exists():
            token = self.hf_token_file.read_text(encoding="utf-8").strip()
            return token or None
        return None

    def max_parameters_for_ladder_stage(
        self, stage: Literal["explore", "promote"] | str | None = None
    ) -> int:
        """Hard param cap for a dual-ladder stage (VAL-RESLAB-003).

        Defaults to the configured ``param_ladder_stage``. Explore →
        ``explore_max_parameters`` (124M); promote → ``promote_max_parameters`` (350M).
        """
        resolved = (stage or self.param_ladder_stage or "explore").strip().lower()
        if resolved == "promote":
            return int(self.promote_max_parameters)
        if resolved == "explore":
            return int(self.explore_max_parameters)
        raise ValueError(f"unknown param ladder stage {stage!r}; expected 'explore' or 'promote'")

    def resolve_admission_max_parameters(
        self, stage: Literal["explore", "promote"] | str | None = None
    ) -> tuple[str, int]:
        """Return ``(stage, cap)`` for emission admission under the dual ladder.

        When ``stage`` is omitted, uses settings ``param_ladder_stage`` and the
        corresponding stage cap. An explicit numeric ``max_parameters`` override on
        settings remains available for lab harnesses but is **not** the dual-ladder
        default path (legacy 150M-only default is gone).
        """
        resolved = (stage or self.param_ladder_stage or "explore").strip().lower()
        if resolved not in ("explore", "promote"):
            raise ValueError(
                f"unknown param ladder stage {stage!r}; expected 'explore' or 'promote'"
            )
        return resolved, self.max_parameters_for_ladder_stage(resolved)

    def resolved_eval_sequence_length(self) -> int:
        """Worker-plane sequence length with max_sequence_length ceiling (no 128-only trap)."""
        seq = int(self.sequence_length)
        cap = int(self.max_sequence_length)
        if seq <= 0:
            raise ValueError(f"sequence_length must be positive; got {seq}")
        if cap <= 0:
            raise ValueError(f"max_sequence_length must be positive; got {cap}")
        if seq > cap:
            raise ValueError(
                f"sequence_length={seq} exceeds max_sequence_length={cap}; "
                "raise max_sequence_length or lower sequence_length"
            )
        return seq

    def resolved_eval_token_budget(self) -> int | None:
        """Optional eval token_budget pass-through (None keeps runner/pin defaults)."""
        if self.token_budget is None:
            return None
        budget = int(self.token_budget)
        if budget <= 0:
            raise ValueError(f"token_budget must be positive; got {budget}")
        return budget

    def prism_context_kwargs(self) -> dict[str, Any]:
        """Kwargs for :class:`PrismContext` from settings (seq + token_budget pass-through).

        Does not invent a default token_budget when unset (emission path stays None).
        Callers building scale-eval pins should set ProtocolPin.token_budget and/or
        settings.token_budget together under a matched pin (VAL-SCALE-006).
        """
        kwargs: dict[str, Any] = {
            "sequence_length": self.resolved_eval_sequence_length(),
            "max_layers": int(self.max_layers),
            "max_parameters": int(self.max_parameters),
            "param_ladder_stage": self.param_ladder_stage,
        }
        budget = self.resolved_eval_token_budget()
        if budget is not None:
            kwargs["token_budget"] = budget
        return kwargs

    def model_post_init(self, __context: Any) -> None:  # noqa: ARG002
        # Validate seq vs max at construct time so misconfigured P1 knobs fail closed early.
        # token_budget None is valid (unset); positive when set is enforced by Field(ge=1).
        _ = self.resolved_eval_sequence_length()


_settings: PrismSettings | None = None


def get_settings() -> PrismSettings:
    """Return process-wide PrismSettings, constructing them on first use.

    Production settings require a broker token/path when ``docker_backend`` is
    ``broker`` (the safe default). Eager ``settings = PrismSettings()`` at module
    import breaks pytest collection and packaging imports that never need production
    credentials. Deploy entrypoints and ``create_app()`` still materialize full
    settings via this helper once environment secrets are present.
    """
    global _settings
    if _settings is None:
        _settings = PrismSettings()
    return _settings


def __getattr__(name: str) -> Any:
    # Preserve ``from prism_challenge.config import settings`` and attribute access
    # without instantiating production settings during package import.
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def configure_logging(app_settings: PrismSettings | None = None) -> None:
    """Configure stdlib root logging at the settings-driven level (default INFO).

    Both deploy entrypoints -- the uvicorn ``prism_challenge.app:app`` API/combined process and the
    standalone ``prism-worker`` CLI -- otherwise run with NO root logging config (uvicorn only sets
    up its own ``uvicorn.*`` loggers), so application INFO propagating to the root logger is
    swallowed by the WARNING-level last-resort handler. Adding a root handler at ``log_level`` makes
    that INFO visible under uvicorn. ``basicConfig`` (no ``force``) is a no-op when the root logger
    already has handlers, so this never displaces uvicorn's handlers nor a test harness's capture
    handlers; it only installs one when the deploy entrypoint has none.
    """
    resolved = app_settings if app_settings is not None else get_settings()
    level = logging.getLevelNamesMapping().get(resolved.log_level.strip().upper(), logging.INFO)
    logging.basicConfig(level=level)
