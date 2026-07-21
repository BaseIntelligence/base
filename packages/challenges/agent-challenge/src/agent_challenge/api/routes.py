"""Public challenge routes proxied by the BASE master."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import zipfile
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, desc, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.responses import JSONResponse, StreamingResponse

from ..analyzer.lifecycle import (
    _artifact_metadata,
    _source_artifact,
    queue_submission_analysis,
)
from ..auth.security import (
    SignedRequestAuth,
    build_owner_signed_auth_dependency,
    build_signed_auth_dependency,
)
from ..canonical.eval_wire import decode_score_f64be
from ..core.config import settings
from ..core.db import database
from ..core.models import (
    AdminReviewDecision,
    AgentSubmission,
    AnalysisRun,
    EvalRun,
    EvaluationAttempt,
    EvaluationJob,
    LlmVerdict,
    OwnerActionAudit,
    PythonAstFeature,
    ReviewAssignment,
    ReviewSession,
    SimilarityMatch,
    SubmissionArtifact,
    SubmissionEnvEncryptionError,
    SubmissionEnvVar,
    SubmissionFamily,
    SubmissionStatusEvent,
    TaskLogEvent,
    TaskResult,
    TerminalBenchTrial,
)
from ..evaluation.authorization import (
    EvalAuthorizationConflict,
    EvalAuthorizationRequired,
    EvalAuthorizationUnavailable,
    cancel_eval_run,
    create_eval_run,
    eval_status_page,
    fail_eval_run,
    load_eval_run_plan,
    retry_eval_run,
)
from ..evaluation.benchmarks import load_benchmark_tasks
from ..evaluation.direct_result import (
    DirectEvalResultError,
    authenticate_eval_token,
    process_direct_eval_result,
    validate_result_bounds,
)
from ..evaluation.replay_audit import (
    REPLAY_AUDIT_LABEL,
    AggregationSpec,
    InvalidReplayTrialsError,
    ReplayAuditWireError,
    accepted_verified_replay_population,
    compare_replay_trials,
    persist_replay_dispute,
    replay_audit_sampler_from_settings,
    replay_request_for_candidate,
    replay_result_from_mapping,
)
from ..evaluation.runner import (
    EvaluationAuthorizationError,
    create_evaluation_job,
    enqueue_evaluation_job_for_submission,
    existing_evaluation_job_for_submission,
    lock_miner_env_for_evaluation,
)
from ..evaluation.task_events import (
    apply_miner_env_redaction,
    record_task_event,
    redact_secrets,
    redact_task_event_message,
)
from ..evaluation.terminal_bench import TERMINAL_BENCH_EVALUATOR
from ..evaluation.validator_executor import (
    finalize_job_if_complete,
    fold_terminally_failed_work_unit,
)
from ..evaluation.weights import is_scoring_submission, scoring_evaluation_jobs_statement
from ..evaluation.work_units import list_pending_work_units
from ..review.artifacts import ReviewArtifactError, load_assignment_artifact
from ..review.canonical import canonical_json_v1, parse_json_object
from ..review.deployment import ReviewDeploymentError, review_input_config_from_settings
from ..review.evidence import ReviewEvidenceError, load_review_evidence_object
from ..review.public_tee import (
    build_public_tee_math_from_assignment,
    public_tee_assignment_qualifies,
    public_tee_unavailable,
)
from ..review.report import (
    DcapReviewQuoteVerifier,
    ReviewMeasurementAllowlist,
    ReviewReportConflict,
    ReviewReportError,
    submit_review_report,
    validate_review_envelope,
)
from ..review.rules import RulesSnapshotCaptureError, capture_rules_bundle
from ..review.schemas import (
    rules_bundle_files,
    validate_model_call_started,
    validate_review_infrastructure_failure,
)
from ..review.sessions import (
    ReviewCapabilityError,
    ReviewConflict,
    ReviewNotFound,
    ReviewRateLimited,
    assignment_artifact,
    assignment_rules,
    authenticate_assignment_capability,
    cancel_review_assignment,
    create_review_session,
    deliver_prepare_token,
    enforce_review_session_mutation_budget,
    expire_assignment_if_needed,
    issue_operator_approval,
    mark_model_call_started,
    mark_review_deployed,
    record_review_submission_status,
    retry_review_assignment,
    review_audit_page,
)
from ..review.sessions import (
    record_review_infrastructure_failure as record_review_infrastructure_failure_state,
)
from ..sdk.auth import (
    build_attempt_stream_auth_dependency,
    build_internal_auth_dependency,
)
from ..sdk.decorators import public_route
from ..sdk.write_retry import run_write_with_lock_retry
from ..submissions.artifacts import (
    MAX_UNCOMPRESSED_BYTES,
    ArtifactMetadata,
    ArtifactReadError,
    ArtifactReadSession,
    ArtifactValidationError,
    ZipManifestEntry,
    store_base64_zip,
    store_zip_bytes,
    store_zip_uri,
)
from ..submissions.miner_env import (
    MinerEnvValidationError,
    validate_miner_env,
)
from ..submissions.rate_limit import (
    RateLimitExceeded,
    consume_submission_rate_limit,
    reserve_submission_rate_limit,
    submission_rate_limit_message,
)
from ..submissions.state_machine import (
    ensure_submission_status,
    public_status_for,
    record_initial_status,
)
from ..submissions.versioning import normalize_submission_name, version_label

router = APIRouter()
DatabaseSession = Annotated[AsyncSession, Depends(database.session_dependency)]
signed_submission_auth = build_signed_auth_dependency(settings)
owner_signed_auth = build_owner_signed_auth_dependency(settings)
internal_bridge_auth = build_internal_auth_dependency(settings)
attempt_stream_auth = build_attempt_stream_auth_dependency(settings)
SignedSubmissionAuth = Annotated[SignedRequestAuth, Depends(signed_submission_auth)]
OwnerSignedAuth = Annotated[SignedRequestAuth, Depends(owner_signed_auth)]
InternalBridgeAuth = Annotated[None, Depends(internal_bridge_auth)]
AttemptStreamAuth = Annotated[None, Depends(attempt_stream_auth)]


async def _read_bounded_result_body(http_request: Request, *, max_bytes: int) -> bytes:
    """Auth-first stream reader for the direct Eval result.

    Content-Length is advisory and may lie; the stream is capped so chunked or
    misleading-length bodies cannot force unbounded pre-verification buffering.
    Callers must authorize the run token before invoking this helper.
    """

    content_length = http_request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={"code": "result_too_large"},
            ) from exc
        if declared_length < 0 or declared_length > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={"code": "result_too_large"},
            )
    chunks: list[bytes] = []
    total = 0
    async for chunk in http_request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={"code": "result_too_large"},
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _public_cache(response: Response, *, max_age: int, swr: int) -> None:
    """Advertise a public GET read as browser/CDN cacheable.

    ``s-maxage`` mirrors ``max-age`` so shared caches (the Cloudflare/Vercel edge
    in front of the flaky origin tunnel) and browsers cache identically, while
    ``stale-while-revalidate`` lets a cache keep serving a slightly stale body
    during an async refresh. Only ever call this on public, non-mutating reads.
    """

    response.headers["Cache-Control"] = (
        f"public, max-age={max_age}, s-maxage={max_age}, stale-while-revalidate={swr}"
    )


#: Hard caps for the real-time log ingest route (record_task_event additionally
#: redacts + enforces per-event / per-task / per-submission byte budgets).
MAX_STREAM_EVENTS_BYTES = 4 * 1024 * 1024
MAX_STREAM_EVENTS_PER_REQUEST = 512
STREAM_LOG_CHANNELS = frozenset({"agent", "harness", "test_stdout", "test_stderr"})
SSE_HEARTBEAT_SECONDS = 15.0
SSE_POLL_SECONDS = 1.0
DEFAULT_TASK_EVENT_REPLAY_LIMIT = 100
MAX_TASK_EVENT_REPLAY_LIMIT = 200
#: Public source endpoint caps: per-file decompressed content cap (bodies beyond
#: this are trimmed) and an overall decompressed payload cap (once reached, no
#: further files are added). Both bound the redacted text placed in the response;
#: the stored zip stays bounded by the artifact ``zip_max_bytes`` guard.
PUBLIC_SOURCE_MAX_FILE_BYTES = 256 * 1024
PUBLIC_SOURCE_MAX_TOTAL_BYTES = 5 * 1024 * 1024
TASK_EVENT_TERMINAL_TYPES = frozenset({"task.completed", "task.failed", "submission.completed"})
PRIVATE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:tmp|var|home|root|workspace|droid)/[^\s,;:'\"<>]+"
)
SENSITIVE_METADATA_KEYS = frozenset(
    {
        "artifact_path",
        "artifact_uri",
        "broker_ref",
        "canonical_artifact_hash",
        "env",
        "environment",
        "harbor_forward_env_vars",
        "family_id",
        "job_dir",
        "lease_owner",
        "execution_provider",
        "job_name",
        "kubernetes_job_name",
        "logs_ref",
        "normalized_name",
        "pod_name",
        "private_path",
        "provider",
        "raw_artifacts_json",
        "raw_ref",
        "signature",
        "signature_message",
        "signature_nonce",
        "signature_payload_sha256",
        "stderr_ref",
        "stdout_ref",
        "token",
        "worker",
        "worker_name",
    }
)
PUBLIC_TASK_PHASE_STATUSES = frozenset({"assigned", "starting", "running", "completed", "failed"})
PUBLIC_SSE_REASON_CODES = frozenset(
    {
        "submission_received",
        "submission_upload_verified",
        "submission_rate_limit_reserved",
        "blocking_analysis_queued",
        "blocking_analysis_claimed",
        "blocking_analysis_ast_completed",
        "blocking_analysis_allowed",
        "waiting_miner_env",
        "blocking_analysis_rejected",
        "blocking_analysis_escalated",
        "blocking_analysis_admin_review_required",
        "blocking_analysis_lease_expired",
        "admin_review_allowed",
        "admin_review_rejected",
        "admin_review_rerun_requested",
        "evaluation_job_queued",
        "evaluation_job_claimed",
        "evaluation_job_running",
        "evaluation_job_completed",
        "evaluation_job_failed",
        "evaluation_failed_before_verdict",
        "evaluation_retry_cap_reached",
        "evaluation_retry_queued",
        "analysis_verdict_recorded",
        "eval_expired",
        "eval_no_result",
        "verifier_unavailable",
        "persistence_unavailable",
        "attestation_verification_failed",
    }
)


class SubmissionRequest(BaseModel):
    """Miner agent submission payload.

    closed schema (extra=forbid): client-supplied ``issued_at`` / ``received_at``
    / freshness booleans cannot override challenge attestation-bound times
    (VAL-ACAT-038).
    """

    model_config = ConfigDict(extra="forbid")

    miner_hotkey: str | None = Field(default=None, min_length=1, max_length=128)
    name: str = Field(default="agent", min_length=1, max_length=128)
    artifact_uri: str | None = Field(default=None, min_length=1)
    artifact_zip_base64: str | None = Field(default=None, min_length=1, repr=False)
    agent_hash: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )


class BaseBridgeHeaders(BaseModel):
    hotkey: str
    nonce: str
    request_hash: str
    filename: str | None = None


class EvaluationSummaryResponse(BaseModel):
    job_id: str
    status: str
    score: float
    passed_tasks: int
    total_tasks: int
    verdict: str | None
    rules_version: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class SubmissionResponse(BaseModel):
    """Submission response returned to the caller."""

    submission_id: int
    name: str
    display_name: str | None
    agent_hash: str
    zip_sha256: str
    family_id: str
    version_number: int
    version_label: str
    version_count: int
    is_latest_version: bool
    latest_submission_id: int
    status: str
    effective_status: str
    submitted_at: datetime
    created_at: datetime
    latest_evaluation: EvaluationSummaryResponse | None


class SubmissionListItem(BaseModel):
    """Submission list item."""

    id: int
    miner_hotkey: str
    name: str
    display_name: str | None
    agent_hash: str
    zip_sha256: str | None
    family_id: str | None
    version_number: int | None
    version_label: str | None
    version_count: int | None
    is_latest_version: bool
    latest_submission_id: int | None
    status: str
    effective_status: str
    env_action_required: bool
    env_keys: list[str]
    env_var_count: int
    env_confirmed_empty: bool
    env_locked: bool
    env_updated_at: datetime | None
    score: float
    submitted_at: datetime
    created_at: datetime
    latest_evaluation: EvaluationSummaryResponse | None
    has_analysis: bool
    analyzer_status: str | None
    analyzer_verdict: str | None
    llm_verdict: str | None
    llm_confidence: float | None
    similarity_max_score_percent: float | None
    similarity_match_count: int
    ast_feature_count: int


class TaskResultResponse(BaseModel):
    """Public task result details."""

    task_id: str
    docker_image: str
    status: str
    score: float
    returncode: int
    duration_seconds: float
    failure_reason: str | None = None
    detail_log: str | None = None


class TaskPhaseResponse(BaseModel):
    task_id: str
    phase: str
    status: str
    updated_at: datetime
    attempt: int | None


class TaskRowResponse(BaseModel):
    task_id: str
    display_name: str
    source: str
    phase: str
    status: str
    updated_at: datetime | None
    attempt: int | None
    has_result: bool = False


class EvaluationResponse(BaseModel):
    """Evaluation progress and score."""

    job_id: str
    submission_id: int
    name: str
    agent_hash: str
    zip_sha256: str | None
    display_name: str | None
    family_id: str | None
    version_number: int | None
    version_label: str | None
    version_count: int | None
    is_latest_version: bool
    latest_submission_id: int | None
    status: str
    effective_status: str
    score: float
    passed_tasks: int
    total_tasks: int
    verdict: str | None
    rules_version: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    tasks: list[TaskResultResponse]
    task_phases: list[TaskPhaseResponse] = Field(default_factory=list)
    task_rows: list[TaskRowResponse] = Field(default_factory=list)


class AgentSourceFile(BaseModel):
    """One file from an agent submission's source zip (secrets redacted)."""

    path: str
    size_bytes: int
    content: str | None
    truncated: bool
    redacted: bool
    binary: bool


class AgentSourceResponse(BaseModel):
    """Redacted source listing for one agent submission."""

    agent_hash: str
    submission_id: int
    agent_name: str | None
    miner_hotkey: str
    available: bool
    total_files: int
    total_bytes: int
    truncated: bool
    files: list[AgentSourceFile]


class AnalyzerStatusResponse(BaseModel):
    phase: str
    status: str | None
    verdict: str | None
    reason_codes: list[str]
    llm_verdict: str | None
    llm_confidence: float | None
    llm_reason_codes: list[str]
    llm_rationale: str | None
    started_at: datetime | None
    finished_at: datetime | None


class SimilarityFilePairResponse(BaseModel):
    source_file_path: str | None
    matched_file_path: str | None
    score_percent: float | None


class SimilarityMatchSummaryResponse(BaseModel):
    matched_submission_id: int | None
    match_kind: str
    score_percent: float
    risk_band: str | None
    algorithm_version: str | None
    top_file_pairs: list[SimilarityFilePairResponse]


class SimilarityStatusResponse(BaseModel):
    max_score_percent: float | None
    match_count: int
    top_matches: list[SimilarityMatchSummaryResponse]


class EvaluationStatusResponse(BaseModel):
    job_id: str | None
    status: str | None
    score: float
    passed_tasks: int
    total_tasks: int
    verdict: str | None
    reason_codes: list[str]
    current_attempt: int | None
    attempt_status: str | None
    task_phases: list[TaskPhaseResponse] = Field(default_factory=list)
    task_rows: list[TaskRowResponse] = Field(default_factory=list)


class ReviewStatusResponse(BaseModel):
    """Deterministic public projection of the current review assignment."""

    session_id: str | None
    assignment_id: str | None
    attempt: int | None
    phase: str | None
    terminal: bool
    verdict: Literal["allow", "reject", "escalate"] | None
    verified: bool
    retryable: bool
    reason_code: str | None
    report_available: bool
    issued_at: datetime | None
    finished_at: datetime | None


class PublicTeeMeasurementResponse(BaseModel):
    """Safe measurement registers for independent public TEE inspection."""

    model_config = ConfigDict(extra="forbid")

    mrtd: str | None = None
    rtmr0: str | None = None
    rtmr1: str | None = None
    rtmr2: str | None = None
    rtmr3: str | None = None
    compose_hash: str | None = None
    os_image_hash: str | None = None
    key_provider: str | None = None
    vm_shape: str | None = None


class PublicTeeVerificationOutcomeResponse(BaseModel):
    """Public subset of the durable review verification outcome."""

    model_config = ConfigDict(extra="forbid")

    status: str | None = None
    measurement_allowlisted: bool | None = None
    report_data_matched: bool | None = None
    verified_at_ms: int | None = None
    reason_code: str | None = None


class PublicTeeReportDataPreimageResponse(BaseModel):
    """Inspectable report_data preimage without raw review_nonce."""

    model_config = ConfigDict(extra="forbid")

    domain: str | None = None
    schema_version: int | None = None
    review_digest: str | None = None
    session_id: str | None = None
    issued_at_ms: int | None = None
    received_at_ms: int | None = None
    review_nonce_sha256: str | None = None


class PublicTeeMathResponse(BaseModel):
    """Public unauthenticated TEE math surface (architecture residual §5.1).

    When no authorizing/current verified report exists the body is exactly
    ``{"available": false}``. When available, only the safe math subset is
    present — never nonce plaintext, tokens, capabilities, evidence bodies,
    model IO, or KEY material.
    """

    model_config = ConfigDict(extra="forbid")

    available: bool
    submission_id: int | str | None = None
    domain: str | None = None
    review_digest: str | None = None
    report_data_hex: str | None = None
    report_data_preimage: PublicTeeReportDataPreimageResponse | None = None
    measurement: PublicTeeMeasurementResponse | None = None
    tdx_quote_hex: str | None = None
    event_log: list[dict[str, Any]] | None = None
    verification_outcome: PublicTeeVerificationOutcomeResponse | None = None
    quote_fingerprint_sha256: str | None = None
    agent_hash: str | None = None
    zip_sha256: str | None = None
    verdict: str | None = None
    assignment_digest: str | None = None
    session_id: str | None = None
    assignment_id: str | None = None


class TerminalBenchStatusResponse(BaseModel):
    total_trials: int
    completed_trials: int
    failed_trials: int
    errored_trials: int
    final_trials: int


class AstStatusResponse(BaseModel):
    feature_count: int
    feature_types: dict[str, int]
    verdict: str | None = None
    verdict_reason: str | None = None


class RuleEvidenceResponse(BaseModel):
    path: str
    line_start: int
    line_end: int
    snippet: str
    reason_code: str
    description: str


class RuleResultResponse(BaseModel):
    rule_id: str
    title: str
    status: str
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[RuleEvidenceResponse] = Field(default_factory=list)


class RulesCheckResponse(BaseModel):
    verdict: str | None = None
    recommended_status: str | None = None
    rules_version: str | None = None
    reviewer_used: bool | None = None
    reason_codes: list[str] = Field(default_factory=list)
    rules: list[RuleResultResponse] = Field(default_factory=list)
    notes: str | None = None


class SubmissionProgressCountsResponse(BaseModel):
    status_events: int
    analysis_runs: int
    similarity_matches: int
    llm_verdicts: int
    evaluation_jobs: int
    evaluation_attempts: int
    terminal_bench_trials: int


class SubmissionStatusResponse(BaseModel):
    submission_id: int
    name: str
    agent_hash: str
    display_name: str | None
    family_id: str | None
    version_number: int | None
    version_label: str | None
    version_count: int | None
    is_latest_version: bool
    latest_submission_id: int | None
    status: str
    public_state: str
    phase: str
    effective_status: str
    env_action_required: bool
    env_keys: list[str]
    env_var_count: int
    env_confirmed_empty: bool
    env_locked: bool
    env_updated_at: datetime | None
    last_event_id: int | None
    last_event_sequence: int | None
    current_attempt: int | None
    analyzer: AnalyzerStatusResponse
    similarity: SimilarityStatusResponse
    ast: AstStatusResponse
    rules_check: RulesCheckResponse | None = None
    # Fully legacy mode intentionally omits this field so flag-off status
    # bytes stay identical to the pre-review response shape.
    review: ReviewStatusResponse | None = None
    evaluation: EvaluationStatusResponse
    terminal_bench: TerminalBenchStatusResponse
    progress: SubmissionProgressCountsResponse
    submitted_at: datetime
    updated_at: datetime | None


class LeaderboardEntry(BaseModel):
    """Leaderboard row."""

    miner_hotkey: str
    submission_id: int
    name: str
    agent_hash: str
    display_name: str | None
    family_id: str | None
    version_number: int | None
    version_label: str | None
    version_count: int | None
    is_latest_version: bool
    latest_submission_id: int | None
    score: float
    passed_tasks: int
    total_tasks: int


class SubmissionVersionItem(BaseModel):
    id: int
    name: str
    agent_hash: str
    zip_sha256: str | None
    display_name: str | None
    family_id: str | None
    version_number: int | None
    version_label: str | None
    version_count: int | None
    is_latest_version: bool
    latest_submission_id: int | None
    status: str
    effective_status: str
    score: float
    submitted_at: datetime
    created_at: datetime
    latest_evaluation: EvaluationSummaryResponse | None


class TaskEventReplayItem(BaseModel):
    id: int
    sequence: int
    submission_id: int
    job_id: str | None
    task_id: str | None
    event_type: str
    stream: str | None
    message: str
    progress: float | None
    status: str | None
    truncated: bool
    cap_reached: bool
    metadata: dict[str, object]
    created_at: datetime


class TaskEventReplayResponse(BaseModel):
    submission_id: int
    name: str
    agent_hash: str
    display_name: str | None
    family_id: str | None
    version_number: int | None
    version_label: str | None
    version_count: int | None
    is_latest_version: bool
    latest_submission_id: int | None
    cursor: int
    next_cursor: int
    limit: int
    has_more: bool
    events: list[TaskEventReplayItem]


class SubmissionCountResponse(BaseModel):
    """Aggregate submission count."""

    count: int


class BenchmarkInfoResponse(BaseModel):
    """Configured benchmark dataset metadata."""

    backend: str
    dataset: str
    task_count: int
    evaluation_concurrency: int


class BenchmarkTaskResponse(BaseModel):
    """Benchmark task visible to miners and operators."""

    task_id: str
    benchmark: str
    docker_image: str
    prompt: str


class WorkUnitResponse(BaseModel):
    """One pending task work unit exposed to the master coordination plane."""

    work_unit_id: str
    submission_id: int
    submission_ref: str
    miner_hotkey: str
    job_id: str
    task_id: str
    docker_image: str
    required_capability: str


class WorkUnitsResponse(BaseModel):
    """The challenge's currently-assignable pending work units."""

    challenge_slug: str
    work_units: list[WorkUnitResponse]


class ReplayAuditRequestResponse(BaseModel):
    """One explicitly labelled replay request for the BASE audit seam."""

    schema_version: int
    audit_label: Literal["agent-challenge.replay-audit.v1"]
    kind: Literal["replay_audit_request"]
    audit_id: str
    submission_id: str
    eval_run_id: str
    replay_attempt: int
    plan_sha256: str
    eval_plan: dict[str, Any]
    k: int
    selected_tasks: list[dict[str, Any]]
    scoring_policy: dict[str, Any]
    scoring_policy_digest: str
    attested_score: float


class ReplayAuditRequestListResponse(BaseModel):
    """Sampled labelled replay requests available to BASE."""

    requests: list[ReplayAuditRequestResponse] = Field(default_factory=list)


class ReplayAuditResultResponse(BaseModel):
    """Outcome of one labelled replay result submission."""

    schema_version: int = 1
    audit_label: Literal["agent-challenge.replay-audit.v1"] = REPLAY_AUDIT_LABEL
    kind: Literal["replay_audit_result"] = "replay_audit_result"
    audit_id: str
    submission_id: str
    eval_run_id: str
    replay_attempt: int
    status: Literal["matched", "mismatch"]
    attested_score: float
    replay_score: float
    delta: float
    dispute_id: int | None = None


class FoldWorkUnitRequest(BaseModel):
    """Master request to fold a permanently-failed (max_attempts) work unit."""

    job_id: str
    task_id: str
    reason: str | None = None


class FoldWorkUnitResponse(BaseModel):
    """Outcome of folding a permanently-failed work unit into its job."""

    work_unit_id: str
    job_id: str
    task_id: str
    status: str
    score: float
    posted: bool
    finalized: bool


class MinerEnvUpdateRequest(BaseModel):
    """Miner-owned submission environment replacement payload."""

    env: dict[str, Any] = Field(default_factory=dict)


class MinerEnvMetadataResponse(BaseModel):
    submission_id: int
    keys: list[str]
    count: int
    updated_at: datetime | None
    locked: bool
    env_confirmed_empty: bool
    env_confirmed_empty_at: datetime | None
    confirmation_state: Literal["pending", "env_vars_present", "empty_confirmed"]


class MinerEnvLaunchResponse(BaseModel):
    submission_id: int
    status: str
    effective_status: str
    job_id: str | None
    env: MinerEnvMetadataResponse


class OwnerRevalidationRequest(BaseModel):
    """Owner request to force a new evaluation job."""

    reason: str = Field(default="", max_length=4000)


class OwnerOverrideRequest(BaseModel):
    """Owner request to override a submission's effective status."""

    status: Literal["overridden_valid", "overridden_invalid"]
    reason: str = Field(min_length=1, max_length=4000)


class OwnerSuspiciousRequest(BaseModel):
    """Owner request to mark or clear suspicious effective status."""

    suspicious: bool = True
    reason: str = Field(min_length=1, max_length=4000)


class AdminEscalationResolutionRequest(BaseModel):
    decision: Literal["admin_allow", "admin_reject", "admin_request_rerun"]
    reason: str = Field(min_length=1, max_length=4000)


class OwnerControlResponse(BaseModel):
    """Owner control response for submission status changes."""

    submission_id: int
    effective_status: str


class OwnerRevalidationResponse(OwnerControlResponse):
    """Owner revalidation response including the new job."""

    job_id: str
    status: str


class AdminEscalationResolutionResponse(OwnerControlResponse):
    decision_id: int
    decision: str
    status: str
    job_id: str | None = None


class OwnerAuditResponse(BaseModel):
    """Owner audit event response."""

    id: int
    submission_id: int
    owner_hotkey: str
    action: str
    reason: str
    request_hash: str
    nonce: str
    signature: str
    request_timestamp: str | None
    before_effective_status: str | None
    after_effective_status: str | None
    created_at: datetime


class ReviewRetryRequest(BaseModel):
    expected_assignment_id: str = Field(min_length=1, max_length=128)
    approval_id: str | None = Field(default=None, min_length=1, max_length=128)
    refresh_rules: bool = False


class ReviewCancelRequest(BaseModel):
    expected_assignment_id: str = Field(min_length=1, max_length=128)


class ReviewDeployedPhalaCreateReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    app_id: str = Field(min_length=1, max_length=128)
    cvm_id: str = Field(min_length=1, max_length=128)
    receipt_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    created_at_ms: int = Field(ge=0, le=2**63 - 1)


class ReviewDeployedComposeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_ref: str = Field(
        min_length=len("a@sha256:") + 64,
        max_length=1024,
        pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$",
    )
    compose_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    app_kms_public_key_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )


class ReviewDeployedRequest(BaseModel):
    """Exact nested Review deployed acknowledgement v1."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    assignment_id: str = Field(min_length=1, max_length=128)
    cvm_id: str = Field(min_length=1, max_length=128)
    phala_create_receipt: ReviewDeployedPhalaCreateReceipt
    compose_identity: ReviewDeployedComposeIdentity


class ReviewApprovalRequest(BaseModel):
    assignment_id: str = Field(min_length=1, max_length=128)
    action: Literal["retry_policy", "refresh_rules"]
    rules_revision_id: str | None = Field(default=None, min_length=1, max_length=128)


class ReviewPrepareResponse(BaseModel):
    session_id: str
    assignment_id: str
    attempt: int
    assignment: dict[str, Any]
    review_session_token: str | None = Field(default=None, repr=False)


class EvalSecretDeliveryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_key: Literal["EVAL_RUN_TOKEN"]
    token: str = Field(min_length=1, repr=False)


class EvalPrepareResponse(BaseModel):
    """One-time signed Eval authorization wrapper."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    plan: dict[str, Any]
    plan_sha256: str
    secret_delivery: EvalSecretDeliveryResponse | None = Field(default=None, repr=False)


class EvalExpectedRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    eval_run_id: str = Field(min_length=1, max_length=128)


class EvalFailureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    eval_run_id: str = Field(min_length=1, max_length=128)
    reason_code: Literal[
        "eval_deploy_failed",
        "eval_tunnel_failed",
        "eval_key_release_unavailable",
        "eval_no_result",
    ]


class EvalHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    submission_id: int
    current_eval_run_id: str | None = None
    items: list[dict[str, Any]]
    next_cursor: str | None
    total_count: int


class EvalMutationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    run: dict[str, Any]


class EvalReceiptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    eval_run_id: str
    receipt_id: str
    body_sha256: str
    received_at_ms: int
    phase: Literal["received", "verifying", "verified", "rejected", "verifier_unavailable"]
    terminal: bool
    verified: bool
    retryable: bool
    reason_code: str | None
    result_available: bool
    finalized_at_ms: int | None


class ReviewAuditResponse(BaseModel):
    session_id: str
    current_assignment_id: str | None
    authorizing_assignment_id: str | None
    items: list[dict[str, Any]]
    next_cursor: str | None
    total_count: int


class ReviewReportReceiptResponse(BaseModel):
    assignment_id: str
    status: str
    terminal: bool
    retryable: bool
    reason_code: str
    nonce_consumed: bool


class ReviewReportSubmission(BaseModel):
    """Credential-free report plus only bounded encrypted-evidence inputs."""

    model_config = ConfigDict(extra="forbid")

    envelope: dict[str, Any]
    evidence: dict[str, str]


@public_route(tags=["submissions"])
@router.post(
    "/submissions",
    response_model=SubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_submission(
    request: SubmissionRequest,
    http_request: Request,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> SubmissionResponse:
    """Store a signed miner submission without starting evaluation."""

    artifact = await asyncio.to_thread(_prepare_artifact, request)
    return await _persist_submission(
        session=session,
        http_request=http_request,
        artifact=artifact,
        miner_hotkey=auth.hotkey,
        name=request.name,
        signature=auth.signature,
        signature_nonce=auth.nonce,
        signature_timestamp=auth.timestamp,
        signature_payload_sha256=auth.body_sha256,
        signature_message=auth.canonical_request,
        route="POST /submissions",
        actor="api",
    )


@router.post(
    "/internal/v1/bridge/submissions",
    response_model=SubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_base_bridge_submission(
    http_request: Request,
    session: DatabaseSession,
    _auth: InternalBridgeAuth,
    x_base_verified_hotkey: Annotated[str | None, Header(alias="X-Base-Verified-Hotkey")] = None,
    x_base_verified_nonce: Annotated[str | None, Header(alias="X-Base-Verified-Nonce")] = None,
    x_base_request_hash: Annotated[str | None, Header(alias="X-Base-Request-Hash")] = None,
    x_submission_filename: Annotated[str | None, Header(alias="X-Submission-Filename")] = None,
) -> SubmissionResponse:
    headers = _base_bridge_headers(
        hotkey=x_base_verified_hotkey,
        nonce=x_base_verified_nonce,
        request_hash=x_base_request_hash,
        filename=x_submission_filename,
    )
    body = await http_request.body()
    artifact = await asyncio.to_thread(_prepare_raw_zip_artifact, body)
    return await _persist_submission(
        session=session,
        http_request=http_request,
        artifact=artifact,
        miner_hotkey=headers.hotkey,
        name=_submission_display_name(headers.filename),
        signature="base-verified",
        signature_nonce=headers.nonce,
        signature_timestamp=None,
        signature_payload_sha256=headers.request_hash,
        signature_message=_base_bridge_signature_message(headers),
        route="POST /internal/v1/bridge/submissions",
        actor="base_bridge",
    )


@router.post(
    "/submissions/{submission_id}/review/prepare",
    response_model=ReviewPrepareResponse,
)
async def prepare_submission_review(
    submission_id: int,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> ReviewPrepareResponse:
    """Return the current immutable assignment and deliver its token once."""

    review_session = await _get_miner_review_session(session, submission_id, auth)
    try:
        assignment, token = await deliver_prepare_token(
            session,
            session_row=review_session,
            settings=settings,
        )
        await session.commit()
    except ReviewRateLimited as exc:
        await session.rollback()
        raise HTTPException(status_code=429, detail={"code": "review_rate_limited"}) from exc
    except ReviewConflict as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail={"code": "review_prepare_conflict"}) from exc
    return _review_prepare_response(review_session, assignment, token)


@router.post(
    "/submissions/{submission_id}/review/retry",
    response_model=ReviewPrepareResponse,
)
async def retry_submission_review(
    submission_id: int,
    request: ReviewRetryRequest,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> ReviewPrepareResponse:
    """Create a fresh assignment only after an eligible terminal predecessor."""

    review_session = await _get_miner_review_session(session, submission_id, auth)
    try:
        refresh_bundle = (
            await asyncio.to_thread(capture_rules_bundle, settings.review_rules_root)
            if request.refresh_rules
            else None
        )
        created = await retry_review_assignment(
            session,
            session_row=review_session,
            expected_assignment_id=request.expected_assignment_id,
            settings=settings,
            approval_id=request.approval_id,
            refresh_rules_files=rules_bundle_files(refresh_bundle) if refresh_bundle else None,
            refresh_rules_revision_id=(
                str(refresh_bundle["revision_id"]) if refresh_bundle is not None else None
            ),
            input_config=review_input_config_from_settings(settings),
        )
        # Retry itself is the authenticated delivery response for the new
        # capability. Record it before committing so a later prepare cannot
        # replay the plaintext token.
        created.assignment.token_delivered_at = datetime.now(UTC)
        await session.commit()
    except RulesSnapshotCaptureError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "review_rules_snapshot_unavailable"},
        ) from exc
    except ReviewDeploymentError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "review_deployment_identity_unavailable"},
        ) from exc
    except ReviewRateLimited as exc:
        await session.rollback()
        raise HTTPException(status_code=429, detail={"code": "review_rate_limited"}) from exc
    except ReviewConflict as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail={"code": "review_retry_conflict"}) from exc
    return _review_prepare_response(review_session, created.assignment, created.session_token)


@router.post("/submissions/{submission_id}/review/cancel")
async def cancel_submission_review(
    submission_id: int,
    request: ReviewCancelRequest,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> dict[str, object]:
    """Atomically revoke only the named current assignment capability and nonce."""

    review_session = await _get_miner_review_session(session, submission_id, auth)
    try:
        assignment = await cancel_review_assignment(
            session,
            session_row=review_session,
            expected_assignment_id=request.expected_assignment_id,
            settings=settings,
        )
        await session.commit()
    except ReviewRateLimited as exc:
        await session.rollback()
        raise HTTPException(status_code=429, detail={"code": "review_rate_limited"}) from exc
    except ReviewConflict as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail={"code": "review_cancel_conflict"}) from exc
    return {
        "session_id": review_session.session_id,
        "assignment_id": assignment.assignment_id,
        "phase": assignment.phase,
    }


def _eval_prepare_response(created: Any) -> EvalPrepareResponse:
    return EvalPrepareResponse(
        schema_version=1,
        plan=created.plan,
        plan_sha256=created.run.plan_sha256,
        secret_delivery=(
            {"env_key": "EVAL_RUN_TOKEN", "token": created.token} if created.token else None
        ),
    )


async def _get_miner_eval_submission(
    session: AsyncSession,
    submission_id: int,
    auth: SignedRequestAuth,
) -> AgentSubmission:
    if not settings.attested_review_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "attested_eval_disabled"},
        )
    return await _get_miner_env_submission(session, submission_id, auth)


def _eval_authorization_unavailable_detail_code(exc: BaseException) -> str:
    """Map plan-build unavailable failures to closed eval prepare ``detail.code``.

    Prefer an explicit ``exc.code`` when set; otherwise fall back to the identity
    family. Never return messages, paths, or tracebacks to the client.
    """

    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return "eval_deployment_identity_unavailable"


@router.post(
    "/submissions/{submission_id}/eval/prepare",
    response_model=EvalPrepareResponse,
)
async def prepare_submission_eval(
    submission_id: int,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> EvalPrepareResponse:
    """Atomically issue the immutable external Eval run after verified allow."""

    submission = await _get_miner_eval_submission(session, submission_id, auth)
    try:
        created = await create_eval_run(session, submission, settings=settings)
        await session.commit()
    except EvalAuthorizationRequired as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "review_allow_required"},
        ) from exc
    except EvalAuthorizationUnavailable as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": _eval_authorization_unavailable_detail_code(exc)},
        ) from exc
    except EvalAuthorizationConflict as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": exc.code},
        ) from exc
    except Exception as exc:
        # Known plan-build failures raise EvalAuthorization*; anything else must
        # still be closed 503 (never bare FastAPI 500 plaintext secrets/traceback).
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "eval_prepare_internal_unavailable"},
        ) from exc
    return _eval_prepare_response(created)


@router.post(
    "/submissions/{submission_id}/eval/retry",
    response_model=EvalPrepareResponse,
)
async def retry_submission_eval(
    submission_id: int,
    request: EvalExpectedRunRequest,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> EvalPrepareResponse:
    submission = await _get_miner_eval_submission(session, submission_id, auth)
    try:
        created = await retry_eval_run(
            session,
            submission,
            expected_run_id=request.eval_run_id,
            settings=settings,
        )
        await session.commit()
    except EvalAuthorizationRequired as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "review_allow_required"},
        ) from exc
    except EvalAuthorizationUnavailable as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": _eval_authorization_unavailable_detail_code(exc)},
        ) from exc
    except EvalAuthorizationConflict as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": exc.code},
        ) from exc
    except Exception as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "eval_prepare_internal_unavailable"},
        ) from exc
    return _eval_prepare_response(created)


@router.post("/submissions/{submission_id}/eval/cancel", response_model=EvalMutationResponse)
async def cancel_submission_eval(
    submission_id: int,
    request: EvalExpectedRunRequest,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> EvalMutationResponse:
    submission = await _get_miner_eval_submission(session, submission_id, auth)
    try:
        await cancel_eval_run(session, submission, request.eval_run_id)
        await session.commit()
    except EvalAuthorizationConflict as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": exc.code},
        ) from exc
    page = await eval_status_page(session, submission)
    return EvalMutationResponse(schema_version=1, run=page["items"][-1])


@router.post("/submissions/{submission_id}/eval/failure", response_model=EvalMutationResponse)
async def fail_submission_eval(
    submission_id: int,
    request: EvalFailureRequest,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> EvalMutationResponse:
    submission = await _get_miner_eval_submission(session, submission_id, auth)
    try:
        await fail_eval_run(
            session,
            submission,
            expected_run_id=request.eval_run_id,
            reason_code=request.reason_code,
        )
        await session.commit()
    except EvalAuthorizationConflict as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": exc.code},
        ) from exc
    page = await eval_status_page(session, submission)
    return EvalMutationResponse(schema_version=1, run=page["items"][-1])


@router.get(
    "/submissions/{submission_id}/eval/status",
    response_model=EvalHistoryResponse,
)
async def get_submission_eval_status(
    submission_id: int,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
    cursor: str | None = Query(default=None, max_length=512),
    limit: int | None = Query(default=None),
) -> EvalHistoryResponse:
    submission = await _get_miner_eval_submission(session, submission_id, auth)
    page_max = settings.eval_status_page_max
    page_default = settings.eval_status_page_default
    effective_limit = page_default if limit is None else limit
    if not 1 <= effective_limit <= page_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "eval_limit_invalid"},
        )
    try:
        page = await eval_status_page(
            session,
            submission,
            cursor=cursor,
            limit=effective_limit,
            cursor_secret=settings.shared_token,
            page_max=page_max,
            settings=settings,
        )
        await session.commit()
    except EvalAuthorizationConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": exc.code},
        ) from exc
    if len(canonical_json_v1(page)) > settings.eval_status_max_response_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "eval_status_too_large"},
        )
    return EvalHistoryResponse(**page)


@router.get(
    "/internal/v1/replay-audits/requests",
    response_model=ReplayAuditRequestListResponse,
)
async def list_replay_audit_requests(
    session: DatabaseSession,
    _auth: InternalBridgeAuth,
) -> ReplayAuditRequestListResponse:
    """Expose only the sampler-selected accepted attested population."""

    if not settings.phala_attestation_enabled or not settings.attested_review_enabled:
        return ReplayAuditRequestListResponse(requests=[])
    candidates = await accepted_verified_replay_population(session, enabled=True)
    sampler = replay_audit_sampler_from_settings(settings)
    selected_ids = set(sampler.sample(candidates))
    requests: list[ReplayAuditRequestResponse] = []
    for candidate in candidates:
        if candidate.submission_id not in selected_ids:
            continue
        try:
            requests.append(
                ReplayAuditRequestResponse(**replay_request_for_candidate(candidate).to_dict())
            )
        except ReplayAuditWireError:
            continue
    return ReplayAuditRequestListResponse(requests=requests)


@router.get(
    "/internal/v1/replay-audits/{eval_run_id}/request",
    response_model=ReplayAuditRequestResponse,
)
async def get_replay_audit_request(
    eval_run_id: str,
    session: DatabaseSession,
    _auth: InternalBridgeAuth,
) -> ReplayAuditRequestResponse:
    """Expose only an accepted full-attested run through the replay seam."""

    if not settings.phala_attestation_enabled or not settings.attested_review_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    candidates = await accepted_verified_replay_population(session, enabled=True)
    sampled_submission_ids = set(replay_audit_sampler_from_settings(settings).sample(candidates))
    candidate = next(
        (
            item
            for item in candidates
            if item.eval_run_id == eval_run_id and item.submission_id in sampled_submission_ids
        ),
        None,
    )
    if candidate is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "replay_population_ineligible"},
        )
    try:
        request = replay_request_for_candidate(candidate)
    except ReplayAuditWireError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "replay_plan_invalid"},
        ) from exc
    return ReplayAuditRequestResponse(**request.to_dict())


@router.post(
    "/internal/v1/replay-audits/{eval_run_id}/result",
    response_model=ReplayAuditResultResponse,
)
async def receive_replay_audit_result(
    eval_run_id: str,
    http_request: Request,
    session: DatabaseSession,
    _auth: InternalBridgeAuth,
) -> ReplayAuditResultResponse:
    """Compare and persist one BASE replay result without mutating accepted state."""

    if not settings.phala_attestation_enabled or not settings.attested_review_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    body = await http_request.body()
    if len(body) > settings.eval_result_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={"code": "replay_result_too_large"},
        )
    try:
        raw = parse_json_object(body)
        replay_result = replay_result_from_mapping(raw)
    except (ValueError, ReplayAuditWireError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "replay_result_invalid"},
        ) from exc
    if replay_result.eval_run_id != eval_run_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "replay_run_mismatch"},
        )
    candidates = await accepted_verified_replay_population(session, enabled=True)
    sampled_submission_ids = set(replay_audit_sampler_from_settings(settings).sample(candidates))
    candidate = next(
        (
            item
            for item in candidates
            if item.eval_run_id == eval_run_id
            and item.submission_id == replay_result.submission_id
            and item.submission_id in sampled_submission_ids
        ),
        None,
    )
    if candidate is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "replay_population_ineligible"},
        )
    try:
        request = replay_request_for_candidate(
            candidate,
            replay_attempt=replay_result.replay_attempt,
        )
        if (
            replay_result.audit_id != request.audit_id
            or replay_result.plan_sha256 != request.plan_sha256
        ):
            raise ReplayAuditWireError("replay result identity does not match request")
        replay_result.validate_against(request)
        comparison = compare_replay_trials(
            candidate,
            replay_result.trial_scores_by_task,
            spec=AggregationSpec.from_eval_plan(candidate.eval_plan),
            tolerance=settings.replay_audit_tolerance,
        )
        dispute = await persist_replay_dispute(
            session,
            candidate=candidate,
            comparison=comparison,
            replay_attempt=replay_result.replay_attempt,
        )
        await session.commit()
    except (ReplayAuditWireError, InvalidReplayTrialsError) as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "replay_result_invalid"},
        ) from exc
    return ReplayAuditResultResponse(
        audit_id=replay_result.audit_id,
        submission_id=replay_result.submission_id,
        eval_run_id=eval_run_id,
        replay_attempt=replay_result.replay_attempt,
        status="mismatch" if comparison.flagged else "matched",
        attested_score=comparison.attested_score,
        replay_score=comparison.replay_score,
        delta=comparison.delta,
        dispute_id=dispute.id if dispute is not None else None,
    )


@router.post(
    "/evaluation/v1/runs/{eval_run_id}/result",
    response_model=EvalReceiptResponse,
)
async def receive_direct_eval_result(
    eval_run_id: str,
    http_request: Request,
    session: DatabaseSession,
    authorization: Annotated[str | None, Header()] = None,
) -> EvalReceiptResponse:
    """Receive one exact token-scoped result from the canonical Eval CVM."""

    if not settings.attested_review_enabled or not settings.phala_attestation_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "attested_eval_disabled"},
        )
    prefix = "Bearer "
    token = (
        authorization[len(prefix) :]
        if isinstance(authorization, str) and authorization.startswith(prefix)
        else None
    )
    content_type = http_request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "result_media_invalid"},
        )
    # Auth-first: resolve the run and authenticate the Bearer token before any
    # transport body allocation. Content-Length is still checked as a cheap
    # prefilter inside the bounded stream reader.
    run = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == eval_run_id).with_for_update()
    )
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "eval_run_unknown"},
        )
    if not authenticate_eval_token(run, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_eval_token"},
        )
    body = await _read_bounded_result_body(
        http_request,
        max_bytes=settings.eval_result_max_bytes,
    )
    try:
        request_body = parse_json_object(body)
        validate_result_bounds(
            request_body,
            max_tasks=settings.eval_result_max_tasks,
            max_event_log_entries=settings.eval_result_max_event_log_entries,
            max_event_log_bytes=settings.eval_result_max_event_log_bytes,
            max_vm_config_bytes=settings.eval_result_max_vm_config_bytes,
            max_string_bytes=settings.eval_result_max_string_bytes,
            max_quote_bytes=settings.eval_result_max_quote_bytes,
        )
        if request_body.get("eval_run_id") != eval_run_id:
            raise DirectEvalResultError(
                "result run does not match route",
                code="result_run_mismatch",
            )
        receipt, created = await process_direct_eval_result(
            session,
            run=run,
            raw_body=body,
            result_request=request_body,
            settings=settings,
        )
    except EvalAuthorizationConflict as exc:
        await session.rollback()
        if exc.code == "eval_result_receipt_conflict":
            status_code = status.HTTP_409_CONFLICT
        elif exc.code in {"eval_result_rate_limited", "eval_result_overloaded"}:
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        else:
            status_code = status.HTTP_410_GONE
        raise HTTPException(status_code=status_code, detail={"code": exc.code}) from exc
    except ValueError as exc:
        await session.rollback()
        code = exc.code if isinstance(exc, DirectEvalResultError) else "result_invalid"
        if code in {
            "result_too_large",
            "result_tasks_too_many",
            "result_event_log_too_large",
            "result_vm_config_too_large",
            "result_string_too_large",
            "result_quote_too_large",
        }:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={"code": code},
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": code},
        ) from exc
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK,
        content=receipt,
    )


@router.post("/submissions/{submission_id}/review/deployed")
async def acknowledge_submission_review_deployment(
    submission_id: int,
    request: ReviewDeployedRequest,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> dict[str, object]:
    """Record immutable deployment metadata without treating it as evidence."""

    review_session = await _get_miner_review_session(session, submission_id, auth)
    try:
        assignment = await mark_review_deployed(
            session,
            session_row=review_session,
            expected_assignment_id=request.assignment_id,
            deployed_receipt=request.model_dump(),
            settings=settings,
        )
        await session.commit()
    except ReviewRateLimited as exc:
        await session.rollback()
        raise HTTPException(status_code=429, detail={"code": "review_rate_limited"}) from exc
    except ReviewConflict as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail={"code": "review_deployed_conflict"}) from exc
    return {
        "session_id": review_session.session_id,
        "assignment_id": assignment.assignment_id,
        "phase": assignment.phase,
    }


@router.get(
    "/submissions/{submission_id}/review/history",
    response_model=ReviewAuditResponse,
)
async def submission_review_history(
    submission_id: int,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
    cursor: str | None = None,
    limit: int | None = None,
) -> ReviewAuditResponse:
    """Return safe, retained immutable assignment history for the owning miner."""

    page_max = settings.review_report_page_max
    page_default = settings.review_report_page_default
    effective_limit = page_default if limit is None else limit
    if not 1 <= effective_limit <= page_max:
        raise HTTPException(
            status_code=422,
            detail={"code": "review_history_limit_invalid"},
        )
    review_session = await _get_miner_review_session(session, submission_id, auth)
    try:
        page = await review_audit_page(
            session,
            session_row=review_session,
            cursor=cursor,
            limit=effective_limit,
            cursor_secret=settings.shared_token,
            page_max=page_max,
        )
        # Persist lazy expiry on history reads so sticky active rows terminalize.
        await session.commit()
        return ReviewAuditResponse(**page)
    except ReviewConflict as exc:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "review_history_cursor_invalid"},
        ) from exc


@router.get(
    "/submissions/{submission_id}/review/report",
    response_model=ReviewAuditResponse,
)
async def submission_review_report(
    submission_id: int,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
    cursor: str | None = None,
    limit: int | None = None,
) -> ReviewAuditResponse:
    """Return only the deterministic redacted public audit projection."""

    page_max = settings.review_report_page_max
    page_default = settings.review_report_page_default
    effective_limit = page_default if limit is None else limit
    if not 1 <= effective_limit <= page_max:
        raise HTTPException(
            status_code=422,
            detail={"code": "review_report_limit_invalid"},
        )
    review_session = await _get_miner_review_session(session, submission_id, auth)
    try:
        page = await review_audit_page(
            session,
            session_row=review_session,
            cursor=cursor,
            limit=effective_limit,
            cursor_secret=settings.shared_token,
            page_max=page_max,
        )
    except ReviewConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "review_report_cursor_invalid"},
        ) from exc
    if not any(item["report_projection"] is not None for item in page["items"]) and not (
        await _review_has_public_projection(session, review_session)
    ):
        raise HTTPException(status_code=404, detail={"code": "review_report_not_available"})
    _enforce_audit_response_size(page, maximum=settings.review_report_max_response_bytes)
    return ReviewAuditResponse(**page)


@router.get(
    "/internal/v1/reviews/{session_id}/report",
    response_model=ReviewAuditResponse,
)
async def internal_review_report(
    session_id: str,
    session: DatabaseSession,
    _auth: InternalBridgeAuth,
    cursor: str | None = None,
    limit: int | None = None,
) -> ReviewAuditResponse:
    """Return the immutable, authenticated internal audit bundle."""

    page_max = settings.review_report_page_max
    page_default = settings.review_report_page_default
    effective_limit = page_default if limit is None else limit
    if not 1 <= effective_limit <= page_max:
        raise HTTPException(
            status_code=422,
            detail={"code": "review_report_limit_invalid"},
        )
    review_session = await session.scalar(
        select(ReviewSession).where(ReviewSession.session_id == session_id)
    )
    if review_session is None:
        raise HTTPException(status_code=404, detail={"code": "review_session_not_found"})
    try:
        page = await review_audit_page(
            session,
            session_row=review_session,
            cursor=cursor,
            limit=effective_limit,
            internal=True,
            cursor_secret=settings.shared_token,
            page_max=page_max,
        )
    except ReviewConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "review_report_cursor_invalid"},
        ) from exc
    _enforce_audit_response_size(page, maximum=settings.review_internal_report_max_response_bytes)
    return ReviewAuditResponse(**page)


@router.post("/internal/v1/reviews/{session_id}/approvals")
async def create_review_operator_approval(
    session_id: str,
    request: ReviewApprovalRequest,
    session: DatabaseSession,
    _auth: InternalBridgeAuth,
) -> dict[str, object]:
    """Create a validator-only one-use approval for a prior immutable attempt."""

    review_session = await session.scalar(
        select(ReviewSession).where(ReviewSession.session_id == session_id)
    )
    if review_session is None:
        raise HTTPException(status_code=404, detail={"code": "review_session_not_found"})
    assignment = await session.scalar(
        select(ReviewAssignment)
        .where(ReviewAssignment.session_id == review_session.id)
        .where(ReviewAssignment.assignment_id == request.assignment_id)
    )
    if assignment is None:
        raise HTTPException(status_code=404, detail={"code": "review_assignment_not_found"})
    try:
        approval = await issue_operator_approval(
            session,
            session_row=review_session,
            assignment=assignment,
            action=request.action,
            rules_revision_id=request.rules_revision_id,
            actor="internal",
            settings=settings,
        )
        await session.commit()
    except ReviewConflict as exc:
        await session.rollback()
        raise HTTPException(status_code=422, detail={"code": "review_approval_invalid"}) from exc
    return {
        "approval_id": approval.approval_id,
        "session_id": review_session.session_id,
        "assignment_id": assignment.assignment_id,
        "action": approval.action,
        "expires_at": approval.expires_at,
    }


@router.get("/review/v1/assignments/{assignment_id}")
async def fetch_review_assignment(
    assignment_id: str,
    session: DatabaseSession,
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    """Return the exact immutable Review assignment v1 to the scoped review CVM.

    Authenticated only by the assignment-scoped bearer delivered through Phala
    ``encrypted_env``.  This discovery surface exists so the measured runtime can
    bootstrap from ``REVIEW_SESSION_TOKEN`` alone (the token embeds the
    assignment_id) without a third secret channel.
    """

    assignment = await _authenticated_review_assignment(session, assignment_id, authorization)
    try:
        body = json.loads(assignment.assignment_bytes)
    except json.JSONDecodeError as exc:  # pragma: no cover - durable corrupt state
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "review_assignment_corrupt"},
        ) from exc
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "review_assignment_corrupt"},
        )
    return body


@router.get("/review/v1/assignments/{assignment_id}/artifact")
async def fetch_review_assignment_artifact(
    assignment_id: str,
    session: DatabaseSession,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """Serve only immutable, rehashed ZIP bytes to the scoped review CVM."""

    assignment = await _authenticated_review_assignment(session, assignment_id, authorization)
    try:
        review_session, submission = await assignment_artifact(session, assignment=assignment)
        artifact = await _source_artifact(session, submission)
        content = load_assignment_artifact(
            assignment=assignment,
            review_session=review_session,
            artifact=artifact,
        )
        await session.commit()
    except ReviewArtifactError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail={"code": "review_artifact_mismatch"}) from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={
            "Content-Length": str(len(content)),
            "X-Content-SHA256": assignment.artifact_sha256,
        },
    )


@router.get("/review/v1/assignments/{assignment_id}/rules")
async def fetch_review_assignment_rules(
    assignment_id: str,
    session: DatabaseSession,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """Serve the exact immutable canonical Rules bundle v1 for an assignment."""

    assignment = await _authenticated_review_assignment(session, assignment_id, authorization)
    try:
        content = await assignment_rules(session, assignment=assignment)
        await session.commit()
    except ReviewNotFound as exc:
        await session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "review_rules_snapshot_missing"},
        ) from exc
    return Response(
        content=content,
        media_type="application/json",
        headers={"X-Content-SHA256": assignment.rules_snapshot_sha256},
    )


@router.post("/review/v1/assignments/{assignment_id}/model-call-started")
async def record_review_model_call_started(
    assignment_id: str,
    http_request: Request,
    session: DatabaseSession,
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    """Durably bind one planned request before the review CVM opens a socket."""

    assignment = await _authenticated_review_assignment(
        session,
        assignment_id,
        authorization,
        revoked_status=status.HTTP_410_GONE,
    )
    body = await http_request.body()
    # Contract precedence: size 413 before media/JSON 400, schema 422, lifecycle
    # 409, and mutation/rate 429. Mutation budget is checked after schema so a
    # multi-fault body prefers transport size, then media, then schema.
    if len(body) > settings.review_max_string_bytes:
        raise HTTPException(status_code=413, detail={"code": "review_marker_too_large"})
    _require_review_json_media(http_request)
    try:
        marker = parse_json_object(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "review_marker_json_invalid"}) from exc
    try:
        validate_model_call_started(marker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "review_marker_invalid"}) from exc
    try:
        # Share mutation-rate budget with other session mutations (VAL-REVIEW-060).
        review_session = await session.get(ReviewSession, assignment.session_id)
        if review_session is not None:
            await enforce_review_session_mutation_budget(
                session,
                session_row=review_session,
                settings=settings,
            )
        started = await mark_model_call_started(
            session, assignment=assignment, marker=marker, settings=settings
        )
        await session.commit()
    except ReviewRateLimited as exc:
        await session.rollback()
        raise HTTPException(status_code=429, detail={"code": "review_rate_limited"}) from exc
    except ReviewConflict as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail={"code": "review_marker_conflict"}) from exc
    return {
        "assignment_id": assignment.assignment_id,
        "model_call_started": True,
        "idempotent_replay": not started,
    }


@router.post("/review/v1/assignments/{assignment_id}/failure")
async def record_review_infrastructure_failure(
    assignment_id: str,
    http_request: Request,
    session: DatabaseSession,
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    """Record one bounded no-report review failure and revoke its capability."""

    assignment = await _authenticated_review_assignment(
        session,
        assignment_id,
        authorization,
        revoked_status=status.HTTP_410_GONE,
        allow_failure_replay=True,
    )
    body = await http_request.body()
    # Same precedence as model-call-started: size → media/JSON → schema → lifecycle → rate.
    if len(body) > settings.review_max_string_bytes:
        raise HTTPException(status_code=413, detail={"code": "review_failure_too_large"})
    _require_review_json_media(http_request)
    try:
        failure = parse_json_object(body)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "review_failure_json_invalid"},
        ) from exc
    try:
        validate_review_infrastructure_failure(failure)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "review_failure_invalid"}) from exc
    try:
        review_session = await session.get(ReviewSession, assignment.session_id)
        if review_session is not None:
            await enforce_review_session_mutation_budget(
                session,
                session_row=review_session,
                settings=settings,
            )
        recorded = await record_review_infrastructure_failure_state(
            session,
            assignment=assignment,
            failure=failure,
            settings=settings,
        )
        await session.commit()
    except ReviewRateLimited as exc:
        await session.rollback()
        raise HTTPException(status_code=429, detail={"code": "review_rate_limited"}) from exc
    except ReviewConflict as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail={"code": "review_failure_conflict"}) from exc
    return {
        "assignment_id": assignment.assignment_id,
        "phase": assignment.phase,
        "idempotent_replay": not recorded,
    }


@router.post(
    "/review/v1/assignments/{assignment_id}/report",
    response_model=ReviewReportReceiptResponse,
)
async def submit_attested_review_report(
    assignment_id: str,
    http_request: Request,
    session: DatabaseSession,
    authorization: Annotated[str | None, Header()] = None,
) -> ReviewReportReceiptResponse:
    """Receipt and verify one immutable quote-bound Review envelope v1."""

    assignment = await _authenticated_review_assignment(
        session,
        assignment_id,
        authorization,
        revoked_status=status.HTTP_410_GONE,
        allow_report_replay=True,
    )
    body = await http_request.body()
    if len(body) > settings.review_max_report_request_bytes:
        raise HTTPException(status_code=413, detail={"code": "review_report_too_large"})
    _require_review_json_media(http_request)
    try:
        raw_payload = parse_json_object(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"code": "review_report_json_invalid"}) from exc
    # Client-smuggled outer time fields on the public bag cannot decide age
    # (VAL-ACAT-038); strip for security decisions. Schema also extra=forbid.
    from agent_challenge.review.attested_times import ignore_client_smuggled_times

    if isinstance(raw_payload, dict):
        raw_payload = ignore_client_smuggled_times(raw_payload)
    try:
        payload = ReviewReportSubmission.model_validate(raw_payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "review_report_invalid"}) from exc
    try:
        envelope = payload.envelope
        validate_review_envelope(envelope, settings=settings)
        evidence = _decode_review_evidence(payload.evidence)
        # Live admission path: bound outcome + report_data times + OR digest
        # fields on openrouter_observation (library helpers alone insufficient).
        from agent_challenge.review.or_outcome_bind import (
            ReviewOrOutcomeError,
            admit_production_from_bound_outcome,
        )

        review_core = envelope.get("review_core") if isinstance(envelope, dict) else None
        if isinstance(review_core, dict):
            admit_production_from_bound_outcome(
                review_core=review_core,
                reported_report_data_hex=str(envelope.get("report_data_hex")),
                require_or_digests=True,
            )
    except ReviewOrOutcomeError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": getattr(exc, "code", "review_report_invalid")},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "review_report_invalid"}) from exc

    try:
        had_receipt = assignment.review_report_envelope_json is not None
        try:
            allowlist = ReviewMeasurementAllowlist.from_measurements(
                settings.review_app_measurement_allowlist
            )
        except ReviewReportError:
            # A missing or malformed validator configuration matches nothing.
            # The report remains a definitive trust failure, never accept-any.
            allowlist = ReviewMeasurementAllowlist()
        outcome = await submit_review_report(
            session,
            assignment=assignment,
            envelope=envelope,
            evidence_objects=evidence,
            evidence_settings=settings,
            quote_verifier=_review_quote_verifier(),
            allowlist=allowlist,
            now=datetime.now(UTC),
        )
        await session.commit()
    except ReviewReportConflict as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail={"code": "review_report_conflict"}) from exc
    except ReviewReportError as exc:
        await session.rollback()
        status_code = (
            status.HTTP_409_CONFLICT if had_receipt else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        # Prefer closed subclass detail codes so guest residual does not collapse
        # solely to http_status when body codes exist (SPEED residual sub26/27).
        code = "review_report_conflict" if had_receipt else _review_report_error_detail_code(exc)
        raise HTTPException(status_code=status_code, detail={"code": code}) from exc
    except ReviewEvidenceError as exc:
        # Defense-in-depth: store may raise ReviewEvidenceError outside the wrap
        # path. Prefer 422 non-retry with closed code (never raw 500 / secrets).
        await session.rollback()
        code = _review_evidence_error_detail_code(exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": code},
        ) from exc

    response = ReviewReportReceiptResponse(
        assignment_id=assignment.assignment_id,
        status=outcome.status,
        terminal=outcome.terminal,
        retryable=outcome.retryable,
        reason_code=outcome.reason_code,
        nonce_consumed=outcome.nonce_consumed,
    )
    if outcome.status == "verifier_unavailable":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(),
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK if had_receipt else status.HTTP_202_ACCEPTED,
        content=response.model_dump(),
    )


@router.get("/internal/v1/reviews/{session_id}/evidence/{object_ref}")
async def read_review_evidence_object(
    session_id: str,
    object_ref: str,
    session: DatabaseSession,
    _auth: InternalBridgeAuth,
    range_header: Annotated[str | None, Header(alias="Range")] = None,
) -> Response:
    """Return one exact encrypted-evidence object, optionally by one byte range."""

    review_session = await session.scalar(
        select(ReviewSession).where(ReviewSession.session_id == session_id)
    )
    if review_session is None:
        raise HTTPException(status_code=404, detail={"code": "review_evidence_not_found"})
    try:
        _row, content = await load_review_evidence_object(
            session,
            review_session=review_session,
            object_ref=object_ref,
            settings=settings,
        )
    except ReviewEvidenceError as exc:
        raise HTTPException(status_code=404, detail={"code": "review_evidence_not_found"}) from exc
    if len(content) > settings.review_evidence_max_object_bytes:
        raise HTTPException(status_code=413, detail={"code": "review_evidence_too_large"})
    try:
        start, end = _parse_evidence_range(
            range_header,
            len(content),
            max_range_bytes=settings.review_evidence_max_range_bytes,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=416, detail={"code": "review_evidence_range_invalid"}
        ) from exc
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
    }
    if range_header is not None:
        headers["Content-Range"] = f"bytes {start}-{end}/{len(content)}"
        return Response(
            content=content[start : end + 1],
            media_type="application/octet-stream",
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            headers=headers,
        )
    return Response(content=content, media_type="application/octet-stream", headers=headers)


@router.post(
    "/internal/v1/evaluations/{attempt_id}/events",
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_evaluation_log_events(
    attempt_id: int,
    http_request: Request,
    session: DatabaseSession,
    _auth: AttemptStreamAuth,
) -> dict[str, int]:
    """Ingest real-time own_runner log events for one Terminal-Bench attempt.

    Authenticated by the per-attempt scoped token (see
    ``build_attempt_stream_auth_dependency``). Each NDJSON ``log`` line is
    attributed to the attempt's own submission/job/task (never values from the
    request body), redacted, and appended via ``record_task_event`` so the live
    SSE feed surfaces it. This route only ever records observability logs; it
    never touches the attempt's score (which stays the authoritative
    ``BASE_BENCHMARK_RESULT=`` stdout line finalized elsewhere).
    """

    attempt = await session.get(EvaluationAttempt, attempt_id)
    if attempt is None or attempt.evaluator_name != TERMINAL_BENCH_EVALUATOR:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation attempt not found",
        )
    body = await http_request.body()
    if len(body) > MAX_STREAM_EVENTS_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="event batch too large",
        )
    # Snapshot the attempt's scalar identity before writing: the lock-retry
    # helper rolls the session back between attempts, which expires ORM objects,
    # and re-reading an expired attribute mid-retry would trigger a synchronous
    # lazy-load that is illegal under async SQLAlchemy. Locals stay valid.
    submission_id = attempt.submission_id
    job_id = attempt.job_id
    task_id = attempt.task_id
    stream_attempt_id = attempt.id
    redaction_values = await _attempt_stream_redaction_values(session, submission_id)
    log_events = [
        coerced
        for event in _parse_stream_events(body)
        if (coerced := _coerce_stream_log_event(event)) is not None
    ]
    recorded = 0
    # Record + commit ONE event per lock-retry transaction. Committing per event
    # (rather than across the whole up-to-512-event request) keeps the single
    # SQLite writer lock held only briefly and, critically, makes each event an
    # independently retryable unit: on a momentary "database is locked" collision
    # (two writers upgrading SHARED -> RESERVED) the helper rolls back and replays
    # only the failing event, so already-committed events are never re-inserted and
    # byte accounting stays exact. Inert on PostgreSQL, where the collision never
    # occurs. O(1) byte accounting keeps each event's write tiny.
    for log_event in log_events:

        async def _record_event(log_event: _StreamLogEvent = log_event) -> None:
            await record_task_event(
                session,
                submission_id=submission_id,
                job_id=job_id,
                task_id=task_id,
                event_type="task.log",
                stream=log_event.stream,
                message=apply_miner_env_redaction(log_event.message, redaction_values),
                status=log_event.status,
                metadata={
                    "evaluator": TERMINAL_BENCH_EVALUATOR,
                    "attempt_id": stream_attempt_id,
                    "trial_name": log_event.trial_name,
                    "streamed": True,
                },
            )

        await run_write_with_lock_retry(session, _record_event)
        recorded += 1
    return {"recorded": recorded}


def _parse_stream_events(body: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in body.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
        if len(events) >= MAX_STREAM_EVENTS_PER_REQUEST:
            break
    return events


@dataclass(frozen=True)
class _StreamLogEvent:
    stream: str
    message: str
    status: str | None
    trial_name: str | None


def _coerce_stream_log_event(event: dict[str, Any]) -> _StreamLogEvent | None:
    """Validate one raw ingest line into a log event, or None if it is not one.

    Collecting the valid events up front lets each one be recorded as a
    self-contained, independently replayable unit by the lock-retry helper.
    """

    if event.get("kind") != "log":
        return None
    stream = event.get("stream")
    message = event.get("message")
    if stream not in STREAM_LOG_CHANNELS or not isinstance(message, str) or not message:
        return None
    event_status = event.get("status")
    trial_name = event.get("trial_name")
    return _StreamLogEvent(
        stream=str(stream),
        message=message,
        status=event_status if isinstance(event_status, str) else None,
        trial_name=trial_name if isinstance(trial_name, str) else None,
    )


async def _attempt_stream_redaction_values(
    session: AsyncSession,
    submission_id: int,
) -> dict[str, str]:
    rows = (
        (
            await session.execute(
                select(SubmissionEnvVar)
                .where(SubmissionEnvVar.submission_id == submission_id)
                .where(SubmissionEnvVar.locked_at.is_not(None))
            )
        )
        .scalars()
        .all()
    )
    values: dict[str, str] = {}
    for row in rows:
        try:
            values[row.key] = row.decrypt_value_for_launch(settings)
        except Exception:  # noqa: BLE001 - skip any undecryptable secret
            continue
    return values


@router.get("/submissions/{submission_id}/env", response_model=MinerEnvMetadataResponse)
async def get_submission_env(
    submission_id: int,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> MinerEnvMetadataResponse:
    submission = await _get_miner_env_submission(session, submission_id, auth)
    env_vars = await _submission_env_vars(session, submission.id)
    return _miner_env_metadata_response(submission, env_vars)


@router.put("/submissions/{submission_id}/env", response_model=MinerEnvMetadataResponse)
async def replace_submission_env(
    submission_id: int,
    request: MinerEnvUpdateRequest,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> MinerEnvMetadataResponse:
    submission = await _get_miner_env_submission(session, submission_id, auth)
    _ensure_miner_env_editable(submission)
    env = _validated_miner_env(request.env)

    await session.execute(
        delete(SubmissionEnvVar).where(SubmissionEnvVar.submission_id == submission.id)
    )
    submission.env_confirmed_empty = False
    submission.env_confirmed_empty_at = None
    for key, value in sorted(env.items()):
        try:
            session.add(
                SubmissionEnvVar.encrypted(
                    submission_id=submission.id,
                    key=key,
                    value=value,
                    settings=settings,
                )
            )
        except SubmissionEnvEncryptionError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="submission env storage unavailable",
            ) from exc
    await session.flush()
    env_vars = await _submission_env_vars(session, submission.id)
    if submission.raw_status == "waiting_miner_env" and env_vars:
        job = await _lock_env_and_enqueue_submission(
            session,
            submission,
            confirmed_empty=False,
        )
        if job is not None and job.trigger_reason is None:
            job.triggered_by_hotkey = auth.hotkey
            job.trigger_reason = "miner_env_update"
    await session.commit()
    env_vars = await _submission_env_vars(session, submission.id)
    await session.refresh(submission)
    return _miner_env_metadata_response(submission, env_vars)


@router.post(
    "/submissions/{submission_id}/env/confirm-empty",
    response_model=MinerEnvMetadataResponse,
)
async def confirm_empty_submission_env(
    submission_id: int,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> MinerEnvMetadataResponse:
    submission = await _get_miner_env_submission(session, submission_id, auth)
    _ensure_miner_env_editable(submission)
    env_vars = await _submission_env_vars(session, submission.id)
    if env_vars:
        raise HTTPException(status_code=409, detail="submission env vars already exist")
    job = await confirm_empty_miner_env_and_enqueue_evaluation(
        session,
        submission,
        actor=auth.hotkey,
    )
    if job is not None and job.trigger_reason is None:
        job.triggered_by_hotkey = auth.hotkey
        job.trigger_reason = "miner_env_confirm_empty"
    await session.commit()
    await session.refresh(submission)
    return _miner_env_metadata_response(submission, [])


@router.post("/submissions/{submission_id}/launch", response_model=MinerEnvLaunchResponse)
async def launch_submission_evaluation(
    submission_id: int,
    session: DatabaseSession,
    auth: SignedSubmissionAuth,
) -> MinerEnvLaunchResponse:
    submission = await _get_miner_env_submission(session, submission_id, auth)
    existing_job = await existing_evaluation_job_for_submission(session, submission)
    if (
        submission.raw_status in {"tb_queued", "tb_running"}
        and existing_job is not None
        and existing_job.status in {"queued", "running"}
    ):
        job = existing_job
    else:
        if submission.raw_status != "waiting_miner_env":
            raise HTTPException(status_code=409, detail="submission env is locked")
        env_vars = await _submission_env_vars(session, submission.id)
        if not env_vars and not submission.env_confirmed_empty:
            raise HTTPException(status_code=409, detail="submission env confirmation is required")
        job = await _lock_env_and_enqueue_submission(
            session,
            submission,
            confirmed_empty=not env_vars,
        )
    if job is not None and job.trigger_reason is None:
        job.triggered_by_hotkey = auth.hotkey
        job.trigger_reason = "miner_env_launch"
    await session.commit()
    await session.refresh(submission)
    if job is not None:
        await session.refresh(job)
    locked_env_vars = await _submission_env_vars(session, submission.id)
    return MinerEnvLaunchResponse(
        submission_id=submission.id,
        status=submission.raw_status,
        effective_status=submission.effective_status,
        job_id=job.job_id if job is not None else None,
        env=_miner_env_metadata_response(submission, locked_env_vars),
    )


@router.post(
    "/owner/submissions/{submission_id}/revalidate",
    response_model=OwnerRevalidationResponse,
)
async def owner_revalidate_submission(
    submission_id: int,
    request: OwnerRevalidationRequest,
    session: DatabaseSession,
    auth: OwnerSignedAuth,
) -> OwnerRevalidationResponse:
    """Force a new evaluation job for an immutable submitted artifact."""

    submission = await _get_submission_or_404(session, submission_id)
    before_status = submission.effective_status
    try:
        job = await create_evaluation_job(session, submission)
    except EvaluationAuthorizationError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "review_authorization_required"},
        ) from exc
    job.triggered_by_hotkey = auth.hotkey
    job.trigger_reason = "revalidate"
    after_status = submission.effective_status
    _append_owner_audit(
        session=session,
        submission=submission,
        auth=auth,
        action="revalidate",
        reason=request.reason.strip(),
        before_status=before_status,
        after_status=after_status,
    )
    await session.commit()
    await session.refresh(submission)
    await session.refresh(job)
    return OwnerRevalidationResponse(
        submission_id=submission.id,
        effective_status=submission.effective_status,
        job_id=job.job_id,
        status=job.status,
    )


@router.post(
    "/owner/submissions/{submission_id}/override",
    response_model=OwnerControlResponse,
)
async def owner_override_submission(
    submission_id: int,
    request: OwnerOverrideRequest,
    session: DatabaseSession,
    auth: OwnerSignedAuth,
) -> OwnerControlResponse:
    """Override only the submission's effective status."""

    reason = _required_reason(request.reason)
    submission = await _get_submission_or_404(session, submission_id)
    if settings.attested_review_enabled and request.status == "overridden_valid":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "review_authorization_required"},
        )
    before_status = submission.effective_status
    submission.effective_status = request.status
    _append_owner_audit(
        session=session,
        submission=submission,
        auth=auth,
        action="override",
        reason=reason,
        before_status=before_status,
        after_status=submission.effective_status,
    )
    await session.commit()
    await session.refresh(submission)
    return OwnerControlResponse(
        submission_id=submission.id,
        effective_status=submission.effective_status,
    )


@router.post(
    "/owner/submissions/{submission_id}/admin-escalation",
    response_model=AdminEscalationResolutionResponse,
)
async def owner_resolve_admin_escalation(
    submission_id: int,
    request: AdminEscalationResolutionRequest,
    session: DatabaseSession,
    auth: OwnerSignedAuth,
) -> AdminEscalationResolutionResponse:
    reason = _required_reason(request.reason)
    submission = await _get_submission_or_404(session, submission_id)
    if settings.attested_review_enabled and settings.phala_attestation_enabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "review_authorization_required"},
        )
    if submission.raw_status not in {"analysis_escalated", "admin_paused"}:
        raise HTTPException(status_code=409, detail="submission is not awaiting admin review")

    before_status = submission.effective_status
    previous_analysis = await _latest_analysis_run(session, submission.id)
    decision = _append_admin_review_decision(
        session=session,
        submission=submission,
        auth=auth,
        decision=request.decision,
        reason=reason,
        before_status=before_status,
        previous_analysis=previous_analysis,
    )
    await session.flush()

    job: EvaluationJob | None = None
    metadata = _admin_decision_status_metadata(decision, previous_analysis)
    if request.decision == "admin_allow":
        await ensure_submission_status(
            session,
            submission,
            "analysis_allowed",
            actor=auth.hotkey,
            reason="admin_review_allowed",
            metadata=metadata,
        )
        if _legacy_confirmed_empty_submission(submission):
            job = await enqueue_evaluation_job_for_submission(session, submission)
            if job is not None:
                job.triggered_by_hotkey = auth.hotkey
                job.trigger_reason = "admin_allow"
        else:
            await ensure_submission_status(
                session,
                submission,
                "waiting_miner_env",
                actor=auth.hotkey,
                reason="waiting_miner_env",
                metadata=metadata,
            )
    elif request.decision == "admin_reject":
        await ensure_submission_status(
            session,
            submission,
            "analysis_rejected",
            actor=auth.hotkey,
            reason="admin_review_rejected",
            metadata=metadata,
        )
    else:
        if submission.raw_status == "analysis_escalated":
            await ensure_submission_status(
                session,
                submission,
                "admin_paused",
                actor=auth.hotkey,
                reason="admin_review_rerun_requested",
                metadata=metadata,
            )
        await ensure_submission_status(
            session,
            submission,
            "analysis_queued",
            actor=auth.hotkey,
            reason="admin_review_rerun_requested",
            metadata=metadata,
        )

    decision.after_effective_status = submission.effective_status
    await session.commit()
    await session.refresh(submission)
    await session.refresh(decision)
    if job is not None:
        await session.refresh(job)
    return AdminEscalationResolutionResponse(
        submission_id=submission.id,
        effective_status=submission.effective_status,
        decision_id=decision.id,
        decision=decision.decision,
        status=submission.raw_status,
        job_id=job.job_id if job is not None else None,
    )


@router.post(
    "/owner/submissions/{submission_id}/suspicious",
    response_model=OwnerControlResponse,
)
async def owner_mark_submission_suspicious(
    submission_id: int,
    request: OwnerSuspiciousRequest,
    session: DatabaseSession,
    auth: OwnerSignedAuth,
) -> OwnerControlResponse:
    """Mark or clear only the submission's suspicious effective status."""

    reason = _required_reason(request.reason)
    submission = await _get_submission_or_404(session, submission_id)
    before_status = submission.effective_status
    submission.effective_status = (
        "suspicious" if request.suspicious else public_status_for(submission.raw_status)
    )
    _append_owner_audit(
        session=session,
        submission=submission,
        auth=auth,
        action="suspicious",
        reason=reason,
        before_status=before_status,
        after_status=submission.effective_status,
    )
    await session.commit()
    await session.refresh(submission)
    return OwnerControlResponse(
        submission_id=submission.id,
        effective_status=submission.effective_status,
    )


@router.get("/owner/audit", response_model=list[OwnerAuditResponse])
async def owner_audit_history(
    session: DatabaseSession,
    auth: OwnerSignedAuth,
) -> list[OwnerAuditResponse]:
    """Return append-only owner action audit history."""

    _ = auth
    result = await session.execute(
        select(OwnerActionAudit).order_by(OwnerActionAudit.created_at, OwnerActionAudit.id)
    )
    return [
        OwnerAuditResponse(
            id=row.id,
            submission_id=row.submission_id,
            owner_hotkey=row.owner_hotkey,
            action=row.action,
            reason=row.reason,
            request_hash=row.request_hash,
            nonce=row.nonce,
            signature=row.signature,
            request_timestamp=row.request_timestamp,
            before_effective_status=row.before_effective_status,
            after_effective_status=row.after_effective_status,
            created_at=row.created_at,
        )
        for row in result.scalars().all()
    ]


@public_route(tags=["benchmarks"])
@router.get("/benchmarks", response_model=BenchmarkInfoResponse)
async def benchmark_info(response: Response) -> BenchmarkInfoResponse:
    """Return the active benchmark configuration."""

    if settings.attested_review_enabled and settings.phala_attestation_enabled:
        _public_cache(response, max_age=60, swr=300)
        return BenchmarkInfoResponse(
            backend="",
            dataset="",
            task_count=0,
            evaluation_concurrency=0,
        )
    tasks = load_benchmark_tasks()
    dataset = (
        settings.terminal_bench_dataset
        if settings.benchmark_backend == "terminal_bench"
        else settings.swe_forge_tree_url
    )
    _public_cache(response, max_age=60, swr=300)
    return BenchmarkInfoResponse(
        backend=settings.benchmark_backend,
        dataset=dataset,
        task_count=len(tasks),
        evaluation_concurrency=settings.evaluation_concurrency,
    )


@public_route(tags=["benchmarks"])
@router.get("/benchmarks/tasks", response_model=list[BenchmarkTaskResponse])
async def benchmark_tasks() -> list[BenchmarkTaskResponse]:
    """Return benchmark tasks or Harbor shards selected by configuration."""

    if settings.attested_review_enabled and settings.phala_attestation_enabled:
        return []
    return [
        BenchmarkTaskResponse(
            task_id=task.task_id,
            benchmark=task.benchmark,
            docker_image=task.docker_image,
            prompt=task.prompt,
        )
        for task in load_benchmark_tasks()
    ]


@router.get("/internal/v1/work_units", response_model=WorkUnitsResponse)
async def list_work_units(
    session: DatabaseSession,
    _auth: InternalBridgeAuth,
) -> WorkUnitsResponse:
    """Expose pending task work units for the master coordination plane.

    Units exist only for submissions whose central AST + LLM gates returned an
    ``allow`` verdict (which created the evaluation job); ``reject``/``escalate``
    submissions and pre-verdict submissions surface nothing here.

    In combined-worker mode the in-process worker owns evaluation end-to-end, so
    the challenge does not participate in the decentralized coordination plane:
    it exposes no work units, giving the master nothing to assign or fold (a
    fold would otherwise clobber the in-process worker's real results).
    """

    if settings.combined_worker:
        return WorkUnitsResponse(challenge_slug=settings.slug, work_units=[])
    if settings.attested_review_enabled and settings.phala_attestation_enabled:
        return WorkUnitsResponse(challenge_slug=settings.slug, work_units=[])

    units = await list_pending_work_units(session)
    return WorkUnitsResponse(
        challenge_slug=settings.slug,
        work_units=[
            WorkUnitResponse(
                work_unit_id=unit.work_unit_id,
                submission_id=unit.submission_id,
                submission_ref=unit.submission_ref,
                miner_hotkey=unit.miner_hotkey,
                job_id=unit.job_id,
                task_id=unit.task_id,
                docker_image=unit.docker_image,
                required_capability=unit.required_capability,
            )
            for unit in units
        ],
    )


@router.post("/internal/v1/work_units/fold", response_model=FoldWorkUnitResponse)
async def fold_work_unit(
    payload: FoldWorkUnitRequest,
    session: DatabaseSession,
    _auth: InternalBridgeAuth,
) -> FoldWorkUnitResponse:
    """Fold a permanently-failed work unit so its EvaluationJob can finalize.

    The master coordination plane calls this when a work unit exhausts
    ``max_attempts`` (no validator will ever report a result for it). The failed
    task is recorded once (status ``failed``, score ``0.0``) and the job is
    finalized if every task is now terminal, so a permanently-failed task never
    hangs its job forever. Idempotent: a task with an existing terminal result is
    left untouched.

    In combined-worker mode the in-process worker owns evaluation end-to-end and
    the challenge exposes no work units, so a fold should never happen. As
    defense-in-depth against an already-assigned unit, folding is a benign no-op
    here: it writes nothing (a failed/score-0 result would clobber the worker's
    real result via the ``(job_id, task_id)`` idempotency race) and finalizes
    nothing, returning a success response without touching the database.
    """

    if settings.combined_worker:
        return FoldWorkUnitResponse(
            work_unit_id=payload.task_id,
            job_id=payload.job_id,
            task_id=payload.task_id,
            status="skipped",
            score=0.0,
            posted=False,
            finalized=False,
        )
    if settings.attested_review_enabled and settings.phala_attestation_enabled:
        return FoldWorkUnitResponse(
            work_unit_id=payload.task_id,
            job_id=payload.job_id,
            task_id=payload.task_id,
            status="skipped",
            score=0.0,
            posted=False,
            finalized=False,
        )

    try:
        outcome = await fold_terminally_failed_work_unit(
            session,
            job_id=payload.job_id,
            task_id=payload.task_id,
            reason=payload.reason,
        )
        summary = await finalize_job_if_complete(session, payload.job_id)
        await session.commit()
    except ValueError as exc:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FoldWorkUnitResponse(
        work_unit_id=outcome.work_unit_id,
        job_id=outcome.job_id,
        task_id=outcome.task_id,
        status=outcome.status,
        score=outcome.score,
        posted=outcome.posted,
        finalized=summary is not None and summary.status == "completed",
    )


@public_route(tags=["submissions"])
@router.get("/submissions", response_model=list[SubmissionListItem])
async def list_submissions(
    session: DatabaseSession,
    response: Response,
) -> list[SubmissionListItem]:
    """Return recent submissions with their latest score."""

    result = await session.execute(
        select(AgentSubmission)
        .options(
            selectinload(AgentSubmission.jobs),
            selectinload(AgentSubmission.env_vars),
            selectinload(AgentSubmission.latest_evaluation_job),
            selectinload(AgentSubmission.submission_family),
        )
        .order_by(desc(AgentSubmission.created_at))
        .limit(100)
    )
    submissions = result.scalars().all()
    summaries = await _submission_analysis_summaries(
        session, [submission.id for submission in submissions]
    )
    _public_cache(response, max_age=5, swr=30)
    return [
        _submission_list_item(submission, summaries.get(submission.id))
        for submission in submissions
    ]


@public_route(tags=["submissions"])
@router.get("/submissions/count", response_model=SubmissionCountResponse)
async def count_submissions(
    session: DatabaseSession,
    response: Response,
) -> SubmissionCountResponse:
    """Return the number of stored submissions."""

    count = await session.scalar(select(func.count(AgentSubmission.id)))
    _public_cache(response, max_age=5, swr=30)
    return SubmissionCountResponse(count=count or 0)


@public_route(tags=["submissions"])
@router.get("/submissions/{submission_id}", response_model=SubmissionListItem)
async def get_submission(
    submission_id: int,
    session: DatabaseSession,
) -> SubmissionListItem:
    """Return one submission by id."""

    submission = await session.scalar(
        select(AgentSubmission)
        .where(AgentSubmission.id == submission_id)
        .options(
            selectinload(AgentSubmission.jobs),
            selectinload(AgentSubmission.env_vars),
            selectinload(AgentSubmission.latest_evaluation_job),
            selectinload(AgentSubmission.submission_family),
        )
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")
    summaries = await _submission_analysis_summaries(session, [submission.id])
    return _submission_list_item(submission, summaries.get(submission.id))


@public_route(tags=["submissions"])
@router.get("/submissions/{submission_id}/versions", response_model=list[SubmissionVersionItem])
async def get_submission_versions(
    submission_id: int,
    session: DatabaseSession,
) -> list[SubmissionVersionItem]:
    submission = await session.scalar(
        select(AgentSubmission)
        .where(AgentSubmission.id == submission_id)
        .options(
            selectinload(AgentSubmission.jobs),
            selectinload(AgentSubmission.latest_evaluation_job),
            selectinload(AgentSubmission.submission_family),
        )
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")
    if submission.submission_family_id is None:
        return [_submission_version_item(submission)]

    result = await session.execute(
        select(AgentSubmission)
        .where(AgentSubmission.submission_family_id == submission.submission_family_id)
        .options(
            selectinload(AgentSubmission.jobs),
            selectinload(AgentSubmission.latest_evaluation_job),
            selectinload(AgentSubmission.submission_family),
        )
        .order_by(AgentSubmission.version_number, AgentSubmission.id)
    )
    return [_submission_version_item(version) for version in result.scalars().all()]


@public_route(tags=["submissions"])
@router.get("/v1/submissions/{submission_id}", response_model=SubmissionListItem)
async def get_v1_submission(
    submission_id: int,
    session: DatabaseSession,
) -> SubmissionListItem:
    return await get_submission(submission_id, session)


@public_route(tags=["submissions"])
@router.get("/submissions/{submission_id}/status", response_model=None)
async def get_submission_status(
    submission_id: int,
    session: DatabaseSession,
    response: Response,
) -> JSONResponse:
    """Return a safe polling snapshot for one submission.

    Fully legacy mode omits the review field entirely so the response bytes stay
    identical to the pre-review schema. Full attested mode includes the safe
    review projection.
    """

    submission = await session.scalar(
        select(AgentSubmission)
        .where(AgentSubmission.id == submission_id)
        .options(selectinload(AgentSubmission.submission_family))
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")
    _public_cache(response, max_age=2, swr=15)
    body = await _submission_status_response(session, submission)
    # Persist lazy review TTL expiry performed during status read.
    if session.dirty or session.new or session.deleted:
        await session.commit()
    payload = body.model_dump(mode="json")
    if not (settings.attested_review_enabled and settings.phala_attestation_enabled):
        payload.pop("review", None)
    return JSONResponse(content=payload, headers=dict(response.headers))


@public_route(tags=["submissions"])
@router.get("/v1/submissions/{submission_id}/status", response_model=None)
async def get_v1_submission_status(
    submission_id: int,
    session: DatabaseSession,
    response: Response,
) -> JSONResponse:
    return await get_submission_status(submission_id, session, response)


@public_route(tags=["submissions"])
@router.get(
    "/submissions/{submission_id}/review/tee",
    response_model=PublicTeeMathResponse,
    response_model_exclude_none=True,
    summary="Public TEE math for independent inspectability",
    responses={
        200: {
            "description": (
                "Safe TEE math when a verified report exists; otherwise the locked "
                'closed form {"available": false}.'
            ),
        },
        404: {"description": "submission not found"},
    },
)
async def get_submission_review_tee(
    submission_id: int,
    session: DatabaseSession,
    response: Response,
) -> JSONResponse:
    """Return the public unauthenticated TEE math projection.

    Same trust class as ``GET .../status``: no miner signature required. Body is
    exactly ``{"available": false}`` when no authorizing/current verified report
    envelope is durable. Never exposes nonce plaintext, tokens, capabilities,
    evidence bodies, model IO, or encryption KEY material.
    """

    submission = await session.scalar(
        select(AgentSubmission).where(AgentSubmission.id == submission_id)
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")
    _public_cache(response, max_age=5, swr=30)
    payload = await _public_tee_math_response(session, submission)
    return JSONResponse(content=payload, headers=dict(response.headers))


@public_route(tags=["submissions"])
@router.get(
    "/v1/submissions/{submission_id}/review/tee",
    response_model=PublicTeeMathResponse,
    response_model_exclude_none=True,
    summary="Public TEE math (v1 alias)",
)
async def get_v1_submission_review_tee(
    submission_id: int,
    session: DatabaseSession,
    response: Response,
) -> JSONResponse:
    """v1 alias for the public TEE math surface (product bridge convention)."""

    return await get_submission_review_tee(submission_id, session, response)


@public_route(tags=["submissions"])
@router.get(
    "/submissions/{submission_id}/task-events",
    response_model=TaskEventReplayResponse,
)
async def get_submission_task_events(
    submission_id: int,
    session: DatabaseSession,
    cursor: str | None = None,
    limit: Annotated[
        int,
        Query(ge=1, le=MAX_TASK_EVENT_REPLAY_LIMIT),
    ] = DEFAULT_TASK_EVENT_REPLAY_LIMIT,
    task_id: str | None = None,
    event_type: str | None = None,
    stream: str | None = None,
) -> TaskEventReplayResponse:
    submission = await session.scalar(
        select(AgentSubmission)
        .where(AgentSubmission.id == submission_id)
        .options(selectinload(AgentSubmission.submission_family))
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")

    parsed_cursor = _parse_task_event_cursor(cursor)
    max_sequence = await _max_task_event_sequence(session, submission_id)
    if parsed_cursor > max_sequence:
        raise _invalid_task_event_cursor(max_sequence)

    events = await _task_events_after_cursor(
        session,
        submission_id=submission_id,
        cursor=parsed_cursor,
        limit=limit,
        task_id=task_id,
        event_type=event_type,
        stream=stream,
    )
    has_more = len(events) > limit
    page_events = events[:limit]
    next_cursor = page_events[-1].sequence if page_events else parsed_cursor

    return TaskEventReplayResponse(
        submission_id=submission.id,
        name=submission.name,
        agent_hash=submission.agent_hash,
        **_version_metadata(submission),
        cursor=parsed_cursor,
        next_cursor=next_cursor,
        limit=limit,
        has_more=has_more,
        events=[_task_event_replay_item(event) for event in page_events],
    )


@public_route(tags=["submissions"])
@router.get("/submissions/{submission_id}/task-events/stream")
async def stream_submission_task_events(
    submission_id: int,
    request: Request,
    session: DatabaseSession,
    cursor: str | None = None,
    stream: str | None = None,
) -> StreamingResponse:
    submission = await session.scalar(
        select(AgentSubmission)
        .where(AgentSubmission.id == submission_id)
        .options(selectinload(AgentSubmission.submission_family))
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")

    parsed_cursor = _task_event_stream_cursor(cursor, request)
    max_sequence = await _max_task_event_sequence(session, submission_id)
    if parsed_cursor > max_sequence:
        raise _invalid_task_event_cursor(max_sequence)

    # Release the request-scoped connection before the (potentially long-lived)
    # stream: the generator opens a fresh short-lived session per poll instead of
    # pinning this one for the whole stream. version_label is a plain column, so
    # capturing it now avoids touching the ORM object after the session closes.
    version_label = submission.version_label
    await session.close()

    return StreamingResponse(
        _submission_task_event_stream(submission_id, version_label, parsed_cursor, stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@public_route(tags=["submissions"])
@router.get("/submissions/{submission_id}/events")
async def stream_submission_events(
    submission_id: int,
    request: Request,
    session: DatabaseSession,
) -> StreamingResponse:
    submission = await session.get(AgentSubmission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")

    last_event_id = _last_event_id_header(request)
    if last_event_id is not None:
        first_event_id = await _first_status_event_id(session, submission_id)
        if not await _status_event_id_exists(session, submission_id, last_event_id):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"detail": "unknown Last-Event-ID", "replay_from": first_event_id},
            )

    # Release the request-scoped connection before streaming; the generator polls
    # with a fresh short-lived session per iteration.
    await session.close()

    return StreamingResponse(
        _submission_event_stream(submission_id, last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@public_route(tags=["evaluations"])
@router.get("/agents/{agent_hash}/evaluation", response_model=EvaluationResponse)
async def get_agent_evaluation(
    agent_hash: str,
    session: DatabaseSession,
    response: Response,
) -> EvaluationResponse:
    """Return evaluation details for an agent hash."""

    if settings.attested_review_enabled and settings.phala_attestation_enabled:
        raise HTTPException(status_code=404, detail="agent evaluation not found")
    result = await session.execute(
        select(EvaluationJob)
        .join(EvaluationJob.submission)
        .where(AgentSubmission.agent_hash == agent_hash)
        .options(
            selectinload(EvaluationJob.submission),
            selectinload(EvaluationJob.submission).selectinload(AgentSubmission.submission_family),
            selectinload(EvaluationJob.task_results),
        )
        .order_by(desc(EvaluationJob.created_at))
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="agent evaluation not found")
    _public_cache(response, max_age=5, swr=30)
    return await _evaluation_response(session, job)


@public_route(tags=["agents"])
@router.get("/agents/{agent_hash}/source", response_model=AgentSourceResponse)
async def get_agent_source(
    agent_hash: str,
    session: DatabaseSession,
    response: Response,
) -> AgentSourceResponse:
    """Return an agent submission's source files with secrets redacted server-side.

    ``agent_hash`` is unique per submission, so this resolves the same submission
    ``get_agent_evaluation`` would (latest by ``created_at`` if ever duplicated).
    Only the code files stored in the submission zip are exposed; miner env
    variable values are never included. When the submission exists but its source
    artifact is missing or unreadable on disk the response is ``available=false``
    with an empty file list rather than an error.
    """

    submission = await session.scalar(
        select(AgentSubmission)
        .where(AgentSubmission.agent_hash == agent_hash)
        .order_by(desc(AgentSubmission.created_at))
        .limit(1)
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="agent source not found")

    files, total_bytes, truncated, available = await _collect_agent_source_files(
        session, submission
    )
    _public_cache(response, max_age=300, swr=86400)
    return AgentSourceResponse(
        agent_hash=submission.agent_hash,
        submission_id=submission.id,
        agent_name=submission.agent_name,
        miner_hotkey=submission.miner_hotkey,
        available=available,
        total_files=len(files),
        total_bytes=total_bytes,
        truncated=truncated,
        files=files,
    )


@public_route(tags=["submissions"])
@router.get("/agents/{agent_hash}/source/download")
async def download_agent_source(
    agent_hash: str,
    session: DatabaseSession,
) -> Response:
    """Return a ZIP of an agent submission's source with secrets redacted.

    Resolves the same submission ``get_agent_source`` would (``agent_hash`` is
    unique; latest by ``created_at`` if ever duplicated) so the frontend can offer
    a "Download ZIP" button. Text entries are passed through ``redact_secrets``
    exactly as the source viewer does, so raw secrets are never exposed; binary
    entries are copied through unchanged. Only manifest-listed, traversal-safe
    paths are written and a total uncompressed cap bounds memory. Responds ``404``
    when the submission or its source artifact is missing/unreadable.
    """

    submission = await session.scalar(
        select(AgentSubmission)
        .where(AgentSubmission.agent_hash == agent_hash)
        .order_by(desc(AgentSubmission.created_at))
        .limit(1)
    )
    if submission is None:
        raise HTTPException(status_code=404, detail="agent source not available")

    zip_bytes = await _build_agent_source_zip(session, submission)
    if zip_bytes is None:
        raise HTTPException(status_code=404, detail="agent source not available")

    response = Response(content=zip_bytes, media_type="application/zip")
    response.headers["Content-Disposition"] = (
        f'attachment; filename="agent-{submission.agent_hash[:8]}.zip"'
    )
    _public_cache(response, max_age=300, swr=86400)
    return response


async def _collect_agent_source_files(
    session: AsyncSession,
    submission: AgentSubmission,
) -> tuple[list[AgentSourceFile], int, bool, bool]:
    """Read, cap, and redact the submission's source zip.

    Returns ``(files, total_bytes, truncated, available)``. ``available`` is
    ``False`` when the source artifact row, its manifest metadata, or the stored
    zip on disk is missing/unreadable (never raising for those cases).
    """

    try:
        artifact = await _source_artifact(session, submission)
        metadata = _artifact_metadata(artifact)
    except (ValueError, KeyError):
        return [], 0, False, False
    try:
        read_session = ArtifactReadSession.from_artifact_metadata(
            metadata,
            per_read_max_bytes=MAX_UNCOMPRESSED_BYTES,
            total_read_budget=MAX_UNCOMPRESSED_BYTES,
        )
    except (OSError, ArtifactReadError, ValueError):
        return [], 0, False, False

    files: list[AgentSourceFile] = []
    total_bytes = 0
    emitted_bytes = 0
    truncated = False
    # ``manifest.entries`` is already sorted by normalized (POSIX, traversal-safe)
    # path, so only zip-listed paths are ever enumerated.
    for entry in metadata.manifest.entries:
        if emitted_bytes >= PUBLIC_SOURCE_MAX_TOTAL_BYTES:
            truncated = True
            break
        total_bytes += entry.size
        if entry.is_binary or not entry.read_eligible:
            files.append(_binary_source_file(entry.normalized_path, entry.size))
            continue
        try:
            raw_text = read_session.read_text(entry.normalized_path)
        except ArtifactReadError:
            files.append(_binary_source_file(entry.normalized_path, entry.size))
            continue
        allowed = min(PUBLIC_SOURCE_MAX_FILE_BYTES, PUBLIC_SOURCE_MAX_TOTAL_BYTES - emitted_bytes)
        encoded = raw_text.encode("utf-8")
        file_truncated = len(encoded) > allowed
        if file_truncated:
            capped_text = encoded[:allowed].decode("utf-8", errors="ignore")
        else:
            capped_text = raw_text
        emitted_bytes += len(capped_text.encode("utf-8"))
        truncated = truncated or file_truncated
        redacted_text, redacted = redact_secrets(capped_text)
        files.append(
            AgentSourceFile(
                path=entry.normalized_path,
                size_bytes=entry.size,
                content=redacted_text,
                truncated=file_truncated,
                redacted=redacted,
                binary=False,
            )
        )
    return files, total_bytes, truncated, True


def _binary_source_file(path: str, size_bytes: int) -> AgentSourceFile:
    return AgentSourceFile(
        path=path,
        size_bytes=size_bytes,
        content=None,
        truncated=False,
        redacted=False,
        binary=True,
    )


async def _build_agent_source_zip(
    session: AsyncSession,
    submission: AgentSubmission,
) -> bytes | None:
    """Build an in-memory redacted ZIP of a submission's source, or ``None``.

    Mirrors ``_collect_agent_source_files`` for text/binary detection and
    redaction so the download stays consistent with the public source viewer:
    text entries are capped (per-file + total) and passed through
    ``redact_secrets`` before being written, binary entries are copied through
    unchanged. Every written name is the manifest's normalized, traversal-safe
    path, so the archive can never contain an entry outside itself. The total
    uncompressed cap bounds memory; once reached a ``TRUNCATED.txt`` note is
    added at the archive root and no further entries are written. Returns
    ``None`` when the artifact row, its manifest, or the stored zip is
    missing/unreadable (never raising for those cases).
    """

    try:
        artifact = await _source_artifact(session, submission)
        metadata = _artifact_metadata(artifact)
    except (ValueError, KeyError):
        return None
    try:
        read_session = ArtifactReadSession.from_artifact_metadata(
            metadata,
            per_read_max_bytes=MAX_UNCOMPRESSED_BYTES,
            total_read_budget=MAX_UNCOMPRESSED_BYTES,
        )
    except (OSError, ArtifactReadError, ValueError):
        return None

    buffer = io.BytesIO()
    emitted_bytes = 0
    truncated = False
    try:
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            # ``manifest.entries`` is sorted by normalized (POSIX, traversal-safe)
            # path, so only zip-listed paths are ever enumerated or written.
            with zipfile.ZipFile(read_session.zip_path) as source_zip:
                for entry in metadata.manifest.entries:
                    remaining = PUBLIC_SOURCE_MAX_TOTAL_BYTES - emitted_bytes
                    if remaining <= 0:
                        truncated = True
                        break
                    if entry.is_binary or not entry.read_eligible:
                        written, exceeded = _write_binary_member(
                            archive, source_zip, entry, remaining
                        )
                        if exceeded:
                            truncated = True
                            break
                        emitted_bytes += written
                        continue
                    try:
                        raw_text = read_session.read_text(entry.normalized_path)
                    except ArtifactReadError:
                        written, exceeded = _write_binary_member(
                            archive, source_zip, entry, remaining
                        )
                        if exceeded:
                            truncated = True
                            break
                        emitted_bytes += written
                        continue
                    allowed = min(PUBLIC_SOURCE_MAX_FILE_BYTES, remaining)
                    encoded = raw_text.encode("utf-8")
                    file_truncated = len(encoded) > allowed
                    if file_truncated:
                        capped_text = encoded[:allowed].decode("utf-8", errors="ignore")
                    else:
                        capped_text = raw_text
                    emitted_bytes += len(capped_text.encode("utf-8"))
                    truncated = truncated or file_truncated
                    redacted_text, _ = redact_secrets(capped_text)
                    archive.writestr(entry.normalized_path, redacted_text)
            if truncated:
                archive.writestr(
                    "TRUNCATED.txt",
                    "This archive was truncated: the agent source exceeded the "
                    f"{PUBLIC_SOURCE_MAX_TOTAL_BYTES}-byte public download cap, so some "
                    "files were shortened or omitted.\n",
                )
    except (OSError, zipfile.BadZipFile):
        return None
    return buffer.getvalue()


def _write_binary_member(
    archive: zipfile.ZipFile,
    source_zip: zipfile.ZipFile,
    entry: ZipManifestEntry,
    remaining: int,
) -> tuple[int, bool]:
    """Copy one binary member's raw bytes into ``archive`` bounded by ``remaining``.

    Returns ``(written, exceeded)``. ``written`` is the number of bytes added
    (``0`` when the member is unreadable and skipped). ``exceeded`` is ``True``
    when the member is larger than ``remaining`` (never decompressing more than
    ``remaining + 1`` bytes into memory), signalling the caller to stop and mark
    the archive truncated rather than write a partial, corrupt binary.
    """

    try:
        with source_zip.open(entry.original_path) as source:
            data = source.read(remaining + 1)
    except (KeyError, OSError, zipfile.BadZipFile):
        return 0, False
    if len(data) > remaining:
        return 0, True
    archive.writestr(entry.normalized_path, data)
    return len(data), False


@public_route(tags=["leaderboard"])
@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def leaderboard(
    session: DatabaseSession,
    response: Response,
) -> list[LeaderboardEntry]:
    """Return the latest score per miner for BASE dashboards."""

    if settings.attested_review_enabled and settings.phala_attestation_enabled:
        _public_cache(response, max_age=5, swr=30)
        return []
    result = await session.execute(
        scoring_evaluation_jobs_statement().options(
            selectinload(EvaluationJob.submission).selectinload(AgentSubmission.submission_family)
        )
    )
    best_by_hotkey: dict[str, LeaderboardEntry] = {}
    for job in result.scalars().all():
        submission = job.submission
        if not is_scoring_submission(submission):
            continue
        if submission.miner_hotkey in best_by_hotkey:
            continue
        best_by_hotkey[submission.miner_hotkey] = LeaderboardEntry(
            miner_hotkey=submission.miner_hotkey,
            submission_id=submission.id,
            name=submission.name,
            agent_hash=submission.agent_hash,
            **_version_metadata(submission),
            score=job.score,
            passed_tasks=job.passed_tasks,
            total_tasks=job.total_tasks,
        )
    _public_cache(response, max_age=5, swr=30)
    return list(best_by_hotkey.values())


async def _get_submission_or_404(
    session: AsyncSession,
    submission_id: int,
) -> AgentSubmission:
    submission = await session.get(AgentSubmission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")
    return submission


async def _get_miner_env_submission(
    session: AsyncSession,
    submission_id: int,
    auth: SignedRequestAuth,
) -> AgentSubmission:
    submission = await _get_submission_or_404(session, submission_id)
    if auth.hotkey != submission.miner_hotkey:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
    return submission


async def _get_miner_review_session(
    session: AsyncSession,
    submission_id: int,
    auth: SignedRequestAuth,
) -> ReviewSession:
    submission = await _get_miner_env_submission(session, submission_id, auth)
    review_session = await session.scalar(
        select(ReviewSession).where(ReviewSession.submission_id == submission.id)
    )
    if review_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "review_session_not_found"},
        )
    return review_session


def _review_prepare_response(
    review_session: ReviewSession,
    assignment: ReviewAssignment,
    token: str | None,
) -> ReviewPrepareResponse:
    try:
        assignment_body = json.loads(assignment.assignment_bytes)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "review_assignment_corrupt"},
        ) from exc
    return ReviewPrepareResponse(
        session_id=review_session.session_id,
        assignment_id=assignment.assignment_id,
        attempt=assignment.attempt,
        assignment=assignment_body,
        review_session_token=token,
    )


async def _authenticated_review_assignment(
    session: AsyncSession,
    assignment_id: str,
    authorization: str | None,
    *,
    revoked_status: int = status.HTTP_401_UNAUTHORIZED,
    allow_failure_replay: bool = False,
    allow_report_replay: bool = False,
) -> ReviewAssignment:
    prefix = "Bearer "
    token = (
        authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
    )
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "review_capability_invalid"},
        )
    try:
        return await authenticate_assignment_capability(
            session,
            assignment_id=assignment_id,
            token=token,
            now=datetime.now(UTC),
            allow_failure_replay=allow_failure_replay,
            allow_report_replay=allow_report_replay,
        )
    except ReviewCapabilityError as exc:
        await session.commit()
        raise HTTPException(
            status_code=revoked_status,
            detail={"code": "review_capability_invalid"},
        ) from exc


def _review_quote_verifier() -> DcapReviewQuoteVerifier:
    """Construct the production DCAP verifier, injectable by offline route tests."""

    return DcapReviewQuoteVerifier()


def _review_report_error_detail_code(exc: BaseException) -> str:
    """Map ReviewReportError message class → closed /report ``detail.code``.

    Guest residual (sub26/27) collapsed solely to ``diag=http_status`` when
    service returned generic ``review_report_invalid`` on every trust refuse.
    Message-word classification only — never re-emits free-form ``str(exc)``
    into the HTTP body, digests, or secrets. Default is
    ``review_report_invalid``.
    """

    text = str(exc).lower()
    # timeline / attested-order / post-receipt residual subclasses
    if any(
        token in text
        for token in (
            "timeline",
            "timestamps are not",
            "times are invalid",
            "receipt precedes",
            "post-receipt",
            "report finished time",
            "future or post",
        )
    ):
        return "report_timeline_invalid"
    # evidence crypto misconfig (missing CHALLENGE_REVIEW_EVIDENCE_ENCRYPTION_KEY)
    # must be checked before the generic evidence token, so residual can map
    # review_evidence_crypto_unavailable → guest diag=evidence (not bare 500).
    if any(
        token in text
        for token in (
            "encryption key",
            "evidence encryption",
            "crypto unavailable",
        )
    ):
        return "review_evidence_crypto_unavailable"
    # evidence refuse subclasses
    if "evidence" in text:
        return "review_evidence_invalid"
    # measurement / quote binding refuse
    if any(
        token in text
        for token in (
            "measurement",
            "quote measurement",
            "allowlist",
            "key provider event",
        )
    ):
        return "review_measurement_mismatch"
    return "review_report_invalid"


def _review_evidence_error_detail_code(exc: BaseException) -> str:
    """Map ReviewEvidenceError → closed /report ``detail.code`` (never secrets).

    Prefer ``review_evidence_crypto_unavailable`` when the Fernet key is missing
    or limp, else generic ``review_evidence_invalid``. Guest mapper already treats
    any ``evidence`` token as ``diag=evidence`` without guest rebuild.
    """

    text = str(exc).lower()
    if any(
        token in text
        for token in (
            "encryption key",
            "evidence encryption",
            "unavailable",
            "not configured",
        )
    ):
        return "review_evidence_crypto_unavailable"
    return "review_evidence_invalid"


def _require_review_json_media(request: Request) -> None:
    """Fail before parsing unless direct review mutations carry JSON bytes."""

    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(status_code=400, detail={"code": "review_json_media_required"})


def _parse_evidence_range(
    value: str | None,
    total_size: int,
    *,
    max_range_bytes: int | None = None,
) -> tuple[int, int]:
    """Parse only one explicit bounded byte range, never a multi-range body."""

    if total_size < 1:
        raise ValueError("evidence object is empty")
    if value is None:
        end = total_size - 1
        if max_range_bytes is not None and total_size > max_range_bytes:
            end = max_range_bytes - 1
        return 0, end
    match = re.fullmatch(r"bytes=(\d+)-(\d*)", value)
    if match is None:
        raise ValueError("range must contain one explicit byte interval")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else total_size - 1
    if start > end or start >= total_size:
        raise ValueError("range is unsatisfiable")
    end = min(end, total_size - 1)
    if max_range_bytes is not None and (end - start + 1) > max_range_bytes:
        raise ValueError("range exceeds configured max")
    return start, end


def _decode_review_evidence(value: Mapping[str, str]) -> dict[str, bytes]:
    """Decode the only report-side raw bytes, before they reach encrypted storage."""

    expected = {
        "planned_request_b64": "planned_request",
        "transport_observation_b64": "transport_observation",
        "request_body_b64": "request_body",
        "response_body_b64": "response_body",
        "metadata_b64": "metadata",
    }
    if not set(value) <= set(expected):
        raise ValueError("review evidence field is invalid")
    decoded: dict[str, bytes] = {}
    for wire_name, object_name in expected.items():
        encoded = value.get(wire_name)
        if encoded is None:
            continue
        if not isinstance(encoded, str):
            raise ValueError("review evidence encoding is invalid")
        try:
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError("review evidence encoding is invalid") from exc
        if not raw:
            raise ValueError("review evidence object is empty")
        decoded[object_name] = raw
    return decoded


async def _review_has_public_projection(
    session: AsyncSession,
    review_session: ReviewSession,
) -> bool:
    return (
        await session.scalar(
            select(ReviewAssignment.id)
            .where(ReviewAssignment.session_id == review_session.id)
            .where(ReviewAssignment.review_public_projection_json.is_not(None))
            .limit(1)
        )
    ) is not None


def _enforce_audit_response_size(value: Mapping[str, object], *, maximum: int) -> None:
    if len(canonical_json_v1(dict(value))) > maximum:
        raise HTTPException(status_code=413, detail={"code": "review_report_too_large"})


async def _submission_env_vars(
    session: AsyncSession,
    submission_id: int,
) -> list[SubmissionEnvVar]:
    result = await session.execute(
        select(SubmissionEnvVar)
        .where(SubmissionEnvVar.submission_id == submission_id)
        .order_by(SubmissionEnvVar.key)
    )
    return list(result.scalars().all())


def _ensure_miner_env_editable(submission: AgentSubmission) -> None:
    if submission.raw_status != "waiting_miner_env" or submission.env_locked_at is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="submission env is locked")


def _validated_miner_env(env: Mapping[str, object]) -> dict[str, str]:
    """Admit miner env keys/tokens only; reject URL/proxy/host injection."""
    try:
        return validate_miner_env(env)
    except MinerEnvValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.reason) from exc


def _miner_env_metadata_response(
    submission: AgentSubmission,
    env_vars: list[SubmissionEnvVar],
) -> MinerEnvMetadataResponse:
    return MinerEnvMetadataResponse(
        submission_id=submission.id,
        keys=[env_var.key for env_var in env_vars],
        count=len(env_vars),
        updated_at=_miner_env_updated_at(submission, env_vars),
        locked=submission.env_locked_at is not None,
        env_confirmed_empty=submission.env_confirmed_empty,
        env_confirmed_empty_at=submission.env_confirmed_empty_at,
        confirmation_state=_miner_env_confirmation_state(submission, env_vars),
    )


def _miner_env_confirmation_state(
    submission: AgentSubmission,
    env_vars: list[SubmissionEnvVar],
) -> Literal["pending", "env_vars_present", "empty_confirmed"]:
    if env_vars:
        return "env_vars_present"
    if submission.env_confirmed_empty:
        return "empty_confirmed"
    return "pending"


def _append_owner_audit(
    *,
    session: AsyncSession,
    submission: AgentSubmission,
    auth: SignedRequestAuth,
    action: str,
    reason: str,
    before_status: str | None,
    after_status: str | None,
) -> OwnerActionAudit:
    audit = OwnerActionAudit(
        submission_id=submission.id,
        owner_hotkey=auth.hotkey,
        action=action,
        reason=reason,
        request_hash=auth.body_sha256,
        nonce=auth.nonce,
        signature=auth.signature,
        request_timestamp=auth.timestamp,
        before_effective_status=before_status,
        after_effective_status=after_status,
    )
    session.add(audit)
    return audit


def _append_admin_review_decision(
    *,
    session: AsyncSession,
    submission: AgentSubmission,
    auth: SignedRequestAuth,
    decision: str,
    reason: str,
    before_status: str | None,
    previous_analysis: AnalysisRun | None,
) -> AdminReviewDecision:
    row = AdminReviewDecision(
        submission_id=submission.id,
        reviewer_hotkey=auth.hotkey,
        decision=decision,
        reason=reason,
        request_hash=auth.body_sha256,
        before_effective_status=before_status,
        after_effective_status=None,
        metadata_json=json.dumps(
            {
                "analysis_run_id": previous_analysis.id if previous_analysis is not None else None,
                "nonce": auth.nonce,
                "previous_status": submission.raw_status,
                "previous_verdict": previous_analysis.verdict
                if previous_analysis is not None
                else None,
                "request_timestamp": auth.timestamp,
                "signature": auth.signature,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    session.add(row)
    return row


def _admin_decision_status_metadata(
    decision: AdminReviewDecision,
    previous_analysis: AnalysisRun | None,
) -> dict[str, object]:
    return {
        "admin_decision_id": decision.id,
        "analysis_run_id": previous_analysis.id if previous_analysis is not None else None,
        "previous_verdict": previous_analysis.verdict if previous_analysis is not None else None,
    }


def _required_reason(reason: str) -> str:
    stripped = reason.strip()
    if not stripped:
        raise HTTPException(status_code=422, detail="reason is required")
    return stripped


@dataclass(frozen=True)
class _SubmissionAnalysisSummary:
    has_analysis: bool
    analyzer_status: str | None
    analyzer_verdict: str | None
    llm_verdict: str | None
    llm_confidence: float | None
    similarity_max_score_percent: float | None
    similarity_match_count: int
    ast_feature_count: int


_EMPTY_ANALYSIS_SUMMARY = _SubmissionAnalysisSummary(
    has_analysis=False,
    analyzer_status=None,
    analyzer_verdict=None,
    llm_verdict=None,
    llm_confidence=None,
    similarity_max_score_percent=None,
    similarity_match_count=0,
    ast_feature_count=0,
)


async def _submission_analysis_summaries(
    session: AsyncSession,
    submission_ids: list[int],
) -> dict[int, _SubmissionAnalysisSummary]:
    if not submission_ids:
        return {}
    # max(id) selects the latest-per-group row portably across Postgres and SQLite.
    latest_run_ids = (
        select(func.max(AnalysisRun.id))
        .where(AnalysisRun.submission_id.in_(submission_ids))
        .group_by(AnalysisRun.submission_id)
    )
    analysis_rows = (
        await session.execute(
            select(
                AnalysisRun.id,
                AnalysisRun.submission_id,
                AnalysisRun.status,
                AnalysisRun.verdict,
            ).where(AnalysisRun.id.in_(latest_run_ids))
        )
    ).all()
    if not analysis_rows:
        return {}
    run_ids = [row.id for row in analysis_rows]

    latest_llm_ids = (
        select(func.max(LlmVerdict.id))
        .where(LlmVerdict.analysis_run_id.in_(run_ids))
        .group_by(LlmVerdict.analysis_run_id)
    )
    llm_rows = (
        await session.execute(
            select(
                LlmVerdict.analysis_run_id,
                LlmVerdict.verdict,
                LlmVerdict.confidence,
            ).where(LlmVerdict.id.in_(latest_llm_ids))
        )
    ).all()
    llm_by_run = {row.analysis_run_id: row for row in llm_rows}

    similarity_rows = (
        await session.execute(
            select(
                SimilarityMatch.analysis_run_id,
                func.max(SimilarityMatch.score),
                func.count(SimilarityMatch.id),
            )
            .where(SimilarityMatch.analysis_run_id.in_(run_ids))
            .group_by(SimilarityMatch.analysis_run_id)
        )
    ).all()
    similarity_by_run = {row[0]: (row[1], row[2]) for row in similarity_rows}

    ast_rows = (
        await session.execute(
            select(
                PythonAstFeature.analysis_run_id,
                func.count(PythonAstFeature.id),
            )
            .where(PythonAstFeature.analysis_run_id.in_(run_ids))
            .group_by(PythonAstFeature.analysis_run_id)
        )
    ).all()
    ast_by_run = {row[0]: row[1] for row in ast_rows}

    summaries: dict[int, _SubmissionAnalysisSummary] = {}
    for row in analysis_rows:
        llm = llm_by_run.get(row.id)
        max_score, match_count = similarity_by_run.get(row.id, (None, 0))
        summaries[row.submission_id] = _SubmissionAnalysisSummary(
            has_analysis=True,
            analyzer_status=row.status,
            analyzer_verdict=row.verdict,
            llm_verdict=llm.verdict if llm is not None else None,
            llm_confidence=llm.confidence if llm is not None else None,
            similarity_max_score_percent=max_score,
            similarity_match_count=match_count,
            ast_feature_count=ast_by_run.get(row.id, 0),
        )
    return summaries


def _submission_list_item(
    submission: AgentSubmission,
    summary: _SubmissionAnalysisSummary | None = None,
) -> SubmissionListItem:
    latest = (
        None
        if settings.attested_review_enabled and settings.phala_attestation_enabled
        else _latest_submission_job(submission)
    )
    env_vars = _loaded_submission_env_vars(submission)
    analysis = summary if summary is not None else _EMPTY_ANALYSIS_SUMMARY
    return SubmissionListItem(
        id=submission.id,
        miner_hotkey=submission.miner_hotkey,
        name=submission.name,
        agent_hash=submission.agent_hash,
        zip_sha256=submission.zip_sha256,
        **_version_metadata(submission),
        status=submission.effective_status,
        effective_status=submission.effective_status,
        **_public_env_action_metadata(submission, env_vars),
        score=latest.score if latest else 0.0,
        submitted_at=submission.submitted_at,
        created_at=submission.created_at,
        latest_evaluation=_evaluation_summary_response(latest) if latest else None,
        has_analysis=analysis.has_analysis,
        analyzer_status=analysis.analyzer_status,
        analyzer_verdict=analysis.analyzer_verdict,
        llm_verdict=analysis.llm_verdict,
        llm_confidence=analysis.llm_confidence,
        similarity_max_score_percent=analysis.similarity_max_score_percent,
        similarity_match_count=analysis.similarity_match_count,
        ast_feature_count=analysis.ast_feature_count,
    )


def _submission_version_item(submission: AgentSubmission) -> SubmissionVersionItem:
    latest = (
        None
        if settings.attested_review_enabled and settings.phala_attestation_enabled
        else _latest_submission_job(submission)
    )
    return SubmissionVersionItem(
        id=submission.id,
        name=submission.name,
        agent_hash=submission.agent_hash,
        zip_sha256=submission.zip_sha256,
        **_version_metadata(submission),
        status=submission.effective_status,
        effective_status=submission.effective_status,
        score=latest.score if latest else 0.0,
        submitted_at=submission.submitted_at,
        created_at=submission.created_at,
        latest_evaluation=_evaluation_summary_response(latest) if latest else None,
    )


def _version_metadata(submission: AgentSubmission) -> dict[str, object]:
    family = submission.submission_family
    return {
        "display_name": family.display_name if family is not None else submission.name,
        "family_id": family.public_family_id if family is not None else None,
        "version_number": submission.version_number,
        "version_label": submission.version_label,
        "version_count": family.version_count if family is not None else None,
        "is_latest_version": submission.is_latest_version,
        "latest_submission_id": family.latest_submission_id if family is not None else None,
    }


async def _submission_status_response(
    session: AsyncSession,
    submission: AgentSubmission,
) -> SubmissionStatusResponse:
    latest_event = await _latest_status_event(session, submission.id)
    raw_status = latest_event.to_status if latest_event is not None else submission.raw_status
    public_state = public_status_for(raw_status)
    analysis = await _latest_analysis_run(session, submission.id)
    llm = await _latest_llm_verdict(session, analysis.id) if analysis is not None else None
    matches = await _similarity_matches(session, analysis.id) if analysis is not None else []
    ast_features = await _python_ast_features(session, analysis.id) if analysis is not None else []
    job = (
        None
        if settings.attested_review_enabled and settings.phala_attestation_enabled
        else await _latest_evaluation_job_for_submission(session, submission.id)
    )
    attempt = (
        None
        if settings.attested_review_enabled and settings.phala_attestation_enabled
        else await _latest_evaluation_attempt(session, submission.id)
    )
    task_phases = await _latest_task_phases_for_job(
        session,
        submission_id=submission.id,
        job_id=job.id if job is not None else None,
    )
    task_results = await _task_results_for_job(session, job.id if job is not None else None)
    task_rows = _task_rows_response(job, task_phases, task_results)
    trial_counts = await _terminal_bench_trial_counts(session, submission.id)
    env_vars = await _submission_env_vars(session, submission.id)
    review_enabled = settings.attested_review_enabled and settings.phala_attestation_enabled
    # Fully legacy mode must not query review tables or emit a review field.
    review = await _review_status_response(session, submission) if review_enabled else None
    latest_run: EvalRun | None = None
    if review_enabled:
        eval_history = await eval_status_page(
            session,
            submission,
            cursor_secret=settings.shared_token,
        )
        latest_eval = eval_history["items"][-1] if eval_history["items"] else None
        # Dual-flag EvaluationJob is forced None; project task_rows + ledger from
        # the latest EvalRun plan / score record only (VAL-DFROWS-001..005).
        latest_run = await _latest_eval_run_for_submission(session, submission.id)
    else:
        latest_eval = None
    if latest_eval is not None:
        if latest_run is not None:
            task_rows = _task_rows_from_eval_run(latest_run)
            score = float(latest_run.score) if latest_run.score is not None else 0.0
            passed_tasks = (
                int(latest_run.passed_tasks) if latest_run.passed_tasks is not None else 0
            )
            total_tasks = int(latest_run.total_tasks) if latest_run.total_tasks is not None else 0
        else:
            score = 0.0
            passed_tasks = 0
            total_tasks = 0
        eval_status = EvaluationStatusResponse(
            job_id=latest_eval["eval_run_id"],
            status=latest_eval["phase"],
            score=score,
            passed_tasks=passed_tasks,
            total_tasks=total_tasks,
            verdict="verified" if latest_eval["verified"] else None,
            reason_codes=[latest_eval["reason_code"]] if latest_eval["reason_code"] else [],
            current_attempt=None,
            attempt_status=latest_eval["phase"],
            task_phases=task_phases,
            task_rows=task_rows,
        )
    else:
        eval_status = _evaluation_status_response(job, attempt, task_phases, task_rows)

    return SubmissionStatusResponse(
        submission_id=submission.id,
        agent_hash=submission.agent_hash,
        name=submission.name,
        **_version_metadata(submission),
        status=public_state,
        public_state=public_state,
        phase=_public_phase(raw_status),
        effective_status=public_status_for(submission.effective_status),
        **_public_env_action_metadata(submission, env_vars),
        last_event_id=latest_event.id if latest_event is not None else None,
        last_event_sequence=latest_event.sequence if latest_event is not None else None,
        current_attempt=attempt.attempt_number if attempt is not None else None,
        analyzer=_analyzer_status_response(raw_status, analysis, llm),
        similarity=_similarity_status_response(matches),
        ast=_ast_status_response(ast_features, analysis),
        rules_check=_rules_check_response(analysis),
        review=review,
        evaluation=eval_status,
        terminal_bench=TerminalBenchStatusResponse(**trial_counts),
        progress=SubmissionProgressCountsResponse(
            status_events=await _count_rows(session, SubmissionStatusEvent, submission.id),
            analysis_runs=await _count_rows(session, AnalysisRun, submission.id),
            similarity_matches=len(matches),
            llm_verdicts=1 if llm is not None else 0,
            evaluation_jobs=(
                0 if review_enabled else await _count_rows(session, EvaluationJob, submission.id)
            ),
            evaluation_attempts=(
                0
                if review_enabled
                else await _count_rows(session, EvaluationAttempt, submission.id)
            ),
            terminal_bench_trials=trial_counts["total_trials"],
        ),
        submitted_at=submission.submitted_at,
        updated_at=_latest_timestamp(latest_event, analysis, job, attempt),
    )


_TERMINAL_REVIEW_PHASES = frozenset(
    {
        "review_allowed",
        "review_rejected",
        "review_escalated",
        "review_expired",
        "review_cancelled",
        "review_error",
    }
)
_RETRYABLE_REVIEW_PHASES = frozenset({"review_expired", "review_cancelled", "review_error"})


async def _review_status_response(
    session: AsyncSession,
    submission: AgentSubmission,
) -> ReviewStatusResponse:
    review_session = await session.scalar(
        select(ReviewSession).where(ReviewSession.submission_id == submission.id).limit(1)
    )
    if review_session is None or review_session.current_assignment_id is None:
        return ReviewStatusResponse(
            session_id=None,
            assignment_id=None,
            attempt=None,
            phase=None,
            terminal=False,
            verdict=None,
            verified=False,
            retryable=False,
            reason_code=None,
            report_available=False,
            issued_at=None,
            finished_at=None,
        )
    assignment = await session.scalar(
        select(ReviewAssignment)
        .where(ReviewAssignment.session_id == review_session.id)
        .where(ReviewAssignment.assignment_id == review_session.current_assignment_id)
        .limit(1)
    )
    if assignment is None:
        return ReviewStatusResponse(
            session_id=review_session.session_id,
            assignment_id=None,
            attempt=None,
            phase=None,
            terminal=False,
            verdict=None,
            verified=False,
            retryable=False,
            reason_code=None,
            report_available=False,
            issued_at=None,
            finished_at=None,
        )
    # Lazy-expire sticky active rows past expires_at so public status is honest.
    await expire_assignment_if_needed(session, assignment, now=datetime.now(UTC))
    outcome = _json_object(assignment.review_verification_outcome_json or "{}")
    outcome_status = outcome.get("status")
    verdict = {
        "verified_allow": "allow",
        "verified_reject": "reject",
        "verified_escalate": "escalate",
    }.get(outcome_status)
    verified = outcome_status in {"verified_allow", "verified_reject", "verified_escalate"}
    return ReviewStatusResponse(
        session_id=review_session.session_id,
        assignment_id=assignment.assignment_id,
        attempt=assignment.attempt,
        phase=assignment.phase,
        terminal=assignment.phase in _TERMINAL_REVIEW_PHASES,
        verdict=verdict,
        verified=verified,
        retryable=(
            bool(outcome.get("retryable"))
            if outcome_status is not None
            else assignment.phase in _RETRYABLE_REVIEW_PHASES
        ),
        reason_code=assignment.reason_code,
        report_available=assignment.review_public_projection_json is not None,
        issued_at=assignment.issued_at,
        finished_at=assignment.finished_at,
    )


def _assignment_public_tee_qualifies(assignment: ReviewAssignment | None) -> bool:
    """True when durable assignment material may back public available:true math.

    Aligns with status ``report_available`` (projection present) and dual-flag
    verified_* outcomes. Envelope alone or verifier_unavailable does not qualify.
    """

    if assignment is None:
        return False
    return public_tee_assignment_qualifies(
        envelope_json=assignment.review_report_envelope_json,
        outcome_json=assignment.review_verification_outcome_json,
        public_projection_json=assignment.review_public_projection_json,
    )


async def _public_tee_math_response(
    session: AsyncSession,
    submission: AgentSubmission,
) -> dict[str, Any]:
    """Load authorizing/current assignment envelope and project safe TEE math.

    Prefer the session's authorizing assignment (verified allow) when present;
    otherwise fall back to the current assignment when it carries a durable
    *verified* report (projection and/or verified_* outcome). Locked closed form
    when nothing durable-and-verified exists. Never 500 on builder ValueError.
    """

    review_session = await session.scalar(
        select(ReviewSession).where(ReviewSession.submission_id == submission.id).limit(1)
    )
    if review_session is None:
        return public_tee_unavailable()

    assignment: ReviewAssignment | None = None
    preferred_ids: list[str] = []
    if review_session.authorizing_assignment_id:
        preferred_ids.append(review_session.authorizing_assignment_id)
    if (
        review_session.current_assignment_id
        and review_session.current_assignment_id not in preferred_ids
    ):
        preferred_ids.append(review_session.current_assignment_id)

    for assignment_id in preferred_ids:
        candidate = await session.scalar(
            select(ReviewAssignment)
            .where(ReviewAssignment.session_id == review_session.id)
            .where(ReviewAssignment.assignment_id == assignment_id)
            .limit(1)
        )
        if _assignment_public_tee_qualifies(candidate):
            assignment = candidate
            break

    if assignment is None:
        # Last resort: any durable *verified* envelope on the session (e.g.
        # verified reject with public projection but no authorizing_assignment_id).
        # Gate the same way as preferred-id path: envelope alone is insufficient.
        candidates = (
            await session.scalars(
                select(ReviewAssignment)
                .where(ReviewAssignment.session_id == review_session.id)
                .where(ReviewAssignment.review_report_envelope_json.is_not(None))
                .order_by(desc(ReviewAssignment.id))
            )
        ).all()
        for candidate in candidates:
            if _assignment_public_tee_qualifies(candidate):
                assignment = candidate
                break

    if assignment is None or not _assignment_public_tee_qualifies(assignment):
        return public_tee_unavailable()

    try:
        return build_public_tee_math_from_assignment(
            submission_id=submission.id,
            envelope_json=assignment.review_report_envelope_json,
            outcome_json=assignment.review_verification_outcome_json,
            public_projection_json=assignment.review_public_projection_json,
        )
    except ValueError:
        # Builder fail-closed: never leak a public 500 from deny-list / schema
        # asserts. Locked closed form is the stable public contract.
        return public_tee_unavailable()


def _loaded_submission_env_vars(submission: AgentSubmission) -> list[SubmissionEnvVar]:
    return list(submission.__dict__.get("env_vars") or [])


def _public_env_action_metadata(
    submission: AgentSubmission,
    env_vars: list[SubmissionEnvVar],
) -> dict[str, object]:
    if submission.raw_status != "waiting_miner_env":
        return {
            "env_action_required": False,
            "env_keys": [],
            "env_var_count": 0,
            "env_confirmed_empty": False,
            "env_locked": False,
            "env_updated_at": None,
        }
    return {
        "env_action_required": submission.env_locked_at is None,
        "env_keys": [env_var.key for env_var in env_vars],
        "env_var_count": len(env_vars),
        "env_confirmed_empty": submission.env_confirmed_empty,
        "env_locked": submission.env_locked_at is not None,
        "env_updated_at": _miner_env_updated_at(submission, env_vars),
    }


def _miner_env_updated_at(
    submission: AgentSubmission,
    env_vars: list[SubmissionEnvVar],
) -> datetime | None:
    updated_values = [env_var.updated_at for env_var in env_vars]
    if submission.env_confirmed_empty_at is not None:
        updated_values.append(submission.env_confirmed_empty_at)
    return max(updated_values) if updated_values else None


async def _latest_status_event(
    session: AsyncSession,
    submission_id: int,
) -> SubmissionStatusEvent | None:
    return (
        await session.execute(
            select(SubmissionStatusEvent)
            .where(SubmissionStatusEvent.submission_id == submission_id)
            .order_by(desc(SubmissionStatusEvent.sequence), desc(SubmissionStatusEvent.id))
            .limit(1)
        )
    ).scalar_one_or_none()


def _parse_task_event_cursor(raw_cursor: str | None) -> int:
    if raw_cursor is None or raw_cursor == "":
        return 0
    try:
        cursor = int(raw_cursor)
    except ValueError as exc:
        raise _invalid_task_event_cursor() from exc
    if cursor < 0:
        raise _invalid_task_event_cursor()
    return cursor


def _task_event_stream_cursor(raw_cursor: str | None, request: Request) -> int:
    if raw_cursor is not None:
        return _parse_task_event_cursor(raw_cursor)
    return _parse_task_event_cursor(request.headers.get("last-event-id"))


def _invalid_task_event_cursor(max_sequence: int | None = None) -> HTTPException:
    detail: dict[str, object] = {
        "code": "task_event_cursor_invalid",
        "message": "cursor must be an integer between 0 and the current max task event sequence",
    }
    if max_sequence is not None:
        detail["max_sequence"] = max_sequence
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


async def _max_task_event_sequence(session: AsyncSession, submission_id: int) -> int:
    value = await session.scalar(
        select(func.max(TaskLogEvent.sequence)).where(TaskLogEvent.submission_id == submission_id)
    )
    return int(value or 0)


async def _task_events_after_cursor(
    session: AsyncSession,
    *,
    submission_id: int,
    cursor: int,
    limit: int,
    task_id: str | None,
    event_type: str | None,
    stream: str | None = None,
) -> list[TaskLogEvent]:
    statement = (
        select(TaskLogEvent)
        .where(TaskLogEvent.submission_id == submission_id)
        .where(TaskLogEvent.sequence > cursor)
        .options(selectinload(TaskLogEvent.job))
    )
    if task_id is not None:
        statement = statement.where(TaskLogEvent.task_id == task_id)
    if event_type is not None:
        statement = statement.where(TaskLogEvent.event_type == event_type)
    if stream is not None:
        statement = statement.where(TaskLogEvent.stream == stream)
    result = await session.execute(
        statement.order_by(TaskLogEvent.sequence, TaskLogEvent.id).limit(limit + 1)
    )
    return list(result.scalars().all())


def _task_event_replay_item(event: TaskLogEvent) -> TaskEventReplayItem:
    return TaskEventReplayItem(
        id=event.id,
        sequence=event.sequence,
        submission_id=event.submission_id,
        job_id=event.job.job_id if event.job is not None else None,
        task_id=event.task_id,
        event_type=event.event_type,
        stream=event.stream,
        message=_public_task_event_text(event.message),
        progress=event.progress,
        status=event.status,
        truncated=event.truncated,
        cap_reached=event.cap_reached,
        metadata=_public_task_event_metadata(_json_object(event.metadata_json)),
        created_at=event.created_at,
    )


def _public_task_event_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    public: dict[str, object] = {}
    for key, value in metadata.items():
        if _is_sensitive_task_event_metadata_key(key):
            continue
        public[str(key)] = _public_task_event_metadata_value(value)
    return public


def _public_task_event_metadata_value(value: object) -> object:
    if isinstance(value, str):
        return _public_task_event_text(value)
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, Mapping):
        return _public_task_event_metadata(value)
    if isinstance(value, list):
        return [_public_task_event_metadata_value(item) for item in value]
    return _public_task_event_text(str(value))


def _is_sensitive_task_event_metadata_key(key: str) -> bool:
    normalized = key.lower()
    return normalized in SENSITIVE_METADATA_KEYS or any(
        marker in normalized
        for marker in ("api_key", "secret", "signature", "token", "_ref", "path")
    )


def _public_task_event_text(value: str) -> str:
    sanitized = PRIVATE_PATH_RE.sub("[REDACTED_PATH]", redact_task_event_message(value))
    sanitized = re.sub(r"\b(?:platform_sdk|base_sdk)\b", "base", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(
        r"\bagent_challenge_runner\.[A-Za-z0-9_.]+",
        "[REDACTED_INTERNAL]",
        sanitized,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]*(?:secret|token|raw-ref|broker-ref|pod-)[A-Za-z0-9_.-]*(?![A-Za-z0-9_.-])",
        "[REDACTED_SECRET]",
        sanitized,
        flags=re.IGNORECASE,
    )


async def _submission_task_event_stream(
    submission_id: int,
    version_label: str | None,
    cursor: int,
    stream: str | None = None,
) -> AsyncIterator[str]:
    last_sent_sequence = cursor
    last_heartbeat_at = asyncio.get_running_loop().time()
    while True:
        # A fresh short-lived session per poll: the pooled connection (and, on
        # SQLite, its read transaction / WAL snapshot) is released between polls
        # instead of being pinned for the whole stream. ``async with`` also closes
        # the session cleanly on client disconnect / CancelledError. Events are
        # rendered to strings inside the session (they eager-load ``job``), then
        # yielded after it closes so the connection is not held during network IO.
        async with database.session() as poll_session:
            events = await _task_events_after_cursor(
                poll_session,
                submission_id=submission_id,
                cursor=last_sent_sequence,
                limit=MAX_TASK_EVENT_REPLAY_LIMIT,
                task_id=None,
                event_type=None,
                stream=stream,
            )
            rendered = [
                (event.sequence, _format_task_event_sse(event, version_label))
                for event in events[:MAX_TASK_EVENT_REPLAY_LIMIT]
            ]
            terminal_sequence = next(
                (
                    event.sequence
                    for event in events[:MAX_TASK_EVENT_REPLAY_LIMIT]
                    if _is_terminal_task_event(event.event_type)
                ),
                None,
            )
            stream_complete = not events and await _submission_task_stream_is_complete(
                poll_session,
                submission_id,
            )

        for sequence, payload in rendered:
            last_sent_sequence = sequence
            yield payload
            if sequence == terminal_sequence:
                return
        if stream_complete:
            return
        if rendered:
            last_heartbeat_at = asyncio.get_running_loop().time()
            continue

        now = asyncio.get_running_loop().time()
        if now - last_heartbeat_at >= SSE_HEARTBEAT_SECONDS:
            yield ": heartbeat\n\n"
            last_heartbeat_at = now
        await asyncio.sleep(SSE_POLL_SECONDS)


def _format_task_event_sse(event: TaskLogEvent, version_label: str | None) -> str:
    payload = _task_event_replay_item(event).model_dump(mode="json")
    payload["id"] = event.sequence
    payload["version_label"] = version_label
    return (
        f"id: {event.sequence}\n"
        f"event: {event.event_type}\n"
        f"data: {json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n\n"
    )


def _is_terminal_task_event(event_type: str) -> bool:
    return event_type in TASK_EVENT_TERMINAL_TYPES


async def _submission_task_stream_is_complete(
    session: AsyncSession,
    submission_id: int,
) -> bool:
    event_type = await session.scalar(
        select(TaskLogEvent.event_type)
        .where(TaskLogEvent.submission_id == submission_id)
        .order_by(TaskLogEvent.sequence.desc())
        .limit(1)
    )
    return _is_terminal_task_event(event_type) if event_type is not None else False


async def _submission_event_stream(
    submission_id: int,
    last_event_id: int | None,
) -> AsyncIterator[str]:
    last_sent_id = last_event_id or 0
    last_heartbeat_at = asyncio.get_running_loop().time()
    while True:
        # Fresh short-lived session per poll so the pooled connection is released
        # between polls; ``async with`` also closes it on client disconnect.
        async with database.session() as poll_session:
            events = await _status_events_after_id(poll_session, submission_id, last_sent_id)
            rendered = [
                (event.id, _format_sse_event(event), _is_terminal_status(event.to_status))
                for event in events
            ]
            stream_complete = not events and await _last_sent_status_is_terminal(
                poll_session, submission_id, last_sent_id
            )

        for event_id, payload, is_terminal in rendered:
            last_sent_id = event_id
            yield payload
            if is_terminal:
                return
        if stream_complete:
            return
        if rendered:
            last_heartbeat_at = asyncio.get_running_loop().time()
            continue

        now = asyncio.get_running_loop().time()
        if now - last_heartbeat_at >= SSE_HEARTBEAT_SECONDS:
            yield ": heartbeat\n\n"
            last_heartbeat_at = now
        await asyncio.sleep(SSE_POLL_SECONDS)


async def _status_events_after_id(
    session: AsyncSession,
    submission_id: int,
    event_id: int,
) -> list[SubmissionStatusEvent]:
    return (
        (
            await session.execute(
                select(SubmissionStatusEvent)
                .where(SubmissionStatusEvent.submission_id == submission_id)
                .where(SubmissionStatusEvent.id > event_id)
                .order_by(SubmissionStatusEvent.sequence, SubmissionStatusEvent.id)
            )
        )
        .scalars()
        .all()
    )


async def _last_sent_status_is_terminal(
    session: AsyncSession,
    submission_id: int,
    event_id: int,
) -> bool:
    if event_id <= 0:
        return False
    raw_status = await session.scalar(
        select(SubmissionStatusEvent.to_status)
        .where(SubmissionStatusEvent.submission_id == submission_id)
        .where(SubmissionStatusEvent.id == event_id)
        .limit(1)
    )
    return _is_terminal_status(raw_status) if raw_status is not None else False


async def _first_status_event_id(session: AsyncSession, submission_id: int) -> int | None:
    return await session.scalar(
        select(SubmissionStatusEvent.id)
        .where(SubmissionStatusEvent.submission_id == submission_id)
        .order_by(SubmissionStatusEvent.sequence, SubmissionStatusEvent.id)
        .limit(1)
    )


async def _status_event_id_exists(
    session: AsyncSession,
    submission_id: int,
    event_id: int,
) -> bool:
    value = await session.scalar(
        select(SubmissionStatusEvent.id)
        .where(SubmissionStatusEvent.submission_id == submission_id)
        .where(SubmissionStatusEvent.id == event_id)
        .limit(1)
    )
    return value is not None


def _last_event_id_header(request: Request) -> int | None:
    raw = request.headers.get("last-event-id")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid Last-Event-ID") from exc


def _format_sse_event(event: SubmissionStatusEvent) -> str:
    public_state = public_status_for(event.to_status)
    data = {
        "id": event.id,
        "sequence": event.sequence,
        "submission_id": event.submission_id,
        "status": public_state,
        "public_state": public_state,
        "phase": _public_phase(event.to_status),
        "created_at": event.created_at.isoformat(),
    }
    if event.reason in PUBLIC_SSE_REASON_CODES:
        data["reason_code"] = event.reason
    if event.actor in {"api", "analysis", "worker", "evaluation", "review-cvm"}:
        data["actor"] = event.actor
    if event.to_status.startswith("review_"):
        metadata = _json_object(event.metadata_json)
        review = metadata.get("review")
        if isinstance(review, dict):
            data["review"] = review
    return (
        f"id: {event.id}\n"
        "event: submission.status\n"
        f"data: {json.dumps(data, sort_keys=True, separators=(',', ':'))}\n\n"
    )


def _is_terminal_status(raw_status: str) -> bool:
    return raw_status in {
        "review_allowed",
        "review_rejected",
        "review_escalated",
        "review_expired",
        "review_cancelled",
        "review_error",
        "analysis_rejected",
        "tb_completed",
        "tb_failed_final",
        "cancelled",
        "valid",
        "invalid",
        "error",
        "completed",
    }


async def _latest_analysis_run(session: AsyncSession, submission_id: int) -> AnalysisRun | None:
    return (
        await session.execute(
            select(AnalysisRun)
            .where(AnalysisRun.submission_id == submission_id)
            .order_by(desc(AnalysisRun.created_at), desc(AnalysisRun.id))
            .limit(1)
        )
    ).scalar_one_or_none()


async def _latest_llm_verdict(session: AsyncSession, analysis_run_id: int) -> LlmVerdict | None:
    return (
        await session.execute(
            select(LlmVerdict)
            .where(LlmVerdict.analysis_run_id == analysis_run_id)
            .order_by(desc(LlmVerdict.created_at), desc(LlmVerdict.id))
            .limit(1)
        )
    ).scalar_one_or_none()


async def _similarity_matches(
    session: AsyncSession,
    analysis_run_id: int,
) -> list[SimilarityMatch]:
    return (
        (
            await session.execute(
                select(SimilarityMatch)
                .where(SimilarityMatch.analysis_run_id == analysis_run_id)
                .order_by(desc(SimilarityMatch.score), desc(SimilarityMatch.id))
            )
        )
        .scalars()
        .all()
    )


async def _python_ast_features(
    session: AsyncSession,
    analysis_run_id: int,
) -> list[PythonAstFeature]:
    return (
        (
            await session.execute(
                select(PythonAstFeature)
                .where(PythonAstFeature.analysis_run_id == analysis_run_id)
                .order_by(PythonAstFeature.feature_type, PythonAstFeature.id)
            )
        )
        .scalars()
        .all()
    )


async def _latest_evaluation_job_for_submission(
    session: AsyncSession,
    submission_id: int,
) -> EvaluationJob | None:
    return (
        await session.execute(
            select(EvaluationJob)
            .where(EvaluationJob.submission_id == submission_id)
            .order_by(desc(EvaluationJob.created_at), desc(EvaluationJob.id))
            .limit(1)
        )
    ).scalar_one_or_none()


async def _latest_task_phases_for_job(
    session: AsyncSession,
    *,
    submission_id: int,
    job_id: int | None,
) -> list[TaskPhaseResponse]:
    if job_id is None:
        return []
    result = await session.execute(
        select(TaskLogEvent)
        .where(TaskLogEvent.submission_id == submission_id)
        .where(TaskLogEvent.job_id == job_id)
        .where(TaskLogEvent.event_type == "task.status")
        .where(TaskLogEvent.task_id.is_not(None))
        .where(TaskLogEvent.status.in_(PUBLIC_TASK_PHASE_STATUSES))
        .order_by(desc(TaskLogEvent.sequence), desc(TaskLogEvent.id))
    )
    latest_by_task_id: dict[str, TaskPhaseResponse] = {}
    for event in result.scalars().all():
        if event.task_id is None or event.task_id in latest_by_task_id:
            continue
        task_phase = _task_phase_response(event)
        if task_phase is not None:
            latest_by_task_id[event.task_id] = task_phase
    return sorted(latest_by_task_id.values(), key=lambda item: item.task_id)


async def _latest_eval_run_for_submission(
    session: AsyncSession,
    submission_id: int,
) -> EvalRun | None:
    """Return the newest EvalRun for public dual-flag status projection."""

    return await session.scalar(
        select(EvalRun)
        .where(EvalRun.submission_id == submission_id)
        .order_by(desc(EvalRun.created_at), desc(EvalRun.id))
        .limit(1)
    )


def _task_rows_from_eval_run(run: EvalRun) -> list[TaskRowResponse]:
    """Project dual-flag public task_rows from EvalRun.plan_json.selected_tasks.

    Never requires EvaluationJob. Overlays canonical_score_record_json outcomes
    for planned task ids only (no invented unplanned rows or guest log bodies).
    """

    selected_items = _eval_run_selected_task_items(run)
    if not selected_items:
        return []

    outcomes = _eval_run_score_outcomes_by_task_id(run)
    rows: list[TaskRowResponse] = []
    seen: set[str] = set()
    for item in selected_items:
        task_id = _selected_task_id(item)
        if task_id is None or task_id in seen:
            continue
        seen.add(task_id)
        outcome = outcomes.get(task_id)
        if outcome is None:
            phase = "assigned"
            has_result = False
        else:
            has_result = True
            phase = "completed" if outcome.get("passed") else "failed"
        rows.append(
            TaskRowResponse(
                task_id=task_id,
                display_name=task_id,
                source=_selected_task_source(item),
                phase=phase,
                status=phase,
                updated_at=run.updated_at or run.created_at or run.issued_at,
                attempt=run.attempt if has_result else None,
                has_result=has_result,
            )
        )
    return rows


def _eval_run_selected_task_items(run: EvalRun) -> list[object]:
    """Load selected_tasks from the immutable EvalRun plan (soft public path)."""

    try:
        plan = load_eval_run_plan(run)
        selected = plan.get("selected_tasks")
        return list(selected) if isinstance(selected, list) else []
    except Exception:
        # Public status must not 500 on a bad plan row; fall back to raw JSON.
        raw = run.plan_json
        if not isinstance(raw, str) or not raw:
            return []
        parsed = _json_value(raw, {})
        if not isinstance(parsed, Mapping):
            return []
        selected = parsed.get("selected_tasks")
        return list(selected) if isinstance(selected, list) else []


def _eval_run_score_outcomes_by_task_id(run: EvalRun) -> dict[str, dict[str, object]]:
    """Map planned task_id -> safe outcome flags from canonical_score_record_json."""

    raw = run.canonical_score_record_json
    if not isinstance(raw, str) or not raw:
        return {}
    parsed = _json_value(raw, {})
    if not isinstance(parsed, Mapping):
        return {}
    tasks = parsed.get("tasks")
    if not isinstance(tasks, list):
        return {}
    outcomes: dict[str, dict[str, object]] = {}
    for item in tasks:
        if not isinstance(item, Mapping):
            continue
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        aggregate = item.get("aggregate_score_f64be")
        passed = False
        if isinstance(aggregate, str) and aggregate:
            try:
                passed = decode_score_f64be(aggregate) == 1.0
            except Exception:
                passed = False
        else:
            # Some fixtures may already surface a float aggregate.
            if isinstance(aggregate, int | float) and not isinstance(aggregate, bool):
                passed = float(aggregate) == 1.0
        outcomes[task_id] = {"passed": passed, "has_result": True}
    return outcomes


def _task_rows_response(
    job: EvaluationJob | None,
    task_phases: list[TaskPhaseResponse],
    task_results: list[TaskResult],
) -> list[TaskRowResponse]:
    if job is None:
        return []

    rows_by_task_id: dict[str, TaskRowResponse] = {}
    ordered_task_ids: list[str] = []
    for planned in _planned_task_rows(job):
        rows_by_task_id[planned.task_id] = planned
        ordered_task_ids.append(planned.task_id)

    for phase in task_phases:
        row = rows_by_task_id.get(phase.task_id)
        if row is None:
            row = TaskRowResponse(
                task_id=phase.task_id,
                display_name=phase.task_id,
                source="benchmark",
                phase=phase.phase,
                status=phase.status,
                updated_at=phase.updated_at,
                attempt=phase.attempt,
            )
            rows_by_task_id[phase.task_id] = row
            ordered_task_ids.append(phase.task_id)
            continue
        rows_by_task_id[phase.task_id] = row.model_copy(
            update={
                "phase": phase.phase,
                "status": phase.status,
                "updated_at": phase.updated_at,
                "attempt": phase.attempt,
            }
        )

    for result in task_results:
        result_phase = _task_result_phase(result.status)
        row = rows_by_task_id.get(result.task_id)
        if row is None:
            rows_by_task_id[result.task_id] = TaskRowResponse(
                task_id=result.task_id,
                display_name=result.task_id,
                source="benchmark",
                phase=result_phase,
                status=result_phase,
                updated_at=result.created_at,
                attempt=None,
                has_result=True,
            )
            ordered_task_ids.append(result.task_id)
            continue
        update: dict[str, object] = {"has_result": True}
        if row.phase == "assigned":
            update["phase"] = result_phase
            update["status"] = result_phase
        if row.updated_at is None or result.created_at > row.updated_at:
            update["updated_at"] = result.created_at
        rows_by_task_id[result.task_id] = row.model_copy(update=update)

    return [rows_by_task_id[task_id] for task_id in ordered_task_ids]


def _planned_task_rows(job: EvaluationJob) -> list[TaskRowResponse]:
    rows: list[TaskRowResponse] = []
    seen: set[str] = set()
    for item in _selected_task_items(job.selected_tasks_json):
        task_id = _selected_task_id(item)
        if task_id is None or task_id in seen:
            continue
        seen.add(task_id)
        rows.append(
            TaskRowResponse(
                task_id=task_id,
                display_name=task_id,
                source=_selected_task_source(item),
                phase="assigned",
                status="assigned",
                updated_at=job.created_at,
                attempt=None,
            )
        )
    return rows


def _selected_task_items(raw: str) -> list[object]:
    value = _json_value(raw, [])
    return value if isinstance(value, list) else []


def _selected_task_id(item: object) -> str | None:
    if isinstance(item, str) and item:
        return item
    if isinstance(item, Mapping):
        task_id = item.get("task_id")
        if isinstance(task_id, str) and task_id:
            return task_id
    return None


def _selected_task_source(item: object) -> str:
    if isinstance(item, Mapping):
        benchmark = item.get("benchmark")
        if benchmark in {"swe_forge", "terminal_bench"}:
            return str(benchmark)
    return "benchmark"


def _task_result_phase(status: str) -> str:
    if status in {"failed", "error"}:
        return "failed"
    return "completed"


def _task_phase_response(event: TaskLogEvent) -> TaskPhaseResponse | None:
    metadata = _json_object(event.metadata_json)
    phase = metadata.get("phase")
    if not isinstance(phase, str) or phase not in PUBLIC_TASK_PHASE_STATUSES:
        return None
    if event.status not in PUBLIC_TASK_PHASE_STATUSES:
        return None
    attempt_value = metadata.get("attempt")
    attempt = (
        attempt_value
        if isinstance(attempt_value, int) and not isinstance(attempt_value, bool)
        else None
    )
    return TaskPhaseResponse(
        task_id=str(event.task_id),
        phase=phase,
        status=event.status,
        updated_at=event.created_at,
        attempt=attempt,
    )


async def _task_results_for_job(
    session: AsyncSession,
    job_id: int | None,
) -> list[TaskResult]:
    if job_id is None:
        return []
    return list(
        (
            await session.execute(
                select(TaskResult)
                .where(TaskResult.job_id == job_id)
                .order_by(TaskResult.task_id, TaskResult.id)
            )
        )
        .scalars()
        .all()
    )


async def _latest_evaluation_attempt(
    session: AsyncSession,
    submission_id: int,
) -> EvaluationAttempt | None:
    return (
        await session.execute(
            select(EvaluationAttempt)
            .where(EvaluationAttempt.submission_id == submission_id)
            .order_by(desc(EvaluationAttempt.attempt_number), desc(EvaluationAttempt.id))
            .limit(1)
        )
    ).scalar_one_or_none()


async def _terminal_bench_trial_counts(
    session: AsyncSession,
    submission_id: int,
) -> dict[str, int]:
    if settings.attested_review_enabled and settings.phala_attestation_enabled:
        return {
            "total_trials": 0,
            "completed_trials": 0,
            "failed_trials": 0,
            "errored_trials": 0,
            "final_trials": 0,
        }
    attempts = (
        (
            await session.execute(
                select(EvaluationAttempt.id).where(EvaluationAttempt.submission_id == submission_id)
            )
        )
        .scalars()
        .all()
    )
    if not attempts:
        return {
            "total_trials": 0,
            "completed_trials": 0,
            "failed_trials": 0,
            "errored_trials": 0,
            "final_trials": 0,
        }
    trials = (
        (
            await session.execute(
                select(TerminalBenchTrial).where(
                    TerminalBenchTrial.evaluation_attempt_id.in_(attempts)
                )
            )
        )
        .scalars()
        .all()
    )
    return {
        "total_trials": len(trials),
        "completed_trials": sum(1 for trial in trials if trial.status == "completed"),
        "failed_trials": sum(1 for trial in trials if trial.status == "failed"),
        "errored_trials": sum(1 for trial in trials if trial.status == "errored"),
        "final_trials": sum(1 for trial in trials if trial.is_final),
    }


async def _count_rows(session: AsyncSession, model: type[Any], submission_id: int) -> int:
    value = await session.scalar(
        select(func.count(model.id)).where(model.submission_id == submission_id)
    )
    return int(value or 0)


def _analyzer_status_response(
    raw_status: str,
    analysis: AnalysisRun | None,
    llm: LlmVerdict | None,
) -> AnalyzerStatusResponse:
    return AnalyzerStatusResponse(
        phase=_analyzer_phase(raw_status, analysis),
        status=analysis.status if analysis is not None else None,
        verdict=analysis.verdict if analysis is not None else None,
        reason_codes=_json_string_list(
            analysis.reason_codes_json if analysis is not None else "[]"
        ),
        llm_verdict=llm.verdict if llm is not None else None,
        llm_confidence=llm.confidence if llm is not None else None,
        llm_reason_codes=_json_string_list(llm.reason_codes_json if llm is not None else "[]"),
        llm_rationale=_llm_public_rationale(llm),
        started_at=analysis.started_at if analysis is not None else None,
        finished_at=analysis.finished_at if analysis is not None else None,
    )


def _similarity_status_response(matches: list[SimilarityMatch]) -> SimilarityStatusResponse:
    top_matches: list[SimilarityMatchSummaryResponse] = []
    for match in matches[:5]:
        evidence = _json_object(match.evidence_json)
        top_matches.append(
            SimilarityMatchSummaryResponse(
                matched_submission_id=match.matched_submission_id,
                match_kind=match.match_kind,
                score_percent=match.score,
                risk_band=_optional_str(evidence.get("risk_band")),
                algorithm_version=_optional_str(evidence.get("algorithm_version")),
                top_file_pairs=_similarity_file_pairs(evidence),
            )
        )
    return SimilarityStatusResponse(
        max_score_percent=matches[0].score if matches else None,
        match_count=len(matches),
        top_matches=top_matches,
    )


def _llm_public_rationale(llm: LlmVerdict | None) -> str | None:
    if llm is None:
        return None
    response = _json_object(llm.raw_response_json)
    verdict_json = response.get("verdict_json")
    rationale = None
    if isinstance(verdict_json, Mapping):
        rationale = verdict_json.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        return _public_task_event_text(rationale.strip())[:1000]
    reason_codes = _json_string_list(llm.reason_codes_json)
    if reason_codes:
        fallback = f"LLM verdict {llm.verdict} with reason codes: {', '.join(reason_codes)}."
    else:
        fallback = f"LLM verdict recorded: {llm.verdict}."
    return _public_task_event_text(fallback)[:1000]


def _similarity_file_pairs(evidence: Mapping[str, object]) -> list[SimilarityFilePairResponse]:
    raw_pairs = evidence.get("top_file_pairs")
    if not isinstance(raw_pairs, list):
        return []
    pairs: list[SimilarityFilePairResponse] = []
    for raw_pair in raw_pairs[:5]:
        if not isinstance(raw_pair, Mapping):
            continue
        pairs.append(
            SimilarityFilePairResponse(
                source_file_path=_public_path_or_none(raw_pair.get("source_file_path")),
                matched_file_path=_public_path_or_none(raw_pair.get("matched_file_path")),
                score_percent=_optional_float(raw_pair.get("score_percent")),
            )
        )
    return pairs


def _public_path_or_none(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return _public_task_event_text(value)[:1000]


def _ast_status_response(
    features: list[PythonAstFeature],
    analysis: AnalysisRun | None,
) -> AstStatusResponse:
    feature_types: dict[str, int] = {}
    for feature in features:
        feature_types[feature.feature_type] = feature_types.get(feature.feature_type, 0) + 1
    verdict: str | None = None
    verdict_reason: str | None = None
    if analysis is not None:
        ast_report = _json_object(analysis.report_json).get("ast")
        if isinstance(ast_report, Mapping):
            verdict = _optional_str(ast_report.get("verdict"))
            verdict_reason = _optional_str(ast_report.get("verdict_reason"))
    return AstStatusResponse(
        feature_count=len(features),
        feature_types=feature_types,
        verdict=verdict,
        verdict_reason=verdict_reason,
    )


_MAX_RULES_CHECK_RULES = 50
_MAX_RULES_CHECK_EVIDENCE = 50
_MAX_RULES_CHECK_SNIPPET_CHARS = 2000
_MAX_RULES_CHECK_NOTES_CHARS = 1000


def _rules_check_response(analysis: AnalysisRun | None) -> RulesCheckResponse | None:
    if analysis is None:
        return None
    rules_check = _json_object(analysis.report_json).get("rules_check")
    if not isinstance(rules_check, Mapping):
        return None
    rules: list[RuleResultResponse] = []
    raw_rules = rules_check.get("rule_results")
    if isinstance(raw_rules, list):
        for raw_rule in raw_rules[:_MAX_RULES_CHECK_RULES]:
            if isinstance(raw_rule, Mapping):
                rules.append(_rule_result_response(raw_rule))
    reviewer_notes = rules_check.get("reviewer_notes")
    notes: str | None = None
    if isinstance(reviewer_notes, str) and reviewer_notes.strip():
        notes = redact_secrets(reviewer_notes)[0][:_MAX_RULES_CHECK_NOTES_CHARS]
    return RulesCheckResponse(
        verdict=_optional_str(rules_check.get("overall_verdict")),
        recommended_status=_optional_str(rules_check.get("recommended_status")),
        rules_version=_optional_str(rules_check.get("rules_version")),
        reviewer_used=_optional_bool(rules_check.get("reviewer_used")),
        reason_codes=_object_string_list(rules_check.get("reason_codes")),
        rules=rules,
        notes=notes,
    )


def _rule_result_response(raw_rule: Mapping[str, object]) -> RuleResultResponse:
    evidence: list[RuleEvidenceResponse] = []
    raw_evidence = raw_rule.get("evidence")
    if isinstance(raw_evidence, list):
        for raw_item in raw_evidence[:_MAX_RULES_CHECK_EVIDENCE]:
            if isinstance(raw_item, Mapping):
                evidence.append(_rule_evidence_response(raw_item))
    return RuleResultResponse(
        rule_id=_optional_str(raw_rule.get("rule_id")) or "",
        title=_optional_str(raw_rule.get("title")) or "",
        status=_optional_str(raw_rule.get("status")) or "",
        reason_codes=_object_string_list(raw_rule.get("reason_codes")),
        evidence=evidence,
    )


def _rule_evidence_response(raw_item: Mapping[str, object]) -> RuleEvidenceResponse:
    raw_snippet = raw_item.get("snippet")
    snippet = raw_snippet if isinstance(raw_snippet, str) else ""
    raw_description = raw_item.get("description")
    description = raw_description if isinstance(raw_description, str) else ""
    raw_path = raw_item.get("path")
    path = raw_path if isinstance(raw_path, str) else ""
    return RuleEvidenceResponse(
        path=redact_secrets(path)[0],
        line_start=_optional_int(raw_item.get("line_start")) or 0,
        line_end=_optional_int(raw_item.get("line_end")) or 0,
        snippet=redact_secrets(snippet)[0][:_MAX_RULES_CHECK_SNIPPET_CHARS],
        reason_code=_optional_str(raw_item.get("reason_code")) or "",
        description=redact_secrets(description)[0][:_MAX_RULES_CHECK_SNIPPET_CHARS],
    )


def _evaluation_status_response(
    job: EvaluationJob | None,
    attempt: EvaluationAttempt | None,
    task_phases: list[TaskPhaseResponse] | None = None,
    task_rows: list[TaskRowResponse] | None = None,
) -> EvaluationStatusResponse:
    return EvaluationStatusResponse(
        job_id=job.job_id if job is not None else None,
        status=job.status if job is not None else None,
        score=job.score if job is not None else 0.0,
        passed_tasks=job.passed_tasks if job is not None else 0,
        total_tasks=job.total_tasks if job is not None else 0,
        verdict=job.verdict if job is not None else None,
        reason_codes=_json_string_list(job.reason_codes_json if job is not None else "[]"),
        current_attempt=attempt.attempt_number if attempt is not None else None,
        attempt_status=attempt.status if attempt is not None else None,
        task_phases=task_phases or [],
        task_rows=task_rows or [],
    )


def _legacy_confirmed_empty_submission(submission: AgentSubmission) -> bool:
    return bool(
        submission.env_confirmed_empty
        and submission.env_locked_at is not None
        and submission.env_compatibility_reason == "pre_env_gate_analysis_allowed"
    )


async def confirm_empty_miner_env_and_enqueue_evaluation(
    session: AsyncSession,
    submission: AgentSubmission,
    *,
    actor: str,
    reason: str = "miner_env_confirmed_empty",
) -> EvaluationJob | None:
    return await _lock_env_and_enqueue_submission(
        session,
        submission,
        confirmed_empty=True,
        actor=actor,
        reason=reason,
    )


async def _lock_env_and_enqueue_submission(
    session: AsyncSession,
    submission: AgentSubmission,
    *,
    confirmed_empty: bool,
    actor: str = "evaluation",
    reason: str = "miner_env_ready",
) -> EvaluationJob | None:
    locked = await lock_miner_env_for_evaluation(
        session,
        submission,
        confirmed_empty=confirmed_empty,
    )
    if not locked:
        raise HTTPException(status_code=409, detail="submission env confirmation is required")
    if submission.raw_status == "analysis_allowed":
        await ensure_submission_status(
            session,
            submission,
            "waiting_miner_env",
            actor=actor,
            reason="waiting_miner_env",
            metadata={"env_ready": True},
        )
    return await enqueue_evaluation_job_for_submission(
        session,
        submission,
        confirmed_miner_env=True,
    )


def _public_phase(raw_status: str) -> str:
    if raw_status in {"received", "upload_verified", "rate_limit_reserved"}:
        return "intake"
    if raw_status == "review_queued":
        return "review_queued"
    if raw_status == "review_cvm_running":
        return "review_cvm_running"
    if raw_status == "review_provider_standby":
        return "review_provider_standby"
    if raw_status == "review_verifying":
        return "review_verifying"
    if raw_status == "review_allowed":
        return "review_allowed"
    if raw_status == "review_rejected":
        return "review_rejected"
    if raw_status == "review_escalated":
        return "review_escalated"
    if raw_status == "review_expired":
        return "review_expired"
    if raw_status == "review_cancelled":
        return "review_cancelled"
    if raw_status == "review_error":
        return "review_error"
    if raw_status == "analysis_queued":
        return "analysis"
    if raw_status == "ast_running":
        return "ast_review"
    if raw_status == "llm_running":
        return "llm_review"
    if raw_status == "llm_standby":
        return "llm_standby"
    if raw_status in {"analysis_rejected"}:
        return "analysis_complete"
    if raw_status in {"analysis_escalated", "admin_paused"}:
        return "admin_review"
    if raw_status == "waiting_miner_env":
        return "waiting_environments"
    if raw_status in {"analysis_allowed", "tb_queued"}:
        return "evaluation_queued"
    if raw_status in {"tb_running", "tb_failed_retryable", "evaluating"}:
        return "evaluation"
    if raw_status in {"tb_completed", "valid", "completed", "overridden_valid"}:
        return "complete"
    if raw_status in {"tb_failed_final", "error", "invalid", "overridden_invalid"}:
        return "failed"
    if raw_status in {"eval_expired", "eval_no_result", "eval_rejected"}:
        return "failed"
    if raw_status == "cancelled":
        return "cancelled"
    if raw_status == "suspicious":
        return "admin_review"
    return public_status_for(raw_status)


def _analyzer_phase(raw_status: str, analysis: AnalysisRun | None) -> str:
    if raw_status in {"analysis_queued", "ast_running", "llm_running", "llm_standby"}:
        return "running"
    if raw_status in {
        "analysis_allowed",
        "waiting_miner_env",
        "analysis_rejected",
        "analysis_escalated",
        "admin_paused",
    }:
        return "completed"
    if analysis is not None:
        return analysis.status
    return "pending"


def _latest_timestamp(*rows: object | None) -> datetime | None:
    values: list[datetime] = []
    for row in rows:
        if row is None:
            continue
        for field_name in ("finished_at", "created_at", "submitted_at"):
            value = getattr(row, field_name, None)
            if isinstance(value, datetime):
                values.append(value)
                break
    return max(values) if values else None


def _json_string_list(raw: str) -> list[str]:
    value = _json_value(raw, [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _json_object(raw: str) -> dict[str, object]:
    value = _json_value(raw, {})
    return value if isinstance(value, dict) else {}


def _json_value(raw: str, default: object) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _object_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _latest_submission_job(submission: AgentSubmission) -> EvaluationJob | None:
    if submission.latest_evaluation_job is not None:
        return submission.latest_evaluation_job
    return max(submission.jobs, key=lambda job: job.created_at, default=None)


def _evaluation_summary_response(job: EvaluationJob) -> EvaluationSummaryResponse:
    return EvaluationSummaryResponse(
        job_id=job.job_id,
        status=job.status,
        score=job.score,
        passed_tasks=job.passed_tasks,
        total_tasks=job.total_tasks,
        verdict=job.verdict,
        rules_version=job.rules_version,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _task_result_response(result: TaskResult) -> TaskResultResponse:
    return TaskResultResponse(
        task_id=result.task_id,
        docker_image=result.docker_image,
        status=result.status,
        score=result.score,
        returncode=result.returncode,
        duration_seconds=result.duration_seconds,
        failure_reason=_task_result_failure_reason(result),
        detail_log=_task_result_detail_log(result),
    )


def _task_result_failure_reason(result: TaskResult) -> str | None:
    if result.status not in {"failed", "error"} and result.returncode == 0:
        return None
    for value in (result.stderr, result.stdout):
        for line in value.splitlines():
            sanitized = _public_task_result_text(line).strip()
            if sanitized:
                return sanitized[:500]
    if result.returncode != 0:
        return f"Task process exited with code {result.returncode}."
    return "Task completed without a passing score."


def _task_result_detail_log(result: TaskResult) -> str | None:
    if result.status not in {"failed", "error"} and result.returncode == 0:
        return None
    sections: list[str] = [
        f"Task: {_public_task_display_name(result.task_id)}",
        f"Status: {result.status}",
        f"Score: {result.score:.4f}",
        f"Return code: {result.returncode}",
        f"Duration seconds: {result.duration_seconds:.3f}",
    ]
    error_log = _public_task_result_text(result.stderr).strip()
    output_log = _public_task_result_text(result.stdout).strip()
    if error_log:
        sections.extend(["", "Error log:", error_log[:4000]])
    if output_log:
        sections.extend(["", "Output log:", output_log[:4000]])
    if len(sections) <= 5:
        return None
    return "\n".join(sections)[:8000]


def _public_task_display_name(task_id: str) -> str:
    return task_id.removeprefix("terminal-bench/")


def _public_task_result_text(value: str) -> str:
    sanitized = _public_task_event_text(value)
    return re.sub(r"\bstd(?:out|err)\b", "task log", sanitized, flags=re.IGNORECASE)


async def _evaluation_response(session: AsyncSession, job: EvaluationJob) -> EvaluationResponse:
    task_phases = await _latest_task_phases_for_job(
        session,
        submission_id=job.submission_id,
        job_id=job.id,
    )
    task_results = list(job.task_results)
    task_rows = _task_rows_response(job, task_phases, task_results)
    return EvaluationResponse(
        job_id=job.job_id,
        submission_id=job.submission.id,
        name=job.submission.name,
        agent_hash=job.submission.agent_hash,
        zip_sha256=job.submission.zip_sha256,
        **_version_metadata(job.submission),
        status=job.status,
        effective_status=job.submission.effective_status,
        score=job.score,
        passed_tasks=job.passed_tasks,
        total_tasks=job.total_tasks,
        verdict=job.verdict,
        rules_version=job.rules_version,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        tasks=[_task_result_response(result) for result in job.task_results],
        task_phases=task_phases,
        task_rows=task_rows,
    )


async def _persist_submission(
    *,
    session: AsyncSession,
    http_request: Request,
    artifact: ArtifactMetadata,
    miner_hotkey: str,
    name: str,
    signature: str | None,
    signature_nonce: str | None,
    signature_timestamp: str | None,
    signature_payload_sha256: str | None,
    signature_message: str | None,
    route: str,
    actor: str,
    retry_on_version_conflict: bool = True,
) -> SubmissionResponse:
    canonical_artifact_hash = artifact.zip_sha256
    try:
        normalized_name = normalize_submission_name(name)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_submission_name", "message": str(exc)},
        ) from exc

    existing_artifact = await session.scalar(
        select(AgentSubmission).where(
            AgentSubmission.canonical_artifact_hash == canonical_artifact_hash
        )
    )
    if existing_artifact is None:
        existing_artifact = await session.scalar(
            select(AgentSubmission).where(AgentSubmission.agent_hash == canonical_artifact_hash)
        )
    if existing_artifact is not None:
        raise HTTPException(status_code=409, detail=_duplicate_code_hash_detail())

    family = await session.scalar(
        select(SubmissionFamily).where(SubmissionFamily.normalized_name == normalized_name)
    )
    if family is not None and family.owner_hotkey != miner_hotkey:
        raise HTTPException(status_code=409, detail=_name_taken_detail())

    window_seconds = settings.submission_rate_limit_window_seconds
    try:
        reservation = await reserve_submission_rate_limit(
            session=session,
            hotkey=miner_hotkey,
            artifact_hash=canonical_artifact_hash,
            zip_sha256=artifact.zip_sha256,
            zip_size_bytes=artifact.zip_size_bytes,
            request_ip=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("user-agent"),
            route=route,
            window_seconds=window_seconds,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "submission_rate_limited",
                "message": submission_rate_limit_message(window_seconds),
                "next_allowed_at": exc.next_allowed_at.isoformat(),
            },
        ) from exc

    if family is None:
        family = SubmissionFamily(
            public_family_id=uuid4().hex,
            owner_hotkey=miner_hotkey,
            display_name=name,
            normalized_name=normalized_name,
            version_count=0,
        )
        session.add(family)
        try:
            await session.flush()
        except IntegrityError as exc:
            await session.rollback()
            family = await session.scalar(
                select(SubmissionFamily).where(SubmissionFamily.normalized_name == normalized_name)
            )
            if family is None or family.owner_hotkey != miner_hotkey:
                raise _submission_conflict_from_integrity_error(exc) from exc
            if not retry_on_version_conflict:
                raise _submission_conflict_from_integrity_error(exc) from exc
            return await _persist_submission(
                session=session,
                http_request=http_request,
                artifact=artifact,
                miner_hotkey=miner_hotkey,
                name=name,
                signature=signature,
                signature_nonce=signature_nonce,
                signature_timestamp=signature_timestamp,
                signature_payload_sha256=signature_payload_sha256,
                signature_message=signature_message,
                route=route,
                actor=actor,
                retry_on_version_conflict=False,
            )

    version_number = family.version_count + 1
    submission = AgentSubmission(
        miner_hotkey=miner_hotkey,
        name=name,
        agent_name=name,
        agent_hash=canonical_artifact_hash,
        artifact_uri=artifact.artifact_path,
        submission_family_id=family.id,
        version_number=version_number,
        version_label=version_label(version_number),
        canonical_artifact_hash=canonical_artifact_hash,
        is_latest_version=True,
        status="received",
        zip_sha256=artifact.zip_sha256,
        zip_size_bytes=artifact.zip_size_bytes,
        artifact_path=artifact.artifact_path,
        raw_status="received",
        effective_status="received",
        env_confirmed_empty=True,
        env_confirmed_empty_at=datetime.now(UTC),
        signature=signature,
        signature_nonce=signature_nonce,
        signature_timestamp=signature_timestamp,
        signature_payload_sha256=signature_payload_sha256,
        signature_message=signature_message,
    )
    session.add(submission)
    try:
        await session.flush()
        await session.execute(
            update(AgentSubmission)
            .where(
                AgentSubmission.submission_family_id == family.id,
                AgentSubmission.id != submission.id,
            )
            .values(is_latest_version=False)
        )
        family.latest_submission_id = submission.id
        family.version_count = version_number
        session.add(
            SubmissionArtifact(
                submission_id=submission.id,
                artifact_kind="source_zip",
                uri=artifact.artifact_path,
                sha256=artifact.zip_sha256,
                size_bytes=artifact.zip_size_bytes,
                metadata_json=json.dumps(
                    {
                        "content_type": "application/zip",
                        "manifest_path": artifact.manifest_path,
                        "manifest": artifact.manifest.to_dict(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
        await record_initial_status(
            session,
            submission,
            actor=actor,
            reason="submission_received",
            metadata={"agent_hash": canonical_artifact_hash},
        )
        consume_submission_rate_limit(reservation)
        if settings.attested_review_enabled:
            await ensure_submission_status(
                session,
                submission,
                "upload_verified",
                actor=actor,
                reason="submission_upload_verified",
                metadata={"zip_sha256": submission.zip_sha256 or ""},
            )
            await ensure_submission_status(
                session,
                submission,
                "rate_limit_reserved",
                actor=actor,
                reason="submission_rate_limit_reserved",
                metadata={"agent_hash": submission.agent_hash},
            )
            try:
                rules_bundle = await asyncio.to_thread(
                    capture_rules_bundle,
                    settings.review_rules_root,
                )
                manifest = artifact.manifest.to_dict()
                created_review = await create_review_session(
                    session,
                    submission=submission,
                    artifact_bytes=Path(artifact.artifact_path).read_bytes(),
                    rules_files=rules_bundle_files(rules_bundle),
                    rules_revision_id=str(rules_bundle["revision_id"]),
                    settings=settings,
                    manifest_sha256=sha256(canonical_json_v1(manifest)).hexdigest(),
                    manifest_entries_sha256=sha256(
                        canonical_json_v1(manifest["entries"])
                    ).hexdigest(),
                    input_config=review_input_config_from_settings(settings),
                )
            except ReviewRateLimited as exc:
                # Keep rate/concurrency faults on 429 independent of conflict and
                # availability so capacity floods never collapse into 503.
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={"code": "review_rate_limited"},
                ) from exc
            except (ReviewConflict, RulesSnapshotCaptureError, ReviewDeploymentError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "code": "review_session_unavailable",
                        "message": "review is unavailable",
                    },
                ) from exc
            await record_review_submission_status(
                session,
                review_session=created_review.session,
                assignment=created_review.assignment,
                raw_status="review_queued",
                reason="review_assignment_issued",
            )
        else:
            await queue_submission_analysis(session, submission, actor=actor)
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        if retry_on_version_conflict and _is_version_allocation_conflict(exc):
            return await _persist_submission(
                session=session,
                http_request=http_request,
                artifact=artifact,
                miner_hotkey=miner_hotkey,
                name=name,
                signature=signature,
                signature_nonce=signature_nonce,
                signature_timestamp=signature_timestamp,
                signature_payload_sha256=signature_payload_sha256,
                signature_message=signature_message,
                route=route,
                actor=actor,
                retry_on_version_conflict=False,
            )
        raise _submission_conflict_from_integrity_error(exc) from exc

    await session.refresh(submission)
    return SubmissionResponse(
        submission_id=submission.id,
        name=submission.name,
        display_name=family.display_name,
        agent_hash=submission.agent_hash,
        zip_sha256=artifact.zip_sha256,
        family_id=family.public_family_id,
        version_number=version_number,
        version_label=submission.version_label or version_label(version_number),
        version_count=family.version_count,
        is_latest_version=submission.is_latest_version,
        latest_submission_id=family.latest_submission_id,
        status=submission.effective_status,
        effective_status=submission.effective_status,
        submitted_at=submission.submitted_at,
        created_at=submission.created_at,
        latest_evaluation=None,
    )


def _is_version_allocation_conflict(exc: IntegrityError) -> bool:
    message = str(exc.orig).lower()
    return "family_version" in message or (
        "submission_family_id" in message and "version_number" in message
    )


def _duplicate_code_hash_detail() -> dict[str, str]:
    return {
        "code": "duplicate_code_hash",
        "message": "submission artifact has already been submitted",
    }


def _name_taken_detail() -> dict[str, str]:
    return {
        "code": "name_taken",
        "message": "submission name is already claimed by another owner",
    }


def _submission_conflict_from_integrity_error(exc: IntegrityError) -> HTTPException:
    message = str(exc.orig).lower()
    if "canonical_artifact_hash" in message or "agent_hash" in message:
        return HTTPException(status_code=409, detail=_duplicate_code_hash_detail())
    if "normalized_name" in message:
        return HTTPException(status_code=409, detail=_name_taken_detail())
    return HTTPException(
        status_code=409,
        detail={
            "code": "submission_conflict",
            "message": "submission conflicts with existing data",
        },
    )


def _base_bridge_headers(
    *,
    hotkey: str | None,
    nonce: str | None,
    request_hash: str | None,
    filename: str | None,
) -> BaseBridgeHeaders:
    return BaseBridgeHeaders(
        hotkey=_required_base_header(hotkey, "X-Base-Verified-Hotkey"),
        nonce=_required_base_header(nonce, "X-Base-Verified-Nonce"),
        request_hash=_required_base_header(request_hash, "X-Base-Request-Hash"),
        filename=filename,
    )


def _required_base_header(value: str | None, header_name: str) -> str:
    if value is None or not value.strip():
        raise HTTPException(status_code=400, detail=f"missing {header_name}")
    return value.strip()


def _submission_display_name(filename: str | None) -> str:
    if filename is None:
        return "agent"
    display_name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return (display_name or "agent")[:128]


def _base_bridge_signature_message(headers: BaseBridgeHeaders) -> str:
    return json.dumps(
        {
            "base_challenge_slug": settings.slug,
            "base_verified_nonce": headers.nonce,
            "base_request_hash": headers.request_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _prepare_raw_zip_artifact(zip_bytes: bytes) -> ArtifactMetadata:
    try:
        return store_zip_bytes(
            zip_bytes=zip_bytes,
            artifact_root=settings.artifact_root,
            max_zip_bytes=settings.zip_max_bytes,
        )
    except ArtifactValidationError as exc:
        status_code = 413 if exc.reason_code == "zip_too_large" else 400
        raise HTTPException(
            status_code=status_code,
            detail={"code": exc.reason_code, "message": exc.message},
        ) from exc


def _prepare_artifact(request: SubmissionRequest) -> ArtifactMetadata:
    artifact_source_count = int(request.artifact_uri is not None) + int(
        request.artifact_zip_base64 is not None
    )
    if artifact_source_count != 1:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_artifact_source_count",
                "message": "exactly one artifact source is required",
            },
        )
    try:
        if request.artifact_zip_base64 is not None:
            return store_base64_zip(
                encoded_zip=request.artifact_zip_base64,
                artifact_root=settings.artifact_root,
                max_zip_bytes=settings.zip_max_bytes,
            )
        if request.artifact_uri is None:
            raise AssertionError("artifact source count validation failed")
        return store_zip_uri(
            artifact_uri=request.artifact_uri,
            artifact_root=settings.artifact_root,
            max_zip_bytes=settings.zip_max_bytes,
        )
    except ArtifactValidationError as exc:
        status_code = 413 if exc.reason_code == "zip_too_large" else 400
        raise HTTPException(
            status_code=status_code,
            detail={"code": exc.reason_code, "message": exc.message},
        ) from exc
