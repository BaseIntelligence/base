"""Canonical task / job / submission status literals (single source of truth).

The evaluation runner, reconciler, validator executor, work-unit exposure, and
submission state machine all reason about the same status strings. Centralizing
them here keeps the cross-module status sets (terminal / assignable / halted)
from silently drifting when a status is added or renamed.

``TaskStatus.TIMED_OUT`` is a TERMINAL, non-passing task status: a timed-out
task is scored ``0`` and counted exactly once, so its work unit neither
re-dispatches forever, discards a later result, nor blocks job finalization.
"""

from __future__ import annotations

from enum import StrEnum


class TaskStatus(StrEnum):
    """Persisted ``TaskResult.status`` values."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"
    TIMED_OUT = "timed_out"


class JobStatus(StrEnum):
    """Persisted ``EvaluationJob.status`` values."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


class SubmissionStatus(StrEnum):
    """Persisted ``AgentSubmission.raw_status`` values (internal + legacy)."""

    # Internal lifecycle
    RECEIVED = "received"
    UPLOAD_VERIFIED = "upload_verified"
    RATE_LIMIT_RESERVED = "rate_limit_reserved"
    REVIEW_QUEUED = "review_queued"
    REVIEW_CVM_RUNNING = "review_cvm_running"
    REVIEW_PROVIDER_STANDBY = "review_provider_standby"
    REVIEW_VERIFYING = "review_verifying"
    REVIEW_ALLOWED = "review_allowed"
    REVIEW_REJECTED = "review_rejected"
    REVIEW_ESCALATED = "review_escalated"
    REVIEW_EXPIRED = "review_expired"
    REVIEW_CANCELLED = "review_cancelled"
    REVIEW_ERROR = "review_error"
    ANALYSIS_QUEUED = "analysis_queued"
    AST_RUNNING = "ast_running"
    LLM_RUNNING = "llm_running"
    LLM_STANDBY = "llm_standby"
    ANALYSIS_ALLOWED = "analysis_allowed"
    WAITING_MINER_ENV = "waiting_miner_env"
    ANALYSIS_REJECTED = "analysis_rejected"
    ANALYSIS_ESCALATED = "analysis_escalated"
    TB_QUEUED = "tb_queued"
    TB_RUNNING = "tb_running"
    TB_COMPLETED = "tb_completed"
    TB_FAILED_RETRYABLE = "tb_failed_retryable"
    TB_FAILED_FINAL = "tb_failed_final"
    CANCELLED = "cancelled"
    ADMIN_PAUSED = "admin_paused"
    # Legacy lifecycle
    PENDING = "pending"
    QUEUED = "queued"
    EVALUATING = "evaluating"
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    ERROR = "error"
    COMPLETED = "completed"
    OVERRIDDEN_VALID = "overridden_valid"
    OVERRIDDEN_INVALID = "overridden_invalid"


INTERNAL_SUBMISSION_STATUSES: frozenset[SubmissionStatus] = frozenset(
    {
        SubmissionStatus.RECEIVED,
        SubmissionStatus.UPLOAD_VERIFIED,
        SubmissionStatus.RATE_LIMIT_RESERVED,
        SubmissionStatus.REVIEW_QUEUED,
        SubmissionStatus.REVIEW_CVM_RUNNING,
        SubmissionStatus.REVIEW_PROVIDER_STANDBY,
        SubmissionStatus.REVIEW_VERIFYING,
        SubmissionStatus.REVIEW_ALLOWED,
        SubmissionStatus.REVIEW_REJECTED,
        SubmissionStatus.REVIEW_ESCALATED,
        SubmissionStatus.REVIEW_EXPIRED,
        SubmissionStatus.REVIEW_CANCELLED,
        SubmissionStatus.REVIEW_ERROR,
        SubmissionStatus.ANALYSIS_QUEUED,
        SubmissionStatus.AST_RUNNING,
        SubmissionStatus.LLM_RUNNING,
        SubmissionStatus.LLM_STANDBY,
        SubmissionStatus.ANALYSIS_ALLOWED,
        SubmissionStatus.WAITING_MINER_ENV,
        SubmissionStatus.ANALYSIS_REJECTED,
        SubmissionStatus.ANALYSIS_ESCALATED,
        SubmissionStatus.TB_QUEUED,
        SubmissionStatus.TB_RUNNING,
        SubmissionStatus.TB_COMPLETED,
        SubmissionStatus.TB_FAILED_RETRYABLE,
        SubmissionStatus.TB_FAILED_FINAL,
        SubmissionStatus.CANCELLED,
        SubmissionStatus.ADMIN_PAUSED,
    }
)

LEGACY_SUBMISSION_STATUSES: frozenset[SubmissionStatus] = frozenset(
    {
        SubmissionStatus.PENDING,
        SubmissionStatus.QUEUED,
        SubmissionStatus.EVALUATING,
        SubmissionStatus.VALID,
        SubmissionStatus.INVALID,
        SubmissionStatus.SUSPICIOUS,
        SubmissionStatus.ERROR,
        SubmissionStatus.COMPLETED,
        SubmissionStatus.OVERRIDDEN_VALID,
        SubmissionStatus.OVERRIDDEN_INVALID,
    }
)

#: A ``TaskResult`` is finished (no longer a pending work unit) once its status
#: is terminal. ``timed_out`` is terminal and NON-PASSING.
TERMINAL_TASK_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.ERROR,
        TaskStatus.TIMED_OUT,
    }
)

#: Terminal ``EvaluationJob.status`` values (a job itself never carries
#: ``timed_out``; that is a per-task result status only).
TERMINAL_JOB_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.ERROR,
    }
)

#: Evaluation-job statuses whose tasks are still assignable work. Terminal jobs
#: expose no work.
ASSIGNABLE_JOB_STATUSES: tuple[JobStatus, ...] = (
    JobStatus.PENDING,
    JobStatus.QUEUED,
    JobStatus.RUNNING,
)

#: Submissions whose gate outcome or lifecycle halted them never expose work,
#: even if a non-terminal job row lingers (e.g. an admin pause after allow).
HALTED_SUBMISSION_STATUSES: tuple[SubmissionStatus, ...] = (
    SubmissionStatus.ANALYSIS_REJECTED,
    SubmissionStatus.ANALYSIS_ESCALATED,
    SubmissionStatus.ADMIN_PAUSED,
    SubmissionStatus.CANCELLED,
    SubmissionStatus.REVIEW_REJECTED,
    SubmissionStatus.REVIEW_ESCALATED,
    SubmissionStatus.REVIEW_EXPIRED,
    SubmissionStatus.REVIEW_CANCELLED,
    SubmissionStatus.REVIEW_ERROR,
    SubmissionStatus.TB_FAILED_FINAL,
)
