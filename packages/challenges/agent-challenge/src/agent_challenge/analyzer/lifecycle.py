from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.analyzer.ast_features import (
    build_python_ast_feature_rows,
    extract_python_ast_features,
)
from agent_challenge.analyzer.gateway_rules_reviewer import (
    RULES_REVIEWER_INFRA_REASON,
    GatewayRulesReviewer,
)
from agent_challenge.analyzer.llm_reviewer import (
    GatewayReviewProvider,
    KimiLlmReviewer,
    LlmProviderRateLimited,
    LlmProviderTimeout,
    LlmProviderUnavailable,
    LlmReviewOutcome,
    LlmReviewProvider,
)
from agent_challenge.analyzer.similarity import (
    ALGORITHM_VERSION,
    persist_same_challenge_similarity_matches,
)
from agent_challenge.core.config import settings
from agent_challenge.core.models import (
    AdminReviewDecision,
    AgentSubmission,
    AnalysisRun,
    EvaluationJob,
    SubmissionArtifact,
)
from agent_challenge.evaluation.runner import (
    enqueue_evaluation_job_for_submission,
    ensure_miner_env_ready_for_evaluation,
)
from agent_challenge.submissions.artifacts import (
    ArtifactMetadata,
    ArtifactReadSession,
    ZipArtifactManifest,
    extract_zip_to_directory,
)
from agent_challenge.submissions.state_machine import ensure_submission_status

if TYPE_CHECKING:
    from agent_challenge.analyzer.pipeline import ReviewerLike
    from agent_challenge.analyzer.schemas import AnalyzerPipelineReport

ANALYZER_NAME = "blocking_analyzer"
ANALYZER_VERSION = "ast-similarity-llm-v1"
DEFAULT_ANALYSIS_LEASE_SECONDS = 900


class AnalyzerReviewer(Protocol):
    def review(
        self,
        *,
        analysis_run_id: int,
        manifest: ZipArtifactManifest,
        read_session: ArtifactReadSession,
        similarity_evidence: list[Mapping[str, object]],
    ) -> LlmReviewOutcome: ...


@dataclass(frozen=True)
class AnalysisSummary:
    analysis_run_id: int
    submission_id: int
    verdict: str
    status: str
    evaluation_job_id: str | None = None


async def queue_submission_analysis(
    session: AsyncSession,
    submission: AgentSubmission,
    *,
    actor: str = "api",
) -> None:
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
    await ensure_submission_status(
        session,
        submission,
        "analysis_queued",
        actor=actor,
        reason="blocking_analysis_queued",
        metadata={"analyzer": ANALYZER_NAME, "analyzer_version": ANALYZER_VERSION},
    )


async def claim_next_analysis_submission(
    session: AsyncSession,
    *,
    lease_owner: str,
    lease_seconds: int = DEFAULT_ANALYSIS_LEASE_SECONDS,
) -> AgentSubmission | None:
    await reclaim_expired_analysis_runs(session, lease_owner=lease_owner)
    if _llm_provider_ready():
        standby_submission = await session.scalar(
            select(AgentSubmission)
            .where(AgentSubmission.raw_status == "llm_standby")
            .order_by(AgentSubmission.created_at, AgentSubmission.id)
            .limit(1)
        )
        if standby_submission is not None:
            await ensure_submission_status(
                session,
                standby_submission,
                "analysis_queued",
                actor=lease_owner,
                reason="llm_provider_ready",
                metadata={"analyzer": ANALYZER_NAME, "analyzer_version": ANALYZER_VERSION},
            )
    submission = await session.scalar(
        select(AgentSubmission)
        .where(AgentSubmission.raw_status == "analysis_queued")
        .order_by(AgentSubmission.created_at, AgentSubmission.id)
        .limit(1)
    )
    if submission is None:
        return None
    await ensure_submission_status(
        session,
        submission,
        "ast_running",
        actor=lease_owner,
        reason="blocking_analysis_claimed",
        metadata={"analyzer": ANALYZER_NAME, "analyzer_version": ANALYZER_VERSION},
    )
    return submission


async def run_next_analysis(
    session: AsyncSession,
    *,
    lease_owner: str,
    lease_seconds: int = DEFAULT_ANALYSIS_LEASE_SECONDS,
    reviewer: AnalyzerReviewer | None = None,
) -> AnalysisSummary | None:
    submission = await claim_next_analysis_submission(
        session,
        lease_owner=lease_owner,
        lease_seconds=lease_seconds,
    )
    if submission is None:
        return None
    return await run_analysis_for_submission(
        session,
        submission.id,
        actor=lease_owner,
        lease_seconds=lease_seconds,
        reviewer=reviewer,
    )


async def run_analysis_for_submission(
    session: AsyncSession,
    submission_id: int,
    *,
    actor: str = "analyzer",
    lease_seconds: int = DEFAULT_ANALYSIS_LEASE_SECONDS,
    reviewer: AnalyzerReviewer | None = None,
) -> AnalysisSummary:
    submission = await session.get(AgentSubmission, submission_id)
    if submission is None:
        raise ValueError(f"unknown submission: {submission_id}")
    if submission.raw_status == "analysis_queued":
        await ensure_submission_status(
            session,
            submission,
            "ast_running",
            actor=actor,
            reason="blocking_analysis_claimed",
            metadata={"analyzer": ANALYZER_NAME, "analyzer_version": ANALYZER_VERSION},
        )

    artifact = await _source_artifact(session, submission)
    artifact_metadata = _artifact_metadata(artifact)
    now = datetime.now(UTC)
    analysis_run = AnalysisRun(
        submission_id=submission.id,
        analyzer_name=ANALYZER_NAME,
        analyzer_version=ANALYZER_VERSION,
        status="running",
        input_artifact_id=artifact.id,
        lease_owner=actor,
        lease_expires_at=now + timedelta(seconds=lease_seconds),
        heartbeat_at=now,
        started_at=now,
    )
    session.add(analysis_run)
    await session.flush()

    # CPU-bound AST compilation over the whole submission runs off the shared
    # event loop so the combined worker cannot freeze HTTP/SSE handling. The
    # read session and manifest are plain data (no ORM/session access), so the
    # thread does pure compute and the loop persists the resulting rows.
    ast_report = await asyncio.to_thread(
        extract_python_ast_features,
        manifest=artifact_metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(artifact_metadata),
    )
    session.add_all(
        build_python_ast_feature_rows(analysis_run_id=analysis_run.id, report=ast_report)
    )
    await session.flush()

    if settings.analyzer_similarity_enabled:
        matches = await persist_same_challenge_similarity_matches(
            session,
            analysis_run_id=analysis_run.id,
        )
    else:
        matches = []
    similarity_evidence = [_json_object(match.evidence_json) for match in matches]

    await ensure_submission_status(
        session,
        submission,
        "llm_running",
        actor=actor,
        reason="blocking_analysis_ast_completed",
        metadata={
            "analysis_run_id": analysis_run.id,
            "python_file_count": ast_report.python_file_count,
            "similarity_matches": len(matches),
        },
    )

    uses_configured_reviewer = reviewer is None
    llm_reviewer = reviewer or build_configured_lifecycle_reviewer()
    if uses_configured_reviewer and _reviewer_missing_gateway_token(llm_reviewer):
        return await _mark_llm_standby(
            session=session,
            submission=submission,
            analysis_run=analysis_run,
            actor=actor,
            ast_report=ast_report.to_dict(),
            similarity_evidence=similarity_evidence,
            reason="missing_llm_gateway_token",
            provider_name=_reviewer_provider_name(llm_reviewer),
            model_name=_reviewer_model_name(llm_reviewer),
        )
    # FIX-2: release the pooled connection before the slow LLM call. Holding an
    # idle cross-node asyncpg socket across it lets NAT/firewall black-hole the
    # connection, hanging the next statement; the refresh() below re-checks-out
    # through pool_pre_ping (FIX-1). expire_on_commit=False keeps objects usable.
    await session.commit()
    try:
        # The reviewer does a blocking httpx.post (up to 720s across retries).
        # Offload it so it cannot stall the shared event loop; the review is pure
        # compute/IO over the artifact + manifest and never touches the session
        # (the commit above already released the pooled connection), and it
        # returns a transient LlmVerdict row the loop persists below.
        outcome = await asyncio.to_thread(
            llm_reviewer.review,
            analysis_run_id=analysis_run.id,
            manifest=artifact_metadata.manifest,
            read_session=ArtifactReadSession.from_artifact_metadata(artifact_metadata),
            similarity_evidence=similarity_evidence,
        )
    except (LlmProviderRateLimited, LlmProviderTimeout, LlmProviderUnavailable) as exc:
        return await _mark_llm_standby(
            session=session,
            submission=submission,
            analysis_run=analysis_run,
            actor=actor,
            ast_report=ast_report.to_dict(),
            similarity_evidence=similarity_evidence,
            reason=_llm_standby_reason(exc, uses_configured_reviewer=uses_configured_reviewer),
            provider_name=_reviewer_provider_name(llm_reviewer),
            model_name=_reviewer_model_name(llm_reviewer),
        )

    # Resilience: a synthetic fail-closed escalate caused by a transient provider
    # error or a recoverable tool-calling miss must NOT permanently park the
    # submission in admin_paused. Route retryable reasons to llm_standby (which
    # re-queues) up to a bounded ceiling; only genuine model verdicts and real
    # exhaustion fall through to _apply_verdict (escalate -> admin_paused).
    if outcome.disposition == "fail_closed":
        reason = outcome.fail_closed_reason or "no_valid_submit_verdict"
        if _is_retryable_review_failure(reason) and (
            await _llm_standby_cycle_count(session, submission.id)
            < settings.llm_reviewer_max_standby_cycles
        ):
            return await _mark_llm_standby(
                session=session,
                submission=submission,
                analysis_run=analysis_run,
                actor=actor,
                ast_report=ast_report.to_dict(),
                similarity_evidence=similarity_evidence,
                reason=reason,
                provider_name=_reviewer_provider_name(llm_reviewer),
                model_name=_reviewer_model_name(llm_reviewer),
            )

    # Run the .rules anti-cheat check in the pre-eval GATE (off the event loop,
    # like the AST/LLM calls) and block cheating BEFORE terminal-bench. The
    # gateway rules reviewer only runs when the master gateway is configured;
    # when it is not, the submission has already parked in llm_standby above, so
    # the rules-check is skipped and the KimiLlmReviewer verdict is preserved.
    rules_report = await _maybe_run_rules_check(outcome, artifact_metadata)

    session.add(outcome.llm_verdict_row)

    verdict = _combine_gate_verdict(outcome.verdict.verdict, rules_report)
    await session.refresh(submission, with_for_update=True)
    await session.refresh(analysis_run, with_for_update=True)
    if analysis_run.status != "running" or analysis_run.lease_owner != actor:
        return _stale_analysis_summary(analysis_run, submission, verdict)
    if submission.raw_status not in {"ast_running", "llm_running"}:
        analysis_run.status = "stale_ignored"
        analysis_run.finished_at = datetime.now(UTC)
        analysis_run.lease_owner = None
        analysis_run.lease_expires_at = None
        analysis_run.heartbeat_at = None
        analysis_run.report_json = _stable_json(
            {
                **_json_object(analysis_run.report_json),
                "stale_completion": {
                    "actor": actor,
                    "submission_status": submission.raw_status,
                },
            }
        )
        await session.flush()
        return _stale_analysis_summary(analysis_run, submission, verdict)
    analysis_run.status = "completed"
    analysis_run.verdict = verdict
    analysis_run.reason_codes_json = _stable_json(
        _merge_gate_reason_codes(outcome.verdict.policy_flags, rules_report)
    )
    ast_verdict, ast_verdict_reason = _ast_similarity_verdict(similarity_evidence)
    # Evidence snippets (AST, similarity, and rules_check) are persisted RAW here
    # for audit. report_json is an internal-only surface; before any public
    # exposure the API lot MUST pass every snippet through
    # ``agent_challenge.evaluation.task_events.redact_secrets``.
    analysis_run.report_json = _stable_json(
        {
            "ast": {
                **ast_report.to_dict(),
                "verdict": ast_verdict,
                "verdict_reason": ast_verdict_reason,
            },
            "llm_verdict": outcome.verdict.model_dump(),
            "similarity": {
                "algorithm_version": ALGORITHM_VERSION,
                "matches": similarity_evidence,
            },
            "rules_check": (
                rules_report.to_json_compatible() if rules_report is not None else None
            ),
        }
    )
    analysis_run.finished_at = datetime.now(UTC)
    analysis_run.lease_owner = None
    analysis_run.lease_expires_at = None
    analysis_run.heartbeat_at = None
    await session.flush()

    job = await _apply_verdict(
        session=session,
        submission=submission,
        analysis_run=analysis_run,
        verdict=verdict,
        actor=actor,
        rationale=outcome.verdict.rationale,
    )
    return AnalysisSummary(
        analysis_run_id=analysis_run.id,
        submission_id=submission.id,
        verdict=verdict,
        status=submission.raw_status,
        evaluation_job_id=job.job_id if job else None,
    )


async def reclaim_expired_analysis_runs(
    session: AsyncSession,
    *,
    lease_owner: str,
) -> AgentSubmission | None:
    now = datetime.now(UTC)
    analysis_run = await session.scalar(
        select(AnalysisRun)
        .join(AnalysisRun.submission)
        .where(AnalysisRun.status == "running")
        .where(AnalysisRun.lease_expires_at.is_not(None))
        .where(AnalysisRun.lease_expires_at <= now)
        .where(AgentSubmission.raw_status.in_({"ast_running", "llm_running"}))
        .order_by(AnalysisRun.started_at, AnalysisRun.id)
        .limit(1)
    )
    if analysis_run is None:
        return None
    await session.refresh(analysis_run, attribute_names=["submission"])
    analysis_run.status = "expired_reclaimed"
    analysis_run.finished_at = now
    analysis_run.lease_owner = None
    analysis_run.lease_expires_at = None
    analysis_run.heartbeat_at = None
    analysis_run.report_json = _stable_json(
        {
            **_json_object(analysis_run.report_json),
            "lease_recovery": {
                "reclaimed_by": lease_owner,
                "reclaimed_at": now.isoformat(),
            },
        }
    )
    if analysis_run.submission.raw_status in {"ast_running", "llm_running"}:
        await ensure_submission_status(
            session,
            analysis_run.submission,
            "analysis_queued",
            actor="lease-reaper",
            reason="blocking_analysis_lease_expired",
            metadata={
                "analysis_run_id": analysis_run.id,
                "lease_owner": lease_owner,
                "lease_recovered_at": now.isoformat(),
            },
        )
    await session.flush()
    return analysis_run.submission


def _stale_analysis_summary(
    analysis_run: AnalysisRun,
    submission: AgentSubmission,
    verdict: str,
) -> AnalysisSummary:
    return AnalysisSummary(
        analysis_run_id=analysis_run.id,
        submission_id=submission.id,
        verdict=verdict,
        status=submission.raw_status,
        evaluation_job_id=None,
    )


def build_configured_lifecycle_reviewer(
    provider: LlmReviewProvider | None = None,
) -> KimiLlmReviewer:
    return KimiLlmReviewer(
        provider=provider or _build_configured_review_provider(),
        max_attempts=settings.llm_reviewer_max_attempts,
        timeout_seconds=settings.llm_reviewer_timeout_seconds,
        expected_model=settings.llm_reviewer_expected_model,
        prompt_cache_enabled=settings.llm_reviewer_prompt_cache_enabled,
    )


def _build_configured_review_provider() -> GatewayReviewProvider:
    base_url = settings.llm_gateway_base_url or ""
    return GatewayReviewProvider(
        gateway_token=settings.llm_gateway_token,
        base_url=gateway_llm_base_url(base_url) if base_url else "",
    )


def gateway_llm_base_url(gateway_base_url: str) -> str:
    """Return the master gateway's OpenAI-compatible ``/llm/v1`` base URL."""

    return f"{gateway_base_url.rstrip('/')}/llm/v1"


def build_configured_rules_reviewer() -> GatewayRulesReviewer | None:
    """VAL-ACAT-015: do **not** wire Base master LLM gateway for rules review.

    Production rules trust is the measured review harness (Phala + OpenRouter
    digests). Residual ``llm_gateway_*`` Settings must not activate a Base
    ``GatewayReviewProvider`` consumer here. Returns ``None`` so the pre-eval
    gate skips this path rather than requiring removed gateway secrets.
    """

    return None


def _resolve_rules_reviewer() -> ReviewerLike | None:
    reviewer = build_configured_rules_reviewer()
    # Configured production path never returns a Base-gateway rules reviewer.
    # Test suites inject a fake reviewer directly into run_rules_analyzer /
    # lifecycle seams when offline rules review is exercised.
    return reviewer


async def _maybe_run_rules_check(
    outcome: LlmReviewOutcome,
    artifact_metadata: ArtifactMetadata,
) -> AnalyzerPipelineReport | None:
    # Only run for a genuine model verdict; a fail-closed synthetic escalate is
    # already heading to standby/admin review, so the rules gateway call is
    # skipped.
    if outcome.disposition != "verdict":
        return None
    reviewer = _resolve_rules_reviewer()
    if reviewer is None:
        return None
    return await asyncio.to_thread(_run_rules_check, artifact_metadata, reviewer)


def _run_rules_check(
    artifact_metadata: ArtifactMetadata,
    reviewer: ReviewerLike,
) -> AnalyzerPipelineReport:
    # Imported lazily: analyzer.pipeline pulls in the evaluation package, and a
    # top-level import here would create an evaluation<->analyzer import cycle.
    from agent_challenge.analyzer.pipeline import run_rules_analyzer

    # Materialize the submission source into a scratch dir so the rules analyzer
    # (which needs a real directory) can scan it. The checked-in service .rules
    # policies are loaded from the default repository root, not the submission.
    with tempfile.TemporaryDirectory(prefix="rules-check-") as scratch_dir:
        extract_zip_to_directory(
            zip_path=artifact_metadata.artifact_path,
            target_directory=scratch_dir,
        )
        return run_rules_analyzer(scratch_dir, reviewer=reviewer)


_GATE_VERDICT_SEVERITY = {"allow": 0, "escalate": 1, "reject": 2}
_RULES_VERDICT_TO_GATE = {
    "valid": "allow",
    "suspicious": "escalate",
    "error": "escalate",
    "invalid": "reject",
}


def _combine_gate_verdict(
    llm_verdict: str,
    rules_report: AnalyzerPipelineReport | None,
) -> str:
    """Combine the LLM-gate verdict with the rules-check; most severe wins.

    A rules ``invalid`` (including any deterministic static hardcoding hit) blocks
    with ``reject``; ``suspicious``/uncertain escalates to admin review; a clean
    rules result never downgrades the KimiLlmReviewer verdict. A rules-reviewer
    GATEWAY INFRA failure (transient outage) is treated as "unaffected": it never
    hard-rejects or escalates, so the KimiLlmReviewer verdict is preserved.
    """

    if rules_report is None or _is_rules_infra_failure(rules_report):
        return llm_verdict
    rules_verdict = _RULES_VERDICT_TO_GATE.get(rules_report.overall_verdict, "escalate")
    if rules_report.hardcoding_findings:
        rules_verdict = "reject"
    return max(
        (llm_verdict, rules_verdict),
        key=lambda candidate: _GATE_VERDICT_SEVERITY.get(candidate, 1),
    )


def _is_rules_infra_failure(rules_report: AnalyzerPipelineReport) -> bool:
    return RULES_REVIEWER_INFRA_REASON in rules_report.reason_codes


def _merge_gate_reason_codes(
    llm_policy_flags: list[str],
    rules_report: AnalyzerPipelineReport | None,
) -> list[str]:
    codes = list(llm_policy_flags)
    if rules_report is not None:
        for code in rules_report.reason_codes:
            if code not in codes:
                codes.append(code)
    return codes


def _ast_similarity_verdict(
    similarity_evidence: list[Mapping[str, object]],
) -> tuple[str, str]:
    max_score = 0.0
    for match in similarity_evidence:
        try:
            score = float(match.get("score_percent"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        max_score = max(max_score, score)
    high = settings.analyzer_similarity_high_risk_threshold
    medium = settings.analyzer_similarity_medium_risk_threshold
    if max_score >= high:
        return "flagged", f"max delta similarity {max_score:.2f}% >= high-risk {high:.2f}%"
    if max_score >= medium:
        return (
            "uncertain",
            f"max delta similarity {max_score:.2f}% in medium-risk band "
            f"[{medium:.2f}%, {high:.2f}%)",
        )
    return "clean", f"max delta similarity {max_score:.2f}% below medium-risk {medium:.2f}%"


async def _mark_llm_standby(
    *,
    session: AsyncSession,
    submission: AgentSubmission,
    analysis_run: AnalysisRun,
    actor: str,
    ast_report: Mapping[str, object],
    similarity_evidence: list[Mapping[str, object]],
    reason: str,
    provider_name: str | None,
    model_name: str | None,
) -> AnalysisSummary:
    now = datetime.now(UTC)
    metadata = _llm_standby_metadata(
        analysis_run_id=analysis_run.id,
        reason=reason,
        provider_name=provider_name,
        model_name=model_name,
    )
    analysis_run.status = "llm_standby"
    analysis_run.verdict = "standby"
    analysis_run.reason_codes_json = _stable_json([reason])
    analysis_run.report_json = _stable_json(
        {
            "ast": ast_report,
            "llm_standby": metadata,
            "similarity": {
                "algorithm_version": ALGORITHM_VERSION,
                "matches": similarity_evidence,
            },
        }
    )
    analysis_run.finished_at = now
    analysis_run.lease_owner = None
    analysis_run.lease_expires_at = None
    analysis_run.heartbeat_at = None
    await ensure_submission_status(
        session,
        submission,
        "llm_standby",
        actor=actor,
        reason=reason,
        metadata=metadata,
    )
    await session.flush()
    return AnalysisSummary(
        analysis_run_id=analysis_run.id,
        submission_id=submission.id,
        verdict="standby",
        status=submission.raw_status,
        evaluation_job_id=None,
    )


def _llm_standby_reason(
    exc: LlmProviderUnavailable,
    *,
    uses_configured_reviewer: bool,
) -> str:
    if uses_configured_reviewer and not _llm_provider_ready():
        return "missing_llm_gateway_token"
    if isinstance(exc, LlmProviderRateLimited):
        return "llm_provider_rate_limited"
    if isinstance(exc, LlmProviderTimeout):
        return "llm_provider_timeout"
    return "llm_provider_unavailable"


def _llm_standby_metadata(
    *,
    analysis_run_id: int,
    reason: str,
    provider_name: str | None,
    model_name: str | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "analysis_run_id": analysis_run_id,
        "reason": reason,
    }
    if provider_name:
        metadata["provider"] = provider_name
    if model_name:
        metadata["model"] = model_name
    return metadata


def _llm_provider_ready() -> bool:
    return bool(settings.llm_gateway_base_url and settings.llm_gateway_token)


def _is_retryable_review_failure(reason: str) -> bool:
    """Classify a fail-closed reason using the (now-live) retry policy.

    A reason is retryable only when it is explicitly included and not excluded,
    so operators can move a reason between standby and admin-escalation purely
    via ``llm_reviewer_retry_include`` / ``llm_reviewer_retry_exclude``.
    """

    if reason in set(settings.llm_reviewer_retry_exclude):
        return False
    return reason in set(settings.llm_reviewer_retry_include)


async def _llm_standby_cycle_count(session: AsyncSession, submission_id: int) -> int:
    """Count prior transient/tool-miss standby cycles for a submission.

    Missing-gateway-token standby is excluded: it parks waiting for the provider
    to become configured (re-queued on token presence) and must not consume the
    transient retry ceiling.
    """

    waiting_reason = _stable_json(["missing_llm_gateway_token"])
    count = await session.scalar(
        select(func.count(AnalysisRun.id))
        .where(AnalysisRun.submission_id == submission_id)
        .where(AnalysisRun.status == "llm_standby")
        .where(AnalysisRun.reason_codes_json != waiting_reason)
    )
    return int(count or 0)


def _reviewer_missing_gateway_token(reviewer: AnalyzerReviewer) -> bool:
    provider = getattr(reviewer, "provider", None)
    if getattr(provider, "provider_name", None) != "gateway":
        return False
    return not getattr(provider, "gateway_token", None)


def _reviewer_provider_name(reviewer: AnalyzerReviewer) -> str | None:
    provider = getattr(reviewer, "provider", None)
    value = getattr(provider, "provider_name", None)
    return value if isinstance(value, str) and value else None


def _reviewer_model_name(reviewer: AnalyzerReviewer) -> str | None:
    provider = getattr(reviewer, "provider", None)
    value = getattr(provider, "model_name", None)
    return value if isinstance(value, str) and value else None


async def _apply_verdict(
    *,
    session: AsyncSession,
    submission: AgentSubmission,
    analysis_run: AnalysisRun,
    verdict: str,
    actor: str,
    rationale: str,
) -> EvaluationJob | None:
    if verdict == "allow":
        metadata = {"analysis_run_id": analysis_run.id}
        await ensure_submission_status(
            session,
            submission,
            "analysis_allowed",
            actor=actor,
            reason="blocking_analysis_allowed",
            metadata=metadata,
        )
        # The attested review application owns authorization in full-attested
        # mode.  Analyzer allow is retained as a safe status event only and
        # must never enter the legacy env/job topology.
        if settings.attested_review_enabled and settings.phala_attestation_enabled:
            return None
        env_ready = await ensure_miner_env_ready_for_evaluation(
            session,
            submission,
            actor=actor,
            metadata=metadata,
        )
        if not env_ready:
            await ensure_submission_status(
                session,
                submission,
                "waiting_miner_env",
                actor=actor,
                reason="waiting_miner_env",
                metadata=metadata,
            )
            return None
        job = await enqueue_evaluation_job_for_submission(
            session,
            submission,
            confirmed_miner_env=True,
        )
        if job is not None and job.trigger_reason is None:
            job.trigger_reason = "analysis_allowed_env_ready"
        return job
    if verdict == "reject":
        await ensure_submission_status(
            session,
            submission,
            "analysis_rejected",
            actor=actor,
            reason="blocking_analysis_rejected",
            metadata={"analysis_run_id": analysis_run.id},
        )
        return None
    if verdict == "escalate":
        before_status = submission.effective_status
        await ensure_submission_status(
            session,
            submission,
            "analysis_escalated",
            actor=actor,
            reason="blocking_analysis_escalated",
            metadata={"analysis_run_id": analysis_run.id},
        )
        await ensure_submission_status(
            session,
            submission,
            "admin_paused",
            actor=actor,
            reason="blocking_analysis_admin_review_required",
            metadata={"analysis_run_id": analysis_run.id},
        )
        session.add(
            AdminReviewDecision(
                submission_id=submission.id,
                reviewer_hotkey="system",
                decision="pending_analysis_review",
                reason=rationale[:4000],
                before_effective_status=before_status,
                after_effective_status=submission.effective_status,
                metadata_json=_stable_json({"analysis_run_id": analysis_run.id}),
            )
        )
        await session.flush()
        return None
    raise ValueError(f"unsupported analysis verdict: {verdict}")


async def _source_artifact(
    session: AsyncSession,
    submission: AgentSubmission,
) -> SubmissionArtifact:
    artifact = await session.scalar(
        select(SubmissionArtifact)
        .where(SubmissionArtifact.submission_id == submission.id)
        .where(SubmissionArtifact.artifact_kind == "source_zip")
        .order_by(SubmissionArtifact.id)
        .limit(1)
    )
    if artifact is None:
        raise ValueError(f"submission {submission.id} has no source artifact")
    return artifact


def _artifact_metadata(artifact: SubmissionArtifact) -> ArtifactMetadata:
    metadata = _json_object(artifact.metadata_json)
    manifest_data = metadata.get("manifest")
    if not isinstance(manifest_data, dict):
        raise ValueError(f"artifact {artifact.id} has no manifest metadata")
    return ArtifactMetadata(
        zip_sha256=artifact.sha256 or str(manifest_data["zip_sha256"]),
        zip_size_bytes=artifact.size_bytes or int(manifest_data["zip_size_bytes"]),
        artifact_path=artifact.uri,
        manifest=ZipArtifactManifest.from_dict(manifest_data),
        manifest_path=str(metadata.get("manifest_path") or ""),
    )


def _json_object(value: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _datetime_to_json(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
