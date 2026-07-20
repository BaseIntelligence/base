"""Transactional lifecycle ledger for immutable review sessions and attempts."""

from __future__ import annotations

import base64
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.core.models import (
    AgentSubmission,
    ReviewAssignment,
    ReviewNonce,
    ReviewOperatorApproval,
    ReviewRulesSnapshot,
    ReviewSession,
)
from agent_challenge.sdk.auth import load_internal_token
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.submissions.state_machine import ensure_submission_status

from .canonical import canonical_json_v1, parse_json_object
from .deployment import ReviewDeploymentError, validate_review_deployed_acknowledgement
from .schemas import (
    ReviewInputConfig,
    build_review_assignment,
    build_rules_bundle,
    rules_snapshot_sha256,
    validate_model_call_started,
    validate_review_infrastructure_failure,
)

REVIEW_ASSIGNMENT_TTL_SECONDS = 1800
MAX_REVIEW_ASSIGNMENTS_PER_SESSION = 16
REVIEW_NONCE_PURPOSE = "review"
_ACTIVE_PHASES = frozenset(
    {"review_queued", "review_cvm_running", "review_provider_standby", "review_verifying"}
)
_RETRYABLE_PHASES = frozenset({"review_cancelled", "review_expired", "review_error"})
_AUDIT_CURSOR_DOMAIN = b"agent-challenge:review-audit-cursor:v1:"
# In-process mutation windows keyed by session_id. Durable pruning for expired
# outstanding nonce/receipt rows still happens through the database.
_MUTATION_WINDOWS: dict[str, list[datetime]] = {}


class ReviewConflict(ValueError):
    """A lifecycle request does not match the immutable active state."""


class ReviewRateLimited(ValueError):
    """A session exceeded its configured mutation rate or outstanding cap.

    Intentionally independent of :class:`ReviewConflict` so intake and route
    handlers can map concurrency/rate floods to HTTP 429 without collapsing into
    conflict (409) or availability (503) catch-all branches.
    """


class ReviewNotFound(ValueError):
    """The referenced review object does not exist."""


class ReviewCapabilityError(PermissionError):
    """The bearer capability cannot access this immutable assignment input."""


@dataclass(frozen=True)
class CreatedReviewSession:
    session: ReviewSession
    assignment: ReviewAssignment
    session_token: str
    harness_identity: dict[str, Any] | None = None


async def create_review_session(
    session: AsyncSession,
    *,
    submission: AgentSubmission,
    artifact_bytes: bytes,
    rules_files: dict[str, bytes],
    rules_revision_id: str,
    settings: ChallengeSettings,
    now: datetime | None = None,
    input_config: ReviewInputConfig | None = None,
    manifest_sha256: str | None = None,
    manifest_entries_sha256: str | None = None,
    entry_script: str | None = None,
    entry_script_identity: str | None = None,
    entry_script_bytes: bytes | None = None,
    harness_kind: str | None = None,
    openrouter_call_attempted: bool = False,
) -> CreatedReviewSession:
    """Create exactly one durable session and first immutable assignment.

    The caller remains responsible for committing its broader intake transaction.
    This function performs no network, CVM, or provider work.

    Product harness gate (ZIP + shipping entry + .rules before OpenRouter):
    ``tools/agent_parity_harness.py`` and other non-product harness kinds are
    refused with stable ``product_review_*`` codes. Default entry identity is
    the shipping selfdeploy marker.

    **Retains harness identity materials** on the session (JSON + digest) so
    later admission can compare against cited identity without re-deriving
    from forgetting call-site kwargs. Cache columns alone never authorize.
    """

    now = _as_utc(now)
    existing = await session.scalar(
        select(ReviewSession).where(ReviewSession.submission_id == submission.id)
    )
    if existing is not None:
        current = await _current_assignment(session, existing)
        if current is None:
            raise ReviewConflict("review session has no current assignment")
        cached_identity: dict[str, Any] | None = None
        if existing.harness_identity_json:
            try:
                parsed = json.loads(existing.harness_identity_json)
                if isinstance(parsed, dict):
                    cached_identity = parsed
            except (TypeError, ValueError):
                cached_identity = None
        return CreatedReviewSession(
            session=existing,
            assignment=current,
            session_token=_derive_session_token(settings, current.assignment_id),
            harness_identity=cached_identity,
        )

    # Sole product review entry: ZIP + entry script + rules pack digests.
    # agent_parity_harness is never a production review path.
    from agent_challenge.review.harness_entry import (
        ProductHarnessAdmissionError,
        admit_product_review_entry,
    )

    try:
        product_identity = admit_product_review_entry(
            agent_zip_bytes=artifact_bytes,
            entry_script=entry_script,
            entry_script_bytes=entry_script_bytes,
            entry_script_identity=entry_script_identity or "python -m agent_challenge.selfdeploy",
            rules_files=rules_files if rules_files else None,
            harness_kind=harness_kind,
            openrouter_call_attempted=openrouter_call_attempted,
        )
    except ProductHarnessAdmissionError as exc:
        raise ReviewConflict(exc.code) from exc

    identity_dict = product_identity.as_dict()
    identity_json = json.dumps(identity_dict, sort_keys=True, separators=(",", ":"))
    identity_sha = product_identity.session_identity_sha256()

    # Challenge-domain receive for submission/send (not guest wall clock).
    # Prefer durable AgentSubmission.submitted_at; fall back to challenge now.
    submission_receive_source = getattr(submission, "submitted_at", None) or now
    if isinstance(submission_receive_source, datetime):
        receive_dt = (
            submission_receive_source
            if submission_receive_source.tzinfo is not None
            else submission_receive_source.replace(tzinfo=UTC)
        )
        submission_received_at_ms = int(receive_dt.timestamp() * 1000)
    else:
        submission_received_at_ms = int(now.timestamp() * 1000)

    artifact_sha256 = sha256(artifact_bytes).hexdigest()
    if submission.zip_sha256 and artifact_sha256 != submission.zip_sha256:
        raise ReviewConflict("committed artifact bytes do not match submission digest")
    max_files = int(getattr(settings, "review_max_rules_files", 128))
    max_rules_bytes = int(getattr(settings, "review_max_rules_bytes", 1_048_576))
    if len(rules_files) > max_files:
        raise ReviewConflict("rules snapshot exceeds review_max_rules_files")
    if sum(len(value) for value in rules_files.values()) > max_rules_bytes:
        raise ReviewConflict("rules snapshot exceeds review_max_rules_bytes")
    bundle = build_rules_bundle(revision_id=rules_revision_id, files=rules_files)
    snapshot_bytes = canonical_json_v1(bundle)
    snapshot_digest = rules_snapshot_sha256(bundle)
    manifest_digest = manifest_sha256 or sha256(canonical_json_v1({"entries": []})).hexdigest()
    entries_digest = manifest_entries_sha256 or sha256(canonical_json_v1([])).hexdigest()

    review_session = ReviewSession(
        session_id=_new_id("rs"),
        submission_id=submission.id,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=len(artifact_bytes),
        manifest_sha256=manifest_digest,
        manifest_entries_sha256=entries_digest,
        harness_identity_json=identity_json,
        harness_identity_sha256=identity_sha,
        submission_received_at_ms=submission_received_at_ms,
    )
    session.add(review_session)
    await session.flush()
    snapshot = await session.scalar(
        select(ReviewRulesSnapshot).where(ReviewRulesSnapshot.snapshot_sha256 == snapshot_digest)
    )
    if snapshot is None:
        snapshot = ReviewRulesSnapshot(
            session_id=review_session.id,
            revision_id=rules_revision_id,
            snapshot_sha256=snapshot_digest,
            canonical_bytes=snapshot_bytes,
        )
        session.add(snapshot)
    created = await _issue_assignment(
        session,
        review_session=review_session,
        submission=submission,
        snapshot=snapshot,
        settings=settings,
        attempt=1,
        now=now,
        input_config=input_config,
    )
    return CreatedReviewSession(
        session=created.session,
        assignment=created.assignment,
        session_token=created.session_token,
        harness_identity=identity_dict,
    )


async def retry_review_assignment(
    session: AsyncSession,
    *,
    session_row: ReviewSession,
    expected_assignment_id: str,
    settings: ChallengeSettings,
    now: datetime | None = None,
    approval_id: str | None = None,
    refresh_rules_files: dict[str, bytes] | None = None,
    refresh_rules_revision_id: str | None = None,
    input_config: ReviewInputConfig | None = None,
) -> CreatedReviewSession:
    """Supersede an eligible attempt, retaining immutable predecessor history."""

    now = _as_utc(now)
    current = await _current_assignment(session, session_row, lock=True)
    if current is None or current.assignment_id != expected_assignment_id:
        raise ReviewConflict("expected assignment is not current")
    await expire_assignment_if_needed(session, current, now=now)
    if current.phase in _ACTIVE_PHASES:
        raise ReviewConflict("review assignment is active")
    if current.phase not in _RETRYABLE_PHASES:
        if current.phase not in {"review_rejected", "review_escalated"}:
            raise ReviewConflict("review assignment is not retryable")
        if approval_id is None:
            raise ReviewConflict("operator approval is required")

    max_attempts = int(
        getattr(settings, "review_max_assignments_per_session", MAX_REVIEW_ASSIGNMENTS_PER_SESSION)
    )
    if current.attempt >= max_attempts:
        raise ReviewConflict("review assignment retry limit reached")
    await enforce_review_session_mutation_budget(
        session,
        session_row=session_row,
        settings=settings,
        now=now,
    )
    snapshot = await _snapshot_for_digest(session, session_row.id, current.rules_snapshot_sha256)
    if refresh_rules_files is not None:
        if approval_id is None:
            raise ReviewConflict("rules refresh requires operator approval")
        approval = await _consume_approval(
            session,
            approval_id=approval_id,
            session_row=session_row,
            assignment=current,
            action="refresh_rules",
            now=now,
        )
        if approval.rules_revision_id is None:
            raise ReviewConflict("rules refresh approval has no revision")
        if (
            refresh_rules_revision_id is not None
            and approval.rules_revision_id != refresh_rules_revision_id
        ):
            raise ReviewConflict("rules refresh approval revision does not match snapshot")
        bundle = build_rules_bundle(
            revision_id=refresh_rules_revision_id or approval.rules_revision_id,
            files=refresh_rules_files,
        )
        snapshot = await _store_snapshot(session, session_row, bundle)
    elif approval_id is not None and current.phase in {"review_rejected", "review_escalated"}:
        await _consume_approval(
            session,
            approval_id=approval_id,
            session_row=session_row,
            assignment=current,
            action="retry_policy",
            now=now,
        )

    current.active_key = None
    current.capability_state = "revoked"
    current.finished_at = current.finished_at or now
    if current.phase in _ACTIVE_PHASES:
        current.phase = "review_cancelled"
    await _set_nonce_state(session, current, "revoked", now=now)
    created = await _issue_assignment(
        session,
        review_session=session_row,
        submission=await _submission_for_session(session, session_row),
        snapshot=snapshot,
        settings=settings,
        attempt=current.attempt + 1,
        now=now,
        input_config=input_config,
    )
    await record_review_submission_status(
        session,
        review_session=session_row,
        assignment=created.assignment,
        raw_status="review_queued",
        reason="review_assignment_retried",
    )
    return created


async def cancel_review_assignment(
    session: AsyncSession,
    *,
    session_row: ReviewSession,
    expected_assignment_id: str,
    now: datetime | None = None,
    settings: ChallengeSettings | None = None,
) -> ReviewAssignment:
    """Cancel only the named current active assignment, never a replacement."""

    now = _as_utc(now)
    assignment = await _current_assignment(session, session_row, lock=True)
    if assignment is None or assignment.assignment_id != expected_assignment_id:
        raise ReviewConflict("expected assignment is not current")
    await expire_assignment_if_needed(session, assignment, now=now)
    if assignment.phase == "review_cancelled":
        return assignment
    if assignment.review_report_envelope_json is not None:
        raise ReviewConflict("receipted review report must resume verification")
    if assignment.phase not in _ACTIVE_PHASES:
        raise ReviewConflict("review assignment is terminal")
    if settings is not None:
        await enforce_review_session_mutation_budget(
            session, session_row=session_row, settings=settings, now=now
        )
    assignment.phase = "review_cancelled"
    assignment.active_key = None
    assignment.capability_state = "revoked"
    assignment.finished_at = now
    assignment.reason_code = "cancelled"
    await _set_nonce_state(session, assignment, "revoked", now=now)
    await record_review_submission_status(
        session,
        review_session=session_row,
        assignment=assignment,
        raw_status="review_cancelled",
        reason="review_cancelled",
    )
    return assignment


async def mark_review_deployed(
    session: AsyncSession,
    *,
    session_row: ReviewSession,
    expected_assignment_id: str,
    deployed_receipt: dict[str, Any],
    now: datetime | None = None,
    settings: ChallengeSettings | None = None,
) -> ReviewAssignment:
    """Record miner deployment metadata as informational, never trust evidence."""

    now = _as_utc(now)
    assignment = await _current_assignment(session, session_row, lock=True)
    if assignment is None or assignment.assignment_id != expected_assignment_id:
        raise ReviewConflict("expected assignment is not current")
    await expire_assignment_if_needed(session, assignment, now=now)
    try:
        assignment_body = json.loads(assignment.assignment_bytes)
        validate_review_deployed_acknowledgement(assignment_body, deployed_receipt)
    except (json.JSONDecodeError, ReviewDeploymentError) as exc:
        raise ReviewConflict(
            "deployment receipt is not bound to immutable review assignment"
        ) from exc
    receipt = json.dumps(deployed_receipt, sort_keys=True, separators=(",", ":"))
    if assignment.deployed_receipt_json is not None:
        if assignment.deployed_receipt_json != receipt:
            raise ReviewConflict("deployment receipt conflicts with prior receipt")
        return assignment
    if assignment.phase != "review_queued":
        raise ReviewConflict("review assignment cannot be deployed in its current phase")
    if settings is not None:
        await enforce_review_session_mutation_budget(
            session, session_row=session_row, settings=settings, now=now
        )
    assignment.phase = "review_cvm_running"
    assignment.deployed_at = now
    assignment.deployed_receipt_json = receipt
    await record_review_submission_status(
        session,
        review_session=session_row,
        assignment=assignment,
        raw_status="review_cvm_running",
        reason="review_cvm_deployed",
    )
    return assignment


async def mark_model_call_started(
    session: AsyncSession,
    *,
    assignment: ReviewAssignment,
    marker: dict[str, Any],
    now: datetime | None = None,
    settings: ChallengeSettings | None = None,
) -> bool:
    """Durably record one exact model-call marker before any network exchange.

    Returns ``True`` on the first durable record and ``False`` for an identical
    replay. A different marker can never replace the first record. Concurrent
    first writers are serialized with a row lock plus a conditional UPDATE that
    admits only one transition from an unset marker.
    """

    now = _as_utc(now)
    try:
        marker_bytes = validate_model_call_started(marker)
    except ValueError as exc:
        raise ReviewConflict("model call marker is malformed") from exc
    if marker["assignment_id"] != assignment.assignment_id:
        raise ReviewConflict("model call marker assignment mismatches path")
    # Serialize concurrent marker advertisers for this assignment. SQLite may
    # ignore FOR UPDATE, so the subsequent conditional UPDATE is the real CAS.
    locked = await session.scalar(
        select(ReviewAssignment).where(ReviewAssignment.id == assignment.id).with_for_update()
    )
    if locked is None:
        raise ReviewConflict("model call marker assignment is missing")
    # Keep the caller's detached/ORM instance views consistent with the lock.
    assignment = locked
    await expire_assignment_if_needed(session, assignment, now=now)
    if assignment.capability_state != "active":
        raise ReviewConflict("model call marker capability is revoked")
    if assignment.phase not in _ACTIVE_PHASES:
        raise ReviewConflict("model call marker assignment is terminal")
    max_string = int(
        getattr(settings, "review_max_string_bytes", 16_384) if settings is not None else 16_384
    )
    if len(marker_bytes) > max_string:
        raise ReviewConflict("model call marker exceeds review_max_string_bytes")
    marker_json = marker_bytes.decode("utf-8")
    marker_digest = sha256(marker_bytes).hexdigest()
    if assignment.model_call_started_json is not None:
        if assignment.model_call_started_json != marker_json:
            raise ReviewConflict("model call marker conflicts with durable marker")
        return False
    if assignment.phase != "review_cvm_running":
        raise ReviewConflict("model call marker requires a running review CVM")
    # Flush any in-session phase transition (e.g. tests set phase on the ORM
    # object) so the conditional update sees the same state as validation.
    await session.flush()
    # Atomic first-writer-wins: only succeed when the durable marker was still null.
    claimed = await session.execute(
        update(ReviewAssignment)
        .where(ReviewAssignment.id == assignment.id)
        .where(ReviewAssignment.model_call_started_json.is_(None))
        .where(ReviewAssignment.capability_state == "active")
        .values(
            model_call_started_json=marker_json,
            model_call_started_sha256=marker_digest,
            planned_request_sha256=str(marker["planned_request_sha256"]),
            request_body_sha256=str(marker["request_body_sha256"]),
            request_body_length=int(marker["request_body_length"]),
            model_call_started_at=now,
            phase="review_provider_standby",
        )
    )
    if claimed.rowcount != 1:
        await session.refresh(assignment)
        if (
            assignment.model_call_started_json is not None
            and assignment.model_call_started_json == marker_json
        ):
            return False
        raise ReviewConflict("model call marker conflicts with durable marker")
    await session.refresh(assignment)
    review_session = await session.get(ReviewSession, assignment.session_id)
    if review_session is not None:
        await record_review_submission_status(
            session,
            review_session=review_session,
            assignment=assignment,
            raw_status="review_provider_standby",
            reason="review_model_call_started",
        )
    await session.flush()
    return True


async def record_review_infrastructure_failure(
    session: AsyncSession,
    *,
    assignment: ReviewAssignment,
    failure: dict[str, Any],
    now: datetime | None = None,
    settings: ChallengeSettings | None = None,
) -> bool:
    """Terminalize one capability-authenticated no-report infrastructure failure."""

    now = _as_utc(now)
    try:
        failure_bytes = validate_review_infrastructure_failure(failure)
    except ValueError as exc:
        raise ReviewConflict("review infrastructure failure is malformed") from exc
    if failure["assignment_id"] != assignment.assignment_id:
        raise ReviewConflict("review failure assignment mismatches path")
    max_string = int(
        getattr(settings, "review_max_string_bytes", 16_384) if settings is not None else 16_384
    )
    if len(failure_bytes) > max_string:
        raise ReviewConflict("review failure exceeds review_max_string_bytes")
    # A durable report receipt owns recovery: never replace it with review_error.
    if assignment.review_report_envelope_json is not None:
        raise ReviewConflict("receipted review report must resume verification")
    planned_digest = failure["planned_request_sha256"]
    if assignment.planned_request_sha256 is None:
        if planned_digest is not None:
            raise ReviewConflict("unannounced review failure must not name a request plan")
    elif planned_digest != assignment.planned_request_sha256:
        raise ReviewConflict("review failure plan does not match durable marker")
    failure_json = failure_bytes.decode("utf-8")
    failure_digest = sha256(failure_bytes).hexdigest()
    if assignment.infrastructure_failure_json is not None:
        if assignment.infrastructure_failure_json != failure_json:
            raise ReviewConflict("review failure conflicts with durable failure")
        return False
    if assignment.phase not in _ACTIVE_PHASES:
        raise ReviewConflict("review failure assignment is terminal")
    assignment.infrastructure_failure_json = failure_json
    assignment.infrastructure_failure_sha256 = failure_digest
    assignment.phase = "review_error"
    assignment.reason_code = str(failure["reason_code"])
    assignment.finished_at = now
    assignment.capability_state = "revoked"
    assignment.active_key = None
    await _set_nonce_state(session, assignment, "revoked", now=now)
    review_session = await session.get(ReviewSession, assignment.session_id)
    if review_session is not None:
        await record_review_submission_status(
            session,
            review_session=review_session,
            assignment=assignment,
            raw_status="review_error",
            reason="review_infrastructure_failure",
        )
    await session.flush()
    return True


def model_call_recovery_grace_seconds(settings: ChallengeSettings | None = None) -> float:
    """Grace window before an announced-but-unreceipted marker fails closed.

    Fresh markers are younger than the OpenRouter exchange + quote/report budget,
    so the reconciler must not convert them to ``report_generation_failed`` on
    every ~5s tick. Explicit ``review_model_call_recovery_grace_seconds`` wins;
    otherwise derive from the HTTPS total timeout plus verification and a
    bounded report/post slack so the window stays minutes-order.
    """

    if settings is not None:
        explicit = getattr(settings, "review_model_call_recovery_grace_seconds", None)
        if explicit is not None and float(explicit) > 0:
            return float(explicit)
    total = float(
        getattr(settings, "review_https_total_timeout_seconds", 300.0)
        if settings is not None
        else 300.0
    )
    verify = float(
        getattr(settings, "attestation_verification_timeout_seconds", 60.0)
        if settings is not None
        else 60.0
    )
    # Quote + report POST + worker jitter; keep minutes order vs reconciler poll.
    return total + verify + 120.0


async def recover_incomplete_model_calls(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    settings: ChallengeSettings | None = None,
) -> int:
    """Fail closed when an announced marker stays unreceipted past grace.

    Markers younger than the OpenRouter/report grace window stay active so a
    measured CVM can finish the exchange. Only stale markers terminalize, and
    only once, preserving the durable planned request digest.
    """

    moment = _as_utc(now)
    grace = timedelta(seconds=model_call_recovery_grace_seconds(settings))
    rows = list(
        (
            await session.scalars(
                select(ReviewAssignment)
                .where(ReviewAssignment.model_call_started_json.is_not(None))
                .where(ReviewAssignment.infrastructure_failure_json.is_(None))
                .where(ReviewAssignment.review_report_envelope_json.is_(None))
                .where(ReviewAssignment.phase.in_(_ACTIVE_PHASES))
                .with_for_update()
            )
        ).all()
    )
    recovered = 0
    for assignment in rows:
        started = assignment.model_call_started_at
        if started is None:
            # Defensive: marker JSON without a stamp is already inconsistent; fail closed.
            started_at = moment - grace - timedelta(seconds=1)
        else:
            started_at = _as_utc(started)
        if moment - started_at < grace:
            continue
        await record_review_infrastructure_failure(
            session,
            assignment=assignment,
            failure={
                "schema_version": 1,
                "assignment_id": assignment.assignment_id,
                "planned_request_sha256": assignment.planned_request_sha256,
                "reason_code": "report_generation_failed",
            },
            now=moment,
            settings=settings,
        )
        recovered += 1
    return recovered


async def recover_pending_review_reports(
    session: AsyncSession,
    *,
    quote_verifier: object,
    allowlist: object,
    now: datetime,
    evidence_settings: ChallengeSettings | None = None,
) -> int:
    """Resume only exact durable receipts left by a transient verifier outage.

    Recovery is receipt-time and evidence-aware: incomplete receipts without a
    durable evidence descriptor are skipped, and non-retryable parked outcomes
    are never reopened as a fresh verification path.
    """

    from .report import (
        ReviewMeasurementAllowlist,
        ReviewReportConflict,
        ReviewReportError,
        ReviewVerificationOutcome,
        submit_review_report,
    )

    if not isinstance(allowlist, ReviewMeasurementAllowlist):
        raise ReviewConflict("review recovery allowlist is invalid")
    rows = list(
        (
            await session.scalars(
                select(ReviewAssignment)
                .where(ReviewAssignment.review_report_envelope_json.is_not(None))
                .where(ReviewAssignment.review_report_received_at.is_not(None))
                .where(ReviewAssignment.review_evidence_descriptor_json.is_not(None))
                .where(ReviewAssignment.phase == "review_verifying")
                .with_for_update()
            )
        ).all()
    )
    recovered = 0
    for assignment in rows:
        if assignment.review_verification_outcome_json is not None:
            try:
                parked = parse_json_object(assignment.review_verification_outcome_json)
            except ValueError:
                continue
            if (
                not isinstance(parked, dict)
                or parked.get("status") != "verifier_unavailable"
                or not parked.get("retryable")
            ):
                # Incomplete/non-transient parked outcomes stay durable and are
                # never accepted after a post-receipt infrastructure failure.
                continue
        try:
            envelope = parse_json_object(assignment.review_report_envelope_json or "")
        except ValueError:
            continue
        try:
            outcome = await submit_review_report(
                session,
                assignment=assignment,
                envelope=envelope,
                evidence_objects=None,
                evidence_settings=evidence_settings,
                quote_verifier=quote_verifier,
                allowlist=allowlist,
                now=now,
            )
        except (ReviewReportConflict, ReviewReportError):
            continue
        if not isinstance(outcome, ReviewVerificationOutcome):  # pragma: no cover - type guard
            raise ReviewConflict("review recovery produced an invalid outcome")
        recovered += 1
    return recovered


async def deliver_prepare_token(
    session: AsyncSession,
    *,
    session_row: ReviewSession,
    settings: ChallengeSettings,
    now: datetime | None = None,
) -> tuple[ReviewAssignment, str | None]:
    """Return current assignment and derive its capability for active undepoyed rows.

    Capability redelivery policy (review-v9 residual):
    - **Active + not deployed** (``review_queued`` / undepoyed active phase):
      re-derive and return the same HMAC token on every prepare. Dry-run prepare
      must not spend the token or force miners into cancel+retry burn.
    - **After deployment receipt** (``deployed_at`` set / ``review_cvm_running``+):
      sticky-null — capability lives only in the measured CVM ciphertext.
    - **Revoked / terminal**: sticky-null.

    First document of delivery still stamps ``token_delivered_at`` for audit, but
    that stamp alone no longer blocks redelivery before deploy.
    """

    assignment = await _current_assignment(session, session_row, lock=True)
    if assignment is None:
        raise ReviewNotFound("review assignment not found")
    now_utc = _as_utc(now)
    await expire_assignment_if_needed(session, assignment, now=now_utc)
    if assignment.capability_state != "active":
        return assignment, None
    # Once the miner has ack'd deployment, token must not leave the CVM path.
    if assignment.deployed_at is not None or assignment.deployed_receipt_json is not None:
        return assignment, None
    if assignment.phase not in _ACTIVE_PHASES:
        return assignment, None
    # Pre-deploy: stamp first delivery for audit, always re-derive capability.
    if assignment.token_delivered_at is None:
        claimed = await session.execute(
            update(ReviewAssignment)
            .where(ReviewAssignment.id == assignment.id)
            .where(ReviewAssignment.token_delivered_at.is_(None))
            .values(token_delivered_at=now_utc)
        )
        if claimed.rowcount == 1:
            assignment.token_delivered_at = now_utc
        else:
            await session.refresh(assignment)
    return assignment, _derive_session_token(settings, assignment.assignment_id)


async def authenticate_assignment_capability(
    session: AsyncSession,
    *,
    assignment_id: str,
    token: str,
    now: datetime | None = None,
    allow_failure_replay: bool = False,
    allow_report_replay: bool = False,
) -> ReviewAssignment:
    """Authenticate a bounded, assignment-scoped bearer without retaining it.

    Every operation verifies:
    - the token digest for this exact assignment_id
    - the durable ReviewSession/submission binding exists for the assignment
    - revocation and `expires_at` (with expire-on-use for active unreceipted rows)
    - only already-receipted failure/report soft-auth may proceed after expiry
    """

    now = _as_utc(now)
    if not isinstance(token, str) or not token:
        raise ReviewCapabilityError("invalid assignment capability")
    # Tokens embed assignment_id as `{assignment_id}.{mac}`. Reject cross-scope
    # presentation early while remaining compatible with pure-mac legacy tokens
    # that do not contain a "." separator.
    if "." in token:
        embedded_id, _sep, _mac = token.partition(".")
        if embedded_id != assignment_id:
            raise ReviewCapabilityError("assignment capability scope is invalid")
    assignment = await session.scalar(
        select(ReviewAssignment)
        .where(ReviewAssignment.assignment_id == assignment_id)
        .with_for_update()
    )
    if assignment is None:
        raise ReviewCapabilityError("unknown assignment capability")
    token_digest = sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(token_digest, assignment.session_token_sha256):
        raise ReviewCapabilityError("invalid assignment capability")

    review_session = await session.get(ReviewSession, assignment.session_id)
    if review_session is None:
        raise ReviewCapabilityError("assignment session binding is missing")
    submission = await session.get(AgentSubmission, review_session.submission_id)
    if submission is None:
        raise ReviewCapabilityError("assignment submission binding is missing")
    # Binding proven for the path-scoped assignment_id: the capability is never
    # portable across sessions/submissions, and the ORM FKs (session_id,
    # submission_id) are re-loaded and asserted non-null on every use.
    if assignment.session_id != review_session.id:
        raise ReviewCapabilityError("assignment session scope is invalid")
    if review_session.submission_id != submission.id:
        raise ReviewCapabilityError("assignment submission scope is invalid")
    if assignment.assignment_id != assignment_id:
        raise ReviewCapabilityError("assignment identity scope is invalid")

    await expire_assignment_if_needed(session, assignment, now=now)

    # Soft authorisation for already-receipted failure/report only. This is the
    # sole post-expiry resumption path and still refuses unreceipted artifact,
    # rules, model-call, and first-report use.
    if allow_failure_replay and assignment.infrastructure_failure_json is not None:
        return assignment
    if allow_report_replay and assignment.review_report_envelope_json is not None:
        return assignment

    if assignment.capability_state != "active":
        raise ReviewCapabilityError("assignment capability is revoked")

    # Active unreceipted capability still requires now < expires_at.
    if now >= _as_utc(assignment.expires_at):
        raise ReviewCapabilityError("assignment capability is expired")
    return assignment


async def assignment_artifact(
    session: AsyncSession,
    *,
    assignment: ReviewAssignment,
) -> tuple[ReviewSession, AgentSubmission]:
    review_session = await session.get(ReviewSession, assignment.session_id)
    if review_session is None:
        raise ReviewNotFound("review session not found")
    submission = await _submission_for_session(session, review_session)
    return review_session, submission


async def assignment_rules(
    session: AsyncSession,
    *,
    assignment: ReviewAssignment,
) -> bytes:
    snapshot = await _snapshot_for_digest(
        session,
        assignment.session_id,
        assignment.rules_snapshot_sha256,
    )
    return snapshot.canonical_bytes


async def issue_operator_approval(
    session: AsyncSession,
    *,
    session_row: ReviewSession,
    assignment: ReviewAssignment,
    action: str,
    actor: str,
    rules_revision_id: str | None = None,
    now: datetime | None = None,
    ttl_seconds: int | None = None,
    settings: ChallengeSettings | None = None,
) -> ReviewOperatorApproval:
    """Create a short-lived one-use approval scoped to an immutable attempt."""

    if action not in {"retry_policy", "refresh_rules"}:
        raise ReviewConflict("unsupported approval action")
    if assignment.session_id != session_row.id:
        raise ReviewConflict("operator approval assignment is outside the review session")
    if action == "refresh_rules" and not rules_revision_id:
        raise ReviewConflict("rules refresh approval requires a revision")
    now = _as_utc(now)
    resolved_ttl = (
        int(ttl_seconds)
        if ttl_seconds is not None
        else int(
            getattr(settings, "review_operator_approval_ttl_seconds", 300)
            if settings is not None
            else 300
        )
    )
    approval_id = _new_id("ra")
    max_approval_bytes = int(
        getattr(settings, "review_max_approval_bytes", 4_096) if settings is not None else 4_096
    )
    # Bound the approval projection before durable allocation so callers cannot
    # create an oversize one-use token under a tightened configuration.
    approval_projection = canonical_json_v1(
        {
            "action": action,
            "actor": actor,
            "approval_id": approval_id,
            "assignment_id": assignment.assignment_id,
            "expires_at_ms": _to_ms(now + timedelta(seconds=resolved_ttl)),
            "rules_revision_id": rules_revision_id,
            "session_id": session_row.session_id,
        }
    )
    if len(approval_projection) > max_approval_bytes:
        raise ReviewConflict("operator approval exceeds review_max_approval_bytes")
    approval = ReviewOperatorApproval(
        approval_id=approval_id,
        session_id=session_row.id,
        assignment_id=assignment.id,
        action=action,
        rules_revision_id=rules_revision_id,
        actor=actor,
        expires_at=now + timedelta(seconds=resolved_ttl),
    )
    session.add(approval)
    await session.flush()
    return approval


async def review_audit_page(
    session: AsyncSession,
    *,
    session_row: ReviewSession,
    cursor: str | None,
    limit: int,
    internal: bool = False,
    cursor_secret: str | None = None,
    page_max: int = 16,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read an immutable attempt history page with a stable snapshot cursor.

    Lazy-expires the current still-active assignment on read so miner-facing
    history/status does not stick forever past ``expires_at`` when no report
    lands (e.g. review CVM cannot reach the validator API). Callers that mutate
    via this session should commit after the page is built.
    """

    if not 1 <= limit <= page_max:
        raise ReviewConflict(f"review history page limit must be 1..{page_max}")
    moment = _as_utc(now)
    current = await _current_assignment(session, session_row, lock=False)
    if current is not None:
        await expire_assignment_if_needed(session, current, now=moment)
    snapshot_id, after_id = _parse_cursor(
        cursor,
        session_id=session_row.session_id,
        secret=cursor_secret or "review-audit-local-default",
    )
    if snapshot_id is None:
        latest = await session.scalar(
            select(ReviewAssignment.id)
            .where(ReviewAssignment.session_id == session_row.id)
            .order_by(ReviewAssignment.id.desc())
            .limit(1)
        )
        snapshot_id = int(latest or 0)
    statement = (
        select(ReviewAssignment)
        .where(ReviewAssignment.session_id == session_row.id)
        .where(ReviewAssignment.id <= snapshot_id)
        .where(ReviewAssignment.id > after_id)
        .order_by(ReviewAssignment.id)
        .limit(limit + 1)
    )
    rows = list((await session.scalars(statement)).all())
    page = rows[:limit]
    next_cursor = (
        _encode_cursor(
            session_id=session_row.session_id,
            snapshot_id=snapshot_id,
            after_id=page[-1].id,
            secret=cursor_secret or "review-audit-local-default",
        )
        if len(rows) > limit and page
        else None
    )
    total_count = await session.scalar(
        select(func.count(ReviewAssignment.id))
        .where(ReviewAssignment.session_id == session_row.id)
        .where(ReviewAssignment.id <= snapshot_id)
    )
    return {
        "session_id": session_row.session_id,
        "current_assignment_id": session_row.current_assignment_id,
        "authorizing_assignment_id": session_row.authorizing_assignment_id,
        "items": [_audit_item(item, internal=internal) for item in page],
        "next_cursor": next_cursor,
        "total_count": int(total_count or 0),
    }


async def expire_assignment_if_needed(
    session: AsyncSession,
    assignment: ReviewAssignment,
    *,
    now: datetime,
) -> bool:
    """Atomically terminalize a still-active unreceipted assignment on expiry."""

    if (
        assignment.review_report_envelope_json is not None
        or assignment.phase not in _ACTIVE_PHASES
        or now < _as_utc(assignment.expires_at)
    ):
        return False
    assignment.phase = "review_expired"
    assignment.active_key = None
    assignment.capability_state = "revoked"
    assignment.finished_at = now
    assignment.reason_code = "expired"
    await _set_nonce_state(session, assignment, "expired", now=now)
    review_session = await session.get(ReviewSession, assignment.session_id)
    if (
        review_session is not None
        and review_session.current_assignment_id == assignment.assignment_id
    ):
        review_session.current_assignment_id = assignment.assignment_id
        await record_review_submission_status(
            session,
            review_session=review_session,
            assignment=assignment,
            raw_status="review_expired",
            reason="review_expired",
        )
    return True


async def _issue_assignment(
    session: AsyncSession,
    *,
    review_session: ReviewSession,
    submission: AgentSubmission,
    snapshot: ReviewRulesSnapshot,
    settings: ChallengeSettings,
    attempt: int,
    now: datetime,
    input_config: ReviewInputConfig | None,
) -> CreatedReviewSession:
    assignment_id = _new_id("ra")
    review_nonce = _new_id("rn")
    token = _derive_session_token(settings, assignment_id)
    ttl_seconds = int(
        getattr(settings, "review_assignment_ttl_seconds", REVIEW_ASSIGNMENT_TTL_SECONDS)
    )
    expires_at = now + timedelta(seconds=ttl_seconds)
    await prune_outstanding_review_records(session, now=now, settings=settings)
    await enforce_outstanding_review_cap(session, settings=settings, now=now)
    artifact = {
        "agent_hash": submission.agent_hash,
        "zip_sha256": review_session.artifact_sha256,
        "zip_size_bytes": review_session.artifact_size_bytes,
        "manifest_sha256": review_session.manifest_sha256,
        "manifest_entries_sha256": review_session.manifest_entries_sha256,
        "fetch_path": f"/review/v1/assignments/{assignment_id}/artifact",
    }
    bound_recv = review_session.submission_received_at_ms
    if bound_recv is None:
        # Fall back to durable submission.submitted_at then challenge-now.
        submitted = getattr(submission, "submitted_at", None)
        if isinstance(submitted, datetime):
            st = submitted if submitted.tzinfo is not None else submitted.replace(tzinfo=UTC)
            bound_recv = int(st.timestamp() * 1000)
        else:
            bound_recv = _to_ms(now)
        review_session.submission_received_at_ms = bound_recv
    _, assignment_bytes, assignment_digest = build_review_assignment(
        session_id=review_session.session_id,
        assignment_id=assignment_id,
        attempt=attempt,
        submission_id=str(submission.id),
        artifact=artifact,
        rules_snapshot_sha256_value=snapshot.snapshot_sha256,
        rules_revision_id=snapshot.revision_id,
        review_nonce=review_nonce,
        issued_at_ms=_to_ms(now),
        expires_at_ms=_to_ms(expires_at),
        session_token_sha256=sha256(token.encode("utf-8")).hexdigest(),
        config=input_config or ReviewInputConfig(),
        submission_received_at_ms=int(bound_recv),
    )
    max_assignment_bytes = int(getattr(settings, "review_max_assignment_bytes", 262_144))
    if len(assignment_bytes) > max_assignment_bytes:
        raise ReviewConflict("review assignment exceeds review_max_assignment_bytes")
    assignment = ReviewAssignment(
        session_id=review_session.id,
        assignment_id=assignment_id,
        attempt=attempt,
        assignment_bytes=assignment_bytes.decode("utf-8"),
        assignment_digest=assignment_digest,
        artifact_sha256=review_session.artifact_sha256,
        rules_snapshot_sha256=snapshot.snapshot_sha256,
        rules_revision_id=snapshot.revision_id,
        review_nonce=review_nonce,
        session_token_sha256=sha256(token.encode("utf-8")).hexdigest(),
        capability_state="active",
        phase="review_queued",
        active_key=review_session.session_id,
        issued_at=now,
        expires_at=expires_at,
    )
    session.add(assignment)
    try:
        await session.flush()
    except IntegrityError as exc:
        if _is_active_assignment_constraint(exc):
            raise ReviewConflict("another review assignment is active") from exc
        raise
    session.add(
        ReviewNonce(
            assignment_id=assignment.id,
            session_id=review_session.id,
            nonce=review_nonce,
            purpose=REVIEW_NONCE_PURPOSE,
            state="active",
            expires_at=expires_at,
        )
    )
    review_session.current_assignment_id = assignment_id
    await session.flush()
    return CreatedReviewSession(
        session=review_session,
        assignment=assignment,
        session_token=token,
    )


async def _current_assignment(
    session: AsyncSession,
    review_session: ReviewSession,
    *,
    lock: bool = False,
) -> ReviewAssignment | None:
    statement = (
        select(ReviewAssignment)
        .where(ReviewAssignment.session_id == review_session.id)
        .where(ReviewAssignment.assignment_id == review_session.current_assignment_id)
    )
    if lock:
        statement = statement.with_for_update()
    return await session.scalar(statement)


async def _snapshot_for_digest(
    session: AsyncSession,
    session_id: int,
    snapshot_sha256: str,
) -> ReviewRulesSnapshot:
    snapshot = await session.scalar(
        select(ReviewRulesSnapshot).where(ReviewRulesSnapshot.snapshot_sha256 == snapshot_sha256)
    )
    if snapshot is None:
        raise ReviewNotFound("rules snapshot not found")
    return snapshot


async def _store_snapshot(
    session: AsyncSession,
    review_session: ReviewSession,
    bundle: dict[str, Any],
) -> ReviewRulesSnapshot:
    digest = rules_snapshot_sha256(bundle)
    existing = await session.scalar(
        select(ReviewRulesSnapshot).where(ReviewRulesSnapshot.snapshot_sha256 == digest)
    )
    if existing is not None:
        return existing
    snapshot = ReviewRulesSnapshot(
        session_id=review_session.id,
        revision_id=str(bundle["revision_id"]),
        snapshot_sha256=digest,
        canonical_bytes=canonical_json_v1(bundle),
    )
    session.add(snapshot)
    await session.flush()
    return snapshot


async def _consume_approval(
    session: AsyncSession,
    *,
    approval_id: str,
    session_row: ReviewSession,
    assignment: ReviewAssignment,
    action: str,
    now: datetime,
) -> ReviewOperatorApproval:
    approval = await session.scalar(
        select(ReviewOperatorApproval)
        .where(ReviewOperatorApproval.approval_id == approval_id)
        .with_for_update()
    )
    if approval is None:
        raise ReviewConflict("operator approval not found")
    if (
        approval.session_id != session_row.id
        or approval.assignment_id != assignment.id
        or approval.action != action
        or approval.used_at is not None
        or now >= _as_utc(approval.expires_at)
    ):
        raise ReviewConflict("operator approval is invalid, expired, or consumed")
    approval.used_at = now
    return approval


async def _set_nonce_state(
    session: AsyncSession,
    assignment: ReviewAssignment,
    state: str,
    *,
    now: datetime,
) -> None:
    nonce = await session.scalar(
        select(ReviewNonce).where(ReviewNonce.assignment_id == assignment.id).with_for_update()
    )
    if nonce is not None and nonce.state == "active":
        nonce.state = state
        nonce.consumed_at = now


async def _submission_for_session(
    session: AsyncSession,
    review_session: ReviewSession,
) -> AgentSubmission:
    submission = await session.get(AgentSubmission, review_session.submission_id)
    if submission is None:
        raise ReviewNotFound("submission not found")
    return submission


async def prune_outstanding_review_records(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    settings: ChallengeSettings | None = None,
) -> int:
    """Expire and drop outstanding review nonces past expiry; return prune count.

    Fully outstanding states (active/consumed receipts past fashion) never delete
    retained evidence. Only unconsumed active nonces with expires_at <= now are
    pruned, so a later valid caller can recover capacity.
    """

    del settings  # settings reserved for typed outstanding caps
    moment = _as_utc(now)
    rows = list(
        (
            await session.scalars(
                select(ReviewNonce)
                .where(ReviewNonce.state == "active")
                .where(ReviewNonce.expires_at <= moment)
                .with_for_update()
            )
        ).all()
    )
    for nonce in rows:
        nonce.state = "expired"
        nonce.consumed_at = moment
    if rows:
        await session.flush()
    return len(rows)


async def enforce_outstanding_review_cap(
    session: AsyncSession,
    *,
    settings: ChallengeSettings,
    now: datetime | None = None,
) -> None:
    """Fail closed once global outstanding review nonces hit the config cap."""

    moment = _as_utc(now)
    await prune_outstanding_review_records(session, now=moment, settings=settings)
    maximum = int(getattr(settings, "attestation_max_outstanding_nonce_receipts", 10_000))
    outstanding = await session.scalar(
        select(func.count()).select_from(ReviewNonce).where(ReviewNonce.state == "active")
    )
    if outstanding is not None and outstanding >= maximum:
        raise ReviewRateLimited("review outstanding nonce capacity is full")


async def enforce_review_session_mutation_budget(
    session: AsyncSession,
    *,
    session_row: ReviewSession,
    settings: ChallengeSettings,
    now: datetime | None = None,
) -> None:
    """Enforce the per-session mutation rate from config before a state change."""

    del session  # rate windows are tracked in process memory keyed by session_id
    moment = _as_utc(now)
    maximum = int(getattr(settings, "review_max_mutations_per_session_per_minute", 10))
    key = session_row.session_id
    window = [
        stamp for stamp in _MUTATION_WINDOWS.get(key, []) if moment - stamp < timedelta(minutes=1)
    ]
    if len(window) >= maximum:
        raise ReviewRateLimited("review session mutation rate limit exceeded")
    window.append(moment)
    _MUTATION_WINDOWS[key] = window


async def record_review_submission_status(
    session: AsyncSession,
    *,
    review_session: ReviewSession,
    assignment: ReviewAssignment,
    raw_status: str,
    reason: str,
) -> None:
    """Mirror assignment lifecycle only through safe submission status events."""

    if assignment.session_id != review_session.id:
        raise ReviewConflict("review assignment is outside the review session")
    submission = await _submission_for_session(session, review_session)
    if submission.raw_status == raw_status:
        return
    if submission.raw_status not in {
        "rate_limit_reserved",
        "review_queued",
        "review_cvm_running",
        "review_provider_standby",
        "review_verifying",
        "review_allowed",
        "review_rejected",
        "review_escalated",
        "review_expired",
        "review_cancelled",
        "review_error",
    }:
        return
    await ensure_submission_status(
        session,
        submission,
        raw_status,
        actor="review-cvm",
        reason=reason,
        metadata={"review": _safe_review_event_projection(review_session, assignment, raw_status)},
    )


def _safe_review_event_projection(
    review_session: ReviewSession,
    assignment: ReviewAssignment,
    phase: str,
) -> dict[str, Any]:
    """Return the redacted durable review projection safe for status events."""

    try:
        outcome = json.loads(assignment.review_verification_outcome_json or "{}")
    except (TypeError, json.JSONDecodeError):
        outcome = {}
    outcome_status = outcome.get("status") if isinstance(outcome, dict) else None
    verdict = {
        "verified_allow": "allow",
        "verified_reject": "reject",
        "verified_escalate": "escalate",
    }.get(outcome_status)
    return {
        "session_id": review_session.session_id,
        "assignment_id": assignment.assignment_id,
        "attempt": assignment.attempt,
        "phase": phase,
        "terminal": phase not in _ACTIVE_PHASES,
        "verdict": verdict,
        "verified": outcome_status in {"verified_allow", "verified_reject", "verified_escalate"},
        "retryable": (
            bool(outcome.get("retryable"))
            if isinstance(outcome, dict) and outcome_status is not None
            else phase in _RETRYABLE_PHASES
        ),
        "reason_code": assignment.reason_code,
        "report_available": assignment.review_public_projection_json is not None,
        "issued_at": assignment.issued_at.isoformat(),
        "finished_at": assignment.finished_at.isoformat() if assignment.finished_at else None,
    }


def _derive_session_token(settings: ChallengeSettings, assignment_id: str) -> str:
    """Derive the assignment bearer from the same loader as internal auth.

    Supports both inline ``shared_token`` and the documented
    ``shared_token_file`` deployment path; never invents a second secret source.

    Format: ``{assignment_id}.{hmac_hex}``.  Embedding the assignment id lets the
    measured review CVM bootstrap from ``REVIEW_SESSION_TOKEN`` alone without a
    third secret name or compose-hash-changing plain env field.
    """

    secret = load_internal_token(settings)
    if not secret:
        raise ReviewConflict("review capability secret is not configured")
    if not isinstance(assignment_id, str) or not assignment_id:
        raise ReviewConflict("review capability assignment id is missing")
    mac = hmac.new(
        secret.encode("utf-8"),
        b"agent-challenge:review-session:v1:" + assignment_id.encode("ascii"),
        sha256,
    ).hexdigest()
    token = f"{assignment_id}.{mac}"
    max_capability_bytes = int(getattr(settings, "review_max_capability_bytes", 4_096))
    if len(token.encode("utf-8")) > max_capability_bytes:
        raise ReviewConflict("review capability exceeds review_max_capability_bytes")
    return token


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(24)}"


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _to_ms(value: datetime) -> int:
    return int(_as_utc(value).timestamp() * 1000)


def _parse_cursor(
    cursor: str | None,
    *,
    session_id: str,
    secret: str,
) -> tuple[int | None, int]:
    if cursor is None:
        return None, 0
    try:
        encoded, signature = cursor.split(".", 1)
        expected = hmac.new(
            secret.encode("utf-8"),
            _AUDIT_CURSOR_DOMAIN + encoded.encode("ascii"),
            sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("audit cursor signature is invalid")
        padded = encoded + ("=" * (-len(encoded) % 4))
        payload = parse_json_object(base64.urlsafe_b64decode(padded))
        if payload["session_id"] != session_id:
            raise ValueError("audit cursor session is invalid")
        snapshot_id = payload["snapshot_id"]
        after_id = payload["after_id"]
        if (
            isinstance(snapshot_id, bool)
            or isinstance(after_id, bool)
            or not isinstance(snapshot_id, int)
            or not isinstance(after_id, int)
        ):
            raise ValueError("audit cursor shape is invalid")
    except (AttributeError, ValueError, UnicodeEncodeError) as exc:
        raise ReviewConflict("invalid review audit cursor") from exc
    if snapshot_id < 0 or after_id < 0 or after_id > snapshot_id:
        raise ReviewConflict("invalid review audit cursor")
    return snapshot_id, after_id


def _encode_cursor(
    *,
    session_id: str,
    snapshot_id: int,
    after_id: int,
    secret: str,
) -> str:
    encoded = (
        base64.urlsafe_b64encode(
            canonical_json_v1(
                {
                    "after_id": after_id,
                    "session_id": session_id,
                    "snapshot_id": snapshot_id,
                }
            )
        )
        .decode("ascii")
        .rstrip("=")
    )
    signature = hmac.new(
        secret.encode("utf-8"),
        _AUDIT_CURSOR_DOMAIN + encoded.encode("ascii"),
        sha256,
    ).hexdigest()
    return f"{encoded}.{signature}"


def _audit_retryable(assignment: ReviewAssignment) -> bool:
    """Match audit retryable flags to durable verification outcomes when present."""

    if assignment.review_verification_outcome_json is not None:
        try:
            outcome = parse_json_object(assignment.review_verification_outcome_json)
        except ValueError:
            outcome = None
        if isinstance(outcome, dict) and "retryable" in outcome:
            return bool(outcome["retryable"])
    return assignment.phase in _RETRYABLE_PHASES


def _audit_item(assignment: ReviewAssignment, *, internal: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "assignment_id": assignment.assignment_id,
        "attempt": assignment.attempt,
        "phase": assignment.phase,
        "terminal": assignment.phase not in _ACTIVE_PHASES,
        "retryable": _audit_retryable(assignment),
        "reason_code": assignment.reason_code,
        "issued_at_ms": _to_ms(assignment.issued_at),
        "finished_at_ms": _to_ms(assignment.finished_at) if assignment.finished_at else None,
        "report_projection": (
            parse_json_object(assignment.review_public_projection_json)
            if assignment.review_public_projection_json is not None
            else None
        ),
    }
    if internal:
        item.update(
            {
                "assignment": parse_json_object(assignment.assignment_bytes),
                "report_envelope": (
                    parse_json_object(assignment.review_report_envelope_json)
                    if assignment.review_report_envelope_json is not None
                    else None
                ),
                "evidence_descriptor": (
                    parse_json_object(assignment.review_evidence_descriptor_json)
                    if assignment.review_evidence_descriptor_json is not None
                    else None
                ),
                "verification_outcome": (
                    parse_json_object(assignment.review_verification_outcome_json)
                    if assignment.review_verification_outcome_json is not None
                    else None
                ),
            }
        )
    return item


def _is_active_assignment_constraint(exc: IntegrityError) -> bool:
    message = str(exc.orig).lower()
    return (
        "uq_review_assignments_one_active_session" in message
        or "review_assignments.active_key" in message
    )
