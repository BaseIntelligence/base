"""Behavioral guards for full-attested review authorization.

These are offline fixtures.  They validate durable database and HTTP behavior,
not a real TDX quote or CVM deployment.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from agent_challenge.api import routes as api_routes
from agent_challenge.app import app
from agent_challenge.auth.security import SignedRequestAuth
from agent_challenge.core.models import (
    AgentSubmission,
    EvaluationJob,
    ReviewAssignment,
    ReviewSession,
    SubmissionStatusEvent,
)
from agent_challenge.evaluation.benchmarks import BenchmarkTask
from agent_challenge.evaluation.runner import (
    EvaluationAuthorizationError,
    create_evaluation_job,
)
from agent_challenge.evaluation.work_units import list_pending_work_units
from agent_challenge.review.authorization import verified_review_assignment_for_submission
from agent_challenge.review.sessions import (
    create_review_session,
    record_review_submission_status,
)
from agent_challenge.sdk.config import ChallengeSettings


def _submission(*, suffix: str, raw_status: str = "review_queued") -> AgentSubmission:
    artifact = f"review-artifact-{suffix}".encode()
    return AgentSubmission(
        miner_hotkey=f"review-miner-{suffix}",
        name=f"review-agent-{suffix}",
        agent_hash=hashlib.sha256(artifact).hexdigest(),
        artifact_uri=f"/tmp/{suffix}.zip",
        artifact_path=f"/tmp/{suffix}.zip",
        zip_sha256=hashlib.sha256(artifact).hexdigest(),
        zip_size_bytes=len(artifact),
        raw_status=raw_status,
        status="queued",
        effective_status="queued",
    )


async def _create_review(
    database_session,
    *,
    suffix: str,
    raw_status: str = "review_queued",
) -> tuple[int, int, int]:
    async with database_session() as session:
        submission = _submission(suffix=suffix, raw_status=raw_status)
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=f"review-artifact-{suffix}".encode(),
            rules_files={".rules/policy.md": b"offline policy"},
            rules_revision_id="rules-v1",
            settings=ChallengeSettings(
                shared_token=api_routes.settings.shared_token or "test-token"
            ),
            now=datetime.now(UTC),
        )
        await session.commit()
        return submission.id, created.session.id, created.assignment.id


def _verified_allow(assignment: ReviewAssignment) -> None:
    assignment.phase = "review_allowed"
    assignment.review_report_envelope_json = '{"schema_version":1}'
    assignment.review_digest = "a" * 64
    assignment.reason_code = "policy_allowed"
    assignment.review_verification_outcome_json = json.dumps(
        {
            "status": "verified_allow",
            "terminal": True,
            "retryable": False,
            "reason_code": "policy_allowed",
            "nonce_consumed": True,
            "measurement_allowlisted": True,
            "report_data_matched": True,
            "verified_at_ms": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.parametrize(
    ("phase", "outcome"),
    [
        ("review_queued", None),
        ("review_cvm_running", None),
        ("review_provider_standby", None),
        ("review_verifying", "verifier_unavailable"),
        ("review_rejected", "verified_reject"),
        ("review_escalated", "verified_escalate"),
        ("review_expired", None),
        ("review_cancelled", None),
        ("review_error", "trust_failed"),
    ],
)
async def test_non_allow_review_states_create_no_work_or_benchmark_metadata(
    client,
    database_session,
    monkeypatch,
    phase: str,
    outcome: str | None,
) -> None:
    monkeypatch.setattr("agent_challenge.core.config.settings.attested_review_enabled", True)
    monkeypatch.setattr("agent_challenge.core.config.settings.phala_attestation_enabled", True)
    selection_calls: list[object] = []
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.load_benchmark_tasks",
        lambda: selection_calls.append(object()) or [],
    )
    submission_id, session_id, assignment_id = await _create_review(
        database_session,
        suffix=phase,
        raw_status=phase,
    )

    async with database_session() as session:
        review_session = await session.get(ReviewSession, session_id)
        assignment = await session.get(ReviewAssignment, assignment_id)
        submission = await session.get(AgentSubmission, submission_id)
        assert review_session is not None
        assert assignment is not None
        assert submission is not None
        assignment.phase = phase
        if outcome is not None:
            assignment.review_verification_outcome_json = json.dumps(
                {
                    "status": outcome,
                    "terminal": outcome != "verifier_unavailable",
                    "retryable": outcome == "verifier_unavailable",
                    "reason_code": "offline",
                    "nonce_consumed": outcome != "verifier_unavailable",
                    "measurement_allowlisted": False,
                    "report_data_matched": False,
                    "verified_at_ms": None,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        with pytest.raises(EvaluationAuthorizationError):
            await create_evaluation_job(session, submission)
        assert await verified_review_assignment_for_submission(session, submission) is None
        assert await session.scalar(select(func.count(EvaluationJob.id))) == 0
        assert await list_pending_work_units(session) == []
        await session.commit()

    assert selection_calls == []
    tasks = await client.get("/benchmarks/tasks")
    assert tasks.status_code == 200
    assert tasks.content == b"[]"
    snapshot = await client.get(f"/submissions/{submission_id}/status")
    assert snapshot.status_code == 200
    assert snapshot.json()["evaluation"] == {
        "job_id": None,
        "status": None,
        "score": 0.0,
        "passed_tasks": 0,
        "total_tasks": 0,
        "verdict": None,
        "reason_codes": [],
        "current_attempt": None,
        "attempt_status": None,
        "task_phases": [],
        "task_rows": [],
    }


async def test_only_exact_persisted_verified_allow_can_prepare_deterministic_work(
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent_challenge.core.config.settings.attested_review_enabled", True)
    monkeypatch.setattr("agent_challenge.core.config.settings.phala_attestation_enabled", True)
    task = BenchmarkTask(
        task_id="terminal-bench/offline-review-gate",
        docker_image="example.test/task@sha256:" + ("a" * 64),
        benchmark="terminal_bench",
    )
    monkeypatch.setattr("agent_challenge.evaluation.runner.load_benchmark_tasks", lambda: [task])
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 1)
    allowed_submission_id, allowed_session_id, allowed_assignment_id = await _create_review(
        database_session,
        suffix="allowed",
        raw_status="queued",
    )
    blocked_submission_id, _blocked_session_id, _blocked_assignment_id = await _create_review(
        database_session,
        suffix="same-miner-new-version",
        raw_status="review_queued",
    )

    async with database_session() as session:
        allowed = await session.get(AgentSubmission, allowed_submission_id)
        blocked = await session.get(AgentSubmission, blocked_submission_id)
        allowed_session = await session.get(ReviewSession, allowed_session_id)
        allowed_assignment = await session.get(ReviewAssignment, allowed_assignment_id)
        assert allowed is not None
        assert blocked is not None
        assert allowed_session is not None
        assert allowed_assignment is not None
        _verified_allow(allowed_assignment)
        allowed_session.authorizing_assignment_id = allowed_assignment.assignment_id
        assert (
            await verified_review_assignment_for_submission(session, allowed)
        ).assignment_id == allowed_assignment.assignment_id
        assert await verified_review_assignment_for_submission(session, blocked) is None
        with pytest.raises(EvaluationAuthorizationError):
            await create_evaluation_job(session, blocked)
        with pytest.raises(EvaluationAuthorizationError):
            await create_evaluation_job(session, allowed)
        await session.commit()

    async with database_session() as session:
        assert await session.scalar(select(func.count(EvaluationJob.id))) == 0
        # Full-attested work is never exposed to the legacy coordination plane.
        assert await list_pending_work_units(session) == []


async def test_full_attested_benchmark_surfaces_are_metadata_free(client, monkeypatch) -> None:
    monkeypatch.setattr("agent_challenge.api.routes.settings.attested_review_enabled", True)
    monkeypatch.setattr("agent_challenge.api.routes.settings.phala_attestation_enabled", True)
    before_info = await client.get("/benchmarks")
    before_tasks = await client.get("/benchmarks/tasks")

    monkeypatch.setattr("agent_challenge.api.routes.settings.terminal_bench_dataset", "sentinel")
    monkeypatch.setattr("agent_challenge.api.routes.settings.evaluation_concurrency", 99)
    after_info = await client.get("/benchmarks", headers={"Authorization": "Bearer ambient"})
    after_tasks = await client.get("/benchmarks/tasks", headers={"X-Hotkey": "ambient"})

    assert before_info.status_code == after_info.status_code == 200
    assert (
        before_info.content
        == after_info.content
        == (b'{"backend":"","dataset":"","task_count":0,"evaluation_concurrency":0}')
    )
    assert before_tasks.status_code == after_tasks.status_code == 200
    assert before_tasks.content == after_tasks.content == b"[]"


async def test_owner_revalidation_cannot_bypass_pending_review(
    client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent_challenge.core.config.settings.attested_review_enabled", True)
    monkeypatch.setattr("agent_challenge.core.config.settings.phala_attestation_enabled", True)
    submission_id, _session_id, _assignment_id = await _create_review(
        database_session,
        suffix="admin-bypass",
    )

    async def owner_auth() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey="owner-hotkey",
            signature="offline",
            nonce="offline-owner-nonce",
            timestamp="2026-07-11T00:00:00+00:00",
            body_sha256="0" * 64,
            canonical_request="POST\n/owner\n0\noffline\nhash",
        )

    app.dependency_overrides[api_routes.owner_signed_auth] = owner_auth
    try:
        response = await client.post(
            f"/owner/submissions/{submission_id}/revalidate",
            json={"reason": "admin must not synthesize review allow"},
        )
    finally:
        app.dependency_overrides.pop(api_routes.owner_signed_auth, None)

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "review_authorization_required"
    async with database_session() as session:
        assert await session.scalar(select(func.count(EvaluationJob.id))) == 0


async def test_review_status_and_sse_publish_only_safe_durable_phase_data(
    client,
    database_session,
    monkeypatch,
) -> None:
    # Status includes the safe review projection only in full attested mode.
    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)
    submission_id, session_id, assignment_id = await _create_review(
        database_session,
        suffix="safe-status",
    )
    async with database_session() as session:
        review_session = await session.get(ReviewSession, session_id)
        assignment = await session.get(ReviewAssignment, assignment_id)
        assert review_session is not None
        assert assignment is not None
        for phase in (
            "review_cvm_running",
            "review_provider_standby",
            "review_verifying",
        ):
            assignment.phase = phase
            await record_review_submission_status(
                session,
                review_session=review_session,
                assignment=assignment,
                raw_status=phase,
                reason=f"offline_{phase}",
            )
        _verified_allow(assignment)
        review_session.authorizing_assignment_id = assignment.assignment_id
        await record_review_submission_status(
            session,
            review_session=review_session,
            assignment=assignment,
            raw_status="review_allowed",
            reason="offline_verified_allow",
        )
        await session.commit()
        events = (
            (
                await session.execute(
                    select(SubmissionStatusEvent)
                    .where(SubmissionStatusEvent.submission_id == submission_id)
                    .order_by(SubmissionStatusEvent.sequence)
                )
            )
            .scalars()
            .all()
        )

    status = await client.get(f"/submissions/{submission_id}/status")
    assert status.status_code == 200
    snapshot = status.json()
    assert snapshot["phase"] == "review_allowed"
    assert snapshot["review"] == {
        "session_id": review_session.session_id,
        "assignment_id": assignment.assignment_id,
        "attempt": 1,
        "phase": "review_allowed",
        "terminal": True,
        "verdict": "allow",
        "verified": True,
        "retryable": False,
        "reason_code": "policy_allowed",
        "report_available": False,
        "issued_at": snapshot["review"]["issued_at"],
        "finished_at": None,
    }
    assert len(events) == 4
    rendered = api_routes._format_sse_event(events[-1])
    event_data = json.loads(rendered.split("data: ", 1)[1].strip())
    assert event_data["phase"] == "review_allowed"
    assert event_data["review"] == snapshot["review"]
    for forbidden in ("selected_tasks", "score", "weight", "review_session_token"):
        assert forbidden not in rendered


async def test_review_mutation_error_precedence_is_auth_then_size_then_media_then_schema(
    client,
    database_session,
    monkeypatch,
) -> None:
    _submission_id, _session_id, assignment_id = await _create_review(
        database_session,
        suffix="route-precedence",
    )
    async with database_session() as session:
        assignment = await session.get(ReviewAssignment, assignment_id)
        assert assignment is not None
        capability_assignment_id = assignment.assignment_id

    async def authenticated_assignment(session, assignment_id, token, **kwargs):
        assignment = await session.scalar(
            select(ReviewAssignment).where(ReviewAssignment.assignment_id == assignment_id)
        )
        assert assignment is not None
        return assignment

    monkeypatch.setattr(api_routes, "authenticate_assignment_capability", authenticated_assignment)
    token = "offline-capability"

    route = f"/review/v1/assignments/{capability_assignment_id}/model-call-started"
    unauthorized = await client.post(route, content=b"x" * (16 * 1024 + 1))
    assert unauthorized.status_code == 401
    oversized = await client.post(
        route,
        content=b"x" * (16 * 1024 + 1),
        headers={"authorization": f"Bearer {token}", "Content-Type": "text/plain"},
    )
    assert oversized.status_code == 413
    wrong_media = await client.post(
        route,
        content=b"{}",
        headers={"authorization": f"Bearer {token}", "Content-Type": "text/plain"},
    )
    assert wrong_media.status_code == 400
    invalid_schema = await client.post(
        route,
        content=b"{}",
        headers={"authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    assert invalid_schema.status_code == 422
    async with database_session() as session:
        assignment = await session.get(ReviewAssignment, assignment_id)
        assert assignment is not None
        assert assignment.phase == "review_queued"
        assert assignment.model_call_started_json is None


async def test_fully_legacy_intake_never_creates_a_review_session(
    client,
    database_session,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agent_challenge.api.routes.settings.attested_review_enabled", False)
    monkeypatch.setattr("agent_challenge.api.routes.settings.phala_attestation_enabled", False)
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        "/tmp/review-gate-legacy",
    )

    async def signed_auth() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey="legacy-miner",
            signature="offline",
            nonce="legacy-intake",
            timestamp="2026-07-11T00:00:00+00:00",
            body_sha256="0" * 64,
            canonical_request="POST\n/submissions\n0\nlegacy\nhash",
        )

    def no_review(*args, **kwargs):
        raise AssertionError("legacy intake must not create a review session")

    monkeypatch.setattr(api_routes, "create_review_session", no_review)
    app.dependency_overrides[api_routes.signed_submission_auth] = signed_auth
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("agent.py", "class Agent:\n    pass\n")
    try:
        response = await client.post(
            "/submissions",
            json={
                "name": "legacy-agent",
                "artifact_zip_base64": base64.b64encode(archive.getvalue()).decode("ascii"),
            },
        )
    finally:
        app.dependency_overrides.pop(api_routes.signed_submission_auth, None)

    assert response.status_code == 201
    submission_id = response.json()["submission_id"]
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        assert submission.raw_status == "analysis_queued"
        assert await session.scalar(select(func.count(ReviewSession.id))) == 0
