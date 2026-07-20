from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.models import (
    AdminReviewDecision,
    AgentSubmission,
    AnalysisRun,
    EvaluationAttempt,
    EvaluationJob,
    LlmVerdict,
    SimilarityMatch,
    SubmissionStatusEvent,
)
from agent_challenge.security import SignedRequestAuth
from agent_challenge.submissions.state_machine import transition_submission_status

NOW = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)


@pytest.fixture
def owner_auth_override() -> AsyncIterator[None]:
    calls = 0

    async def authenticate() -> SignedRequestAuth:
        nonlocal calls
        calls += 1
        return SignedRequestAuth(
            hotkey="owner-hotkey",
            signature=f"owner-signature-{calls}",
            nonce=f"owner-nonce-{calls}",
            timestamp=NOW.isoformat(),
            body_sha256=hashlib.sha256(f"admin-body-{calls}".encode()).hexdigest(),
            canonical_request=f"owner-request-{calls}",
        )

    app.dependency_overrides[routes.owner_signed_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.owner_signed_auth, None)


def owner_headers(*, nonce: str, hotkey: str | None = None) -> dict[str, str]:
    return {
        "X-Hotkey": hotkey or routes.settings.owner_hotkey,
        "X-Signature": "valid-signature",
        "X-Nonce": nonce,
        "X-Timestamp": datetime.now(UTC).isoformat(),
    }


async def create_escalated_submission(database_session, *, raw_status: str = "admin_paused"):
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-admin",
            name="admin-agent",
            agent_hash="admin-agent-hash",
            artifact_uri="/tmp/admin-agent.zip",
            status="received",
            raw_status="received",
            effective_status="received",
            zip_sha256="a" * 64,
            zip_size_bytes=128,
            artifact_path="/tmp/admin-agent.zip",
        )
        session.add(submission)
        await session.flush()
        for to_status, actor, reason in (
            ("received", "api", "received"),
            ("upload_verified", "api", "verified"),
            ("rate_limit_reserved", "api", "reserved"),
            ("analysis_queued", "api", "queued"),
            ("ast_running", "worker", "ast"),
            ("llm_running", "worker", "llm"),
            ("analysis_escalated", "worker", "escalated"),
        ):
            kwargs = {"from_status": None} if to_status == "received" else {}
            await transition_submission_status(
                session,
                submission,
                to_status,
                actor=actor,
                reason=reason,
                **kwargs,
            )
        analysis = AnalysisRun(
            submission_id=submission.id,
            analyzer_name="blocking_analyzer",
            analyzer_version="ast-similarity-llm-v1",
            status="completed",
            verdict="escalate",
            reason_codes_json=json.dumps(["manual_review"]),
            report_json=json.dumps({"evidence": "preserved"}),
        )
        session.add(analysis)
        await session.flush()
        session.add(
            LlmVerdict(
                analysis_run_id=analysis.id,
                reviewer_name="kimi",
                model_name="model",
                verdict="escalate",
                confidence=0.51,
                reason_codes_json=json.dumps(["manual_review"]),
                raw_request_json=json.dumps({"redacted": True}),
                raw_response_json=json.dumps({"verdict": "escalate"}),
            )
        )
        session.add(
            SimilarityMatch(
                analysis_run_id=analysis.id,
                source_submission_id=submission.id,
                matched_submission_id=123,
                matched_artifact_uri="/tmp/private-match.zip",
                match_kind="python_ast_similarity",
                score=91.0,
                evidence_json=json.dumps({"risk_band": "high"}),
            )
        )
        before_status = submission.effective_status
        if raw_status == "admin_paused":
            await transition_submission_status(
                session,
                submission,
                "admin_paused",
                actor="worker",
                reason="admin review required",
                metadata={"analysis_run_id": analysis.id},
            )
        session.add(
            AdminReviewDecision(
                submission_id=submission.id,
                reviewer_hotkey="system",
                decision="pending_analysis_review",
                reason="mock escalate",
                before_effective_status=before_status,
                after_effective_status=submission.effective_status,
                metadata_json=json.dumps({"analysis_run_id": analysis.id}),
            )
        )
        await session.commit()
        return submission.id, analysis.id


async def test_admin_allow_appends_decision_and_waits_for_miner_env(
    client,
    database_session,
    monkeypatch,
    owner_auth_override,
):
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.evaluation.runner.load_benchmark_tasks", lambda: [])
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    submission_id, analysis_id = await create_escalated_submission(database_session)
    evidence_snapshot = await admin_evidence_snapshot(database_session, analysis_id)

    response = await client.post(
        f"/owner/submissions/{submission_id}/admin-escalation",
        json={"decision": "admin_allow", "reason": "review cleared"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == "admin_allow"
    assert payload["status"] == "waiting_miner_env"
    assert payload["effective_status"] == "Waiting environments"
    assert payload["job_id"] is None
    async with database_session() as session:
        decisions = (
            (await session.execute(select(AdminReviewDecision).order_by(AdminReviewDecision.id)))
            .scalars()
            .all()
        )
        submission = await session.get(AgentSubmission, submission_id)
        job = await session.scalar(select(EvaluationJob))
        evidence_after = await admin_evidence_snapshot_for_session(session, analysis_id)
        events = (
            (
                await session.execute(
                    select(SubmissionStatusEvent).order_by(SubmissionStatusEvent.id)
                )
            )
            .scalars()
            .all()
        )

    assert submission is not None
    assert submission.raw_status == "waiting_miner_env"
    assert job is None
    assert [decision.decision for decision in decisions] == [
        "pending_analysis_review",
        "admin_allow",
    ]
    assert decisions[1].before_effective_status == "admin_paused"
    assert decisions[1].after_effective_status == "Waiting environments"
    metadata = json.loads(decisions[1].metadata_json)
    assert metadata["analysis_run_id"] == analysis_id
    assert metadata["previous_verdict"] == "escalate"
    assert metadata["nonce"] == "owner-nonce-1"
    assert metadata["signature"] == "owner-signature-1"
    assert [event.to_status for event in events][-2:] == ["analysis_allowed", "waiting_miner_env"]
    assert [event.reason for event in events][-2:] == ["admin_review_allowed", "waiting_miner_env"]
    assert evidence_after == evidence_snapshot


async def test_legacy_confirmed_empty_admin_allow_queues_terminal_bench(
    client,
    database_session,
    monkeypatch,
    owner_auth_override,
):
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.evaluation.runner.load_benchmark_tasks", lambda: [])
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    submission_id, _analysis_id = await create_escalated_submission(database_session)
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        submission.env_confirmed_empty = True
        submission.env_confirmed_empty_at = NOW
        submission.env_locked_at = NOW
        submission.env_compatibility_reason = "pre_env_gate_analysis_allowed"
        await session.commit()

    response = await client.post(
        f"/owner/submissions/{submission_id}/admin-escalation",
        json={"decision": "admin_allow", "reason": "review cleared"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "tb_queued"
    assert payload["effective_status"] == "evaluation queued"
    assert payload["job_id"] is not None
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        job = await session.scalar(select(EvaluationJob))

    assert submission is not None
    assert submission.raw_status == "tb_queued"
    assert job is not None
    assert job.status == "queued"
    assert job.triggered_by_hotkey == "owner-hotkey"
    assert job.trigger_reason == "admin_allow"


async def test_admin_reject_appends_decision_without_terminal_bench_work(
    client,
    database_session,
    owner_auth_override,
):
    submission_id, analysis_id = await create_escalated_submission(database_session)
    evidence_snapshot = await admin_evidence_snapshot(database_session, analysis_id)

    response = await client.post(
        f"/owner/submissions/{submission_id}/admin-escalation",
        json={"decision": "admin_reject", "reason": "policy violation confirmed"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "submission_id": submission_id,
        "effective_status": "invalid",
        "decision_id": response.json()["decision_id"],
        "decision": "admin_reject",
        "status": "analysis_rejected",
        "job_id": None,
    }
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        attempt_count = await session.scalar(select(func.count(EvaluationAttempt.id)))
        decision = await session.scalar(
            select(AdminReviewDecision).where(AdminReviewDecision.decision == "admin_reject")
        )
        evidence_after = await admin_evidence_snapshot_for_session(session, analysis_id)

    assert submission is not None
    assert submission.raw_status == "analysis_rejected"
    assert submission.effective_status == "invalid"
    assert job_count == 0
    assert attempt_count == 0
    assert decision is not None
    assert json.loads(decision.metadata_json)["analysis_run_id"] == analysis_id
    assert evidence_after == evidence_snapshot


async def test_admin_request_rerun_requeues_analysis_and_preserves_prior_evidence(
    client,
    database_session,
    owner_auth_override,
):
    submission_id, analysis_id = await create_escalated_submission(database_session)
    evidence_snapshot = await admin_evidence_snapshot(database_session, analysis_id)

    response = await client.post(
        f"/owner/submissions/{submission_id}/admin-escalation",
        json={"decision": "admin_request_rerun", "reason": "fresh analyzer pass required"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "analysis_queued"
    assert response.json()["effective_status"] == "queued"
    assert response.json()["job_id"] is None
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        analysis_runs = (await session.execute(select(AnalysisRun))).scalars().all()
        llm_rows = (await session.execute(select(LlmVerdict))).scalars().all()
        matches = (await session.execute(select(SimilarityMatch))).scalars().all()
        decision = await session.scalar(
            select(AdminReviewDecision).where(AdminReviewDecision.decision == "admin_request_rerun")
        )
        last_event = await session.scalar(
            select(SubmissionStatusEvent).order_by(SubmissionStatusEvent.sequence.desc()).limit(1)
        )
        evidence_after = await admin_evidence_snapshot_for_session(session, analysis_id)

    assert submission is not None
    assert submission.raw_status == "analysis_queued"
    assert len(analysis_runs) == 1
    assert analysis_runs[0].id == analysis_id
    assert analysis_runs[0].verdict == "escalate"
    assert len(llm_rows) == 1
    assert llm_rows[0].verdict == "escalate"
    assert len(matches) == 1
    assert matches[0].score == 91.0
    assert decision is not None
    assert decision.after_effective_status == "queued"
    assert last_event is not None
    assert last_event.to_status == "analysis_queued"
    event_metadata = json.loads(last_event.metadata_json)
    assert event_metadata["admin_decision_id"] == decision.id
    assert event_metadata["analysis_run_id"] == analysis_id
    assert evidence_after == evidence_snapshot


async def test_admin_request_rerun_accepts_analysis_escalated_before_pause(
    client,
    database_session,
    owner_auth_override,
):
    submission_id, analysis_id = await create_escalated_submission(
        database_session,
        raw_status="analysis_escalated",
    )
    evidence_snapshot = await admin_evidence_snapshot(database_session, analysis_id)

    response = await client.post(
        f"/owner/submissions/{submission_id}/admin-escalation",
        json={"decision": "admin_request_rerun", "reason": "rerun before pause"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "analysis_queued"
    async with database_session() as session:
        events = (
            (
                await session.execute(
                    select(SubmissionStatusEvent).order_by(SubmissionStatusEvent.sequence)
                )
            )
            .scalars()
            .all()
        )
        evidence_after = await admin_evidence_snapshot_for_session(session, analysis_id)

    assert [event.to_status for event in events][-2:] == ["admin_paused", "analysis_queued"]
    assert evidence_after == evidence_snapshot


async def test_admin_resolution_rejects_invalid_state(
    client,
    database_session,
    owner_auth_override,
):
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-admin",
            name="not-paused",
            agent_hash="not-paused-hash",
            artifact_uri="/tmp/not-paused.zip",
            status="received",
            raw_status="analysis_queued",
            effective_status="queued",
        )
        session.add(submission)
        await session.commit()
        submission_id = submission.id

    response = await client.post(
        f"/owner/submissions/{submission_id}/admin-escalation",
        json={"decision": "admin_allow", "reason": "not paused"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "submission is not awaiting admin review"}
    async with database_session() as session:
        decision_count = await session.scalar(select(func.count(AdminReviewDecision.id)))
    assert decision_count == 0


async def test_admin_resolution_requires_signed_owner_auth(
    client,
    database_session,
    monkeypatch,
):
    submission_id, _analysis_id = await create_escalated_submission(database_session)
    monkeypatch.setattr(
        "agent_challenge.auth.security.verify_substrate_signature",
        lambda _hotkey, _message, signature: signature == "valid-signature",
    )

    missing_auth = await client.post(
        f"/owner/submissions/{submission_id}/admin-escalation",
        json={"decision": "admin_allow", "reason": "missing auth"},
    )
    wrong_owner = await client.post(
        f"/owner/submissions/{submission_id}/admin-escalation",
        json={"decision": "admin_allow", "reason": "wrong owner"},
        headers=owner_headers(nonce="wrong-owner", hotkey="miner-hotkey"),
    )

    assert missing_auth.status_code == 401
    assert wrong_owner.status_code == 403
    assert wrong_owner.json() == {"detail": "forbidden"}
    async with database_session() as session:
        decisions = (
            (await session.execute(select(AdminReviewDecision).order_by(AdminReviewDecision.id)))
            .scalars()
            .all()
        )
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
    assert [decision.decision for decision in decisions] == ["pending_analysis_review"]
    assert job_count == 0


async def admin_evidence_snapshot(database_session, analysis_id: int) -> dict[str, object]:
    async with database_session() as session:
        return await admin_evidence_snapshot_for_session(session, analysis_id)


async def admin_evidence_snapshot_for_session(session, analysis_id: int) -> dict[str, object]:
    analysis = await session.get(AnalysisRun, analysis_id)
    llm = await session.scalar(select(LlmVerdict).where(LlmVerdict.analysis_run_id == analysis_id))
    match = await session.scalar(
        select(SimilarityMatch).where(SimilarityMatch.analysis_run_id == analysis_id)
    )
    assert analysis is not None
    assert llm is not None
    assert match is not None
    return {
        "analysis_status": analysis.status,
        "analysis_verdict": analysis.verdict,
        "analysis_reason_codes_json": analysis.reason_codes_json,
        "analysis_report_json": analysis.report_json,
        "llm_verdict": llm.verdict,
        "llm_confidence": llm.confidence,
        "llm_reason_codes_json": llm.reason_codes_json,
        "llm_raw_request_json": llm.raw_request_json,
        "llm_raw_response_json": llm.raw_response_json,
        "match_kind": match.match_kind,
        "match_score": match.score,
        "matched_submission_id": match.matched_submission_id,
        "matched_artifact_uri": match.matched_artifact_uri,
        "match_evidence_json": match.evidence_json,
    }
