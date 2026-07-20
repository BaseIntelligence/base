from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, func, select

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.models import (
    AgentSubmission,
    AnalyzerReport,
    EvaluationJob,
    OwnerActionAudit,
    RequestNonce,
)
from agent_challenge.security import SignedRequestAuth

NOW = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
async def clean_owner_control_tables(database_session) -> AsyncIterator[None]:
    async with database_session() as session:
        await session.execute(delete(OwnerActionAudit))
        await session.execute(delete(RequestNonce))
        await session.commit()
    yield
    async with database_session() as session:
        await session.execute(delete(OwnerActionAudit))
        await session.execute(delete(RequestNonce))
        await session.commit()


@pytest.fixture
def owner_auth_override() -> AsyncIterator[None]:
    calls = 0

    async def authenticate() -> SignedRequestAuth:
        nonlocal calls
        calls += 1
        body_hash = hashlib.sha256(f"owner-body-{calls}".encode()).hexdigest()
        return SignedRequestAuth(
            hotkey="owner-hotkey",
            signature=f"owner-signature-{calls}",
            nonce=f"owner-nonce-{calls}",
            timestamp=NOW.isoformat(),
            body_sha256=body_hash,
            canonical_request=f"owner-request-{calls}",
        )

    app.dependency_overrides[routes.owner_signed_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.owner_signed_auth, None)


def owner_headers(
    *,
    nonce: str,
    hotkey: str | None = None,
    signature: str = "valid-signature",
) -> dict[str, str]:
    return {
        "X-Hotkey": hotkey or routes.settings.owner_hotkey,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": datetime.now(UTC).isoformat(),
    }


async def create_completed_submission(database_session, tmp_path):
    artifact_path = tmp_path / "agent.zip"
    artifact_path.write_bytes(b"immutable artifact")
    zip_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-hotkey",
            name="agent-a",
            agent_hash="agent-hash-a",
            artifact_uri=str(artifact_path),
            status="completed",
            raw_status="completed",
            effective_status="completed",
            zip_sha256=zip_sha256,
            zip_size_bytes=artifact_path.stat().st_size,
            artifact_path=str(artifact_path),
        )
        session.add(submission)
        await session.flush()
        job = EvaluationJob(
            job_id="original-job",
            submission_id=submission.id,
            status="completed",
            selected_tasks_json="[]",
            score=1.0,
            passed_tasks=0,
            total_tasks=0,
            verdict="valid",
            reason_codes_json='["rules_passed"]',
            logs_ref="logs/original-job.txt",
        )
        session.add(job)
        await session.flush()
        submission.latest_evaluation_job_id = job.id
        report = AnalyzerReport(
            job_id=job.id,
            rules_version="rules-v1",
            verdict="valid",
            reason_codes_json='["rules_passed"]',
            report_json='{"overall_verdict":"valid"}',
            logs_ref="logs/original-job.txt",
        )
        session.add(report)
        await session.commit()
        return submission.id, job.id, job.job_id, submission.agent_hash, submission.artifact_uri


async def create_internal_terminal_submission(database_session, tmp_path, *, raw_status):
    artifact_path = tmp_path / "agent.zip"
    artifact_path.write_bytes(b"immutable artifact")
    zip_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    public_label = "valid" if raw_status == "tb_completed" else "error"
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-hotkey",
            name="agent-internal-terminal",
            agent_hash=f"agent-hash-{raw_status}",
            artifact_uri=str(artifact_path),
            status=public_label,
            raw_status=raw_status,
            effective_status=public_label,
            zip_sha256=zip_sha256,
            zip_size_bytes=artifact_path.stat().st_size,
            artifact_path=str(artifact_path),
        )
        session.add(submission)
        await session.flush()
        job = EvaluationJob(
            job_id=f"original-job-{raw_status}",
            submission_id=submission.id,
            status="completed",
            selected_tasks_json="[]",
            score=1.0,
            passed_tasks=0,
            total_tasks=0,
            verdict="valid",
            reason_codes_json='["rules_passed"]',
            logs_ref=f"logs/original-job-{raw_status}.txt",
        )
        session.add(job)
        await session.flush()
        submission.latest_evaluation_job_id = job.id
        await session.commit()
        return submission.id, job.id, job.job_id, public_label


async def test_owner_revalidate_creates_new_job_for_same_immutable_artifact(
    client,
    database_session,
    monkeypatch,
    owner_auth_override,
    tmp_path,
):
    monkeypatch.setattr("agent_challenge.evaluation.runner.load_benchmark_tasks", lambda: [])
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    (
        submission_id,
        original_job_pk,
        original_job_id,
        agent_hash,
        artifact_uri,
    ) = await create_completed_submission(database_session, tmp_path)

    response = await client.post(
        f"/owner/submissions/{submission_id}/revalidate",
        json={"reason": "rerun with updated rules"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["submission_id"] == submission_id
    assert payload["status"] == "queued"
    assert payload["effective_status"] == "queued"
    assert payload["job_id"] != original_job_id

    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        jobs = (
            (await session.execute(select(EvaluationJob).order_by(EvaluationJob.id)))
            .scalars()
            .all()
        )
        audit = await session.scalar(select(OwnerActionAudit))

    assert submission is not None
    assert submission.agent_hash == agent_hash
    assert submission.artifact_uri == artifact_uri
    assert submission.latest_evaluation_job_id != original_job_pk
    assert len(jobs) == 2
    assert jobs[1].id == submission.latest_evaluation_job_id
    assert jobs[1].triggered_by_hotkey == "owner-hotkey"
    assert jobs[1].trigger_reason == "revalidate"
    assert audit is not None
    assert audit.action == "revalidate"
    assert audit.reason == "rerun with updated rules"
    assert audit.before_effective_status == "completed"
    assert audit.after_effective_status == "queued"


@pytest.mark.parametrize("raw_status", ["tb_completed", "tb_failed_final"])
async def test_owner_revalidate_requeues_internal_terminal_submission(
    client,
    database_session,
    monkeypatch,
    owner_auth_override,
    tmp_path,
    raw_status,
):
    monkeypatch.setattr("agent_challenge.evaluation.runner.load_benchmark_tasks", lambda: [])
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    (
        submission_id,
        original_job_pk,
        original_job_id,
        before_effective_status,
    ) = await create_internal_terminal_submission(database_session, tmp_path, raw_status=raw_status)

    response = await client.post(
        f"/owner/submissions/{submission_id}/revalidate",
        json={"reason": "rerun internal terminal submission"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["submission_id"] == submission_id
    assert payload["status"] == "queued"
    assert payload["effective_status"] == "evaluation queued"
    assert payload["job_id"] != original_job_id

    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        jobs = (
            (await session.execute(select(EvaluationJob).order_by(EvaluationJob.id)))
            .scalars()
            .all()
        )
        audit = await session.scalar(select(OwnerActionAudit))

    assert submission is not None
    assert submission.raw_status == "tb_queued"
    assert submission.effective_status == "evaluation queued"
    assert submission.latest_evaluation_job_id != original_job_pk
    assert len(jobs) == 2
    original = next(job for job in jobs if job.id == original_job_pk)
    new_job = next(job for job in jobs if job.id != original_job_pk)
    assert original.job_id == original_job_id
    assert new_job.id == submission.latest_evaluation_job_id
    assert new_job.status == "queued"
    assert new_job.triggered_by_hotkey == "owner-hotkey"
    assert new_job.trigger_reason == "revalidate"
    assert audit is not None
    assert audit.action == "revalidate"
    assert audit.before_effective_status == before_effective_status
    assert audit.after_effective_status == "evaluation queued"


async def test_owner_override_and_suspicious_only_change_effective_status_and_audit(
    client,
    database_session,
    owner_auth_override,
    tmp_path,
):
    (
        submission_id,
        original_job_pk,
        _job_id,
        _agent_hash,
        _artifact_uri,
    ) = await create_completed_submission(database_session, tmp_path)

    override = await client.post(
        f"/owner/submissions/{submission_id}/override",
        json={"status": "overridden_invalid", "reason": "manual invalidation"},
    )
    suspicious = await client.post(
        f"/owner/submissions/{submission_id}/suspicious",
        json={"suspicious": True, "reason": "needs owner review"},
    )

    assert override.status_code == 200
    assert override.json()["effective_status"] == "overridden_invalid"
    assert suspicious.status_code == 200
    assert suspicious.json()["effective_status"] == "suspicious"

    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        job = await session.get(EvaluationJob, original_job_pk)
        reports = (await session.execute(select(AnalyzerReport))).scalars().all()
        audits = (
            (
                await session.execute(
                    select(OwnerActionAudit).order_by(
                        OwnerActionAudit.created_at,
                        OwnerActionAudit.id,
                    )
                )
            )
            .scalars()
            .all()
        )

    assert submission is not None
    assert job is not None
    assert submission.status == "completed"
    assert submission.raw_status == "completed"
    assert submission.effective_status == "suspicious"
    assert submission.latest_evaluation_job_id == original_job_pk
    assert job.status == "completed"
    assert job.verdict == "valid"
    assert job.reason_codes_json == '["rules_passed"]'
    assert reports[0].verdict == "valid"
    assert reports[0].report_json == '{"overall_verdict":"valid"}'
    assert [audit.action for audit in audits] == ["override", "suspicious"]
    assert audits[0].before_effective_status == "completed"
    assert audits[0].after_effective_status == "overridden_invalid"
    assert audits[1].before_effective_status == "overridden_invalid"
    assert audits[1].after_effective_status == "suspicious"


async def test_owner_suspicious_can_clear_to_raw_status(
    client,
    database_session,
    owner_auth_override,
    tmp_path,
):
    submission_id, _job_pk, _job_id, _agent_hash, _artifact_uri = await create_completed_submission(
        database_session, tmp_path
    )
    mark = await client.post(
        f"/owner/submissions/{submission_id}/suspicious",
        json={"suspicious": True, "reason": "temporary hold"},
    )
    clear = await client.post(
        f"/owner/submissions/{submission_id}/suspicious",
        json={"suspicious": False, "reason": "review cleared"},
    )

    assert mark.status_code == 200
    assert mark.json()["effective_status"] == "suspicious"
    assert clear.status_code == 200
    assert clear.json()["effective_status"] == "completed"

    async with database_session() as session:
        audits = (
            (await session.execute(select(OwnerActionAudit).order_by(OwnerActionAudit.id)))
            .scalars()
            .all()
        )
    assert [audit.after_effective_status for audit in audits] == ["suspicious", "completed"]


async def test_owner_endpoint_rejects_non_owner_hotkey(
    client,
    database_session,
    monkeypatch,
    tmp_path,
):
    submission_id, _job_pk, _job_id, _agent_hash, _artifact_uri = await create_completed_submission(
        database_session, tmp_path
    )
    monkeypatch.setattr(
        "agent_challenge.auth.security.verify_substrate_signature",
        lambda _hotkey, _message, signature: signature == "valid-signature",
    )

    response = await client.post(
        f"/owner/submissions/{submission_id}/override",
        json={"status": "overridden_valid", "reason": "owner only"},
        headers=owner_headers(nonce="non-owner", hotkey="miner-hotkey"),
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "forbidden"}
    async with database_session() as session:
        audit_count = await session.scalar(select(func.count(OwnerActionAudit.id)))
    assert audit_count == 0


async def test_owner_endpoint_rejects_replayed_nonce(
    client,
    database_session,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("agent_challenge.evaluation.runner.load_benchmark_tasks", lambda: [])
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    monkeypatch.setattr(
        "agent_challenge.auth.security.verify_substrate_signature",
        lambda _hotkey, _message, signature: signature == "valid-signature",
    )
    submission_id, _job_pk, _job_id, _agent_hash, _artifact_uri = await create_completed_submission(
        database_session, tmp_path
    )
    headers = owner_headers(nonce="replayed-owner-nonce")

    first = await client.post(
        f"/owner/submissions/{submission_id}/revalidate",
        json={"reason": "first"},
        headers=headers,
    )
    second = await client.post(
        f"/owner/submissions/{submission_id}/revalidate",
        json={"reason": "first"},
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json() == {"detail": "replayed request"}
    async with database_session() as session:
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        audit_count = await session.scalar(select(func.count(OwnerActionAudit.id)))
    assert job_count == 2
    assert audit_count == 1


async def test_owner_override_missing_submission_returns_404(client, owner_auth_override):
    response = await client.post(
        "/owner/submissions/999999/override",
        json={"status": "overridden_valid", "reason": "missing"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "submission not found"}


async def test_owner_audit_returns_append_only_history(
    client,
    database_session,
    owner_auth_override,
    tmp_path,
):
    submission_id, _job_pk, _job_id, _agent_hash, _artifact_uri = await create_completed_submission(
        database_session, tmp_path
    )

    first = await client.post(
        f"/owner/submissions/{submission_id}/override",
        json={"status": "overridden_valid", "reason": "manual pass"},
    )
    second = await client.post(
        f"/owner/submissions/{submission_id}/override",
        json={"status": "overridden_invalid", "reason": "new evidence"},
    )
    audit = await client.get("/owner/audit")

    assert first.status_code == 200
    assert second.status_code == 200
    assert audit.status_code == 200
    rows = audit.json()
    assert len(rows) == 2
    assert [row["action"] for row in rows] == ["override", "override"]
    assert [row["reason"] for row in rows] == ["manual pass", "new evidence"]
    assert rows[0]["id"] < rows[1]["id"]
    assert rows[0]["after_effective_status"] == "overridden_valid"
    assert rows[1]["before_effective_status"] == "overridden_valid"
    assert rows[0]["request_hash"] != rows[1]["request_hash"]
    assert rows[0]["nonce"] == "owner-nonce-1"
    assert rows[1]["nonce"] == "owner-nonce-2"

    async with database_session() as session:
        persisted = (
            (await session.execute(select(OwnerActionAudit).order_by(OwnerActionAudit.id)))
            .scalars()
            .all()
        )
    assert [row.reason for row in persisted] == ["manual pass", "new evidence"]
