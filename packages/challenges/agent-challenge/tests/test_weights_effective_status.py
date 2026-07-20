from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select

from agent_challenge.core.config import settings
from agent_challenge.models import (
    AgentSubmission,
    AnalyzerReport,
    EvaluationJob,
    OwnerActionAudit,
)
from agent_challenge.sdk.config import effective_evaluation_task_count
from agent_challenge.weights import get_weights

NOW = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
FULL_TASK_COUNT = effective_evaluation_task_count(settings.evaluation_task_count)


@dataclass(frozen=True)
class SubmissionCase:
    hotkey: str
    agent_hash: str
    score: float
    effective_status: str
    raw_status: str = "tb_completed"
    job_status: str = "completed"
    verdict: str | None = "valid"
    passed_tasks: int = 1
    total_tasks: int = 1


@pytest.fixture(autouse=True)
async def clean_effective_status_tables(database_session):
    async with database_session() as session:
        await session.execute(delete(OwnerActionAudit))
        await session.execute(delete(AnalyzerReport))
        await session.execute(delete(EvaluationJob))
        await session.execute(delete(AgentSubmission))
        await session.commit()
    yield
    async with database_session() as session:
        await session.execute(delete(OwnerActionAudit))
        await session.execute(delete(AnalyzerReport))
        await session.execute(delete(EvaluationJob))
        await session.execute(delete(AgentSubmission))
        await session.commit()


async def test_weights_require_tb_completed_and_effective_valid_status(database_session):
    async with database_session() as session:
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="valid-hotkey",
                agent_hash="hash-valid",
                score=0.8,
                effective_status="valid",
                total_tasks=FULL_TASK_COUNT,
            ),
        )
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="suspicious-hotkey",
                agent_hash="hash-suspicious",
                score=0.99,
                effective_status="suspicious",
                verdict="suspicious",
            ),
        )
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="invalid-hotkey",
                agent_hash="hash-invalid",
                score=0.95,
                effective_status="invalid",
                raw_status="invalid",
                verdict="invalid",
            ),
        )
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="error-hotkey",
                agent_hash="hash-error",
                score=0.9,
                effective_status="error",
                raw_status="error",
                verdict="error",
            ),
        )
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="overridden-valid-stale-hotkey",
                agent_hash="hash-overridden-valid-stale",
                score=0.7,
                effective_status="overridden_valid",
                raw_status="invalid",
                verdict="invalid",
            ),
        )
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="overridden-invalid-hotkey",
                agent_hash="hash-overridden-invalid",
                score=0.85,
                effective_status="overridden_invalid",
                verdict="valid",
            ),
        )
        await session.commit()

    assert await get_weights() == {"valid-hotkey": 0.8}

    async with database_session() as session:
        overridden_job = await session.scalar(
            select(EvaluationJob)
            .join(EvaluationJob.submission)
            .where(AgentSubmission.agent_hash == "hash-overridden-valid-stale")
        )
    assert overridden_job is not None
    assert overridden_job.verdict == "invalid"


@pytest.mark.parametrize(
    ("raw_status", "effective_status", "verdict"),
    [
        ("analysis_rejected", "invalid", "invalid"),
        ("analysis_rejected", "valid", "valid"),
        ("analysis_escalated", "suspicious", "suspicious"),
        ("admin_paused", "admin_paused", "suspicious"),
        ("admin_paused", "valid", "valid"),
        ("tb_failed_retryable", "valid", "valid"),
        ("tb_failed_final", "error", "error"),
        ("tb_failed_final", "valid", "valid"),
        ("error", "valid", "valid"),
        ("invalid", "valid", "valid"),
        ("suspicious", "valid", "valid"),
        ("overridden_invalid", "overridden_invalid", "valid"),
    ],
)
async def test_weights_exclude_stale_completed_jobs_without_durable_tb_completion(
    database_session,
    raw_status,
    effective_status,
    verdict,
):
    async with database_session() as session:
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey=f"stale-{raw_status}-{effective_status}",
                agent_hash=f"hash-stale-{raw_status}-{effective_status}",
                score=0.99,
                raw_status=raw_status,
                effective_status=effective_status,
                verdict=verdict,
            ),
        )
        await session.commit()

    assert await get_weights() == {}


async def test_weights_include_tb_completed_after_admin_allow_terminal_bench(database_session):
    async with database_session() as session:
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="admin-allowed-hotkey",
                agent_hash="hash-admin-allowed-tb-completed",
                score=0.91,
                raw_status="tb_completed",
                effective_status="valid",
                verdict="valid",
                total_tasks=FULL_TASK_COUNT,
            ),
        )
        await session.commit()

    assert await get_weights() == {"admin-allowed-hotkey": 0.91}


async def test_overridden_valid_scores_only_after_tb_completed_policy(database_session):
    async with database_session() as session:
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="overridden-valid-completed-hotkey",
                agent_hash="hash-overridden-valid-completed",
                score=0.7,
                raw_status="tb_completed",
                effective_status="overridden_valid",
                verdict="invalid",
                total_tasks=FULL_TASK_COUNT,
            ),
        )
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="overridden-invalid-completed-hotkey",
                agent_hash="hash-overridden-invalid-completed",
                score=1.0,
                raw_status="tb_completed",
                effective_status="overridden_invalid",
                verdict="valid",
            ),
        )
        await session.commit()

    assert await get_weights() == {"overridden-valid-completed-hotkey": 0.7}


async def test_weights_exclude_non_completed_jobs_even_after_tb_completed(database_session):
    async with database_session() as session:
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="running-job-hotkey",
                agent_hash="hash-running-job",
                score=1.0,
                raw_status="tb_completed",
                effective_status="valid",
                job_status="running",
            ),
        )
        await session.commit()

    assert await get_weights() == {}


async def test_leaderboard_uses_same_tb_completed_scoring_gate(client, database_session):
    async with database_session() as session:
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="miner-a",
                agent_hash="hash-miner-a-valid",
                score=0.55,
                effective_status="valid",
            ),
        )
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="miner-a",
                agent_hash="hash-miner-a-stale-analysis-rejected",
                score=0.99,
                raw_status="analysis_rejected",
                effective_status="valid",
                verdict="valid",
            ),
        )
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="miner-b",
                agent_hash="hash-miner-b-overridden-valid",
                score=0.75,
                effective_status="overridden_valid",
                verdict="invalid",
            ),
        )
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="miner-c",
                agent_hash="hash-miner-c-overridden-invalid",
                score=1.0,
                effective_status="overridden_invalid",
            ),
        )
        await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="miner-d",
                agent_hash="hash-miner-d-admin-paused-stale",
                score=0.95,
                raw_status="admin_paused",
                effective_status="valid",
            ),
        )
        await session.commit()

    response = await client.get("/leaderboard")

    assert response.status_code == 200
    rows = response.json()
    assert rows == [
        {
            "miner_hotkey": "miner-b",
            "submission_id": rows[0]["submission_id"],
            "name": "agent-hash-miner-b-overridden-valid",
            "agent_hash": "hash-miner-b-overridden-valid",
            "display_name": "agent-hash-miner-b-overridden-valid",
            "family_id": None,
            "version_number": None,
            "version_label": None,
            "version_count": None,
            "is_latest_version": False,
            "latest_submission_id": None,
            "score": 0.75,
            "passed_tasks": 1,
            "total_tasks": 1,
        },
        {
            "miner_hotkey": "miner-a",
            "submission_id": rows[1]["submission_id"],
            "name": "agent-hash-miner-a-valid",
            "agent_hash": "hash-miner-a-valid",
            "display_name": "agent-hash-miner-a-valid",
            "family_id": None,
            "version_number": None,
            "version_label": None,
            "version_count": None,
            "is_latest_version": False,
            "latest_submission_id": None,
            "score": 0.55,
            "passed_tasks": 1,
            "total_tasks": 1,
        },
    ]


async def test_public_submission_status_exposes_bounded_effective_summary(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id = await _create_submission_case(
            session,
            SubmissionCase(
                hotkey="miner-status",
                agent_hash="hash-status",
                score=0.42,
                raw_status="analysis_escalated",
                effective_status="suspicious",
                verdict="suspicious",
            ),
            include_private_report=True,
        )
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "suspicious"
    assert payload["effective_status"] == "suspicious"
    assert payload["zip_sha256"] == "zip-hash-status"
    assert payload["submitted_at"] is not None
    assert payload["created_at"] is not None
    assert payload["latest_evaluation"] == {
        "job_id": "job-hash-status",
        "status": "completed",
        "score": 0.42,
        "passed_tasks": 1,
        "total_tasks": 1,
        "verdict": "suspicious",
        "rules_version": "rules-v1",
        "created_at": "2026-05-22T12:00:00",
        "started_at": "2026-05-22T12:00:01",
        "finished_at": "2026-05-22T12:00:02",
    }
    _assert_public_payload_omits_private_fields(payload)
    _assert_public_payload_omits_private_fields(payload["latest_evaluation"])

    evaluation = await client.get("/agents/hash-status/evaluation")
    assert evaluation.status_code == 200
    evaluation_payload = evaluation.json()
    assert evaluation_payload["effective_status"] == "suspicious"
    assert evaluation_payload["zip_sha256"] == "zip-hash-status"
    assert evaluation_payload["rules_version"] == "rules-v1"
    assert evaluation_payload["verdict"] == "suspicious"
    _assert_public_payload_omits_private_fields(evaluation_payload)


async def _create_submission_case(
    session,
    case: SubmissionCase,
    *,
    include_private_report: bool = False,
) -> int:
    submission = AgentSubmission(
        miner_hotkey=case.hotkey,
        name=f"agent-{case.agent_hash}",
        agent_hash=case.agent_hash,
        artifact_uri=f"/tmp/{case.agent_hash}.zip",
        status=case.raw_status,
        raw_status=case.raw_status,
        effective_status=case.effective_status,
        zip_sha256=f"zip-{case.agent_hash}",
        zip_size_bytes=123,
        artifact_path=f"/tmp/{case.agent_hash}.zip",
        submitted_at=NOW,
        created_at=NOW,
        signature="private-signature",
        signature_nonce="private-nonce",
        signature_timestamp=NOW.isoformat(),
        signature_payload_sha256="private-payload-hash",
        signature_message="private canonical request",
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=f"job-{case.agent_hash}",
        submission_id=submission.id,
        status=case.job_status,
        selected_tasks_json="[]",
        score=case.score,
        passed_tasks=case.passed_tasks,
        total_tasks=case.total_tasks,
        verdict=case.verdict,
        rules_version="rules-v1",
        reason_codes_json='["private_reason"]',
        logs_ref="private/logs.txt",
        created_at=NOW,
        started_at=NOW + timedelta(seconds=1),
        finished_at=NOW + timedelta(seconds=2),
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    if include_private_report:
        session.add(
            AnalyzerReport(
                job_id=job.id,
                rules_version="rules-v1",
                verdict=case.verdict or "suspicious",
                reason_codes_json='["private_report_reason"]',
                report_json='{"private":"analyzer-report"}',
                logs_ref="private/analyzer-logs.txt",
            )
        )
    await session.flush()
    return submission.id


def _assert_public_payload_omits_private_fields(payload: dict[str, object]) -> None:
    private_fields = {
        "logs_ref",
        "report_json",
        "analyzer_reports",
        "reason_codes_json",
        "signature",
        "signature_nonce",
        "signature_timestamp",
        "signature_payload_sha256",
        "signature_message",
        "raw_status",
    }
    assert private_fields.isdisjoint(payload)
