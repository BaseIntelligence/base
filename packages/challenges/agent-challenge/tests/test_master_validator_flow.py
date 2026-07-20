from __future__ import annotations

import base64
import hashlib
import io
import zipfile

import pytest
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.evaluation import (
    claim_next_evaluation_job,
    enqueue_evaluation_job_for_submission,
    run_evaluation_job,
    run_next_evaluation_job,
)
from agent_challenge.models import AgentSubmission, AnalyzerReport, EvaluationJob, TaskResult
from agent_challenge.sdk.executors import DockerRunResult
from agent_challenge.security import SignedRequestAuth
from agent_challenge.swe_forge import SweForgeTask

ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"


def agent_source(contents: str | bytes) -> str | bytes:
    if isinstance(contents, bytes):
        return contents
    if "class Agent" in contents:
        return contents
    return f"{ENTRYPOINT_SOURCE}\n{contents}"


@pytest.fixture
def signed_submission_override():
    async def authenticate() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey="signed-miner-hotkey",
            signature="test-signature",
            nonce="test-nonce",
            timestamp="2026-05-22T12:00:00+00:00",
            body_sha256="test-body-sha256",
            canonical_request="signed-test-request",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.signed_submission_auth, None)


class CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0
        self.tasks: list[str] = []

    def run(self, spec, timeout_seconds: int):
        self.calls += 1
        self.tasks.append(spec.labels["base.task"])
        return DockerRunResult(
            container_name="fake",
            stdout=f"ran {spec.labels['base.task']}",
            stderr="",
            returncode=0,
        )


def make_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("agent.py", agent_source("print('ok')\n"))
    return buffer.getvalue()


def patch_single_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.settings.benchmark_backend",
        "swe_forge",
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.load_swe_forge_tasks",
        lambda: [SweForgeTask(task_id="task-a", docker_image="baseintelligence/swe-forge:task-a")],
    )
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 1)
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_concurrency", 1)


async def test_legacy_normal_role_queues_analysis_like_master(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    """The legacy ``normal`` role is inert: intake queues central analysis."""
    executor = CountingExecutor()
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    monkeypatch.setattr("agent_challenge.api.routes.settings.validator_role", "normal")
    monkeypatch.setattr("agent_challenge.evaluation.runner.build_docker_executor", lambda: executor)
    archive_bytes = make_zip()
    zip_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    response = await client.post(
        "/submissions",
        json={
            "name": "normal-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 201
    assert response.json()["status"] == "queued"
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        task_result_count = await session.scalar(select(func.count(TaskResult.id)))

    assert submission is not None
    assert submission.agent_hash == zip_sha256
    assert submission.status == "queued"
    assert submission.raw_status == "analysis_queued"
    assert submission.effective_status == "queued"
    assert submission.latest_evaluation_job_id is None
    assert job_count == 0
    assert task_result_count == 0
    assert executor.calls == 0


async def test_master_validator_signed_submission_queues_analysis_without_running(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    executor = CountingExecutor()
    patch_single_task(monkeypatch)
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    monkeypatch.setattr("agent_challenge.api.routes.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.evaluation.runner.build_docker_executor", lambda: executor)
    archive_bytes = make_zip()

    response = await client.post(
        "/submissions",
        json={
            "name": "master-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 201
    assert response.json()["status"] == "queued"
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        jobs = (await session.execute(select(EvaluationJob))).scalars().all()
        task_result_count = await session.scalar(select(func.count(TaskResult.id)))

    assert submission is not None
    assert jobs == []
    assert submission.latest_evaluation_job_id is None
    assert submission.status == "queued"
    assert submission.raw_status == "analysis_queued"
    assert submission.effective_status == "queued"
    assert task_result_count == 0
    assert executor.calls == 0


async def test_master_queue_enqueue_claim_and_run_are_idempotent(
    database_session,
    monkeypatch,
    tmp_path,
):
    patch_single_task(monkeypatch)
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    executor = CountingExecutor()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="hotkey-a",
            name="agent-a",
            agent_hash="abc12345",
            artifact_uri=str(agent_dir),
            status="queued",
            raw_status="queued",
            effective_status="queued",
        )
        session.add(submission)
        await session.flush()

        first_job = await enqueue_evaluation_job_for_submission(session, submission)
        second_job = await enqueue_evaluation_job_for_submission(session, submission)
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))

        assert first_job is not None
        assert second_job is not None
        assert first_job.id == second_job.id
        assert job_count == 1
        assert submission.latest_evaluation_job_id == first_job.id

        claimed = await claim_next_evaluation_job(session)
        duplicate_claim = await claim_next_evaluation_job(session)

        assert claimed is not None
        assert claimed.id == first_job.id
        assert claimed.status == "running"
        assert submission.status == "evaluating"
        assert duplicate_claim is None

        empty_run = await run_next_evaluation_job(session, executor=executor)

        assert empty_run is None
        assert executor.calls == 0

        summary = await run_evaluation_job(session, claimed.job_id, executor=executor)
        repeated_summary = await run_evaluation_job(session, claimed.job_id, executor=executor)

        assert summary.status == "completed"
        assert repeated_summary.status == "completed"
        assert executor.calls == 2
        assert executor.tasks == ["analyzer", "task-a"]


async def test_legacy_normal_role_runs_existing_queued_job(
    database_session,
    monkeypatch,
    tmp_path,
):
    """The legacy ``normal`` role is inert: a node still claims and runs work."""
    patch_single_task(monkeypatch)
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "normal")
    executor = CountingExecutor()
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="hotkey-normal",
            name="agent-normal",
            agent_hash="normal12345",
            artifact_uri=str(agent_dir),
            status="queued",
            raw_status="queued",
            effective_status="queued",
        )
        session.add(submission)
        await session.flush()

        job = await enqueue_evaluation_job_for_submission(session, submission)
        assert job is not None
        claimed = await claim_next_evaluation_job(session)
        assert claimed is not None

        summary = await run_evaluation_job(session, claimed.job_id, executor=executor)

        report_count = await session.scalar(select(func.count(AnalyzerReport.id)))
        task_result_count = await session.scalar(select(func.count(TaskResult.id)))

    assert summary.status == "completed"
    assert executor.calls == 2
    assert executor.tasks == ["analyzer", "task-a"]
    assert report_count == 1
    assert task_result_count == 1
