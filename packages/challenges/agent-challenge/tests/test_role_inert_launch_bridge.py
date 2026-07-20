"""Decentralization: validator_role is inert and the central launch bridge is gone.

These encode the M4 removal contract (VAL-AC-001/002/003/027): no
``is_master_validator()`` precondition makes a node inert, every eligible node
executes assigned work identically regardless of ``validator_role``, and the old
centralized internal launch bridge route no longer drives execution.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from agent_challenge.evaluation import (
    claim_next_evaluation_job,
    enqueue_evaluation_job_for_submission,
    run_evaluation_job,
)
from agent_challenge.evaluation.runner import create_evaluation_job
from agent_challenge.evaluation.worker import run_worker_once
from agent_challenge.models import AgentSubmission, EvaluationAttempt, EvaluationJob, TaskResult
from agent_challenge.sdk.executors import DockerRunResult
from agent_challenge.swe_forge import SweForgeTask

ROLES = ("normal", "master")


class ValidReport:
    rules_version = "rules-test"
    overall_verdict = "valid"
    reason_codes = ["rules_passed"]

    def to_json_compatible(self) -> dict[str, object]:
        return {
            "rules_version": self.rules_version,
            "overall_verdict": self.overall_verdict,
            "reason_codes": self.reason_codes,
        }


class RecordingExecutor:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def run(self, spec, timeout_seconds: int):
        task = spec.labels["base.task"]
        self.tasks.append(task)
        return DockerRunResult(
            container_name="fake",
            stdout=f"ran {task}",
            stderr="",
            returncode=0,
        )


class SuccessfulTerminalBenchExecutor:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def run(self, spec, timeout_seconds: int):
        task = spec.labels["base.task"]
        self.tasks.append(task)
        if task == "analyzer":
            return DockerRunResult(container_name="analyzer", stdout="ok", stderr="", returncode=0)
        return DockerRunResult(
            container_name="terminal-bench",
            stdout='BASE_BENCHMARK_RESULT={"score":1.0,"status":"completed"}',
            stderr="",
            returncode=0,
        )


def _patch_swe_forge_env(monkeypatch, *, role: str | None) -> None:
    if role is not None:
        monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", role)
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 1)
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_concurrency", 1)
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.settings.benchmark_backend", "swe_forge"
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.load_swe_forge_tasks",
        lambda: [SweForgeTask(task_id="task-a", docker_image="baseintelligence/swe-forge:task-a")],
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.run_rules_analyzer",
        lambda _workspace, *, reviewer=None: ValidReport(),
    )


def _patch_terminal_bench_env(monkeypatch, tmp_path, *, role: str) -> None:
    settings_paths = (
        "agent_challenge.evaluation.benchmarks.settings",
        "agent_challenge.evaluation.runner.settings",
        "agent_challenge.evaluation.terminal_bench.settings",
    )
    for settings_path in settings_paths:
        monkeypatch.setattr(f"{settings_path}.validator_role", role)
        monkeypatch.setattr(f"{settings_path}.benchmark_backend", "terminal_bench")
        monkeypatch.setattr(f"{settings_path}.terminal_bench_execution_backend", "own_runner")
        monkeypatch.setattr(f"{settings_path}.terminal_bench_task_ids", ("hello-world",))
        monkeypatch.setattr(f"{settings_path}.evaluation_task_count", 1)
        monkeypatch.setattr(f"{settings_path}.evaluation_concurrency", 1)
        monkeypatch.setattr(f"{settings_path}.artifact_root", str(tmp_path / "artifacts"))
        monkeypatch.setattr(f"{settings_path}.docker_enabled", True)
        monkeypatch.setattr(f"{settings_path}.docker_backend", "broker")
        monkeypatch.setattr(f"{settings_path}.docker_broker_url", "https://platform-broker.test")
        monkeypatch.setattr(f"{settings_path}.docker_broker_token", "broker-token")
        monkeypatch.setattr(f"{settings_path}.docker_broker_token_file", None)
        monkeypatch.setattr(
            f"{settings_path}.docker_allowed_images",
            (
                "ghcr.io/baseintelligence/agent-challenge-analyzer:1.0",
                "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
            ),
        )
        monkeypatch.setattr(
            f"{settings_path}.harbor_runner_image",
            "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
        )
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.run_rules_analyzer",
        lambda _workspace, *, reviewer=None: ValidReport(),
    )


async def _create_queued_submission(database_session, tmp_path, *, agent_hash: str) -> str:
    agent_dir = tmp_path / agent_hash
    agent_dir.mkdir(parents=True, exist_ok=True)
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=f"hotkey-{agent_hash}",
            name=f"agent-{agent_hash}",
            agent_hash=agent_hash,
            artifact_uri=str(agent_dir),
            status="queued",
            raw_status="queued",
            effective_status="queued",
        )
        session.add(submission)
        await session.flush()
        job = await enqueue_evaluation_job_for_submission(session, submission)
        assert job is not None
        await session.commit()
        return job.job_id


async def test_worker_once_executes_for_non_master_role(database_session, monkeypatch, tmp_path):
    """VAL-AC-002/003: a normal/unset node claims and runs an assigned task."""
    _patch_swe_forge_env(monkeypatch, role="normal")
    job_id = await _create_queued_submission(database_session, tmp_path, agent_hash="normal-exec")
    executor = RecordingExecutor()

    iteration = await run_worker_once(worker_id="normal-worker", executor=executor)

    assert iteration.summary is not None
    assert iteration.summary.status == "completed"
    assert iteration.summary.total_tasks > 0
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        task_result_count = await session.scalar(select(func.count(TaskResult.id)))
    assert job is not None
    assert job.status == "completed"
    assert task_result_count >= 1


@pytest.mark.parametrize("role", (None, "normal", "master"))
async def test_execution_outcome_is_identical_across_roles(
    database_session, monkeypatch, tmp_path, role
):
    """VAL-AC-001/027: equal completed job + task results regardless of role."""
    _patch_swe_forge_env(monkeypatch, role=role)
    agent_hash = f"role-{role or 'unset'}"
    job_id = await _create_queued_submission(database_session, tmp_path, agent_hash=agent_hash)
    executor = RecordingExecutor()

    async with database_session() as session:
        claimed = await claim_next_evaluation_job(session)
        assert claimed is not None
        assert claimed.job_id == job_id
        summary = await run_evaluation_job(session, claimed.job_id, executor=executor)
        await session.commit()

    assert summary.status == "completed"
    assert summary.total_tasks == 1
    assert summary.passed_tasks == 1
    async with database_session() as session:
        results = (await session.execute(select(TaskResult))).scalars().all()
    assert len(results) == 1
    assert results[0].score == 1.0


async def test_terminal_bench_completion_without_master_role(
    database_session, monkeypatch, tmp_path
):
    """VAL-AC-001: a non-master node drives a submission to tb_completed."""
    for name in ("DAYTONA_API_KEY", "DAYTONA_JWT_TOKEN", "DAYTONA_ORGANIZATION_ID"):
        monkeypatch.delenv(name, raising=False)
    _patch_terminal_bench_env(monkeypatch, tmp_path, role="normal")
    agent_dir = tmp_path / "tb-normal"
    agent_dir.mkdir(parents=True, exist_ok=True)
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="hotkey-tb-normal",
            name="agent-tb-normal",
            agent_hash="tb-normal",
            artifact_uri=str(agent_dir),
        )
        session.add(submission)
        await session.flush()
        job = await create_evaluation_job(session, submission)
        await session.commit()
        job_id = job.job_id

    executor = SuccessfulTerminalBenchExecutor()
    iteration = await run_worker_once(worker_id="tb-normal-worker", executor=executor)

    assert iteration.summary is not None
    assert iteration.summary.status == "completed"
    assert executor.tasks == ["analyzer", "hello-world"]
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        submission = await session.get(AgentSubmission, job.submission_id if job else None)
        attempts = (await session.execute(select(EvaluationAttempt))).scalars().all()
        task_result_count = await session.scalar(select(func.count(TaskResult.id)))
    assert job is not None
    assert job.status == "completed"
    assert submission is not None
    assert submission.raw_status == "tb_completed"
    assert submission.effective_status == "valid"
    assert task_result_count >= 1
    assert [attempt.status for attempt in attempts] == ["completed"]


async def test_internal_launch_bridge_route_is_removed(client, database_session, internal_headers):
    """VAL-AC-027: the centralized internal launch bridge no longer exists."""
    agent_dir = "/tmp/launch-bridge-removed"
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="hotkey-bridge",
            name="agent-bridge",
            agent_hash="bridge-hash",
            artifact_uri=agent_dir,
            status="received",
            raw_status="waiting_miner_env",
            effective_status="waiting environments",
        )
        session.add(submission)
        await session.flush()
        submission_id = submission.id
        await session.commit()

    response = await client.post(
        f"/internal/v1/submissions/{submission_id}/launch",
        headers=internal_headers,
    )

    assert response.status_code == 404
    async with database_session() as session:
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
    assert job_count == 0
