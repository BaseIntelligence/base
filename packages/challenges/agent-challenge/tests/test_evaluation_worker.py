from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from agent_challenge.analyzer.lifecycle import AnalysisSummary
from agent_challenge.evaluation.runner import (
    _terminal_bench_env,
    claim_next_evaluation_job_for_worker,
    create_evaluation_job,
    reset_stale_evaluation_jobs,
)
from agent_challenge.evaluation.worker import run_worker_once
from agent_challenge.models import AgentSubmission, EvaluationAttempt, EvaluationJob, TaskLogEvent
from agent_challenge.sdk.executors import DockerRunResult
from agent_challenge.swe_forge import SweForgeTask

ROOT = Path(__file__).resolve().parents[1]


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
    def __init__(self, *, fail_task: bool = False) -> None:
        self.fail_task = fail_task
        self.tasks: list[str] = []

    def run(self, spec, timeout_seconds: int):
        task = spec.labels["base.task"]
        self.tasks.append(task)
        if self.fail_task and task != "analyzer":
            raise RuntimeError("broker unavailable")
        return DockerRunResult(
            container_name="fake",
            stdout=f"ran {task}",
            stderr="",
            returncode=0,
        )


class RetryableTerminalBenchExecutor:
    def __init__(self) -> None:
        self.tasks: list[str] = []
        self.scripts: list[str] = []

    def run(self, spec, timeout_seconds: int):
        task = spec.labels["base.task"]
        self.tasks.append(task)
        if task == "analyzer":
            return DockerRunResult(
                container_name="analyzer",
                stdout="ok",
                stderr="",
                returncode=0,
            )
        self.scripts.append(spec.command[2])
        return DockerRunResult(
            container_name="terminal-bench",
            stdout=(
                'BASE_BENCHMARK_RESULT={"reason_code":"harbor_broker_connection_failed",'
                '"score":0.0,"status":"failed"}'
            ),
            stderr="broker connection failed",
            returncode=1,
        )


class SuccessfulTerminalBenchExecutor:
    def __init__(self) -> None:
        self.tasks: list[str] = []
        self.scripts: list[str] = []
        self.envs: list[dict[str, str]] = []

    def run(self, spec, timeout_seconds: int):
        task = spec.labels["base.task"]
        self.tasks.append(task)
        if task == "analyzer":
            return DockerRunResult(
                container_name="analyzer",
                stdout="ok",
                stderr="",
                returncode=0,
            )
        self.scripts.append(spec.command[2])
        self.envs.append(dict(spec.env))
        return DockerRunResult(
            container_name="terminal-bench",
            stdout='BASE_BENCHMARK_RESULT={"score":1.0,"status":"completed"}',
            stderr="",
            returncode=0,
        )


def test_worker_console_help_available_from_checkout() -> None:
    env = os.environ.copy()
    env["PATH"] = f"{ROOT}{os.pathsep}{env.get('PATH', '')}"

    result = subprocess.run(
        ["agent-challenge-worker", "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Run the Agent Challenge evaluation worker." in result.stdout
    assert "--once" in result.stdout


def test_terminal_bench_env_keeps_controlled_writable_cache(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.settings.harbor_forward_env_vars",
        ("HOME", "XDG_CACHE_HOME", "OPERATOR_VISIBLE"),
    )
    monkeypatch.setenv("HOME", "/operator-home")
    monkeypatch.setenv("XDG_CACHE_HOME", "/operator-cache")
    monkeypatch.setenv("OPERATOR_VISIBLE", "operator-value")

    env = _terminal_bench_env(
        {
            "HOME": "/.cache",
            "XDG_CACHE_HOME": "/.cache",
            "BASE_AGENT_PATH": "/bad-agent",
            "BASE_BENCHMARK_DATASET": "bad-dataset",
            "OPENROUTER_API_KEY": "sk-or-miner",
            "HTTPS_PROXY": "http://evil.example:8080",
            "MINER_VISIBLE": "miner-value",
        }
    )

    assert env["HOME"] == "/tmp"
    assert env["XDG_CACHE_HOME"] == "/tmp/.cache"
    assert env["BASE_AGENT_PATH"] == "/workspace/agent"
    assert env["BASE_BENCHMARK_DATASET"] == "terminal-bench/terminal-bench-2-1"
    assert env["OPERATOR_VISIBLE"] == "operator-value"
    assert env["OPENROUTER_API_KEY"] == "sk-or-miner"
    # VAL-ACLOCK: plain/non-token and proxy keys are not forwarded.
    assert "MINER_VISIBLE" not in env
    assert "HTTPS_PROXY" not in env


def patch_worker_environment(monkeypatch, *, role: str = "master") -> None:
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", role)
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 1)
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_concurrency", 1)
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.settings.benchmark_backend",
        "swe_forge",
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.load_swe_forge_tasks",
        lambda: [SweForgeTask(task_id="task-a", docker_image="baseintelligence/swe-forge:task-a")],
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.run_rules_analyzer",
        lambda _workspace, *, reviewer=None: ValidReport(),
    )


def patch_own_runner_terminal_bench_worker_environment(monkeypatch, tmp_path) -> None:
    settings_paths = (
        "agent_challenge.evaluation.benchmarks.settings",
        "agent_challenge.evaluation.runner.settings",
        "agent_challenge.evaluation.terminal_bench.settings",
    )
    for settings_path in settings_paths:
        monkeypatch.setattr(f"{settings_path}.validator_role", "master")
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


async def create_submission_with_job(database_session, tmp_path, *, job_id: str = "worker-job"):
    agent_dir = tmp_path / job_id
    agent_dir.mkdir()
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=f"hotkey-{job_id}",
            name=f"agent-{job_id}",
            agent_hash=f"hash-{job_id}",
            artifact_uri=str(agent_dir),
        )
        session.add(submission)
        await session.flush()
        job = await create_evaluation_job(session, submission)
        await session.commit()
        return job.job_id


async def test_worker_once_processes_queued_master_job_to_terminal_status(
    database_session,
    monkeypatch,
    tmp_path,
):
    patch_worker_environment(monkeypatch)
    job_id = await create_submission_with_job(database_session, tmp_path)
    executor = RecordingExecutor()

    iteration = await run_worker_once(worker_id="worker-a", executor=executor)

    assert iteration.summary is not None
    assert iteration.summary.status == "completed"
    assert executor.tasks == ["analyzer", "task-a"]
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert job is not None
    assert job.status == "completed"
    assert job.attempt_count == 1
    assert job.lease_owner is None
    assert job.lease_expires_at is None
    assert job.heartbeat_at is None
    assert job.started_at is not None
    assert job.finished_at is not None


async def test_worker_once_processes_queued_job_on_legacy_normal_role(
    database_session,
    monkeypatch,
    tmp_path,
):
    """The legacy ``normal`` role is inert: a worker iteration still does work."""
    patch_worker_environment(monkeypatch, role="normal")
    job_id = await create_submission_with_job(database_session, tmp_path, job_id="normal-job")

    iteration = await run_worker_once(worker_id="normal-worker", executor=RecordingExecutor())

    assert iteration.summary is not None
    assert iteration.summary.status == "completed"
    assert iteration.summary.total_tasks > 0
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert job is not None
    assert job.status == "completed"
    assert job.attempt_count == 1
    assert job.lease_owner is None


async def test_duplicate_claims_do_not_double_dispatch(database_session, monkeypatch, tmp_path):
    patch_worker_environment(monkeypatch)
    job_id = await create_submission_with_job(database_session, tmp_path, job_id="duplicate-job")

    async with database_session() as session:
        claimed = await claim_next_evaluation_job_for_worker(
            session,
            lease_owner="worker-a",
            lease_seconds=60,
        )
        await session.commit()
    async with database_session() as session:
        duplicate = await claim_next_evaluation_job_for_worker(
            session,
            lease_owner="worker-b",
            lease_seconds=60,
        )
        await session.commit()

    assert claimed is not None
    assert claimed.job_id == job_id
    assert duplicate is None
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert job is not None
    assert job.status == "running"
    assert job.lease_owner == "worker-a"
    assert job.attempt_count == 1


async def test_claim_sets_makespan_lease_not_default(database_session, monkeypatch, tmp_path):
    patch_worker_environment(monkeypatch)
    runner_settings = "agent_challenge.evaluation.runner.settings"
    monkeypatch.setattr(f"{runner_settings}.evaluation_task_count", 20)
    monkeypatch.setattr(f"{runner_settings}.evaluation_concurrency", 4)
    monkeypatch.setattr(f"{runner_settings}.evaluation_timeout_seconds", 3600)
    await create_submission_with_job(database_session, tmp_path, job_id="lease-job")

    async with database_session() as session:
        claimed = await claim_next_evaluation_job_for_worker(session, lease_owner="worker-a")
        await session.commit()

    assert claimed is not None
    assert claimed.lease_expires_at is not None
    expires = claimed.lease_expires_at
    now_ref = datetime.now(UTC)
    if expires.tzinfo is None:
        now_ref = now_ref.replace(tzinfo=None)
    remaining = (expires - now_ref).total_seconds()
    assert 21000 <= remaining <= 21600


async def test_own_runner_worker_claims_queued_terminal_bench_job_as_running_evaluating(
    database_session,
    monkeypatch,
    tmp_path,
):
    for name in ("DAYTONA_API_KEY", "DAYTONA_JWT_TOKEN", "DAYTONA_ORGANIZATION_ID"):
        monkeypatch.delenv(name, raising=False)
    patch_own_runner_terminal_bench_worker_environment(monkeypatch, tmp_path)
    job_id = await create_submission_with_job(
        database_session,
        tmp_path,
        job_id="platform-sdk-claim-job",
    )

    async with database_session() as session:
        claimed = await claim_next_evaluation_job_for_worker(
            session,
            lease_owner="platform-sdk-claimer",
            lease_seconds=60,
        )
        await session.commit()

    assert claimed is not None
    assert claimed.job_id == job_id
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        submission = await session.get(AgentSubmission, job.submission_id if job else None)

    assert job is not None
    assert submission is not None
    assert job.status == "running"
    assert job.attempt_count == 1
    assert job.lease_owner == "platform-sdk-claimer"
    assert job.lease_expires_at is not None
    assert job.heartbeat_at is not None
    assert submission.raw_status == "evaluating"
    assert submission.effective_status == "evaluating"


async def test_claim_errors_orphaned_queued_job_instead_of_crashing(
    database_session,
    monkeypatch,
    tmp_path,
):
    """An orphaned queued job (submission already finalized to a terminal state)
    must be errored on claim, not crash the worker with an invalid transition.

    Regression: previously the invalid ``tb_failed_final -> evaluating`` claim
    transition raised and the worker re-claimed the same poison job every loop,
    blocking all evaluation progress.
    """
    patch_worker_environment(monkeypatch)
    job_id = await create_submission_with_job(database_session, tmp_path, job_id="orphan-claim-job")
    # Simulate the reconciler finalizing the submission to a terminal state while
    # its job stayed queued: tb_failed_final cannot transition to evaluating.
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        submission = await session.get(AgentSubmission, job.submission_id)
        submission.raw_status = "tb_failed_final"
        await session.commit()

    async with database_session() as session:
        claimed = await claim_next_evaluation_job_for_worker(
            session,
            lease_owner="orphan-claimer",
            lease_seconds=60,
        )
        await session.commit()

    assert claimed is None
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert job is not None
    assert job.status == "error"
    assert "claim aborted" in job.last_error
    assert job.lease_owner is None
    assert job.lease_expires_at is None


async def test_stale_leases_requeue_until_retry_cap_then_error(
    database_session,
    monkeypatch,
    tmp_path,
):
    patch_worker_environment(monkeypatch)
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    agent_dir = tmp_path / "stale-agent"
    agent_dir.mkdir()
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="stale-hotkey",
            name="stale-agent",
            agent_hash="stale-hash",
            artifact_uri=str(agent_dir),
        )
        session.add(submission)
        await session.flush()
        retry_job = EvaluationJob(
            job_id="stale-retry",
            submission_id=submission.id,
            status="running",
            selected_tasks_json="[]",
            total_tasks=0,
            lease_owner="dead-worker",
            lease_expires_at=expired_at,
            heartbeat_at=expired_at,
            attempt_count=2,
        )
        capped_job = EvaluationJob(
            job_id="stale-capped",
            submission_id=submission.id,
            status="running",
            selected_tasks_json="[]",
            total_tasks=0,
            lease_owner="dead-worker",
            lease_expires_at=expired_at,
            heartbeat_at=expired_at,
            attempt_count=3,
        )
        session.add_all([retry_job, capped_job])
        await session.commit()

    async with database_session() as session:
        stale_count = await reset_stale_evaluation_jobs(session)
        await session.commit()

    assert stale_count == 2
    async with database_session() as session:
        retry_job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == "stale-retry")
        )
        capped_job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == "stale-capped")
        )
    assert retry_job is not None
    assert retry_job.status == "queued"
    assert retry_job.last_error == "stale lease expired"
    assert retry_job.lease_owner is None
    assert capped_job is not None
    assert capped_job.status == "error"
    assert capped_job.last_error == "stale lease expired"
    assert capped_job.error == "stale lease expired"


async def test_worker_retries_transient_failures_until_attempt_cap(
    database_session,
    monkeypatch,
    tmp_path,
):
    patch_worker_environment(monkeypatch)
    job_id = await create_submission_with_job(database_session, tmp_path, job_id="retry-cap-job")
    executor = RecordingExecutor(fail_task=True)

    first = await run_worker_once(worker_id="worker-a", executor=executor)
    second = await run_worker_once(worker_id="worker-a", executor=executor)
    third = await run_worker_once(worker_id="worker-a", executor=executor)

    assert first.summary is not None
    assert second.summary is not None
    assert third.summary is not None
    assert first.summary.status == "failed"
    assert second.summary.status == "failed"
    assert third.summary.status == "error"
    assert executor.tasks == ["analyzer", "task-a", "analyzer", "task-a", "analyzer", "task-a"]
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert job is not None
    assert job.status == "error"
    assert job.attempt_count == 3
    assert job.last_error == "broker unavailable"
    assert job.error == "broker unavailable"


async def test_own_runner_worker_requeues_retryable_terminal_bench_broker_failure(
    database_session,
    monkeypatch,
    tmp_path,
):
    for name in ("DAYTONA_API_KEY", "DAYTONA_JWT_TOKEN", "DAYTONA_ORGANIZATION_ID"):
        monkeypatch.delenv(name, raising=False)
    patch_own_runner_terminal_bench_worker_environment(monkeypatch, tmp_path)
    job_id = await create_submission_with_job(
        database_session,
        tmp_path,
        job_id="platform-sdk-retry-cap-job",
    )
    executor = RetryableTerminalBenchExecutor()

    first = await run_worker_once(worker_id="platform-sdk-worker", executor=executor)
    second = await run_worker_once(worker_id="platform-sdk-worker", executor=executor)
    third = await run_worker_once(worker_id="platform-sdk-worker", executor=executor)

    assert first.summary is not None
    assert second.summary is not None
    assert third.summary is not None
    assert first.summary.status == "failed"
    assert second.summary.status == "failed"
    assert third.summary.status == "error"
    assert executor.tasks == [
        "analyzer",
        "hello-world",
        "analyzer",
        "hello-world",
        "analyzer",
        "hello-world",
    ]
    assert len(executor.scripts) == 3
    assert all("--environment-import-path" not in script for script in executor.scripts)
    assert all("--env daytona" not in script for script in executor.scripts)
    assert all("DAYTONA_" not in script for script in executor.scripts)
    assert all("broker-token" not in script for script in executor.scripts)

    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        attempts = (
            (
                await session.execute(
                    select(EvaluationAttempt).order_by(EvaluationAttempt.attempt_number)
                )
            )
            .scalars()
            .all()
        )
        submission = await session.get(AgentSubmission, job.submission_id if job else None)
        task_events = (
            (
                await session.execute(
                    select(TaskLogEvent)
                    .where(TaskLogEvent.submission_id == (job.submission_id if job else None))
                    .where(TaskLogEvent.task_id == "hello-world")
                    .order_by(TaskLogEvent.sequence)
                )
            )
            .scalars()
            .all()
        )

    assert job is not None
    assert submission is not None
    assert job.status == "error"
    assert job.attempt_count == 3
    assert job.last_error == "harbor_broker_connection_failed"
    assert job.error == "harbor_broker_connection_failed"
    assert submission.raw_status == "valid"
    assert [(attempt.attempt_number, attempt.status, attempt.error) for attempt in attempts] == [
        (1, "failed_retryable", "harbor_broker_connection_failed"),
        (2, "failed_retryable", "harbor_broker_connection_failed"),
        (3, "failed", "harbor_broker_connection_failed"),
    ]
    status_events = [event for event in task_events if event.event_type == "task.status"]
    assert [event.status for event in status_events] == [
        "assigned",
        "starting",
        "waiting",
        "running",
        "failed",
        "assigned",
        "starting",
        "waiting",
        "running",
        "failed",
        "assigned",
        "starting",
        "waiting",
        "running",
        "failed",
    ]
    serialized = json.dumps(
        [
            {
                "event_type": event.event_type,
                "message": event.message,
                "metadata": json.loads(event.metadata_json),
                "status": event.status,
                "task_id": event.task_id,
            }
            for event in status_events
        ],
        sort_keys=True,
    )
    for forbidden in (
        "platform_sdk",
        "base_sdk",
        "tb21-",
        "broker-token",
        "platform-sdk-worker",
        "terminal-bench/jobs",
        "platform-terminal-bench-command.sh",
        "agent_challenge_runner.platform_environment",
        "provider",
        "job_dir",
        "raw_ref",
        "pod",
        "worker",
        "command",
        "token",
    ):
        assert forbidden not in serialized


async def test_own_runner_worker_records_safe_task_phase_sequence(
    database_session,
    monkeypatch,
    tmp_path,
):
    for name in ("DAYTONA_API_KEY", "DAYTONA_JWT_TOKEN", "DAYTONA_ORGANIZATION_ID"):
        monkeypatch.delenv(name, raising=False)
    patch_own_runner_terminal_bench_worker_environment(monkeypatch, tmp_path)
    job_id = await create_submission_with_job(
        database_session,
        tmp_path,
        job_id="platform-sdk-phase-job",
    )
    executor = SuccessfulTerminalBenchExecutor()

    iteration = await run_worker_once(worker_id="platform-sdk-phase-worker", executor=executor)

    assert iteration.summary is not None
    assert iteration.summary.status == "completed"
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        assert job is not None
        events = (
            (
                await session.execute(
                    select(TaskLogEvent)
                    .where(TaskLogEvent.submission_id == job.submission_id)
                    .where(TaskLogEvent.task_id == "hello-world")
                    .order_by(TaskLogEvent.sequence)
                )
            )
            .scalars()
            .all()
        )

    status_events = [event for event in events if event.event_type == "task.status"]
    assert [event.status for event in status_events] == [
        "assigned",
        "starting",
        "waiting",
        "running",
        "completed",
    ]
    first_terminal_sequence = min(
        event.sequence
        for event in events
        if event.event_type in {"task.progress", "task.completed", "task.failed"}
    )
    assert status_events[3].sequence < first_terminal_sequence
    assert [json.loads(event.metadata_json) for event in status_events] == [
        {"benchmark": "terminal_bench", "phase": "assigned"},
        {"benchmark": "terminal_bench", "phase": "starting"},
        {"attempt": 1, "benchmark": "terminal_bench", "phase": "waiting"},
        {"attempt": 1, "benchmark": "terminal_bench", "phase": "running"},
        {"attempt": 1, "benchmark": "terminal_bench", "phase": "completed"},
    ]


async def test_own_runner_worker_completes_successful_terminal_bench_without_daytona(
    database_session,
    monkeypatch,
    tmp_path,
):
    for name in ("DAYTONA_API_KEY", "DAYTONA_JWT_TOKEN", "DAYTONA_ORGANIZATION_ID"):
        monkeypatch.delenv(name, raising=False)
    patch_own_runner_terminal_bench_worker_environment(monkeypatch, tmp_path)
    job_id = await create_submission_with_job(
        database_session,
        tmp_path,
        job_id="platform-sdk-success-job",
    )
    executor = SuccessfulTerminalBenchExecutor()

    iteration = await run_worker_once(worker_id="platform-sdk-success-worker", executor=executor)

    assert iteration.summary is not None
    assert iteration.summary.status == "completed"
    assert executor.tasks == ["analyzer", "hello-world"]
    assert len(executor.scripts) == 1
    script = executor.scripts[0]
    assert "--environment-import-path" not in script
    assert "--env daytona" not in script
    assert "--env platform" not in script
    assert "DAYTONA_" not in script
    assert "broker-token" not in script
    assert executor.envs == [
        {
            "BASE_AGENT_PATH": "/workspace/agent",
            "BASE_BENCHMARK_DATASET": "terminal-bench/terminal-bench-2-1",
            "HOME": "/tmp",
            "XDG_CACHE_HOME": "/tmp/.cache",
        }
    ]

    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        attempts = (
            (
                await session.execute(
                    select(EvaluationAttempt).order_by(EvaluationAttempt.attempt_number)
                )
            )
            .scalars()
            .all()
        )
        submission = await session.get(AgentSubmission, job.submission_id if job else None)

    assert job is not None
    assert submission is not None
    assert job.status == "completed"
    assert job.score == 1.0
    assert job.attempt_count == 1
    assert job.error == ""
    assert submission.raw_status == "tb_completed"
    assert submission.effective_status == "valid"
    assert [(attempt.attempt_number, attempt.status, attempt.error) for attempt in attempts] == [
        (1, "completed", ""),
    ]


async def test_worker_claims_eval_after_running_analysis(database_session, monkeypatch, tmp_path):
    patch_worker_environment(monkeypatch)
    job_id = await create_submission_with_job(
        database_session, tmp_path, job_id="post-analysis-job"
    )
    executor = RecordingExecutor()

    fake_summary = AnalysisSummary(
        analysis_run_id=1,
        submission_id=999,
        verdict="allow",
        status="allow",
        evaluation_job_id=None,
    )

    async def fake_run_next_analysis(session, *, lease_owner, **kwargs):
        return fake_summary

    monkeypatch.setattr(
        "agent_challenge.evaluation.worker.run_next_analysis", fake_run_next_analysis
    )

    iteration = await run_worker_once(worker_id="worker-a", executor=executor)

    assert iteration.analysis_summary is fake_summary
    assert iteration.summary is not None
    assert iteration.summary.status == "completed"
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert job is not None
    assert job.status == "completed"


async def test_worker_survives_analysis_exception(database_session, monkeypatch, tmp_path):
    patch_worker_environment(monkeypatch)
    job_id = await create_submission_with_job(database_session, tmp_path, job_id="resilient-job")
    executor = RecordingExecutor()

    async def boom(session, *, lease_owner, **kwargs):
        raise RuntimeError("connection was closed in the middle of operation")

    monkeypatch.setattr("agent_challenge.evaluation.worker.run_next_analysis", boom)

    iteration = await run_worker_once(worker_id="worker-a", executor=executor)

    assert iteration.analysis_summary is None
    assert iteration.summary is not None
    assert iteration.summary.status == "completed"
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert job is not None
    assert job.status == "completed"


async def test_run_evaluation_job_offloads_blocking_analyzer_work_off_event_loop(
    database_session, monkeypatch, tmp_path
):
    """Fix #1: the analyzer container Docker call and the CPU-bound rules analysis
    must run off the shared event-loop thread (via asyncio.to_thread)."""
    import threading

    patch_worker_environment(monkeypatch)
    main_thread = threading.get_ident()
    rules_threads: list[int] = []

    def _recording_rules(_workspace, *, reviewer=None):
        rules_threads.append(threading.get_ident())
        return ValidReport()

    monkeypatch.setattr("agent_challenge.evaluation.runner.run_rules_analyzer", _recording_rules)

    class ThreadRecordingExecutor:
        def __init__(self) -> None:
            self.run_threads: dict[str, int] = {}

        def run(self, spec, timeout_seconds: int):
            task = spec.labels["base.task"]
            self.run_threads[task] = threading.get_ident()
            return DockerRunResult(
                container_name="fake",
                stdout=f"ran {task}",
                stderr="",
                returncode=0,
            )

    await create_submission_with_job(database_session, tmp_path, job_id="offload-job")
    executor = ThreadRecordingExecutor()

    iteration = await run_worker_once(worker_id="offload-worker", executor=executor)

    assert iteration.summary is not None
    assert iteration.summary.status == "completed"
    # The analyzer container Docker call ran off the event-loop thread.
    assert executor.run_threads.get("analyzer") not in (None, main_thread)
    # The CPU-bound rules analysis ran off the event-loop thread.
    assert rules_threads and all(thread != main_thread for thread in rules_threads)
