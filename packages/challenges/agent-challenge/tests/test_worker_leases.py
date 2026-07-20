from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from agent_challenge.analyzer.lifecycle import claim_next_analysis_submission
from agent_challenge.evaluation import runner as runner_module
from agent_challenge.evaluation.benchmarks import BenchmarkTask
from agent_challenge.evaluation.leases import heartbeat_evaluation_attempt
from agent_challenge.evaluation.runner import EvaluationSummary, run_evaluation_job
from agent_challenge.evaluation.terminal_bench import (
    create_terminal_bench_attempt,
    finalize_terminal_bench_attempt,
    reconcile_stale_terminal_bench_attempts,
)
from agent_challenge.models import (
    AgentSubmission,
    AnalysisRun,
    EvaluationAttempt,
    EvaluationJob,
    SubmissionStatusEvent,
)


async def test_expired_analysis_lease_requeues_and_claims(database_session, monkeypatch, tmp_path):
    monkeypatch.setattr("agent_challenge.analyzer.lifecycle.settings.validator_role", "master")
    expired_at = datetime.now(UTC) - timedelta(seconds=5)
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-analysis-lease",
            name="analysis-lease-agent",
            agent_hash="analysis-lease-hash",
            artifact_uri=str(tmp_path / "agent.zip"),
            raw_status="llm_running",
            status="analysis_running",
            effective_status="analysis_running",
        )
        session.add(submission)
        await session.flush()
        run = AnalysisRun(
            submission_id=submission.id,
            analyzer_name="blocking_analyzer",
            analyzer_version="test",
            status="running",
            lease_owner="dead-analyzer",
            lease_expires_at=expired_at,
            heartbeat_at=expired_at,
            started_at=expired_at,
        )
        session.add(run)
        await session.commit()

    async with database_session() as session:
        claimed = await claim_next_analysis_submission(session, lease_owner="new-analyzer")
        await session.commit()

    assert claimed is not None
    assert claimed.raw_status == "ast_running"
    async with database_session() as session:
        run = await session.scalar(select(AnalysisRun))
        events = (
            (
                await session.execute(
                    select(SubmissionStatusEvent).order_by(SubmissionStatusEvent.sequence)
                )
            )
            .scalars()
            .all()
        )

    assert run is not None
    assert run.status == "expired_reclaimed"
    assert run.lease_owner is None
    assert json.loads(run.report_json)["lease_recovery"]["reclaimed_by"] == "new-analyzer"
    assert [event.to_status for event in events] == ["analysis_queued", "ast_running"]
    assert events[0].reason == "blocking_analysis_lease_expired"


async def test_terminal_bench_expired_attempt_reconciles_before_stale_completion(
    database_session,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root", str(tmp_path)
    )
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(session, tmp_path, raw_status="tb_running")
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
            lease_seconds=-1,
        )
        await session.commit()

    async with database_session() as session:
        reconciled = await reconcile_stale_terminal_bench_attempts(session)
        stale_outcome = await finalize_terminal_bench_attempt(
            session,
            plan=plan,
            task=task,
            run_payload={"status": "completed", "score": 1.0},
            normalized_status="completed",
            normalized_score=1.0,
            reason_code=None,
            returncode=0,
            timed_out=False,
        )
        await session.commit()

    assert reconciled == 1
    assert stale_outcome.status == "stale"
    async with database_session() as session:
        attempt = await session.scalar(select(EvaluationAttempt))

    assert attempt is not None
    assert attempt.status == "failed_retryable"
    assert attempt.error == "terminal_bench_lease_expired"
    assert attempt.lease_owner is None
    assert json.loads(attempt.metadata_json)["lease_recovery"]["retryable"] is True


async def test_attempt_heartbeat_updates_running_lease(database_session, tmp_path):
    old = datetime.now(UTC) - timedelta(minutes=10)
    async with database_session() as session:
        submission, job = await _submission_and_job(session, tmp_path)
        attempt = EvaluationAttempt(
            submission_id=submission.id,
            job_id=job.id,
            attempt_number=1,
            evaluator_name="terminal_bench",
            status="running",
            lease_owner="worker-a",
            lease_expires_at=old,
            heartbeat_at=old,
            started_at=old,
        )
        session.add(attempt)
        await session.commit()
        attempt_id = attempt.id

    async with database_session() as session:
        updated = await heartbeat_evaluation_attempt(
            session,
            attempt_id,
            lease_owner="worker-a",
            lease_seconds=60,
        )
        await session.commit()

    assert updated is True
    async with database_session() as session:
        attempt = await session.get(EvaluationAttempt, attempt_id)

    assert attempt is not None
    old_naive = old.replace(tzinfo=None)
    assert attempt.heartbeat_at is not None and attempt.heartbeat_at > old_naive
    assert attempt.lease_expires_at is not None and attempt.lease_expires_at > old_naive


async def test_stale_evaluation_completion_does_not_overwrite_newer_terminal_state(
    database_session,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    async with database_session() as session:
        submission, job = await _submission_and_job(session, tmp_path, raw_status="tb_running")
        job.status = "running"
        job.lease_owner = "worker-a"
        job.lease_expires_at = datetime.now(UTC) + timedelta(minutes=5)
        job.heartbeat_at = datetime.now(UTC)
        job.selected_tasks_json = "[]"
        await session.commit()
        job_id = job.job_id

    async def finish_elsewhere(session, executor, submission, job, tasks, *, lease_owner=None):
        job.status = "completed"
        job.score = 0.25
        submission.raw_status = "tb_completed"
        submission.status = "valid"
        submission.effective_status = "valid"
        await session.flush()
        return []

    monkeypatch.setattr("agent_challenge.evaluation.runner._run_tasks", finish_elsewhere)
    async with database_session() as session:
        summary = await run_evaluation_job(session, job_id)
        await session.commit()

    assert isinstance(summary, EvaluationSummary)
    assert summary.status == "completed"
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        submission = await session.scalar(select(AgentSubmission))

    assert job is not None
    assert job.score == 0.25
    assert submission is not None
    assert submission.raw_status == "tb_completed"


async def _submission_and_job(
    session,
    tmp_path: Path,
    *,
    raw_status: str = "tb_queued",
):
    agent_dir = tmp_path / f"agent-{raw_status}"
    agent_dir.mkdir(exist_ok=True)
    submission = AgentSubmission(
        miner_hotkey=f"miner-{raw_status}",
        name=f"agent-{raw_status}",
        agent_hash=f"hash-{raw_status}",
        artifact_uri=str(agent_dir),
        raw_status=raw_status,
        status="evaluating" if raw_status == "tb_running" else "queued",
        effective_status="evaluating" if raw_status == "tb_running" else "queued",
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=f"job-{raw_status}",
        submission_id=submission.id,
        status="running",
        selected_tasks_json="[]",
        total_tasks=0,
        attempt_count=1,
        lease_owner="worker-a",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        heartbeat_at=datetime.now(UTC),
    )
    session.add(job)
    await session.flush()
    return submission, job


def _terminal_bench_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="hello-world",
        docker_image="ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
        benchmark="terminal_bench",
        metadata={"task_id": "hello-world"},
    )


async def test_mark_job_running_is_idempotent_when_submission_already_running(
    database_session, tmp_path
):
    async with database_session() as session:
        submission, job = await _submission_and_job(session, tmp_path, raw_status="tb_running")
        await session.refresh(job, attribute_names=["submission"])

        await runner_module._mark_job_running(session, job)

        events = (
            (
                await session.execute(
                    select(SubmissionStatusEvent).where(
                        SubmissionStatusEvent.submission_id == submission.id
                    )
                )
            )
            .scalars()
            .all()
        )

    assert events == []
    assert job.status == "running"
    assert job.started_at is not None
    assert job.heartbeat_at is not None
    assert submission.raw_status == "tb_running"
