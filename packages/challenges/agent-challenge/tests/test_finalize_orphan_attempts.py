"""Regression: the work-unit/fold finalize path leaves zero ``running`` attempts.

Two execution paths race on the same job. The combined worker's durable tasks
(``runner._run_terminal_bench_task_durable``) each COMMIT one ``running``
:class:`EvaluationAttempt` (with a long lease) before awaiting their container.
The master coordination / work-unit path folds every task into a ``TaskResult``
and :func:`finalize_job_if_complete` marks the job ``completed`` -- but that path
never created those attempts and, before this fix, never finalized them, so a job
could complete while all its attempts were stuck ``running`` (frozen lease,
``finished_at`` unset) until the reconciler's lease sweep churned over them.

These tests lock the invariant that after a job reaches a terminal status via the
work-unit/fold path it has ZERO ``running`` attempts, and that a slow durable
container finalizing its own attempt AFTER the job was completed is a safe no-op.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.terminal_bench import (
    create_terminal_bench_attempt,
    finalize_terminal_bench_attempt,
)
from agent_challenge.evaluation.validator_executor import (
    finalize_job_if_complete,
    fold_terminally_failed_work_unit,
)
from agent_challenge.models import (
    AgentSubmission,
    EvaluationAttempt,
    EvaluationJob,
    TerminalBenchTrial,
)


def _terminal_bench_tasks(count: int) -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id=f"terminal-bench/task-{index}",
            docker_image=f"ghcr.io/baseintelligence/terminal-bench-runner:{index}",
            prompt=f"task {index}",
            benchmark="terminal_bench",
            metadata={"task_id": f"terminal-bench/task-{index}"},
        )
        for index in range(count)
    ]


async def _create_running_job(
    session,
    *,
    agent_hash: str,
    tasks: list[BenchmarkTask],
    tmp_path,
) -> tuple[AgentSubmission, EvaluationJob]:
    """Create a ``tb_running`` submission + ``running`` job for ``tasks``."""

    agent_dir = tmp_path / agent_hash
    agent_dir.mkdir(parents=True, exist_ok=True)
    submission = AgentSubmission(
        miner_hotkey=f"hotkey-{agent_hash}",
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=str(agent_dir),
        status="evaluating",
        raw_status="tb_running",
        effective_status="evaluating",
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=uuid.uuid4().hex,
        submission_id=submission.id,
        status="running",
        selected_tasks_json=benchmark_tasks_to_json(tasks),
        total_tasks=len(tasks),
        verdict="valid",
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    return submission, job


async def _commit_running_attempts(session, submission, job, tasks):
    """Mirror the combined worker committing one ``running`` attempt per task."""

    for task in tasks:
        await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "run"),
            lease_owner="worker-a",
        )


async def test_finalize_sweeps_orphaned_running_attempts(database_session, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root", str(tmp_path)
    )
    tasks = _terminal_bench_tasks(3)
    async with database_session() as session:
        submission, job = await _create_running_job(
            session, agent_hash="orphan", tasks=tasks, tmp_path=tmp_path
        )
        # Path 1 (combined worker): a running attempt per task, live lease held.
        await _commit_running_attempts(session, submission, job, tasks)
        await session.commit()
        job_id = job.job_id
        job_pk = job.id

    # Sanity: every attempt is running with a frozen lease and no finish stamp.
    async with database_session() as session:
        running = (await session.execute(select(EvaluationAttempt))).scalars().all()
    assert len(running) == 3
    assert all(attempt.status == "running" for attempt in running)
    assert all(attempt.lease_owner == "worker-a" for attempt in running)
    assert all(attempt.finished_at is None for attempt in running)

    # Path 2 (master work-unit/fold): every task recorded as a TaskResult WITHOUT
    # going through the durable task's normal completion (which would finalize the
    # attempt). This is the exact production race that left attempts orphaned.
    async with database_session() as session:
        for task in tasks:
            await fold_terminally_failed_work_unit(session, job_id=job_id, task_id=task.task_id)
        await session.commit()

    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()

    assert summary is not None
    assert summary.status == "completed"

    async with database_session() as session:
        job_row = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        attempts = (
            (
                await session.execute(
                    select(EvaluationAttempt).where(EvaluationAttempt.job_id == job_pk)
                )
            )
            .scalars()
            .all()
        )
        submission_row = await session.scalar(select(AgentSubmission))

    assert job_row.status == "completed"
    assert submission_row.raw_status == "tb_completed"
    # THE INVARIANT: a terminal job has zero running attempts.
    assert [attempt for attempt in attempts if attempt.status == "running"] == []
    assert len(attempts) == 3
    for attempt in attempts:
        assert attempt.status in {"failed", "failed_retryable"}
        assert attempt.finished_at is not None
        assert attempt.lease_owner is None
        assert attempt.lease_expires_at is None
        assert attempt.heartbeat_at is None


async def test_late_container_finalize_after_job_completed_is_safe(
    database_session, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root", str(tmp_path)
    )
    tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        submission, job = await _create_running_job(
            session, agent_hash="late", tasks=tasks, tmp_path=tmp_path
        )
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=tasks[0],
            command=("bash", "-lc", "run"),
            lease_owner="worker-a",
        )
        await session.commit()
        job_id = job.job_id
        job_pk = job.id
        attempt_id = plan.attempt_id

    # Fold + finalize: the work-unit path completes the job and sweeps the still
    # running attempt to a terminal status.
    async with database_session() as session:
        await fold_terminally_failed_work_unit(session, job_id=job_id, task_id=tasks[0].task_id)
        await session.commit()
    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()
    assert summary is not None
    assert summary.status == "completed"

    async with database_session() as session:
        swept = await session.get(EvaluationAttempt, attempt_id)
    assert swept.status in {"failed", "failed_retryable"}
    swept_status = swept.status
    swept_finished_at = swept.finished_at

    # The slow durable container finally returns and finalizes ITS attempt AFTER
    # the job was already completed. This must be an idempotent no-op: no raise,
    # no double-write, no resurrection of the attempt to running.
    async with database_session() as session:
        outcome = await finalize_terminal_bench_attempt(
            session,
            plan=plan,
            task=tasks[0],
            run_payload={"status": "completed", "score": 1.0},
            normalized_status="completed",
            normalized_score=1.0,
            reason_code=None,
            returncode=0,
            timed_out=False,
        )
        await session.commit()
    assert outcome.status == "stale"

    async with database_session() as session:
        attempt = await session.get(EvaluationAttempt, attempt_id)
        trials = (
            (
                await session.execute(
                    select(TerminalBenchTrial).where(
                        TerminalBenchTrial.evaluation_attempt_id == attempt_id
                    )
                )
            )
            .scalars()
            .all()
        )
        job_row = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        still_running = (
            (
                await session.execute(
                    select(EvaluationAttempt)
                    .where(EvaluationAttempt.job_id == job_pk)
                    .where(EvaluationAttempt.status == "running")
                )
            )
            .scalars()
            .all()
        )

    # The late finalize left the swept attempt untouched, wrote no trial rows, and
    # the job stayed completed with zero running attempts.
    assert attempt.status == swept_status
    assert attempt.finished_at == swept_finished_at
    assert trials == []
    assert job_row.status == "completed"
    assert still_running == []

    # Re-finalizing the already-terminal job is itself a safe no-op.
    async with database_session() as session:
        again = await finalize_job_if_complete(session, job_id)
        await session.commit()
    assert again is not None
    assert again.status == "completed"
