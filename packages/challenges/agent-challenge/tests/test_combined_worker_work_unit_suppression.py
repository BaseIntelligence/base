"""Combined-worker mode suppresses the decentralized coordination plane.

When ``settings.combined_worker`` is true the in-process worker owns evaluation
end-to-end, so the challenge must NOT expose evaluation work units to the master:

1. ``GET /internal/v1/work_units`` returns an empty list even when assignable
   jobs with pending tasks exist, so the master has nothing to assign or fold.
2. ``POST /internal/v1/work_units/fold`` is a benign no-op: it writes no
   ``work_unit_max_attempts_exhausted`` result and does not finalize the job, so
   an already-assigned unit can never clobber the in-process worker's result.

With ``combined_worker`` off (the default) the existing decentralized behavior is
unchanged: units are exposed and folds are recorded. The internal
``list_pending_work_units`` derivation is never gated (the decentralized executor
still relies on it), only the two API routes.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select

from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.work_units import list_pending_work_units
from agent_challenge.models import AgentSubmission, EvaluationJob, TaskResult


def _tasks(count: int) -> list[BenchmarkTask]:
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


async def _create_job(
    session, *, agent_hash: str, tasks: list[BenchmarkTask], tmp_path
) -> tuple[AgentSubmission, EvaluationJob]:
    agent_dir = tmp_path / agent_hash
    agent_dir.mkdir(parents=True, exist_ok=True)
    submission = AgentSubmission(
        miner_hotkey=f"hotkey-{agent_hash}",
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=str(agent_dir),
        status="evaluation queued",
        raw_status="tb_queued",
        effective_status="evaluation queued",
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=uuid.uuid4().hex,
        submission_id=submission.id,
        status="queued",
        selected_tasks_json=benchmark_tasks_to_json(tasks),
        total_tasks=len(tasks),
        verdict="valid",
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    return submission, job


# --------------------------------------------------------------------------- #
# GET /internal/v1/work_units
# --------------------------------------------------------------------------- #
async def test_combined_worker_hides_work_units_endpoint(
    client, internal_headers, database_session, monkeypatch, tmp_path
):
    monkeypatch.setattr("agent_challenge.api.routes.settings.combined_worker", True)
    async with database_session() as session:
        await _create_job(session, agent_hash="combined-hide", tasks=_tasks(3), tmp_path=tmp_path)
        await session.commit()

    # The decentralized derivation still sees the pending units: only the API
    # route is gated, so the ``combined_worker`` OFF execution path is intact.
    async with database_session() as session:
        assert len(await list_pending_work_units(session)) == 3

    response = await client.get("/internal/v1/work_units", headers=internal_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["challenge_slug"] == "agent-challenge"
    assert body["work_units"] == []


async def test_default_mode_exposes_work_units_endpoint(
    client, internal_headers, database_session, tmp_path
):
    # combined_worker defaults to False: existing behavior is unchanged.
    async with database_session() as session:
        await _create_job(session, agent_hash="default-expose", tasks=_tasks(3), tmp_path=tmp_path)
        await session.commit()

    response = await client.get("/internal/v1/work_units", headers=internal_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["challenge_slug"] == "agent-challenge"
    assert len(body["work_units"]) == 3


# --------------------------------------------------------------------------- #
# POST /internal/v1/work_units/fold
# --------------------------------------------------------------------------- #
async def test_combined_worker_fold_is_noop(
    client, internal_headers, database_session, monkeypatch, tmp_path
):
    monkeypatch.setattr("agent_challenge.api.routes.settings.combined_worker", True)
    tasks = _tasks(1)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="combined-fold", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_id = job.job_id

    response = await client.post(
        "/internal/v1/work_units/fold",
        headers=internal_headers,
        json={"job_id": job_id, "task_id": tasks[0].task_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["posted"] is False
    assert body["finalized"] is False
    assert body["score"] == 0.0
    assert body["status"] != "failed"

    # The fold wrote nothing and did not finalize the job, so the in-process
    # worker's real result can never be clobbered.
    async with database_session() as session:
        result_count = await session.scalar(select(func.count(TaskResult.id)))
        job_row = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert result_count == 0
    assert job_row.status != "completed"


async def test_combined_worker_fold_unknown_job_is_still_noop_success(
    client, internal_headers, database_session, monkeypatch
):
    monkeypatch.setattr("agent_challenge.api.routes.settings.combined_worker", True)

    response = await client.post(
        "/internal/v1/work_units/fold",
        headers=internal_headers,
        json={"job_id": "does-not-exist", "task_id": "terminal-bench/task-0"},
    )
    assert response.status_code == 200
    assert response.json()["posted"] is False

    async with database_session() as session:
        result_count = await session.scalar(select(func.count(TaskResult.id)))
    assert result_count == 0


async def test_default_mode_fold_records_failed_result(
    client, internal_headers, database_session, tmp_path
):
    # combined_worker defaults to False: the fold records a terminal result and
    # finalizes the job as before.
    tasks = _tasks(1)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="default-fold", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_id = job.job_id

    response = await client.post(
        "/internal/v1/work_units/fold",
        headers=internal_headers,
        json={"job_id": job_id, "task_id": tasks[0].task_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["posted"] is True
    assert body["finalized"] is True

    async with database_session() as session:
        results = (await session.execute(select(TaskResult))).scalars().all()
        job_row = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert len(results) == 1
    assert results[0].stderr == "work_unit_max_attempts_exhausted"
    assert job_row.status == "completed"
