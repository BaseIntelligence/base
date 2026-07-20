"""Validator pull / execute-via-own-broker / post-results (VAL-AC-010..013, 022, 023).

A decentralized validator pulls its assigned task subset from the coordination
plane, runs Terminal-Bench ``own_runner`` on its OWN broker (faked here), and
posts one immutable per-task result. Tasks split across validators in parallel;
per-task posting is idempotent and re-running after a crash never double-counts.
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import func, select

from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.validator_executor import (
    execute_work_unit,
    finalize_job_if_complete,
    pull_assigned_work_units,
    run_validator_cycle,
)
from agent_challenge.evaluation.work_units import (
    PendingWorkUnit,
    list_pending_work_units,
    work_unit_id_for,
)
from agent_challenge.models import AgentSubmission, EvaluationJob, TaskResult
from agent_challenge.sdk.executors import DockerRunResult


# --------------------------------------------------------------------------- #
# Faked validator-owned broker + helpers
# --------------------------------------------------------------------------- #
class FakeBrokerExecutor:
    """Stands in for the validator's OWN broker-backed DockerExecutor.

    Records each dispatched run spec and emits a Terminal-Bench
    ``BASE_BENCHMARK_RESULT=`` stdout payload the runner parses into a result.
    """

    def __init__(self, *, scores: dict[str, float] | None = None) -> None:
        self.runs: list[dict[str, object]] = []
        self.scores = dict(scores or {})

    def run(self, spec, timeout_seconds: int):
        task_id = spec.labels["base.task"]
        self.runs.append({"image": spec.image, "task": task_id, "command": spec.command})
        score = self.scores.get(task_id, 1.0)
        status = "completed" if score >= 1.0 else "failed"
        if status == "failed":
            score = 0.0
        payload = json.dumps({"score": score, "status": status})
        return DockerRunResult(
            container_name="broker-fake",
            stdout=f"BASE_BENCHMARK_RESULT={payload}",
            stderr="",
            returncode=0,
        )


def _patch_terminal_bench(monkeypatch, tmp_path) -> None:
    base = "agent_challenge.evaluation.runner.settings"
    monkeypatch.setattr(f"{base}.benchmark_backend", "terminal_bench")
    monkeypatch.setattr(f"{base}.terminal_bench_execution_backend", "own_runner")
    monkeypatch.setattr(f"{base}.evaluation_concurrency", 1)
    monkeypatch.setattr(f"{base}.docker_enabled", True)
    monkeypatch.setattr(f"{base}.docker_backend", "broker")
    monkeypatch.setattr(f"{base}.docker_broker_url", "https://broker.test")
    monkeypatch.setattr(f"{base}.docker_broker_token", "broker-token")
    monkeypatch.setattr(f"{base}.docker_broker_token_file", None)
    harbor = tmp_path / "harbor-runs"
    harbor.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(f"{base}.harbor_output_dir", str(harbor))


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


async def _create_job(
    session,
    *,
    agent_hash: str,
    tasks: list[BenchmarkTask],
    tmp_path,
    miner_hotkey: str | None = None,
) -> tuple[AgentSubmission, EvaluationJob]:
    agent_dir = tmp_path / agent_hash
    agent_dir.mkdir(parents=True, exist_ok=True)
    submission = AgentSubmission(
        miner_hotkey=miner_hotkey or f"hotkey-{agent_hash}",
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
# VAL-AC-010: execute the assigned task via the validator's OWN broker
# --------------------------------------------------------------------------- #
async def test_executes_assigned_task_via_own_broker(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        submission, job = await _create_job(
            session, agent_hash="broker-one", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()

    async with database_session() as session:
        units = await list_pending_work_units(session)
    assert len(units) == 1

    fake = FakeBrokerExecutor(scores={"terminal-bench/task-0": 1.0})
    async with database_session() as session:
        outcome = await execute_work_unit(session, units[0], executor=fake)
        await session.commit()

    # The faked broker received the run spec for the assigned task image, and the
    # run is Terminal-Bench own_runner.
    assert len(fake.runs) == 1
    assert fake.runs[0]["image"] == tasks[0].docker_image
    assert fake.runs[0]["task"] == "terminal-bench/task-0"
    assert "agent_challenge.evaluation.own_runner_backend" in fake.runs[0]["command"][-1]

    # The TaskResult is parsed from the broker BASE_BENCHMARK_RESULT= payload.
    assert outcome.executed is True
    assert outcome.posted is True
    assert outcome.status == "completed"
    assert outcome.score == 1.0
    async with database_session() as session:
        result = await session.scalar(select(TaskResult))
    assert result is not None
    assert result.task_id == "terminal-bench/task-0"
    assert 0.0 <= result.score <= 1.0
    assert result.score == 1.0


# --------------------------------------------------------------------------- #
# VAL-AC-011: per-task results are posted and persisted (one per task)
# --------------------------------------------------------------------------- #
async def test_per_task_results_persisted(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(3)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="per-task", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()

    fake = FakeBrokerExecutor(scores={"terminal-bench/task-1": 0.0})
    summary = await run_validator_cycle(executor=fake)

    assert summary.pulled == 3
    assert summary.posted == 3
    async with database_session() as session:
        results = (await session.execute(select(TaskResult))).scalars().all()
    assert len(results) == 3
    assert {r.task_id for r in results} == {t.task_id for t in tasks}
    # exactly one immutable row per (job_id, task_id) with a score in [0, 1]
    assert len({(r.job_id, r.task_id) for r in results}) == 3
    assert all(0.0 <= r.score <= 1.0 for r in results)
    statuses = {r.task_id: r.status for r in results}
    assert statuses["terminal-bench/task-1"] == "failed"
    assert statuses["terminal-bench/task-0"] == "completed"


# --------------------------------------------------------------------------- #
# VAL-AC-012: tasks split across validators (disjoint subsets, full coverage)
# --------------------------------------------------------------------------- #
async def test_tasks_split_across_validators(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(4)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="split", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()

    async with database_session() as session:
        units = await list_pending_work_units(session)
    assert len(units) == 4
    ordered_ids = sorted(unit.work_unit_id for unit in units)
    subset_a = ordered_ids[:2]
    subset_b = ordered_ids[2:]

    fake_a = FakeBrokerExecutor()
    fake_b = FakeBrokerExecutor()
    summary_a = await run_validator_cycle(work_unit_ids=subset_a, executor=fake_a)
    summary_b = await run_validator_cycle(work_unit_ids=subset_b, executor=fake_b)

    # Each validator ran ONLY its assigned subset; subsets are disjoint and their
    # union is the full selected task set.
    tasks_a = {run["task"] for run in fake_a.runs}
    tasks_b = {run["task"] for run in fake_b.runs}
    assert summary_a.posted == 2
    assert summary_b.posted == 2
    assert tasks_a.isdisjoint(tasks_b)
    assert tasks_a | tasks_b == {task.task_id for task in tasks}

    async with database_session() as session:
        results = (await session.execute(select(TaskResult))).scalars().all()
        job_row = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == job.job_id)
        )
    assert {r.task_id for r in results} == {task.task_id for task in tasks}
    assert job_row.total_tasks == 4
    assert job_row.status == "completed"
    assert job_row.passed_tasks == 4
    assert job_row.score == 1.0


# --------------------------------------------------------------------------- #
# VAL-AC-013: execution is driven by the coordination-plane pull, not a launch
# --------------------------------------------------------------------------- #
async def test_execution_driven_by_pull_not_central_launch(
    client, database_session, internal_headers, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(2)
    async with database_session() as session:
        submission, _job = await _create_job(
            session, agent_hash="pull-driven", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        submission_id = submission.id

    # The centralized launch bridge no longer exists / cannot trigger execution.
    launch = await client.post(
        f"/internal/v1/submissions/{submission_id}/launch",
        headers=internal_headers,
    )
    assert launch.status_code == 404

    # Execution still proceeds purely from the pull.
    fake = FakeBrokerExecutor()
    summary = await run_validator_cycle(executor=fake)
    assert summary.pulled == 2
    assert summary.posted == 2
    async with database_session() as session:
        result_count = await session.scalar(select(func.count(TaskResult.id)))
    assert result_count == 2


# --------------------------------------------------------------------------- #
# VAL-AC-022: re-posting a completed task is an idempotent no-op success
# --------------------------------------------------------------------------- #
async def test_repost_completed_task_is_idempotent(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(2)
    async with database_session() as session:
        submission, job = await _create_job(
            session, agent_hash="idempotent", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        submission_id = submission.id
        miner_hotkey = submission.miner_hotkey
        agent_hash = submission.agent_hash

    await run_validator_cycle(executor=FakeBrokerExecutor())

    async with database_session() as session:
        before_count = await session.scalar(select(func.count(TaskResult.id)))
        before_job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == job.job_id)
        )
        before_score = before_job.score
        before_status = before_job.status
    assert before_count == 2
    assert before_status == "completed"

    # Re-post the result for an already-completed task; the broker must NOT run
    # again and no duplicate row may appear.
    completed_unit = PendingWorkUnit(
        work_unit_id=work_unit_id_for(submission_id, "terminal-bench/task-0"),
        submission_id=submission_id,
        submission_ref=agent_hash,
        miner_hotkey=miner_hotkey,
        job_id=job.job_id,
        task_id="terminal-bench/task-0",
        docker_image=tasks[0].docker_image,
    )
    repost_executor = FakeBrokerExecutor()
    async with database_session() as session:
        outcome = await execute_work_unit(session, completed_unit, executor=repost_executor)
        await session.commit()
    assert outcome.executed is False
    assert outcome.posted is False
    assert repost_executor.runs == []

    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job.job_id)
        await session.commit()
        after_count = await session.scalar(select(func.count(TaskResult.id)))
        after_job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == job.job_id)
        )
    assert summary is not None
    assert after_count == before_count
    assert after_job.score == before_score


# --------------------------------------------------------------------------- #
# VAL-AC-023: re-running a task after a crash is safe (no double counting)
# --------------------------------------------------------------------------- #
async def test_rerun_after_crash_counts_each_task_once(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(2)
    async with database_session() as session:
        submission, job = await _create_job(
            session, agent_hash="rerun-safe", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()

    async with database_session() as session:
        units = {u.task_id: u for u in await list_pending_work_units(session)}
    unit0 = units["terminal-bench/task-0"]
    unit1 = units["terminal-bench/task-1"]

    # Validator posts task-0, then crashes before finalizing the job.
    async with database_session() as session:
        await execute_work_unit(session, unit0, executor=FakeBrokerExecutor())
        await session.commit()

    # Re-run after the crash: the same task-0 unit is re-attempted but NOT
    # re-executed (already terminal), so no second row is produced.
    rerun_executor = FakeBrokerExecutor()
    async with database_session() as session:
        rerun = await execute_work_unit(session, unit0, executor=rerun_executor)
        await session.commit()
    assert rerun.executed is False
    assert rerun.posted is False
    assert rerun_executor.runs == []

    # The recovered validator finishes the remaining task and finalizes.
    async with database_session() as session:
        await execute_work_unit(session, unit1, executor=FakeBrokerExecutor())
        await session.commit()
    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job.job_id)
        await session.commit()

    assert summary is not None
    async with database_session() as session:
        results = (await session.execute(select(TaskResult))).scalars().all()
        job_row = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.job_id == job.job_id)
        )
    # Exactly one result per selected task; job aggregates reflect the unique set.
    assert len(results) == 2
    assert len({(r.job_id, r.task_id) for r in results}) == 2
    assert job_row.total_tasks == 2
    assert job_row.passed_tasks == 2
    assert job_row.score == 1.0


# --------------------------------------------------------------------------- #
# Re-run safety at the persistence seam: a duplicate post dedupes to one row.
# --------------------------------------------------------------------------- #
async def test_double_execution_dedupes_to_single_result(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="dedupe", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()

    async with database_session() as session:
        unit = (await list_pending_work_units(session))[0]

    async with database_session() as session:
        first = await execute_work_unit(session, unit, executor=FakeBrokerExecutor())
        await session.commit()
    assert first.posted is True

    async with database_session() as session:
        second = await execute_work_unit(session, unit, executor=FakeBrokerExecutor())
        await session.commit()
    assert second.posted is False

    async with database_session() as session:
        result_count = await session.scalar(
            select(func.count(TaskResult.id)).where(TaskResult.job_id == job.id)
        )
    assert result_count == 1


# --------------------------------------------------------------------------- #
# Pull semantics: only the assigned subset is returned.
# --------------------------------------------------------------------------- #
async def test_pull_returns_only_assigned_subset(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(3)
    async with database_session() as session:
        submission, _job = await _create_job(
            session, agent_hash="subset", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        assigned = work_unit_id_for(submission.id, "terminal-bench/task-1")

    async with database_session() as session:
        pulled = await pull_assigned_work_units(session, work_unit_ids=[assigned])
    assert [unit.work_unit_id for unit in pulled] == [assigned]
