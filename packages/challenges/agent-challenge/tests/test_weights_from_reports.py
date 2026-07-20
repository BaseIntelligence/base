"""get_weights from validator-reported task results + failed-task folding.

VAL-AC-014: ``/internal/v1/get_weights`` response shape is unchanged.
VAL-AC-015: weights are derived from validator-reported, completed results.
VAL-AC-016: best score per hotkey wins.
VAL-AC-017: only valid/overridden_valid completed submissions count.
VAL-AC-018: an empty store yields an empty weights map (no crash).
VAL-AC-025: a job is not finalized/scored until every task unit is terminal.
VAL-AC-026: a terminally-failed task is folded as one non-passing task.

The validator reporting path (own-broker execution faked here) is the SOURCE of
the per-task results; ``get_weights`` reads only what was reported and finalized
into the challenge eval store, never a master-only precomputed value.
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import func, select

from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.validator_executor import (
    execute_work_unit,
    finalize_job_if_complete,
    fold_terminally_failed_work_unit,
    run_validator_cycle,
)
from agent_challenge.evaluation.weights import get_weights
from agent_challenge.evaluation.work_units import list_pending_work_units
from agent_challenge.models import AgentSubmission, EvaluationJob, TaskResult
from agent_challenge.sdk.executors import DockerRunResult


# --------------------------------------------------------------------------- #
# Faked validator-owned broker + helpers
# --------------------------------------------------------------------------- #
class FakeBrokerExecutor:
    """Emit a Terminal-Bench ``BASE_BENCHMARK_RESULT=`` payload per task."""

    def __init__(self, *, scores: dict[str, float] | None = None) -> None:
        self.runs: list[str] = []
        self.scores = dict(scores or {})

    def run(self, spec, timeout_seconds: int):
        task_id = spec.labels["base.task"]
        self.runs.append(task_id)
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
    monkeypatch.setattr(f"{base}.evaluation_task_count", 1)
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
# VAL-AC-014 + VAL-AC-018: response shape unchanged; empty store -> {}
# --------------------------------------------------------------------------- #
async def test_get_weights_shape_and_empty_store(client, internal_headers):
    response = await client.get("/internal/v1/get_weights", headers=internal_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["challenge_slug"] == "agent-challenge"
    assert body["epoch"] is None or isinstance(body["epoch"], int)
    assert isinstance(body["weights"], dict)
    assert body["weights"] == {}


# --------------------------------------------------------------------------- #
# VAL-AC-015: weights are computed from validator-reported task results
# --------------------------------------------------------------------------- #
async def test_weights_from_reported_results(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(2)
    async with database_session() as session:
        _submission, job = await _create_job(
            session,
            agent_hash="reported",
            tasks=tasks,
            tmp_path=tmp_path,
            miner_hotkey="hk-reported",
        )
        await session.commit()
        job_id = job.job_id
        job_pk = job.id

    # No reported results yet -> the hotkey is absent.
    assert await get_weights() == {}

    fake = FakeBrokerExecutor(scores={"terminal-bench/task-1": 0.0})
    summary = await run_validator_cycle(executor=fake)
    assert job_id in summary.finalized_jobs

    async with database_session() as session:
        job_row = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        results = (
            (await session.execute(select(TaskResult).where(TaskResult.job_id == job_pk)))
            .scalars()
            .all()
        )
    assert job_row.status == "completed"
    expected = sum(result.score for result in results) / len(results)

    weights = await get_weights()
    # The weight equals the aggregate of the reported per-task results.
    assert weights == {"hk-reported": expected}
    assert weights["hk-reported"] == job_row.score
    assert weights["hk-reported"] == 0.5


# --------------------------------------------------------------------------- #
# VAL-AC-016: best score per hotkey wins
# --------------------------------------------------------------------------- #
async def test_best_score_per_hotkey(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)

    # Low-scoring submission for the hotkey (task-1 fails -> 0.5).
    async with database_session() as session:
        await _create_job(
            session,
            agent_hash="best-low",
            tasks=_terminal_bench_tasks(2),
            tmp_path=tmp_path,
            miner_hotkey="hk-best",
        )
        await session.commit()
    await run_validator_cycle(executor=FakeBrokerExecutor(scores={"terminal-bench/task-1": 0.0}))

    # High-scoring submission for the SAME hotkey (both pass -> 1.0).
    async with database_session() as session:
        await _create_job(
            session,
            agent_hash="best-high",
            tasks=_terminal_bench_tasks(2),
            tmp_path=tmp_path,
            miner_hotkey="hk-best",
        )
        await session.commit()
    await run_validator_cycle(executor=FakeBrokerExecutor())

    weights = await get_weights()
    assert weights == {"hk-best": 1.0}


# --------------------------------------------------------------------------- #
# VAL-AC-017: only valid/overridden_valid completed submissions count
# --------------------------------------------------------------------------- #
async def test_only_valid_effective_status_counts(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)

    # A genuinely reported + finalized valid submission contributes.
    async with database_session() as session:
        await _create_job(
            session,
            agent_hash="incl-valid",
            tasks=_terminal_bench_tasks(1),
            tmp_path=tmp_path,
            miner_hotkey="hk-incl",
        )
        await session.commit()
    await run_validator_cycle(executor=FakeBrokerExecutor())

    # Equally reported + finalized submissions whose effective status lands on an
    # excluded value (suspicious/invalid/error) must NOT contribute a weight.
    for agent_hash, hotkey, excluded_status in (
        ("excl-susp", "hk-susp", "suspicious"),
        ("excl-invalid", "hk-invalid", "invalid"),
        ("excl-error", "hk-error", "error"),
    ):
        async with database_session() as session:
            await _create_job(
                session,
                agent_hash=agent_hash,
                tasks=_terminal_bench_tasks(1),
                tmp_path=tmp_path,
                miner_hotkey=hotkey,
            )
            await session.commit()
        await run_validator_cycle(executor=FakeBrokerExecutor())
        async with database_session() as session:
            submission = await session.scalar(
                select(AgentSubmission).where(AgentSubmission.agent_hash == agent_hash)
            )
            submission.effective_status = excluded_status
            await session.commit()

    weights = await get_weights()
    assert set(weights) == {"hk-incl"}


# --------------------------------------------------------------------------- #
# VAL-AC-025: a job is not finalized/scored until every unit is terminal
# --------------------------------------------------------------------------- #
async def test_job_not_finalized_until_all_units_terminal(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(2)
    async with database_session() as session:
        _submission, job = await _create_job(
            session,
            agent_hash="partial",
            tasks=tasks,
            tmp_path=tmp_path,
            miner_hotkey="hk-partial",
        )
        await session.commit()
        job_id = job.job_id

    async with database_session() as session:
        unit_ids = sorted(unit.work_unit_id for unit in await list_pending_work_units(session))

    # Execute only ONE of the two units.
    summary = await run_validator_cycle(work_unit_ids=unit_ids[:1], executor=FakeBrokerExecutor())
    assert summary.posted == 1
    assert summary.finalized_jobs == ()

    async with database_session() as session:
        job_row = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert job_row.status != "completed"
    # While a unit is still outstanding the hotkey does NOT appear.
    assert "hk-partial" not in await get_weights()

    # Finalizing now is a safe no-op (no hang, no partial scoring).
    async with database_session() as session:
        assert await finalize_job_if_complete(session, job_id) is None
        await session.commit()

    # Complete the remaining unit -> the job finalizes and the hotkey appears.
    summary2 = await run_validator_cycle(work_unit_ids=unit_ids[1:], executor=FakeBrokerExecutor())
    assert job_id in summary2.finalized_jobs
    assert "hk-partial" in await get_weights()


# --------------------------------------------------------------------------- #
# VAL-AC-026: a terminally-failed task is folded deterministically (once)
# --------------------------------------------------------------------------- #
async def test_terminally_failed_task_folded_once(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(2)
    async with database_session() as session:
        _submission, job = await _create_job(
            session,
            agent_hash="failfold",
            tasks=tasks,
            tmp_path=tmp_path,
            miner_hotkey="hk-fold",
        )
        await session.commit()
        job_id = job.job_id
        job_pk = job.id

    # task-0 reports completed; task-1 never reports (it permanently fails after
    # max_attempts on the coordination plane).
    async with database_session() as session:
        units = {unit.task_id: unit for unit in await list_pending_work_units(session)}
    async with database_session() as session:
        await execute_work_unit(
            session,
            units["terminal-bench/task-0"],
            executor=FakeBrokerExecutor(),
        )
        await session.commit()

    # While task-1 is outstanding the job must not finalize (no hang).
    async with database_session() as session:
        assert await finalize_job_if_complete(session, job_id) is None
        await session.commit()
    assert "hk-fold" not in await get_weights()

    # The coordination plane gives up on task-1: fold it as one non-passing task.
    async with database_session() as session:
        outcome = await fold_terminally_failed_work_unit(
            session,
            job_id=job_id,
            task_id="terminal-bench/task-1",
            reason="work_unit_max_attempts_exhausted",
        )
        await session.commit()
    assert outcome.status == "failed"
    assert outcome.score == 0.0
    assert outcome.posted is True
    assert outcome.executed is False

    # Folding again is idempotent (single non-passing row, no double count).
    async with database_session() as session:
        repeat = await fold_terminally_failed_work_unit(
            session,
            job_id=job_id,
            task_id="terminal-bench/task-1",
        )
        await session.commit()
    assert repeat.posted is False

    # Now every unit is terminal -> the job finalizes deterministically.
    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()
    assert summary is not None
    assert summary.status == "completed"
    assert summary.total_tasks == 2
    assert summary.passed_tasks == 1
    assert summary.score == 0.5

    async with database_session() as session:
        results = (
            (await session.execute(select(TaskResult).where(TaskResult.job_id == job_pk)))
            .scalars()
            .all()
        )
        result_count = await session.scalar(
            select(func.count(TaskResult.id)).where(TaskResult.job_id == job_pk)
        )
    # The failed task is counted exactly once.
    assert result_count == 2
    statuses = {result.task_id: result.status for result in results}
    assert statuses["terminal-bench/task-1"] == "failed"
    assert statuses["terminal-bench/task-0"] == "completed"

    # Downstream weight reflects the deterministic aggregate.
    assert await get_weights() == {"hk-fold": 0.5}


# --------------------------------------------------------------------------- #
# Folding never overwrites an already-reported terminal result.
# --------------------------------------------------------------------------- #
async def test_fold_does_not_overwrite_reported_result(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        _submission, job = await _create_job(
            session,
            agent_hash="fold-noop",
            tasks=tasks,
            tmp_path=tmp_path,
            miner_hotkey="hk-fold-noop",
        )
        await session.commit()
        job_id = job.job_id

    async with database_session() as session:
        unit = (await list_pending_work_units(session))[0]
    async with database_session() as session:
        await execute_work_unit(session, unit, executor=FakeBrokerExecutor())
        await session.commit()

    # The task already has a completed result; folding is a no-op.
    async with database_session() as session:
        outcome = await fold_terminally_failed_work_unit(
            session,
            job_id=job_id,
            task_id="terminal-bench/task-0",
        )
        await session.commit()
    assert outcome.status == "completed"
    assert outcome.score == 1.0
    assert outcome.posted is False
