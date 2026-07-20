"""Agent-challenge hardening: timed_out terminal trap + shared status constants.

Covers the optional M4 hardening feature (``m-misc-agent-challenge-hardening``):

(1) ``timed_out`` is a TERMINAL, non-passing task status. A timed-out task work
    unit must NOT re-dispatch every cycle, must NOT discard a later result, and
    must NOT block :func:`finalize_job_if_complete`.

(2) Task / job / submission status literals come from ONE shared source
    (:mod:`agent_challenge.core.statuses`) used across ``work_units``,
    ``runner``, ``validator_executor``, ``reconciler``, and the submission
    ``state_machine`` (so a rename cannot silently drift the cross-module sets).
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import func, select

from agent_challenge.core import statuses
from agent_challenge.core.statuses import (
    JobStatus,
    SubmissionStatus,
    TaskStatus,
)
from agent_challenge.evaluation import runner, validator_executor, work_units
from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.validator_executor import (
    execute_work_unit,
    finalize_job_if_complete,
)
from agent_challenge.evaluation.work_units import list_pending_work_units
from agent_challenge.models import AgentSubmission, EvaluationJob, TaskResult
from agent_challenge.sdk.executors import DockerRunResult
from agent_challenge.submissions import state_machine


class FakeBrokerExecutor:
    """Validator-owned broker stand-in that can also simulate a timeout."""

    def __init__(
        self,
        *,
        scores: dict[str, float] | None = None,
        timed_out_tasks: set[str] | None = None,
    ) -> None:
        self.runs: list[dict[str, object]] = []
        self.scores = dict(scores or {})
        self.timed_out_tasks = set(timed_out_tasks or set())

    def run(self, spec, timeout_seconds: int):
        task_id = spec.labels["base.task"]
        self.runs.append({"image": spec.image, "task": task_id})
        if task_id in self.timed_out_tasks:
            return DockerRunResult(
                container_name="broker-fake",
                stdout="",
                stderr="",
                returncode=124,
                timed_out=True,
            )
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
# (1) timed_out is terminal & non-passing
# --------------------------------------------------------------------------- #
def test_timed_out_is_terminal_task_status_but_not_terminal_job_status():
    assert "timed_out" in work_units.TERMINAL_TASK_STATUSES
    assert TaskStatus.TIMED_OUT in statuses.TERMINAL_TASK_STATUSES
    # A timed-out result must NOT be treated as a terminal job status (a job
    # never carries timed_out); keeping the sets distinct preserves job logic.
    assert "timed_out" not in runner.TERMINAL_JOB_STATUSES
    assert "timed_out" not in statuses.TERMINAL_JOB_STATUSES


async def test_timed_out_task_does_not_redispatch_forever(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        _submission, _job = await _create_job(
            session, agent_hash="timeout-redispatch", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()

    async with database_session() as session:
        units = await list_pending_work_units(session)
    assert len(units) == 1

    fake = FakeBrokerExecutor(timed_out_tasks={"terminal-bench/task-0"})
    async with database_session() as session:
        outcome = await execute_work_unit(session, units[0], executor=fake)
        await session.commit()
    assert outcome.status == "timed_out"
    assert outcome.score == 0.0
    assert outcome.posted is True

    # The timed-out task now has a terminal result, so it is no longer a pending
    # work unit -> it is NOT re-dispatched on the next pull.
    async with database_session() as session:
        remaining = await list_pending_work_units(session)
    assert remaining == []

    # A second execute is an idempotent no-op (broker not re-run, no new row).
    rerun = FakeBrokerExecutor()
    async with database_session() as session:
        repeat = await execute_work_unit(session, units[0], executor=rerun)
        await session.commit()
    assert repeat.executed is False
    assert repeat.posted is False
    assert rerun.runs == []
    async with database_session() as session:
        count = await session.scalar(select(func.count(TaskResult.id)))
    assert count == 1


async def test_timed_out_task_does_not_block_finalize(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _terminal_bench_tasks(2)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="timeout-finalize", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()

    async with database_session() as session:
        units = {u.task_id: u for u in await list_pending_work_units(session)}

    fake = FakeBrokerExecutor(timed_out_tasks={"terminal-bench/task-0"})
    async with database_session() as session:
        await execute_work_unit(session, units["terminal-bench/task-0"], executor=fake)
        await session.commit()
    async with database_session() as session:
        await execute_work_unit(session, units["terminal-bench/task-1"], executor=fake)
        await session.commit()

    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job.job_id)
        await session.commit()

    assert summary is not None
    assert summary.status == "completed"
    # Timed-out task counted exactly once and as NON-passing (score 0).
    assert summary.total_tasks == 2
    assert summary.passed_tasks == 1
    assert summary.score == 0.5
    async with database_session() as session:
        rows = (await session.execute(select(TaskResult))).scalars().all()
    statuses_by_task = {r.task_id: r.status for r in rows}
    assert statuses_by_task["terminal-bench/task-0"] == "timed_out"
    assert statuses_by_task["terminal-bench/task-1"] == "completed"


# --------------------------------------------------------------------------- #
# (2) single shared source of truth for status literals
# --------------------------------------------------------------------------- #
def test_status_sets_share_one_source_of_truth():
    # work_units / runner / validator_executor all reference the SAME objects
    # defined in core.statuses (not hand-copied frozensets).
    assert work_units.TERMINAL_TASK_STATUSES is statuses.TERMINAL_TASK_STATUSES
    assert work_units.ASSIGNABLE_JOB_STATUSES is statuses.ASSIGNABLE_JOB_STATUSES
    assert work_units.HALTED_SUBMISSION_STATUSES is statuses.HALTED_SUBMISSION_STATUSES
    assert runner.TERMINAL_JOB_STATUSES is statuses.TERMINAL_JOB_STATUSES
    assert validator_executor.TERMINAL_TASK_STATUSES is statuses.TERMINAL_TASK_STATUSES
    assert validator_executor.TERMINAL_JOB_STATUSES is statuses.TERMINAL_JOB_STATUSES


def test_state_machine_status_membership_from_shared_source():
    assert state_machine.INTERNAL_STATUSES is statuses.INTERNAL_SUBMISSION_STATUSES
    assert state_machine.LEGACY_STATUSES is statuses.LEGACY_SUBMISSION_STATUSES
    # Every submission status the state machine validates is a SubmissionStatus
    # enum member (the single canonical vocabulary).
    union = statuses.INTERNAL_SUBMISSION_STATUSES | statuses.LEGACY_SUBMISSION_STATUSES
    assert {str(member) for member in union} == set(SubmissionStatus)


def test_enum_values_match_persisted_literals():
    assert JobStatus.COMPLETED == "completed"
    assert JobStatus.QUEUED == "queued"
    assert TaskStatus.TIMED_OUT == "timed_out"
    assert SubmissionStatus.TB_FAILED_FINAL == "tb_failed_final"
    # Halted set is exactly the cross-module work-suppression vocabulary.
    assert {str(s) for s in statuses.HALTED_SUBMISSION_STATUSES} == {
        "analysis_rejected",
        "analysis_escalated",
        "admin_paused",
        "cancelled",
        "review_rejected",
        "review_escalated",
        "review_expired",
        "review_cancelled",
        "review_error",
        "tb_failed_final",
    }
