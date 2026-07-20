"""M6 cross-feature wiring on the agent-challenge side.

Two guarantees the master-orchestration-driver depends on:

1. The challenge-side fold route (``POST /internal/v1/work_units/fold``) records
   a permanently-failed (max_attempts) work unit once and finalizes its
   EvaluationJob, so a permanently-failed task never hangs its job forever (the
   fold logic gains its production caller here).
2. The production validator cycle (``run_assigned_validator_cycle``) ALWAYS
   builds a ``GatewayExecutionConfig`` from the assignment payload and never
   dispatches an eval run with ``gateway=None`` - so no raw miner ``*_API_KEY``
   reaches the eval container and DeepSeek always routes through the master
   gateway (VAL-AC-019 end-to-end).
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy import select

from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.gateway import GatewayExecutionConfig
from agent_challenge.evaluation.validator_executor import (
    AssignedWorkUnit,
    run_assigned_validator_cycle,
)
from agent_challenge.evaluation.work_units import list_pending_work_units, work_unit_id_for
from agent_challenge.models import AgentSubmission, EvaluationJob, TaskResult
from agent_challenge.sdk.executors import DockerRunResult


# --------------------------------------------------------------------------- #
# Faked validator-owned broker recording every dispatched run spec.
# --------------------------------------------------------------------------- #
class RecordingBrokerExecutor:
    def __init__(self, *, score: float = 1.0) -> None:
        self.specs: list[object] = []
        self.score = score

    def run(self, spec, timeout_seconds: int):
        self.specs.append(spec)
        status = "completed" if self.score >= 1.0 else "failed"
        payload = json.dumps({"score": self.score, "status": status})
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
    session, *, agent_hash, tasks, tmp_path, miner_hotkey=None
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
# Fold route: permanently-failed work unit finalizes its job
# --------------------------------------------------------------------------- #
async def test_fold_route_finalizes_job_when_last_task_is_dead(
    client, internal_headers, database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _tasks(2)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="fold-job", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_id = job.job_id
        job_pk = job.id

    # One task gets a real reported result; the other is permanently failed.
    submission_id = await _submission_id_for(database_session, job_id)
    run_unit_id = work_unit_id_for(submission_id, tasks[0].task_id)
    await run_assigned_validator_cycle(
        [AssignedWorkUnit(work_unit_id=run_unit_id, payload={"gateway_token": "tok"})],
        gateway_base_url="https://master-gateway.test",
        executor=RecordingBrokerExecutor(),
    )

    # The job is NOT finalized while task-1 is still outstanding.
    async with database_session() as session:
        job_row = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert job_row.status != "completed"

    # The master folds the dead task-1 -> the job now finalizes.
    response = await client.post(
        "/internal/v1/work_units/fold",
        headers=internal_headers,
        json={"job_id": job_id, "task_id": tasks[1].task_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["score"] == 0.0
    assert body["posted"] is True
    assert body["finalized"] is True

    async with database_session() as session:
        job_row = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        results = (
            (await session.execute(select(TaskResult).where(TaskResult.job_id == job_pk)))
            .scalars()
            .all()
        )
    assert job_row.status == "completed"
    # Each task counted once: one pass (1.0) + one fold (0.0) -> score 0.5.
    assert job_row.score == 0.5
    assert len(results) == 2


async def test_fold_route_is_idempotent(client, internal_headers, database_session, tmp_path):
    tasks = _tasks(1)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="fold-idem", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_id = job.job_id

    first = await client.post(
        "/internal/v1/work_units/fold",
        headers=internal_headers,
        json={"job_id": job_id, "task_id": tasks[0].task_id, "reason": "exhausted"},
    )
    assert first.status_code == 200
    assert first.json()["posted"] is True

    second = await client.post(
        "/internal/v1/work_units/fold",
        headers=internal_headers,
        json={"job_id": job_id, "task_id": tasks[0].task_id},
    )
    assert second.status_code == 200
    # A second fold does not create a duplicate result row.
    assert second.json()["posted"] is False

    async with database_session() as session:
        results = (await session.execute(select(TaskResult))).scalars().all()
    assert len(results) == 1


async def test_fold_route_requires_internal_token(client):
    response = await client.post(
        "/internal/v1/work_units/fold",
        json={"job_id": "missing", "task_id": "t"},
    )
    assert response.status_code in (401, 403)


async def test_fold_route_unknown_job_returns_404(client, internal_headers):
    response = await client.post(
        "/internal/v1/work_units/fold",
        headers=internal_headers,
        json={"job_id": "does-not-exist", "task_id": "t"},
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# VAL-ACAT-013: production cycle is Base-gateway-free end-to-end
# --------------------------------------------------------------------------- #
async def test_production_cycle_dispatches_gateway_none(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _tasks(1)
    async with database_session() as session:
        await _create_job(session, agent_hash="gw-always", tasks=tasks, tmp_path=tmp_path)
        await session.commit()

    async with database_session() as session:
        units = await list_pending_work_units(session)
    assert len(units) == 1

    captured: list[object] = []

    real_run_validator_cycle = None

    async def _capturing_cycle(
        *, work_unit_ids=None, executor=None, gateway=None, attestation_gate=None
    ):
        captured.append(gateway)
        # VAL-ACAT-013: production always passes gateway=None.
        assert gateway is None
        return await real_run_validator_cycle(
            work_unit_ids=work_unit_ids,
            executor=executor,
            gateway=gateway,
            attestation_gate=attestation_gate,
        )

    from agent_challenge.evaluation import validator_executor as ve

    real_run_validator_cycle = ve.run_validator_cycle
    monkeypatch.setattr(ve, "run_validator_cycle", _capturing_cycle)

    fake = RecordingBrokerExecutor()
    summary = await run_assigned_validator_cycle(
        [
            AssignedWorkUnit(
                work_unit_id=units[0].work_unit_id,
                payload={"gateway_token": "scoped-assignment-token"},
            )
        ],
        gateway_base_url="https://master-gateway.test",
        executor=fake,
    )

    assert summary.posted == 1
    assert captured and all(g is None for g in captured)
    env = fake.specs[0].env
    assert "BASE_LLM_GATEWAY_URL" not in env
    assert "BASE_GATEWAY_TOKEN" not in env
    assert "DEEPSEEK_API_KEY" not in env


async def test_production_cycle_runs_without_gateway_token_payload(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = _tasks(1)
    async with database_session() as session:
        await _create_job(session, agent_hash="gw-missing", tasks=tasks, tmp_path=tmp_path)
        await session.commit()
    async with database_session() as session:
        units = await list_pending_work_units(session)

    fake = RecordingBrokerExecutor()
    # VAL-ACAT-013: missing gateway token is legal tools-only.
    summary = await run_assigned_validator_cycle(
        [AssignedWorkUnit(work_unit_id=units[0].work_unit_id, payload={})],
        gateway_base_url="https://master-gateway.test",
        executor=fake,
    )
    assert summary.posted == 1
    assert "BASE_GATEWAY_TOKEN" not in fake.specs[0].env


def test_gateway_from_payload_builds_config_residual_only():
    """Residual helper still parses payloads; production cycle does not use it."""

    gateway = GatewayExecutionConfig.from_assignment_payload(
        {"gateway_token": "tok"},
        base_url="https://gw.test",
    )
    assert gateway.token == "tok"
    assert gateway.base_url == "https://gw.test"
    assert gateway.llm_gateway_url == "https://gw.test/llm/v1"


async def _submission_id_for(database_session, job_id: str) -> int:
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
        return job.submission_id
