"""Agent DeepSeek via the master gateway; no master-only secrets on validators.

Covers VAL-AC-019..021: during decentralized task execution the agent's DeepSeek
calls are routed at the master LLM gateway (no raw provider key on the
validator), execution no longer requires a master-only Fernet env-decryption key
to obtain LLM credentials, and miner-env values plus the scoped gateway token are
redacted from persisted result stdout/stderr.
"""

from __future__ import annotations

import json
import uuid

from cryptography.fernet import Fernet
from sqlalchemy import select

from agent_challenge.evaluation import runner
from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.gateway import GatewayExecutionConfig
from agent_challenge.evaluation.validator_executor import execute_work_unit
from agent_challenge.evaluation.work_units import list_pending_work_units
from agent_challenge.models import AgentSubmission, EvaluationJob, SubmissionEnvVar, TaskResult
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.sdk.executors import DockerRunResult


# --------------------------------------------------------------------------- #
# Faked validator-owned broker that records the dispatched run spec and can echo
# secrets into stdout/stderr to exercise redaction.
# --------------------------------------------------------------------------- #
class RecordingBrokerExecutor:
    def __init__(self, *, leak: tuple[str, ...] = ()) -> None:
        self.specs: list[object] = []
        self.leak = leak

    def run(self, spec, timeout_seconds: int):
        self.specs.append(spec)
        payload = json.dumps({"score": 1.0, "status": "completed"})
        leak_text = " ".join(self.leak)
        return DockerRunResult(
            container_name="broker-fake",
            stdout=f"agent log: {leak_text}\nBASE_BENCHMARK_RESULT={payload}",
            stderr=f"diagnostic: {leak_text}",
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


def _terminal_bench_task(index: int = 0) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=f"terminal-bench/task-{index}",
        docker_image=f"ghcr.io/baseintelligence/terminal-bench-runner:{index}",
        prompt=f"task {index}",
        benchmark="terminal_bench",
        metadata={"task_id": f"terminal-bench/task-{index}"},
    )


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


async def _add_locked_env_var(
    session,
    submission: AgentSubmission,
    *,
    key: str,
    value: str,
    enc_settings: ChallengeSettings,
) -> None:
    env_var = SubmissionEnvVar.encrypted(
        submission_id=submission.id,
        key=key,
        value=value,
        settings=enc_settings,
    )
    env_var.locked_at = submission.created_at
    session.add(env_var)
    submission.env_locked_at = submission.created_at


def test_gateway_agent_env_composition_is_legacy_shape_only():
    """Residual GatewayExecutionConfig still has helper shape, but product does not inject it."""

    gateway = GatewayExecutionConfig(
        base_url="https://master-gateway.test/",
        token="tok",
    )
    # Helper remains for residual library compatibility; VAL-ACAT-013 ensures
    # runner / own_runner never apply agent_env() into production containers.
    env = gateway.agent_env()
    assert env["BASE_LLM_GATEWAY_URL"].endswith("/llm/v1")
    assert env["BASE_GATEWAY_TOKEN"] == "tok"
    assert "LLM_MODEL" not in env
    assert "DEEPSEEK_API_KEY" not in env


# --------------------------------------------------------------------------- #
# VAL-AC-019: agent LLM calls route through the master gateway
# --------------------------------------------------------------------------- #
async def test_deepseek_no_longer_routes_through_master_gateway(
    database_session, monkeypatch, tmp_path
):
    """VAL-ACAT-013: residual gateway parameter must not inject BASE_* agent env."""

    _patch_terminal_bench(monkeypatch, tmp_path)
    tasks = [_terminal_bench_task(0)]
    async with database_session() as session:
        await _create_job(session, agent_hash="gw-deepseek", tasks=tasks, tmp_path=tmp_path)
        await session.commit()

    async with database_session() as session:
        units = await list_pending_work_units(session)
    assert len(units) == 1

    gateway = GatewayExecutionConfig(
        base_url="https://master-gateway.test",
        token="scoped-assignment-token",
    )
    fake = RecordingBrokerExecutor()
    async with database_session() as session:
        outcome = await execute_work_unit(session, units[0], executor=fake, gateway=gateway)
        await session.commit()

    assert outcome.status == "completed"
    assert len(fake.specs) == 1
    env = fake.specs[0].env
    assert "BASE_LLM_GATEWAY_URL" not in env
    assert "BASE_GATEWAY_TOKEN" not in env
    assert "DEEPSEEK_API_KEY" not in env
    assert "LLM_MODEL" not in env
    serialized = json.dumps(env, sort_keys=True)
    assert "api.deepseek.com" not in serialized


# --------------------------------------------------------------------------- #
# VAL-AC-019: a miner-supplied raw provider key never reaches the eval container
# --------------------------------------------------------------------------- #
async def test_raw_provider_key_is_stripped_when_routing_via_gateway(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    key_file = tmp_path / "env-key"
    key_file.write_text(Fernet.generate_key().decode("ascii"), encoding="utf-8")
    enc_settings = ChallengeSettings(submission_env_encryption_key_file=str(key_file))
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.settings.submission_env_encryption_key_file",
        str(key_file),
    )

    tasks = [_terminal_bench_task(0)]
    async with database_session() as session:
        submission, _job = await _create_job(
            session, agent_hash="gw-strip-key", tasks=tasks, tmp_path=tmp_path
        )
        await _add_locked_env_var(
            session,
            submission,
            key="DEEPSEEK_API_KEY",
            value="sk-raw-miner-provider-secret",
            enc_settings=enc_settings,
        )
        await session.commit()

    async with database_session() as session:
        units = await list_pending_work_units(session)

    gateway = GatewayExecutionConfig(base_url="https://master-gateway.test", token="tok-abc")
    fake = RecordingBrokerExecutor()
    async with database_session() as session:
        await execute_work_unit(session, units[0], executor=fake, gateway=gateway)
        await session.commit()

    env = fake.specs[0].env
    assert "DEEPSEEK_API_KEY" not in env
    assert "sk-raw-miner-provider-secret" not in json.dumps(env, sort_keys=True)
    assert "BASE_LLM_GATEWAY_URL" not in env
    assert "BASE_GATEWAY_TOKEN" not in env


# --------------------------------------------------------------------------- #
# VAL-AC-020: no master-only env-decryption is required for LLM creds
# --------------------------------------------------------------------------- #
async def test_execution_without_env_decryption_key(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    # Encrypt + lock a miner env var with a key that the EXECUTOR will NOT have.
    key_file = tmp_path / "env-key"
    key_file.write_text(Fernet.generate_key().decode("ascii"), encoding="utf-8")
    enc_settings = ChallengeSettings(submission_env_encryption_key_file=str(key_file))

    tasks = [_terminal_bench_task(0)]
    async with database_session() as session:
        submission, _job = await _create_job(
            session, agent_hash="no-key", tasks=tasks, tmp_path=tmp_path
        )
        await _add_locked_env_var(
            session,
            submission,
            key="TASK_SECRET",
            value="locked-miner-secret",
            enc_settings=enc_settings,
        )
        await session.commit()

    # The validator/executor has NO submission-env encryption key configured.
    assert runner.settings.submission_env_encryption_key_file is None

    async with database_session() as session:
        units = await list_pending_work_units(session)
    assert len(units) == 1

    gateway = GatewayExecutionConfig(base_url="https://master-gateway.test", token="tok-xyz")
    fake = RecordingBrokerExecutor()
    async with database_session() as session:
        outcome = await execute_work_unit(session, units[0], executor=fake, gateway=gateway)
        await session.commit()

    # Execution completes to a TaskResult despite the missing decryption key.
    assert outcome.executed is True
    assert outcome.posted is True
    assert outcome.status == "completed"
    async with database_session() as session:
        result = await session.scalar(select(TaskResult))
    assert result is not None
    assert result.status == "completed"
    # The undecryptable value is not forwarded into the eval container.
    assert "locked-miner-secret" not in json.dumps(fake.specs[0].env, sort_keys=True)


# --------------------------------------------------------------------------- #
# VAL-AC-021: miner-env values and the gateway token are redacted in results
# --------------------------------------------------------------------------- #
async def test_miner_env_and_gateway_token_redacted_in_results(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    key_file = tmp_path / "env-key"
    key_file.write_text(Fernet.generate_key().decode("ascii"), encoding="utf-8")
    enc_settings = ChallengeSettings(submission_env_encryption_key_file=str(key_file))
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.settings.submission_env_encryption_key_file",
        str(key_file),
    )

    miner_secret = "super-secret-miner-value"
    gateway_token = "scoped-gateway-token-7f3a"
    tasks = [_terminal_bench_task(0)]
    async with database_session() as session:
        submission, _job = await _create_job(
            session, agent_hash="redact", tasks=tasks, tmp_path=tmp_path
        )
        await _add_locked_env_var(
            session,
            submission,
            key="MINER_SECRET",
            value=miner_secret,
            enc_settings=enc_settings,
        )
        await session.commit()

    async with database_session() as session:
        units = await list_pending_work_units(session)

    gateway = GatewayExecutionConfig(base_url="https://master-gateway.test", token=gateway_token)
    # The agent echoes both the miner secret and the scoped token into its logs.
    fake = RecordingBrokerExecutor(leak=(miner_secret, gateway_token))
    async with database_session() as session:
        await execute_work_unit(session, units[0], executor=fake, gateway=gateway)
        await session.commit()

    async with database_session() as session:
        result = await session.scalar(select(TaskResult))
    assert result is not None
    persisted = json.dumps({"stdout": result.stdout, "stderr": result.stderr}, sort_keys=True)
    assert "[REDACTED_MINER_ENV]" in persisted
    assert miner_secret not in persisted
    assert gateway_token not in persisted
