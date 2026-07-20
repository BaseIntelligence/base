"""Combined-worker / own_runner: Base LLM gateway injection removed (VAL-ACAT-013).

Historical VAL-LLM-CODE-011 wired master ``BASE_LLM_GATEWAY_URL`` + token into
the agent sandbox. Attestation-only track forbids that path. These tests pin:

* residual Settings gateway fields never inject Base gateway agent env;
* residual process ``BASE_*`` gateway vars are stripped from agent env;
* tools-only (no gateway env) remains legal;
* Main never fails solely because Base gateway tokens are missing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent_challenge.evaluation import own_runner_backend, runner
from agent_challenge.evaluation.benchmarks import BenchmarkTask
from agent_challenge.evaluation.gateway import (
    agent_gateway_config_from_settings,
)
from agent_challenge.evaluation.own_runner.result_schema import build_benchmark_result
from agent_challenge.models import AgentSubmission, EvaluationJob
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.sdk.executors import DockerRunResult

GATEWAY_BASE_URL = "https://master-gateway.test"
AGENT_TOKEN = "agent-scoped-token"
ANALYZER_TOKEN = "analyzer-central-gate-token"
GATEWAY_ENV_KEYS = ("CHALLENGE_LLM_GATEWAY_BASE_URL", "CHALLENGE_AGENT_GATEWAY_TOKEN")


def _clear_gateway_env(monkeypatch) -> None:
    for name in (
        "CHALLENGE_LLM_GATEWAY_BASE_URL",
        "CHALLENGE_AGENT_GATEWAY_TOKEN",
        "CHALLENGE_AGENT_GATEWAY_TOKEN_FILE",
        "CHALLENGE_LLM_GATEWAY_TOKEN",
        "CHALLENGE_LLM_GATEWAY_TOKEN_FILE",
        # Also clear the agent-facing BASE_* env the broker injects into the
        # runner container, so a truly ungatewayed environment yields None.
        "BASE_LLM_GATEWAY_URL",
        "BASE_GATEWAY_TOKEN",
        "LLM_COST_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# config helper (agent_gateway_config_from_settings)
# --------------------------------------------------------------------------- #
def test_agent_gateway_config_never_injects_residual_settings() -> None:
    """VAL-ACAT-013: residual Settings gateway bags never produce agent gateway env."""

    settings = ChallengeSettings(
        llm_gateway_base_url=GATEWAY_BASE_URL,
        agent_gateway_token=AGENT_TOKEN,
        llm_gateway_token=ANALYZER_TOKEN,
    )
    assert agent_gateway_config_from_settings(settings) is None


def test_agent_gateway_token_file_still_loads_but_does_not_inject(tmp_path) -> None:
    token_file = tmp_path / "agent-gateway-token"
    token_file.write_text("file-backed-agent-token\n", encoding="utf-8")

    settings = ChallengeSettings(
        llm_gateway_base_url=GATEWAY_BASE_URL,
        agent_gateway_token_file=str(token_file),
    )

    assert settings.agent_gateway_token == "file-backed-agent-token"
    assert agent_gateway_config_from_settings(settings) is None
    assert "file-backed-agent-token" not in str(settings.safe_model_dump())
    assert "file-backed-agent-token" not in repr(settings)


def test_agent_gateway_config_backcompat_when_no_gateway(monkeypatch) -> None:
    _clear_gateway_env(monkeypatch)
    settings = ChallengeSettings()
    assert settings.llm_gateway_base_url is None
    assert agent_gateway_config_from_settings(settings) is None


# --------------------------------------------------------------------------- #
# own_runner backend main() (the combined-worker in-container entry point)
# --------------------------------------------------------------------------- #
def _capture_run_own_runner_job(monkeypatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    async def fake_run(**kwargs):
        captured["agent_env"] = kwargs.get("agent_env")
        return SimpleNamespace(
            benchmark_result=build_benchmark_result(
                status="completed",
                score=1.0,
                resolved=1,
                total=1,
                reason_code=None,
            )
        )

    monkeypatch.setattr(own_runner_backend, "run_own_runner_job", fake_run)
    return captured


def test_own_runner_main_does_not_inject_base_gateway_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CHALLENGE_LLM_GATEWAY_BASE_URL", GATEWAY_BASE_URL)
    monkeypatch.setenv("CHALLENGE_AGENT_GATEWAY_TOKEN", AGENT_TOKEN)
    monkeypatch.setenv("CHALLENGE_LLM_GATEWAY_TOKEN", ANALYZER_TOKEN)
    captured = _capture_run_own_runner_job(monkeypatch)

    rc = own_runner_backend.main(["run", "--task", "task-1", "--job-dir", str(tmp_path / "job")])

    assert rc == 0
    agent_env = captured.get("agent_env") or {}
    assert "BASE_LLM_GATEWAY_URL" not in agent_env
    assert "BASE_GATEWAY_TOKEN" not in agent_env
    assert ANALYZER_TOKEN not in json.dumps(agent_env)


def test_own_runner_main_runs_tools_only_without_gateway_token(monkeypatch, tmp_path) -> None:
    """VAL-ACAT-013: missing Base gateway tokens must not block tools-only eval."""

    monkeypatch.delenv("CHALLENGE_LLM_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("CHALLENGE_AGENT_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("CHALLENGE_AGENT_GATEWAY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("BASE_LLM_GATEWAY_URL", raising=False)
    monkeypatch.delenv("BASE_GATEWAY_TOKEN", raising=False)
    captured = _capture_run_own_runner_job(monkeypatch)

    rc = own_runner_backend.main(["run", "--task", "task-1", "--job-dir", str(tmp_path / "job")])

    assert rc == 0
    agent_env = captured.get("agent_env")
    assert agent_env in (None, {})


def test_own_runner_main_backcompat_no_gateway_env(monkeypatch, tmp_path) -> None:
    # With NEITHER the CHALLENGE_* settings NOR the injected BASE_* env present,
    # the resolver yields no gateway env (back-compat).
    _clear_gateway_env(monkeypatch)
    captured = _capture_run_own_runner_job(monkeypatch)

    rc = own_runner_backend.main(["run", "--task", "task-1", "--job-dir", str(tmp_path / "job")])

    assert rc == 0
    assert captured["agent_env"] is None


def test_resolve_agent_gateway_env_strips_residual_base_gateway(monkeypatch) -> None:
    """VAL-ACAT-013: residual BASE_* gateway env must not re-enter agent sandbox."""

    _clear_gateway_env(monkeypatch)
    monkeypatch.setenv("BASE_LLM_GATEWAY_URL", f"{GATEWAY_BASE_URL}/llm/v1")
    monkeypatch.setenv("BASE_GATEWAY_TOKEN", AGENT_TOKEN)
    monkeypatch.setenv("LLM_COST_LIMIT", "12.5")

    assert own_runner_backend._resolve_agent_gateway_env() == {
        "LLM_COST_LIMIT": "12.5",
    }


def test_resolve_agent_gateway_env_without_token_is_tools_only(monkeypatch) -> None:
    # Missing Base gateway tokens is legal (tools-only); do not fail closed.
    _clear_gateway_env(monkeypatch)
    monkeypatch.setenv("BASE_LLM_GATEWAY_URL", f"{GATEWAY_BASE_URL}/llm/v1")
    monkeypatch.delenv("BASE_GATEWAY_TOKEN", raising=False)

    assert own_runner_backend._resolve_agent_gateway_env() is None


# --------------------------------------------------------------------------- #
# durable eval path (_run_terminal_bench_task_durable) container env wiring
# --------------------------------------------------------------------------- #
class _RecordingTerminalBenchExecutor:
    def __init__(self, *, leak: tuple[str, ...] = ()) -> None:
        self.specs: list[object] = []
        self.leak = leak

    def run(self, spec, timeout_seconds: int) -> DockerRunResult:
        self.specs.append(spec)
        leak_text = " ".join(self.leak)
        payload = json.dumps({"score": 1.0, "status": "completed"})
        return DockerRunResult(
            container_name="fake",
            stdout=f"agent log: {leak_text}\nBASE_BENCHMARK_RESULT={payload}",
            stderr=f"diagnostic: {leak_text}",
            returncode=0,
        )

    @property
    def spec(self):
        return self.specs[-1] if self.specs else None


def _configure_durable_gateway(monkeypatch, tmp_path, *, base_url, agent_token) -> None:
    harbor = tmp_path / "harbor-runs"
    harbor.mkdir(parents=True, exist_ok=True)
    for module in ("runner", "terminal_bench"):
        base = f"agent_challenge.evaluation.{module}.settings"
        monkeypatch.setattr(f"{base}.artifact_root", str(tmp_path))
        monkeypatch.setattr(f"{base}.harbor_output_dir", str(harbor))
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.docker_backend", "cli")
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.llm_gateway_base_url", base_url)
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.settings.agent_gateway_token", agent_token
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.settings.llm_gateway_token", ANALYZER_TOKEN
    )


async def _durable_submission_and_job(session, tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(exist_ok=True)
    submission = AgentSubmission(
        miner_hotkey="hotkey-durable-gw",
        name="agent-durable-gw",
        agent_hash="durable-gw-hash",
        artifact_uri=str(agent_dir),
        raw_status="tb_running",
        effective_status="evaluating",
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id="job-durable-gw",
        submission_id=submission.id,
        status="running",
        selected_tasks_json="[]",
    )
    session.add(job)
    await session.flush()
    return submission, job


def _terminal_bench_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="task-durable-gw",
        docker_image="ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
        benchmark="terminal_bench",
    )


async def test_durable_path_never_injects_base_gateway_env(database_session, monkeypatch, tmp_path):
    _configure_durable_gateway(
        monkeypatch, tmp_path, base_url=GATEWAY_BASE_URL, agent_token=AGENT_TOKEN
    )
    executor = _RecordingTerminalBenchExecutor()
    async with database_session() as session:
        submission, job = await _durable_submission_and_job(session, tmp_path)
        result = await runner._run_terminal_bench_task_durable(
            session, executor, submission, job, _terminal_bench_task()
        )

    assert result.status == "completed"
    env = executor.spec.env
    assert "BASE_LLM_GATEWAY_URL" not in env
    assert "BASE_GATEWAY_TOKEN" not in env
    assert ANALYZER_TOKEN not in json.dumps(env, sort_keys=True)


async def test_durable_path_runs_without_agent_gateway_token(
    database_session, monkeypatch, tmp_path
):
    """VAL-ACAT-013: residual gateway URL without token is not a hard fail."""

    _configure_durable_gateway(monkeypatch, tmp_path, base_url=GATEWAY_BASE_URL, agent_token=None)
    executor = _RecordingTerminalBenchExecutor()
    async with database_session() as session:
        submission, job = await _durable_submission_and_job(session, tmp_path)
        result = await runner._run_terminal_bench_task_durable(
            session, executor, submission, job, _terminal_bench_task()
        )
    assert result.status == "completed"
    assert "BASE_GATEWAY_TOKEN" not in executor.spec.env


async def test_durable_path_backcompat_no_gateway_env(database_session, monkeypatch, tmp_path):
    _configure_durable_gateway(monkeypatch, tmp_path, base_url=None, agent_token=None)
    executor = _RecordingTerminalBenchExecutor()
    async with database_session() as session:
        submission, job = await _durable_submission_and_job(session, tmp_path)
        result = await runner._run_terminal_bench_task_durable(
            session, executor, submission, job, _terminal_bench_task()
        )

    assert result.status == "completed"
    env = executor.spec.env
    assert "BASE_LLM_GATEWAY_URL" not in env
    assert "BASE_GATEWAY_TOKEN" not in env


async def test_durable_path_does_not_require_base_gateway_redaction(
    database_session, monkeypatch, tmp_path
):
    """With Base gateway gone, residual agent token is never shipped into the container."""

    _configure_durable_gateway(
        monkeypatch, tmp_path, base_url=GATEWAY_BASE_URL, agent_token=AGENT_TOKEN
    )
    executor = _RecordingTerminalBenchExecutor(leak=(AGENT_TOKEN,))
    async with database_session() as session:
        submission, job = await _durable_submission_and_job(session, tmp_path)
        result = await runner._run_terminal_bench_task_durable(
            session, executor, submission, job, _terminal_bench_task()
        )

    env = executor.spec.env
    assert "BASE_GATEWAY_TOKEN" not in env
    # Token may still appear if executor "leaks" synthetic stdout for the test;
    # product path simply does not inject the Base gateway secret into env.
    assert result.status == "completed"
