"""Tests for the own-runner execution backend composition (Task 16).

The backend (:mod:`agent_challenge.evaluation.own_runner_backend`) is the END-TO-END
glue that composes the eight already-built own-runner modules into a single
runnable pipeline that emits a valid ``BASE_BENCHMARK_RESULT=<json>`` line --
the third selectable Terminal-Bench execution backend ("own_runner") alongside
the existing "harbor" (default) and "base_sdk" backends.

Layers (mirrors the sibling own-runner test modules):

* **Composition** (no docker): :func:`run_own_runner_job` wired to a REAL
  :class:`AgentDriver` (injected fake agent class) + an injected preparer/verifier
  seam, proving the orchestrate -> aggregate -> benchmark-result path.
* **CLI** (no docker): :func:`main` parses args, runs the job, and prints the
  ``BASE_BENCHMARK_RESULT=`` line to stdout.
* **Selection / dispatch** (no docker): the additive "own_runner" backend is
  selectable and routes through the runner dispatch seams without disturbing the
  default "harbor" backend.
* **Script generation** (no docker): the runner emits an own-runner container
  script that invokes the backend (no harbor CLI).
* **Docker integration** (``@docker_required``): the full backend with the real
  ``run_verifier`` against a throwaway ``python:3.12-slim`` container, k=2 oracle
  pass, asserting a real emitted ``BASE_BENCHMARK_RESULT=`` line.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.evaluation.own_runner.orchestrator import (
    JobResult,
    PreparedTrial,
    TaskSpec,
    TrialId,
)
from agent_challenge.evaluation.own_runner.result_schema import (
    RESULT_LINE_PREFIX,
    build_benchmark_result,
)
from agent_challenge.evaluation.own_runner.verifier_runner import VerifierOutcome
from agent_challenge.evaluation.own_runner_backend import (
    CACHE_ROOT_ENV,
    DEFAULT_CACHE_ROOT,
    EVALUATION_CONCURRENCY_ENV,
    _build_default_preparer,
    _concurrency_cap_from_env,
    _resolve_cache_root,
    _resolve_concurrency_cap,
    main,
    run_own_runner_job,
)


# ===========================================================================
# Test doubles
# ===========================================================================
class _OracleAgent:
    """A trivial agent whose setup/run succeed (drive -> completed)."""

    def __init__(self, *, logs_dir: Any = None, model_name: Any = None, **kwargs: Any) -> None:
        self.logs_dir = logs_dir
        self.model_name = model_name

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> str:
        return "DONE"


class _FakeEnvironment:
    """Minimal recording exec-bridge stand-in (duck-typed TerminalEnvironment)."""

    def __init__(self) -> None:
        self.removed = False
        self.commands: list[str] = []

    async def exec(self, command: str, **kwargs: Any) -> Any:
        self.commands.append(command)
        return type("R", (), {"return_code": 0, "stdout": "", "stderr": None})()

    def remove(self) -> None:
        self.removed = True


# ===========================================================================
# Composition (no docker): run_own_runner_job over injected seams
# ===========================================================================
async def test_run_own_runner_job_composes_pipeline_to_completed(tmp_path: Path) -> None:
    envs: list[_FakeEnvironment] = []

    async def _preparer(trial_id: TrialId, task: TaskSpec) -> PreparedTrial:
        env = _FakeEnvironment()
        envs.append(env)
        return PreparedTrial(
            environment=env,
            instruction="do it",
            tests_source_dir=tmp_path / "tests",
            start_session=False,
        )

    async def _fake_verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        return VerifierOutcome(
            status="completed",
            score=1.0,
            resolved=1,
            total=1,
            reason_code=None,
            rewards={"reward": 1.0},
        )

    result = await run_own_runner_job(
        task_ids=["task"],
        job_dir=tmp_path / "job",
        agent_class=_OracleAgent,
        preparer=_preparer,
        verifier=_fake_verifier,
        n_attempts=2,
        n_concurrent=2,
    )

    assert isinstance(result, JobResult)
    assert result.status == "completed"
    assert result.score == 1.0
    assert result.resolved == 2
    assert result.total == 2
    # The validated benchmark-result dict carries exactly the five core fields.
    assert result.benchmark_result["status"] == "completed"
    assert set(result.benchmark_result) == {
        "status",
        "score",
        "resolved",
        "total",
        "reason_code",
    }
    # Every environment was torn down by the runner after verification.
    assert all(env.removed for env in envs)


async def test_run_own_runner_job_agent_crash_yields_failed(tmp_path: Path) -> None:
    class _CrashAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def setup(self, environment: Any) -> None:
            return None

        async def run(self, instruction: str, environment: Any, context: Any) -> str:
            raise RuntimeError("boom")

    async def _preparer(trial_id: TrialId, task: TaskSpec) -> PreparedTrial:
        return PreparedTrial(
            environment=_FakeEnvironment(),
            instruction="do it",
            tests_source_dir=tmp_path / "tests",
            start_session=False,
        )

    verifier_called = False

    async def _fake_verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        nonlocal verifier_called
        verifier_called = True
        return VerifierOutcome(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None, rewards={"r": 1}
        )

    result = await run_own_runner_job(
        task_ids=["task"],
        job_dir=tmp_path / "job",
        agent_class=_CrashAgent,
        preparer=_preparer,
        verifier=_fake_verifier,
    )

    # Agent crashed -> errored trial, verifier skipped, job failed.
    assert result.status == "failed"
    assert result.n_errored_trials == 1
    assert verifier_called is False
    assert result.benchmark_result["status"] == "failed"


# ===========================================================================
# CLI (no docker): main() prints the BASE_BENCHMARK_RESULT line
# ===========================================================================
def test_main_emits_benchmark_result_line(monkeypatch, tmp_path, capsys) -> None:
    canned = JobResult(
        status="completed",
        score=1.0,
        resolved=1,
        total=1,
        reason_code=None,
        pass_at_k={},
        n_total_trials=1,
        n_completed_trials=1,
        n_errored_trials=0,
        trial_outcomes=[],
        benchmark_result=build_benchmark_result(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None
        ),
    )

    captured_kwargs: dict[str, Any] = {}

    async def _fake_run(**kwargs: Any) -> JobResult:
        captured_kwargs.update(kwargs)
        return canned

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.run_own_runner_job", _fake_run
    )

    rc = main(
        [
            "run",
            "--task",
            "hello-world",
            "--job-dir",
            str(tmp_path / "job"),
            "--n-attempts",
            "1",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith(RESULT_LINE_PREFIX)]
    assert len(line) == 1
    assert '"status": "completed"' in line[0]
    assert captured_kwargs["task_ids"] == ["hello-world"]


def test_main_fail_closed_emits_failed_line_on_error(monkeypatch, tmp_path, capsys) -> None:
    async def _boom(**kwargs: Any) -> JobResult:
        raise RuntimeError("explode")

    monkeypatch.setattr("agent_challenge.evaluation.own_runner_backend.run_own_runner_job", _boom)

    rc = main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])

    # Fail-closed: a crash still emits a valid (failed) benchmark-result line.
    assert rc != 0
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith(RESULT_LINE_PREFIX)]
    assert len(line) == 1
    assert '"status": "failed"' in line[0]


def test_resolve_cache_root_precedence(monkeypatch, tmp_path) -> None:
    # explicit arg wins over env + default.
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "from-env"))
    assert _resolve_cache_root(tmp_path / "explicit") == tmp_path / "explicit"
    # env wins over the built-in default.
    assert _resolve_cache_root(None) == tmp_path / "from-env"
    # default when neither explicit nor env is set.
    monkeypatch.delenv(CACHE_ROOT_ENV, raising=False)
    assert _resolve_cache_root(None) == DEFAULT_CACHE_ROOT


def test_main_threads_cache_root_from_env(monkeypatch, tmp_path, capsys) -> None:
    canned = JobResult(
        status="completed",
        score=1.0,
        resolved=1,
        total=1,
        reason_code=None,
        pass_at_k={},
        n_total_trials=1,
        n_completed_trials=1,
        n_errored_trials=0,
        trial_outcomes=[],
        benchmark_result=build_benchmark_result(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None
        ),
    )
    captured_kwargs: dict[str, Any] = {}

    async def _fake_run(**kwargs: Any) -> JobResult:
        captured_kwargs.update(kwargs)
        return canned

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.run_own_runner_job", _fake_run
    )
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "cache"))

    rc = main(["run", "--task", "t", "--job-dir", str(tmp_path / "job")])

    assert rc == 0
    assert captured_kwargs["cache_root"] == tmp_path / "cache"


# ===========================================================================
# Concurrency cap default (settings.evaluation_concurrency bounds the orchestrator)
# ===========================================================================
def test_concurrency_cap_from_env_parses_and_validates(monkeypatch) -> None:
    monkeypatch.setenv(EVALUATION_CONCURRENCY_ENV, "3")
    assert _concurrency_cap_from_env() == 3
    # Unset -> no cap (pure auto-sizing).
    monkeypatch.delenv(EVALUATION_CONCURRENCY_ENV, raising=False)
    assert _concurrency_cap_from_env() is None
    # Non-integer / below-one -> ignored (no cap).
    monkeypatch.setenv(EVALUATION_CONCURRENCY_ENV, "not-a-number")
    assert _concurrency_cap_from_env() is None
    monkeypatch.setenv(EVALUATION_CONCURRENCY_ENV, "0")
    assert _concurrency_cap_from_env() is None


def test_resolve_concurrency_cap_explicit_flag_wins(monkeypatch) -> None:
    # An explicit --concurrency-cap always overrides the configured default.
    monkeypatch.setenv(EVALUATION_CONCURRENCY_ENV, "3")
    assert _resolve_concurrency_cap(7) == 7


def test_resolve_concurrency_cap_defaults_from_settings(monkeypatch) -> None:
    # No explicit flag -> the miner-configured evaluation_concurrency bounds it.
    monkeypatch.setenv(EVALUATION_CONCURRENCY_ENV, "3")
    assert _resolve_concurrency_cap(None) == 3


def test_resolve_concurrency_cap_falls_back_to_env_without_settings(monkeypatch) -> None:
    # The lean canonical image lacks pydantic-settings, so ChallengeSettings
    # cannot be constructed; the cap must still be read from the same
    # CHALLENGE_EVALUATION_CONCURRENCY env var (fallback), never crashing main().
    import agent_challenge.sdk.config as cfg

    class _NoSettings:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            raise ModuleNotFoundError("No module named 'pydantic_settings'")

    monkeypatch.setattr(cfg, "ChallengeSettings", _NoSettings)
    monkeypatch.setenv(EVALUATION_CONCURRENCY_ENV, "5")
    assert _resolve_concurrency_cap(None) == 5


def _capture_run_kwargs(monkeypatch) -> dict[str, Any]:
    canned = JobResult(
        status="completed",
        score=1.0,
        resolved=1,
        total=1,
        reason_code=None,
        pass_at_k={},
        n_total_trials=1,
        n_completed_trials=1,
        n_errored_trials=0,
        trial_outcomes=[],
        benchmark_result=build_benchmark_result(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None
        ),
    )
    captured: dict[str, Any] = {}

    async def _fake_run(**kwargs: Any) -> JobResult:
        captured.update(kwargs)
        return canned

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.run_own_runner_job", _fake_run
    )
    return captured


def test_main_defaults_concurrency_cap_from_settings(monkeypatch, tmp_path) -> None:
    captured = _capture_run_kwargs(monkeypatch)
    monkeypatch.setenv(EVALUATION_CONCURRENCY_ENV, "3")

    rc = main(["run", "--task", "t", "--job-dir", str(tmp_path / "job")])

    assert rc == 0
    assert captured["concurrency_cap"] == 3


def test_main_explicit_concurrency_cap_overrides_settings(monkeypatch, tmp_path) -> None:
    captured = _capture_run_kwargs(monkeypatch)
    monkeypatch.setenv(EVALUATION_CONCURRENCY_ENV, "3")

    rc = main(
        [
            "run",
            "--task",
            "t",
            "--job-dir",
            str(tmp_path / "job"),
            "--concurrency-cap",
            "9",
        ]
    )

    assert rc == 0
    assert captured["concurrency_cap"] == 9


# ===========================================================================
# Docker integration: full backend with the real run_verifier
# ===========================================================================
_IMAGE = "python:3.12-slim"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", _IMAGE], capture_output=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


docker_required = pytest.mark.skipif(
    not _docker_ready(), reason=f"docker + {_IMAGE} image required for backend container tests"
)

_BINARY_TEST_SH = """#!/bin/bash
if [ -f /app/solved ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def _make_tests_dir(tmp_path: Path) -> Path:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_BINARY_TEST_SH)
    test_sh.chmod(0o755)
    return tests_dir


@docker_required
async def test_backend_docker_oracle_emits_result_line(tmp_path: Path) -> None:
    from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment
    from agent_challenge.evaluation.own_runner.result_schema import (
        format_benchmark_result_line,
    )
    from agent_challenge.evaluation.own_runner.verifier_runner import run_verifier

    tests_dir = _make_tests_dir(tmp_path)

    async def _preparer(trial_id: TrialId, task: TaskSpec) -> PreparedTrial:
        env = DockerExecEnvironment.launch(_IMAGE, network="host")
        # Oracle: solve the task before the agent runs so the verifier writes 1.
        await env.exec("touch /app/solved", user="root")
        return PreparedTrial(
            environment=env,
            instruction="solve it",
            tests_source_dir=tests_dir,
            start_session=False,
        )

    result = await run_own_runner_job(
        task_ids=["task"],
        job_dir=tmp_path / "job",
        agent_class=_OracleAgent,
        preparer=_preparer,
        verifier=run_verifier,
        n_attempts=2,
        n_concurrent=2,
    )

    assert result.status == "completed"
    assert result.score == 1.0
    assert result.resolved == 2
    assert result.total == 2

    # The real emitted wire line is a valid BASE_BENCHMARK_RESULT= line.
    line = format_benchmark_result_line(result.benchmark_result)
    assert line.startswith(RESULT_LINE_PREFIX)
    assert '"status": "completed"' in line


# ===========================================================================
# Selection / dispatch wiring (own_runner as a third selectable backend)
# ===========================================================================
def test_own_runner_provider_constant_distinct() -> None:
    from agent_challenge.evaluation.terminal_bench import (
        TERMINAL_BENCH_BASE_SDK_PROVIDER,
        TERMINAL_BENCH_HARBOR_PROVIDER,
        TERMINAL_BENCH_OWN_RUNNER_PROVIDER,
    )

    assert TERMINAL_BENCH_OWN_RUNNER_PROVIDER == "own_runner"
    assert TERMINAL_BENCH_OWN_RUNNER_PROVIDER != TERMINAL_BENCH_HARBOR_PROVIDER
    assert TERMINAL_BENCH_OWN_RUNNER_PROVIDER != TERMINAL_BENCH_BASE_SDK_PROVIDER


def test_own_runner_in_attempt_providers() -> None:
    from agent_challenge.evaluation.terminal_bench import (
        TERMINAL_BENCH_ATTEMPT_PROVIDERS,
        TERMINAL_BENCH_BASE_SDK_PROVIDER,
        TERMINAL_BENCH_HARBOR_PROVIDER,
        TERMINAL_BENCH_LEGACY_BASE_SDK_PROVIDER,
        TERMINAL_BENCH_OWN_RUNNER_PROVIDER,
    )

    assert TERMINAL_BENCH_OWN_RUNNER_PROVIDER in TERMINAL_BENCH_ATTEMPT_PROVIDERS
    assert TERMINAL_BENCH_HARBOR_PROVIDER in TERMINAL_BENCH_ATTEMPT_PROVIDERS
    assert TERMINAL_BENCH_BASE_SDK_PROVIDER in TERMINAL_BENCH_ATTEMPT_PROVIDERS
    assert TERMINAL_BENCH_LEGACY_BASE_SDK_PROVIDER in TERMINAL_BENCH_ATTEMPT_PROVIDERS


def test_execution_provider_resolves_own_runner() -> None:
    from agent_challenge.evaluation.runner import _terminal_bench_execution_provider

    assert _terminal_bench_execution_provider("own_runner") == "own_runner"


def test_runner_image_resolves_own_runner_to_task_image() -> None:
    from agent_challenge.evaluation.benchmarks import BenchmarkTask
    from agent_challenge.evaluation.runner import _terminal_bench_runner_image

    task = BenchmarkTask(task_id="t", docker_image="python:3.12-slim")
    # own_runner reuses the task image (like harbor), not the base_sdk image.
    assert _terminal_bench_runner_image(task, "own_runner") == "python:3.12-slim"


# ===========================================================================
# Task 17: the agent's own output becomes a DISTINCT stream=agent task-event
# ===========================================================================
class _MarkerAgent:
    """Agent whose run() returns a distinctive marker (the agent's OWN output)."""

    MARKER = "AGENT-OWN-OUTPUT::backend-marker-9b21"

    def __init__(self, *, logs_dir: Any = None, model_name: Any = None, **kwargs: Any) -> None:
        pass

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> str:
        return self.MARKER


async def _make_submission(session: Any) -> Any:
    from agent_challenge.models import AgentSubmission

    submission = AgentSubmission(
        miner_hotkey="miner-task17",
        name="agent-task17",
        agent_hash="task17hash",
        artifact_uri="/tmp/task17.zip",
        raw_status="received",
        effective_status="received",
    )
    session.add(submission)
    await session.flush()
    return submission


async def test_own_runner_agent_output_recorded_only_under_stream_agent(
    tmp_path: Path, database_session: Any
) -> None:
    from sqlalchemy import select

    from agent_challenge.evaluation.task_events import record_separated_trial_logs
    from agent_challenge.evaluation.terminal_bench import _separated_log_refs
    from agent_challenge.models import TaskLogEvent

    install_marker = "HARNESS-INSTALL-OUTPUT::pip-install-marker-3c8d"

    async def _preparer(trial_id: TrialId, task: TaskSpec) -> PreparedTrial:
        return PreparedTrial(
            environment=_FakeEnvironment(),
            instruction="do it",
            tests_source_dir=tmp_path / "tests",
            start_session=False,
        )

    async def _fake_verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        return VerifierOutcome(
            status="completed",
            score=1.0,
            resolved=1,
            total=1,
            reason_code=None,
            rewards={"reward": 1.0},
        )

    job_dir = tmp_path / "job"
    result = await run_own_runner_job(
        task_ids=["task"],
        job_dir=job_dir,
        agent_class=_MarkerAgent,
        preparer=_preparer,
        verifier=_fake_verifier,
        n_attempts=1,
    )
    assert result.status == "completed"

    trial_dirs = sorted((job_dir / "trials").iterdir())
    assert len(trial_dirs) == 1
    trial_dir = trial_dirs[0]

    # The orchestrator wrote the agent's OWN output under <trial_dir>/agent/**.
    agent_files = list((trial_dir / "agent").rglob("*"))
    assert any(p.is_file() for p in agent_files)
    agent_blob = "\n".join(p.read_text() for p in agent_files if p.is_file())
    assert _MarkerAgent.MARKER in agent_blob

    # Simulate harness/install output landing on the trial-runner log (harness
    # stream) -- it must NEVER be tagged stream=agent.
    (trial_dir / "trial.log").write_text(install_marker)

    refs = _separated_log_refs(trial_dir)
    assert refs.get("agent_log_files")
    assert refs.get("trial_log_ref")

    async with database_session() as session:
        submission = await _make_submission(session)
        await record_separated_trial_logs(
            session,
            submission_id=submission.id,
            job_id=None,
            task_result_id=None,
            task_id="task",
            artifacts=refs,
            status="completed",
        )
        await session.commit()

        rows = list(
            (
                await session.execute(
                    select(TaskLogEvent)
                    .where(TaskLogEvent.submission_id == submission.id)
                    .order_by(TaskLogEvent.sequence)
                )
            )
            .scalars()
            .all()
        )

    by_stream: dict[str | None, str] = {}
    for row in rows:
        by_stream.setdefault(row.stream, "")
        by_stream[row.stream] += row.message

    # The agent marker appears ONLY under stream=agent.
    assert _MarkerAgent.MARKER in by_stream.get("agent", "")
    for stream, blob in by_stream.items():
        if stream != "agent":
            assert _MarkerAgent.MARKER not in blob
    # The harness/install marker is tagged harness, NEVER agent.
    assert install_marker in by_stream.get("harness", "")
    assert install_marker not in by_stream.get("agent", "")


# ===========================================================================
# Task 22: additive stage_solution flag (baseline oracle parity gate)
# ===========================================================================
# The production preparer must NOT stage a solution by default (miner path
# unchanged: agents bring their own behaviour). When the backend is run with
# stage_solution=True -- the reference-oracle baseline mode -- the preparer
# copies <task_root>/solution into the container at /solution BEFORE the agent
# runs, so a harbor-free OracleAgent can exec the staged solve.sh.
_STAGE_TASK_ID = "stage-solution-probe"

_STAGE_TASK_TOML = """\
[task]
name = "stage/probe"

[environment]
docker_image = "python:3.12-slim"
cpus = 1
memory_mb = 512

[agent]
timeout_sec = 60.0

[verifier]
timeout_sec = 60.0
"""

_STAGE_DOCKERFILE = "FROM python:3.12-slim\nWORKDIR /app\n"
_STAGE_INSTRUCTION = "do the thing\n"
_STAGE_SOLVE = "#!/bin/bash\ntouch /app/solved\n"
_STAGE_TEST_SH = "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"


def _make_stage_task_root(tmp_path: Path) -> Path:
    root = tmp_path / "cache" / _STAGE_TASK_ID
    root.mkdir(parents=True)
    (root / "task.toml").write_text(_STAGE_TASK_TOML, encoding="utf-8")
    (root / "instruction.md").write_text(_STAGE_INSTRUCTION, encoding="utf-8")
    env = root / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text(_STAGE_DOCKERFILE, encoding="utf-8")
    sol = root / "solution"
    sol.mkdir()
    (sol / "solve.sh").write_text(_STAGE_SOLVE, encoding="utf-8")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(_STAGE_TEST_SH, encoding="utf-8")
    return root


class _RecordingEnv:
    """Records upload_dir calls so we can assert solution staging."""

    def __init__(self) -> None:
        self.uploads: list[tuple[Path, str]] = []
        self.removed = False

    def upload_dir(self, source_dir: Any, target_dir: str) -> None:
        self.uploads.append((Path(source_dir), target_dir))

    async def exec(self, command: str, **kwargs: Any) -> Any:
        return type("R", (), {"return_code": 0, "stdout": "", "stderr": None})()

    def remove(self) -> None:
        self.removed = True


class _FakeBuilder:
    """Builder stand-in whose prepare() returns a recording env (no docker)."""

    def __init__(self) -> None:
        self.env = _RecordingEnv()

    def prepare(self, task: Any) -> Any:
        return type("Built", (), {"env": self.env})()


def _stage_manifest(task_root: Path) -> dict[str, Any]:
    from agent_challenge.evaluation.own_runner.taskdefs import compute_task_digest

    return {"tasks": {_STAGE_TASK_ID: {"content_digest_sha256": compute_task_digest(task_root)}}}


async def test_default_preparer_does_not_stage_solution_by_default(tmp_path: Path) -> None:
    """Miner path unchanged: no solution staged when stage_solution is omitted."""
    task_root = _make_stage_task_root(tmp_path)
    builder = _FakeBuilder()

    preparer = _build_default_preparer(
        task_ids=[_STAGE_TASK_ID],
        cache_root=task_root.parent,
        digest_manifest=_stage_manifest(task_root),
        digest_manifest_path=None,
        builder=builder,  # type: ignore[arg-type]
        agent_env=None,
    )
    prepared = await preparer(
        TrialId(task_name=_STAGE_TASK_ID, attempt=0),
        TaskSpec(task_name=_STAGE_TASK_ID),
    )

    assert prepared.environment is builder.env
    assert builder.env.uploads == []


async def test_default_preparer_stages_solution_when_enabled(tmp_path: Path) -> None:
    """stage_solution=True copies <task_root>/solution into the container at /solution."""
    task_root = _make_stage_task_root(tmp_path)
    builder = _FakeBuilder()

    preparer = _build_default_preparer(
        task_ids=[_STAGE_TASK_ID],
        cache_root=task_root.parent,
        digest_manifest=_stage_manifest(task_root),
        digest_manifest_path=None,
        builder=builder,  # type: ignore[arg-type]
        agent_env=None,
        stage_solution=True,
    )
    await preparer(
        TrialId(task_name=_STAGE_TASK_ID, attempt=0),
        TaskSpec(task_name=_STAGE_TASK_ID),
    )

    assert builder.env.uploads == [(task_root / "solution", "/solution")]


async def test_run_own_runner_job_threads_stage_solution_to_default_preparer(
    tmp_path: Path,
) -> None:
    """run_own_runner_job forwards stage_solution into the production preparer."""
    task_root = _make_stage_task_root(tmp_path)
    builder = _FakeBuilder()

    async def _fake_verifier(environment: Any, **kwargs: Any) -> VerifierOutcome:
        return VerifierOutcome(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None, rewards={"r": 1}
        )

    result = await run_own_runner_job(
        task_ids=[_STAGE_TASK_ID],
        job_dir=tmp_path / "job",
        cache_root=task_root.parent,
        digest_manifest=_stage_manifest(task_root),
        agent_class=_OracleAgent,
        verifier=_fake_verifier,
        builder=builder,  # type: ignore[arg-type]
        stage_solution=True,
        n_attempts=1,
        n_concurrent=1,
    )

    assert result.status == "completed"
    assert builder.env.uploads == [(task_root / "solution", "/solution")]
