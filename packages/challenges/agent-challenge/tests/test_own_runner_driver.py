"""Tests for the own-runner agent invocation driver (Task 13).

The driver loads the submitted agent (``agent:Agent``, honoring the harbor-free
BaseAgent SDK contract), drives ``setup`` -> ``run`` against the task
environment, wires an in-container tmux session, enforces a two-clock timeout
model (wall-clock around the whole run + a per-command exec ceiling), and maps
load failures / crashes / timeouts onto the centralized reason-code taxonomy.

Two layers, mirroring ``test_own_runner_session.py``:

* **Unit tests** (no Docker): duck-typed fake agents and a recording fake
  environment pin the driver's orchestration, env forwarding, timeout handling
  and cleanup -- a fast RED -> GREEN signal. The fakes do NOT subclass the
  baseagent ``BaseAgent`` (agent-challenge cannot import ``src.sdk``); they only
  honor the structural contract the driver depends on.
* **Integration test** (Docker): a throwaway ``python:3.12-slim`` container is
  driven end-to-end by a real agent class that runs a command through
  ``environment.exec`` and returns output. Skipped when Docker / the image is
  unavailable, and asserts the container is removed afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess

import pytest

from agent_challenge.evaluation.own_runner.driver import (
    AGENT_CRASH_REASON_CODE,
    AGENT_LOAD_FAILED_REASON_CODE,
    AGENT_TIMEOUT_REASON_CODE,
    DEFAULT_AGENT_IMPORT_PATH,
    AgentDriver,
    AgentLoadError,
    AgentRunResult,
    DriverContext,
    load_agent_class,
)
from agent_challenge.evaluation.own_runner.exec_bridge import ExecResult
from agent_challenge.evaluation.own_runner.reason_codes import REASON_CODES

# ---------------------------------------------------------------------------
# Recording fake environment (structural match for DockerExecEnvironment.exec).
# Crucially the signature does NOT accept ``timeout`` or ``workdir`` so the
# baseagent harbor_registry adapter's ``timeout=`` / ``workdir=`` attempts raise
# TypeError and cascade to ``timeout_sec=`` -- exactly as against real harbor.
# ---------------------------------------------------------------------------


class FakeEnvironment:
    """Records ``exec`` calls and returns canned :class:`ExecResult` objects."""

    def __init__(self, responses: dict[str, ExecResult] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[dict[str, object]] = []
        self.commands: list[str] = []

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_sec": timeout_sec,
                "user": user,
            }
        )
        self.commands.append(command)
        for needle, result in self.responses.items():
            if needle in command:
                return result
        return ExecResult(stdout=None, stderr=None, return_code=0)


def _tmux_present_env() -> FakeEnvironment:
    """A fake env that reports tmux is already installed (probe returns 0)."""
    return FakeEnvironment({"tmux -V": ExecResult(stdout="tmux 3.4\n", return_code=0)})


class FakeContainer:
    """Stands in for a launched task container; records removal."""

    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


# ---------------------------------------------------------------------------
# Duck-typed fake agents (honor the SDK contract WITHOUT importing it).
# ---------------------------------------------------------------------------


class RecordingFakeAgent:
    """A well-behaved agent that records how the driver invokes it."""

    last_instance: RecordingFakeAgent | None = None

    def __init__(self, logs_dir=None, model_name=None, **kwargs) -> None:
        self.logs_dir = logs_dir
        self.model_name = model_name
        self.init_kwargs = kwargs
        self.setup_env: object | None = None
        self.run_args: tuple[object, object, object] | None = None
        self.order: list[str] = []
        type(self).last_instance = self

    @staticmethod
    def name() -> str:
        return "fake-agent"

    @staticmethod
    def version() -> str:
        return "0.1.0"

    async def setup(self, environment) -> None:
        self.order.append("setup")
        self.setup_env = environment

    async def run(self, instruction, environment, context) -> str:
        self.order.append("run")
        self.run_args = (instruction, environment, context)
        return "fake output"


class CrashingFakeAgent(RecordingFakeAgent):
    async def run(self, instruction, environment, context) -> str:
        raise RuntimeError("boom")


class CommandTimeoutCrashFakeAgent(RecordingFakeAgent):
    """Simulates a per-command exec timeout surfacing as a RuntimeError."""

    async def run(self, instruction, environment, context) -> str:
        raise RuntimeError("Command timed out after 120 seconds")


class InnerTimeoutFakeAgent(RecordingFakeAgent):
    """Raises its OWN TimeoutError (not the driver's wall-clock)."""

    async def run(self, instruction, environment, context) -> str:
        raise TimeoutError("inner blocking send timed out")


class SleepingFakeAgent(RecordingFakeAgent):
    async def run(self, instruction, environment, context) -> str:
        await asyncio.sleep(5)
        return "never"


class ExecCallingFakeAgent(RecordingFakeAgent):
    """Calls ``environment.exec`` to exercise the per-command timeout wrapper."""

    async def run(self, instruction, environment, context) -> str:
        await environment.exec("echo hi", timeout_sec=600)
        await environment.exec("echo bye")
        return "exec done"


class ConstructCrashFakeAgent:
    def __init__(self, logs_dir=None, model_name=None, **kwargs) -> None:
        raise RuntimeError("cannot construct")

    @staticmethod
    def name() -> str:
        return "broken"

    @staticmethod
    def version() -> str:
        return "0.0.0"

    async def setup(self, environment) -> None:  # pragma: no cover - never reached
        ...

    async def run(self, instruction, environment, context) -> str:  # pragma: no cover
        return ""


# ---------------------------------------------------------------------------
# load_agent_class
# ---------------------------------------------------------------------------


def test_default_import_path_is_agent_colon_agent() -> None:
    assert DEFAULT_AGENT_IMPORT_PATH == "agent:Agent"


def test_load_agent_class_valid() -> None:
    cls = load_agent_class(
        "agent_challenge.evaluation.own_runner.exec_bridge:DockerExecEnvironment"
    )
    from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment

    assert cls is DockerExecEnvironment


def test_load_agent_class_missing_colon() -> None:
    with pytest.raises(AgentLoadError):
        load_agent_class("agent.Agent")


def test_load_agent_class_missing_module() -> None:
    with pytest.raises(AgentLoadError):
        load_agent_class("agent_challenge.no_such_module_xyz:Thing")


def test_load_agent_class_missing_attribute() -> None:
    with pytest.raises(AgentLoadError):
        load_agent_class("agent_challenge.evaluation.own_runner.exec_bridge:NoSuchClass")


def test_load_agent_class_not_a_type() -> None:
    with pytest.raises(AgentLoadError):
        load_agent_class("agent_challenge.evaluation.own_runner.exec_bridge:DEFAULT_WORKDIR")


def test_agent_load_error_carries_load_failed_reason_code() -> None:
    try:
        load_agent_class("agent.Agent")
    except AgentLoadError as exc:
        assert exc.reason_code == AGENT_LOAD_FAILED_REASON_CODE
    else:  # pragma: no cover
        pytest.fail("expected AgentLoadError")


# ---------------------------------------------------------------------------
# Reason-code taxonomy membership (no invented codes)
# ---------------------------------------------------------------------------


def test_driver_reason_codes_are_known() -> None:
    assert AGENT_LOAD_FAILED_REASON_CODE in REASON_CODES
    assert AGENT_CRASH_REASON_CODE in REASON_CODES
    assert AGENT_TIMEOUT_REASON_CODE in REASON_CODES
    # Pin the exact harbor strings miners/dashboards key on.
    assert AGENT_LOAD_FAILED_REASON_CODE == "harbor_submission_code_failed"
    assert AGENT_CRASH_REASON_CODE == "harbor_trial_failed"
    assert AGENT_TIMEOUT_REASON_CODE == "harbor_agent_timeout_error"


# ---------------------------------------------------------------------------
# drive(): happy path
# ---------------------------------------------------------------------------


async def test_drive_calls_setup_then_run_and_completes() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=RecordingFakeAgent)
    result = await driver.drive(
        environment=env,
        instruction="do the thing",
        start_session=False,
    )
    assert isinstance(result, AgentRunResult)
    assert result.status == "completed"
    assert result.reason_code is None
    assert result.output == "fake output"
    agent = RecordingFakeAgent.last_instance
    assert agent is not None
    assert agent.order == ["setup", "run"]  # setup strictly before run
    assert agent.run_args is not None
    instruction, run_env, context = agent.run_args
    assert instruction == "do the thing"
    assert isinstance(context, DriverContext)


async def test_drive_forwards_logs_dir_and_model_name() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=RecordingFakeAgent)
    await driver.drive(
        environment=env,
        instruction="x",
        model_name="deepseek-v4-pro",
        start_session=False,
    )
    agent = RecordingFakeAgent.last_instance
    assert agent is not None
    assert agent.model_name == "deepseek-v4-pro"


# ---------------------------------------------------------------------------
# drive(): agent env forwarding (constructor extra_env + context.env), and
# os.environ is NEVER mutated.
# ---------------------------------------------------------------------------


async def test_drive_passes_agent_env_via_context_and_constructor() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=RecordingFakeAgent)
    agent_env = {"BASE_LLM_GATEWAY_URL": "https://gw.test/llm/v1", "BASE_GATEWAY_TOKEN": "tok"}
    await driver.drive(
        environment=env,
        instruction="x",
        agent_env=agent_env,
        start_session=False,
    )
    agent = RecordingFakeAgent.last_instance
    assert agent is not None
    # Constructor parity with the harbor factory: extra_env keyword.
    assert agent.init_kwargs.get("extra_env") == agent_env
    # SDK run contract: context.env carries the same mapping.
    assert agent.run_args is not None
    _, _, context = agent.run_args
    assert context.env == agent_env


async def test_drive_does_not_mutate_os_environ() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=RecordingFakeAgent)
    before = dict(os.environ)
    await driver.drive(
        environment=env,
        instruction="x",
        agent_env={"SECRET_TOKEN_XYZ": "leaked?"},
        start_session=False,
    )
    assert "SECRET_TOKEN_XYZ" not in os.environ
    assert dict(os.environ) == before


# ---------------------------------------------------------------------------
# drive(): tmux session wiring on the RAW environment
# ---------------------------------------------------------------------------


async def test_drive_starts_and_stops_session_on_raw_environment() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=RecordingFakeAgent)
    result = await driver.drive(
        environment=env,
        instruction="x",
        agent_env={"FOO": "bar"},
        start_session=True,
        session_name="agent",
    )
    assert result.status == "completed"
    joined = "\n".join(env.commands)
    assert "tmux new-session" in joined  # created
    assert "kill-session -t agent" in joined  # torn down
    # session env vars are injected into the pane (harbor -e KEY=value).
    assert "FOO=bar" in joined


# ---------------------------------------------------------------------------
# drive(): crash -> harbor_trial_failed
# ---------------------------------------------------------------------------


async def test_drive_crash_maps_to_trial_failed() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=CrashingFakeAgent)
    result = await driver.drive(
        environment=env,
        instruction="x",
        start_session=False,
    )
    assert result.status == "failed"
    assert result.reason_code == AGENT_CRASH_REASON_CODE
    assert result.error and "boom" in result.error


async def test_drive_per_command_runtime_timeout_is_trial_failed_not_agent_timeout() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=CommandTimeoutCrashFakeAgent)
    result = await driver.drive(
        environment=env,
        instruction="x",
        wall_clock_sec=30,
        start_session=False,
    )
    assert result.status == "failed"
    # A per-command exec timeout surfaces as RuntimeError -> crash, NOT the
    # wall-clock agent-timeout code.
    assert result.reason_code == AGENT_CRASH_REASON_CODE


async def test_drive_inner_timeout_error_is_crash_not_agent_timeout() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=InnerTimeoutFakeAgent)
    result = await driver.drive(
        environment=env,
        instruction="x",
        wall_clock_sec=30,  # generous: the wall-clock will NOT expire
        start_session=False,
    )
    assert result.status == "failed"
    assert result.reason_code == AGENT_CRASH_REASON_CODE


# ---------------------------------------------------------------------------
# drive(): load / construct failure -> harbor_submission_code_failed
# ---------------------------------------------------------------------------


async def test_drive_load_failure_maps_to_submission_code_failed() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(import_path="agent_challenge.no_such_mod_zzz:Agent")
    result = await driver.drive(
        environment=env,
        instruction="x",
        start_session=False,
    )
    assert result.status == "failed"
    assert result.reason_code == AGENT_LOAD_FAILED_REASON_CODE


async def test_drive_construct_failure_maps_to_submission_code_failed() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=ConstructCrashFakeAgent)
    result = await driver.drive(
        environment=env,
        instruction="x",
        start_session=False,
    )
    assert result.status == "failed"
    assert result.reason_code == AGENT_LOAD_FAILED_REASON_CODE


# ---------------------------------------------------------------------------
# drive(): wall-clock timeout -> harbor_agent_timeout_error
# ---------------------------------------------------------------------------


async def test_drive_wall_clock_timeout_maps_to_agent_timeout() -> None:
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=SleepingFakeAgent)
    result = await driver.drive(
        environment=env,
        instruction="x",
        wall_clock_sec=0.1,
        start_session=False,
    )
    assert result.status == "failed"
    assert result.reason_code == AGENT_TIMEOUT_REASON_CODE


# ---------------------------------------------------------------------------
# drive(): per-command timeout ceiling injection / capping
# ---------------------------------------------------------------------------


async def test_drive_per_command_ceiling_caps_and_injects_timeout() -> None:
    env = FakeEnvironment()
    driver = AgentDriver(agent_class=ExecCallingFakeAgent)
    result = await driver.drive(
        environment=env,
        instruction="x",
        command_timeout_sec=120,
        start_session=False,
    )
    assert result.status == "completed"
    timeouts = [c["timeout_sec"] for c in env.calls]
    # 600 capped down to 120; None injected up to 120.
    assert timeouts == [120, 120]


async def test_drive_without_ceiling_passes_timeout_through_unchanged() -> None:
    env = FakeEnvironment()
    driver = AgentDriver(agent_class=ExecCallingFakeAgent)
    await driver.drive(
        environment=env,
        instruction="x",
        start_session=False,
    )
    timeouts = [c["timeout_sec"] for c in env.calls]
    assert timeouts == [600, None]  # untouched


# ---------------------------------------------------------------------------
# drive(): container cleanup on EVERY exit path
# ---------------------------------------------------------------------------


async def test_drive_removes_container_on_success() -> None:
    env = _tmux_present_env()
    container = FakeContainer()
    driver = AgentDriver(agent_class=RecordingFakeAgent)
    await driver.drive(
        environment=env,
        instruction="x",
        container=container,
        start_session=False,
    )
    assert container.removed is True


async def test_drive_removes_container_on_crash() -> None:
    env = _tmux_present_env()
    container = FakeContainer()
    driver = AgentDriver(agent_class=CrashingFakeAgent)
    await driver.drive(
        environment=env,
        instruction="x",
        container=container,
        start_session=False,
    )
    assert container.removed is True


async def test_drive_removes_container_on_wall_clock_timeout() -> None:
    env = _tmux_present_env()
    container = FakeContainer()
    driver = AgentDriver(agent_class=SleepingFakeAgent)
    await driver.drive(
        environment=env,
        instruction="x",
        container=container,
        wall_clock_sec=0.1,
        start_session=False,
    )
    assert container.removed is True


# ---------------------------------------------------------------------------
# drive(): live pane tailer (best-effort real-time agent-terminal streaming)
# ---------------------------------------------------------------------------


class _ScriptedPaneSession:
    """Feeds a scripted sequence of pane captures / raised errors to the tailer.

    Each ``capture_pane`` call consumes the next step: a string is returned, a
    :class:`BaseException` is raised. Once the script is exhausted it keeps
    returning the last string (so duplicate captures are produced and skipped).
    """

    def __init__(self, steps: list[object]) -> None:
        self._steps = list(steps)
        self._index = 0
        self._last = ""
        self.capture_calls = 0

    async def capture_pane(self, capture_entire: bool = False) -> str:
        self.capture_calls += 1
        if self._index < len(self._steps):
            step = self._steps[self._index]
            self._index += 1
            if isinstance(step, BaseException):
                raise step
            self._last = str(step)
            return self._last
        return self._last


async def _run_tailer_until(session: _ScriptedPaneSession, on_incremental, done: asyncio.Event):
    task = asyncio.create_task(AgentDriver._pane_tailer(session, on_incremental, interval_sec=0))
    try:
        await asyncio.wait_for(done.wait(), timeout=2.0)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_pane_tailer_streams_only_new_suffix() -> None:
    # "" -> abc -> abcdef (suffix) -> abcdef (dup, skipped) -> xyznew (whole).
    session = _ScriptedPaneSession(["abc", "abcdef", "abcdef", "xyznew"])
    deltas: list[str] = []
    done = asyncio.Event()

    async def _on_incremental(delta: str) -> None:
        deltas.append(delta)
        if delta == "xyznew":
            done.set()

    await _run_tailer_until(session, _on_incremental, done)
    assert deltas == ["abc", "def", "xyznew"]


async def test_pane_tailer_swallows_capture_and_emit_failures() -> None:
    # First capture raises, then an emit raises -- both swallowed; the tailer
    # keeps running and still delivers the later delta.
    session = _ScriptedPaneSession([RuntimeError("capture boom"), "hello", "helloworld"])
    delivered: list[str] = []
    done = asyncio.Event()

    async def _on_incremental(delta: str) -> None:
        if delta == "hello":
            raise RuntimeError("emit boom")
        delivered.append(delta)
        done.set()

    await _run_tailer_until(session, _on_incremental, done)
    assert delivered == ["world"]
    assert session.capture_calls >= 3


async def test_drive_with_on_incremental_completes_and_cancels_tailer() -> None:
    # A fast agent completes before the (default-interval) tailer fires; drive()
    # must still create + cleanly cancel the tailer and report completion.
    env = _tmux_present_env()
    driver = AgentDriver(agent_class=RecordingFakeAgent)
    deltas: list[str] = []

    async def _on_incremental(delta: str) -> None:
        deltas.append(delta)

    result = await driver.drive(
        environment=env,
        instruction="x",
        start_session=True,
        session_name="agent",
        on_incremental=_on_incremental,
    )
    assert result.status == "completed"
    joined = "\n".join(env.commands)
    assert "kill-session -t agent" in joined  # session still torn down


# ---------------------------------------------------------------------------
# Integration: drive a real agent against a throwaway container.
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, text=True, check=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


class _RealExecAgent:
    """A real agent that runs one command through the environment exec-bridge."""

    def __init__(self, logs_dir=None, model_name=None, **kwargs) -> None:
        self.observed: str | None = None

    @staticmethod
    def name() -> str:
        return "real-exec-agent"

    @staticmethod
    def version() -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        await environment.exec("true")

    async def run(self, instruction, environment, context) -> str:
        result = await environment.exec("echo driver-ok")
        self.observed = (result.stdout or "").strip()
        return self.observed


@pytest.mark.skipif(not _docker_available(), reason="docker unavailable")
async def test_drive_end_to_end_against_real_container() -> None:
    from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment

    image = "python:3.12-slim"
    pull = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    if pull.returncode != 0:
        pulled = subprocess.run(
            ["docker", "pull", image], capture_output=True, text=True, timeout=300
        )
        if pulled.returncode != 0:
            pytest.skip(f"cannot pull {image}")

    # host network so the stand-in tmux install works (python:3.12-slim ships no
    # tmux); mirrors the session integration test. tmux is baked into task images
    # at BUILD time now, so the runtime _ensure_tmux only probes -- pre-install it
    # here to stand in for that baked layer (skip if no network).
    container = DockerExecEnvironment.launch(image, network="host")
    probe = subprocess.run(
        [
            "docker",
            "exec",
            "-u",
            "root",
            container.container_name,
            "bash",
            "-lc",
            "command -v tmux",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        try:
            install = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-u",
                    "root",
                    container.container_name,
                    "bash",
                    "-lc",
                    "apt-get update && apt-get install -y --no-install-recommends tmux",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            container.remove()
            pytest.skip("no network to install tmux into the task container")
        if install.returncode != 0:
            container.remove()
            pytest.skip("tmux unavailable in the task container (no build network)")
    driver = AgentDriver(agent_class=_RealExecAgent)
    result = await driver.drive(
        environment=container,
        instruction="echo something",
        container=container,
        command_timeout_sec=60,
        wall_clock_sec=120,
        start_session=True,
    )
    assert result.status == "completed"
    assert result.output == "driver-ok"

    # Container must have been removed by the driver's cleanup.
    inspect = subprocess.run(
        ["docker", "container", "inspect", container.container_name],
        capture_output=True,
        text=True,
    )
    assert inspect.returncode != 0, "driver must remove the container on exit"
