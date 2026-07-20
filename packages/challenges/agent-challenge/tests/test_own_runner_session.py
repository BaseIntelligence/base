"""Tests for the own-runner in-container tmux session manager (Task 12).

These tests pin the :class:`TmuxSession` contract against harbor 0.13.1's
``harbor.agents.terminus_2.tmux_session.TmuxSession`` -- the pane lifecycle
(``tmux new-session``), command injection (``tmux send-keys``), capture
(``tmux capture-pane``), liveness (``tmux has-session``) and teardown
(``tmux kill-session``) semantics the agent driver (Task 13) drives.

There are two layers:

* **Unit tests** (no Docker): a recording fake ``environment`` captures the
  exact command strings our session builds and asserts they are byte-faithful
  to harbor's command construction. These give a fast RED -> GREEN signal.
* **Integration tests** (Docker + tmux): a throwaway ``python:3.12-slim``
  container exercises the full create -> send -> capture -> kill lifecycle and
  asserts NO residual tmux sessions remain after teardown. Skipped when Docker
  or the image is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid

import pytest

from agent_challenge.evaluation.own_runner.exec_bridge import (
    DockerExecEnvironment,
    ExecResult,
)
from agent_challenge.evaluation.own_runner.session import TmuxSession

# ---------------------------------------------------------------------------
# Recording fake environment (structural match for DockerExecEnvironment.exec)
# ---------------------------------------------------------------------------


class FakeEnvironment:
    """Records ``exec`` calls and returns canned :class:`ExecResult` objects.

    ``responses`` maps a substring -> ExecResult; the first substring found in
    the command wins. Anything unmatched returns ``ExecResult(return_code=0)``.
    """

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


def _tmux_present() -> ExecResult:
    return ExecResult(stdout="tmux 3.4\n", stderr=None, return_code=0)


# ---------------------------------------------------------------------------
# Unit: construction / validation
# ---------------------------------------------------------------------------


def test_pane_dimensions_must_be_positive_integers() -> None:
    env = FakeEnvironment()
    with pytest.raises(ValueError):
        TmuxSession("s", env, pane_width=0)
    with pytest.raises(ValueError):
        TmuxSession("s", env, pane_height=-1)
    with pytest.raises(ValueError):
        TmuxSession("s", env, pane_width="wide")  # type: ignore[arg-type]


def test_default_pane_geometry_matches_harbor() -> None:
    env = FakeEnvironment()
    session = TmuxSession("agent", env)
    # harbor defaults: 160x40.
    assert session.pane_width == 160
    assert session.pane_height == 40
    assert session.session_name == "agent"


# ---------------------------------------------------------------------------
# Unit: start() / pane lifecycle command construction
# ---------------------------------------------------------------------------


async def test_start_builds_harbor_new_session_command() -> None:
    env = FakeEnvironment(responses={"tmux -V": _tmux_present()})
    session = TmuxSession("agent", env, pane_width=80, pane_height=24)
    await session.start()

    new_session = next(c for c in env.commands if "tmux new-session" in c)
    # harbor: export TERM + SHELL, detached (-d), named (-s), geometry, bash --login.
    assert "export TERM=xterm-256color" in new_session
    assert "export SHELL=/bin/bash" in new_session
    assert "-x 80 -y 24" in new_session
    assert "-d -s agent" in new_session
    assert "'bash --login'" in new_session
    # harbor bumps the scrollback history-limit after creating the session.
    assert any("set-option -g history-limit" in c for c in env.commands)


async def test_start_injects_extra_env_as_tmux_e_options() -> None:
    env = FakeEnvironment(responses={"tmux -V": _tmux_present()})
    session = TmuxSession("agent", env, extra_env={"FOO": "bar baz"})
    await session.start()
    new_session = next(c for c in env.commands if "tmux new-session" in c)
    # harbor passes env via `-e KEY=value`, shell-quoted (value has a space).
    assert "-e 'FOO=bar baz'" in new_session


async def test_start_raises_when_new_session_fails() -> None:
    env = FakeEnvironment(
        responses={
            "tmux -V": _tmux_present(),
            "tmux new-session": ExecResult(stdout="boom", stderr=None, return_code=1),
        }
    )
    session = TmuxSession("agent", env)
    with pytest.raises(RuntimeError):
        await session.start()


# ---------------------------------------------------------------------------
# Unit: _ensure_tmux is a bounded probe with NO offline runtime install
# ---------------------------------------------------------------------------


async def test_ensure_tmux_probes_with_bounded_timeout_and_no_install() -> None:
    # tmux is baked into the task image at BUILD time; the runtime probe just
    # confirms it -- a single bounded `tmux -V`, no package-manager install.
    env = FakeEnvironment(responses={"tmux -V": _tmux_present()})
    session = TmuxSession("agent", env)
    await session._ensure_tmux()
    assert env.commands == ["tmux -V"]
    assert env.calls[0]["timeout_sec"] == TmuxSession._TMUX_PROBE_TIMEOUT_SEC
    assert env.calls[0]["user"] == "root"


async def test_ensure_tmux_fails_fast_when_missing_offline() -> None:
    # The eval runtime is --network none, so an offline apt-get/apk/... could
    # only hang. Missing tmux must raise PROMPTLY with no install attempt.
    env = FakeEnvironment(responses={"tmux -V": ExecResult(stdout="", return_code=127)})
    session = TmuxSession("agent", env)

    start = time.monotonic()
    with pytest.raises(RuntimeError):
        await session._ensure_tmux()
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"_ensure_tmux hung for {elapsed:.2f}s"
    assert env.commands == ["tmux -V"]
    joined = "\n".join(env.commands)
    for forbidden in ("apt-get", "apk add", "dnf install", "yum install"):
        assert forbidden not in joined


async def test_start_raises_fast_when_tmux_missing() -> None:
    env = FakeEnvironment(responses={"tmux -V": ExecResult(stdout="", return_code=127)})
    session = TmuxSession("agent", env)
    with pytest.raises(RuntimeError):
        await session.start()
    # Never advanced to creating the pane.
    assert not any("tmux new-session" in c for c in env.commands)


async def test_ensure_tmux_propagates_probe_timeout_without_hanging() -> None:
    # If the bounded probe itself breaches its timeout, the exec-bridge raises
    # RuntimeError("Command timed out ..."); _ensure_tmux must propagate it
    # (fail fast) instead of swallowing it into an unbounded install loop.
    class _TimingOutEnvironment:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def exec(
            self,
            command: str,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ) -> ExecResult:
            self.commands.append(command)
            raise RuntimeError("Command timed out after 10 seconds")

    env = _TimingOutEnvironment()
    session = TmuxSession("agent", env)
    with pytest.raises(RuntimeError):
        await session._ensure_tmux()
    assert env.commands == ["tmux -V"]


# ---------------------------------------------------------------------------
# Unit: send_keys construction (blocking vs non-blocking) -- harbor parity
# ---------------------------------------------------------------------------


async def test_send_keys_uses_double_dash_option_terminator() -> None:
    env = FakeEnvironment()
    session = TmuxSession("agent", env)
    await session.send_keys(["ls -la"])
    send = next(c for c in env.commands if "send-keys" in c)
    # harbor inserts `--` so keys are never parsed as options.
    assert "tmux send-keys -t agent --" in send
    assert "'ls -la'" in send


async def test_blocking_send_keys_appends_completion_sentinel_and_waits() -> None:
    env = FakeEnvironment()
    session = TmuxSession("agent", env)
    # harbor blocks only when the final key executes the command (Enter).
    await session.send_keys(["echo hi", "Enter"], block=True)
    joined = "\n".join(env.commands)
    # harbor: append `; tmux wait -S done` + Enter, then block on `tmux wait done`.
    assert "tmux wait -S done" in joined
    assert "tmux wait done" in joined


async def test_block_without_executor_key_is_non_blocking() -> None:
    env = FakeEnvironment()
    session = TmuxSession("agent", env)
    # No terminating key -> harbor treats this as a keystroke, NOT a command:
    # it is sent non-blocking even under block=True (no completion sentinel).
    await session.send_keys("echo hi", block=True)
    joined = "\n".join(env.commands)
    assert "tmux wait -S done" not in joined
    assert "tmux wait done" not in joined


async def test_blocking_send_keys_strips_trailing_enter_before_sentinel() -> None:
    env = FakeEnvironment()
    session = TmuxSession("agent", env)
    # When the caller already terminates with Enter, harbor pops it, appends the
    # sentinel, then a single Enter -- so we never double-execute.
    await session.send_keys(["echo hi", "Enter"], block=True)
    send = next(c for c in env.commands if "send-keys" in c and "tmux wait -S done" in c)
    # The sentinel command is present exactly once and followed by Enter.
    assert send.count("tmux wait -S done") == 1


async def test_blocking_timeout_raises_timeouterror() -> None:
    env = FakeEnvironment(
        responses={"tmux wait done": ExecResult(stdout="", stderr=None, return_code=124)}
    )
    session = TmuxSession("agent", env)
    with pytest.raises(TimeoutError):
        await session.send_keys(["sleep 999", "Enter"], block=True, max_timeout_sec=1)


async def test_oversized_keys_split_across_multiple_send_commands() -> None:
    env = FakeEnvironment()
    session = TmuxSession("agent", env)
    big = "x" * 40_000
    await session.send_keys([big])
    send_cmds = [c for c in env.commands if "send-keys" in c]
    # Exceeds the ~16 KB tmux command ceiling -> must be chunked.
    assert len(send_cmds) >= 2
    for cmd in send_cmds:
        assert len(cmd.encode("utf-8")) <= TmuxSession._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH


# ---------------------------------------------------------------------------
# Unit: capture / liveness / teardown command construction
# ---------------------------------------------------------------------------


async def test_capture_pane_visible_and_entire() -> None:
    env = FakeEnvironment(
        responses={"capture-pane": ExecResult(stdout="screen\n", stderr=None, return_code=0)}
    )
    session = TmuxSession("agent", env)
    out = await session.capture_pane()
    assert out == "screen\n"
    visible = next(c for c in env.commands if "capture-pane" in c)
    assert visible == "tmux capture-pane -p -t agent"
    env.commands.clear()
    await session.capture_pane(capture_entire=True)
    entire = next(c for c in env.commands if "capture-pane" in c)
    # Entire scrollback uses `-S -`.
    assert entire == "tmux capture-pane -p -S - -t agent"


async def test_capture_pane_returns_empty_string_when_no_output() -> None:
    env = FakeEnvironment(
        responses={"capture-pane": ExecResult(stdout=None, stderr=None, return_code=0)}
    )
    session = TmuxSession("agent", env)
    assert await session.capture_pane() == ""


async def test_is_session_alive_maps_has_session_return_code() -> None:
    alive = FakeEnvironment(
        responses={"has-session": ExecResult(stdout="", stderr=None, return_code=0)}
    )
    dead = FakeEnvironment(
        responses={"has-session": ExecResult(stdout="", stderr=None, return_code=1)}
    )
    assert await TmuxSession("agent", alive).is_session_alive() is True
    assert await TmuxSession("agent", dead).is_session_alive() is False
    cmd = next(c for c in alive.commands if "has-session" in c)
    assert cmd == "tmux has-session -t agent"


async def test_stop_kills_session_and_is_idempotent() -> None:
    env = FakeEnvironment()
    session = TmuxSession("agent", env)
    await session.stop()
    kill = next(c for c in env.commands if "kill-session" in c)
    assert kill == "tmux kill-session -t agent"
    # Idempotent: a second stop must not raise even though session is gone.
    env.commands.clear()
    await session.stop()


async def test_async_context_manager_starts_and_tears_down() -> None:
    env = FakeEnvironment(responses={"tmux -V": _tmux_present()})
    async with TmuxSession("agent", env) as session:
        assert isinstance(session, TmuxSession)
    assert any("tmux new-session" in c for c in env.commands)
    assert any("tmux kill-session -t agent" in c for c in env.commands)


# ---------------------------------------------------------------------------
# Integration: real tmux inside a throwaway container
# ---------------------------------------------------------------------------

_IMAGE = "python:3.12-slim"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", _IMAGE],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


docker_required = pytest.mark.skipif(
    not _docker_ready(),
    reason=f"docker + {_IMAGE} image required for tmux session container tests",
)


@pytest.fixture
def container():
    # host network so the stand-in tmux install works (slim lacks it); the
    # default bridge is unavailable in some CI sandboxes.
    environment = DockerExecEnvironment.launch(_IMAGE, network="host")
    # tmux is now baked into task images at BUILD time; the runtime _ensure_tmux
    # only probes (no offline install). Stand in for that baked layer by ensuring
    # tmux is present before the lifecycle test drives the session.
    probe = subprocess.run(
        [
            "docker",
            "exec",
            "-u",
            "root",
            environment.container_name,
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
                    environment.container_name,
                    "bash",
                    "-lc",
                    "apt-get update && apt-get install -y --no-install-recommends tmux",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            environment.remove()
            pytest.skip("no network to install tmux into the task container")
        if install.returncode != 0:
            environment.remove()
            pytest.skip("tmux unavailable in the task container (no build network)")
    try:
        yield environment
    finally:
        environment.remove()


@docker_required
async def test_full_lifecycle_create_send_capture_teardown(container) -> None:
    name = f"agent-{uuid.uuid4().hex[:8]}"
    session = TmuxSession(name, container, pane_width=120, pane_height=30)
    await session.start()
    try:
        assert await session.is_session_alive() is True
        await session.send_keys(["echo HARBOR_PARITY_OK", "Enter"], block=True, max_timeout_sec=30)
        captured = await session.capture_pane(capture_entire=True)
        assert "HARBOR_PARITY_OK" in captured
    finally:
        await session.stop()

    # Teardown leaves NO residual sessions.
    assert await session.is_session_alive() is False
    ls = await container.exec("tmux ls 2>&1 || true")
    out = ls.stdout or ""
    assert name not in out
