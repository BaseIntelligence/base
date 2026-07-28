"""Offline tests for the Terminal-Bench miner agent (scripts/miner_agent).

Scenarios (contract):
  S1 stubbed LLM → known-good shell tool call applied via environment.exec
  S2 missing OPENROUTER_API_KEY → clean typed failure, no secret leakage
  S3 malformed LLM output → task scored as miss, no process abort
  S4 per-task timeout → miss, run returns
  S5 packaged ZIP < 1 MiB and contains no credential-shaped literals
  S6 entrypoint matches own-runner driver contract (ctor + setup + run)
  S7 miner _miner_* modules resolve inside scripts/miner_agent (no tools/ shadow)
"""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.evaluation.own_runner.driver import AgentDriver

_MINER_DIR = Path(__file__).resolve().parents[1] / "scripts" / "miner_agent"
_AGENT_PATH = _MINER_DIR / "agent.py"
_MAX_ZIP_BYTES = 1_048_576


def _load_miner_module() -> Any:
    import sys

    # Prefer the agent directory over any host package root so flat sibling
    # imports (_miner_*) cannot be shadowed by namespace packages.
    miner_dir = str(_MINER_DIR.resolve())
    try:
        while miner_dir in sys.path:
            sys.path.remove(miner_dir)
    except ValueError:
        pass
    sys.path.insert(0, miner_dir)

    # Drop stale generic module names that a prior test may have imported.
    for stale in ("tools", "loop", "openrouter"):
        mod = sys.modules.get(stale)
        if mod is None:
            continue
        origin = getattr(mod, "__file__", None) or ""
        if "miner_agent" not in origin.replace("\\", "/"):
            sys.modules.pop(stale, None)

    spec = importlib.util.spec_from_file_location("miner_agent_agent", _AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_agent_class() -> type:
    return _load_miner_module().Agent


def test_miner_modules_not_shadowed_by_package_tools_namespace() -> None:
    """Regression: package-root `tools/` must not shadow miner siblings.

    Full-suite runs put ``packages/challenges/agent-challenge`` on sys.path,
    where ``tools/`` is a real directory (namespace package). Generic names
    like ``tools`` / ``loop`` then resolve to the wrong place. Miner internals
    must use unique ``_miner_*`` module names and resolve inside miner_agent/.
    """
    import sys

    package_root = str(Path(__file__).resolve().parents[1])
    # Simulate full-suite path ordering: package root first.
    if package_root in sys.path:
        sys.path.remove(package_root)
    sys.path.insert(0, package_root)

    # Ensure a bare `import tools` would hit the package namespace, not miner.
    import tools as host_tools  # noqa: F401

    host_file = getattr(host_tools, "__file__", None)
    # Namespace package has __file__ is None; regular module has a path.
    # Either way it must NOT be under scripts/miner_agent.
    if host_file is not None:
        assert "scripts/miner_agent" not in Path(host_file).as_posix()

    module = _load_miner_module()
    assert hasattr(module, "Agent")

    import _miner_loop
    import _miner_openrouter
    import _miner_tools

    for mod in (_miner_tools, _miner_loop, _miner_openrouter):
        origin = Path(mod.__file__).resolve()
        assert origin.is_file(), mod
        assert _MINER_DIR.resolve() in origin.parents or origin.parent == _MINER_DIR.resolve()
        assert origin.name.startswith("_miner_"), origin.name

    # agent.py itself must still be named agent.py (driver contract).
    assert _AGENT_PATH.name == "agent.py"
    assert "class Agent" in _AGENT_PATH.read_text(encoding="utf-8")


class _RecordingEnv:
    """Duck-typed exec bridge that records commands and returns canned results."""

    def __init__(self) -> None:
        self.exec_calls: list[tuple[str, dict[str, Any]]] = []
        self._files: dict[str, str] = {}

    async def exec(self, command: str, **kwargs: Any) -> Any:
        self.exec_calls.append((command, kwargs))
        # Minimal filesystem simulation for write/read patterns used by the agent.
        if command.startswith("cat ") or command.startswith("/bin/cat "):
            path = command.split(None, 1)[1].strip().strip("'\"")
            content = self._files.get(path, "")
            return type(
                "R",
                (),
                {"return_code": 0 if path in self._files else 1, "stdout": content, "stderr": ""},
            )()
        if "tee " in command or command.startswith("printf ") or " > " in command:
            # Best-effort capture of simple redirects for stub E2E.
            if " > " in command:
                parts = command.rsplit(" > ", 1)
                if len(parts) == 2:
                    path = parts[1].strip().split()[0].strip("'\"")
                    # Extract payload from echo/printf when present.
                    payload = "ok"
                    if "echo " in parts[0]:
                        payload = parts[0].split("echo ", 1)[1].strip().strip("'\"")
                    self._files[path] = payload
        if command.strip() in {"pwd", "pwd -P"}:
            return type("R", (), {"return_code": 0, "stdout": "/app\n", "stderr": ""})()
        if command.startswith("ls") or command.startswith("find "):
            return type("R", (), {"return_code": 0, "stdout": "README.md\n", "stderr": ""})()
        return type("R", (), {"return_code": 0, "stdout": "", "stderr": ""})()


class _StubLLM:
    """Deterministic LLM: first call issues a shell write; second call finishes."""

    def __init__(self, mode: str = "good") -> None:
        self.mode = mode
        self.calls = 0

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        self.calls += 1
        if self.mode == "garbage":
            return {"content": "not-json-{{{", "tool_calls": None, "raw": "@@@"}
        if self.mode == "malformed_tools":
            return {
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "nope", "arguments": "{"}}],
            }
        if self.calls == 1:
            args = json.dumps(
                {
                    "command": "echo agent-solved-ok > /tmp/agent-solved-ok",
                    "workdir": "/app",
                }
            )
            return {
                "content": "writing marker",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "shell_command", "arguments": args},
                    }
                ],
            }
        return {"content": "done", "tool_calls": None}


# ---------------------------------------------------------------------------
# S6 — entrypoint contract
# ---------------------------------------------------------------------------


def test_miner_agent_constructs_per_driver_contract() -> None:
    Agent = _load_agent_class()
    agent = Agent(
        logs_dir=Path("/tmp/logs"),
        model_name="x-ai/grok-4.5",
        extra_env={"OPENROUTER_API_KEY": "test-key-not-real"},
        unexpected_extra="ignored",
    )
    assert agent is not None


def test_miner_agent_constructs_with_defaults() -> None:
    Agent = _load_agent_class()
    agent = Agent(logs_dir=None, model_name=None)
    assert agent is not None


async def test_miner_agent_setup_is_callable() -> None:
    Agent = _load_agent_class()
    env = _RecordingEnv()
    agent = Agent(logs_dir=None, model_name=None, extra_env={"OPENROUTER_API_KEY": "k"})
    await agent.setup(env)


# ---------------------------------------------------------------------------
# S2 — missing API key
# ---------------------------------------------------------------------------


async def test_missing_openrouter_key_clean_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_miner_module()
    Agent = module.Agent
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = _RecordingEnv()
    agent = Agent(logs_dir=None, model_name=None, extra_env={})

    with pytest.raises(module.MissingAPIKeyError) as exc_info:
        await agent.setup(env)
        await agent.run("do something", env, context=None)

    msg = str(exc_info.value)
    assert "OPENROUTER_API_KEY" in msg
    assert "sk-" not in msg
    assert "traceback" not in msg.lower()


# ---------------------------------------------------------------------------
# S1 — stubbed LLM end-to-end offline
# ---------------------------------------------------------------------------


async def test_stubbed_llm_applies_solution_via_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_miner_module()
    Agent = module.Agent
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-offline-only")
    env = _RecordingEnv()
    stub = _StubLLM(mode="good")
    agent = Agent(
        logs_dir=None,
        model_name="x-ai/grok-4.5",
        extra_env={"OPENROUTER_API_KEY": "test-key-offline-only"},
        llm_client=stub,
    )

    await agent.setup(env)
    output = await agent.run(
        "Create /tmp/agent-solved-ok containing agent-solved-ok",
        env,
        context=type("C", (), {"env": {"OPENROUTER_API_KEY": "test-key-offline-only"}})(),
    )

    assert stub.calls >= 1
    assert any("agent-solved-ok" in cmd for cmd, _ in env.exec_calls)
    assert isinstance(output, str)
    assert output  # non-empty summary


async def test_stubbed_llm_via_real_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_miner_module()
    Agent = module.Agent
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-offline-only")
    env = _RecordingEnv()
    stub = _StubLLM(mode="good")

    class _Factory(Agent):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("llm_client", stub)
            kwargs.setdefault("extra_env", {"OPENROUTER_API_KEY": "test-key-offline-only"})
            super().__init__(*args, **kwargs)

    driver = AgentDriver(agent_class=_Factory)
    result = await driver.drive(
        environment=env,
        instruction="Create /tmp/agent-solved-ok containing agent-solved-ok",
        start_session=False,
        agent_env={"OPENROUTER_API_KEY": "test-key-offline-only"},
    )
    assert result.status == "completed"
    assert any("agent-solved-ok" in cmd for cmd, _ in env.exec_calls)


# ---------------------------------------------------------------------------
# S3 — garbage / malformed LLM
# ---------------------------------------------------------------------------


async def test_garbage_llm_is_miss_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_miner_module()
    Agent = module.Agent
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-offline-only")
    env = _RecordingEnv()
    stub = _StubLLM(mode="garbage")
    agent = Agent(
        logs_dir=None,
        model_name=None,
        extra_env={"OPENROUTER_API_KEY": "test-key-offline-only"},
        llm_client=stub,
        max_steps=3,
    )
    await agent.setup(env)
    output = await agent.run("impossible", env, context=None)
    assert isinstance(output, str)
    # Miss summary — must not raise.


async def test_malformed_tool_args_is_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_miner_module()
    Agent = module.Agent
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-offline-only")
    env = _RecordingEnv()
    stub = _StubLLM(mode="malformed_tools")
    agent = Agent(
        logs_dir=None,
        model_name=None,
        extra_env={"OPENROUTER_API_KEY": "test-key-offline-only"},
        llm_client=stub,
        max_steps=3,
    )
    await agent.setup(env)
    output = await agent.run("x", env, context=None)
    assert isinstance(output, str)


# ---------------------------------------------------------------------------
# S4 — timeout
# ---------------------------------------------------------------------------


async def test_task_timeout_is_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_miner_module()
    Agent = module.Agent
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-offline-only")
    env = _RecordingEnv()

    class _SlowLLM:
        calls = 0

        def chat(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
        ) -> dict[str, Any]:
            import time

            time.sleep(0.5)
            self.calls += 1
            return {"content": "still working", "tool_calls": None}

    agent = Agent(
        logs_dir=None,
        model_name=None,
        extra_env={"OPENROUTER_API_KEY": "test-key-offline-only"},
        llm_client=_SlowLLM(),
        task_timeout_sec=0.2,
        max_steps=50,
    )
    await agent.setup(env)
    output = await agent.run("slow task", env, context=None)
    assert isinstance(output, str)
    assert "timeout" in output.lower() or "miss" in output.lower() or "time" in output.lower()


# ---------------------------------------------------------------------------
# S5 — ZIP packaging hygiene
# ---------------------------------------------------------------------------


def test_build_zip_under_1mib_and_no_credentials() -> None:
    # Prefer the package helper if present; else use submit_agent.build_agent_zip.
    build_path = _MINER_DIR / "build_zip.py"
    if build_path.is_file():
        spec = importlib.util.spec_from_file_location("miner_build_zip", build_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        data = mod.build_zip(_MINER_DIR)
    else:
        # Fall back to loading submit_agent.py by path.
        sa_path = Path(__file__).resolve().parents[1] / "scripts" / "submit_agent.py"
        spec = importlib.util.spec_from_file_location("submit_agent_mod", sa_path)
        assert spec is not None and spec.loader is not None
        sa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sa)
        data = sa.build_agent_zip(_MINER_DIR)

    assert isinstance(data, (bytes, bytearray))
    assert len(data) < _MAX_ZIP_BYTES
    assert len(data) > 0
    # No credential-shaped literals anywhere in the archive bytes.
    assert b"sk-" not in data
    assert b"Bearer " not in data
    # Must contain agent.py at root.
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert "agent.py" in names
        source = zf.read("agent.py")
        assert b"class Agent" in source
        assert b"sk-" not in source


def test_miner_source_has_no_embedded_secrets() -> None:
    for path in _MINER_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sk-" not in text
        assert "Bearer " not in text


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
