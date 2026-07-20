"""Agent invocation driver (own-runner backend, Task 13).

This module owns the loop that takes a submitted agent and actually *runs* it
against a prepared task container. It loads the agent class (honoring the
harbor-free BaseAgent SDK contract -- ``agent:Agent`` by default), constructs it
with ``(logs_dir=, model_name=, **kwargs)``, wires an in-container tmux session
(Task 12) on the task environment, drives ``setup(environment)`` then
``run(instruction, environment, context)``, and maps load failures, crashes and
timeouts onto the centralized reason-code taxonomy (Task 7).

Why faithful matters: the submitted agent (baseagent ``agent:Agent``) is the same
artifact that runs under stock harbor. The driver therefore honors the exact SDK
shape harbor uses -- ``setup`` before ``run``, the agent's env supplied both via
the constructor (harbor's factory passes ``extra_env``) and via ``context.env``
(the run-time contract, read by ``agent._extract_context_env``) -- and never
mutates ``os.environ`` to inject agent secrets.

Two-clock timeout model (independent, never conflated):

* **Wall-clock** -- an :func:`asyncio.timeout` around the whole ``setup``+``run``.
  When IT expires the attempt is an agent timeout
  (``harbor_agent_timeout_error``). A :class:`TimeoutError` raised from *inside*
  the agent (e.g. a tmux blocking-send ceiling) is distinguished via the timeout
  context's :meth:`expired` and treated as a crash, not a wall-clock timeout.
* **Per-command** -- a thin environment wrapper (:class:`_CommandTimeoutEnvironment`)
  caps/injects ``timeout_sec`` on each ``environment.exec`` the agent issues. A
  command exceeding that ceiling surfaces from the exec-bridge as a
  ``RuntimeError`` -> mapped to a crash (``harbor_trial_failed``), NOT the
  wall-clock code. The wrapper is given ONLY to the agent; the tmux session runs
  on the raw environment (its own ``max_timeout_sec`` governs blocking sends).

This module builds on the Task-10 exec-bridge contract and the Task-12
:class:`TmuxSession`; it does NOT modify either, nor the SDK contract. Package
wiring (``__init__`` / ``pyproject``) is deferred to Task 16.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_challenge.evaluation.own_runner.exec_bridge import ExecResult
from agent_challenge.evaluation.own_runner.reason_codes import REASON_CODES
from agent_challenge.evaluation.own_runner.session import TmuxSession

#: Default agent entry point. The baseagent submission exposes ``agent:Agent``
#: (``agent.py``'s ``Agent.import_path()``), so this mirrors stock harbor.
DEFAULT_AGENT_IMPORT_PATH = "agent:Agent"

#: Default tmux session name for the agent's interactive pane.
DEFAULT_SESSION_NAME = "agent"

#: Default poll interval (seconds) for the best-effort live pane tailer. Kept at
#: >= 2s and paired with a byte-diff so live streaming bounds event volume / DB
#: writes even with several concurrent trials per worker.
DEFAULT_INCREMENTAL_INTERVAL_SEC = 3.0

#: The agent ZIP could not be loaded or constructed -- the submission's own code
#: is at fault. Maps to harbor's submission-code-failed final code.
AGENT_LOAD_FAILED_REASON_CODE = "harbor_submission_code_failed"

#: The agent's ``setup``/``run`` raised (including a per-command exec timeout
#: surfacing as ``RuntimeError``). Maps to harbor's generic trial-failed code.
AGENT_CRASH_REASON_CODE = "harbor_trial_failed"

#: The whole ``setup``+``run`` exceeded the wall-clock budget. Maps to harbor's
#: agent-timeout code -- kept distinct from per-command crashes.
AGENT_TIMEOUT_REASON_CODE = "harbor_agent_timeout_error"


class AgentLoadError(Exception):
    """The agent class could not be located / loaded (typed, fail-fast).

    Carries a ``reason_code`` drawn from the own-runner taxonomy so the failure
    maps cleanly onto a known outcome.
    """

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = AGENT_LOAD_FAILED_REASON_CODE,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass
class AgentRunResult:
    """Outcome of a single :meth:`AgentDriver.drive` invocation.

    ``status`` is ``"completed"`` when ``run`` returned normally, otherwise
    ``"failed"`` with a ``reason_code`` from the taxonomy. ``output`` is the
    agent's returned summary on success; ``error`` carries the failure detail.
    Scoring / reward extraction is a downstream concern (Task 14), NOT the
    driver's -- the driver reports only how the *invocation* went.
    """

    status: str
    reason_code: str | None = None
    output: str | None = None
    error: str | None = None
    agent_info: Any | None = None


@dataclass
class DriverContext:
    """The ``context`` object handed to ``agent.run``.

    Honors the SDK run contract's ``context.env`` access path
    (``agent._extract_context_env`` reads ``getattr(context, "env", {})``), so
    the agent receives its runtime configuration without any reliance on
    ``os.environ``.
    """

    env: dict[str, str] = field(default_factory=dict)


def load_agent_class(import_path: str) -> type:
    """Resolve a ``"module:Qualname"`` string to the agent class object.

    Raises :class:`AgentLoadError` (carrying
    :data:`AGENT_LOAD_FAILED_REASON_CODE`) on a malformed path, a missing
    module, a missing attribute, or a target that is not a class -- every
    failure mode is the submission's fault, so all map to one reason code.
    """
    if ":" not in import_path:
        raise AgentLoadError(f"agent import path must be 'module:Qualname', got {import_path!r}")
    module_path, _, qualname = import_path.partition(":")
    if not module_path or not qualname:
        raise AgentLoadError(f"agent import path must be 'module:Qualname', got {import_path!r}")

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise AgentLoadError(f"could not import agent module {module_path!r}: {exc}") from exc

    obj: Any = module
    try:
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except AttributeError as exc:
        raise AgentLoadError(
            f"agent {qualname!r} not found in module {module_path!r}: {exc}"
        ) from exc

    if not isinstance(obj, type):
        raise AgentLoadError(
            f"agent target {import_path!r} resolved to {type(obj).__name__}, not a class"
        )
    return obj


class _CommandTimeoutEnvironment:
    """Wraps an exec-bridge to enforce a per-command ``timeout_sec`` ceiling.

    Given ONLY to the agent (never the tmux session). The signature mirrors the
    Task-10 exec-bridge exactly -- notably it accepts NO ``timeout`` / ``workdir``
    kwargs, so the baseagent adapter's ``timeout=``/``workdir=`` probes raise
    ``TypeError`` and cascade to ``timeout_sec=`` precisely as against real
    harbor. Unknown attributes forward to the wrapped environment.
    """

    def __init__(self, inner: Any, ceiling_sec: int) -> None:
        self._inner = inner
        self._ceiling_sec = ceiling_sec

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if timeout_sec is None:
            effective = self._ceiling_sec
        else:
            effective = min(timeout_sec, self._ceiling_sec)
        return await self._inner.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_sec=effective,
            user=user,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class AgentDriver:
    """Loads and drives a submitted agent against a prepared task environment.

    Construct with either an ``import_path`` (resolved lazily at drive time) or
    an injected ``agent_class`` (test/embedding seam). ``extra_init_kwargs`` are
    forwarded verbatim to the agent constructor (harbor-factory parity).
    """

    def __init__(
        self,
        *,
        import_path: str = DEFAULT_AGENT_IMPORT_PATH,
        agent_class: type | None = None,
        extra_init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._import_path = import_path
        self._agent_class = agent_class
        self._extra_init_kwargs = dict(extra_init_kwargs or {})

    async def drive(
        self,
        *,
        environment: Any,
        instruction: str,
        container: Any | None = None,
        logs_dir: Path | str | None = None,
        model_name: str | None = None,
        agent_env: dict[str, str] | None = None,
        wall_clock_sec: float | None = None,
        command_timeout_sec: int | None = None,
        session_name: str | None = None,
        start_session: bool = True,
        user: str | int | None = None,
        on_incremental: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentRunResult:
        """Run the agent end-to-end and report the invocation outcome.

        On every exit path the tmux session is torn down and (when given) the
        ``container`` is removed. Failures never raise -- they are reported as a
        ``"failed"`` :class:`AgentRunResult` with a taxonomy reason code.

        When ``on_incremental`` is given (and a session exists), a best-effort
        background task tails the live tmux pane while the agent runs and forwards
        each fresh delta to it for real-time streaming. That tailer is purely
        observability: every fault is swallowed and it can never change the
        outcome or crash the trial.
        """
        resolved_agent_env = dict(agent_env or {})
        session: TmuxSession | None = None
        try:
            # 1. Load + construct the agent. Any failure here is the
            #    submission's fault -> submission_code_failed.
            try:
                agent_cls = self._agent_class or load_agent_class(self._import_path)
                init_kwargs = dict(self._extra_init_kwargs)
                if resolved_agent_env:
                    init_kwargs.setdefault("extra_env", dict(resolved_agent_env))
                agent = agent_cls(
                    logs_dir=Path(logs_dir) if logs_dir is not None else None,
                    model_name=model_name,
                    **init_kwargs,
                )
            except AgentLoadError as exc:
                return AgentRunResult(status="failed", reason_code=exc.reason_code, error=str(exc))
            except Exception as exc:
                return AgentRunResult(
                    status="failed",
                    reason_code=AGENT_LOAD_FAILED_REASON_CODE,
                    error=f"agent construction failed: {exc}",
                )

            agent_info = self._safe_agent_info(agent)

            # 2. Create the tmux session on the RAW environment (its own
            #    max_timeout_sec governs blocking sends).
            if start_session:
                session = TmuxSession(
                    session_name or DEFAULT_SESSION_NAME,
                    environment,
                    extra_env=dict(resolved_agent_env),
                    user=user,
                )
                await session.start()

            # 3. The agent sees a per-command-capped view of the environment.
            agent_environment: Any = environment
            if command_timeout_sec is not None:
                agent_environment = _CommandTimeoutEnvironment(environment, command_timeout_sec)

            context = DriverContext(env=dict(resolved_agent_env))

            # 4. Drive setup -> run under the wall-clock budget. A best-effort
            #    background task tails the live pane while the agent runs; it is
            #    cancelled in the finally BEFORE teardown and never affects the
            #    outcome.
            tailer: asyncio.Task[None] | None = None
            if on_incremental is not None and session is not None:
                tailer = asyncio.create_task(self._pane_tailer(session, on_incremental))
            try:
                async with asyncio.timeout(wall_clock_sec) as wall_clock:
                    await agent.setup(agent_environment)
                    output = await agent.run(instruction, agent_environment, context)
            except TimeoutError as exc:
                # Distinguish OUR wall-clock from an inner TimeoutError (e.g. a
                # tmux blocking-send ceiling propagating out of the agent).
                if wall_clock.expired():
                    return AgentRunResult(
                        status="failed",
                        reason_code=AGENT_TIMEOUT_REASON_CODE,
                        error=f"agent exceeded wall-clock budget of {wall_clock_sec}s",
                        agent_info=agent_info,
                    )
                return AgentRunResult(
                    status="failed",
                    reason_code=AGENT_CRASH_REASON_CODE,
                    error=f"agent run raised TimeoutError: {exc}",
                    agent_info=agent_info,
                )
            except Exception as exc:
                return AgentRunResult(
                    status="failed",
                    reason_code=AGENT_CRASH_REASON_CODE,
                    error=f"agent run failed: {exc}",
                    agent_info=agent_info,
                )
            finally:
                # Stop the live tailer BEFORE session teardown so it never
                # captures a dying pane; its only non-swallowed exit is the
                # cancellation we raise here.
                if tailer is not None:
                    tailer.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await tailer

            return AgentRunResult(
                status="completed",
                output=output,
                agent_info=agent_info,
            )
        finally:
            await self._teardown(session, container)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    async def _pane_tailer(
        session: TmuxSession,
        on_incremental: Callable[[str], Awaitable[None]],
        *,
        interval_sec: float = DEFAULT_INCREMENTAL_INTERVAL_SEC,
    ) -> None:
        """Best-effort live tail of the agent's tmux pane (observability only).

        Loops until cancelled: wait ``interval_sec``, full-capture the pane, and
        forward only the newly-appended suffix to ``on_incremental``. Diffing
        against the previous capture bounds event volume; if the new capture does
        not extend the previous one (e.g. the pane was cleared / scrolled) the
        whole capture is sent. EVERY fault (capture, decode, emit) is swallowed so
        the tailer can never disturb the agent run or the trial outcome -- the
        completion batch remains authoritative. It exits only via cancellation
        (``CancelledError`` is intentionally NOT caught).
        """

        previous = ""
        while True:
            await asyncio.sleep(interval_sec)
            try:
                full = await session.capture_pane(capture_entire=True)
            except Exception:
                continue
            if not full or full == previous:
                continue
            delta = full[len(previous) :] if previous and full.startswith(previous) else full
            if not delta:
                continue
            previous = full
            try:
                await on_incremental(delta)
            except Exception:
                continue

    @staticmethod
    def _safe_agent_info(agent: Any) -> Any | None:
        to_info = getattr(agent, "to_agent_info", None)
        if callable(to_info):
            try:
                return to_info()
            except Exception:
                return None
        return None

    @staticmethod
    async def _teardown(session: TmuxSession | None, container: Any | None) -> None:
        """Best-effort cleanup: kill the session, then remove the container.

        Both are suppressed so a teardown failure never masks the real outcome,
        and the container is removed on EVERY exit path (success / crash /
        timeout) to leave no residual resources.
        """
        if session is not None:
            try:
                await session.stop()
            except Exception:
                pass
        if container is not None:
            remove = getattr(container, "remove", None)
            if callable(remove):
                try:
                    result = remove()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass


# Fail fast at import time if the reason codes drift out of the taxonomy
# (mirrors container_builder.py's import-time assertions).
assert AGENT_LOAD_FAILED_REASON_CODE in REASON_CODES
assert AGENT_CRASH_REASON_CODE in REASON_CODES
assert AGENT_TIMEOUT_REASON_CODE in REASON_CODES


__all__ = [
    "AGENT_CRASH_REASON_CODE",
    "AGENT_LOAD_FAILED_REASON_CODE",
    "AGENT_TIMEOUT_REASON_CODE",
    "DEFAULT_AGENT_IMPORT_PATH",
    "DEFAULT_SESSION_NAME",
    "AgentDriver",
    "AgentLoadError",
    "AgentRunResult",
    "DriverContext",
    "load_agent_class",
]
