"""Recorded-deterministic parity agent (harbor-free side) — Task 23.

This module is the SHARED, harbor-free core of the agent-driving parity probe.
It owns:

* :data:`TRANSCRIPTS` -- a FIXED, recorded sequence of ``environment.exec``
  commands per mode (NO LLM, NO network). The same dict is imported by the
  harbor-side adapter (:mod:`recorded_harbor_agent`) so BOTH execution paths
  drive byte-identical command sequences.
* :func:`run_transcript` -- the shared driver loop, so the own-runner agent and
  the harbor adapter execute the transcript through the exact same code.
* :class:`RecordedAgent` -- the own-runner agent class. It honors the own-runner
  driver construction contract ``agent_cls(logs_dir=, model_name=, **kwargs)``
  (Task 13 ``AgentDriver.drive``) and the harbor SDK run shape
  (``setup`` then ``run(instruction, environment, context)``).

Deliberately imports NOTHING from harbor: it must keep importing cleanly after
harbor is uninstalled (Task 24). The harbor ``BaseAgent`` subclass lives in the
sibling :mod:`recorded_harbor_agent`, which is loaded ONLY inside the harbor
subprocess via ``--agent-import-path`` and which delegates to this module.

Mode selection mirrors the harbor contract: ``--ae RECORDED_MODE=...`` arrives in
the constructor ``extra_env`` (harbor's factory passes ``extra_env`` into the
agent ``__init__``; see ``harbor/agents/factory.py``), with an ``os.environ``
fallback for completeness. The default ``probe`` transcript is intentionally
container-agnostic and harmless (exit-code 0 in every subset task image), so the
verifier scores it ``resolved=0`` on BOTH paths -- parity holds regardless of the
verifier outcome, which is exactly what this gate proves.
"""

from __future__ import annotations

import os
from typing import Any

#: Default mode when neither ``extra_env`` nor ``os.environ`` selects one.
DEFAULT_MODE = "probe"

#: Recorded transcripts: ``mode -> [(command, exec_kwargs), ...]``. Each command
#: is run via ``environment.exec(command, **exec_kwargs)``. The ``probe`` mode is
#: container-agnostic (rc=0 in any task image) and uses a RELATIVE cwd default so
#: it exercises harbor's "cwd defaults to the task workdir" behaviour identically
#: on both paths.
TRANSCRIPTS: dict[str, list[tuple[str, dict[str, Any]]]] = {
    "probe": [
        ("echo recorded-parity-probe", {"timeout_sec": 30}),
        ("pwd", {"timeout_sec": 30}),
    ],
}


def select_mode(extra_env: dict[str, str] | None) -> str:
    """Resolve the recorded mode: ``extra_env`` wins, then ``os.environ``, then default.

    Mirrors harbor's import-path agent contract where ``--ae KEY=VALUE`` vars are
    delivered through the constructor ``extra_env`` (NOT ``os.environ``); the
    ``os.environ`` fallback keeps the agent usable when driven directly.
    """
    env = extra_env or {}
    return env.get("RECORDED_MODE") or os.environ.get("RECORDED_MODE", DEFAULT_MODE)


async def run_transcript(
    environment: Any,
    *,
    mode: str,
    extra_env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Execute the recorded ``mode`` transcript through ``environment.exec``.

    Returns one log entry per command (command + return_code + stdout + stderr),
    and mirrors each to stdout for log capture. This is the SINGLE place the
    transcript is driven, so the own-runner agent and the harbor adapter run an
    identical command sequence by construction.
    """
    transcript = TRANSCRIPTS[mode]
    log: list[dict[str, Any]] = []
    for command, kwargs in transcript:
        result = await environment.exec(command, **kwargs)
        entry = {
            "command": command,
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        log.append(entry)
        print(
            f"[recorded:{mode}] cmd={command!r} rc={result.return_code} "
            f"stdout={result.stdout!r} stderr={result.stderr!r}",
            flush=True,
        )
    return log


class RecordedAgent:
    """Own-runner recorded-deterministic agent (harbor-free).

    Constructed by the Task-13 driver as ``RecordedAgent(logs_dir=, model_name=,
    extra_env=, **kwargs)``; ``extra_env`` carries the trial's ``agent_env`` (the
    own-runner driver forwards ``agent_env`` as ``extra_env``), matching how
    harbor delivers ``--ae`` vars. ``setup`` is a no-op; ``run`` drives the
    recorded transcript via :func:`run_transcript`.
    """

    def __init__(
        self,
        *,
        logs_dir: Any | None = None,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._logs_dir = logs_dir
        self._model_name = model_name
        self._extra_env: dict[str, str] = dict(extra_env or {})

    @staticmethod
    def name() -> str:
        return "recorded-parity"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: Any) -> None:
        """No-op, matching harbor's trivial-agent ``setup``."""
        return None

    async def run(self, instruction: str, environment: Any, context: Any | None = None) -> str:
        """Drive the recorded transcript and return a short summary string."""
        mode = select_mode(self._extra_env)
        log = await run_transcript(environment, mode=mode, extra_env=self._extra_env)
        return f"recorded-parity ran mode={mode} ({len(log)} command(s))"


__all__ = [
    "DEFAULT_MODE",
    "TRANSCRIPTS",
    "RecordedAgent",
    "run_transcript",
    "select_mode",
]
