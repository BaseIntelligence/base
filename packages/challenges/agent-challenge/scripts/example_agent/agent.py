"""Minimal valid Agent Challenge submission entrypoint (no-LLM).

Contract:
- This file MUST be named ``agent.py`` and live at the ZIP archive root.
- It MUST define a top-level ``class Agent``.
- Production validators (and the own-runner driver) import ``agent:Agent`` and
  run it inside the Terminal-Bench benchmark workspace.

The own-runner driver constructs the agent as
``Agent(logs_dir=, model_name=, **extra)`` (``extra`` may carry ``extra_env``),
then calls ``setup`` once before ``run``. This example satisfies that lifecycle
and proves in-container code execution via ``environment.exec`` while making NO
model calls, so it can exercise the end-to-end submission pipeline (signing,
upload, analyzer, env gate, terminal-bench launch + eval) without any provider
configuration.

A real miner builds this from ``BaseIntelligence/baseagent``. Coded agents call
the platform LLM gateway using the env the validator injects at launch; the
platform selects the provider and model, so the submission embeds no provider
API key, base URL, or model name::

    # Injected by the validator at launch (do not hardcode in the submission):
    #   BASE_LLM_GATEWAY_URL  -> platform LLM gateway base URL
    #   BASE_GATEWAY_TOKEN    -> per-assignment scoped token (auth)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Marker echoed inside the task container to prove the agent's code executed.
EXECUTION_MARKER = "agent-challenge-ok"


class Agent:
    """A no-op, no-LLM agent that satisfies the own-runner driver contract."""

    def __init__(
        self,
        *,
        logs_dir: Path | str | None = None,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._logs_dir = Path(logs_dir) if logs_dir is not None else None
        self._model_name = model_name
        self._extra_env: dict[str, str] = dict(extra_env or {})

    async def setup(self, environment: Any) -> None:
        """Called once before :meth:`run`; nothing to prepare for a no-op agent."""
        return None

    async def run(self, instruction: str, environment: Any, context: Any | None = None) -> str:
        """Prove in-container execution via ``environment.exec`` and return the marker.

        Makes no model calls. The single exec echoes a marker (and writes it to a
        file) so the run leaves observable evidence the submitted code ran inside
        the task container; the returned string becomes the agent's summary.
        """
        result = await environment.exec(
            f"echo {EXECUTION_MARKER} | tee /tmp/{EXECUTION_MARKER}",
            env=self._extra_env or None,
        )
        return (result.stdout or "").strip() or EXECUTION_MARKER
