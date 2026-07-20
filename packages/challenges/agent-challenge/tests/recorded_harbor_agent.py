"""Recorded-deterministic parity agent (harbor side) — Task 23.

Thin ``harbor.agents.base.BaseAgent`` adapter used ONLY inside the harbor
subprocess via ``harbor run --agent-import-path recorded_harbor_agent:Agent``.
It imports the SHARED transcript + driver loop from the harbor-free
:mod:`recorded_parity_agent`, so the stock-harbor execution path and the
own-runner execution path drive byte-identical command sequences against the
task container.

This module is the ONLY one that imports harbor. It is never imported by the
committed pytest suite (which stays harbor-free for Task 24); harbor loads it by
module name, so the directory containing it must be on ``PYTHONPATH`` /
``sys.path`` when ``harbor run`` is invoked (the parity harness sets this).

Construction parity: harbor's factory calls
``Agent(logs_dir=, model_name=, extra_env=, **kwargs)`` and delivers ``--ae``
vars through ``extra_env`` (``harbor/agents/factory.py``
``create_agent_from_import_path``). ``__init__`` captures ``extra_env`` and
forwards the rest to ``BaseAgent.__init__`` unchanged.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.base import BaseAgent
from recorded_parity_agent import run_transcript, select_mode


class Agent(BaseAgent):
    """Harbor ``BaseAgent`` adapter delegating to the shared recorded transcript."""

    def __init__(self, *args: Any, extra_env: dict[str, str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # harbor delivers --ae/--agent-env vars here (NOT into os.environ for
        # import-path agents); the shared loop also honors an os.environ fallback.
        self._extra_env: dict[str, str] = dict(extra_env or {})

    @staticmethod
    def name() -> str:
        return "recorded-parity"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: Any) -> None:
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        mode = select_mode(self._extra_env)
        await run_transcript(environment, mode=mode, extra_env=self._extra_env)


__all__ = ["Agent"]
