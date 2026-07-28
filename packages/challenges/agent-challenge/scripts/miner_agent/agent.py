"""Terminal-Bench miner agent entrypoint (ZIP root ``agent.py``).

Contract (own-runner / Harbor):
- This file MUST be named ``agent.py`` at the ZIP archive root.
- It MUST define a top-level ``class Agent``.
- Driver constructs ``Agent(logs_dir=, model_name=, **extra)`` (``extra`` may
  carry ``extra_env``), then ``await setup(environment)`` once before
  ``await run(instruction, environment, context)``.

Runtime credentials:
- ``OPENROUTER_API_KEY`` is read from process env, constructor ``extra_env``,
  or ``context.env``. Never hardcode, log, or write the key.
- Model defaults to ``x-ai/grok-4.5`` via OpenRouter. No Base LLM gateway.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Sibling modules ship inside the same ZIP / agent directory.
# Force this directory to sys.path[0] so flat ZIP-root imports cannot be
# shadowed by host packages named tools/loop/openrouter (namespace packages).
_HERE = Path(__file__).resolve().parent
_here_s = str(_HERE)
if sys.path and sys.path[0] == _here_s:
    pass
else:
    try:
        while _here_s in sys.path:
            sys.path.remove(_here_s)
    except ValueError:
        pass
    sys.path.insert(0, _here_s)

from _miner_loop import run_solve_loop  # noqa: E402
from _miner_openrouter import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    OpenRouterClient,
)


class MissingAPIKeyError(RuntimeError):
    """Raised when OPENROUTER_API_KEY is absent. Message never contains secrets."""

    def __init__(self) -> None:
        super().__init__(
            "OPENROUTER_API_KEY is not set. Provide it via the miner env gate "
            "(encrypted eval env). No API key was found in the process environment, "
            "constructor extra_env, or context.env."
        )


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, default: float | None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_int(value: str | None, default: int) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _merge_env(
    extra_env: dict[str, str] | None,
    context: Any | None,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    # Process env first (lowest priority for overrides we care about).
    for key, val in os.environ.items():
        if val is not None:
            merged[str(key)] = str(val)
    if extra_env:
        merged.update({str(k): str(v) for k, v in extra_env.items()})
    if context is not None:
        if isinstance(context, dict):
            raw = context.get("env") or {}
        else:
            raw = getattr(context, "env", None) or {}
        if isinstance(raw, dict):
            merged.update({str(k): str(v) for k, v in raw.items()})
    return merged


def _resolve_api_key(env_map: dict[str, str]) -> str | None:
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        val = env_map.get(name)
        if val and str(val).strip():
            return str(val).strip()
    return None


class Agent:
    """Harbor / own-runner compatible agent using OpenRouter + shell tools."""

    def __init__(
        self,
        *,
        logs_dir: Path | str | None = None,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        llm_client: Any | None = None,
        max_steps: int | None = None,
        task_timeout_sec: float | None = None,
        reasoning_effort: str | None = None,
        cost_limit_usd: float | None = None,
        **kwargs: Any,
    ) -> None:
        self._logs_dir = Path(logs_dir) if logs_dir is not None else None
        self._model_name = model_name or DEFAULT_MODEL
        self._extra_env: dict[str, str] = dict(extra_env or {})
        self._llm_client = llm_client
        self._max_steps = max_steps
        self._task_timeout_sec = task_timeout_sec
        self._reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
        self._cost_limit_usd = cost_limit_usd
        self._environment: Any | None = None
        # kwargs tolerated for harbor factory extras (unexpected_extra, etc.).
        self._kwargs = kwargs

    @staticmethod
    def name() -> str:
        return "BaseMinerAgent"

    @staticmethod
    def version() -> str:
        return "1.0.0"

    @staticmethod
    def import_path() -> str:
        return "agent:Agent"

    def to_agent_info(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "version": self.version(),
            "model_info": {"name": self._model_name, "provider": "openrouter"},
        }

    async def setup(self, environment: Any) -> None:
        """Called once before :meth:`run`."""
        self._environment = environment

    async def run(
        self,
        instruction: str,
        environment: Any,
        context: Any | None = None,
    ) -> str:
        """Solve one task. Recoverable failures return a miss summary string."""
        active = environment if environment is not None else self._environment
        if active is None:
            return "miss: no environment"

        env_map = _merge_env(self._extra_env, context)
        api_key = _resolve_api_key(env_map)
        if not api_key and self._llm_client is None:
            raise MissingAPIKeyError()

        max_steps = self._max_steps
        if max_steps is None:
            max_steps = _as_int(env_map.get("AGENT_MAX_STEPS"), 40)

        task_timeout = self._task_timeout_sec
        if task_timeout is None:
            task_timeout = _as_float(env_map.get("AGENT_TASK_TIMEOUT_SEC"), 900.0) or 900.0

        cost_limit = self._cost_limit_usd
        if cost_limit is None:
            cost_limit = _as_float(env_map.get("LLM_COST_LIMIT"), 5.0)

        reasoning = (
            env_map.get("REASONING_EFFORT")
            or env_map.get("AGENT_REASONING_EFFORT")
            or self._reasoning_effort
        )
        model = (
            env_map.get("LLM_MODEL")
            or env_map.get("OPENROUTER_MODEL")
            or self._model_name
            or DEFAULT_MODEL
        )

        llm = self._llm_client
        if llm is None:
            assert api_key is not None  # guarded above
            llm = OpenRouterClient(
                api_key=api_key,
                model=model,
                reasoning_effort=str(reasoning),
                cost_limit_usd=cost_limit,
            )

        # Do not forward API keys into shell env inside the task container.
        command_env = {
            k: v
            for k, v in self._extra_env.items()
            if not k.upper().endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))
        } or None

        try:
            return await run_solve_loop(
                instruction=instruction,
                environment=active,
                llm=llm,
                extra_env=self._extra_env,
                max_steps=int(max_steps),
                task_timeout_sec=float(task_timeout),
                command_env=command_env,
            )
        except MissingAPIKeyError:
            raise
        except Exception as exc:  # noqa: BLE001 — never abort the trial suite
            return f"miss: agent error: {type(exc).__name__}: {exc}"
