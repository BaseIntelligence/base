"""Autonomous solve loop for one Terminal-Bench task."""

from __future__ import annotations

import time
from typing import Any, Protocol

from _miner_tools import TOOLS, dispatch_tool, parse_tool_arguments

SYSTEM_PROMPT = """You are an autonomous coding agent solving a Terminal-Bench task.
You run non-interactively inside a container. Never ask questions. Never wait for humans.

Rules:
- Explore with shell_command (pwd, ls, find, cat) before editing.
- Prefer small, correct changes. Verify with commands after edits.
- Do not read or modify hidden test harness files under /tests unless the instruction requires it.
- Do not commit secrets. Do not print API keys.
- When the task is fully done, respond with a short plain-text summary and NO tool calls.
- If stuck after several attempts, summarize what you tried and stop.
"""


class LLMClientProto(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...


async def run_solve_loop(
    *,
    instruction: str,
    environment: Any,
    llm: LLMClientProto,
    extra_env: dict[str, str] | None = None,
    max_steps: int = 40,
    task_timeout_sec: float = 900.0,
    command_env: dict[str, str] | None = None,
) -> str:
    """Drive LLM ↔ shell until completion, timeout, or step budget.

    Never raises for recoverable LLM/tool failures — returns a miss summary.
    """
    started = time.monotonic()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task instruction:\n{instruction}\n\n"
                "Start by inspecting the workspace, then solve the task completely."
            ),
        },
    ]

    last_summary = "miss: no progress"
    for step in range(max_steps):
        if time.monotonic() - started > task_timeout_sec:
            return f"miss: task timeout after {task_timeout_sec:.1f}s (step {step})"

        try:
            response = llm.chat(messages, tools=TOOLS)
        except Exception as exc:  # noqa: BLE001 — miss, do not abort suite
            return f"miss: llm error: {type(exc).__name__}: {exc}"

        content = str(response.get("content") or "")
        tool_calls = response.get("tool_calls")

        # Enforce wall-clock even when the LLM call itself overran the budget.
        if time.monotonic() - started > task_timeout_sec:
            return f"miss: task timeout after {task_timeout_sec:.1f}s (step {step})"

        if not tool_calls:
            if content.strip():
                return content.strip()
            return last_summary

        # Record assistant turn with tool_calls for the next round.
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content or None}
        assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        for call in tool_calls:
            if time.monotonic() - started > task_timeout_sec:
                return f"miss: task timeout after {task_timeout_sec:.1f}s (during tools)"
            call_id = str(call.get("id") or f"call_{step}")
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            args = parse_tool_arguments(fn.get("arguments"))
            if not name:
                result = "error: missing tool name"
            else:
                result = await dispatch_tool(
                    environment,
                    name,
                    args,
                    extra_env=command_env,
                )
            last_summary = f"step {step + 1}: {name}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                }
            )

        # Bound context growth: keep system + user + last N messages.
        if len(messages) > 60:
            head = messages[:2]
            tail = messages[-40:]
            messages = head + [{"role": "user", "content": "[earlier steps compacted]"}] + tail

    return f"miss: exceeded max_steps={max_steps}; last={last_summary}"
