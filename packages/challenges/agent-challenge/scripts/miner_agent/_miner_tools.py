"""Tool specs + environment.exec bridge for the miner agent."""

from __future__ import annotations

import inspect
import json
from typing import Any

DEFAULT_CWD = "/app"
MAX_OUTPUT_CHARS = 40_000

SHELL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "shell_command",
        "description": (
            "Run a shell command inside the task container. "
            "Use workdir for the working directory. Prefer non-interactive flags. "
            "Do not ask the user questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "workdir": {
                    "type": "string",
                    "description": f"Working directory (default {DEFAULT_CWD})",
                },
                "timeout_sec": {
                    "type": "integer",
                    "description": "Command timeout in seconds (default 120)",
                },
            },
            "required": ["command"],
        },
    },
}

TOOLS: list[dict[str, Any]] = [SHELL_TOOL]


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    keep = limit // 2 - 40
    return f"{text[:keep]}\n\n[...truncated {len(text) - limit} chars...]\n\n{text[-keep:]}"


async def exec_command(
    environment: Any,
    command: str,
    *,
    cwd: str | None = None,
    timeout_sec: int = 120,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Call environment.exec with signature fallbacks; return (exit_code, output)."""
    exec_fn = environment.exec
    workdir = cwd or DEFAULT_CWD
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        ((command,), {"cwd": workdir, "timeout_sec": timeout_sec, "env": extra_env}),
        ((command,), {"cwd": workdir, "timeout_sec": timeout_sec}),
        ((command,), {"cwd": workdir, "timeout": timeout_sec}),
        ((command,), {"cwd": workdir}),
        ((command,), {}),
    ]
    last_type: TypeError | None = None
    value: Any = None
    for args, kwargs in attempts:
        # Drop None env to avoid surprising TypeErrors.
        clean = {k: v for k, v in kwargs.items() if v is not None}
        try:
            value = exec_fn(*args, **clean)
            if inspect.isawaitable(value):
                value = await value
            break
        except TypeError as exc:
            last_type = exc
            value = None
            continue
    else:
        raise last_type or TypeError("environment.exec could not be called")

    return _normalize(value)


def _normalize(value: Any) -> tuple[int, str]:
    if isinstance(value, str):
        return 0, truncate(value)
    if isinstance(value, dict):
        stdout = str(value.get("stdout") or value.get("output") or "")
        stderr = str(value.get("stderr") or "")
        code = value.get("return_code", value.get("exit_code", value.get("returncode", 0)))
        return int(code or 0), truncate(_join(stdout, stderr))
    stdout = getattr(value, "stdout", None)
    output = getattr(value, "output", None)
    stderr = getattr(value, "stderr", None)
    code = getattr(value, "return_code", None)
    if code is None:
        code = getattr(value, "exit_code", None)
    if code is None:
        code = getattr(value, "returncode", 0)
    text = _join(
        stdout if stdout is not None else output,
        stderr,
    )
    return int(code or 0), truncate(text)


def _join(stdout: Any, stderr: Any) -> str:
    out = "" if stdout is None else str(stdout)
    err = "" if stderr is None else str(stderr)
    if err:
        return f"{out}\n{err}" if out else err
    return out


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raw = str(raw)
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def dispatch_tool(
    environment: Any,
    name: str,
    arguments: dict[str, Any],
    *,
    extra_env: dict[str, str] | None = None,
) -> str:
    if name != "shell_command":
        return f"error: unknown tool {name!r}"
    command = str(arguments.get("command") or "").strip()
    if not command:
        return "error: empty command"
    workdir = str(arguments.get("workdir") or DEFAULT_CWD)
    try:
        timeout_sec = int(arguments.get("timeout_sec") or 120)
    except (TypeError, ValueError):
        timeout_sec = 120
    timeout_sec = max(1, min(timeout_sec, 600))
    try:
        code, output = await exec_command(
            environment,
            command,
            cwd=workdir,
            timeout_sec=timeout_sec,
            extra_env=extra_env,
        )
    except Exception as exc:  # noqa: BLE001 — tool result must never abort the loop
        return f"error: exec failed: {type(exc).__name__}: {exc}"
    return f"exit_code={code}\n{output}"
