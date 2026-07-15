from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from base.config.settings import Settings


def _parse_env_value(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith(("[", "{")):
        parsed = yaml.safe_load(stripped)
        if isinstance(parsed, list | dict):
            return parsed
    return value


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    node = data
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value


def _apply_env(data: dict[str, Any], prefix: str = "BASE_") -> dict[str, Any]:
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        raw = key[len(prefix) :].lower()
        _set_nested(data, raw.split("__"), _parse_env_value(value))
    return data


# Removed LLM-gateway configuration surfaces. Presence must fail closed rather than
# silently ignore or reacquire a provider route.
_REMOVED_GATEWAY_TOP_LEVEL_KEYS = frozenset({"gateway"})
_REMOVED_GATEWAY_ENV_PREFIXES = (
    "BASE_GATEWAY",
    "BASE_LLM_GATEWAY",
    "GATEWAY_TOKEN",
    "CENTRAL_GATEWAY_TOKEN",
    "PRISM_GATEWAY",
    "PRISM_LLM_GATEWAY",
    "CHALLENGE_LLM_GATEWAY",
)
_REMOVED_NESTED_GATEWAY_KEYS = frozenset(
    {
        "gateway_url",
        "gateway_token",
        "gateway_token_file",
        "llm_gateway_url",
        "llm_gateway_token",
        "llm_gateway_token_file",
    }
)

#: Operator env that is intentionally gateway-adjacent in name only: allowlist of
#: attestation-only / gateway-free agent-challenge image digests (not provider keys).
#: Read by :mod:`base.master.agent_challenge_compat`; must not trip removal reject.
_ALLOWED_GATEWAY_ADJACENT_ENV = frozenset(
    {
        "BASE_AGENT_CHALLENGE_GATEWAY_FREE_DIGESTS",
    }
)


def _reject_removed_gateway_config(data: dict[str, Any]) -> None:
    """Fail closed when legacy LLM-gateway configuration is supplied."""

    unknown: list[str] = []
    for key in data:
        if key in _REMOVED_GATEWAY_TOP_LEVEL_KEYS or key.startswith("gateway"):
            unknown.append(str(key))
    for section_name in ("validator", "worker", "master"):
        section = data.get(section_name)
        if not isinstance(section, dict):
            continue
        agent = section.get("agent")
        candidates = [section]
        if isinstance(agent, dict):
            candidates.append(agent)
        for candidate in candidates:
            for key in candidate:
                if key in _REMOVED_NESTED_GATEWAY_KEYS or "gateway" in str(key).lower():
                    unknown.append(f"{section_name}.{key}")
    for env_key in os.environ:
        upper = env_key.upper()
        if upper in _ALLOWED_GATEWAY_ADJACENT_ENV:
            continue
        if any(
            upper == prefix
            or upper.startswith(f"{prefix}_")
            or upper.startswith(f"{prefix}__")
            for prefix in _REMOVED_GATEWAY_ENV_PREFIXES
        ):
            unknown.append(env_key)
        if upper.startswith("BASE_") and "GATEWAY" in upper:
            unknown.append(env_key)
    if unknown:
        unique = ", ".join(sorted(set(unknown)))
        raise ValueError(
            "Unsupported removed LLM gateway configuration keys: "
            f"{unique}. The LLM gateway has been removed; do not set provider "
            "tokens, gateway URLs, or gateway blocks."
        )


def load_settings(path: str | Path | None = None) -> Settings:
    data: dict[str, Any] = {}
    if path:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file must contain a mapping: {config_path}")
        data.update(loaded)
    merged = _apply_env(data)
    _reject_removed_gateway_config(merged)
    return Settings.model_validate(merged)
