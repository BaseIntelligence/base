"""Derive in-instance sidecar HTTP base URL from a raw Lium pod payload.

Lium ``GET /pods/{id}`` does not expose a generic HTTP endpoint for custom
container ports. Callers must compose ``executor.executor_ip_address`` with the
published host port from ``ports_mapping`` for the sidecar's internal port.

This module is pure and fail-closed: it never falls back to ``jupyter_url``,
never parses ``ssh_connect_cmd``, never scans/guesses ports, and never defaults
to the internal port when unmapped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

_MIN_PORT: Final[int] = 1
_MAX_PORT: Final[int] = 65535
_HOST_PORT_KEYS: Final[tuple[str, ...]] = (
    "HostPort",
    "host_port",
    "hostPort",
    "port",
    "Port",
)


class SidecarEndpointMissReason(StrEnum):
    """Machine-consumed miss codes for sidecar base URL resolution."""

    SIDECAR_PORT_UNPUBLISHED = "sidecar_port_unpublished"
    POD_ENDPOINT_MISSING = "pod_endpoint_missing"
    PORTS_MAPPING_UNPARSEABLE = "ports_mapping_unparseable"


@dataclass(frozen=True, slots=True)
class SidecarEndpointHit:
    """Resolved sidecar HTTP base URL (no trailing slash)."""

    base_url: str

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class SidecarEndpointMiss:
    """Fail-closed resolution outcome."""

    reason: SidecarEndpointMissReason

    def __bool__(self) -> bool:
        return False


SidecarEndpointResult = SidecarEndpointHit | SidecarEndpointMiss


def resolve_sidecar_base_url(
    pod_raw: Mapping[str, Any],
    *,
    internal_port: int,
    scheme: str = "http",
) -> SidecarEndpointResult:
    """Resolve ``{scheme}://{host}:{external_port}`` for the sidecar.

    Parameters
    ----------
    pod_raw:
        Raw Lium ``PodDetailResponse`` mapping (e.g. ``LiumPodRead.raw``).
    internal_port:
        Container-internal sidecar listen port to look up in ``ports_mapping``.
    scheme:
        URL scheme; defaults to ``http``.
    """
    host = _extract_host(pod_raw)
    if host is None:
        return SidecarEndpointMiss(SidecarEndpointMissReason.POD_ENDPOINT_MISSING)

    mapping_result = _normalize_ports_mapping(pod_raw.get("ports_mapping"))
    if isinstance(mapping_result, SidecarEndpointMissReason):
        return SidecarEndpointMiss(mapping_result)

    external = _lookup_external_port(mapping_result, internal_port)
    if external is None:
        return SidecarEndpointMiss(SidecarEndpointMissReason.SIDECAR_PORT_UNPUBLISHED)

    return SidecarEndpointHit(base_url=f"{scheme}://{_format_host(host)}:{external}")


def _extract_host(pod_raw: Mapping[str, Any]) -> str | None:
    executor = pod_raw.get("executor")
    if not isinstance(executor, Mapping):
        return None
    ip = executor.get("executor_ip_address")
    if not isinstance(ip, str):
        return None
    host = ip.strip()
    if not host:
        return None
    return host


def _normalize_ports_mapping(
    raw: object,
) -> dict[Any, Any] | SidecarEndpointMissReason:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return SidecarEndpointMissReason.PORTS_MAPPING_UNPARSEABLE
        try:
            parsed: object = json.loads(text)
        except json.JSONDecodeError:
            return SidecarEndpointMissReason.PORTS_MAPPING_UNPARSEABLE
        if not isinstance(parsed, dict):
            return SidecarEndpointMissReason.PORTS_MAPPING_UNPARSEABLE
        return parsed
    return SidecarEndpointMissReason.PORTS_MAPPING_UNPARSEABLE


def _lookup_external_port(
    mapping: Mapping[Any, Any],
    internal_port: int,
) -> int | None:
    value = _mapping_get(mapping, internal_port)
    if value is None:
        return None
    return _coerce_external_port(value)


def _mapping_get(mapping: Mapping[Any, Any], internal_port: int) -> object | None:
    if internal_port in mapping:
        return mapping[internal_port]
    as_str = str(internal_port)
    if as_str in mapping:
        return mapping[as_str]
    return None


def _coerce_external_port(value: object) -> int | None:
    direct = _as_port_number(value)
    if direct is not None:
        return direct
    if isinstance(value, Mapping):
        return _port_from_hostport_dict(value)
    if isinstance(value, list):
        for item in value:
            port = _coerce_external_port(item)
            if port is not None:
                return port
        return None
    return None


def _port_from_hostport_dict(value: Mapping[Any, Any]) -> int | None:
    for key in _HOST_PORT_KEYS:
        if key in value:
            port = _as_port_number(value[key])
            if port is not None:
                return port
    return None


def _as_port_number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if _MIN_PORT <= value <= _MAX_PORT else None
    if isinstance(value, str):
        text = value.strip()
        if not text or not text.isdigit():
            return None
        port = int(text)
        return port if _MIN_PORT <= port <= _MAX_PORT else None
    return None


def _format_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


__all__ = [
    "SidecarEndpointHit",
    "SidecarEndpointMiss",
    "SidecarEndpointMissReason",
    "SidecarEndpointResult",
    "resolve_sidecar_base_url",
]
