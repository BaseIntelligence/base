"""Canonical challenge health and version response types."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from .schemas import HealthCheck, HealthResponse, RuntimeStatusResponse, VersionResponse

ReadinessCheck = Callable[[], bool | Awaitable[bool]]
_PROBE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class ReadinessProbe:
    """One bounded, non-secret readiness signal owned by the server runtime."""

    name: str
    check: ReadinessCheck
    required: bool = True
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not _PROBE_NAME_RE.fullmatch(self.name):
            raise ValueError("readiness probe names must be lowercase identifiers")
        if self.timeout_seconds <= 0:
            raise ValueError("readiness probe timeout must be positive")


class HealthcheckDatabase(Protocol):
    async def healthcheck(self) -> bool: ...


async def evaluate_readiness(
    probes: Sequence[ReadinessProbe],
) -> tuple[HealthCheck, ...]:
    """Evaluate readiness without exposing exceptions or dependency coordinates."""

    checks: list[HealthCheck] = []
    for probe in probes:
        try:
            result = probe.check()
            healthy = (
                bool(await asyncio.wait_for(result, timeout=probe.timeout_seconds))
                if inspect.isawaitable(result)
                else bool(result)
            )
        except Exception:
            healthy = False
        checks.append(
            HealthCheck(
                name=probe.name,
                status="ok" if healthy else "unhealthy",
                required=probe.required,
            )
        )
    return tuple(checks)


def health_from_checks(
    *,
    slug: str,
    version: str,
    role: Literal["master", "validator", "challenge", "worker"],
    capabilities: tuple[str, ...],
    checks: tuple[HealthCheck, ...],
) -> HealthResponse:
    required_healthy = all(check.status == "ok" for check in checks if check.required)
    status: Literal["ok", "degraded", "unhealthy"] = (
        "unhealthy"
        if not required_healthy
        else "degraded"
        if any(check.status != "ok" for check in checks)
        else "ok"
    )
    return HealthResponse(
        status=status,
        slug=slug,
        version=version,
        role=role,
        ready=required_healthy,
        capabilities=capabilities,
        checks=checks,
    )


__all__ = [
    "HealthCheck",
    "HealthResponse",
    "HealthcheckDatabase",
    "ReadinessCheck",
    "ReadinessProbe",
    "RuntimeStatusResponse",
    "VersionResponse",
    "evaluate_readiness",
    "health_from_checks",
]
