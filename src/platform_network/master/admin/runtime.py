from __future__ import annotations

from typing import Protocol

from platform_network.schemas.challenge import RuntimeOperationResponse


class RuntimeController(Protocol):
    async def pull(self, slug: str) -> RuntimeOperationResponse: ...
    async def restart(self, slug: str) -> RuntimeOperationResponse: ...
    async def status(self, slug: str) -> RuntimeOperationResponse: ...


class NoopRuntimeController:
    async def pull(self, slug: str) -> RuntimeOperationResponse:
        return RuntimeOperationResponse(
            slug=slug,
            operation="pull",
            status="not_configured",
            detail="Runtime controller is not configured.",
        )

    async def restart(self, slug: str) -> RuntimeOperationResponse:
        return RuntimeOperationResponse(
            slug=slug,
            operation="restart",
            status="not_configured",
            detail="Runtime controller is not configured.",
        )

    async def status(self, slug: str) -> RuntimeOperationResponse:
        return RuntimeOperationResponse(
            slug=slug,
            operation="status",
            status="not_configured",
            detail="Runtime controller is not configured.",
        )
