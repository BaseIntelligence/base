from __future__ import annotations

from typing import Protocol

from base.schemas.challenge import RuntimeOperationResponse


class RuntimeController(Protocol):
    async def pull(self, slug: str) -> RuntimeOperationResponse: ...
    async def restart(self, slug: str) -> RuntimeOperationResponse: ...
    async def status(self, slug: str) -> RuntimeOperationResponse: ...
