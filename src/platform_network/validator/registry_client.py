from __future__ import annotations

import httpx

from platform_network.schemas.challenge import RegistryResponse


class RegistryClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def fetch_registry(self) -> RegistryResponse:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(f"{self.base_url}/v1/registry")
            response.raise_for_status()
            return RegistryResponse.model_validate(response.json())
