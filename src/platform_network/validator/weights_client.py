from __future__ import annotations

import asyncio

import httpx

from platform_network.schemas.weights import MasterWeightsResponse


class WeightsClient:
    def __init__(
        self, base_url: str, *, timeout_seconds: float = 15.0, retries: int = 3
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    async def fetch_latest(self) -> MasterWeightsResponse:
        last_error: Exception | None = None
        for attempt in range(max(1, self.retries + 1)):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(f"{self.base_url}/v1/weights/latest")
                    response.raise_for_status()
                    return MasterWeightsResponse.model_validate(response.json())
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    await asyncio.sleep(0)
        raise last_error or RuntimeError("weights fetch failed")
