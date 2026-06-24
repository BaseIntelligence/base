from __future__ import annotations

import asyncio

import httpx

from base.schemas.weights import (
    ChallengeWeightsResponse,
    ChallengeWeightsResult,
)


class ChallengeClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        retries: int = 3,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    async def get_weights(
        self,
        *,
        slug: str,
        base_url: str,
        token: str,
        emission_percent: float,
    ) -> ChallengeWeightsResult:
        url = f"{base_url.rstrip('/')}/internal/v1/get_weights"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Base-Challenge-Slug": slug,
            "Accept": "application/json",
        }
        last_error = "unknown error"
        for attempt in range(max(self.retries, 1)):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                payload = ChallengeWeightsResponse.model_validate(response.json())
                return ChallengeWeightsResult(
                    slug=slug,
                    emission_percent=emission_percent,
                    weights=payload.weights,
                    ok=True,
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt + 1 < max(self.retries, 1):
                    await asyncio.sleep(0.5 * (attempt + 1))
        return ChallengeWeightsResult(
            slug=slug,
            emission_percent=emission_percent,
            weights={},
            ok=False,
            error=last_error,
        )
