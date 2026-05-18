from __future__ import annotations

import asyncio
from typing import Any

import httpx

from platform_network.kubernetes.agent import KubernetesAgentClient
from platform_network.schemas.weights import (
    ChallengeWeightsResponse,
    ChallengeWeightsResult,
)


class ChallengeClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        retries: int = 3,
        kubernetes_target_registry: Any | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.kubernetes_target_registry = kubernetes_target_registry

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
            "X-Platform-Challenge-Slug": slug,
            "Accept": "application/json",
        }
        last_error = "unknown error"
        for attempt in range(max(self.retries, 1)):
            try:
                agent = self._agent_client(slug)
                if agent is not None:
                    response = await agent.forward_challenge_request(
                        slug=slug,
                        method="GET",
                        path="/internal/v1/get_weights",
                        headers=headers,
                    )
                else:
                    async with httpx.AsyncClient(
                        timeout=self.timeout_seconds
                    ) as client:
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

    def _agent_client(self, slug: str) -> KubernetesAgentClient | None:
        registry = self.kubernetes_target_registry
        if registry is None or not hasattr(registry, "get_assignment"):
            return None
        target_id = registry.get_assignment(slug)
        if not target_id:
            return None
        target = registry.get(target_id)
        if target.mode != "agent" or not target.agent_url:
            return None
        token = registry.get_agent_token(target.id)
        if not token:
            return None
        return KubernetesAgentClient(
            target_id=target.id,
            base_url=target.agent_url,
            token=token,
            timeout_seconds=target.timeout_seconds,
            verify_tls=target.verify_tls,
        )
