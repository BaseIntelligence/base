from __future__ import annotations

import asyncio
import logging

from platform_network.master.docker_orchestrator import (
    ChallengeSpec,
    DockerOrchestrator,
)
from platform_network.validator.registry_client import RegistryClient

logger = logging.getLogger(__name__)


class NormalValidatorRunner:
    def __init__(
        self,
        *,
        registry_client: RegistryClient,
        orchestrator: DockerOrchestrator,
        retry_seconds: int = 15,
    ) -> None:
        self.registry_client = registry_client
        self.orchestrator = orchestrator
        self.retry_seconds = retry_seconds

    async def run_once(self) -> None:
        registry = await self.registry_client.fetch_registry()
        for challenge in registry.challenges:
            spec = ChallengeSpec(
                slug=challenge.slug,
                image=challenge.image,
                version=challenge.version,
                env=challenge.env,
                required_capabilities=tuple(challenge.required_capabilities),
            )
            self.orchestrator.start_challenge(spec)

    async def run_forever(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception:
                logger.exception("registry sync failed; retrying")
            await asyncio.sleep(self.retry_seconds)
