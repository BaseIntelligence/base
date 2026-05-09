from __future__ import annotations

from dataclasses import dataclass

from platform_network.master.docker_orchestrator import ChallengeResources
from platform_network.schemas.gpu_server import GpuServerRecord


@dataclass(frozen=True)
class CapabilityDecision:
    can_run: bool
    reason: str | None = None


class ResourceCapabilityChecker:
    def __init__(self, gpu_servers: dict[str, GpuServerRecord] | None = None) -> None:
        self.gpu_servers = gpu_servers or {}

    def check(self, resources: ChallengeResources) -> CapabilityDecision:
        if resources.gpu_count is not None and resources.gpu_count <= 0:
            return CapabilityDecision(False, "invalid_gpu_count")
        if not resources.gpu_server:
            return CapabilityDecision(True)
        server = self.gpu_servers.get(resources.gpu_server)
        if server is None:
            return CapabilityDecision(False, "gpu_server_unknown")
        if not server.enabled:
            return CapabilityDecision(False, "gpu_server_disabled")
        required_gpus = resources.gpu_count or 1
        if server.min_gpu_count < required_gpus:
            return CapabilityDecision(False, "gpu_capacity_insufficient")
        return CapabilityDecision(True)
