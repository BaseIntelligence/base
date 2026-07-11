"""Canonical, versioned Base challenge SDK public surface."""

from .app_factory import ChallengeDatabase, create_challenge_app
from .auth import build_internal_auth_dependency, load_shared_token
from .config import ChallengeSettings, DockerExecutorSettings
from .executors.docker import (
    DockerContainerInfo,
    DockerExecutor,
    DockerExecutorError,
    DockerLimits,
    DockerMount,
    DockerRunResult,
    DockerRunSpec,
)
from .roles import is_public_route, public_route
from .schemas import HealthResponse, VersionResponse, WeightsResponse
from .version import (
    API_VERSION,
    ARTIFACT_VERSION,
    DISTRIBUTION_NAME,
    RELEASE_ID,
    RELEASE_MANIFEST,
    SDK_CONTRACT_VERSION,
)

__all__ = [
    "API_VERSION",
    "ARTIFACT_VERSION",
    "ChallengeDatabase",
    "ChallengeSettings",
    "DISTRIBUTION_NAME",
    "DockerContainerInfo",
    "DockerExecutor",
    "DockerExecutorError",
    "DockerExecutorSettings",
    "DockerLimits",
    "DockerMount",
    "DockerRunResult",
    "DockerRunSpec",
    "HealthResponse",
    "RELEASE_ID",
    "RELEASE_MANIFEST",
    "SDK_CONTRACT_VERSION",
    "VersionResponse",
    "WeightsResponse",
    "build_internal_auth_dependency",
    "create_challenge_app",
    "is_public_route",
    "load_shared_token",
    "public_route",
]
