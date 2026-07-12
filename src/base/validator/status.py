from __future__ import annotations

from base.challenge_sdk.health import health_from_checks
from base.challenge_sdk.roles import Capability, Role, capabilities_for_role
from base.challenge_sdk.schemas import (
    HealthCheck,
    HealthResponse,
    RuntimeStatusResponse,
    VersionResponse,
)
from base.challenge_sdk.version import (
    API_VERSION,
    ARTIFACT_VERSION,
    DISTRIBUTION_NAME,
    RELEASE_ID,
    SDK_CONTRACT_VERSION,
)


def validator_capabilities(*, submission_enabled: bool) -> tuple[str, ...]:
    capabilities = capabilities_for_role(Role.VALIDATOR)
    if submission_enabled:
        return capabilities
    return tuple(
        token for token in capabilities if token != Capability.VALIDATOR_OWN_SET_WEIGHTS
    )


def validator_runtime_status(
    *,
    master_health: HealthResponse | None,
    master_version: VersionResponse | None,
    submission_enabled: bool,
) -> RuntimeStatusResponse:
    master_compatible = (
        master_health is not None
        and master_health.role == Role.MASTER
        and master_health.ready
        and master_version is not None
        and master_version.role == Role.MASTER
        and master_version.api_version == API_VERSION
        and master_version.sdk_contract_version == SDK_CONTRACT_VERSION
    )
    capabilities = validator_capabilities(submission_enabled=submission_enabled)
    health = health_from_checks(
        slug="base-validator",
        version=ARTIFACT_VERSION,
        role=Role.VALIDATOR.value,
        capabilities=capabilities,
        checks=(
            HealthCheck(
                name="master",
                status="ok" if master_compatible else "unhealthy",
                required=True,
            ),
        ),
    )
    version = VersionResponse(
        distribution_name=DISTRIBUTION_NAME,
        artifact_version=ARTIFACT_VERSION,
        release_id=RELEASE_ID,
        api_version=API_VERSION,
        challenge_slug=None,
        challenge_version=ARTIFACT_VERSION,
        sdk_contract_version=SDK_CONTRACT_VERSION,
        sdk_version=SDK_CONTRACT_VERSION,
        role=Role.VALIDATOR.value,
        capabilities=capabilities,
    )
    return RuntimeStatusResponse(health=health, version=version)


__all__ = [
    "validator_capabilities",
    "validator_runtime_status",
]
