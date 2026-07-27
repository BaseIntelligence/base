"""Compute provider clients for the miner-funded GPU worker plane.

Provider-agnostic contract (:mod:`base.compute.provider`) plus concrete clients
(Lium; Targon added alongside) used to provision, inspect, and tear down the GPU
instances that run worker agents, under mandatory cost guardrails.
"""

from __future__ import annotations

from base.compute.attestation_nonce import (
    AttestationNonceService,
    NonceBinding,
    NonceConsumeHit,
    NonceConsumeMiss,
    NonceConsumeReason,
    NonceRecord,
    NonceSnapshot,
)
from base.compute.constation_corroboration import (
    CorroborationOutcome,
    evaluate_corroboration,
)
from base.compute.constation_custody import (
    LiumKeyCustody,
    generate_custody_master_key,
)
from base.compute.constation_pod import (
    assert_pod_bound,
    extract_miner_hotkey,
    pod_is_running,
)
from base.compute.constation_poller import (
    ContinuousConstationPoller,
    PollerConfig,
    PollerResult,
)
from base.compute.constation_runner import (
    ConstationRunner,
    ConstationRunRequest,
)
from base.compute.constation_sidecar_client import (
    HttpSidecarAttestor,
    SidecarAttestHit,
    SidecarAttestMiss,
    SidecarAttestResult,
)
from base.compute.constation_sidecar_endpoint import (
    SidecarEndpointHit,
    SidecarEndpointMiss,
    SidecarEndpointMissReason,
    resolve_sidecar_base_url,
)
from base.compute.constation_triangle import (
    DigestTriangleResult,
    evaluate_digest_triangle,
)
from base.compute.constation_types import (
    ConstationFailCode,
    ConstationRunRecord,
    ConstationVerdict,
    CorroborationStatus,
    FaultClass,
    PollSample,
)
from base.compute.digest_allowlist import (
    AllowlistHit,
    AllowlistMiss,
    AllowlistMissReason,
    DigestAllowlist,
    DigestRecord,
    ImageVariant,
)
from base.compute.lium import (
    LIUM_API_BASE_URL,
    LiumAuthError,
    LiumClient,
    LiumError,
    LiumNotFoundError,
    LiumPodRead,
    LiumRateLimitError,
    LiumTemplateDigestMismatchError,
    LiumTemplateRead,
)
from base.compute.provider import (
    CostGuardrailError,
    Instance,
    InstanceSpec,
    Offer,
    ProviderClient,
    ProviderError,
)
from base.compute.targon import (
    TARGON_API_BASE_URL,
    BalanceUnavailableError,
    InsufficientCreditsError,
    TargonClient,
    TargonError,
)
from base.compute.worker_deployment import (
    WORKER_IMAGE,
    WORKER_IMAGE_DIGEST,
    WORKER_IMAGE_TAG,
    WORKER_STARTUP_COMMANDS,
    build_lium_worker_template,
    build_targon_worker_app,
    is_loopback_url,
    is_metachar_free,
    is_pinned_digest,
    pinned_image_reference,
)

__all__ = [
    "AllowlistHit",
    "AttestationNonceService",
    "NonceBinding",
    "NonceConsumeHit",
    "NonceConsumeMiss",
    "NonceConsumeReason",
    "NonceRecord",
    "NonceSnapshot",
    "AllowlistMiss",
    "AllowlistMissReason",
    "DigestAllowlist",
    "DigestRecord",
    "ImageVariant",
    "LIUM_API_BASE_URL",
    "TARGON_API_BASE_URL",
    "WORKER_IMAGE",
    "WORKER_IMAGE_DIGEST",
    "WORKER_IMAGE_TAG",
    "WORKER_STARTUP_COMMANDS",
    "BalanceUnavailableError",
    "CostGuardrailError",
    "Instance",
    "InstanceSpec",
    "InsufficientCreditsError",
    "LiumClient",
    "LiumError",
    "LiumTemplateRead",
    "LiumTemplateDigestMismatchError",
    "LiumRateLimitError",
    "LiumPodRead",
    "LiumNotFoundError",
    "LiumAuthError",
    "Offer",
    "ProviderClient",
    "ProviderError",
    "TargonClient",
    "TargonError",
    "build_lium_worker_template",
    "build_targon_worker_app",
    "is_loopback_url",
    "is_metachar_free",
    "is_pinned_digest",
    "pinned_image_reference",
    "ConstationFailCode",
    "ConstationRunRecord",
    "ConstationRunRequest",
    "ConstationRunner",
    "ConstationVerdict",
    "ContinuousConstationPoller",
    "CorroborationOutcome",
    "CorroborationStatus",
    "FaultClass",
    "LiumKeyCustody",
    "PollSample",
    "PollerConfig",
    "PollerResult",
    "evaluate_corroboration",
    "DigestTriangleResult",
    "evaluate_digest_triangle",
    "generate_custody_master_key",
    "HttpSidecarAttestor",
    "SidecarAttestHit",
    "SidecarAttestMiss",
    "SidecarAttestResult",
    "SidecarEndpointHit",
    "SidecarEndpointMiss",
    "SidecarEndpointMissReason",
    "resolve_sidecar_base_url",
    "assert_pod_bound",
    "extract_miner_hotkey",
    "pod_is_running",
]
