"""Compute provider clients for the miner-funded GPU worker plane.

Provider-agnostic contract (:mod:`base.compute.provider`) plus concrete clients
(Lium; Targon added alongside) used to provision, inspect, and tear down the GPU
instances that run worker agents, under mandatory cost guardrails.
"""

from __future__ import annotations

from base.compute.lium import LIUM_API_BASE_URL, LiumClient, LiumError
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
]
