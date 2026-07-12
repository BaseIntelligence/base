"""Deploy planning + orchestration for ``base worker deploy`` (architecture 3.2).

Pure, network-free helpers the ``base worker deploy`` CLI composes:

* :func:`require_provider_api_key` enforces the provider key env BEFORE any
  network call (a missing key is a typed, actionable refusal).
* :func:`select_worker_offer` / :func:`rank_worker_offers` pick a rentable offer
  under the ``--max-price`` cap, preferring executors whose ``gpu_count`` matches
  the request (a live-learned Lium constraint: renting a partial slice of a
  multi-GPU executor 400s), with next-cheapest fallback ordering.
* :func:`build_signed_binding` produces the miner-signed enrollment binding.
* :func:`build_worker_pod_env` builds the env handed to a provisioned pod's agent
  and NEVER includes a provider API key (architecture security invariant 1).

The provider API key lives only in the CLI/agent environment: it authenticates
the provider client's own HTTP calls and is never placed in a pod env, a binding,
or any master-bound request.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Mapping
from typing import Any

from base.compute.provider import Offer
from base.compute.worker_deployment import is_loopback_url, is_pinned_digest
from base.security.worker_auth import worker_binding_message
from base.validator.agent.signing import RequestSigner
from base.worker.runtime import WorkerBinding

logger = logging.getLogger(__name__)

LOCAL_PROVIDER = "local"
SUPPORTED_PROVIDERS: tuple[str, ...] = ("local", "lium", "targon")

#: Provider -> the environment variable that MUST hold the miner's provider key.
PROVIDER_KEY_ENV: dict[str, str] = {
    "lium": "LIUM_API_KEY",
    "targon": "TARGON_API_KEY",
}


class WorkerDeployError(RuntimeError):
    """A worker deploy could not be completed."""


class UnsupportedProviderError(WorkerDeployError):
    """The requested provider is not one of ``local``/``lium``/``targon``."""


class MissingProviderKeyError(WorkerDeployError):
    """A provider deploy was attempted without its API key env var set.

    Carries the missing ``env_var`` so the CLI can name it in an actionable
    error; raised BEFORE any provider or master network call.
    """

    def __init__(self, provider: str, env_var: str) -> None:
        self.provider = provider
        self.env_var = env_var
        super().__init__(
            f"provider '{provider}' requires the {env_var} environment variable; "
            "set it and retry. No provider or master call was made."
        )


class NoOfferWithinBudgetError(WorkerDeployError):
    """No rentable offer is at or below the requested ``--max-price`` cap."""


class WorkerImageNotConfiguredError(WorkerDeployError):
    """A provider deploy lacks an explicit, digest-pinned worker image.

    A provider (``lium``/``targon``) deploy MUST pin a PUBLICLY-pullable worker
    image; the baked-in placeholder is a PRIVATE-namespace GHCR image that fails
    Lium pod creation (``CREATION_FAILED``), so we refuse to silently pin it and
    surface an actionable error naming the config keys + env vars instead.
    """


def normalize_provider(provider: str) -> str:
    """Normalize + validate a ``--provider`` value to a supported provider."""

    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise UnsupportedProviderError(
            f"unsupported provider '{provider}'; expected one of "
            f"{', '.join(SUPPORTED_PROVIDERS)}"
        )
    return normalized


def require_provider_api_key(
    provider: str, *, environ: Mapping[str, str] | None = None
) -> str:
    """Return the provider's API key from the env, or raise a typed refusal.

    ``local`` needs no key. For ``lium``/``targon`` a missing/blank key raises
    :class:`MissingProviderKeyError` naming the exact env var, BEFORE any network
    call is made.
    """

    env = os.environ if environ is None else environ
    env_var = PROVIDER_KEY_ENV.get(provider)
    if env_var is None:
        raise UnsupportedProviderError(
            f"provider '{provider}' has no API key requirement"
        )
    value = env.get(env_var)
    if value is None or not value.strip():
        raise MissingProviderKeyError(provider, env_var)
    return value


def require_worker_image(
    *, image: str | None, image_digest: str | None, provider: str
) -> tuple[str, str]:
    """Return the ``(image, digest)`` a provider deploy must pin, or refuse.

    A ``lium``/``targon`` deploy MUST reference a PUBLICLY-pullable, digest-pinned
    worker image via ``worker.deploy.image`` + ``worker.deploy.image_digest`` (env
    ``BASE_WORKER__DEPLOY__IMAGE`` / ``BASE_WORKER__DEPLOY__IMAGE_DIGEST``). We do
    NOT fall back to a baked-in placeholder: a private-namespace GHCR image makes
    Lium pod creation fail with ``CREATION_FAILED``, so silently pinning it would
    provision a pod that never boots. Raises :class:`WorkerImageNotConfiguredError`
    (naming the config keys) when unset or when the digest is not ``sha256:<64 hex>``.
    See docs/miner/worker-plane.md ("Publishing the worker image").
    """

    resolved_image = image.strip() if image else ""
    resolved_digest = image_digest.strip() if image_digest else ""
    if not resolved_image or not resolved_digest:
        raise WorkerImageNotConfiguredError(
            f"provider '{provider}' deploy requires an explicit worker image: set "
            "worker.deploy.image + worker.deploy.image_digest (env "
            "BASE_WORKER__DEPLOY__IMAGE / BASE_WORKER__DEPLOY__IMAGE_DIGEST) to a "
            "PUBLICLY-pullable, digest-pinned image. The deploy refuses to pin a "
            "baked-in placeholder because a private-namespace GHCR image fails Lium "
            "pod creation (CREATION_FAILED). See docs/miner/worker-plane.md."
        )
    if not is_pinned_digest(resolved_digest):
        raise WorkerImageNotConfiguredError(
            "worker.deploy.image_digest must be a pinned digest of the form "
            f"'sha256:<64 hex>' (got {resolved_digest!r}); a mutable tag is not an "
            "immutable pin. See docs/miner/worker-plane.md."
        )
    return resolved_image, resolved_digest


def rank_worker_offers(
    offers: list[Offer], *, gpu_count: int, max_price: float | None = None
) -> list[Offer]:
    """Return offers within budget, best-first for worker deployment.

    Filters to ``price_per_hour <= max_price`` (when a cap is given), then orders
    so an executor whose ``gpu_count`` equals the request comes first (renting a
    partial slice of a multi-GPU executor 400s on Lium), breaking ties by ascending
    price then id. The full ordered list supports next-cheapest fallback when a
    rent loses an availability race.
    """

    eligible = [
        offer
        for offer in offers
        if max_price is None or offer.price_per_hour <= max_price
    ]
    return sorted(
        eligible,
        key=lambda offer: (
            offer.gpu_count != gpu_count,
            offer.price_per_hour,
            offer.id,
        ),
    )


def select_worker_offer(
    offers: list[Offer], *, gpu_count: int, max_price: float | None = None
) -> Offer:
    """Select the best in-budget offer, or raise :class:`NoOfferWithinBudgetError`."""

    ranked = rank_worker_offers(offers, gpu_count=gpu_count, max_price=max_price)
    if not ranked:
        cap = "unbounded" if max_price is None else f"{max_price}/GPU/hr"
        raise NoOfferWithinBudgetError(
            f"no rentable offer within budget ({cap}); nothing was provisioned"
        )
    return ranked[0]


async def plan_provider_deployment(
    client: Any, *, gpu_count: int, max_price: float | None
) -> Offer:
    """List provider offers under the cap and select the best (no rent issued)."""

    offers = await client.list_offers(max_price_per_hour=max_price)
    return select_worker_offer(offers, gpu_count=gpu_count, max_price=max_price)


def build_signed_binding(
    *,
    worker_pubkey: str,
    miner_signer: RequestSigner,
    nonce: str | None = None,
) -> WorkerBinding:
    """Sign the enrollment binding with the miner keypair (fresh nonce default).

    The message is the pinned ``worker-binding:{worker_pubkey}:{miner_hotkey}:
    {nonce}`` bytes; ``miner_signer`` is the MINER keypair (its hotkey must be on
    the metagraph). A fresh ``nonce`` per call makes a restart's re-enroll
    idempotent while a replay is rejected.
    """

    resolved_nonce = nonce or uuid.uuid4().hex
    message = worker_binding_message(
        worker_pubkey=worker_pubkey,
        miner_hotkey=miner_signer.hotkey,
        nonce=resolved_nonce,
    )
    return WorkerBinding(
        miner_hotkey=miner_signer.hotkey,
        signature=miner_signer.sign(message),
        nonce=resolved_nonce,
    )


def build_worker_pod_env(
    *,
    master_url: str,
    provider: str,
    binding: WorkerBinding,
    worker_key_uri: str | None = None,
    worker_key_mnemonic: str | None = None,
    broker_url: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the ``BASE_``-prefixed env for a provisioned pod's worker agent.

    Carries only non-secret coordination config plus the miner-signed binding
    (the pod never holds the miner key, so the binding is pre-signed here). It
    NEVER contains a provider API key: the key authenticates the CLI's provider
    client and must not leave the CLI/agent environment (architecture invariant 1).

    Loopback coordination URLs (master/broker) are OMITTED: Lium's edge WAF
    403s on any request body carrying a loopback URL and this env is baked into the
    WAF-sensitive template body, while the pod config already defaults these to
    loopback (the agent resolves them at runtime, reaching a local master via an SSH
    tunnel). A real (public) URL is passed through as an override.
    """

    env: dict[str, str] = {
        "BASE_COMPUTE__WORKER_PLANE_ENABLED": "true",
        "BASE_WORKER__DEPLOY__PROVIDER": provider,
        "BASE_WORKER__IDENTITY__MINER_HOTKEY": binding.miner_hotkey,
        "BASE_WORKER__IDENTITY__BINDING_SIGNATURE": binding.signature,
        "BASE_WORKER__IDENTITY__BINDING_NONCE": binding.nonce,
    }
    _set_non_loopback(env, "BASE_WORKER__AGENT__MASTER_URL", master_url)
    if worker_key_uri:
        env["BASE_WORKER__IDENTITY__KEY_URI"] = worker_key_uri
    if worker_key_mnemonic:
        env["BASE_WORKER__IDENTITY__KEY_MNEMONIC"] = worker_key_mnemonic
    if broker_url:
        _set_non_loopback(env, "BASE_WORKER__AGENT__BROKER_URL", broker_url)
    if extra:
        for key, value in extra.items():
            if key not in PROVIDER_KEY_ENV.values() and "GATEWAY" not in key.upper():
                env[key] = value
    return env


def _set_non_loopback(env: dict[str, str], key: str, value: str) -> None:
    """Set ``env[key] = value`` unless ``value`` is a WAF-triggering loopback URL."""

    if is_loopback_url(value):
        logger.info(
            "omitting loopback URL from worker pod env %s (Lium WAF 403s on loopback "
            "URLs; the agent resolves it at runtime from config)",
            key,
        )
        return
    env[key] = value


__all__ = [
    "LOCAL_PROVIDER",
    "PROVIDER_KEY_ENV",
    "SUPPORTED_PROVIDERS",
    "MissingProviderKeyError",
    "NoOfferWithinBudgetError",
    "UnsupportedProviderError",
    "WorkerDeployError",
    "WorkerImageNotConfiguredError",
    "build_signed_binding",
    "build_worker_pod_env",
    "normalize_provider",
    "plan_provider_deployment",
    "rank_worker_offers",
    "require_provider_api_key",
    "require_worker_image",
    "select_worker_offer",
]
