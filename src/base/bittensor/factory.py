from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from base.bittensor.identity_cache import IdentityCache, self_declared_identity
from base.bittensor.metagraph_cache import MetagraphCache
from base.bittensor.weight_setter import WeightSetter
from base.config.settings import Settings


class BittensorDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class BittensorRuntime:
    metagraph_cache: MetagraphCache
    weight_setter: WeightSetter | None = None
    identity_cache: IdentityCache | None = None


def _load_bittensor() -> Any:
    try:
        return importlib.import_module("bittensor")
    except ImportError as exc:
        raise BittensorDependencyError(
            "Install the bittensor extra to submit weights: "
            "`pip install 'base[bittensor]'`."
        ) from exc


def _create_subtensor(settings: Settings) -> Any:
    bittensor = _load_bittensor()
    subtensor_kwargs = {}
    if settings.network.chain_endpoint:
        subtensor_kwargs["network"] = settings.network.chain_endpoint
    return bittensor.Subtensor(**subtensor_kwargs)


def _create_wallet(settings: Settings) -> Any:
    bittensor = _load_bittensor()
    wallet_kwargs = {
        "name": settings.network.wallet_name,
        "hotkey": settings.network.wallet_hotkey,
    }
    if settings.network.wallet_path:
        wallet_kwargs["path"] = settings.network.wallet_path
    return bittensor.Wallet(**wallet_kwargs)


def create_validator_keypair(settings: Settings) -> Any:
    """Return the validator hotkey keypair used to sign coordination requests."""

    return _create_wallet(settings).hotkey


def _keypair_from_uri_or_mnemonic(
    *, uri: str | None, mnemonic: str | None
) -> Any | None:
    """Build an sr25519 keypair from a dev URI or a mnemonic (None if neither)."""

    bittensor = _load_bittensor()
    if uri:
        return bittensor.Keypair.create_from_uri(uri)
    if mnemonic:
        return bittensor.Keypair.create_from_mnemonic(mnemonic)
    return None


def _wallet_keypair(
    *, name: str | None, hotkey: str | None, path: str | None
) -> Any | None:
    """Build a keypair from a bittensor wallet's hotkey (None if unconfigured)."""

    if not name and not hotkey and not path:
        return None
    bittensor = _load_bittensor()
    wallet_kwargs: dict[str, Any] = {}
    if name:
        wallet_kwargs["name"] = name
    if hotkey:
        wallet_kwargs["hotkey"] = hotkey
    if path:
        wallet_kwargs["path"] = path
    return bittensor.Wallet(**wallet_kwargs).hotkey


def create_worker_keypair(settings: Settings) -> Any:
    """Return the WORKER keypair signing coordination requests + proofs.

    Resolves an sr25519 dev URI, then a mnemonic, then a dedicated bittensor
    wallet, then falls back to ``network.wallet`` (the miner's own hotkey) so a
    single-wallet local deploy works without extra key material.
    """

    identity = settings.worker.identity
    keypair = _keypair_from_uri_or_mnemonic(
        uri=identity.key_uri, mnemonic=identity.key_mnemonic
    )
    if keypair is not None:
        return keypair
    keypair = _wallet_keypair(
        name=identity.wallet_name,
        hotkey=identity.wallet_hotkey,
        path=identity.wallet_path,
    )
    if keypair is not None:
        return keypair
    return _create_wallet(settings).hotkey


def create_worker_miner_keypair(settings: Settings) -> Any | None:
    """Return the MINER keypair that signs the enrollment binding, or None.

    Resolves an sr25519 dev URI, then a mnemonic, then a bittensor wallet. Returns
    None when no miner key is configured (the caller then requires a pre-signed
    binding, e.g. a pod that never holds the miner key).
    """

    identity = settings.worker.identity
    keypair = _keypair_from_uri_or_mnemonic(
        uri=identity.miner_key_uri, mnemonic=identity.miner_key_mnemonic
    )
    if keypair is not None:
        return keypair
    return _wallet_keypair(
        name=identity.miner_wallet_name,
        hotkey=identity.miner_wallet_hotkey,
        path=identity.miner_wallet_path,
    )


def _seed_mock_metagraph_cache(settings: Settings) -> MetagraphCache | None:
    """Seed a static ``MetagraphCache`` from ``network.mock_metagraph``.

    Returns ``None`` (the seam is OFF) when no static nodes are configured, so
    the live-metagraph path is unchanged. When configured the cache is marked
    ``static`` and carries no subtensor, so eligibility is served entirely from
    the configured set without ever constructing a live Subtensor.
    """

    nodes = settings.network.mock_metagraph
    if not nodes:
        return None
    cache = MetagraphCache(
        netuid=settings.network.netuid,
        ttl_seconds=settings.master.metagraph_cache_ttl_seconds,
        subtensor=None,
        static=True,
    )
    cache.update_from_metagraph(
        [node.hotkey for node in nodes],
        uids=[
            node.uid if node.uid is not None else index
            for index, node in enumerate(nodes)
        ],
        validator_permits=[node.validator_permit for node in nodes],
        stakes=[node.stake for node in nodes],
    )
    return cache


def _seed_mock_identity_cache(settings: Settings) -> IdentityCache:
    """Seed a static identity cache from ``mock_metagraph`` self-declared identity.

    Used by the no-chain mock deploy (architecture.md sec 7.2): each configured
    node's optional ``display_name``/``logo_url`` becomes an UNTRUSTED
    self-declared identity. The cache carries no subtensor, so resolution never
    reaches a chain. Nodes without a self-declared identity are simply absent
    (None-safe on lookup).
    """

    cache = IdentityCache(
        netuid=settings.network.netuid,
        ttl_seconds=settings.master.metagraph_cache_ttl_seconds,
        subtensor=None,
        static=True,
    )
    identities = {}
    for node in settings.network.mock_metagraph:
        identity = self_declared_identity(node.display_name, node.logo_url)
        if identity is not None:
            identities[node.hotkey] = identity
    cache.seed_static(identities)
    return cache


def create_bittensor_runtime(settings: Settings) -> BittensorRuntime:
    mock_cache = _seed_mock_metagraph_cache(settings)
    if mock_cache is not None:
        return BittensorRuntime(
            metagraph_cache=mock_cache,
            identity_cache=_seed_mock_identity_cache(settings),
        )
    subtensor = _create_subtensor(settings)
    return BittensorRuntime(
        metagraph_cache=MetagraphCache(
            netuid=settings.network.netuid,
            ttl_seconds=settings.master.metagraph_cache_ttl_seconds,
            subtensor=subtensor,
        ),
        identity_cache=IdentityCache(
            netuid=settings.network.netuid,
            ttl_seconds=settings.master.metagraph_cache_ttl_seconds,
            subtensor=subtensor,
        ),
    )


def create_bittensor_submit_runtime(settings: Settings) -> BittensorRuntime:
    subtensor = _create_subtensor(settings)
    wallet = _create_wallet(settings)
    return BittensorRuntime(
        metagraph_cache=MetagraphCache(
            netuid=settings.network.netuid,
            ttl_seconds=settings.master.metagraph_cache_ttl_seconds,
            subtensor=subtensor,
        ),
        identity_cache=IdentityCache(
            netuid=settings.network.netuid,
            ttl_seconds=settings.master.metagraph_cache_ttl_seconds,
            subtensor=subtensor,
        ),
        weight_setter=WeightSetter(
            subtensor=subtensor,
            wallet=wallet,
            netuid=settings.network.netuid,
        ),
    )
