from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from platform_network.bittensor.metagraph_cache import MetagraphCache
from platform_network.bittensor.weight_setter import WeightSetter
from platform_network.config.settings import Settings


class BittensorDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class BittensorRuntime:
    metagraph_cache: MetagraphCache
    weight_setter: WeightSetter


def _load_bittensor(*, required: bool) -> Any | None:
    try:
        return importlib.import_module("bittensor")
    except ImportError as exc:
        if required:
            raise BittensorDependencyError(
                "Install the bittensor extra to submit weights: "
                "`pip install 'platform-network[bittensor]'`."
            ) from exc
        return None


def create_bittensor_runtime(
    settings: Settings, *, dry_run: bool = True
) -> BittensorRuntime:
    bittensor = _load_bittensor(required=not dry_run)
    subtensor = None
    wallet = None
    if bittensor is not None:
        subtensor_kwargs = {}
        if settings.network.chain_endpoint:
            subtensor_kwargs["network"] = settings.network.chain_endpoint
        subtensor = bittensor.Subtensor(**subtensor_kwargs)
        if not dry_run:
            wallet = bittensor.Wallet(
                name=settings.network.wallet_name,
                hotkey=settings.network.wallet_hotkey,
            )

    return BittensorRuntime(
        metagraph_cache=MetagraphCache(
            netuid=settings.network.netuid,
            ttl_seconds=settings.master.metagraph_cache_ttl_seconds,
            subtensor=subtensor,
        ),
        weight_setter=WeightSetter(
            subtensor=None if dry_run else subtensor,
            wallet=wallet,
            netuid=settings.network.netuid,
        ),
    )
