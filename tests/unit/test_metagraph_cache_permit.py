from __future__ import annotations

from types import SimpleNamespace

from base.bittensor.metagraph_cache import MetagraphCache


def test_update_from_hotkeys_defaults_permit_and_stake() -> None:
    cache = MetagraphCache(netuid=1)
    assert cache.update_from_hotkeys(["a", "b"]) == {"a": 0, "b": 1}
    assert cache.validator_permit("a") is False
    assert cache.stake("a") == 0.0
    assert cache.is_validator("a") is False


def test_update_from_metagraph_exposes_permit_and_stake() -> None:
    cache = MetagraphCache(netuid=1)
    cache.update_from_metagraph(
        ["validator", "miner"],
        validator_permits=[True, False],
        stakes=[123.5, 1.0],
    )

    assert cache.hotkey_to_uid == {"validator": 0, "miner": 1}
    assert cache.validator_permit("validator") is True
    assert cache.validator_permit("miner") is False
    assert cache.stake("validator") == 123.5
    assert cache.stake("miner") == 1.0

    # On metagraph AND permitted.
    assert cache.is_validator("validator") is True
    # On metagraph but NOT permitted.
    assert cache.is_validator("miner") is False
    # Absent from metagraph.
    assert cache.is_validator("ghost") is False
    assert cache.validator_permit("ghost") is False
    assert cache.stake("ghost") == 0.0


def test_refresh_reads_permit_and_stake_from_subtensor() -> None:
    metagraph = SimpleNamespace(
        hotkeys=["v0", "v1"],
        validator_permit=[True, False],
        S=[10.0, 2.0],
    )
    subtensor = SimpleNamespace(metagraph=lambda netuid: metagraph)
    cache = MetagraphCache(netuid=7, ttl_seconds=0, subtensor=subtensor)

    result = cache.get(force=True)

    assert result == {"v0": 0, "v1": 1}
    assert cache.validator_permit("v0") is True
    assert cache.validator_permit("v1") is False
    assert cache.stake("v0") == 10.0
    assert cache.is_validator("v0") is True
    assert cache.is_validator("v1") is False
