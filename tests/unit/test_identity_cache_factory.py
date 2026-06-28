"""Identity-cache wiring in the bittensor runtime factory (architecture.md 7.2).

The runtime factory seeds a STATIC identity cache from the no-chain
``mock_metagraph`` self-declared identities (never constructing a live
Subtensor), and shares the live Subtensor with the identity cache on the
chain-enabled path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from base.bittensor.factory import (
    create_bittensor_runtime,
    create_bittensor_submit_runtime,
)
from base.bittensor.identity_cache import SOURCE_SELF_DECLARED
from base.config.loader import load_settings

VALIDATOR_HOTKEY = "validator-hotkey"
PLAIN_HOTKEY = "plain-hotkey"


def _explode_bittensor(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("identity cache must not construct a live Subtensor")

    monkeypatch.setitem(
        sys.modules, "bittensor", SimpleNamespace(Subtensor=_Boom, Wallet=_Boom)
    )


def _mock_settings(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "network:",
                "  netuid: 7",
                "  chain_endpoint: ws://localhost:9944",
                "  mock_metagraph:",
                f"    - hotkey: {VALIDATOR_HOTKEY}",
                "      uid: 3",
                "      validator_permit: true",
                "      stake: 1000.0",
                "      display_name: Acme Validator",
                "      logo_url: https://acme/logo.png",
                f"    - hotkey: {PLAIN_HOTKEY}",
                "      uid: 4",
                "      validator_permit: true",
                "master:",
                "  metagraph_cache_ttl_seconds: 300",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(config)


def test_settings_parse_self_declared_identity(tmp_path: Path) -> None:
    nodes = _mock_settings(tmp_path).network.mock_metagraph
    assert nodes[0].display_name == "Acme Validator"
    assert nodes[0].logo_url == "https://acme/logo.png"
    # Identity fields are optional (default None).
    assert nodes[1].display_name is None
    assert nodes[1].logo_url is None


def test_factory_seeds_static_identity_cache_without_subtensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _explode_bittensor(monkeypatch)

    runtime = create_bittensor_runtime(_mock_settings(tmp_path))

    cache = runtime.identity_cache
    assert cache is not None
    assert cache.static is True
    assert cache.subtensor is None
    assert cache.netuid == 7

    identity = cache.get(VALIDATOR_HOTKEY)
    assert identity is not None
    assert identity.display_name == "Acme Validator"
    assert identity.logo_url == "https://acme/logo.png"
    assert identity.source == SOURCE_SELF_DECLARED
    # A mock node with no self-declared identity is None-safe.
    assert cache.get(PLAIN_HOTKEY) is None


def test_factory_live_identity_cache_shares_subtensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instances: list[object] = []

    class _Subtensor:
        def __init__(self, **kwargs: object) -> None:
            instances.append(self)

    monkeypatch.setitem(
        sys.modules, "bittensor", SimpleNamespace(Subtensor=_Subtensor, Wallet=object)
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "network:\n  netuid: 42\n  chain_endpoint: ws://localhost:9944\n",
        encoding="utf-8",
    )
    runtime = create_bittensor_runtime(load_settings(config))

    assert runtime.identity_cache is not None
    assert runtime.identity_cache.static is False
    assert runtime.identity_cache.netuid == 42
    # Exactly one Subtensor is built and shared by the metagraph + identity cache.
    assert len(instances) == 1
    assert runtime.identity_cache.subtensor is runtime.metagraph_cache.subtensor


def test_submit_runtime_includes_identity_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Subtensor:
        def __init__(self, **kwargs: object) -> None:
            pass

    class _Wallet:
        def __init__(self, **kwargs: object) -> None:
            self.hotkey = object()

    monkeypatch.setitem(
        sys.modules, "bittensor", SimpleNamespace(Subtensor=_Subtensor, Wallet=_Wallet)
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "network:\n  netuid: 9\n  chain_endpoint: ws://localhost:9944\n",
        encoding="utf-8",
    )
    runtime = create_bittensor_submit_runtime(load_settings(config))
    assert runtime.identity_cache is not None
    assert runtime.identity_cache.subtensor is runtime.metagraph_cache.subtensor
