"""Validator subnet identity resolution (architecture.md sec 7.2).

Encodes the VAL-VDIR-ID-001..003 assertions for the on-chain identity reader,
the self-declared fallback, and the graceful-degradation guards:

* ID-001: with a (mocked) live chain, identity resolves ``{display_name,
  logo_url}`` by hotkey.
* ID-002: with no chain, identity resolves from the self-declared fallback
  (mock_metagraph and/or the validator ``last_seen_meta``); ``None`` when
  neither source is set.
* ID-003: the reader degrades gracefully -- it NEVER constructs a live
  ``Subtensor`` when the chain is disabled, and does not crash on missing or
  renamed chain fields.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from base.bittensor.identity_cache import (
    SOURCE_CHAIN,
    SOURCE_SELF_DECLARED,
    IdentityCache,
    ResolvedIdentity,
    ValidatorIdentityResolver,
    identity_from_meta,
    self_declared_identity,
)

VALIDATOR_HOTKEY = "validator-hotkey"
OTHER_HOTKEY = "other-hotkey"


class _ChainIdentity:
    """A faked ``ChainIdentity`` (bittensor>=9 neuron/delegate identity)."""

    def __init__(self, name: str | None = None, image: str | None = None) -> None:
        self.name = name
        self.image = image


class _SubnetIdentity:
    def __init__(
        self, subnet_name: str | None = None, logo_url: str | None = None
    ) -> None:
        self.subnet_name = subnet_name
        self.logo_url = logo_url


class _RecordingSubtensor:
    """A faked Subtensor recording the per-hotkey identity queries it serves."""

    def __init__(
        self,
        identities: dict[str, Any] | None = None,
        subnet: Any | None = None,
    ) -> None:
        self._identities = identities or {}
        self._subnet = subnet
        self.identity_calls: list[str] = []
        self.subnet_calls: list[int] = []

    def query_identity(self, hotkey: str, block: int | None = None) -> Any:
        self.identity_calls.append(hotkey)
        return self._identities.get(hotkey)

    def get_subnet_identity(self, netuid: int) -> Any:
        self.subnet_calls.append(netuid)
        return self._subnet


def _live_cache(subtensor: Any, *, netuid: int = 7, ttl: int = 300) -> IdentityCache:
    return IdentityCache(netuid=netuid, ttl_seconds=ttl, subtensor=subtensor)


# ---------------------------------------------------------------------------
# VAL-VDIR-ID-001: on-chain identity resolved by hotkey
# ---------------------------------------------------------------------------


def test_id001_on_chain_identity_resolved_by_hotkey() -> None:
    subtensor = _RecordingSubtensor(
        identities={
            VALIDATOR_HOTKEY: _ChainIdentity(name="Acme Validator", image="ipfs://logo")
        }
    )
    cache = _live_cache(subtensor)

    identity = cache.get(VALIDATOR_HOTKEY)

    assert identity == ResolvedIdentity(
        display_name="Acme Validator", logo_url="ipfs://logo", source=SOURCE_CHAIN
    )
    assert subtensor.identity_calls == [VALIDATOR_HOTKEY]


def test_id001_on_chain_identity_is_ttl_cached() -> None:
    subtensor = _RecordingSubtensor(
        identities={VALIDATOR_HOTKEY: _ChainIdentity(name="Acme")}
    )
    cache = _live_cache(subtensor)

    assert cache.get(VALIDATOR_HOTKEY) is not None
    assert cache.get(VALIDATOR_HOTKEY) is not None
    # Second lookup is served from the cache (single chain query within the TTL).
    assert subtensor.identity_calls == [VALIDATOR_HOTKEY]


def test_id001_on_chain_identity_requeries_after_ttl() -> None:
    subtensor = _RecordingSubtensor(
        identities={VALIDATOR_HOTKEY: _ChainIdentity(name="Acme")}
    )
    cache = _live_cache(subtensor, ttl=300)

    assert cache.get(VALIDATOR_HOTKEY) is not None
    cache._updated_at = 0.0  # force the snapshot to look long-expired
    assert cache.get(VALIDATOR_HOTKEY) is not None
    assert subtensor.identity_calls == [VALIDATOR_HOTKEY, VALIDATOR_HOTKEY]


def test_id001_subnet_identity_resolved() -> None:
    subtensor = _RecordingSubtensor(
        subnet=_SubnetIdentity(subnet_name="Base", logo_url="https://base/logo.png")
    )
    cache = _live_cache(subtensor)

    subnet = cache.subnet_identity()

    assert subnet == ResolvedIdentity(
        display_name="Base", logo_url="https://base/logo.png", source=SOURCE_CHAIN
    )
    # Cached: a second read does not re-query the chain.
    assert cache.subnet_identity() == subnet
    assert subtensor.subnet_calls == [7]


def test_id001_missing_hotkey_returns_none_without_error() -> None:
    cache = _live_cache(_RecordingSubtensor(identities={}))
    assert cache.get(OTHER_HOTKEY) is None


# ---------------------------------------------------------------------------
# VAL-VDIR-ID-003: graceful degradation; never builds a live Subtensor
# ---------------------------------------------------------------------------


def test_id003_chain_disabled_never_constructs_subtensor() -> None:
    # No subtensor and not static => chain disabled. get()/subnet_identity()
    # return None and there is no way to construct a Subtensor (the field is
    # None and is never assigned).
    cache = IdentityCache(netuid=7, subtensor=None)

    assert cache.get(VALIDATOR_HOTKEY) is None
    assert cache.subnet_identity() is None
    assert cache.subtensor is None


def test_id003_defensive_against_missing_query_identity_method() -> None:
    # A bittensor build whose Subtensor lacks query_identity must not crash.
    subtensor = SimpleNamespace()
    cache = _live_cache(subtensor)
    assert cache.get(VALIDATOR_HOTKEY) is None


def test_id003_defensive_against_missing_get_subnet_identity_method() -> None:
    # bittensor 10.x has no get_subnet_identity on Subtensor (drift): degrade.
    subtensor = SimpleNamespace(query_identity=lambda hotkey, block=None: None)
    cache = _live_cache(subtensor)
    assert cache.subnet_identity() is None


def test_id003_subnet_identity_none_when_chain_returns_none() -> None:
    subtensor = _RecordingSubtensor(subnet=None)
    cache = _live_cache(subtensor)
    assert cache.subnet_identity() is None


def test_id003_subnet_identity_none_on_renamed_fields() -> None:
    subtensor = _RecordingSubtensor(subnet=SimpleNamespace(title="x", icon="y"))
    cache = _live_cache(subtensor)
    assert cache.subnet_identity() is None


def test_id003_subnet_query_raising_degrades() -> None:
    def _boom(netuid: int) -> Any:
        raise RuntimeError("chain unavailable")

    subtensor = SimpleNamespace(get_subnet_identity=_boom)
    cache = _live_cache(subtensor)
    assert cache.subnet_identity() is None


def test_id003_defensive_against_renamed_chain_fields() -> None:
    # A ChainIdentity carrying neither name/image nor any known alias resolves
    # to None rather than raising on the absent attributes.
    subtensor = _RecordingSubtensor(
        identities={VALIDATOR_HOTKEY: SimpleNamespace(handle="x", avatar="y")}
    )
    cache = _live_cache(subtensor)
    assert cache.get(VALIDATOR_HOTKEY) is None


def test_id003_defensive_against_query_raising() -> None:
    def _boom(hotkey: str, block: int | None = None) -> Any:
        raise RuntimeError("chain unavailable")

    subtensor = SimpleNamespace(query_identity=_boom)
    cache = _live_cache(subtensor)
    assert cache.get(VALIDATOR_HOTKEY) is None


def test_id003_partial_chain_fields_resolve() -> None:
    # Only one of name/image present still resolves (no crash on the other).
    subtensor = _RecordingSubtensor(
        identities={VALIDATOR_HOTKEY: _ChainIdentity(name="OnlyName")}
    )
    cache = _live_cache(subtensor)
    assert cache.get(VALIDATOR_HOTKEY) == ResolvedIdentity(
        display_name="OnlyName", logo_url=None, source=SOURCE_CHAIN
    )


# ---------------------------------------------------------------------------
# VAL-VDIR-ID-002: self-declared fallback (no chain)
# ---------------------------------------------------------------------------


def test_id002_static_cache_resolves_seeded_identity_without_chain() -> None:
    cache = IdentityCache(netuid=7, subtensor=None)
    cache.seed_static(
        {
            VALIDATOR_HOTKEY: ResolvedIdentity(
                display_name="Self Declared",
                logo_url="https://x/logo.png",
                source=SOURCE_SELF_DECLARED,
            )
        }
    )

    assert cache.static is True
    assert cache.subtensor is None
    identity = cache.get(VALIDATOR_HOTKEY)
    assert identity is not None
    assert identity.display_name == "Self Declared"
    assert identity.source == SOURCE_SELF_DECLARED
    # Unseeded hotkey is None-safe.
    assert cache.get(OTHER_HOTKEY) is None


def test_id002_identity_from_meta_parses_self_declared_fields() -> None:
    identity = identity_from_meta(
        {"display_name": " Node A ", "logo_url": " https://a/l.png "}
    )
    assert identity == ResolvedIdentity(
        display_name="Node A",
        logo_url="https://a/l.png",
        source=SOURCE_SELF_DECLARED,
    )


def test_id002_static_subnet_identity_resolves_without_chain() -> None:
    cache = IdentityCache(netuid=7, subtensor=None)
    cache.seed_static(
        {},
        subnet_identity=ResolvedIdentity(
            display_name="Base", source=SOURCE_SELF_DECLARED
        ),
    )
    subnet = cache.subnet_identity()
    assert subnet is not None
    assert subnet.display_name == "Base"


def test_id002_identity_from_meta_none_when_absent_or_blank() -> None:
    assert identity_from_meta(None) is None
    assert identity_from_meta({}) is None
    assert identity_from_meta({"display_name": "  ", "logo_url": ""}) is None
    assert identity_from_meta({"capabilities": ["cpu"]}) is None


def test_self_declared_identity_helper() -> None:
    assert self_declared_identity(None, None) is None
    assert self_declared_identity("  ", "") is None
    assert self_declared_identity("Name", None) == ResolvedIdentity(
        display_name="Name", logo_url=None, source=SOURCE_SELF_DECLARED
    )


# ---------------------------------------------------------------------------
# Resolver: on-chain > self-declared fallback > None
# ---------------------------------------------------------------------------


def test_resolver_prefers_on_chain_over_self_declared() -> None:
    subtensor = _RecordingSubtensor(
        identities={VALIDATOR_HOTKEY: _ChainIdentity(name="Chain Name", image="img")}
    )
    resolver = ValidatorIdentityResolver(cache=_live_cache(subtensor))

    identity = resolver.resolve(
        VALIDATOR_HOTKEY,
        last_seen_meta={"display_name": "Self Name", "logo_url": "self-logo"},
    )
    assert identity is not None
    assert identity.display_name == "Chain Name"
    assert identity.source == SOURCE_CHAIN


def test_resolver_falls_back_to_self_declared_meta_without_chain() -> None:
    resolver = ValidatorIdentityResolver(cache=IdentityCache(netuid=7, subtensor=None))

    identity = resolver.resolve(
        VALIDATOR_HOTKEY,
        last_seen_meta={"display_name": "Self Name", "logo_url": "self-logo"},
    )
    assert identity is not None
    assert identity.display_name == "Self Name"
    assert identity.logo_url == "self-logo"
    assert identity.source == SOURCE_SELF_DECLARED


def test_resolver_prefers_static_seed_over_meta() -> None:
    cache = IdentityCache(netuid=7, subtensor=None)
    seeded = self_declared_identity("Operator Seeded", "op-logo")
    assert seeded is not None
    cache.seed_static({VALIDATOR_HOTKEY: seeded})
    resolver = ValidatorIdentityResolver(cache=cache)

    identity = resolver.resolve(
        VALIDATOR_HOTKEY,
        last_seen_meta={"display_name": "Validator Meta", "logo_url": "vm-logo"},
    )
    assert identity is not None
    assert identity.display_name == "Operator Seeded"


def test_resolver_none_when_neither_source_set() -> None:
    resolver = ValidatorIdentityResolver(cache=IdentityCache(netuid=7, subtensor=None))
    assert resolver.resolve(VALIDATOR_HOTKEY, last_seen_meta=None) is None
    assert resolver.resolve(VALIDATOR_HOTKEY, last_seen_meta={}) is None


def test_resolver_none_safe_without_cache() -> None:
    resolver = ValidatorIdentityResolver(cache=None)
    assert resolver.resolve(VALIDATOR_HOTKEY, last_seen_meta=None) is None
    assert resolver.subnet_identity() is None
    fallback = resolver.resolve(
        VALIDATOR_HOTKEY, last_seen_meta={"display_name": "Only Meta"}
    )
    assert fallback is not None
    assert fallback.display_name == "Only Meta"


def test_resolver_exposes_subnet_identity() -> None:
    subtensor = _RecordingSubtensor(subnet=_SubnetIdentity(subnet_name="Base"))
    resolver = ValidatorIdentityResolver(cache=_live_cache(subtensor))
    subnet = resolver.subnet_identity()
    assert subnet is not None
    assert subnet.display_name == "Base"


def test_resolved_identity_is_empty() -> None:
    assert ResolvedIdentity().is_empty is True
    assert ResolvedIdentity(display_name="x").is_empty is False
    assert ResolvedIdentity(logo_url="x").is_empty is False
