"""Validator subnet identity resolution (architecture.md sec 7.2).

Mirrors :mod:`base.bittensor.metagraph_cache` (``subtensor``/``static``/``ttl``/
``get``) for validator display identity (display name + logo). The PRIMARY
source is on-chain Bittensor identity by hotkey; the FALLBACK is self-declared
config (the no-chain ``mock_metagraph`` seed and/or the validator-config values
surfaced via ``last_seen_meta``). Self-declared identity is UNTRUSTED.

The reader degrades gracefully: it NEVER constructs a live ``Subtensor`` (it only
uses one handed to it), returns ``None`` when the chain is disabled, and uses
defensive ``getattr`` so a renamed/missing chain field (bittensor>=9 drift) never
raises.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Trusted on-chain identity.
SOURCE_CHAIN = "chain"
#: UNTRUSTED operator/validator self-declared identity.
SOURCE_SELF_DECLARED = "self_declared"

#: ``last_seen_meta`` keys carrying the validator-config self-declared identity.
IDENTITY_DISPLAY_NAME_KEY = "display_name"
IDENTITY_LOGO_URL_KEY = "logo_url"

# Defensive field aliases (bittensor>=9 drift): the first non-empty wins.
_CHAIN_NAME_FIELDS = ("name", "display_name", "display")
_CHAIN_IMAGE_FIELDS = ("image", "logo_url", "logo", "image_url")
_SUBNET_NAME_FIELDS = ("subnet_name", "name", "display_name")
_SUBNET_LOGO_FIELDS = ("logo_url", "image", "logo", "image_url")


@dataclass(frozen=True)
class ResolvedIdentity:
    """A resolved validator identity (display name + logo URL).

    ``source`` records provenance: :data:`SOURCE_CHAIN` (trusted) or
    :data:`SOURCE_SELF_DECLARED` (UNTRUSTED). Consumers MUST sanitize a
    self-declared identity on render and never execute the logo URL.
    """

    display_name: str | None = None
    logo_url: str | None = None
    source: str | None = None

    @property
    def is_empty(self) -> bool:
        return self.display_name is None and self.logo_url is None


def _clean(value: Any) -> str | None:
    """Coerce ``value`` to a trimmed non-empty string, else ``None``."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_attr(obj: Any, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = _clean(getattr(obj, name, None))
        if value is not None:
            return value
    return None


def self_declared_identity(display_name: Any, logo_url: Any) -> ResolvedIdentity | None:
    """Build an UNTRUSTED self-declared identity, or ``None`` when both blank."""

    name = _clean(display_name)
    logo = _clean(logo_url)
    if name is None and logo is None:
        return None
    return ResolvedIdentity(
        display_name=name, logo_url=logo, source=SOURCE_SELF_DECLARED
    )


def identity_from_meta(
    last_seen_meta: Mapping[str, Any] | None,
) -> ResolvedIdentity | None:
    """Extract the self-declared identity from a validator's ``last_seen_meta``."""

    if not last_seen_meta:
        return None
    return self_declared_identity(
        last_seen_meta.get(IDENTITY_DISPLAY_NAME_KEY),
        last_seen_meta.get(IDENTITY_LOGO_URL_KEY),
    )


@dataclass
class IdentityCache:
    """On-chain (or static self-declared) validator identity, keyed by hotkey.

    A ``static`` cache is seeded from config (the no-chain ``mock_metagraph``
    seam) and never reaches a chain. A live cache lazily queries ``subtensor``
    per hotkey (TTL-cached). When ``subtensor`` is ``None`` and the cache is not
    ``static`` the chain is disabled: every lookup returns ``None`` and NO
    ``Subtensor`` is ever constructed.
    """

    netuid: int
    ttl_seconds: int = 300
    subtensor: Any | None = None
    static: bool = False
    _identities: dict[str, ResolvedIdentity | None] = field(default_factory=dict)
    _subnet_identity: ResolvedIdentity | None = None
    _subnet_fetched: bool = False
    _updated_at: float = 0.0

    def seed_static(
        self,
        identities: Mapping[str, ResolvedIdentity],
        *,
        subnet_identity: ResolvedIdentity | None = None,
    ) -> None:
        """Seed a static (no-chain) snapshot from self-declared config."""

        self.static = True
        self.subtensor = None
        self._identities = dict(identities)
        self._subnet_identity = subnet_identity
        self._subnet_fetched = True

    def get(self, hotkey: str) -> ResolvedIdentity | None:
        """Resolve ``hotkey``'s on-chain (or static self-declared) identity."""

        if self.static:
            return self._identities.get(hotkey)
        if self.subtensor is None:
            return None
        self._expire_if_stale()
        if hotkey not in self._identities:
            self._identities[hotkey] = self._query_chain_identity(hotkey)
        return self._identities[hotkey]

    def subnet_identity(self) -> ResolvedIdentity | None:
        """Resolve the subnet-level identity (top-level display name + logo)."""

        if self.static:
            return self._subnet_identity
        if self.subtensor is None:
            return None
        self._expire_if_stale()
        if not self._subnet_fetched:
            self._subnet_identity = self._query_subnet_identity()
            self._subnet_fetched = True
        return self._subnet_identity

    def _expire_if_stale(self) -> None:
        now = time.time()
        if now - self._updated_at > self.ttl_seconds:
            self._identities.clear()
            self._subnet_identity = None
            self._subnet_fetched = False
            self._updated_at = now

    def _query_chain_identity(self, hotkey: str) -> ResolvedIdentity | None:
        query = getattr(self.subtensor, "query_identity", None)
        if query is None:
            return None
        try:
            chain_identity = query(hotkey)
        except Exception:
            logger.debug("chain identity query failed for %s", hotkey, exc_info=True)
            return None
        if chain_identity is None:
            return None
        display_name = _first_attr(chain_identity, _CHAIN_NAME_FIELDS)
        logo_url = _first_attr(chain_identity, _CHAIN_IMAGE_FIELDS)
        if display_name is None and logo_url is None:
            return None
        return ResolvedIdentity(
            display_name=display_name, logo_url=logo_url, source=SOURCE_CHAIN
        )

    def _query_subnet_identity(self) -> ResolvedIdentity | None:
        getter = getattr(self.subtensor, "get_subnet_identity", None)
        if getter is None:
            return None
        try:
            subnet_identity = getter(self.netuid)
        except Exception:
            logger.debug(
                "subnet identity query failed for netuid %s",
                self.netuid,
                exc_info=True,
            )
            return None
        if subnet_identity is None:
            return None
        display_name = _first_attr(subnet_identity, _SUBNET_NAME_FIELDS)
        logo_url = _first_attr(subnet_identity, _SUBNET_LOGO_FIELDS)
        if display_name is None and logo_url is None:
            return None
        return ResolvedIdentity(
            display_name=display_name, logo_url=logo_url, source=SOURCE_CHAIN
        )


@dataclass(frozen=True)
class ValidatorIdentityResolver:
    """Resolve a validator's display identity (architecture.md sec 7.2).

    Resolution order: on-chain identity (or the static self-declared seed) from
    the :class:`IdentityCache`, else the validator's self-declared
    ``last_seen_meta`` fallback, else ``None``. Self-declared identities are
    UNTRUSTED (``source == SOURCE_SELF_DECLARED``).
    """

    cache: IdentityCache | None = None

    def resolve(
        self,
        hotkey: str,
        last_seen_meta: Mapping[str, Any] | None = None,
    ) -> ResolvedIdentity | None:
        if self.cache is not None:
            on_chain = self.cache.get(hotkey)
            if on_chain is not None and not on_chain.is_empty:
                return on_chain
        return identity_from_meta(last_seen_meta)

    def subnet_identity(self) -> ResolvedIdentity | None:
        if self.cache is None:
            return None
        return self.cache.subnet_identity()
