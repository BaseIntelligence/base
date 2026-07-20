"""Fresh, single-use validator nonces for the golden key-release protocol.

The validator key-release endpoint hands the CVM a fresh, high-entropy nonce
which the enclave must bind into its quote's ``report_data`` (architecture.md
§4 C3). A nonce is:

* **fresh + unpredictable** -- 256-bit, drawn from :func:`secrets.token_urlsafe`,
  so it cannot be guessed or derived from anything the requester observes;
* **time-bounded** -- rejected once its validity window (TTL) elapses; and
* **single-use** -- consumed on the first completed release attempt (whether the
  attempt released the key or was denied), so a captured quote bound to a nonce
  can never be replayed against a second release.

This store is the authority for nonce state; it is deliberately in-memory and
validator-local (the allowlist authority is validator-owned, never the miner).
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from enum import Enum

#: Default nonce validity window (seconds). A quote bound to a nonce presented
#: after this window has elapsed is rejected as stale.
DEFAULT_NONCE_TTL_SECONDS = 120.0

#: Entropy of an issued nonce in bytes (256-bit) -> unpredictable / unguessable.
NONCE_ENTROPY_BYTES = 32


class NonceState(Enum):
    """The outcome of presenting a nonce for consumption."""

    #: Fresh, issued-by-us, unexpired, unconsumed -> consumed now, release may proceed.
    OK = "ok"
    #: Never issued by this endpoint (attacker-chosen value).
    UNKNOWN = "unknown"
    #: Issued but presented after its validity window elapsed.
    EXPIRED = "expired"
    #: Already consumed by a prior completed release attempt.
    CONSUMED = "consumed"


class NonceStore:
    """Thread-safe issuer/validator of fresh, single-use, time-bounded nonces.

    ``clock`` is injectable so nonce expiry is testable without real sleeps; it
    defaults to :func:`time.monotonic` (immune to wall-clock adjustments).
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_NONCE_TTL_SECONDS,
        entropy_bytes: int = NONCE_ENTROPY_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("nonce ttl must be positive")
        if entropy_bytes < 16:
            raise ValueError("nonce entropy must be at least 128-bit")
        self._ttl = float(ttl_seconds)
        self._entropy_bytes = int(entropy_bytes)
        self._clock = clock
        self._issued: dict[str, float] = {}
        self._consumed: set[str] = set()
        self._lock = threading.Lock()

    def issue(self) -> str:
        """Return a fresh, high-entropy nonce and record it as outstanding."""

        nonce = secrets.token_urlsafe(self._entropy_bytes)
        with self._lock:
            self._issued[nonce] = self._clock()
        return nonce

    def is_outstanding(self, nonce: str) -> bool:
        """Whether ``nonce`` is currently issued, unexpired, and unconsumed."""

        with self._lock:
            issued_at = self._issued.get(nonce)
            if issued_at is None:
                return False
            return (self._clock() - issued_at) <= self._ttl

    def consume(self, nonce: str) -> NonceState:
        """Atomically validate and consume ``nonce`` for a release attempt.

        Returns :class:`NonceState`. A fresh, unexpired, previously-unconsumed
        nonce transitions to consumed and returns :attr:`NonceState.OK`; every
        subsequent presentation of the same value returns
        :attr:`NonceState.CONSUMED`. Unknown/expired values are rejected without
        yielding OK, so no key can ever be released for them.
        """

        with self._lock:
            if nonce in self._consumed:
                return NonceState.CONSUMED
            issued_at = self._issued.get(nonce)
            if issued_at is None:
                return NonceState.UNKNOWN
            if (self._clock() - issued_at) > self._ttl:
                # Burn the stale nonce so it cannot be presented again.
                del self._issued[nonce]
                self._consumed.add(nonce)
                return NonceState.EXPIRED
            del self._issued[nonce]
            self._consumed.add(nonce)
            return NonceState.OK

    def outstanding_count(self) -> int:
        """Number of currently-issued (not yet consumed) nonces (diagnostics)."""

        with self._lock:
            return len(self._issued)


__all__ = [
    "DEFAULT_NONCE_TTL_SECONDS",
    "NONCE_ENTROPY_BYTES",
    "NonceState",
    "NonceStore",
]
