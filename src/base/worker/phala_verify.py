"""Validator-owned measurement allowlist, nonce freshness, and run binding (M4).

Supporting types for the Phala-tier verifier in
:func:`base.worker.proof.verify_execution_proof` (architecture.md sec 4 C4 / 6 / 7):

* :class:`MeasurementAllowlist` -- the **validator-owned** set of canonical
  measurements a genuine quote must match. Membership (not mere quote validity)
  governs acceptance; an EMPTY allowlist fails closed (matches nothing), never
  accept-any. No requester-supplied value can widen it.
* :class:`NonceValidator` / :class:`InMemoryNonceValidator` -- validator-issued,
  single-use, TTL-bounded nonces. The nonce bound into a quote's ``report_data``
  must be one the validator issued and has not consumed/expired, defeating
  quote replay and cross-submission repurposing.
* :class:`PhalaBinding` -- the run identity the VALIDATOR expects for a submission
  (agent_hash, task_ids, scores_digest, validator_nonce) that ``report_data`` must
  bind. It is the validator's own record, never trusted from the attested payload.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from base.schemas.worker import PhalaMeasurement

#: The static, allowlist-pinnable measurement fields (excludes runtime ``rtmr3``).
CANONICAL_MEASUREMENT_FIELDS: tuple[str, ...] = (
    "mrtd",
    "rtmr0",
    "rtmr1",
    "rtmr2",
    "compose_hash",
    "os_image_hash",
)

#: Env var naming a JSON file with the validator's canonical measurement
#: allowlist. Unset/empty ⇒ an unconfigured validator ⇒ empty (fail-closed).
MEASUREMENT_ALLOWLIST_FILE_ENV = "BASE_PHALA_MEASUREMENT_ALLOWLIST_FILE"


def canonical_measurement_mapping(
    measurement: PhalaMeasurement | Mapping[str, object],
) -> dict[str, str]:
    """The static canonical subset of a measurement as a ``{field: str}`` mapping."""

    if isinstance(measurement, PhalaMeasurement):
        return measurement.canonical()
    return {field: str(measurement[field]) for field in CANONICAL_MEASUREMENT_FIELDS}


@dataclass(frozen=True)
class MeasurementAllowlist:
    """A validator-owned set of canonical measurements a quote must match.

    Matching is exact across ALL canonical registers. The allowlist can hold
    MORE THAN ONE entry so that during a canonical-image rotation both the
    outgoing and incoming measurements are trusted simultaneously -- a quote
    matching ANY entry passes, one matching none is rejected. An empty allowlist
    matches nothing (fail closed) -- an unconfigured validator never accepts a
    quote, and every load path below fails closed (to empty) on a missing,
    unreadable, or unparseable source rather than defaulting to accept-any.
    """

    entries: tuple[dict[str, str], ...] = ()

    @classmethod
    def from_measurements(
        cls, measurements: Iterable[PhalaMeasurement | Mapping[str, object]]
    ) -> MeasurementAllowlist:
        return cls(tuple(canonical_measurement_mapping(m) for m in measurements))

    @classmethod
    def from_json(cls, text: str) -> MeasurementAllowlist:
        """Parse a JSON allowlist, FAILING CLOSED (empty) on any malformed input.

        Accepts either a bare ``[entry, ...]`` list or a ``{"entries": [...]}``
        object. Invalid JSON, an unexpected top-level shape, a non-mapping entry,
        or an entry missing a canonical register yields an EMPTY allowlist (which
        rejects everything) -- never an accept-any allowlist and never an
        exception a caller might mistake for success (VAL-VERIFY-025).
        """

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return cls()
        if isinstance(data, Mapping):
            data = data.get("entries", [])
        if not isinstance(data, list):
            return cls()
        entries: list[dict[str, str]] = []
        for item in data:
            if not isinstance(item, Mapping):
                return cls()
            try:
                entries.append(canonical_measurement_mapping(item))
            except (KeyError, TypeError):
                return cls()
        return cls(tuple(entries))

    @classmethod
    def from_file(cls, path: str | Path) -> MeasurementAllowlist:
        """Load an allowlist from a JSON file, FAILING CLOSED on any I/O error.

        A missing or unreadable file yields an EMPTY allowlist (fail closed)
        rather than raising or accepting anything.
        """

        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            return cls()
        return cls.from_json(text)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MeasurementAllowlist:
        """Load the allowlist named by :data:`MEASUREMENT_ALLOWLIST_FILE_ENV`.

        When the env var is unset/empty the validator is UNCONFIGURED, which
        fails closed to an empty allowlist (accepts nothing).
        """

        environ = os.environ if env is None else env
        path = environ.get(MEASUREMENT_ALLOWLIST_FILE_ENV)
        if not path:
            return cls()
        return cls.from_file(path)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def contains(self, measurement: PhalaMeasurement | Mapping[str, object]) -> bool:
        """Whether ``measurement`` exactly matches a canonical allowlist entry."""

        candidate = canonical_measurement_mapping(measurement)
        return any(candidate == entry for entry in self.entries)


class NonceState(StrEnum):
    """Outcome of consuming a validator nonce."""

    OK = "ok"
    UNKNOWN = "unknown"
    EXPIRED = "expired"
    CONSUMED = "consumed"


@runtime_checkable
class NonceValidator(Protocol):
    """Consumes a validator-issued nonce, reporting its freshness state.

    ``consume`` is single-use: the first consume of a known, unexpired nonce
    returns :attr:`NonceState.OK` and marks it consumed; any later consume of the
    same nonce returns :attr:`NonceState.CONSUMED`.
    """

    def consume(self, nonce: str) -> NonceState:  # pragma: no cover - protocol
        ...


@dataclass
class InMemoryNonceValidator:
    """A single-use, TTL-bounded :class:`NonceValidator` (reference / tests).

    Nonces are 256-bit ``secrets``-random unless a value is supplied to
    :meth:`issue`. ``clock`` is injectable so expiry can be exercised
    deterministically.
    """

    ttl_seconds: float = 120.0
    clock: Callable[[], float] = time.time
    _issued: dict[str, float] = field(default_factory=dict)
    _consumed: set[str] = field(default_factory=set)

    def issue(self, nonce: str | None = None) -> str:
        value = nonce if nonce is not None else secrets.token_urlsafe(32)
        self._issued[value] = self.clock()
        return value

    def is_outstanding(self, nonce: str) -> bool:
        if not nonce or nonce in self._consumed or nonce not in self._issued:
            return False
        return (self.clock() - self._issued[nonce]) <= self.ttl_seconds

    def consume(self, nonce: str) -> NonceState:
        if not nonce or nonce not in self._issued:
            return NonceState.UNKNOWN
        if nonce in self._consumed:
            return NonceState.CONSUMED
        if (self.clock() - self._issued[nonce]) > self.ttl_seconds:
            return NonceState.EXPIRED
        self._consumed.add(nonce)
        return NonceState.OK


@dataclass(frozen=True)
class PhalaBinding:
    """The run identity a validator expects a submission's ``report_data`` to bind.

    Sourced from the validator's OWN records (the submission's agent hash, the
    accepted work unit's task ids, the scores the result reports, and the nonce
    the validator issued) -- never trusted from the attested payload. ``task_ids``
    are compared order-insensitively (sorted before hashing).
    """

    agent_hash: str
    task_ids: tuple[str, ...]
    scores_digest: str
    validator_nonce: str = ""
    eval_run_id: str | None = None
    score_nonce: str | None = None

    def __post_init__(self) -> None:
        if (self.eval_run_id is None) != (self.score_nonce is None):
            raise ValueError("eval_run_id and score_nonce must be supplied together")
        if self.is_eval_v2 and self.validator_nonce:
            raise ValueError("schema-version-2 bindings do not use validator_nonce")

    @property
    def is_eval_v2(self) -> bool:
        """Whether this immutable binding uses the schema-version-2 Eval shape."""

        return self.eval_run_id is not None or self.score_nonce is not None

    @property
    def nonce(self) -> str:
        """The purpose-scoped nonce consumed after a successful verification."""

        return (
            self.score_nonce if self.score_nonce is not None else self.validator_nonce
        )


__all__ = [
    "CANONICAL_MEASUREMENT_FIELDS",
    "MEASUREMENT_ALLOWLIST_FILE_ENV",
    "InMemoryNonceValidator",
    "MeasurementAllowlist",
    "NonceState",
    "NonceValidator",
    "PhalaBinding",
    "canonical_measurement_mapping",
]
