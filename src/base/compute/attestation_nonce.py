"""BASE-issued single-use attestation nonces (mechanism 1).

BASE issues a UUID nonce bound to ``(work_unit_id, miner_hotkey, pod_id)`` with
a TTL covering job wall-clock plus skew. Exactly one successful consume is
allowed. Freshness is derived from **BASE issue and receive time only** — the
guest clock is untrusted and is never accepted as a security input (M3).

This module is pure in-memory with a snapshot boundary so a DB adapter can
persist without re-implementing the rules. SQLAlchemy table + alembic migration
live alongside for durable master storage.

Consume returns distinct machine reason codes (no prose matching):

* ``already_consumed`` — nonce was already successfully consumed (replay)
* ``unknown_nonce`` — nonce was never issued by this service
* ``expired`` — BASE receive time is past ``expires_at``
* ``work_unit_mismatch`` / ``miner_hotkey_mismatch`` / ``pod_mismatch`` —
  binding does not match the issued triple
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_nonblank(field_name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return normalized


class NonceConsumeReason(StrEnum):
    """Machine-consumed miss codes for nonce consume."""

    ALREADY_CONSUMED = "already_consumed"
    UNKNOWN_NONCE = "unknown_nonce"
    EXPIRED = "expired"
    WORK_UNIT_MISMATCH = "work_unit_mismatch"
    MINER_HOTKEY_MISMATCH = "miner_hotkey_mismatch"
    POD_MISMATCH = "pod_mismatch"


@dataclass(frozen=True, slots=True)
class NonceBinding:
    """Triple that a nonce is issued against and must match on consume."""

    work_unit_id: str
    miner_hotkey: str
    pod_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "work_unit_id", _require_nonblank("work_unit_id", self.work_unit_id)
        )
        object.__setattr__(
            self,
            "miner_hotkey",
            _require_nonblank("miner_hotkey", self.miner_hotkey),
        )
        object.__setattr__(self, "pod_id", _require_nonblank("pod_id", self.pod_id))


@dataclass(frozen=True, slots=True)
class NonceRecord:
    """One BASE-issued nonce and its lifecycle timestamps (BASE clocks only)."""

    nonce: str
    binding: NonceBinding
    issued_at: datetime
    expires_at: datetime
    consumed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class NonceConsumeHit:
    """Consume succeeded: first use, unexpired, binding matched."""

    record: NonceRecord
    received_at: datetime


@dataclass(frozen=True, slots=True)
class NonceConsumeMiss:
    """Consume failed with a distinct machine reason."""

    reason: NonceConsumeReason


NonceConsumeResult = NonceConsumeHit | NonceConsumeMiss


@dataclass(frozen=True, slots=True)
class NonceSnapshot:
    """Serializable state for DB/file adapters."""

    records: tuple[NonceRecord, ...]
    ttl: timedelta


@dataclass
class AttestationNonceService:
    """In-memory single-use nonce issuer/consumer.

    ``now_fn`` is the BASE clock. Callers may pass ``received_at`` on consume to
    pin the BASE receive instant; there is no guest-clock parameter.
    """

    ttl: timedelta
    now_fn: Callable[[], datetime] = field(default=_utc_now)
    _by_nonce: dict[str, NonceRecord] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.ttl <= timedelta(0):
            raise ValueError(f"ttl must be positive, got {self.ttl!r}")

    def issue(self, binding: NonceBinding) -> NonceRecord:
        """Issue a fresh UUID nonce bound to ``binding`` at BASE ``now_fn`` time."""
        # Re-validate via constructor path (binding already normalized).
        bound = NonceBinding(
            work_unit_id=binding.work_unit_id,
            miner_hotkey=binding.miner_hotkey,
            pod_id=binding.pod_id,
        )
        issued_at = self.now_fn()
        if issued_at.tzinfo is None:
            raise ValueError("BASE now_fn must return timezone-aware datetime")
        record = NonceRecord(
            nonce=str(uuid.uuid4()),
            binding=bound,
            issued_at=issued_at,
            expires_at=issued_at + self.ttl,
            consumed_at=None,
        )
        self._by_nonce[record.nonce] = record
        return record

    def consume(
        self,
        nonce: str,
        binding: NonceBinding,
        *,
        received_at: datetime | None = None,
    ) -> NonceConsumeResult:
        """Consume ``nonce`` once if unexpired and binding matches.

        Check order (first match wins):
        1. unknown nonce → ``unknown_nonce``
        2. already consumed → ``already_consumed``
        3. BASE receive time past expires_at → ``expired``
        4. work_unit / hotkey / pod mismatch → distinct binding reasons
        5. else hit (marks consumed_at = receive time)
        """
        want = NonceBinding(
            work_unit_id=binding.work_unit_id,
            miner_hotkey=binding.miner_hotkey,
            pod_id=binding.pod_id,
        )
        receive = received_at if received_at is not None else self.now_fn()
        if receive.tzinfo is None:
            raise ValueError("BASE received_at must be timezone-aware")

        key = nonce.strip()
        record = self._by_nonce.get(key)
        if record is None:
            return NonceConsumeMiss(reason=NonceConsumeReason.UNKNOWN_NONCE)

        if record.consumed_at is not None:
            return NonceConsumeMiss(reason=NonceConsumeReason.ALREADY_CONSUMED)

        if receive > record.expires_at:
            return NonceConsumeMiss(reason=NonceConsumeReason.EXPIRED)

        if record.binding.work_unit_id != want.work_unit_id:
            return NonceConsumeMiss(reason=NonceConsumeReason.WORK_UNIT_MISMATCH)
        if record.binding.miner_hotkey != want.miner_hotkey:
            return NonceConsumeMiss(reason=NonceConsumeReason.MINER_HOTKEY_MISMATCH)
        if record.binding.pod_id != want.pod_id:
            return NonceConsumeMiss(reason=NonceConsumeReason.POD_MISMATCH)

        consumed = NonceRecord(
            nonce=record.nonce,
            binding=record.binding,
            issued_at=record.issued_at,
            expires_at=record.expires_at,
            consumed_at=receive,
        )
        self._by_nonce[key] = consumed
        return NonceConsumeHit(record=consumed, received_at=receive)

    def snapshot(self) -> NonceSnapshot:
        """Export durable state for persistence adapters."""
        return NonceSnapshot(records=tuple(self._by_nonce.values()), ttl=self.ttl)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: NonceSnapshot,
        *,
        ttl: timedelta | None = None,
        now_fn: Callable[[], datetime] = _utc_now,
    ) -> AttestationNonceService:
        """Restore a service from :meth:`snapshot` output."""
        service = cls(ttl=ttl if ttl is not None else snapshot.ttl, now_fn=now_fn)
        for record in snapshot.records:
            service._by_nonce[record.nonce] = record
        return service
