"""Durable SQLAlchemy host for attestation nonces (mechanism 1 storage).

Issue/consume *rules* match :class:`base.compute.attestation_nonce.AttestationNonceService`.
Consume uses ``UPDATE … WHERE consumed_at IS NULL`` so multi-worker races still
yield exactly one HIT (S4).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.compute.attestation_nonce import (
    NonceBinding,
    NonceConsumeHit,
    NonceConsumeMiss,
    NonceConsumeReason,
    NonceConsumeResult,
    NonceRecord,
)
from base.db.models import AttestationNonce
from base.db.session import session_scope


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_aware(dt: datetime) -> datetime:
    """SQLite often returns naive UTC; normalize for comparisons."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)



class DurableAttestationNonceService:
    """BASE-clock nonce issuer/consumer backed by ``attestation_nonces``."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        ttl: timedelta,
        now_fn: Callable[[], datetime] = _utc_now,
        session_scope_fn: Callable[..., Any] = session_scope,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError(f"ttl must be positive, got {ttl!r}")
        self._session_factory = session_factory
        self._ttl = ttl
        self._now_fn = now_fn
        self._session_scope = session_scope_fn

    async def issue(self, binding: NonceBinding) -> NonceRecord:
        """Issue a fresh UUID nonce bound to ``binding`` at BASE ``now_fn`` time."""
        bound = NonceBinding(
            work_unit_id=binding.work_unit_id,
            miner_hotkey=binding.miner_hotkey,
            pod_id=binding.pod_id,
        )
        issued_at = self._now_fn()
        if issued_at.tzinfo is None:
            raise ValueError("BASE now_fn must return timezone-aware datetime")
        record = NonceRecord(
            nonce=str(uuid.uuid4()),
            binding=bound,
            issued_at=issued_at,
            expires_at=issued_at + self._ttl,
            consumed_at=None,
        )
        async with self._session_scope(self._session_factory) as session:
            session.add(
                AttestationNonce(
                    nonce=record.nonce,
                    work_unit_id=bound.work_unit_id,
                    miner_hotkey=bound.miner_hotkey,
                    pod_id=bound.pod_id,
                    issued_at=record.issued_at,
                    expires_at=record.expires_at,
                    consumed_at=None,
                )
            )
        return record

    async def consume(
        self,
        nonce: str,
        binding: NonceBinding,
        *,
        received_at: datetime | None = None,
    ) -> NonceConsumeResult:
        """Consume ``nonce`` once if unexpired and binding matches.

        Same check order as the pure in-memory service. Atomic claim via
        ``UPDATE … WHERE consumed_at IS NULL``.
        """
        want = NonceBinding(
            work_unit_id=binding.work_unit_id,
            miner_hotkey=binding.miner_hotkey,
            pod_id=binding.pod_id,
        )
        receive = received_at if received_at is not None else self._now_fn()
        if receive.tzinfo is None:
            raise ValueError("BASE received_at must be timezone-aware")

        key = nonce.strip()
        async with self._session_scope(self._session_factory) as session:
            row = (
                await session.execute(
                    select(AttestationNonce).where(AttestationNonce.nonce == key)
                )
            ).scalar_one_or_none()
            if row is None:
                return NonceConsumeMiss(reason=NonceConsumeReason.UNKNOWN_NONCE)

            if row.consumed_at is not None:
                return NonceConsumeMiss(reason=NonceConsumeReason.ALREADY_CONSUMED)

            if receive > _as_aware(row.expires_at):
                return NonceConsumeMiss(reason=NonceConsumeReason.EXPIRED)

            if row.work_unit_id != want.work_unit_id:
                return NonceConsumeMiss(reason=NonceConsumeReason.WORK_UNIT_MISMATCH)
            if row.miner_hotkey != want.miner_hotkey:
                return NonceConsumeMiss(reason=NonceConsumeReason.MINER_HOTKEY_MISMATCH)
            if row.pod_id != want.pod_id:
                return NonceConsumeMiss(reason=NonceConsumeReason.POD_MISMATCH)

            # Atomic single-use claim.
            result = await session.execute(
                update(AttestationNonce)
                .where(
                    AttestationNonce.nonce == key,
                    AttestationNonce.consumed_at.is_(None),
                )
                .values(consumed_at=receive)
            )
            if result.rowcount != 1:  # type: ignore[attr-defined]
                # Lost race after passes — treat as already consumed.
                return NonceConsumeMiss(reason=NonceConsumeReason.ALREADY_CONSUMED)

            consumed = NonceRecord(
                nonce=row.nonce,
                binding=NonceBinding(
                    work_unit_id=row.work_unit_id,
                    miner_hotkey=row.miner_hotkey,
                    pod_id=row.pod_id,
                ),
                issued_at=_as_aware(row.issued_at),
                expires_at=_as_aware(row.expires_at),
                consumed_at=receive,
            )
            return NonceConsumeHit(record=consumed, received_at=receive)


__all__ = ["DurableAttestationNonceService"]
