"""TDD: durable SQLAlchemy adapter for AttestationNonceService (production host).

Pure consume order stays in ``base.compute.attestation_nonce``; this repository
issues and consumes against ``attestation_nonces`` with atomic SQL consume for S4.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from base.compute.attestation_nonce import (
    NonceBinding,
    NonceConsumeHit,
    NonceConsumeMiss,
    NonceConsumeReason,
)
from base.db.models import Base
from base.master.constation.nonce_repository import DurableAttestationNonceService

BINDING = NonceBinding(
    work_unit_id="wu-1",
    miner_hotkey="hk-miner",
    pod_id="pod-abc",
)
OTHER = NonceBinding(
    work_unit_id="wu-2",
    miner_hotkey="hk-miner",
    pod_id="pod-abc",
)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now = self._now + delta


@pytest.fixture
async def session_factory(
    tmp_path: Path,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    db_path = tmp_path / "nonces.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_issue_persists_across_service_instances(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    clock = _Clock(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))
    svc = DurableAttestationNonceService(
        session_factory, ttl=timedelta(hours=1), now_fn=clock.now
    )
    record = await svc.issue(BINDING)

    other = DurableAttestationNonceService(
        session_factory, ttl=timedelta(hours=1), now_fn=clock.now
    )
    result = await other.consume(record.nonce, BINDING)
    assert isinstance(result, NonceConsumeHit)
    assert result.record.nonce == record.nonce
    assert result.record.binding == BINDING


@pytest.mark.asyncio
async def test_second_consume_is_already_consumed_s4(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """S4 nonce replay after successful consume."""
    clock = _Clock(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))
    svc = DurableAttestationNonceService(
        session_factory, ttl=timedelta(hours=1), now_fn=clock.now
    )
    record = await svc.issue(BINDING)
    first = await svc.consume(record.nonce, BINDING)
    second = await svc.consume(record.nonce, BINDING)
    assert isinstance(first, NonceConsumeHit)
    assert isinstance(second, NonceConsumeMiss)
    assert second.reason is NonceConsumeReason.ALREADY_CONSUMED


@pytest.mark.asyncio
async def test_unknown_nonce(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    clock = _Clock(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))
    svc = DurableAttestationNonceService(
        session_factory, ttl=timedelta(hours=1), now_fn=clock.now
    )
    result = await svc.consume("00000000-0000-0000-0000-000000000000", BINDING)
    assert isinstance(result, NonceConsumeMiss)
    assert result.reason is NonceConsumeReason.UNKNOWN_NONCE


@pytest.mark.asyncio
async def test_expired_nonce(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    clock = _Clock(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))
    svc = DurableAttestationNonceService(
        session_factory, ttl=timedelta(seconds=30), now_fn=clock.now
    )
    record = await svc.issue(BINDING)
    clock.advance(timedelta(seconds=60))
    result = await svc.consume(record.nonce, BINDING)
    assert isinstance(result, NonceConsumeMiss)
    assert result.reason is NonceConsumeReason.EXPIRED


@pytest.mark.asyncio
async def test_work_unit_mismatch(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    clock = _Clock(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))
    svc = DurableAttestationNonceService(
        session_factory, ttl=timedelta(hours=1), now_fn=clock.now
    )
    record = await svc.issue(BINDING)
    result = await svc.consume(record.nonce, OTHER)
    assert isinstance(result, NonceConsumeMiss)
    assert result.reason is NonceConsumeReason.WORK_UNIT_MISMATCH


@pytest.mark.asyncio
async def test_atomic_double_consume_from_two_handles(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Two durable services race: only one HIT, one ALREADY_CONSUMED."""
    clock = _Clock(datetime(2026, 7, 26, 12, 0, tzinfo=UTC))
    a = DurableAttestationNonceService(
        session_factory, ttl=timedelta(hours=1), now_fn=clock.now
    )
    b = DurableAttestationNonceService(
        session_factory, ttl=timedelta(hours=1), now_fn=clock.now
    )
    record = await a.issue(BINDING)
    r1 = await a.consume(record.nonce, BINDING)
    r2 = await b.consume(record.nonce, BINDING)
    hits = sum(1 for r in (r1, r2) if isinstance(r, NonceConsumeHit))
    misses = [r for r in (r1, r2) if isinstance(r, NonceConsumeMiss)]
    assert hits == 1
    assert len(misses) == 1
    assert misses[0].reason is NonceConsumeReason.ALREADY_CONSUMED
