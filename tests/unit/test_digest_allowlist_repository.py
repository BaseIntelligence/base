"""TDD: durable SQLAlchemy adapter for DigestAllowlist (production host).

Pure lookup rules stay in ``base.compute.digest_allowlist``; this repository
persists to ``ImageDigestAllowlistEntry`` / denied tables and reloads into a
``DigestAllowlist`` for S5 allowlist-miss scenarios.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from base.compute.digest_allowlist import (
    AllowlistHit,
    AllowlistMiss,
    AllowlistMissReason,
    DigestRecord,
    ImageVariant,
)
from base.db.models import Base
from base.db.session import session_scope
from base.master.constation.allowlist_repository import DigestAllowlistRepository

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
TREE_A = "c" * 40
TREE_B = "d" * 40
DIGEST_CUDA = "sha256:" + ("1" * 64)
DIGEST_CPU = "sha256:" + ("2" * 64)
DIGEST_OTHER = "sha256:" + ("3" * 64)


def _record(
    *,
    commit_sha: str = COMMIT_A,
    tree_sha: str = TREE_A,
    variant: ImageVariant = ImageVariant.CUDA,
    digest: str = DIGEST_CUDA,
) -> DigestRecord:
    return DigestRecord(
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        variant=variant,
        digest=digest,
    )


@pytest.fixture
async def session_factory(
    tmp_path: Path,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    db_path = tmp_path / "allowlist.sqlite3"
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
async def test_register_persists_and_reloads_lookup_hit(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Given register via repository, When new instance loads, Then lookup hits."""
    repo = DigestAllowlistRepository(session_factory)
    record = _record()
    await repo.register(record)

    reloaded = DigestAllowlistRepository(session_factory)
    allowlist = await reloaded.load_allowlist()
    result = allowlist.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )
    assert isinstance(result, AllowlistHit)
    assert result.record == record


@pytest.mark.asyncio
async def test_lookup_unknown_digest_is_unknown(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repo = DigestAllowlistRepository(session_factory)
    await repo.register(_record())
    allowlist = await repo.load_allowlist()
    result = allowlist.lookup(
        digest=DIGEST_OTHER,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )
    assert isinstance(result, AllowlistMiss)
    assert result.reason is AllowlistMissReason.UNKNOWN_DIGEST


@pytest.mark.asyncio
async def test_revoke_digest_loads_as_revoked(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """S5: denied digest remains revoked after reload."""
    repo = DigestAllowlistRepository(session_factory)
    await repo.register(_record())
    await repo.revoke_digest(DIGEST_CUDA, reason="ops-revoke")

    allowlist = await DigestAllowlistRepository(session_factory).load_allowlist()
    result = allowlist.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )
    assert isinstance(result, AllowlistMiss)
    assert result.reason is AllowlistMissReason.REVOKED


@pytest.mark.asyncio
async def test_revoke_commit_loads_as_revoked(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repo = DigestAllowlistRepository(session_factory)
    await repo.register(_record())
    await repo.revoke_commit(COMMIT_A, reason="commit-yanked")

    allowlist = await DigestAllowlistRepository(session_factory).load_allowlist()
    result = allowlist.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )
    assert isinstance(result, AllowlistMiss)
    assert result.reason is AllowlistMissReason.REVOKED


@pytest.mark.asyncio
async def test_multiple_records_roundtrip(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repo = DigestAllowlistRepository(session_factory)
    cuda = _record(variant=ImageVariant.CUDA, digest=DIGEST_CUDA)
    cpu = _record(
        commit_sha=COMMIT_B,
        tree_sha=TREE_B,
        variant=ImageVariant.CPU,
        digest=DIGEST_CPU,
    )
    await repo.register(cuda)
    await repo.register(cpu)

    allowlist = await DigestAllowlistRepository(session_factory).load_allowlist()
    hit_cuda = allowlist.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant="cuda",
    )
    hit_cpu = allowlist.lookup(
        digest=DIGEST_CPU,
        commit_sha=COMMIT_B,
        tree_sha=TREE_B,
        variant="cpu",
    )
    assert isinstance(hit_cuda, AllowlistHit)
    assert isinstance(hit_cpu, AllowlistHit)
    assert len(allowlist.snapshot().records) == 2


@pytest.mark.asyncio
async def test_identical_reregister_is_noop(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repo = DigestAllowlistRepository(session_factory)
    await repo.register(_record())
    await repo.register(_record())  # identical
    async with session_scope(session_factory) as session:
        from sqlalchemy import func, select

        from base.db.models import ImageDigestAllowlistEntry

        count = await session.scalar(
            select(func.count()).select_from(ImageDigestAllowlistEntry)
        )
    assert count == 1
