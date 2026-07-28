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
    sealed_manifest_hashes: dict[str, str] | None = None,
) -> DigestRecord:
    hashes = (
        sealed_manifest_hashes
        if sealed_manifest_hashes is not None
        else {"default.py": "e" * 64}
    )
    return DigestRecord(
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        variant=variant,
        digest=digest,
        sealed_manifest_hashes=hashes,
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


@pytest.mark.asyncio
async def test_allowlist_repository_roundtrips_sealed_hashes(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Given sealed hashes, When load_allowlist, Then they round-trip."""
    sealed = {"harness.py": "a" * 64, "lib/mod.py": "b" * 64}
    repo = DigestAllowlistRepository(session_factory)
    record = _record(sealed_manifest_hashes=sealed)
    await repo.register(record)

    allowlist = await DigestAllowlistRepository(session_factory).load_allowlist()
    result = allowlist.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )
    assert isinstance(result, AllowlistHit)
    assert dict(result.record.sealed_manifest_hashes) == sealed


@pytest.mark.asyncio
async def test_get_active_pin_returns_single_non_revoked(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Given exactly one non-revoked pin for variant, When get_active_pin, Then it."""
    repo = DigestAllowlistRepository(session_factory)
    sealed = {"default.py": "e" * 64}
    record = _record(sealed_manifest_hashes=sealed)
    await repo.register(record)

    pin = await repo.get_active_pin(variant=ImageVariant.CUDA)
    assert pin == record
    assert dict(pin.sealed_manifest_hashes) == sealed


@pytest.mark.asyncio
async def test_get_active_pin_zero_candidates_returns_none(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Given no pins for variant, When get_active_pin, Then None (fail-closed)."""
    repo = DigestAllowlistRepository(session_factory)
    await repo.register(_record(variant=ImageVariant.CPU, digest=DIGEST_CPU))

    assert await repo.get_active_pin(variant=ImageVariant.CUDA) is None
    assert await repo.get_active_pin(variant="cuda") is None


@pytest.mark.asyncio
async def test_get_active_pin_ambiguous_multiple_returns_none(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Given two non-revoked pins for same variant, When get_active_pin, Then None.

    Deterministic fail-closed: never silently pick newest/random among several.
    """
    repo = DigestAllowlistRepository(session_factory)
    await repo.register(_record(digest=DIGEST_CUDA))
    await repo.register(
        _record(
            commit_sha=COMMIT_B,
            tree_sha=TREE_B,
            digest=DIGEST_OTHER,
            variant=ImageVariant.CUDA,
        )
    )

    assert await repo.get_active_pin(variant=ImageVariant.CUDA) is None


@pytest.mark.asyncio
async def test_get_active_pin_skips_revoked_digest(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Given sole pin revoked by digest, When get_active_pin, Then None."""
    repo = DigestAllowlistRepository(session_factory)
    await repo.register(_record())
    await repo.revoke_digest(DIGEST_CUDA)

    assert await repo.get_active_pin(variant=ImageVariant.CUDA) is None


@pytest.mark.asyncio
async def test_get_active_pin_skips_revoked_commit_leaves_other(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Given two pins, one commit revoked, When get_active_pin, Then the other."""
    repo = DigestAllowlistRepository(session_factory)
    kept = _record(
        commit_sha=COMMIT_B,
        tree_sha=TREE_B,
        digest=DIGEST_OTHER,
        variant=ImageVariant.CUDA,
    )
    await repo.register(_record())  # COMMIT_A / DIGEST_CUDA
    await repo.register(kept)
    await repo.revoke_commit(COMMIT_A)

    pin = await repo.get_active_pin(variant=ImageVariant.CUDA)
    assert pin == kept


def test_constation_identity_payload_has_hook_keys() -> None:
    """Given DigestRecord, When identity payload, Then five hook keys present."""
    from base.master.constation.allowlist_repository import (
        constation_identity_payload,
    )

    sealed = {"a.py": "f" * 64}
    record = _record(sealed_manifest_hashes=sealed)
    payload = constation_identity_payload(record)
    assert payload["required_digest"] == DIGEST_CUDA
    assert payload["commit_sha"] == COMMIT_A
    assert payload["tree_sha"] == TREE_A
    assert payload["variant"] == "cuda"
    assert payload["sealed_manifest_hashes"] == sealed
    # Must not invent pod/instance identity at stamp time.
    assert "pod_id" not in payload
    assert "instance_id" not in payload
