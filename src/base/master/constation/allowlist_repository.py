"""Durable SQLAlchemy host for DigestAllowlist (mechanism 4 storage).

Rules remain in base.compute.digest_allowlist; this module is the master
persistence boundary over image_digest_allowlist and deny tables.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.compute.digest_allowlist import (
    DigestAllowlist,
    DigestRecord,
    ImageVariant,
    is_full_git_sha,
    is_image_digest,
    normalize_image_digest,
)
from base.db.models import (
    DeniedImageCommit,
    DeniedImageDigest,
    ImageDigestAllowlistEntry,
)
from base.db.session import session_scope


def _normalize_digest(value: str) -> str:
    dig = normalize_image_digest(value)
    if not is_image_digest(dig):
        raise ValueError(f"digest must be sha256:<64 lowercase hex>, got {value!r}")
    return dig


def _normalize_commit(value: str) -> str:
    commit = value.strip().lower()
    if not is_full_git_sha(commit):
        raise ValueError(
            f"commit_sha must be a full 40-char lowercase hex git SHA, got {value!r}"
        )
    return commit


class DigestAllowlistRepository:
    """Persist and reload BASE-produced image digest bindings."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        session_scope_fn: Callable[..., Any] = session_scope,
    ) -> None:
        self._session_factory = session_factory
        self._session_scope = session_scope_fn

    async def register(self, record: DigestRecord) -> None:
        """Insert a binding; identical re-register is a no-op.

        Conflicting rebind (same digest, different commit/tree/variant) raises
        ValueError (matches pure DigestAllowlist.register).
        """
        async with self._session_scope(self._session_factory) as session:
            row = (
                await session.execute(
                    select(ImageDigestAllowlistEntry).where(
                        ImageDigestAllowlistEntry.digest == record.digest
                    )
                )
            ).scalar_one_or_none()
            if row is not None:
                bound = DigestRecord(
                    commit_sha=row.commit_sha,
                    tree_sha=row.tree_sha,
                    variant=row.variant,
                    digest=row.digest,
                    sealed_manifest_hashes=dict(row.sealed_manifest_hashes or {}),
                )
                if bound != record:
                    raise ValueError(
                        f"digest {record.digest!r} already bound to "
                        f"commit={bound.commit_sha} tree={bound.tree_sha} "
                        f"variant={bound.variant.value}; cannot rebind to "
                        f"commit={record.commit_sha} tree={record.tree_sha} "
                        f"variant={record.variant.value}"
                    )
                return
            session.add(
                ImageDigestAllowlistEntry(
                    commit_sha=record.commit_sha,
                    tree_sha=record.tree_sha,
                    variant=record.variant.value,
                    digest=record.digest,
                    sealed_manifest_hashes=dict(record.sealed_manifest_hashes),
                )
            )

    async def revoke_digest(self, digest: str, *, reason: str | None = None) -> None:
        """Add durable digest deny entry."""
        dig = _normalize_digest(digest)
        async with self._session_scope(self._session_factory) as session:
            existing = await session.get(DeniedImageDigest, dig)
            if existing is None:
                session.add(DeniedImageDigest(digest=dig, reason=reason))
            elif reason is not None and existing.reason is None:
                existing.reason = reason

    async def revoke_commit(
        self, commit_sha: str, *, reason: str | None = None
    ) -> None:
        """Add durable commit deny entry."""
        commit = _normalize_commit(commit_sha)
        async with self._session_scope(self._session_factory) as session:
            existing = await session.get(DeniedImageCommit, commit)
            if existing is None:
                session.add(DeniedImageCommit(commit_sha=commit, reason=reason))
            elif reason is not None and existing.reason is None:
                existing.reason = reason

    async def load_allowlist(self) -> DigestAllowlist:
        """Materialize a pure in-memory allowlist from durable tables."""
        async with self._session_scope(self._session_factory) as session:
            entries = (
                (await session.execute(select(ImageDigestAllowlistEntry)))
                .scalars()
                .all()
            )
            denied_d = (
                (await session.execute(select(DeniedImageDigest))).scalars().all()
            )
            denied_c = (
                (await session.execute(select(DeniedImageCommit))).scalars().all()
            )

            allowlist = DigestAllowlist()
            for row in entries:
                allowlist.register(
                    DigestRecord(
                        commit_sha=row.commit_sha,
                        tree_sha=row.tree_sha,
                        variant=ImageVariant(row.variant),
                        digest=row.digest,
                        sealed_manifest_hashes=dict(row.sealed_manifest_hashes or {}),
                    )
                )
            for row in denied_d:
                allowlist.revoke_digest(row.digest)
            for row in denied_c:
                allowlist.revoke_commit(row.commit_sha)
            return allowlist


__all__ = ["DigestAllowlistRepository"]
