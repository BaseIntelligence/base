"""BASE-produced image digest allowlist with post-hoc revocation.

Mechanism 4 of prism-lium image attestation: only a digest BASE itself produced
from a registered ``(commit_sha, tree_sha, variant)`` is scoreable. This module
is an **allowlist only** — it does not perform constation, nonce checks, or
signature verification. Those arrive in later waves.

Lookup is pure and returns distinct miss reasons so callers (prism
``constation_ok``) can fail closed without string-matching prose:

* ``unknown_digest`` — digest was never registered by BASE
* ``variant_mismatch`` — digest is known but for a different variant (cpu/cuda)
* ``commit_mismatch`` — digest is known but bound to a different commit/tree
* ``revoked`` — digest or its commit is on the denylist

Storage is in-memory with a snapshot boundary so a DB/file adapter can persist
without re-implementing the rules. SQLAlchemy tables + alembic migration live
alongside for durable master storage.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

_FULL_GIT_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class ImageVariant(StrEnum):
    """Build/runtime image variant. Cross-variant scoring is always a miss."""

    CPU = "cpu"
    CUDA = "cuda"


class AllowlistMissReason(StrEnum):
    """Machine-consumed miss codes for allowlist lookup."""

    UNKNOWN_DIGEST = "unknown_digest"
    VARIANT_MISMATCH = "variant_mismatch"
    REVOKED = "revoked"
    COMMIT_MISMATCH = "commit_mismatch"


def is_full_git_sha(value: str) -> bool:
    """Return True if ``value`` is a lowercase 40-hex git object id."""
    return bool(_FULL_GIT_SHA_RE.fullmatch(value))


def is_image_digest(value: str) -> bool:
    """Return True if ``value`` is a lowercase ``sha256:<64-hex>`` digest."""
    return bool(_IMAGE_DIGEST_RE.fullmatch(value))


def normalize_image_digest(value: str) -> str:
    """Lowercase a digest string; does not validate shape."""
    return value.strip().lower()


def _require_full_git_sha(field_name: str, value: str) -> str:
    normalized = value.strip().lower()
    if not is_full_git_sha(normalized):
        raise ValueError(
            f"{field_name} must be a full 40-char lowercase hex git SHA, got {value!r}"
        )
    return normalized


def _require_image_digest(value: str) -> str:
    normalized = normalize_image_digest(value)
    if not is_image_digest(normalized):
        raise ValueError(f"digest must be sha256:<64 lowercase hex>, got {value!r}")
    return normalized


def _require_variant(value: ImageVariant | str) -> ImageVariant:
    if isinstance(value, ImageVariant):
        return value
    try:
        return ImageVariant(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"variant must be one of {[v.value for v in ImageVariant]}, got {value!r}"
        ) from exc


def _require_sealed_manifest_hashes(
    value: Mapping[str, str],
) -> Mapping[str, str]:
    """Validate path → 64-hex lowercase sealed hashes; reject empty maps."""
    if not isinstance(value, Mapping):
        raise ValueError("sealed_manifest_hashes must be a mapping")
    if not value:
        raise ValueError("sealed_manifest_hashes must be non-empty")
    out: dict[str, str] = {}
    for raw_path, raw_hash in value.items():
        path = str(raw_path).strip()
        if not path:
            raise ValueError(
                "sealed_manifest_hashes path keys must be non-blank strings"
            )
        digest = str(raw_hash).strip().lower()
        if not _HEX64_RE.fullmatch(digest):
            raise ValueError(
                f"sealed_manifest_hashes[{path!r}] must be 64-char lowercase hex, "
                f"got {raw_hash!r}"
            )
        if path in out:
            raise ValueError(f"duplicate sealed_manifest_hashes path: {path!r}")
        out[path] = digest
    return MappingProxyType(out)


@dataclass(frozen=True, slots=True)
class DigestRecord:
    """One BASE-produced image binding.

    Keyed conceptually as ``(commit_sha, tree_sha, variant, digest)``. A given
    digest maps to exactly one binding. ``sealed_manifest_hashes`` is the
    build-time sealed surface (path → 64-hex SHA-256) prism compares as
    ``expected_sealed_manifest_hashes``; must be non-empty at registration.
    """

    commit_sha: str
    tree_sha: str
    variant: ImageVariant
    digest: str
    sealed_manifest_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "commit_sha", _require_full_git_sha("commit_sha", self.commit_sha)
        )
        object.__setattr__(
            self, "tree_sha", _require_full_git_sha("tree_sha", self.tree_sha)
        )
        object.__setattr__(self, "variant", _require_variant(self.variant))
        object.__setattr__(self, "digest", _require_image_digest(self.digest))
        object.__setattr__(
            self,
            "sealed_manifest_hashes",
            _require_sealed_manifest_hashes(self.sealed_manifest_hashes),
        )


@dataclass(frozen=True, slots=True)
class AllowlistHit:
    """Lookup succeeded: digest is BASE-produced for the requested binding."""

    record: DigestRecord


@dataclass(frozen=True, slots=True)
class AllowlistMiss:
    """Lookup failed with a distinct machine reason."""

    reason: AllowlistMissReason


AllowlistLookupResult = AllowlistHit | AllowlistMiss


@dataclass(frozen=True, slots=True)
class AllowlistSnapshot:
    """Serializable state for DB/file adapters."""

    records: tuple[DigestRecord, ...]
    denied_digests: frozenset[str]
    denied_commits: frozenset[str]


@dataclass
class DigestAllowlist:
    """In-memory digest allowlist with revocation denylists.

    Pure register / lookup / revoke surface intended for prism and BASE build
    pipeline callers. Not a network client and not attestation.
    """

    _by_digest: dict[str, DigestRecord] = field(default_factory=dict, init=False)
    _denied_digests: set[str] = field(default_factory=set, init=False)
    _denied_commits: set[str] = field(default_factory=set, init=False)

    def register(self, record: DigestRecord) -> None:
        """Record a BASE-produced digest binding.

        Re-registering the identical binding is a no-op. Binding the same digest
        to a different commit/tree/variant raises ``ValueError``.
        """
        existing = self._by_digest.get(record.digest)
        if existing is not None and existing != record:
            raise ValueError(
                f"digest {record.digest!r} already bound to "
                f"commit={existing.commit_sha} tree={existing.tree_sha} "
                f"variant={existing.variant.value}; cannot rebind to "
                f"commit={record.commit_sha} tree={record.tree_sha} "
                f"variant={record.variant.value}"
            )
        self._by_digest[record.digest] = record

    def revoke_digest(self, digest: str) -> None:
        """Deny a single image digest permanently (post-hoc revocation)."""
        self._denied_digests.add(_require_image_digest(digest))

    def revoke_commit(self, commit_sha: str) -> None:
        """Deny every digest bound to ``commit_sha`` (and future registrations)."""
        self._denied_commits.add(_require_full_git_sha("commit_sha", commit_sha))

    def lookup(
        self,
        *,
        digest: str,
        commit_sha: str,
        tree_sha: str,
        variant: ImageVariant | str,
    ) -> AllowlistLookupResult:
        """Return hit only when digest matches registered commit, tree, and variant.

        Check order (first match wins):
        1. revoked digest or revoked commit → ``revoked``
        2. digest absent → ``unknown_digest``
        3. commit_sha or tree_sha differ → ``commit_mismatch``
        4. variant differs → ``variant_mismatch``
        5. else hit
        """
        dig = _require_image_digest(digest)
        commit = _require_full_git_sha("commit_sha", commit_sha)
        tree = _require_full_git_sha("tree_sha", tree_sha)
        want_variant = _require_variant(variant)

        if dig in self._denied_digests or commit in self._denied_commits:
            return AllowlistMiss(reason=AllowlistMissReason.REVOKED)

        record = self._by_digest.get(dig)
        if record is None:
            return AllowlistMiss(reason=AllowlistMissReason.UNKNOWN_DIGEST)

        # Also revoke if the *registered* commit is denied (lookup commit may
        # already have matched above; this covers digest-only lookups that pass
        # a non-denied commit while the binding's commit is denied).
        if record.commit_sha in self._denied_commits:
            return AllowlistMiss(reason=AllowlistMissReason.REVOKED)

        if record.commit_sha != commit or record.tree_sha != tree:
            return AllowlistMiss(reason=AllowlistMissReason.COMMIT_MISMATCH)

        if record.variant is not want_variant:
            return AllowlistMiss(reason=AllowlistMissReason.VARIANT_MISMATCH)

        return AllowlistHit(record=record)

    def snapshot(self) -> AllowlistSnapshot:
        """Export durable state for persistence adapters."""
        return AllowlistSnapshot(
            records=tuple(self._by_digest.values()),
            denied_digests=frozenset(self._denied_digests),
            denied_commits=frozenset(self._denied_commits),
        )

    @classmethod
    def from_snapshot(cls, snapshot: AllowlistSnapshot) -> DigestAllowlist:
        """Restore a registry from :meth:`snapshot` output."""
        registry = cls()
        for record in snapshot.records:
            registry.register(record)
        for digest in snapshot.denied_digests:
            registry.revoke_digest(digest)
        for commit in snapshot.denied_commits:
            registry.revoke_commit(commit)
        return registry

    def load_records(self, records: Mapping[str, DigestRecord] | None = None) -> None:
        """Bulk-load records (used by DB adapters)."""
        if not records:
            return
        for record in records.values():
            self.register(record)
