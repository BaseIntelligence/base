"""TDD tests for BASE-produced image digest allowlist with revocation.

This is an allowlist only (mechanism 4). It does not perform constation or
attestation; it records digests BASE built and answers lookup with distinct
miss reasons so prism can fail closed later.
"""

from __future__ import annotations

import pytest

from base.compute.digest_allowlist import (
    AllowlistHit,
    AllowlistMiss,
    AllowlistMissReason,
    DigestAllowlist,
    DigestRecord,
    ImageVariant,
    is_full_git_sha,
    is_image_digest,
    normalize_image_digest,
)

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


def test_register_and_lookup_hit_for_matching_commit_and_variant() -> None:
    """Given a BASE-registered cuda digest, When lookup matches, Then hit."""
    registry = DigestAllowlist()
    record = _record(variant=ImageVariant.CUDA, digest=DIGEST_CUDA)
    registry.register(record)

    result = registry.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )

    assert isinstance(result, AllowlistHit)
    assert result.record == record


def test_lookup_unknown_digest_misses() -> None:
    registry = DigestAllowlist()
    registry.register(_record())

    result = registry.lookup(
        digest=DIGEST_OTHER,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )

    assert isinstance(result, AllowlistMiss)
    assert result.reason is AllowlistMissReason.UNKNOWN_DIGEST


def test_cross_variant_scoring_rejected_with_variant_mismatch() -> None:
    """cpu digest cannot score as cuda (and vice versa)."""
    registry = DigestAllowlist()
    registry.register(
        _record(variant=ImageVariant.CPU, digest=DIGEST_CPU),
    )

    result = registry.lookup(
        digest=DIGEST_CPU,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )

    assert isinstance(result, AllowlistMiss)
    assert result.reason is AllowlistMissReason.VARIANT_MISMATCH


def test_commit_mismatch_when_digest_bound_to_other_commit() -> None:
    registry = DigestAllowlist()
    registry.register(_record(commit_sha=COMMIT_A, tree_sha=TREE_A))

    result = registry.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_B,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )

    assert isinstance(result, AllowlistMiss)
    assert result.reason is AllowlistMissReason.COMMIT_MISMATCH


def test_tree_sha_mismatch_is_commit_mismatch() -> None:
    registry = DigestAllowlist()
    registry.register(_record(commit_sha=COMMIT_A, tree_sha=TREE_A))

    result = registry.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_A,
        tree_sha=TREE_B,
        variant=ImageVariant.CUDA,
    )

    assert isinstance(result, AllowlistMiss)
    assert result.reason is AllowlistMissReason.COMMIT_MISMATCH


def test_revoked_digest_never_hits() -> None:
    registry = DigestAllowlist()
    registry.register(_record())
    registry.revoke_digest(DIGEST_CUDA)

    result = registry.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )

    assert isinstance(result, AllowlistMiss)
    assert result.reason is AllowlistMissReason.REVOKED


def test_revoked_commit_never_hits_any_digest_for_that_commit() -> None:
    registry = DigestAllowlist()
    registry.register(_record(digest=DIGEST_CUDA))
    registry.register(
        _record(variant=ImageVariant.CPU, digest=DIGEST_CPU),
    )
    registry.revoke_commit(COMMIT_A)

    for digest, variant in (
        (DIGEST_CUDA, ImageVariant.CUDA),
        (DIGEST_CPU, ImageVariant.CPU),
    ):
        result = registry.lookup(
            digest=digest,
            commit_sha=COMMIT_A,
            tree_sha=TREE_A,
            variant=variant,
        )
        assert isinstance(result, AllowlistMiss)
        assert result.reason is AllowlistMissReason.REVOKED


def test_revoked_takes_precedence_over_variant_mismatch() -> None:
    registry = DigestAllowlist()
    registry.register(_record(variant=ImageVariant.CPU, digest=DIGEST_CPU))
    registry.revoke_digest(DIGEST_CPU)

    result = registry.lookup(
        digest=DIGEST_CPU,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )

    assert isinstance(result, AllowlistMiss)
    assert result.reason is AllowlistMissReason.REVOKED


def test_register_rejects_invalid_shapes() -> None:
    registry = DigestAllowlist()
    with pytest.raises(ValueError, match="commit_sha"):
        registry.register(_record(commit_sha="main"))
    with pytest.raises(ValueError, match="tree_sha"):
        registry.register(_record(tree_sha="short"))
    with pytest.raises(ValueError, match="digest"):
        registry.register(_record(digest="latest"))


def test_register_rejects_duplicate_digest_with_conflicting_binding() -> None:
    registry = DigestAllowlist()
    registry.register(_record(commit_sha=COMMIT_A, digest=DIGEST_CUDA))
    with pytest.raises(ValueError, match="digest"):
        registry.register(_record(commit_sha=COMMIT_B, digest=DIGEST_CUDA))


def test_idempotent_reregister_same_record() -> None:
    registry = DigestAllowlist()
    record = _record()
    registry.register(record)
    registry.register(record)
    result = registry.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )
    assert isinstance(result, AllowlistHit)


def test_normalize_and_validators() -> None:
    assert is_full_git_sha(COMMIT_A)
    assert not is_full_git_sha("abc")
    assert is_image_digest(DIGEST_CUDA)
    assert not is_image_digest("sha256:xyz")
    assert normalize_image_digest("SHA256:" + ("a" * 64)) == "sha256:" + ("a" * 64)


def test_snapshot_roundtrip_for_persistence_boundary() -> None:
    """Pure snapshot so a DB/file adapter can load without re-implementing rules."""
    registry = DigestAllowlist()
    registry.register(_record())
    registry.register(_record(variant=ImageVariant.CPU, digest=DIGEST_CPU))
    registry.revoke_digest(DIGEST_OTHER)
    registry.revoke_commit(COMMIT_B)

    snap = registry.snapshot()
    restored = DigestAllowlist.from_snapshot(snap)

    hit = restored.lookup(
        digest=DIGEST_CUDA,
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA,
    )
    assert isinstance(hit, AllowlistHit)
    assert DIGEST_OTHER in snap.denied_digests
    assert COMMIT_B in snap.denied_commits


def test_orm_models_and_metadata_tables_exist() -> None:
    """Durable tables match the pure registry concepts (migration 0017)."""
    from base.db import (
        Base,
        DeniedImageCommit,
        DeniedImageDigest,
        ImageDigestAllowlistEntry,
    )

    entry = ImageDigestAllowlistEntry(
        commit_sha=COMMIT_A,
        tree_sha=TREE_A,
        variant=ImageVariant.CUDA.value,
        digest=DIGEST_CUDA,
    )
    assert entry.digest == DIGEST_CUDA
    assert "image_digest_allowlist" in Base.metadata.tables
    assert "denied_image_digests" in Base.metadata.tables
    assert "denied_image_commits" in Base.metadata.tables
    assert DeniedImageDigest.__tablename__ == "denied_image_digests"
    assert DeniedImageCommit.__tablename__ == "denied_image_commits"


def test_alembic_migration_0017_is_chained_from_watcher_state() -> None:
    import ast
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "0017_image_digest_allowlist.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in {"revision", "down_revision"} and isinstance(
                node.value, ast.Constant
            ):
                values[node.target.id] = node.value.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "revision",
                    "down_revision",
                }:
                    if isinstance(node.value, ast.Constant):
                        values[target.id] = node.value.value
    assert values["revision"] == "0017_digest_allowlist"
    assert values["down_revision"] == "0016_watcher_state"
    text = path.read_text(encoding="utf-8")
    assert "image_digest_allowlist" in text
    assert "denied_image_digests" in text
    assert "denied_image_commits" in text
