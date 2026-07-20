"""Assignment-scoped verification of immutable content-addressed submissions."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from agent_challenge.core.models import ReviewAssignment, ReviewSession, SubmissionArtifact
from agent_challenge.submissions.artifacts import ArtifactValidationError, build_zip_manifest

from .canonical import canonical_json_v1


class ReviewArtifactError(ValueError):
    """The committed artifact cannot satisfy the immutable assignment binding."""


def manifest_digests(metadata_json: str) -> tuple[str, str]:
    """Return canonical manifest and normalized-entry digests from stored metadata."""

    try:
        metadata = json.loads(metadata_json)
        manifest = metadata["manifest"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ReviewArtifactError("missing committed artifact manifest") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
        raise ReviewArtifactError("malformed committed artifact manifest")
    return (
        sha256(canonical_json_v1(manifest)).hexdigest(),
        sha256(canonical_json_v1(manifest["entries"])).hexdigest(),
    )


def load_assignment_artifact(
    *,
    assignment: ReviewAssignment,
    review_session: ReviewSession,
    artifact: SubmissionArtifact,
) -> bytes:
    """Read and independently revalidate exactly the committed ZIP bytes."""

    if (
        artifact.sha256 != review_session.artifact_sha256
        or artifact.size_bytes != review_session.artifact_size_bytes
    ):
        raise ReviewArtifactError("stored artifact metadata does not match review session")
    manifest_digest, entries_digest = manifest_digests(artifact.metadata_json)
    if (
        manifest_digest != review_session.manifest_sha256
        or entries_digest != review_session.manifest_entries_sha256
    ):
        raise ReviewArtifactError("stored artifact manifest does not match review session")
    try:
        zip_bytes = Path(artifact.uri).read_bytes()
    except OSError as exc:
        raise ReviewArtifactError("committed artifact is unavailable") from exc
    if (
        len(zip_bytes) != review_session.artifact_size_bytes
        or sha256(zip_bytes).hexdigest() != review_session.artifact_sha256
    ):
        raise ReviewArtifactError("committed artifact bytes changed")
    try:
        rebuilt = build_zip_manifest(
            zip_bytes=zip_bytes,
            artifact_reference=artifact.uri,
            zip_sha256=review_session.artifact_sha256,
        )
    except ArtifactValidationError as exc:
        raise ReviewArtifactError("committed artifact archive is invalid") from exc
    rebuilt_manifest = rebuilt.to_dict()
    if (
        sha256(canonical_json_v1(rebuilt_manifest)).hexdigest() != review_session.manifest_sha256
        or sha256(canonical_json_v1(rebuilt_manifest["entries"])).hexdigest()
        != review_session.manifest_entries_sha256
    ):
        raise ReviewArtifactError("committed artifact manifest changed")
    if assignment.artifact_sha256 != review_session.artifact_sha256:
        raise ReviewArtifactError("assignment artifact digest changed")
    return zip_bytes
