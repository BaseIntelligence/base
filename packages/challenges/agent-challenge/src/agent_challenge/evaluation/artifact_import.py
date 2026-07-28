"""Fail-closed guest-side artifact import for the eval CVM.

Fetches the miner ZIP over HTTPS, recomputes agent_hash and package_tree_sha
from real bytes, and materializes the package on disk. Every mismatch raises;
there is no environment-variable fallback and tokens are never logged.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_challenge.canonical.eval_wire import agent_artifact_sha256_hex
from agent_challenge.submissions.artifacts import (
    MAX_ZIP_BYTES,
    ArtifactValidationError,
    compute_package_tree_sha_from_zip_bytes,
    extract_zip_to_directory,
)

# Bound fetch bodies to the same ceiling as submission ZIP validation.
_DEFAULT_MAX_FETCH_BYTES = MAX_ZIP_BYTES


class ArtifactImportError(Exception):
    """Raised when guest-side artifact fetch/verify/materialize fails closed."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        super().__init__(message or reason_code)
        self.reason_code = reason_code
        self.message = message or reason_code


@dataclass(frozen=True)
class ArtifactProof:
    """Guest-computed proof of the bytes written to disk."""

    agent_hash: str
    package_tree_sha: str
    zip_size_bytes: int
    zip_path: Path
    package_root: Path


def fetch_eval_artifact(url: str, token: str, *, timeout: float) -> bytes:
    """HTTPS GET the eval artifact with a bearer token.

    Does not log the token or URL query string. Non-200 and transport failures
    raise ``ArtifactImportError(reason_code="fetch_failed")``. Bodies larger
    than ``MAX_ZIP_BYTES`` are refused.
    """

    if not isinstance(url, str) or not url.startswith("https://"):
        raise ArtifactImportError("fetch_failed", "artifact url must be https")
    if not isinstance(token, str) or not token:
        raise ArtifactImportError("fetch_failed", "artifact token missing")
    if timeout <= 0:
        raise ArtifactImportError("fetch_failed", "timeout must be positive")

    request = Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/zip, application/octet-stream",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 — https-only above
            status = int(getattr(response, "status", 0) or 0)
            if status != 200:
                raise ArtifactImportError("fetch_failed", f"artifact fetch status={status}")
            content_length = _content_length(response)
            if content_length is not None and content_length > _DEFAULT_MAX_FETCH_BYTES:
                raise ArtifactImportError("fetch_failed", "artifact exceeds max size")
            body = _read_bounded(response, max_bytes=_DEFAULT_MAX_FETCH_BYTES)
    except ArtifactImportError:
        raise
    except HTTPError as exc:
        # Do not include response body or URL (may carry credentials in query).
        raise ArtifactImportError(
            "fetch_failed",
            f"artifact fetch status={int(exc.code)}",
        ) from None
    except (URLError, TimeoutError, OSError, ValueError):
        raise ArtifactImportError("fetch_failed", "artifact fetch transport failed") from None

    if not body:
        raise ArtifactImportError("fetch_failed", "artifact body empty")
    return body


def verify_zip_bytes(
    zip_bytes: bytes,
    *,
    expected_agent_hash: str,
    expected_package_tree_sha: str,
) -> ArtifactProof:
    """Recompute both digests from ``zip_bytes`` and compare constant-time.

    Returns an ``ArtifactProof`` with path fields left as empty placeholders
    (``zip_path`` / ``package_root`` are filled by ``materialize_agent_artifact``).
    Callers that only need the digests may ignore the path fields.
    """

    if not isinstance(zip_bytes, (bytes, bytearray)) or not zip_bytes:
        raise ArtifactImportError("digest_mismatch", "artifact bytes missing")

    raw = bytes(zip_bytes)
    try:
        actual_agent_hash = agent_artifact_sha256_hex(raw)
        actual_tree_sha = compute_package_tree_sha_from_zip_bytes(raw)
    except (ArtifactValidationError, ValueError, TypeError) as exc:
        raise ArtifactImportError("digest_mismatch", "artifact bytes unreadable") from exc

    if not _digest_equal(actual_agent_hash, expected_agent_hash):
        raise ArtifactImportError("digest_mismatch", "agent_hash mismatch")
    if not _digest_equal(actual_tree_sha, expected_package_tree_sha):
        raise ArtifactImportError("tree_mismatch", "package_tree_sha mismatch")

    return ArtifactProof(
        agent_hash=actual_agent_hash,
        package_tree_sha=actual_tree_sha,
        zip_size_bytes=len(raw),
        zip_path=Path(),
        package_root=Path(),
    )


def materialize_agent_artifact(
    zip_bytes: bytes,
    *,
    zip_dest: Path,
    package_dest: Path,
) -> ArtifactProof:
    """Write the ZIP, extract the package tree, return proof of written bytes."""

    if not isinstance(zip_bytes, (bytes, bytearray)) or not zip_bytes:
        raise ArtifactImportError("digest_mismatch", "artifact bytes missing")
    raw = bytes(zip_bytes)
    if len(raw) > _DEFAULT_MAX_FETCH_BYTES:
        raise ArtifactImportError("digest_mismatch", "artifact exceeds max size")

    try:
        agent_hash = agent_artifact_sha256_hex(raw)
        package_tree_sha = compute_package_tree_sha_from_zip_bytes(raw)
    except (ArtifactValidationError, ValueError, TypeError) as exc:
        raise ArtifactImportError("digest_mismatch", "artifact bytes unreadable") from exc

    zip_path = Path(zip_dest)
    package_root = Path(package_dest)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(raw)

    # Re-hash the bytes actually on disk (fail closed if write was truncated).
    written = zip_path.read_bytes()
    if not hmac.compare_digest(written, raw):
        raise ArtifactImportError("digest_mismatch", "zip write integrity failed")

    try:
        extract_zip_to_directory(zip_path=zip_path, target_directory=package_root)
    except ArtifactValidationError as exc:
        raise ArtifactImportError("digest_mismatch", "artifact extract failed") from exc

    return ArtifactProof(
        agent_hash=agent_hash,
        package_tree_sha=package_tree_sha,
        zip_size_bytes=len(written),
        zip_path=zip_path,
        package_root=package_root,
    )


def _digest_equal(actual: str, expected: str) -> bool:
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    actual_b = actual.encode("utf-8")
    expected_b = expected.encode("utf-8")
    if len(actual_b) != len(expected_b):
        # compare_digest requires equal length; unequal length is a mismatch.
        return False
    return hmac.compare_digest(actual_b, expected_b)


def _content_length(response: object) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Content-Length") if hasattr(headers, "get") else None
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _read_bounded(response: object, *, max_bytes: int) -> bytes:
    read = getattr(response, "read", None)
    if not callable(read):
        raise ArtifactImportError("fetch_failed", "artifact response unreadable")
    # Read one extra byte past the bound so oversize bodies are detected.
    chunk = read(max_bytes + 1)
    if not isinstance(chunk, (bytes, bytearray)):
        raise ArtifactImportError("fetch_failed", "artifact response unreadable")
    data = bytes(chunk)
    if len(data) > max_bytes:
        raise ArtifactImportError("fetch_failed", "artifact exceeds max size")
    return data
