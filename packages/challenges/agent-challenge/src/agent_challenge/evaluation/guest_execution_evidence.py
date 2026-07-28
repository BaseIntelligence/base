"""Guest-side dual-hash proof that executed agent bytes match the plan.

The TEE guest recomputes SHA-256 over (1) the artifact bytes as downloaded and
(2) the artifact bytes actually handed to the orchestrator entrypoint. Both must
equal the immutable plan ``agent_hash``. Callers never supply precomputed digests
as proof — only raw bytes (or an on-disk path that is read twice).

The structured :class:`GuestArtifactExecutionEvidence` is the stable input a
later task folds into the attestation envelope. Do not log tokens or grant URLs;
hashes and sizes are safe.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from agent_challenge.canonical.eval_wire import (
    agent_artifact_sha256_hex,
    canonical_json_v1,
)

#: Public schema version for :class:`GuestArtifactExecutionEvidence.to_dict`.
GUEST_ARTIFACT_EXECUTION_EVIDENCE_SCHEMA_VERSION: Final[int] = 1

#: Stable field order for deterministic serialization (canonical JSON keys sorted
#: by the shared profile; this tuple documents the public contract).
GUEST_ARTIFACT_EXECUTION_EVIDENCE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "expected_hash",
    "download_hash",
    "executed_hash",
    "byte_size",
    "match",
)


@dataclass(frozen=True, slots=True)
class GuestArtifactExecutionEvidence:
    """Structured guest proof that download + executed bytes match the plan.

    Field contract (stable — later attestation code binds to these names):

    - ``schema_version``: int, currently ``1``
    - ``expected_hash``: plan ``agent_hash`` (hex SHA-256 of submitted ZIP)
    - ``download_hash``: guest-computed SHA-256 of bytes as downloaded
    - ``executed_hash``: guest-computed SHA-256 of bytes unpacked/handed to entry
    - ``byte_size``: length in bytes of the download observation
    - ``match``: True only when both guest hashes equal ``expected_hash``
    """

    expected_hash: str
    download_hash: str
    executed_hash: str
    byte_size: int
    match: bool
    schema_version: int = GUEST_ARTIFACT_EXECUTION_EVIDENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-ready mapping with the stable public field set."""

        return {
            "schema_version": int(self.schema_version),
            "expected_hash": self.expected_hash,
            "download_hash": self.download_hash,
            "executed_hash": self.executed_hash,
            "byte_size": int(self.byte_size),
            "match": bool(self.match),
        }


def serialize_guest_artifact_execution_evidence(
    evidence: GuestArtifactExecutionEvidence,
) -> bytes:
    """Canonical JSON bytes for ``evidence`` (same input → byte-identical)."""

    if not isinstance(evidence, GuestArtifactExecutionEvidence):
        raise TypeError("evidence must be GuestArtifactExecutionEvidence")
    return canonical_json_v1(evidence.to_dict())


def prove_guest_artifact_execution(
    *,
    plan_agent_hash: str,
    download_bytes: bytes,
    executed_bytes: bytes,
) -> GuestArtifactExecutionEvidence:
    """Recompute download + executed digests from real bytes; fail closed on mismatch.

    Never accepts caller-supplied ``download_hash`` / ``executed_hash`` — those
    names are intentionally absent from the signature so an env-echo or host
    injection cannot substitute for guest computation.
    """

    expected = _require_plan_hash(plan_agent_hash)
    download = _require_bytes(download_bytes, label="download")
    executed = _require_bytes(executed_bytes, label="executed")

    download_hash = agent_artifact_sha256_hex(download)
    executed_hash = agent_artifact_sha256_hex(executed)

    download_ok = _digest_hex_equal(download_hash, expected)
    executed_ok = _digest_hex_equal(executed_hash, expected)
    pair_ok = _digest_hex_equal(download_hash, executed_hash)
    matched = download_ok and executed_ok and pair_ok

    evidence = GuestArtifactExecutionEvidence(
        expected_hash=expected,
        download_hash=download_hash,
        executed_hash=executed_hash,
        byte_size=len(download),
        match=matched,
    )
    if not matched:
        raise ValueError(_mismatch_message(evidence))
    return evidence


def prove_guest_artifact_execution_from_path(
    *,
    plan_agent_hash: str,
    artifact_path: Path | str,
) -> GuestArtifactExecutionEvidence:
    """Dual-read on-disk ZIP: first observation = download, second = executed.

    Two separate reads so a swap between observations is detectable when the
    caller stages download bytes then rewrites the path before the second read
    (or when this helper is split across download vs exec stages).
    """

    if artifact_path is None:
        raise ValueError(
            "agent artifact bytes unavailable; guest cannot verify plan agent_hash "
            "(refusing environment/declared digest echo)"
        )
    path = Path(artifact_path)
    try:
        download_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"agent artifact cannot be read: {path}") from exc
    try:
        executed_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"agent artifact cannot be read for execution: {path}") from exc
    return prove_guest_artifact_execution(
        plan_agent_hash=plan_agent_hash,
        download_bytes=download_bytes,
        executed_bytes=executed_bytes,
    )


def evidence_from_download_and_path(
    *,
    plan_agent_hash: str,
    download_bytes: bytes,
    executed_artifact_path: Path | str,
) -> GuestArtifactExecutionEvidence:
    """Hash download buffer + re-read path bytes at exec time (swap-detecting)."""

    path = Path(executed_artifact_path)
    try:
        executed_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"agent artifact cannot be read for execution: {path}") from exc
    return prove_guest_artifact_execution(
        plan_agent_hash=plan_agent_hash,
        download_bytes=download_bytes,
        executed_bytes=executed_bytes,
    )


def _require_plan_hash(plan_agent_hash: str) -> str:
    if not isinstance(plan_agent_hash, str):
        raise ValueError(
            "immutable Eval plan agent_hash is missing; guest cannot verify artifact identity"
        )
    expected = plan_agent_hash.strip()
    if not expected:
        raise ValueError(
            "immutable Eval plan agent_hash is missing; guest cannot verify artifact identity"
        )
    return expected


def _require_bytes(value: bytes, *, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError(f"agent artifact {label} bytes unavailable")
    raw = bytes(value)
    if not raw:
        raise ValueError(f"agent artifact {label} bytes unavailable")
    return raw


def _digest_hex_equal(actual: str, expected: str) -> bool:
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    actual_b = actual.encode("utf-8")
    expected_b = expected.encode("utf-8")
    if len(actual_b) != len(expected_b):
        return False
    return hmac.compare_digest(actual_b, expected_b)


def _mismatch_message(evidence: GuestArtifactExecutionEvidence) -> str:
    # Hashes and sizes only — never tokens/URLs.
    if not _digest_hex_equal(evidence.download_hash, evidence.expected_hash):
        return (
            "agent artifact download hash does not match immutable Eval plan agent_hash "
            f"(expected {evidence.expected_hash}, got {evidence.download_hash})"
        )
    if not _digest_hex_equal(evidence.executed_hash, evidence.expected_hash):
        return (
            "agent artifact executed hash does not match immutable Eval plan agent_hash "
            f"(expected {evidence.expected_hash}, got {evidence.executed_hash})"
        )
    if not _digest_hex_equal(evidence.download_hash, evidence.executed_hash):
        return (
            "agent artifact download/executed hash mismatch "
            f"(download {evidence.download_hash}, executed {evidence.executed_hash})"
        )
    return (
        "agent artifact does not match immutable Eval plan agent_hash "
        f"(expected {evidence.expected_hash}, "
        f"download {evidence.download_hash}, executed {evidence.executed_hash})"
    )


def evidence_public_mapping(
    evidence: GuestArtifactExecutionEvidence,
) -> Mapping[str, Any]:
    """Readonly view of the public evidence dict (attestation fold input)."""

    return evidence.to_dict()
