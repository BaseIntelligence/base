"""Encrypted internal storage for raw bounded review transport objects."""

from __future__ import annotations

import base64
import secrets
from collections.abc import Mapping
from hashlib import sha256

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.core.models import ReviewAssignment, ReviewEvidenceObject, ReviewSession
from agent_challenge.sdk.config import ChallengeSettings

from .schemas import (
    MAX_OPENROUTER_METADATA_BYTES,
    MAX_OPENROUTER_REQUEST_BYTES,
    MAX_OPENROUTER_RESPONSE_BYTES,
)

MAX_REVIEW_EVIDENCE_BYTES = 6 * 1024 * 1024
REVIEW_EVIDENCE_ENCRYPTION_PROFILE = "review-evidence-fernet-v1"


def _evidence_limits(settings: ChallengeSettings | None = None) -> dict[str, int]:
    """Resolve per-object evidence caps, preferring live ChallengeSettings."""

    if settings is None:
        return {
            "planned_request": MAX_OPENROUTER_REQUEST_BYTES,
            "transport_observation": 256 * 1024,
            "request_body": MAX_OPENROUTER_REQUEST_BYTES,
            "response_body": MAX_OPENROUTER_RESPONSE_BYTES,
            "metadata": MAX_OPENROUTER_METADATA_BYTES,
        }
    return {
        "planned_request": int(settings.review_max_openrouter_request_bytes),
        "transport_observation": int(settings.review_max_openrouter_metadata_bytes),
        "request_body": int(settings.review_max_openrouter_request_bytes),
        "response_body": int(settings.review_max_openrouter_response_bytes),
        "metadata": int(settings.review_max_openrouter_metadata_bytes),
    }


class ReviewEvidenceError(ValueError):
    """Evidence storage or read cannot prove its immutable encrypted binding."""


async def store_review_evidence_objects(
    session: AsyncSession,
    *,
    assignment: ReviewAssignment,
    settings: ChallengeSettings,
    objects: Mapping[str, bytes],
) -> dict[str, dict[str, object]]:
    """Encrypt exact raw objects atomically, returning credential-free descriptors.

    The aggregate cap is enforced against Fernet ciphertext expansion plus the
    persisted descriptor JSON, not plaintext lengths alone.
    """

    limits = _evidence_limits(settings)
    if not objects or not set(objects) <= set(limits):
        raise ReviewEvidenceError("review evidence object kinds are invalid")
    for kind, value in objects.items():
        if not isinstance(value, bytes) or not value:
            raise ReviewEvidenceError("review evidence must contain non-empty bytes")
        if len(value) > limits[kind]:
            raise ReviewEvidenceError("review evidence object exceeds its configured bound")
    aggregate_cap = int(
        getattr(settings, "review_max_encrypted_evidence_bytes", MAX_REVIEW_EVIDENCE_BYTES)
    )
    fernet = _evidence_fernet(settings)
    pending: list[tuple[str, bytes, str, bytes | ReviewEvidenceObject]] = []
    descriptors: dict[str, dict[str, object]] = {}
    ciphertext_total = 0
    for kind, value in objects.items():
        digest = sha256(value).hexdigest()
        row = await session.scalar(
            select(ReviewEvidenceObject)
            .where(ReviewEvidenceObject.assignment_id == assignment.id)
            .where(ReviewEvidenceObject.object_kind == kind)
        )
        if row is not None:
            if row.sha256 != digest or row.size_bytes != len(value):
                raise ReviewEvidenceError(
                    "review evidence object conflicts with immutable prior bytes"
                )
            ciphertext_total += len(row.ciphertext)
            descriptors[kind] = {
                "object_ref": row.object_ref,
                "sha256": row.sha256,
                "length": row.size_bytes,
            }
            pending.append((kind, value, digest, row))
            continue
        ciphertext = fernet.encrypt(value)
        ciphertext_total += len(ciphertext)
        object_ref = f"re_{secrets.token_urlsafe(24)}"
        descriptors[kind] = {
            "object_ref": object_ref,
            "sha256": digest,
            "length": len(value),
        }
        pending.append((kind, value, digest, ciphertext))
    # Descriptor bytes are part of the architecture's encrypted-evidence aggregate.
    import json

    descriptor_bytes = json.dumps(descriptors, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if ciphertext_total + len(descriptor_bytes) > aggregate_cap:
        raise ReviewEvidenceError("review evidence aggregate exceeds configured bound")
    for kind, value, digest, ciphertext_or_row in pending:
        if isinstance(ciphertext_or_row, ReviewEvidenceObject):
            continue
        row = ReviewEvidenceObject(
            object_ref=str(descriptors[kind]["object_ref"]),
            session_id=assignment.session_id,
            assignment_id=assignment.id,
            object_kind=kind,
            sha256=digest,
            size_bytes=len(value),
            encryption_profile=REVIEW_EVIDENCE_ENCRYPTION_PROFILE,
            ciphertext=ciphertext_or_row,
        )
        session.add(row)
        await session.flush()
    return descriptors


async def load_review_evidence_object(
    session: AsyncSession,
    *,
    review_session: ReviewSession,
    object_ref: str,
    settings: ChallengeSettings,
) -> tuple[ReviewEvidenceObject, bytes]:
    """Decrypt exactly one opaque object ref only within its owning session."""

    row = await session.scalar(
        select(ReviewEvidenceObject)
        .where(ReviewEvidenceObject.session_id == review_session.id)
        .where(ReviewEvidenceObject.object_ref == object_ref)
    )
    if row is None:
        raise ReviewEvidenceError("review evidence object was not found")
    if row.encryption_profile != REVIEW_EVIDENCE_ENCRYPTION_PROFILE:
        raise ReviewEvidenceError("review evidence encryption profile is unsupported")
    try:
        value = _evidence_fernet(settings).decrypt(row.ciphertext)
    except InvalidToken as exc:
        raise ReviewEvidenceError("review evidence cannot be authenticated") from exc
    if len(value) != row.size_bytes or sha256(value).hexdigest() != row.sha256:
        raise ReviewEvidenceError("review evidence bytes do not match immutable descriptor")
    return row, value


def _evidence_fernet(settings: ChallengeSettings) -> Fernet:
    """Derive Fernet material only from the dedicated evidence encryption key."""

    try:
        secret = settings.load_review_evidence_encryption_key()
    except ValueError as exc:
        raise ReviewEvidenceError("review evidence encryption key is unavailable") from exc
    material = sha256(b"agent-challenge:review-evidence:v1:" + secret.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(material)
    return Fernet(key)


__all__ = [
    "MAX_REVIEW_EVIDENCE_BYTES",
    "REVIEW_EVIDENCE_ENCRYPTION_PROFILE",
    "ReviewEvidenceError",
    "load_review_evidence_object",
    "store_review_evidence_objects",
]
