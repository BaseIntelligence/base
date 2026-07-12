"""Hotkey-signed + metagraph-permit auth for validator-facing master routes.

The canonical signed-request scheme matches the one used across the repos:
headers ``X-Hotkey``/``X-Signature``/``X-Nonce``/``X-Timestamp`` with the
canonical string::

    METHOD\\nPATH_WITH_SORTED_QUERY\\nTIMESTAMP\\nNONCE\\nSHA256_HEX(body)

Requests are rejected with ``401`` for signed-request failures (missing header,
bad signature, tampered body, stale timestamp), ``409`` for a replayed
``(hotkey, nonce)`` pair, and ``403`` when the hotkey is not an eligible
validator (absent from the metagraph or lacking a validator permit).
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Protocol
from urllib.parse import parse_qsl, urlencode

from fastapi import HTTPException, Request, status
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.bittensor.metagraph_cache import MetagraphCache
from base.db.models import ValidatorRequestNonce
from base.db.session import session_scope
from base.security.miner_auth import SignatureVerifier, verify_substrate_signature

HOTKEY_HEADER = "X-Hotkey"
SIGNATURE_HEADER = "X-Signature"
NONCE_HEADER = "X-Nonce"
TIMESTAMP_HEADER = "X-Timestamp"

INVALID_AUTH_DETAIL = "invalid signed request"
REPLAY_AUTH_DETAIL = "replayed request"
INELIGIBLE_DETAIL = "validator not eligible"


class ValidatorAuthError(ValueError):
    """Signed-request authentication failure (maps to HTTP 401)."""


class ValidatorReplayError(ValidatorAuthError):
    """A previously-seen ``(hotkey, nonce)`` was reused (maps to HTTP 409)."""


class ValidatorEligibilityError(ValueError):
    """Hotkey is not an eligible validator (maps to HTTP 403)."""


@dataclass(frozen=True)
class ValidatorIdentity:
    """Verified identity of a validator-facing signed request."""

    hotkey: str
    uid: int | None
    nonce: str
    timestamp: int
    body_hash: str
    canonical_request: str


class ValidatorNonceStore(Protocol):
    async def reserve(
        self,
        *,
        hotkey: str,
        nonce: str,
        body_hash: str,
        created_at: datetime,
    ) -> None: ...


class ValidatorEligibility(Protocol):
    def is_eligible(self, hotkey: str) -> bool: ...

    def uid_for(self, hotkey: str) -> int | None: ...


class SqlAlchemyValidatorNonceStore:
    """Persisted ``(hotkey, nonce)`` replay guard for validator requests.

    Exact retry of the same signed body (same body hash) is accepted as an
    idempotent reserve so registration/result delivery can safely re-POST with
    the original nonce. Reuse of a nonce with different content is a conflict.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        ttl_seconds: int = 86_400,
    ) -> None:
        self.session_factory = session_factory
        self.ttl_seconds = ttl_seconds

    async def reserve(
        self,
        *,
        hotkey: str,
        nonce: str,
        body_hash: str,
        created_at: datetime,
    ) -> None:
        from sqlalchemy import select

        cutoff = created_at - timedelta(seconds=self.ttl_seconds)
        try:
            async with session_scope(self.session_factory) as session:
                await session.execute(
                    delete(ValidatorRequestNonce).where(
                        ValidatorRequestNonce.created_at < cutoff
                    )
                )
                existing = (
                    await session.execute(
                        select(ValidatorRequestNonce).where(
                            ValidatorRequestNonce.hotkey == hotkey,
                            ValidatorRequestNonce.nonce == nonce,
                        )
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    if existing.body_hash == body_hash:
                        return
                    raise ValidatorReplayError("nonce already used with different body")
                session.add(
                    ValidatorRequestNonce(
                        id=uuid.uuid4(),
                        hotkey=hotkey,
                        nonce=nonce,
                        body_hash=body_hash,
                        created_at=created_at,
                    )
                )
        except IntegrityError as exc:
            raise ValidatorReplayError("nonce already used") from exc


@dataclass
class MetagraphValidatorEligibility:
    """Eligibility backed by the metagraph cache (on-graph AND permitted)."""

    metagraph_cache: MetagraphCache

    def is_eligible(self, hotkey: str) -> bool:
        self.metagraph_cache.get()
        return self.metagraph_cache.is_validator(hotkey)

    def uid_for(self, hotkey: str) -> int | None:
        return self.metagraph_cache.hotkey_to_uid.get(hotkey)


def body_sha256(body: bytes) -> str:
    return sha256(body).hexdigest()


def sorted_path_with_query(path: str, query_string: str | bytes = "") -> str:
    if isinstance(query_string, bytes):
        query_string = query_string.decode("utf-8")
    if not query_string:
        return path
    pairs = sorted(parse_qsl(query_string, keep_blank_values=True))
    return f"{path}?{urlencode(pairs)}"


def canonical_validator_request(
    *,
    method: str,
    path: str,
    query_string: str | bytes,
    timestamp: str,
    nonce: str,
    body: bytes,
) -> str:
    return "\n".join(
        (
            method.upper(),
            sorted_path_with_query(path, query_string),
            timestamp,
            nonce,
            body_sha256(body),
        )
    )


class ValidatorSignedRequestVerifier:
    """Verify a hotkey-signed, metagraph-permit-gated validator request."""

    def __init__(
        self,
        *,
        nonce_store: ValidatorNonceStore,
        eligibility: ValidatorEligibility,
        signature_verifier: SignatureVerifier = verify_substrate_signature,
        ttl_seconds: int = 300,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.nonce_store = nonce_store
        self.eligibility = eligibility
        self.signature_verifier = signature_verifier
        self.ttl_seconds = ttl_seconds
        self.now_fn = now_fn

    async def verify(
        self,
        *,
        method: str,
        path: str,
        query_string: str | bytes,
        headers: Mapping[str, str],
        body: bytes,
    ) -> ValidatorIdentity:
        hotkey = _required_header(headers, HOTKEY_HEADER)
        signature = _required_header(headers, SIGNATURE_HEADER)
        nonce = _required_header(headers, NONCE_HEADER)
        timestamp_raw = _required_header(headers, TIMESTAMP_HEADER)

        timestamp = _parse_epoch(timestamp_raw)
        now = float(self.now_fn())
        if abs(now - timestamp) > self.ttl_seconds:
            raise ValidatorAuthError("stale signature")

        canonical_request = canonical_validator_request(
            method=method,
            path=path,
            query_string=query_string,
            timestamp=timestamp_raw,
            nonce=nonce,
            body=body,
        )
        if not self.signature_verifier(hotkey, canonical_request.encode(), signature):
            raise ValidatorAuthError("invalid signature")

        if not self.eligibility.is_eligible(hotkey):
            raise ValidatorEligibilityError("hotkey is not an eligible validator")

        body_hash = body_sha256(body)
        await self.nonce_store.reserve(
            hotkey=hotkey,
            nonce=nonce,
            body_hash=body_hash,
            created_at=datetime.fromtimestamp(now, UTC),
        )
        return ValidatorIdentity(
            hotkey=hotkey,
            uid=self.eligibility.uid_for(hotkey),
            nonce=nonce,
            timestamp=int(timestamp),
            body_hash=body_hash,
            canonical_request=canonical_request,
        )


def build_validator_auth_dependency(
    verifier: ValidatorSignedRequestVerifier,
) -> Callable[[Request], object]:
    """Build a FastAPI dependency enforcing validator signed-request auth."""

    async def authenticate(request: Request) -> ValidatorIdentity:
        body = await request.body()
        try:
            return await verifier.verify(
                method=request.method,
                path=request.url.path,
                query_string=request.url.query,
                headers=request.headers,
                body=body,
            )
        except ValidatorReplayError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=REPLAY_AUTH_DETAIL
            ) from exc
        except ValidatorEligibilityError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=INELIGIBLE_DETAIL
            ) from exc
        except ValidatorAuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=INVALID_AUTH_DETAIL
            ) from exc

    return authenticate


def _parse_epoch(raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValidatorAuthError("invalid timestamp") from exc
    if not math.isfinite(value):
        raise ValidatorAuthError("invalid timestamp")
    return value


def _required_header(headers: Mapping[str, str], name: str) -> str:
    value = headers.get(name) or headers.get(name.lower())
    if not value or not value.strip():
        raise ValidatorAuthError(f"missing {name}")
    return value.strip()
