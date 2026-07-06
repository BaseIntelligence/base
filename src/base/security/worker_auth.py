"""Worker-plane auth: miner binding verification + worker signed requests.

Two distinct authentications back the master worker surface (architecture.md
sec 3.3):

* **Registration** is authenticated by the MINER's sr25519 signature over the
  binding message ``worker-binding:{worker_pubkey}:{miner_hotkey}:{nonce}``,
  verified against the (mock) metagraph. See :func:`worker_binding_message` and
  :class:`MetagraphMinerMembership`.
* **Heartbeat + fleet reads** are authenticated by a hotkey-signed request in the
  canonical scheme shared with the validator plane
  (``METHOD\\nPATH\\nTIMESTAMP\\nNONCE\\nSHA256_HEX(body)``). The signer is the
  WORKER keypair (heartbeat) or any registered worker / eligible validator (fleet
  reads); the gate is registration status, NOT a metagraph validator permit.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from fastapi import HTTPException, Request, status
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.bittensor.metagraph_cache import MetagraphCache
from base.db.models import WorkerRegistration, WorkerRequestNonce
from base.db.session import session_scope
from base.security.miner_auth import SignatureVerifier, verify_substrate_signature
from base.security.validator_auth import (
    body_sha256,
    canonical_validator_request,
)

HOTKEY_HEADER = "X-Hotkey"
SIGNATURE_HEADER = "X-Signature"
NONCE_HEADER = "X-Nonce"
TIMESTAMP_HEADER = "X-Timestamp"

WORKER_BINDING_MESSAGE_PREFIX = "worker-binding"

INVALID_AUTH_DETAIL = "invalid signed request"
REPLAY_AUTH_DETAIL = "replayed request"
INELIGIBLE_DETAIL = "worker not eligible"


def worker_binding_message(
    *, worker_pubkey: str, miner_hotkey: str, nonce: str
) -> bytes:
    """Canonical miner binding message signed by the miner hotkey (sr25519)."""

    return (
        f"{WORKER_BINDING_MESSAGE_PREFIX}:{worker_pubkey}:{miner_hotkey}:{nonce}"
    ).encode()


class WorkerAuthError(ValueError):
    """Signed-request authentication failure (maps to HTTP 401)."""


class WorkerReplayError(WorkerAuthError):
    """A previously-seen ``(hotkey, nonce)`` was reused (maps to HTTP 409)."""


class WorkerEligibilityError(ValueError):
    """Signer is neither a registered worker nor an eligible validator (403)."""


@dataclass(frozen=True)
class WorkerIdentity:
    """Verified identity of a worker-plane signed request."""

    hotkey: str
    nonce: str
    timestamp: int
    body_hash: str
    canonical_request: str


class WorkerNonceStore(Protocol):
    async def reserve(
        self,
        *,
        hotkey: str,
        nonce: str,
        body_hash: str,
        created_at: datetime,
    ) -> None: ...


class WorkerAuthEligibility(Protocol):
    async def is_eligible(self, hotkey: str) -> bool: ...


@dataclass
class MetagraphMinerMembership:
    """Miner membership backed by the metagraph cache (on-graph, no permit).

    The binding is signed by a MINER, so membership only requires the hotkey to
    be present in the metagraph snapshot (it need NOT hold a validator permit).
    """

    metagraph_cache: MetagraphCache

    def is_registered(self, hotkey: str) -> bool:
        return hotkey in self.metagraph_cache.get()


class SqlAlchemyWorkerNonceStore:
    """Persisted ``(hotkey, nonce)`` replay guard for worker-plane nonces."""

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
        cutoff = created_at - timedelta(seconds=self.ttl_seconds)
        try:
            async with session_scope(self.session_factory) as session:
                await session.execute(
                    delete(WorkerRequestNonce).where(
                        WorkerRequestNonce.created_at < cutoff
                    )
                )
                session.add(
                    WorkerRequestNonce(
                        id=uuid.uuid4(),
                        hotkey=hotkey,
                        nonce=nonce,
                        body_hash=body_hash,
                        created_at=created_at,
                    )
                )
        except IntegrityError as exc:
            raise WorkerReplayError("nonce already used") from exc


@dataclass
class RegisteredWorkerEligibility:
    """Eligible when the signer pubkey has a ``worker_registrations`` row.

    Any lifecycle status qualifies for authentication; status-sensitive gating
    (e.g. a retired worker not being resurrected) is enforced by the service so
    the caller still gets a meaningful, authenticated response.
    """

    session_factory: async_sessionmaker[AsyncSession]

    async def is_eligible(self, hotkey: str) -> bool:
        async with self.session_factory() as session:
            found = (
                await session.execute(
                    select(WorkerRegistration.id).where(
                        WorkerRegistration.worker_pubkey == hotkey
                    )
                )
            ).first()
        return found is not None


@dataclass
class CoordinationReadEligibility:
    """Eligible when the signer is a registered worker OR an eligible validator.

    Fleet reads (``GET /v1/workers`` / ``GET /v1/workers/active``) are
    authenticated-but-not-admin: any registered coordination identity may read.
    """

    session_factory: async_sessionmaker[AsyncSession]
    metagraph_cache: MetagraphCache

    async def is_eligible(self, hotkey: str) -> bool:
        self.metagraph_cache.get()
        if self.metagraph_cache.is_validator(hotkey):
            return True
        return await RegisteredWorkerEligibility(self.session_factory).is_eligible(
            hotkey
        )


class WorkerSignedRequestVerifier:
    """Verify a hotkey-signed worker-plane request (heartbeat + fleet reads)."""

    def __init__(
        self,
        *,
        nonce_store: WorkerNonceStore,
        eligibility: WorkerAuthEligibility,
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
    ) -> WorkerIdentity:
        hotkey = _required_header(headers, HOTKEY_HEADER)
        signature = _required_header(headers, SIGNATURE_HEADER)
        nonce = _required_header(headers, NONCE_HEADER)
        timestamp_raw = _required_header(headers, TIMESTAMP_HEADER)

        timestamp = _parse_epoch(timestamp_raw)
        now = float(self.now_fn())
        if abs(now - timestamp) > self.ttl_seconds:
            raise WorkerAuthError("stale signature")

        canonical_request = canonical_validator_request(
            method=method,
            path=path,
            query_string=query_string,
            timestamp=timestamp_raw,
            nonce=nonce,
            body=body,
        )
        if not self.signature_verifier(hotkey, canonical_request.encode(), signature):
            raise WorkerAuthError("invalid signature")

        if not await self.eligibility.is_eligible(hotkey):
            raise WorkerEligibilityError("signer is not an eligible worker identity")

        body_hash = body_sha256(body)
        await self.nonce_store.reserve(
            hotkey=hotkey,
            nonce=nonce,
            body_hash=body_hash,
            created_at=datetime.fromtimestamp(now, UTC),
        )
        return WorkerIdentity(
            hotkey=hotkey,
            nonce=nonce,
            timestamp=int(timestamp),
            body_hash=body_hash,
            canonical_request=canonical_request,
        )


def build_worker_auth_dependency(
    verifier: WorkerSignedRequestVerifier,
) -> Callable[[Request], object]:
    """Build a FastAPI dependency enforcing worker signed-request auth."""

    async def authenticate(request: Request) -> WorkerIdentity:
        body = await request.body()
        try:
            return await verifier.verify(
                method=request.method,
                path=request.url.path,
                query_string=request.url.query,
                headers=request.headers,
                body=body,
            )
        except WorkerReplayError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=REPLAY_AUTH_DETAIL
            ) from exc
        except WorkerEligibilityError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=INELIGIBLE_DETAIL
            ) from exc
        except WorkerAuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=INVALID_AUTH_DETAIL
            ) from exc

    return authenticate


def _parse_epoch(raw: str) -> float:
    import math

    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise WorkerAuthError("invalid timestamp") from exc
    if not math.isfinite(value):
        raise WorkerAuthError("invalid timestamp")
    return value


def _required_header(headers: Mapping[str, str], name: str) -> str:
    value = headers.get(name) or headers.get(name.lower())
    if not value or not value.strip():
        raise WorkerAuthError(f"missing {name}")
    return value.strip()


__all__ = [
    "WORKER_BINDING_MESSAGE_PREFIX",
    "CoordinationReadEligibility",
    "MetagraphMinerMembership",
    "RegisteredWorkerEligibility",
    "SqlAlchemyWorkerNonceStore",
    "WorkerAuthEligibility",
    "WorkerAuthError",
    "WorkerEligibilityError",
    "WorkerIdentity",
    "WorkerNonceStore",
    "WorkerReplayError",
    "WorkerSignedRequestVerifier",
    "build_worker_auth_dependency",
    "worker_binding_message",
]
