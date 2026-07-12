"""Authenticated challenge raw-weight push ingress on the master.

Implements the closed, digest-bound, hotkey-keyed push protocol
(`POST /internal/v1/challenges/{slug}/raw-weights`):

* challenge credential verification (bearer token bound to slug)
* strict schema / digest / freshness validation
* durable immutable snapshot persistence as the acknowledgement boundary
* exact concurrent delivery idempotence
* conflict rejection without mutation
* monotonic revision selection before sealing
"""

from __future__ import annotations

import asyncio
import hmac
import inspect
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.challenge_sdk.roles import Capability, Role, role_contract
from base.challenge_sdk.schemas import (
    RawWeightPushAcknowledgement,
    RawWeightPushRequest,
)
from base.db.models import (
    AggregationEpoch,
    AggregationEpochStatus,
    RawWeightNonce,
    RawWeightSnapshot,
)
from base.db.session import session_scope
from base.security.tokens import hash_token, verify_token

# Published policy: freshness uses server receipt with bounded clock skew.
RAW_WEIGHT_FRESHNESS_POLICY_VERSION = "raw-weight-freshness.v1"
DEFAULT_MAX_CLOCK_SKEW_SECONDS = 30
DEFAULT_MAX_BODY_BYTES = 256_000
DEFAULT_MAX_WEIGHT_KEYS = 4_096
DEFAULT_MAX_FUTURE_EPOCH_AHEAD = 2
PROTOCOL_MAJOR = "1"

UNAUTHORIZED_DETAIL = "Unauthorized"
FORBIDDEN_DETAIL = "Forbidden"
CONFLICT_DETAIL = "conflicting raw weight payload"
STALE_EPOCH_DETAIL = "stale or sealed epoch"
FRESHNESS_DETAIL = "snapshot outside freshness window"
DIGEST_DETAIL = "payload_digest mismatch"
SEALED_REVISION_DETAIL = "epoch is sealed; revision rejected"
SCHEMA_DETAIL = "invalid raw weight payload"
PAYLOAD_TOO_LARGE_DETAIL = "payload too large"
UNSUPPORTED_MEDIA_DETAIL = "unsupported media type"
UNSUPPORTED_VERSION_DETAIL = "unsupported protocol version"
FUTURE_EPOCH_DETAIL = "epoch too far in the future"


class RawWeightAuthError(PermissionError):
    """Missing, malformed, or unknown challenge credential (HTTP 401)."""


class RawWeightForbiddenError(PermissionError):
    """Credential does not match route/body challenge binding (HTTP 403)."""


class RawWeightConflictError(ValueError):
    """Replay/conflict under an existing operation identity (HTTP 409)."""


class RawWeightFreshnessError(ValueError):
    """Outside the documented receipt-time skew window (HTTP 422)."""


class RawWeightSchemaError(ValueError):
    """Malformed, oversized, or non-canonical payload (HTTP 422)."""


class RawWeightSealedError(ValueError):
    """Epoch sealed; higher/lower late revisions fail closed (HTTP 409)."""


@dataclass(frozen=True)
class PushOutcome:
    """Durable acknowledgement fields after a successful commit path."""

    snapshot_id: str
    payload_digest: str
    challenge_slug: str
    epoch: int
    revision: int
    protocol_version: str
    idempotent: bool


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        return ""
    return authorization.split(" ", 1)[1].strip()


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def canonical_challenge_push_request(
    *,
    method: str,
    path: str,
    challenge_slug: str,
    timestamp: str,
    body: bytes,
) -> str:
    """Canonical string signed by the challenge credential for raw-weight push.

    Binds method, path, route challenge identity, freshness timestamp, and body.
    """

    body_digest = sha256(body).hexdigest()
    return f"{method.upper()}\n{path}\n{challenge_slug}\n{timestamp}\n{body_digest}"


def sign_challenge_push_request(*, token: str, canonical: str) -> str:
    """HMAC-SHA256 challenge credential over the canonical request string."""

    return hmac.new(
        token.encode("utf-8"),
        canonical.encode("utf-8"),
        sha256,
    ).hexdigest()


class ChallengeCredentialStore:
    """Resolve a challenge slug to plain token (for HMAC) and publication hash."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    async def token_for(self, slug: str) -> str | None:
        get_token = getattr(self._registry, "get_token", None)
        if not callable(get_token):
            return None
        try:
            token = await _resolve(get_token(slug))
        except Exception:  # noqa: BLE001 - fail closed on unknown challenge
            return None
        if not token:
            return None
        return str(token)

    async def token_hash_for(self, slug: str) -> str | None:
        get = getattr(self._registry, "get", None)
        if callable(get):
            try:
                record = await _resolve(get(slug))
            except Exception:  # noqa: BLE001
                record = None
            if record is not None:
                token_hash = getattr(record, "token_hash", None)
                if isinstance(token_hash, str) and token_hash:
                    return token_hash
        token = await self.token_for(slug)
        if token is None:
            return None
        return hash_token(token)


class RawWeightIngressService:
    """Persist authenticated challenge raw-weight snapshots."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        credential_store: ChallengeCredentialStore,
        max_clock_skew_seconds: int = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        max_weight_keys: int = DEFAULT_MAX_WEIGHT_KEYS,
        max_future_epoch_ahead: int = DEFAULT_MAX_FUTURE_EPOCH_AHEAD,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
        accepted_versions: frozenset[str] = frozenset({"1.0"}),
    ) -> None:
        self._session_factory = session_factory
        self._credentials = credential_store
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self.max_body_bytes = max_body_bytes
        self.max_weight_keys = max_weight_keys
        self.max_future_epoch_ahead = max_future_epoch_ahead
        self._now_fn = now_fn
        self.accepted_versions = accepted_versions
        self.freshness_policy_version = RAW_WEIGHT_FRESHNESS_POLICY_VERSION

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_RAW_WEIGHT_INGRESS)
    async def accept_push(
        self,
        *,
        route_slug: str,
        method: str,
        path: str,
        authorization: str | None,
        content_type: str | None,
        raw_body: bytes,
        signature: str | None,
        timestamp_header: str | None,
        challenge_slug_header: str | None,
    ) -> PushOutcome:
        """Validate, authenticate, and durable-commit a raw-weight push."""

        self._validate_content_type(content_type)
        if len(raw_body) > self.max_body_bytes:
            raise RawWeightSchemaError(PAYLOAD_TOO_LARGE_DETAIL)

        body_hash = sha256(raw_body).hexdigest()
        payload = self._parse_payload(raw_body)
        self._validate_route_binding(
            route_slug=route_slug,
            header_slug=challenge_slug_header,
            payload=payload,
        )
        await self._authenticate(
            route_slug=route_slug,
            body_slug=payload.challenge_slug,
            authorization=authorization,
            method=method,
            path=path,
            signature=signature,
            timestamp_header=timestamp_header,
            raw_body=raw_body,
        )
        receipt = self._now_fn()
        self._validate_freshness(payload, receipt=receipt)
        await self._validate_epoch_window(payload)

        try:
            return await self._commit(
                payload=payload,
                body_hash=body_hash,
                receipt=receipt,
            )
        except IntegrityError:
            # Lost a concurrent exact/conflicting race — re-read for idempotence
            # or raise a stable conflict without mutation.
            return await self._resolve_after_race(payload, body_hash=body_hash)
        except Exception as exc:
            message = str(exc).lower()
            if any(
                token in message
                for token in ("unique", "locked", "constraint", "integrity")
            ):
                return await self._resolve_after_race(payload, body_hash=body_hash)
            raise

    async def seal_epoch(self, epoch: int) -> AggregationEpoch:
        """Mark an epoch sealed so subsequent revisions fail closed."""

        now = self._now_fn()
        async with session_scope(self._session_factory) as session:
            row = (
                await session.execute(
                    select(AggregationEpoch).where(AggregationEpoch.epoch == epoch)
                )
            ).scalar_one_or_none()
            if row is None:
                row = AggregationEpoch(
                    id=uuid.uuid4(),
                    epoch=epoch,
                    status=AggregationEpochStatus.SEALED,
                    sealed_at=now,
                )
                session.add(row)
            else:
                row.status = AggregationEpochStatus.SEALED
                row.sealed_at = now
            await session.flush()
            await session.refresh(row)
            return row

    def _validate_content_type(self, content_type: str | None) -> None:
        if content_type is None:
            raise RawWeightSchemaError(UNSUPPORTED_MEDIA_DETAIL)
        media = content_type.split(";", 1)[0].strip().lower()
        if media != "application/json":
            raise RawWeightSchemaError(UNSUPPORTED_MEDIA_DETAIL)

    def _parse_payload(self, raw_body: bytes) -> RawWeightPushRequest:
        try:
            # Reject BOM-prefixed / non-UTF-8 bodies early.
            text = raw_body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RawWeightSchemaError(SCHEMA_DETAIL) from exc
        try:
            # model_validate_json rejects non-object roots and coercion.
            payload = RawWeightPushRequest.model_validate_json(text)
        except ValidationError as exc:
            raise RawWeightSchemaError(SCHEMA_DETAIL) from exc
        if payload.protocol_version not in self.accepted_versions:
            raise RawWeightSchemaError(UNSUPPORTED_VERSION_DETAIL)
        if not payload.protocol_version.startswith(f"{PROTOCOL_MAJOR}."):
            raise RawWeightSchemaError(UNSUPPORTED_VERSION_DETAIL)
        if len(payload.weights) > self.max_weight_keys:
            raise RawWeightSchemaError(PAYLOAD_TOO_LARGE_DETAIL)
        # Recompute server digest over the signed field set (exclude digest).
        canonical = payload.model_dump(mode="json", exclude={"payload_digest"})
        expected = RawWeightPushRequest.compute_digest(canonical)
        if expected != payload.payload_digest:
            raise RawWeightSchemaError(DIGEST_DETAIL)
        return payload

    def _validate_route_binding(
        self,
        *,
        route_slug: str,
        header_slug: str | None,
        payload: RawWeightPushRequest,
    ) -> None:
        if route_slug != payload.challenge_slug:
            raise RawWeightForbiddenError(FORBIDDEN_DETAIL)
        if header_slug is not None and header_slug != route_slug:
            raise RawWeightForbiddenError(FORBIDDEN_DETAIL)

    async def _authenticate(
        self,
        *,
        route_slug: str,
        body_slug: str,
        authorization: str | None,
        method: str,
        path: str,
        signature: str | None,
        timestamp_header: str | None,
        raw_body: bytes,
    ) -> None:
        token = _bearer_token(authorization)
        if not token:
            raise RawWeightAuthError(UNAUTHORIZED_DETAIL)
        if route_slug != body_slug:
            raise RawWeightForbiddenError(FORBIDDEN_DETAIL)
        expected_token = await self._credentials.token_for(route_slug)
        if expected_token is None:
            # Try hash material only when plaintext is not mounted (revoked/absent).
            token_hash = await self._credentials.token_hash_for(route_slug)
            if token_hash is None or not verify_token(token, token_hash):
                raise RawWeightAuthError(UNAUTHORIZED_DETAIL)
            # Signature verification requires the exact credential secret.
            raise RawWeightAuthError(UNAUTHORIZED_DETAIL)
        if not hmac.compare_digest(token, expected_token):
            raise RawWeightAuthError(UNAUTHORIZED_DETAIL)
        if not signature or not timestamp_header:
            raise RawWeightAuthError(UNAUTHORIZED_DETAIL)
        try:
            ts = int(timestamp_header)
        except (TypeError, ValueError) as exc:
            raise RawWeightAuthError(UNAUTHORIZED_DETAIL) from exc
        skew = self.max_clock_skew_seconds
        now = int(self._now_fn().timestamp())
        if abs(now - ts) > skew:
            raise RawWeightAuthError(UNAUTHORIZED_DETAIL)
        canonical = canonical_challenge_push_request(
            method=method,
            path=path,
            challenge_slug=route_slug,
            timestamp=str(ts),
            body=raw_body,
        )
        expected_sig = sign_challenge_push_request(
            token=expected_token, canonical=canonical
        )
        if not hmac.compare_digest(signature.lower(), expected_sig.lower()):
            raise RawWeightAuthError(UNAUTHORIZED_DETAIL)

    def _validate_freshness(
        self, payload: RawWeightPushRequest, *, receipt: datetime
    ) -> None:
        """Accept when contribution is still fresh under bounded skew.

        Policy ``raw-weight-freshness.v1``:
        ``computed_at - max_clock_skew <= receipt_time < expires_at + max_clock_skew``
        and ``expires_at > computed_at`` (schema-enforced).
        """

        skew = timedelta(seconds=self.max_clock_skew_seconds)
        computed = _as_utc(payload.computed_at)
        expires = _as_utc(payload.expires_at)
        receipt_utc = _as_utc(receipt)
        lower = computed - skew
        upper = expires + skew
        if not (lower <= receipt_utc < upper):
            raise RawWeightFreshnessError(FRESHNESS_DETAIL)

    async def _validate_epoch_window(self, payload: RawWeightPushRequest) -> None:
        async with session_scope(self._session_factory) as session:
            highest_sealed = (
                await session.execute(
                    select(AggregationEpoch.epoch)
                    .where(AggregationEpoch.status == AggregationEpochStatus.SEALED)
                    .order_by(AggregationEpoch.epoch.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if highest_sealed is not None and payload.epoch < int(highest_sealed):
                raise RawWeightSealedError(STALE_EPOCH_DETAIL)
            if highest_sealed is not None and payload.epoch == int(highest_sealed):
                raise RawWeightSealedError(SEALED_REVISION_DETAIL)
            if highest_sealed is not None:
                ahead = payload.epoch - int(highest_sealed)
                if ahead > self.max_future_epoch_ahead:
                    raise RawWeightSchemaError(FUTURE_EPOCH_DETAIL)
            epoch_row = (
                await session.execute(
                    select(AggregationEpoch).where(
                        AggregationEpoch.epoch == payload.epoch
                    )
                )
            ).scalar_one_or_none()
            if (
                epoch_row is not None
                and epoch_row.status == AggregationEpochStatus.SEALED
            ):
                raise RawWeightSealedError(SEALED_REVISION_DETAIL)

    async def _commit(
        self,
        *,
        payload: RawWeightPushRequest,
        body_hash: str,
        receipt: datetime,
    ) -> PushOutcome:
        async with session_scope(self._session_factory) as session:
            existing_nonce = (
                await session.execute(
                    select(RawWeightNonce).where(
                        RawWeightNonce.challenge_slug == payload.challenge_slug,
                        RawWeightNonce.nonce == payload.nonce,
                    )
                )
            ).scalar_one_or_none()
            if existing_nonce is not None:
                return await self._existing_nonce_outcome(
                    session, existing_nonce, payload=payload, body_hash=body_hash
                )

            existing_rev = (
                await session.execute(
                    select(RawWeightSnapshot).where(
                        RawWeightSnapshot.challenge_slug == payload.challenge_slug,
                        RawWeightSnapshot.epoch == payload.epoch,
                        RawWeightSnapshot.revision == payload.revision,
                    )
                )
            ).scalar_one_or_none()
            if existing_rev is not None:
                if existing_rev.payload_digest == payload.payload_digest:
                    return PushOutcome(
                        snapshot_id=str(existing_rev.id),
                        payload_digest=existing_rev.payload_digest,
                        challenge_slug=existing_rev.challenge_slug,
                        epoch=int(existing_rev.epoch),
                        revision=int(existing_rev.revision),
                        protocol_version=existing_rev.protocol_version,
                        idempotent=True,
                    )
                raise RawWeightConflictError(CONFLICT_DETAIL)

            await self._ensure_open_epoch(session, payload.epoch)

            canonical_text = RawWeightPushRequest.canonicalize(
                payload.model_dump(mode="json", exclude={"payload_digest"})
            ).decode("utf-8")
            snapshot_id = uuid.uuid4()
            snapshot = RawWeightSnapshot(
                id=snapshot_id,
                challenge_slug=payload.challenge_slug,
                epoch=payload.epoch,
                revision=payload.revision,
                protocol_version=payload.protocol_version,
                computed_at=_as_utc(payload.computed_at),
                expires_at=_as_utc(payload.expires_at),
                nonce=payload.nonce,
                payload_digest=payload.payload_digest,
                canonical_payload=canonical_text,
                weights=dict(payload.weights),
                is_selected_source=False,
                received_at=_as_utc(receipt),
            )
            session.add(snapshot)
            await session.flush()

            # Selection: highest accepted revision for the open epoch wins.
            await self._update_selected_source(
                session,
                challenge_slug=payload.challenge_slug,
                epoch=payload.epoch,
            )

            session.add(
                RawWeightNonce(
                    id=uuid.uuid4(),
                    challenge_slug=payload.challenge_slug,
                    nonce=payload.nonce,
                    body_hash=body_hash,
                    payload_digest=payload.payload_digest,
                    epoch=payload.epoch,
                    revision=payload.revision,
                    snapshot_id=snapshot_id,
                    created_at=_as_utc(receipt),
                )
            )
            await session.flush()
            return PushOutcome(
                snapshot_id=str(snapshot_id),
                payload_digest=payload.payload_digest,
                challenge_slug=payload.challenge_slug,
                epoch=payload.epoch,
                revision=payload.revision,
                protocol_version=payload.protocol_version,
                idempotent=False,
            )

    async def _ensure_open_epoch(self, session: AsyncSession, epoch: int) -> None:
        """Create or load the open aggregation epoch.

        Concurrent first-open of the same epoch is resolved by nested savepoint:
        the loser reloads the winner without aborting the outer unit of work.
        """

        epoch_row = (
            await session.execute(
                select(AggregationEpoch).where(AggregationEpoch.epoch == epoch)
            )
        ).scalar_one_or_none()
        if epoch_row is not None:
            if epoch_row.status == AggregationEpochStatus.SEALED:
                raise RawWeightSealedError(SEALED_REVISION_DETAIL)
            return
        try:
            async with session.begin_nested():
                session.add(
                    AggregationEpoch(
                        id=uuid.uuid4(),
                        epoch=epoch,
                        status=AggregationEpochStatus.OPEN,
                    )
                )
                await session.flush()
        except IntegrityError as exc:
            epoch_row = (
                await session.execute(
                    select(AggregationEpoch).where(AggregationEpoch.epoch == epoch)
                )
            ).scalar_one_or_none()
            if epoch_row is None:
                raise
            if epoch_row.status == AggregationEpochStatus.SEALED:
                raise RawWeightSealedError(SEALED_REVISION_DETAIL) from exc

    async def _existing_nonce_outcome(
        self,
        session: AsyncSession,
        existing: RawWeightNonce,
        *,
        payload: RawWeightPushRequest,
        body_hash: str,
    ) -> PushOutcome:
        if (
            existing.body_hash != body_hash
            or existing.payload_digest != payload.payload_digest
        ):
            raise RawWeightConflictError(CONFLICT_DETAIL)
        if (
            existing.epoch != payload.epoch
            or existing.revision != payload.revision
            or existing.challenge_slug != payload.challenge_slug
        ):
            raise RawWeightConflictError(CONFLICT_DETAIL)
        snapshot: RawWeightSnapshot | None = None
        if existing.snapshot_id is not None:
            snapshot = (
                await session.execute(
                    select(RawWeightSnapshot).where(
                        RawWeightSnapshot.id == existing.snapshot_id
                    )
                )
            ).scalar_one_or_none()
        if snapshot is None:
            snapshot = (
                await session.execute(
                    select(RawWeightSnapshot).where(
                        RawWeightSnapshot.challenge_slug == payload.challenge_slug,
                        RawWeightSnapshot.epoch == payload.epoch,
                        RawWeightSnapshot.revision == payload.revision,
                    )
                )
            ).scalar_one_or_none()
        if snapshot is None:
            raise RawWeightConflictError(CONFLICT_DETAIL)
        return PushOutcome(
            snapshot_id=str(snapshot.id),
            payload_digest=snapshot.payload_digest,
            challenge_slug=snapshot.challenge_slug,
            epoch=int(snapshot.epoch),
            revision=int(snapshot.revision),
            protocol_version=snapshot.protocol_version,
            idempotent=True,
        )

    async def _update_selected_source(
        self,
        session: AsyncSession,
        *,
        challenge_slug: str,
        epoch: int,
    ) -> None:
        """Atomically select the highest accepted revision for (slug, epoch).

        Serializes concurrent multi-revision races under a per-(slug, epoch)
        barrier so a lower revision cannot remain ``is_selected_source`` after a
        concurrent higher-revision commit (VAL-WEIGHT-017/095).

        PostgreSQL/SQLite: lock the matching aggregation_epochs row with
        ``FOR UPDATE``; on SQLite without row locking, nested savepoints still
        force one writer through identity of ``is_selected_source`` via
        exclusive update of the selected flag set after re-reading.
        """

        # Serialize selection with the open epoch barrier (if present). Nested
        # savepoint keeps outer commit boundaries intact when FOR UPDATE is
        # unsupported by the dialect (test SQLite variants).
        try:
            async with session.begin_nested():
                await self._select_highest_revision_locked(
                    session,
                    challenge_slug=challenge_slug,
                    epoch=epoch,
                    for_update=True,
                )
        except RawWeightSealedError:
            raise
        except (OperationalError, ProgrammingError, NotImplementedError):
            await self._select_highest_revision_locked(
                session,
                challenge_slug=challenge_slug,
                epoch=epoch,
                for_update=False,
            )

    async def _select_highest_revision_locked(
        self,
        session: AsyncSession,
        *,
        challenge_slug: str,
        epoch: int,
        for_update: bool,
    ) -> None:
        epoch_stmt = select(AggregationEpoch).where(
            AggregationEpoch.epoch == int(epoch)
        )
        snap_stmt = (
            select(RawWeightSnapshot)
            .where(
                RawWeightSnapshot.challenge_slug == challenge_slug,
                RawWeightSnapshot.epoch == epoch,
            )
            .order_by(
                RawWeightSnapshot.revision.desc(),
                RawWeightSnapshot.received_at.desc(),
            )
        )
        if for_update:
            epoch_stmt = epoch_stmt.with_for_update()
            snap_stmt = snap_stmt.with_for_update()

        epoch_row = (await session.execute(epoch_stmt)).scalar_one_or_none()
        if epoch_row is not None and epoch_row.status == AggregationEpochStatus.SEALED:
            raise RawWeightSealedError(SEALED_REVISION_DETAIL)

        rows = (await session.execute(snap_stmt)).scalars().all()
        if not rows:
            return
        winner = rows[0]
        await session.execute(
            update(RawWeightSnapshot)
            .where(
                RawWeightSnapshot.challenge_slug == challenge_slug,
                RawWeightSnapshot.epoch == epoch,
            )
            .values(is_selected_source=False)
        )
        await session.execute(
            update(RawWeightSnapshot)
            .where(RawWeightSnapshot.id == winner.id)
            .values(is_selected_source=True)
        )
        await session.flush()

    async def _resolve_after_race(
        self, payload: RawWeightPushRequest, *, body_hash: str
    ) -> PushOutcome:
        # Concurrent exact delivery may lose the insert race before the winner
        # commits; retry briefly so every loser converges on one snapshot.
        last_error: Exception | None = None
        for attempt in range(8):
            try:
                async with session_scope(self._session_factory) as session:
                    existing_nonce = (
                        await session.execute(
                            select(RawWeightNonce).where(
                                RawWeightNonce.challenge_slug == payload.challenge_slug,
                                RawWeightNonce.nonce == payload.nonce,
                            )
                        )
                    ).scalar_one_or_none()
                    if existing_nonce is not None:
                        return await self._existing_nonce_outcome(
                            session,
                            existing_nonce,
                            payload=payload,
                            body_hash=body_hash,
                        )
                    existing_rev = (
                        await session.execute(
                            select(RawWeightSnapshot).where(
                                RawWeightSnapshot.challenge_slug
                                == payload.challenge_slug,
                                RawWeightSnapshot.epoch == payload.epoch,
                                RawWeightSnapshot.revision == payload.revision,
                            )
                        )
                    ).scalar_one_or_none()
                    if existing_rev is None:
                        raise RawWeightConflictError(CONFLICT_DETAIL)
                    if existing_rev.payload_digest != payload.payload_digest:
                        raise RawWeightConflictError(CONFLICT_DETAIL)
                    return PushOutcome(
                        snapshot_id=str(existing_rev.id),
                        payload_digest=existing_rev.payload_digest,
                        challenge_slug=existing_rev.challenge_slug,
                        epoch=int(existing_rev.epoch),
                        revision=int(existing_rev.revision),
                        protocol_version=existing_rev.protocol_version,
                        idempotent=True,
                    )
            except RawWeightConflictError as exc:
                last_error = exc
                if attempt + 1 >= 8:
                    raise
                await asyncio.sleep(0.01 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RawWeightConflictError(CONFLICT_DETAIL)


def build_raw_weight_ingress_router(
    *,
    service: RawWeightIngressService,
) -> APIRouter:
    """Build the challenge raw-weight push router served by the master."""

    router = APIRouter()

    @router.post(
        "/internal/v1/challenges/{slug}/raw-weights",
        response_model=RawWeightPushAcknowledgement,
        responses={
            401: {"description": "unauthorized"},
            403: {"description": "forbidden"},
            409: {"description": "conflict/replay"},
            413: {"description": "payload too large"},
            415: {"description": "unsupported media type"},
            422: {"description": "schema/freshness/digest rejection"},
            503: {"description": "unavailable"},
        },
    )
    async def push_raw_weights(
        slug: str,
        request: Request,
        authorization: str | None = Header(default=None),
        content_type: str | None = Header(default=None, alias="Content-Type"),
        x_role: str | None = Header(default=None, alias="X-Role"),
        x_signature: str | None = Header(default=None, alias="X-Signature"),
        x_timestamp: str | None = Header(default=None, alias="X-Timestamp"),
        x_base_challenge_slug: str | None = Header(
            default=None, alias="X-Base-Challenge-Slug"
        ),
    ) -> RawWeightPushAcknowledgement | JSONResponse:
        # Role headers never establish identity or capability.
        _ = x_role
        raw_body = await request.body()
        try:
            outcome = await service.accept_push(
                route_slug=slug,
                method=request.method,
                path=request.url.path,
                authorization=authorization,
                content_type=content_type or request.headers.get("content-type"),
                raw_body=raw_body,
                signature=x_signature,
                timestamp_header=x_timestamp,
                challenge_slug_header=x_base_challenge_slug,
            )
        except RawWeightAuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=UNAUTHORIZED_DETAIL,
            ) from exc
        except RawWeightForbiddenError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=FORBIDDEN_DETAIL,
            ) from exc
        except RawWeightConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=CONFLICT_DETAIL,
            ) from exc
        except RawWeightSealedError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc) or SEALED_REVISION_DETAIL,
            ) from exc
        except RawWeightFreshnessError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=FRESHNESS_DETAIL,
            ) from exc
        except RawWeightSchemaError as exc:
            message = str(exc) or SCHEMA_DETAIL
            if message == PAYLOAD_TOO_LARGE_DETAIL:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=PAYLOAD_TOO_LARGE_DETAIL,
                ) from exc
            if message == UNSUPPORTED_MEDIA_DETAIL:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=UNSUPPORTED_MEDIA_DETAIL,
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=message,
            ) from exc
        return RawWeightPushAcknowledgement(
            protocol_version=outcome.protocol_version,
            challenge_slug=outcome.challenge_slug,
            epoch=outcome.epoch,
            revision=outcome.revision,
            snapshot_id=outcome.snapshot_id,
            payload_digest=outcome.payload_digest,
            accepted=True,
            idempotent=outcome.idempotent,
        )

    return router


__all__ = [
    "ChallengeCredentialStore",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_MAX_CLOCK_SKEW_SECONDS",
    "DEFAULT_MAX_FUTURE_EPOCH_AHEAD",
    "DEFAULT_MAX_WEIGHT_KEYS",
    "PushOutcome",
    "RAW_WEIGHT_FRESHNESS_POLICY_VERSION",
    "RawWeightAuthError",
    "RawWeightConflictError",
    "RawWeightForbiddenError",
    "RawWeightFreshnessError",
    "RawWeightIngressService",
    "RawWeightSchemaError",
    "RawWeightSealedError",
    "build_raw_weight_ingress_router",
    "canonical_challenge_push_request",
    "sign_challenge_push_request",
]
