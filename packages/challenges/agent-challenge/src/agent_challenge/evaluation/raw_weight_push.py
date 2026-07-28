"""Challenge-owned raw-weight push to the master with durable ack checkpointing.

Agent Challenge builds a closed, digest-bound hotkey-weight snapshot (typically
the winner-take-all map from ``get_weights``), signs it with the challenge
credential, and posts to master
``POST /internal/v1/challenges/agent-challenge/raw-weights``. Delivery cursor
advances only after an acknowledgement whose snapshot digest and epoch/revision
match the attempted payload exactly. Timeouts and bad acks leave the cursor
unchanged so restart retries the same logical snapshot.

Wire format, HMAC binding, and ack gate mirror Prism's
``prism_challenge.raw_weight_push`` so master accepts both challenges identically.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

import httpx
from base.challenge_sdk.roles import Capability, Role, activate_role, role_contract
from base.challenge_sdk.schemas import (
    RawWeightPushAcknowledgement,
    RawWeightPushRequest,
)
from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from agent_challenge.sdk.db import Base, Database

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "1.0"
DEFAULT_FRESHNESS_SECONDS = 300
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_CHALLENGE_SLUG = "agent-challenge"

WeightsFn = Callable[[], Mapping[str, float] | Awaitable[Mapping[str, float]]]
EpochFn = Callable[[], int]


def canonical_challenge_push_request(
    *,
    method: str,
    path: str,
    challenge_slug: str,
    timestamp: str,
    body: bytes,
) -> str:
    """Mirror the master's challenge-push signature binding (method/path/slug/ts/body)."""

    body_digest = sha256(body).hexdigest()
    return f"{method.upper()}\n{path}\n{challenge_slug}\n{timestamp}\n{body_digest}"


def sign_challenge_push_request(*, token: str, canonical: str) -> str:
    return hmac.new(
        token.encode("utf-8"),
        canonical.encode("utf-8"),
        sha256,
    ).hexdigest()


@dataclass(frozen=True)
class PushCursor:
    """Durable last-acknowledged delivery identity for one challenge."""

    epoch: int
    revision: int
    payload_digest: str
    snapshot_id: str | None
    acknowledged_at: str | None


@dataclass(frozen=True)
class PushAttemptResult:
    """Outcome of a single push attempt (cursor may or may not advance)."""

    status: str
    epoch: int
    revision: int
    payload_digest: str
    snapshot_id: str | None
    cursor_advanced: bool
    error: str | None = None


class RawWeightPushLedger(Base):
    """Singleton SQLAlchemy row for durable attempt/ack checkpointing.

    Matches Prism's ``raw_weight_push_ledger`` columns so restart semantics are
    identical; stored via AC's shared Database (SQLite or PostgreSQL), not a
    side-car SQLite file.
    """

    __tablename__ = "raw_weight_push_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    challenge_slug: Mapped[str] = mapped_column(String(128), nullable=False)
    last_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_payload_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_snapshot_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_canonical_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_nonce: Mapped[str | None] = mapped_column(String(256), nullable=True)
    acknowledged_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pending_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pending_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pending_payload_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pending_canonical_payload: Mapped[str | None] = mapped_column(Text, nullable=True)
    pending_nonce: Mapped[str | None] = mapped_column(String(256), nullable=True)
    pending_attempted_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class RawWeightPushStore:
    """SQLAlchemy durable attempt/ack ledger for agent-challenge raw-weight delivery."""

    def __init__(self, database: Database, *, challenge_slug: str) -> None:
        self.database = database
        self.challenge_slug = challenge_slug

    async def init(self) -> None:
        async with self.database.engine.begin() as connection:
            await connection.run_sync(
                lambda sync_conn: RawWeightPushLedger.__table__.create(
                    sync_conn, checkfirst=True
                )
            )

    async def _get_row(self, session: Any) -> RawWeightPushLedger | None:
        return await session.get(RawWeightPushLedger, 1)

    async def get_cursor(self) -> PushCursor | None:
        async with self.database.session() as session:
            row = await self._get_row(session)
            if row is None or row.last_epoch is None:
                return None
            return PushCursor(
                epoch=int(row.last_epoch),
                revision=int(row.last_revision or 0),
                payload_digest=str(row.last_payload_digest or ""),
                snapshot_id=(
                    str(row.last_snapshot_id) if row.last_snapshot_id is not None else None
                ),
                acknowledged_at=(
                    str(row.acknowledged_at) if row.acknowledged_at is not None else None
                ),
            )

    async def get_pending(self) -> dict[str, Any] | None:
        async with self.database.session() as session:
            row = await self._get_row(session)
            if row is None or row.pending_epoch is None:
                return None
            return {
                "epoch": int(row.pending_epoch),
                "revision": int(row.pending_revision or 0),
                "payload_digest": str(row.pending_payload_digest or ""),
                "canonical_payload": str(row.pending_canonical_payload or ""),
                "nonce": str(row.pending_nonce or ""),
                "attempted_at": row.pending_attempted_at,
            }

    async def record_pending(
        self,
        *,
        epoch: int,
        revision: int,
        payload_digest: str,
        canonical_payload: str,
        nonce: str,
        attempted_at: str,
    ) -> None:
        async with self.database.session() as session:
            row = await self._get_row(session)
            if row is None:
                row = RawWeightPushLedger(
                    id=1,
                    challenge_slug=self.challenge_slug,
                    updated_at=attempted_at,
                )
                session.add(row)
            row.challenge_slug = self.challenge_slug
            row.pending_epoch = epoch
            row.pending_revision = revision
            row.pending_payload_digest = payload_digest
            row.pending_canonical_payload = canonical_payload
            row.pending_nonce = nonce
            row.pending_attempted_at = attempted_at
            row.updated_at = attempted_at
            await session.commit()

    async def acknowledge(
        self,
        *,
        epoch: int,
        revision: int,
        payload_digest: str,
        snapshot_id: str,
        canonical_payload: str,
        nonce: str,
        acknowledged_at: str,
    ) -> None:
        """Advance the durable delivery cursor after an exact matching ack."""

        async with self.database.session() as session:
            row = await self._get_row(session)
            if row is None:
                row = RawWeightPushLedger(
                    id=1,
                    challenge_slug=self.challenge_slug,
                    updated_at=acknowledged_at,
                )
                session.add(row)
            row.challenge_slug = self.challenge_slug
            row.last_epoch = epoch
            row.last_revision = revision
            row.last_payload_digest = payload_digest
            row.last_snapshot_id = snapshot_id
            row.last_canonical_payload = canonical_payload
            row.last_nonce = nonce
            row.acknowledged_at = acknowledged_at
            row.pending_epoch = None
            row.pending_revision = None
            row.pending_payload_digest = None
            row.pending_canonical_payload = None
            row.pending_nonce = None
            row.pending_attempted_at = None
            row.updated_at = acknowledged_at
            await session.commit()


async def _resolve_weights(
    weights: Mapping[str, float] | None,
    weights_fn: WeightsFn | None,
) -> dict[str, float]:
    if weights is not None:
        return dict(weights)
    if weights_fn is None:
        return {}
    loaded = weights_fn()
    if isinstance(loaded, Awaitable):
        loaded = await loaded
    return dict(loaded)


class RawWeightPushClient:
    """Build, sign, and push raw weights; checkpoint only on exact acks."""

    def __init__(
        self,
        *,
        database: Database,
        challenge_slug: str,
        master_base_url: str,
        shared_token: str,
        weights_fn: WeightsFn | None = None,
        epoch_fn: EpochFn | None = None,
        freshness_seconds: int = DEFAULT_FRESHNESS_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.database = database
        self.challenge_slug = challenge_slug
        self.master_base_url = master_base_url.rstrip("/")
        self.shared_token = shared_token
        self.weights_fn = weights_fn
        self.epoch_fn = epoch_fn
        self.freshness_seconds = freshness_seconds
        self.timeout_seconds = timeout_seconds
        self._now_fn = now_fn
        self._http = http_client
        self.store = RawWeightPushStore(database, challenge_slug=challenge_slug)

    async def init(self) -> None:
        await self.store.init()

    def _path_for(self) -> str:
        return f"/internal/v1/challenges/{self.challenge_slug}/raw-weights"

    def _next_revision(self, cursor: PushCursor | None, epoch: int) -> int:
        if cursor is None:
            return 1
        if cursor.epoch == epoch:
            return cursor.revision + 1
        return 1

    def _build_payload(
        self,
        *,
        weights: Mapping[str, float],
        epoch: int,
        revision: int,
        nonce: str,
        now: datetime,
    ) -> tuple[RawWeightPushRequest, bytes]:
        computed_at = now.replace(microsecond=0)
        expires_at = computed_at + timedelta(seconds=self.freshness_seconds)
        body: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "challenge_slug": self.challenge_slug,
            "epoch": epoch,
            "revision": revision,
            "computed_at": computed_at.isoformat().replace("+00:00", "Z"),
            "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
            "nonce": nonce,
            "weights": {str(hotkey): float(weight) for hotkey, weight in weights.items()},
        }
        if not body["weights"]:
            raise ValueError("empty weight map has no push surface")
        digest = RawWeightPushRequest.compute_digest(body)
        body["payload_digest"] = digest
        payload = RawWeightPushRequest.model_validate(body)
        raw_bytes = payload.canonical_bytes()
        return payload, raw_bytes

    def _headers(self, *, path: str, body: bytes, timestamp: int) -> dict[str, str]:
        canonical = canonical_challenge_push_request(
            method="POST",
            path=path,
            challenge_slug=self.challenge_slug,
            timestamp=str(timestamp),
            body=body,
        )
        signature = sign_challenge_push_request(token=self.shared_token, canonical=canonical)
        return {
            "Authorization": f"Bearer {self.shared_token}",
            "Content-Type": "application/json",
            "X-Base-Challenge-Slug": self.challenge_slug,
            "X-Signature": signature,
            "X-Timestamp": str(timestamp),
            "Accept": "application/json",
        }

    def _ack_matches(
        self, ack: RawWeightPushAcknowledgement, *, payload: RawWeightPushRequest
    ) -> bool:
        return (
            ack.accepted is True
            and ack.challenge_slug == payload.challenge_slug
            and ack.epoch == payload.epoch
            and ack.revision == payload.revision
            and ack.payload_digest == payload.payload_digest
            and bool(ack.snapshot_id)
        )

    @role_contract(role=Role.CHALLENGE, capability=Capability.CHALLENGE_RAW_WEIGHT_PUSH)
    async def push_once(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        epoch: int | None = None,
        force_revision: int | None = None,
        reuse_pending: bool = True,
    ) -> PushAttemptResult:
        """Push one snapshot. Cursor advances only on exact durable acknowledgement."""

        await self.store.init()
        now = self._now_fn()
        cursor = await self.store.get_cursor()
        pending = await self.store.get_pending() if reuse_pending else None
        payload: RawWeightPushRequest | None = None
        raw_bytes: bytes | None = None

        if pending is not None:
            # Retry exact previous bytes after timeout/restart (no new revision).
            try:
                pending_bytes = str(pending["canonical_payload"]).encode("utf-8")
                payload = RawWeightPushRequest.model_validate_json(pending_bytes)
                raw_bytes = payload.canonical_bytes()
            except Exception:  # noqa: BLE001 - corrupt pending is rebuilt
                pending = None
                payload = None
                raw_bytes = None

        if payload is None or raw_bytes is None:
            resolved_weights = await _resolve_weights(weights, self.weights_fn)
            # Positive hotkey weights only when synthesizing from get_weights;
            # explicit zero maps (caller-supplied zeros) are preserved as zero-contribution.
            if weights is not None:
                cleaned = {
                    str(hotkey): float(value) for hotkey, value in resolved_weights.items()
                }
            else:
                cleaned = {
                    str(hotkey): float(value)
                    for hotkey, value in resolved_weights.items()
                    if float(value) > 0.0
                }
            if not cleaned:
                return PushAttemptResult(
                    status="skipped_empty",
                    epoch=0,
                    revision=0,
                    payload_digest="",
                    snapshot_id=None,
                    cursor_advanced=False,
                    error="empty weights",
                )
            resolved_epoch = (
                int(epoch)
                if epoch is not None
                else int(self.epoch_fn())
                if self.epoch_fn is not None
                else int(now.timestamp()) // 3600
            )
            revision = (
                int(force_revision)
                if force_revision is not None
                else self._next_revision(cursor, resolved_epoch)
            )
            nonce = f"agent-challenge-{uuid.uuid4().hex}"
            payload, raw_bytes = self._build_payload(
                weights=cleaned,
                epoch=resolved_epoch,
                revision=revision,
                nonce=nonce,
                now=now,
            )
            await self.store.record_pending(
                epoch=payload.epoch,
                revision=payload.revision,
                payload_digest=payload.payload_digest,
                canonical_payload=raw_bytes.decode("utf-8"),
                nonce=payload.nonce,
                attempted_at=now.isoformat(),
            )

        path = self._path_for()
        url = f"{self.master_base_url}{path}"
        headers = self._headers(path=path, body=raw_bytes, timestamp=int(now.timestamp()))
        client = self._http
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout_seconds)
        assert client is not None
        try:
            response = await client.post(url, content=raw_bytes, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            # Never log Authorization / token material — digest prefix only.
            logger.info(
                "raw weight push transport failure",
                extra={
                    "epoch": payload.epoch,
                    "revision": payload.revision,
                    "digest": payload.payload_digest[:12],
                },
            )
            return PushAttemptResult(
                status="transport_error",
                epoch=payload.epoch,
                revision=payload.revision,
                payload_digest=payload.payload_digest,
                snapshot_id=None,
                cursor_advanced=False,
                error=str(exc),
            )
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code >= 500:
            return PushAttemptResult(
                status="server_error",
                epoch=payload.epoch,
                revision=payload.revision,
                payload_digest=payload.payload_digest,
                snapshot_id=None,
                cursor_advanced=False,
                error=f"status={response.status_code}",
            )
        if response.status_code not in {200, 201}:
            return PushAttemptResult(
                status="rejected",
                epoch=payload.epoch,
                revision=payload.revision,
                payload_digest=payload.payload_digest,
                snapshot_id=None,
                cursor_advanced=False,
                error=f"status={response.status_code}",
            )
        try:
            ack = RawWeightPushAcknowledgement.model_validate(response.json())
        except Exception as exc:  # noqa: BLE001
            return PushAttemptResult(
                status="malformed_ack",
                epoch=payload.epoch,
                revision=payload.revision,
                payload_digest=payload.payload_digest,
                snapshot_id=None,
                cursor_advanced=False,
                error=str(exc),
            )
        if not self._ack_matches(ack, payload=payload):
            return PushAttemptResult(
                status="ack_mismatch",
                epoch=payload.epoch,
                revision=payload.revision,
                payload_digest=payload.payload_digest,
                snapshot_id=ack.snapshot_id if hasattr(ack, "snapshot_id") else None,
                cursor_advanced=False,
                error="acknowledgement identity mismatch",
            )
        ack_time = self._now_fn().isoformat()
        await self.store.acknowledge(
            epoch=payload.epoch,
            revision=payload.revision,
            payload_digest=payload.payload_digest,
            snapshot_id=ack.snapshot_id,
            canonical_payload=raw_bytes.decode("utf-8"),
            nonce=payload.nonce,
            acknowledged_at=ack_time,
        )
        return PushAttemptResult(
            status="acknowledged",
            epoch=payload.epoch,
            revision=payload.revision,
            payload_digest=payload.payload_digest,
            snapshot_id=ack.snapshot_id,
            cursor_advanced=True,
        )


def build_weights_loader(
    get_weights_fn: Callable[[], Awaitable[Mapping[str, float]]] | None = None,
) -> WeightsFn:
    """Wrap agent-challenge ``get_weights`` (async WTA map) for the push client."""

    async def _load() -> dict[str, float]:
        if get_weights_fn is None:
            from agent_challenge.evaluation.weights import get_weights

            return dict(await get_weights())
        return dict(await get_weights_fn())

    return _load


async def run_raw_weight_push_loop(
    client: RawWeightPushClient,
    *,
    interval_seconds: float = 30.0,
    resilient: bool = True,
) -> None:
    """Background loop: push scored hotkey weights when master + token enable it.

    Retries the same durable pending identity on transport failures. Cancellation
    always propagates so app lifespan can stop the task cleanly.
    """

    await client.init()
    logger.info(
        "raw weight push loop started",
        extra={"master": client.master_base_url, "slug": client.challenge_slug},
    )
    while True:
        try:
            with activate_role(
                Role.CHALLENGE, capabilities=(Capability.CHALLENGE_RAW_WEIGHT_PUSH,)
            ):
                result = await client.push_once()
            logger.info(
                "raw weight push attempt",
                extra={
                    "status": result.status,
                    "epoch": result.epoch,
                    "revision": result.revision,
                    "cursor_advanced": result.cursor_advanced,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if not resilient:
                raise
            logger.exception("raw weight push loop iteration failed")
        await asyncio.sleep(max(float(interval_seconds), 0.1))


def maybe_build_push_client_from_settings(
    *,
    settings: Any,
    database: Database,
    get_weights_fn: Callable[[], Awaitable[Mapping[str, float]]] | None = None,
) -> RawWeightPushClient | None:
    """Construct a push client when master_base_url + token enable raw-weight push."""

    if not bool(getattr(settings, "raw_weight_push_enabled", True)):
        return None
    master_url = getattr(settings, "master_base_url", None) or getattr(
        getattr(settings, "worker_plane", None), "master_base_url", None
    )
    if not master_url:
        return None
    token_loader = getattr(settings, "internal_token", None)
    token = token_loader() if callable(token_loader) else None
    if not token:
        shared = getattr(settings, "shared_token", None) or getattr(
            settings, "challenge_shared_token", None
        )
        token = str(shared) if shared else None
    if not token:
        return None
    epoch_seconds = int(getattr(settings, "epoch_seconds", 3600) or 3600)
    interval_hint = float(getattr(settings, "raw_weight_push_interval_seconds", 30.0))
    slug = str(
        getattr(settings, "slug", None)
        or getattr(settings, "challenge_slug", None)
        or DEFAULT_CHALLENGE_SLUG
    )

    def _epoch() -> int:
        return int(datetime.now(UTC).timestamp()) // max(epoch_seconds, 1)

    client = RawWeightPushClient(
        database=database,
        challenge_slug=slug,
        master_base_url=str(master_url),
        shared_token=str(token),
        weights_fn=build_weights_loader(get_weights_fn),
        epoch_fn=_epoch,
        freshness_seconds=int(
            getattr(settings, "raw_weight_push_freshness_seconds", DEFAULT_FRESHNESS_SECONDS)
        ),
        timeout_seconds=float(
            getattr(settings, "raw_weight_push_timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        ),
    )
    client.push_interval_seconds = interval_hint  # type: ignore[attr-defined]
    return client


__all__ = [
    "PushAttemptResult",
    "PushCursor",
    "RawWeightPushClient",
    "RawWeightPushLedger",
    "RawWeightPushStore",
    "build_weights_loader",
    "canonical_challenge_push_request",
    "maybe_build_push_client_from_settings",
    "run_raw_weight_push_loop",
    "sign_challenge_push_request",
]
