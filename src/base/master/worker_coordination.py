"""Master worker coordination plane (architecture.md sec 3.3).

Persists the miner-funded GPU worker registry and its enrollment surface,
mirroring :mod:`base.master.validator_coordination`:

* ``POST /v1/workers/register`` verifies the miner's sr25519 binding signature
  against the (mock) metagraph, protects against binding-nonce replay, refuses a
  silent rebind of a pubkey to a different miner hotkey, and creates a
  ``pending`` worker (idempotent re-enroll for the same owner).
* ``POST /v1/workers/{worker_id}/heartbeat`` flips ``pending``/``stale`` ->
  ``active`` (never resurrects ``retired``).
* ``GET /v1/workers`` is the fleet view (status/owner/provider/last-seen/faults).
* ``GET /v1/workers/active?hotkey=`` is the admission surface (exactly the ACTIVE
  workers of a miner hotkey).

Lifecycle: ``pending -> active -> stale -> retired``. ``active`` requires a
verified binding AND a heartbeat within ``worker_heartbeat_ttl_seconds``;
``retired`` is terminal.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.db.models import WorkerFault, WorkerRegistration, WorkerStatus
from base.db.session import session_scope
from base.schemas.worker import (
    WorkerFaultView,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerListResponse,
    WorkerRegisterRequest,
    WorkerRegisterResponse,
    WorkerView,
)
from base.security.miner_auth import SignatureVerifier, verify_substrate_signature
from base.security.validator_auth import body_sha256
from base.security.worker_auth import (
    WorkerNonceStore,
    WorkerReplayError,
    worker_binding_message,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKER_HEARTBEAT_TTL_SECONDS = 120


class MinerNotInMetagraphError(ValueError):
    """Binding miner hotkey is absent from the (mock) metagraph (HTTP 403)."""


class WorkerBindingSignatureError(ValueError):
    """Binding signature does not verify against the miner hotkey (HTTP 401)."""


class WorkerRebindError(ValueError):
    """A pubkey re-registration under a different miner hotkey (HTTP 409)."""


class WorkerNotRegisteredError(LookupError):
    """Heartbeat for a ``worker_id`` without a row (HTTP 404)."""


class WorkerOwnershipError(ValueError):
    """Heartbeat signer does not own the target ``worker_id`` (HTTP 403)."""


class MinerMembership(Protocol):
    def is_registered(self, hotkey: str) -> bool: ...


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class WorkerCoordinationService:
    """Persist worker registration, heartbeat liveness, and lifecycle."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        miner_membership: MinerMembership,
        binding_nonce_store: WorkerNonceStore,
        signature_verifier: SignatureVerifier = verify_substrate_signature,
        heartbeat_ttl_seconds: int = DEFAULT_WORKER_HEARTBEAT_TTL_SECONDS,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
        worker_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._session_factory = session_factory
        self._miner_membership = miner_membership
        self._binding_nonce_store = binding_nonce_store
        self._signature_verifier = signature_verifier
        self.heartbeat_ttl_seconds = heartbeat_ttl_seconds
        self._now_fn = now_fn
        self._worker_id_factory = worker_id_factory

    def current_time(self) -> datetime:
        return self._now_fn()

    def effective_status(
        self, worker: WorkerRegistration, now: datetime
    ) -> WorkerStatus:
        """Derive the reported status from persisted status + heartbeat freshness.

        ``retired`` and ``pending`` are returned verbatim (``retired`` is
        terminal; ``pending`` awaits a first heartbeat). An ``active``/``stale``
        worker is reported ``active`` only while its heartbeat is within the TTL,
        so a query is correct regardless of whether the background staleness pass
        has run yet.
        """

        current = WorkerStatus(worker.status)
        if current in (WorkerStatus.RETIRED, WorkerStatus.PENDING):
            return current
        last = worker.last_heartbeat_at
        if last is None:
            return WorkerStatus.STALE
        if now - _as_utc(last) > timedelta(seconds=self.heartbeat_ttl_seconds):
            return WorkerStatus.STALE
        return WorkerStatus.ACTIVE

    async def register(
        self,
        *,
        worker_pubkey: str,
        miner_hotkey: str,
        binding_signature: str,
        nonce: str,
        provider: str,
        provider_instance_ref: str | None,
        capabilities: list[str],
        last_seen_meta: Mapping[str, Any] | None = None,
    ) -> WorkerRegistration:
        """Enroll a worker after verifying the miner binding.

        Verifies (1) the miner hotkey is on the metagraph, (2) the sr25519
        binding signature, (3) the binding nonce has not been replayed, then
        upserts the ``worker_registrations`` row as ``pending``. A pubkey already
        bound to a DIFFERENT miner hotkey raises :class:`WorkerRebindError` (no
        silent rebind); the same owner re-enrolling with a fresh nonce updates
        the single existing row (idempotent, restart-safe).
        """

        now = self._now_fn()
        if not self._miner_membership.is_registered(miner_hotkey):
            raise MinerNotInMetagraphError(miner_hotkey)

        message = worker_binding_message(
            worker_pubkey=worker_pubkey, miner_hotkey=miner_hotkey, nonce=nonce
        )
        if not self._signature_verifier(miner_hotkey, message, binding_signature):
            raise WorkerBindingSignatureError(worker_pubkey)

        await self._binding_nonce_store.reserve(
            hotkey=miner_hotkey,
            nonce=nonce,
            body_hash=body_sha256(message),
            created_at=now,
        )

        try:
            async with session_scope(self._session_factory) as session:
                return await self._register_in_session(
                    session,
                    now=now,
                    worker_pubkey=worker_pubkey,
                    miner_hotkey=miner_hotkey,
                    binding_signature=binding_signature,
                    nonce=nonce,
                    provider=provider,
                    provider_instance_ref=provider_instance_ref,
                    capabilities=capabilities,
                    last_seen_meta=last_seen_meta,
                )
        except IntegrityError:
            # Lost the first-register race on worker_pubkey: re-run as an update.
            async with session_scope(self._session_factory) as session:
                return await self._register_in_session(
                    session,
                    now=now,
                    worker_pubkey=worker_pubkey,
                    miner_hotkey=miner_hotkey,
                    binding_signature=binding_signature,
                    nonce=nonce,
                    provider=provider,
                    provider_instance_ref=provider_instance_ref,
                    capabilities=capabilities,
                    last_seen_meta=last_seen_meta,
                )

    async def _register_in_session(
        self,
        session: AsyncSession,
        *,
        now: datetime,
        worker_pubkey: str,
        miner_hotkey: str,
        binding_signature: str,
        nonce: str,
        provider: str,
        provider_instance_ref: str | None,
        capabilities: list[str],
        last_seen_meta: Mapping[str, Any] | None,
    ) -> WorkerRegistration:
        existing = (
            await session.execute(
                select(WorkerRegistration).where(
                    WorkerRegistration.worker_pubkey == worker_pubkey
                )
            )
        ).scalar_one_or_none()

        if existing is not None:
            if existing.miner_hotkey != miner_hotkey:
                raise WorkerRebindError(worker_pubkey)
            existing.binding_signature = binding_signature
            existing.binding_nonce = nonce
            existing.provider = provider
            existing.provider_instance_ref = provider_instance_ref
            existing.capabilities = list(capabilities)
            existing.status = WorkerStatus.PENDING
            existing.last_heartbeat_at = None
            if last_seen_meta is not None:
                existing.last_seen_meta = dict(last_seen_meta)
            return existing

        worker = WorkerRegistration(
            worker_id=self._worker_id_factory(),
            worker_pubkey=worker_pubkey,
            miner_hotkey=miner_hotkey,
            binding_signature=binding_signature,
            binding_nonce=nonce,
            provider=provider,
            provider_instance_ref=provider_instance_ref,
            capabilities=list(capabilities),
            status=WorkerStatus.PENDING,
            last_seen_meta=dict(last_seen_meta or {}),
            last_heartbeat_at=None,
            created_at=now,
        )
        session.add(worker)
        return worker

    async def heartbeat(
        self,
        *,
        worker_id: str,
        worker_pubkey: str,
        last_seen_meta: Mapping[str, Any] | None = None,
    ) -> tuple[WorkerRegistration, datetime]:
        """Refresh liveness; flip ``pending``/``stale`` -> ``active``.

        ``retired`` is terminal: a heartbeat never resurrects it (recovery needs
        a fresh registration). Raises :class:`WorkerNotRegisteredError` for an
        unknown ``worker_id`` and :class:`WorkerOwnershipError` when the signer
        does not own it.
        """

        now = self._now_fn()
        async with session_scope(self._session_factory) as session:
            worker = (
                await session.execute(
                    select(WorkerRegistration).where(
                        WorkerRegistration.worker_id == worker_id
                    )
                )
            ).scalar_one_or_none()
            if worker is None:
                raise WorkerNotRegisteredError(worker_id)
            if worker.worker_pubkey != worker_pubkey:
                raise WorkerOwnershipError(worker_id)
            if WorkerStatus(worker.status) == WorkerStatus.RETIRED:
                return worker, now
            worker.last_heartbeat_at = now
            worker.status = WorkerStatus.ACTIVE
            if last_seen_meta is not None:
                worker.last_seen_meta = dict(last_seen_meta)
            return worker, now

    async def list_workers(self) -> list[WorkerRegistration]:
        """Return all workers ordered for stable fleet observability."""

        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(WorkerRegistration).order_by(
                            WorkerRegistration.created_at,
                            WorkerRegistration.worker_id,
                        )
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    async def faults_by_worker(self) -> dict[str, list[WorkerFault]]:
        """Return each worker's faults keyed by ``worker_id`` (time-ordered)."""

        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(WorkerFault).order_by(WorkerFault.created_at)
                    )
                )
                .scalars()
                .all()
            )
        grouped: dict[str, list[WorkerFault]] = {}
        for fault in rows:
            grouped.setdefault(fault.worker_id, []).append(fault)
        return grouped

    async def active_workers(self, miner_hotkey: str) -> list[WorkerRegistration]:
        """Return exactly the ACTIVE workers bound to ``miner_hotkey``.

        Retired/stale/pending workers and other owners are excluded; staleness is
        derived from the heartbeat freshness window at query time.
        """

        now = self._now_fn()
        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(WorkerRegistration)
                        .where(WorkerRegistration.miner_hotkey == miner_hotkey)
                        .order_by(
                            WorkerRegistration.created_at,
                            WorkerRegistration.worker_id,
                        )
                    )
                )
                .scalars()
                .all()
            )
        return [
            row
            for row in rows
            if self.effective_status(row, now) == WorkerStatus.ACTIVE
        ]

    async def detect_stale_workers(
        self, *, session: AsyncSession | None = None
    ) -> list[str]:
        """Persist ``active`` -> ``stale`` for workers past the TTL.

        Edge-triggered: only ``active`` workers are considered. Returns the
        ``worker_id``s that transitioned this pass.
        """

        now = self._now_fn()
        if session is not None:
            return await self._detect_stale_in_session(session, now)
        async with session_scope(self._session_factory) as own_session:
            return await self._detect_stale_in_session(own_session, now)

    async def _detect_stale_in_session(
        self, session: AsyncSession, now: datetime
    ) -> list[str]:
        ttl = timedelta(seconds=self.heartbeat_ttl_seconds)
        transitioned: list[str] = []
        rows = (
            (
                await session.execute(
                    select(WorkerRegistration).where(
                        WorkerRegistration.status == WorkerStatus.ACTIVE
                    )
                )
            )
            .scalars()
            .all()
        )
        for worker in rows:
            last = worker.last_heartbeat_at
            if last is None:
                continue
            if now - _as_utc(last) > ttl:
                worker.status = WorkerStatus.STALE
                transitioned.append(worker.worker_id)
        return transitioned


def worker_fault_to_view(fault: WorkerFault) -> WorkerFaultView:
    return WorkerFaultView(
        work_unit_id=fault.work_unit_id,
        challenge_slug=fault.challenge_slug,
        detail=fault.detail,
        created_at=fault.created_at,
    )


def worker_to_view(
    worker: WorkerRegistration,
    *,
    service: WorkerCoordinationService,
    now: datetime,
    faults: list[WorkerFault] | None = None,
) -> WorkerView:
    """Build the fleet view for a worker with its derived (effective) status."""

    return WorkerView(
        worker_id=worker.worker_id,
        worker_pubkey=worker.worker_pubkey,
        miner_hotkey=worker.miner_hotkey,
        provider=worker.provider,
        provider_instance_ref=worker.provider_instance_ref,
        capabilities=list(worker.capabilities),
        status=service.effective_status(worker, now).value,
        last_heartbeat_at=worker.last_heartbeat_at,
        created_at=worker.created_at,
        faults=[worker_fault_to_view(fault) for fault in (faults or [])],
    )


def build_worker_coordination_router(
    *,
    service: WorkerCoordinationService,
    auth_dependency: Callable[..., Any],
) -> APIRouter:
    """Build the worker coordination router (register/heartbeat/fleet reads).

    ``auth_dependency`` authenticates heartbeat + fleet reads as a registered
    worker (or, for reads, an eligible validator). Registration is authenticated
    by the miner binding signature carried in the body, not this dependency.
    """

    router = APIRouter()

    @router.post("/v1/workers/register", response_model=WorkerRegisterResponse)
    async def register_worker(
        payload: WorkerRegisterRequest,
    ) -> WorkerRegisterResponse:
        try:
            worker = await service.register(
                worker_pubkey=payload.worker_pubkey,
                miner_hotkey=payload.miner_hotkey,
                binding_signature=payload.binding_signature,
                nonce=payload.nonce,
                provider=payload.provider,
                provider_instance_ref=payload.provider_instance_ref,
                capabilities=payload.capabilities,
                last_seen_meta=payload.last_seen_meta,
            )
        except MinerNotInMetagraphError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="miner hotkey is not in the metagraph",
            ) from exc
        except WorkerBindingSignatureError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid worker binding signature",
            ) from exc
        except WorkerReplayError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="binding nonce already used",
            ) from exc
        except WorkerRebindError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="worker pubkey already bound to a different miner hotkey",
            ) from exc
        now = service.current_time()
        return WorkerRegisterResponse(
            worker=worker_to_view(worker, service=service, now=now),
            heartbeat_ttl_seconds=service.heartbeat_ttl_seconds,
        )

    @router.post(
        "/v1/workers/{worker_id}/heartbeat",
        response_model=WorkerHeartbeatResponse,
    )
    async def heartbeat_worker(
        worker_id: str,
        payload: WorkerHeartbeatRequest,
        identity: Any = Depends(auth_dependency),
    ) -> WorkerHeartbeatResponse:
        try:
            worker, now = await service.heartbeat(
                worker_id=worker_id,
                worker_pubkey=identity.hotkey,
                last_seen_meta=payload.last_seen_meta,
            )
        except WorkerNotRegisteredError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="worker not registered",
            ) from exc
        except WorkerOwnershipError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="worker id not owned by signer",
            ) from exc
        return WorkerHeartbeatResponse(
            status=service.effective_status(worker, now).value,
            now=now,
        )

    @router.get(
        "/v1/workers",
        response_model=WorkerListResponse,
        dependencies=[Depends(auth_dependency)],
    )
    async def list_workers() -> WorkerListResponse:
        now = service.current_time()
        rows = await service.list_workers()
        faults = await service.faults_by_worker()
        return WorkerListResponse(
            workers=[
                worker_to_view(
                    row, service=service, now=now, faults=faults.get(row.worker_id)
                )
                for row in rows
            ]
        )

    @router.get(
        "/v1/workers/active",
        response_model=WorkerListResponse,
        dependencies=[Depends(auth_dependency)],
    )
    async def list_active_workers(
        hotkey: str = Query(...),
    ) -> WorkerListResponse:
        now = service.current_time()
        rows = await service.active_workers(hotkey)
        return WorkerListResponse(
            workers=[worker_to_view(row, service=service, now=now) for row in rows]
        )

    return router


async def run_worker_health_loop(
    service: WorkerCoordinationService,
    *,
    interval_seconds: float,
    shutdown_event: asyncio.Event,
) -> None:
    """Run the staleness pass every ``interval_seconds`` until shutdown."""

    while not shutdown_event.is_set():
        try:
            await service.detect_stale_workers()
        except Exception:
            logger.exception("worker staleness detection pass failed")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue


def build_worker_health_lifespan(
    service: WorkerCoordinationService | None,
    interval_seconds: float | None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]] | None:
    """Build a FastAPI lifespan running the staleness loop (None when disabled)."""

    if service is None or interval_seconds is None or interval_seconds <= 0:
        return None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            run_worker_health_loop(
                service,
                interval_seconds=interval_seconds,
                shutdown_event=shutdown,
            )
        )
        try:
            yield
        finally:
            shutdown.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    return lifespan
