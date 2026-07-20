"""FastAPI app factory for BASE challenge repositories."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from time import time

from fastapi import APIRouter, Depends, FastAPI

from .auth import build_internal_auth_dependency
from .config import ChallengeSettings
from .db import Database
from .schemas import HealthResponse, VersionResponse, WeightsResponse

logger = logging.getLogger(__name__)

GetWeightsFn = Callable[[], Awaitable[dict[str, float]]]
WorkerMain = Callable[[], Awaitable[None]]


def _handle_worker_task_done(task: asyncio.Task[None]) -> None:
    """Fail loud if the combined-mode worker loop exits on its own.

    Normal shutdown cancels the task (``task.cancelled()`` is True) -> no-op.
    Any other exit -- whether it RAISED or RETURNED normally -- means the eval
    queue silently stopped draining while ``/health`` still returns 200, so we
    log CRITICAL and raise SIGTERM to let uvicorn run graceful shutdown and Swarm
    restart the single combined service.
    """

    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.critical("combined-mode worker loop exited unexpectedly", exc_info=exc)
    else:
        logger.critical("combined-mode worker loop returned unexpectedly")
    signal.raise_signal(signal.SIGTERM)


def create_challenge_app(
    *,
    settings: ChallengeSettings,
    database: Database,
    public_router: APIRouter,
    get_weights_fn: GetWeightsFn,
    challenge_internal_router: APIRouter | None = None,
    worker_main: WorkerMain | None = None,
) -> FastAPI:
    """Create a complete FastAPI challenge app with standard BASE routes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Production full-attested deployments cannot accept results without an
        # endpoint-owned signer rebind; fail closed at startup rather than after
        # receipt.
        settings.require_eval_result_signer_for_production()
        # Same fail-closed for evidence Fernet key: admit only when dual attestation
        # is ON and the dedicated key material is loadable (no secret leak).
        settings.require_review_evidence_encryption_for_production()
        # Dual-flag production also needs dcap-qvl on PATH (baked into runtime
        # image). Binary presence only; no PCS network / secret / trust-root invent.
        settings.require_dcap_qvl_binary_for_production()
        await database.init()
        worker_task: asyncio.Task[None] | None = None
        if worker_main is not None:
            worker_task = asyncio.create_task(worker_main(), name="combined-worker-loop")
            worker_task.add_done_callback(_handle_worker_task_done)
        try:
            yield
        finally:
            if worker_task is not None:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # Already surfaced by the done-callback; keep shutdown clean.
                    logger.exception("combined-mode worker loop crashed during shutdown")
            await database.close()

    app = FastAPI(title=settings.name, version=settings.version, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    async def health() -> HealthResponse:
        return HealthResponse(slug=settings.slug, version=settings.version)

    @app.get("/version", response_model=VersionResponse, include_in_schema=False)
    async def version() -> VersionResponse:
        capabilities = ["get_weights", "proxy_routes", "sqlite", "swe_forge"]
        if settings.docker_enabled:
            capabilities.append("docker_executor")
        return VersionResponse(
            api_version=settings.api_version,
            challenge_version=settings.version,
            sdk_version=settings.sdk_version,
            capabilities=capabilities,
        )

    internal_router = APIRouter(
        prefix="/internal/v1",
        dependencies=[Depends(build_internal_auth_dependency(settings))],
    )

    @internal_router.get("/get_weights", response_model=WeightsResponse)
    async def get_weights() -> WeightsResponse:
        weights = await get_weights_fn()
        return WeightsResponse(
            challenge_slug=settings.slug,
            epoch=int(time()),
            weights=weights,
        )

    if challenge_internal_router is not None:
        internal_router.include_router(challenge_internal_router)

    app.include_router(internal_router)
    app.include_router(public_router)
    return app
