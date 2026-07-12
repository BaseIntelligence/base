"""Canonical FastAPI application factory for independently packaged challenges."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from time import time
from typing import Any, Protocol

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .auth import build_internal_auth_dependency, load_shared_token
from .config import ChallengeSettings
from .health import ReadinessProbe, evaluate_readiness, health_from_checks
from .roles import Capability, Role, activate_role, role_contract
from .schemas import HealthResponse, VersionResponse, WeightsResponse
from .version import (
    API_VERSION,
    ARTIFACT_VERSION,
    DISTRIBUTION_NAME,
    RELEASE_ID,
    SDK_CONTRACT_VERSION,
)

GetWeightsFn = Callable[[], Awaitable[dict[str, float]]]
BackgroundTaskFactory = Callable[[FastAPI], Coroutine[Any, Any, None]]

_logger = logging.getLogger("base.challenge_sdk.app_factory")


class ChallengeDatabase(Protocol):
    async def init(self) -> None: ...

    async def close(self) -> None: ...


def _log_unexpected_background_exit(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        _logger.critical("background task exited unexpectedly", exc_info=error)
    else:
        _logger.critical("background task exited unexpectedly without error")
    signal.raise_signal(signal.SIGTERM)


def create_challenge_app(
    *,
    settings: ChallengeSettings,
    database: ChallengeDatabase,
    public_router: APIRouter,
    get_weights_fn: GetWeightsFn,
    background_tasks: Sequence[BackgroundTaskFactory] = (),
    readiness_probes: Sequence[ReadinessProbe] = (),
) -> FastAPI:
    """Create a challenge app with canonical lifecycle and identity routes."""

    if settings.api_version != API_VERSION:
        raise ValueError(
            "Incompatible challenge API version: "
            f"expected {API_VERSION!r}, actual {settings.api_version!r}"
        )
    if settings.sdk_version != SDK_CONTRACT_VERSION:
        raise ValueError(
            "Incompatible challenge SDK version: "
            f"expected {SDK_CONTRACT_VERSION!r}, actual {settings.sdk_version!r}"
        )
    server_capabilities = tuple(settings.capabilities)
    configured_probes = list(readiness_probes)
    database_healthcheck = getattr(database, "healthcheck", None)
    if callable(database_healthcheck):
        configured_probes.insert(
            0,
            ReadinessProbe(name="database", check=database_healthcheck),
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_shared_token(settings) is None:
            raise RuntimeError(
                "challenge authentication secret is missing or empty; refusing to start"
            )
        await database.init()
        tasks: list[asyncio.Task[None]] = []
        try:
            with activate_role(
                settings.role,
                capabilities=server_capabilities,
            ):
                tasks = [
                    asyncio.create_task(factory(app), name="challenge-background-task")
                    for factory in background_tasks
                ]
                app.state.challenge_background_tasks = tuple(tasks)
                for task in tasks:
                    task.add_done_callback(_log_unexpected_background_exit)
                yield
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            app.state.challenge_background_tasks = ()
            await database.close()

    app = FastAPI(title=settings.name, version=settings.version, lifespan=lifespan)
    server_capabilities = tuple(settings.capabilities)

    @app.middleware("http")
    async def establish_challenge_role(
        request: Request,
        call_next: Callable[[Request], Coroutine[Any, Any, Response]],
    ) -> Response:
        with activate_role(
            settings.role,
            capabilities=server_capabilities,
        ):
            return await call_next(request)

    @app.middleware("http")
    async def refuse_mutations_while_unready(
        request: Request,
        call_next: Callable[[Request], Coroutine[Any, Any, Response]],
    ) -> Response:
        discovery_path = request.url.path in {"/health", "/ready", "/version"}
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not discovery_path:
            health = await current_health()
            if not health.ready:
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": {
                            "code": "runtime_not_ready",
                            "detail": (
                                "mandatory runtime dependencies are unavailable"
                            ),
                        }
                    },
                )
        return await call_next(request)

    if background_tasks:

        def required_worker_running() -> bool:
            tasks = getattr(app.state, "challenge_background_tasks", ())
            return len(tasks) == len(background_tasks) and all(
                not task.done() for task in tasks
            )

        configured_probes.append(
            ReadinessProbe(name="worker", check=required_worker_running)
        )

    async def current_health() -> HealthResponse:
        checks = await evaluate_readiness(configured_probes)
        return health_from_checks(
            slug=settings.slug,
            version=settings.version,
            role=Role.CHALLENGE.value,
            capabilities=server_capabilities,
            checks=checks,
        )

    @app.api_route(
        "/health",
        methods=["GET", "HEAD"],
        response_model=HealthResponse,
        include_in_schema=False,
    )
    @role_contract(role=Role.CHALLENGE, capability=Capability.CHALLENGE_STATE)
    async def health() -> HealthResponse:
        return await current_health()

    @app.api_route(
        "/ready",
        methods=["GET", "HEAD"],
        response_model=HealthResponse,
        include_in_schema=False,
    )
    @role_contract(role=Role.CHALLENGE, capability=Capability.CHALLENGE_STATE)
    async def ready() -> HealthResponse | JSONResponse:
        response = await current_health()
        if response.ready:
            return response
        return JSONResponse(
            status_code=503,
            content=response.model_dump(mode="json"),
        )

    @app.api_route(
        "/version",
        methods=["GET", "HEAD"],
        response_model=VersionResponse,
        include_in_schema=False,
    )
    @role_contract(role=Role.CHALLENGE, capability=Capability.CHALLENGE_STATE)
    async def version() -> VersionResponse:
        return VersionResponse(
            distribution_name=DISTRIBUTION_NAME,
            artifact_version=ARTIFACT_VERSION,
            release_id=RELEASE_ID,
            api_version=API_VERSION,
            challenge_slug=settings.slug,
            challenge_version=settings.version,
            sdk_contract_version=SDK_CONTRACT_VERSION,
            sdk_version=SDK_CONTRACT_VERSION,
            capabilities=server_capabilities,
        )

    internal_router = APIRouter(
        prefix="/internal/v1",
        dependencies=[Depends(build_internal_auth_dependency(settings))],
    )

    @internal_router.get("/get_weights", response_model=WeightsResponse)
    @role_contract(role=Role.CHALLENGE, capability=Capability.CHALLENGE_STATE)
    async def get_weights() -> WeightsResponse:
        weights = await get_weights_fn()
        return WeightsResponse(
            challenge_slug=settings.slug,
            epoch=int(time()),
            weights=weights,
        )

    app.include_router(internal_router)
    app.include_router(public_router)
    return app


__all__ = ["ChallengeDatabase", "create_challenge_app"]
