"""Canonical FastAPI application factory for independently packaged challenges."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from time import time
from typing import Any, Protocol

from fastapi import APIRouter, Depends, FastAPI

from .auth import build_internal_auth_dependency
from .config import ChallengeSettings
from .roles import Role
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await database.init()
        tasks = [
            asyncio.create_task(factory(app), name="challenge-background-task")
            for factory in background_tasks
        ]
        for task in tasks:
            task.add_done_callback(_log_unexpected_background_exit)
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await database.close()

    app = FastAPI(title=settings.name, version=settings.version, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    async def health() -> HealthResponse:
        return HealthResponse(
            slug=settings.slug,
            version=settings.version,
            role=Role.CHALLENGE.value,
            capabilities=tuple(settings.capabilities),
        )

    @app.get("/version", response_model=VersionResponse, include_in_schema=False)
    async def version() -> VersionResponse:
        return VersionResponse(
            distribution_name=DISTRIBUTION_NAME,
            artifact_version=ARTIFACT_VERSION,
            release_id=RELEASE_ID,
            api_version=API_VERSION,
            challenge_version=settings.version,
            sdk_contract_version=SDK_CONTRACT_VERSION,
            sdk_version=SDK_CONTRACT_VERSION,
            capabilities=tuple(settings.capabilities),
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

    app.include_router(internal_router)
    app.include_router(public_router)
    return app


__all__ = ["ChallengeDatabase", "create_challenge_app"]
