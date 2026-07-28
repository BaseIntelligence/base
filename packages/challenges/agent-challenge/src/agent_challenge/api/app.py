"""FastAPI application entrypoint for Agent Challenge."""

from __future__ import annotations

import functools

from fastapi import FastAPI

from ..core import models as _models  # noqa: F401 - register SQLAlchemy models
from ..core.config import settings
from ..core.db import database
from ..evaluation import raw_weight_push as raw_weight_push_module
from ..evaluation.weights import get_weights
from ..evaluation.worker import run_worker_loop
from ..sdk.app_factory import BackgroundTaskFactory, WorkerMain, create_challenge_app
from ..sdk.config import ChallengeSettings
from ..sdk.db import Database
from ..sdk.observability import configure_root_logging
from .routes import router

# Configure stdlib root logging at import so application INFO is visible under
# uvicorn (whose default config installs no root handler). Runs before the app is
# built, and also covers the in-process worker loop in combined mode.
configure_root_logging(settings)


def build_worker_main(challenge_settings: ChallengeSettings) -> WorkerMain | None:
    """Return the combined-mode worker entrypoint, or None when disabled."""

    if not challenge_settings.combined_worker:
        return None
    # Reuse the exact loop the ``agent-challenge-worker`` CLI runs; the API
    # lifespan owns the shared Database, so the loop must not init/close it.
    return functools.partial(run_worker_loop, manage_database=False)


def create_app(
    *,
    challenge_settings: ChallengeSettings | None = None,
    db: Database | None = None,
) -> FastAPI:
    """Build the Agent Challenge FastAPI app (production and tests).

    Wires the optional combined worker and the authenticated raw-weight push
    loop the same way Prism does: build the client when settings enable it,
    append a background task factory, stash the client on ``app.state``.
    """

    app_settings = challenge_settings if challenge_settings is not None else settings
    app_db = db if db is not None else database
    configure_root_logging(app_settings)

    background_tasks: list[BackgroundTaskFactory] = []
    # Attribute access (not a bound import) so tests can monkeypatch the module.
    push_client = raw_weight_push_module.maybe_build_push_client_from_settings(
        settings=app_settings,
        database=app_db,
        get_weights_fn=get_weights,
    )
    if push_client is not None:
        interval = float(getattr(push_client, "push_interval_seconds", 30.0))

        async def _run_raw_weight_push(app: FastAPI) -> None:
            client = getattr(app.state, "raw_weight_push_client", push_client)
            await raw_weight_push_module.run_raw_weight_push_loop(
                client,
                interval_seconds=interval,
                resilient=True,
            )

        background_tasks.append(_run_raw_weight_push)

    app = create_challenge_app(
        settings=app_settings,
        database=app_db,
        public_router=router,
        get_weights_fn=get_weights,
        worker_main=build_worker_main(app_settings),
        background_tasks=tuple(background_tasks),
    )
    if push_client is not None:
        app.state.raw_weight_push_client = push_client
    return app


app = create_app()
