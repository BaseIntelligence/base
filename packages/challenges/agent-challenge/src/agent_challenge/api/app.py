"""FastAPI application entrypoint for Agent Challenge."""

from __future__ import annotations

import functools

from ..core import models as _models  # noqa: F401 - register SQLAlchemy models
from ..core.config import settings
from ..core.db import database
from ..evaluation.weights import get_weights
from ..evaluation.worker import run_worker_loop
from ..sdk.app_factory import WorkerMain, create_challenge_app
from ..sdk.config import ChallengeSettings
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


app = create_challenge_app(
    settings=settings,
    database=database,
    public_router=router,
    get_weights_fn=get_weights,
    worker_main=build_worker_main(settings),
)
