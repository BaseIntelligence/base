"""Root logging configuration shared by the Agent Challenge entrypoints.

Uvicorn's default logging config installs no handler on the stdlib root logger,
so without an explicit call here every application logger (``evaluation.worker``,
``own_runner.orchestrator``, the runner, ...) is silent under both the API and
the worker service. Each process entrypoint calls :func:`configure_root_logging`
once so INFO-level application logs are actually emitted.
"""

from __future__ import annotations

import logging

from base.observability.logging import configure_logging

from .config import ChallengeSettings


def resolve_log_level(value: str | int) -> int:
    """Map a ``CHALLENGE_LOG_LEVEL`` value to a stdlib logging level int.

    Accepts a level name (``"INFO"``), a numeric string (``"20"``), or an int.
    Unknown names fall back to :data:`logging.INFO` rather than raising, so a
    typo can never crash an entrypoint before logging is even configured.
    """

    if isinstance(value, int):
        return value
    text = value.strip().upper()
    if text.isdigit():
        return int(text)
    resolved = logging.getLevelName(text)
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_root_logging(settings: ChallengeSettings) -> None:
    """Configure stdlib root logging at the ``CHALLENGE_LOG_LEVEL`` level.

    Uses the shared :func:`base.observability.logging.configure_logging` helper
    (JSON logs, matching the rest of the BASE platform) with ``force=True`` so it
    is safe to call exactly once at each process entrypoint.
    """

    configure_logging(json_logs=True, level=resolve_log_level(settings.log_level))
