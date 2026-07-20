"""Submission admission rule: require >=1 active worker for the submitting hotkey.

architecture.md 3.5: when ``worker_plane.admission_requires_worker`` is on, a prism submission
(direct ``POST /v1/submissions`` AND the base bridge path ``POST /internal/v1/bridge/submissions``,
identical enforcement) is rejected with HTTP 403 code ``NO_ACTIVE_WORKER`` unless the base master's
``GET /v1/workers/active?hotkey=`` confirms at least one active worker bound to that hotkey.

Fail-closed and bounded (VAL-PRISM-020): a master that is unreachable, times out, or returns a
non-2xx (5xx/4xx) can never confirm a worker, so the submission is rejected with the SAME 403
``NO_ACTIVE_WORKER`` shape used for an explicit zero-worker answer -- one deterministic response,
capped by ``worker_plane.admission_timeout_seconds``. With the flag off the check is a no-op and no
master call is made, so submission behavior is byte-identical to legacy (VAL-PRISM-015).
"""

from __future__ import annotations

import logging

import httpx
from fastapi import HTTPException, status

from .config import PrismSettings

logger = logging.getLogger(__name__)

#: Error code returned in the 403 body when admission is denied (no confirmed active worker).
NO_ACTIVE_WORKER_CODE = "NO_ACTIVE_WORKER"

#: Master coordination path the admission rule queries for a hotkey's active workers.
ACTIVE_WORKERS_PATH = "/v1/workers/active"


def _bridge_bearer(settings: PrismSettings) -> str | None:
    """Reuse the prism<->master bridge shared token as the admission-query bearer, if configured."""
    try:
        return settings.internal_token()
    except RuntimeError:
        return None


async def count_active_workers(settings: PrismSettings, hotkey: str) -> int | None:
    """Return the number of ACTIVE workers the master reports for ``hotkey``.

    Returns ``None`` when the master cannot be consulted (unset URL, connection error, timeout, or
    a non-2xx response) so the caller fails closed. The query is bounded by
    ``worker_plane.admission_timeout_seconds`` so it never hangs a submission.
    """
    worker_plane = settings.worker_plane
    base_url = worker_plane.master_base_url
    if not base_url:
        logger.warning(
            "admission_requires_worker is on but worker_plane.master_base_url is unset; "
            "failing closed"
        )
        return None
    headers: dict[str, str] = {}
    token = _bridge_bearer(settings)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=worker_plane.admission_timeout_seconds,
        ) as client:
            response = await client.get(
                ACTIVE_WORKERS_PATH, params={"hotkey": hotkey}, headers=headers
            )
    except httpx.HTTPError as exc:
        logger.warning("admission master query failed (%s); failing closed", type(exc).__name__)
        return None
    if response.status_code >= 400:
        logger.warning("admission master returned HTTP %s; failing closed", response.status_code)
        return None
    try:
        payload = response.json()
    except ValueError:
        logger.warning("admission master returned non-JSON body; failing closed")
        return None
    workers = payload.get("workers") if isinstance(payload, dict) else None
    if not isinstance(workers, list):
        logger.warning("admission master response missing a workers list; failing closed")
        return None
    return len(workers)


async def enforce_admission(settings: PrismSettings, hotkey: str) -> None:
    """Reject a submission from ``hotkey`` unless the master confirms >=1 active worker.

    No-op unless ``worker_plane.admission_requires_worker`` is on (flag off => zero master calls,
    legacy behavior). On denial (or any fail-closed condition) raises HTTP 403 with the
    ``NO_ACTIVE_WORKER`` code so no submission row is ever created.
    """
    if not settings.worker_plane.admission_requires_worker:
        return
    active = await count_active_workers(settings, hotkey)
    if active is None or active < 1:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            {
                "code": NO_ACTIVE_WORKER_CODE,
                "detail": "no active worker bound to the submitting hotkey",
            },
        )
