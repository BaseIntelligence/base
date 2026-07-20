from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.models import RateLimitReservation

SUBMISSION_LIMIT_KEY = "submission:create"
SUBMISSION_RESERVATION_KEY = "submission"
DEFAULT_SUBMISSION_WINDOW_SECONDS = 3 * 60 * 60


@dataclass(eq=False)
class RateLimitExceeded(Exception):
    # Not frozen: a frozen dataclass exception blocks Python from setting
    # __traceback__ during propagation (raises FrozenInstanceError that masks the
    # real error). eq=False keeps identity hashing/equality.
    next_allowed_at: datetime


@dataclass(frozen=True)
class SubmissionRateLimitReservation:
    row: RateLimitReservation
    next_allowed_at: datetime


def effective_submission_window_seconds(window_seconds: int) -> int:
    """Clamp product/operator window to the enforcement floor of 1 second.

    Settings may load 0, but 0 is not a disable switch; enforcement always uses
    max(window_seconds, 1).
    """
    return max(int(window_seconds), 1)


def submission_rate_limit_message(window_seconds: int) -> str:
    """Operator-facing 429 detail matching the active window (not a fixed 3h string)."""
    effective = effective_submission_window_seconds(window_seconds)
    if effective % 3600 == 0:
        hours = effective // 3600
        unit = "hour" if hours == 1 else "hours"
        return f"one submission per hotkey is allowed every {hours} {unit}"
    if effective % 60 == 0:
        minutes = effective // 60
        unit = "minute" if minutes == 1 else "minutes"
        return f"one submission per hotkey is allowed every {minutes} {unit}"
    unit = "second" if effective == 1 else "seconds"
    return f"one submission per hotkey is allowed every {effective} {unit}"


async def reserve_submission_rate_limit(
    *,
    session: AsyncSession,
    hotkey: str,
    artifact_hash: str,
    zip_sha256: str,
    zip_size_bytes: int,
    request_ip: str | None = None,
    user_agent: str | None = None,
    route: str | None = None,
    window_seconds: int = DEFAULT_SUBMISSION_WINDOW_SECONDS,
    now: datetime | None = None,
) -> SubmissionRateLimitReservation:
    current_time = _as_utc(now or _utc_now())
    window_seconds = effective_submission_window_seconds(window_seconds)
    window_start = _window_start(current_time, window_seconds)
    next_allowed_at = window_start + timedelta(seconds=window_seconds)
    metadata = _metadata_json(
        artifact_hash=artifact_hash,
        zip_sha256=zip_sha256,
        zip_size_bytes=zip_size_bytes,
        request_ip=request_ip,
        user_agent=user_agent,
        route=route,
    )
    reservation = RateLimitReservation(
        hotkey=hotkey,
        limit_key=SUBMISSION_LIMIT_KEY,
        window_start=window_start,
        window_seconds=window_seconds,
        reservation_key=SUBMISSION_RESERVATION_KEY,
        cost=1,
        status="reserved",
        expires_at=next_allowed_at,
        metadata_json=metadata,
    )
    session.add(reservation)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise RateLimitExceeded(next_allowed_at=next_allowed_at) from exc
    return SubmissionRateLimitReservation(row=reservation, next_allowed_at=next_allowed_at)


def consume_submission_rate_limit(reservation: SubmissionRateLimitReservation) -> None:
    reservation.row.status = "consumed"


def _metadata_json(
    *,
    artifact_hash: str,
    zip_sha256: str,
    zip_size_bytes: int,
    request_ip: str | None,
    user_agent: str | None,
    route: str | None,
) -> str:
    metadata: dict[str, Any] = {
        "artifact_hash": artifact_hash,
        "zip_sha256": zip_sha256,
        "zip_size_bytes": zip_size_bytes,
    }
    if request_ip:
        metadata["request_ip"] = request_ip
    if route:
        metadata["route"] = route
    if user_agent:
        metadata["user_agent"] = user_agent[:512]
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _window_start(value: datetime, window_seconds: int) -> datetime:
    timestamp = int(value.timestamp())
    return datetime.fromtimestamp(timestamp - (timestamp % window_seconds), UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)
