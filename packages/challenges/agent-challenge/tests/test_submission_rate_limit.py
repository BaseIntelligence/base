from __future__ import annotations

import asyncio
import base64
import io
import zipfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.models import AgentSubmission, RateLimitReservation
from agent_challenge.security import SignedRequestAuth
from agent_challenge.submissions.rate_limit import (
    DEFAULT_SUBMISSION_WINDOW_SECONDS,
    effective_submission_window_seconds,
    submission_rate_limit_message,
)

NOW = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)
ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"


def agent_source(contents: str | bytes) -> str | bytes:
    if isinstance(contents, bytes):
        return contents
    if "class Agent" in contents:
        return contents
    return f"{ENTRYPOINT_SOURCE}\n{contents}"


@pytest.fixture
def signed_submission_override() -> AsyncIterator[None]:
    async def authenticate() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey="rate-limit-hotkey",
            signature="test-signature",
            nonce="test-nonce",
            timestamp=NOW.isoformat(),
            body_sha256="test-body-sha256",
            canonical_request="signed-test-request",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.signed_submission_auth, None)


@pytest.fixture
def rate_limit_clock(monkeypatch) -> AsyncIterator[None]:
    current_now = NOW

    def set_now(value: datetime) -> None:
        nonlocal current_now
        current_now = value

    monkeypatch.setattr("agent_challenge.submissions.rate_limit._utc_now", lambda: current_now)
    yield set_now


def build_zip(contents: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("agent.py", agent_source(contents))
    return buffer.getvalue()


def submission_payload(name: str, contents: str) -> dict[str, str]:
    return {
        "name": name,
        "artifact_zip_base64": base64.b64encode(build_zip(contents)).decode("ascii"),
    }


def _patch_submission_settings(monkeypatch, tmp_path, *, window_seconds: int) -> None:
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.submission_rate_limit_window_seconds",
        window_seconds,
    )


@pytest.mark.parametrize(
    ("window_seconds", "advance", "expected_next", "expected_message"),
    [
        (
            1,
            timedelta(seconds=1),
            "2026-05-22T12:00:01+00:00",
            "one submission per hotkey is allowed every 1 second",
        ),
        (
            300,
            timedelta(seconds=300),
            "2026-05-22T12:05:00+00:00",
            "one submission per hotkey is allowed every 5 minutes",
        ),
        (
            10_800,
            timedelta(hours=3),
            "2026-05-22T15:00:00+00:00",
            "one submission per hotkey is allowed every 3 hours",
        ),
    ],
)
async def test_create_path_honors_settings_submission_rate_window(
    client,
    database_session,
    monkeypatch,
    rate_limit_clock,
    signed_submission_override,
    tmp_path,
    window_seconds: int,
    advance: timedelta,
    expected_next: str,
    expected_message: str,
):
    """VAL-E2E-009/011/012: live create passes Settings window into reserve."""
    _patch_submission_settings(monkeypatch, tmp_path, window_seconds=window_seconds)

    first = await client.post("/submissions", json=submission_payload("agent-a", "print('a')\n"))
    second = await client.post("/submissions", json=submission_payload("agent-b", "print('b')\n"))
    rate_limit_clock(NOW + advance)
    third = await client.post("/submissions", json=submission_payload("agent-c", "print('c')\n"))

    assert first.status_code == 201
    assert second.status_code == 429
    detail = second.json()["detail"]
    assert detail["code"] == "submission_rate_limited"
    assert detail["message"] == expected_message
    assert detail["next_allowed_at"] == expected_next
    assert "3 hours" not in expected_message or window_seconds == 10_800
    if window_seconds != 10_800:
        assert "3 hours" not in detail["message"]
    assert third.status_code == 201
    async with database_session() as session:
        reservation = await session.scalar(select(RateLimitReservation))
        assert reservation is not None
        assert reservation.window_seconds == window_seconds


async def test_settings_window_zero_still_enforces_floor_one_second(
    client,
    database_session,
    monkeypatch,
    rate_limit_clock,
    signed_submission_override,
    tmp_path,
):
    """VAL-E2E-010: window 0 is not disable; enforcement floors at 1s."""
    _patch_submission_settings(monkeypatch, tmp_path, window_seconds=0)

    first = await client.post("/submissions", json=submission_payload("agent-a", "print('a')\n"))
    second = await client.post("/submissions", json=submission_payload("agent-b", "print('b')\n"))

    assert first.status_code == 201
    assert second.status_code == 429
    detail = second.json()["detail"]
    assert detail["code"] == "submission_rate_limited"
    assert detail["next_allowed_at"] == "2026-05-22T12:00:01+00:00"
    assert "3 hours" not in detail["message"]
    assert "1 second" in detail["message"]
    async with database_session() as session:
        reservation = await session.scalar(select(RateLimitReservation))
        assert reservation is not None
        assert reservation.window_seconds == 1


def test_submission_rate_limit_message_tracks_window() -> None:
    assert submission_rate_limit_message(1) == (
        "one submission per hotkey is allowed every 1 second"
    )
    assert submission_rate_limit_message(60) == (
        "one submission per hotkey is allowed every 1 minute"
    )
    assert submission_rate_limit_message(300) == (
        "one submission per hotkey is allowed every 5 minutes"
    )
    assert submission_rate_limit_message(DEFAULT_SUBMISSION_WINDOW_SECONDS) == (
        "one submission per hotkey is allowed every 3 hours"
    )
    assert submission_rate_limit_message(0) == (
        "one submission per hotkey is allowed every 1 second"
    )


def test_effective_window_floors_at_one() -> None:
    assert effective_submission_window_seconds(0) == 1
    assert effective_submission_window_seconds(-5) == 1
    assert effective_submission_window_seconds(1) == 1
    assert effective_submission_window_seconds(10_800) == 10_800


def test_create_handler_passes_settings_window_kwarg() -> None:
    """VAL-E2E-009/013: static contract — call site uses window_seconds= from settings."""
    from pathlib import Path

    import agent_challenge.api.routes as api_routes

    routes_source = Path(api_routes.__file__).read_text(encoding="utf-8")
    assert "window_seconds=window_seconds" in routes_source
    assert "settings.submission_rate_limit_window_seconds" in routes_source
    assert "submission_rate_limit_message(window_seconds)" in routes_source
    hard_coded = 'message": "one submission per hotkey is allowed every 3 hours"'
    assert hard_coded not in routes_source


async def test_first_submission_consumes_slot_second_returns_429_then_after_3h_succeeds(
    client,
    database_session,
    monkeypatch,
    rate_limit_clock,
    signed_submission_override,
    tmp_path,
):
    _patch_submission_settings(monkeypatch, tmp_path, window_seconds=10_800)

    first = await client.post("/submissions", json=submission_payload("agent-a", "print('a')\n"))
    second = await client.post("/submissions", json=submission_payload("agent-b", "print('b')\n"))
    rate_limit_clock(NOW + timedelta(hours=3))
    third = await client.post("/submissions", json=submission_payload("agent-c", "print('c')\n"))

    assert first.status_code == 201
    assert second.status_code == 429
    assert second.json()["detail"] == {
        "code": "submission_rate_limited",
        "message": "one submission per hotkey is allowed every 3 hours",
        "next_allowed_at": "2026-05-22T15:00:00+00:00",
    }
    assert third.status_code == 201
    async with database_session() as session:
        reservation_count = await session.scalar(select(func.count(RateLimitReservation.id)))
        consumed_count = await session.scalar(
            select(func.count(RateLimitReservation.id)).where(
                RateLimitReservation.status == "consumed"
            )
        )
        submission_count = await session.scalar(select(func.count(AgentSubmission.id)))
    assert reservation_count == 2
    assert consumed_count == 2
    assert submission_count == 2


async def test_invalid_zip_creates_no_reservation_and_does_not_consume_slot(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )

    invalid = await client.post(
        "/submissions",
        json={
            "name": "bad-agent",
            "artifact_zip_base64": base64.b64encode(b"nope").decode("ascii"),
        },
    )
    valid = await client.post("/submissions", json=submission_payload("agent-a", "print('a')\n"))

    assert invalid.status_code == 400
    assert invalid.json()["detail"]["code"] == "invalid_zip"
    assert valid.status_code == 201
    async with database_session() as session:
        reservations = (await session.scalars(select(RateLimitReservation))).all()
    assert len(reservations) == 1
    assert reservations[0].status == "consumed"


async def test_unsigned_submission_creates_no_rate_limit_reservation(
    client,
    database_session,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )

    response = await client.post("/submissions", json=submission_payload("agent-a", "print('a')\n"))

    assert response.status_code == 401
    async with database_session() as session:
        reservation_count = await session.scalar(select(func.count(RateLimitReservation.id)))
    assert reservation_count == 0


async def test_concurrent_same_hotkey_submissions_accept_exactly_one_reservation(
    database_session,
    monkeypatch,
    rate_limit_clock,
    signed_submission_override,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    transport = ASGITransport(app=app)

    async def submit(name: str, contents: str) -> int:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/submissions", json=submission_payload(name, contents))
            return response.status_code

    statuses = await asyncio.gather(
        submit("agent-a", "print('a')\n"),
        submit("agent-b", "print('b')\n"),
    )

    assert sorted(statuses) == [201, 429]
    async with database_session() as session:
        reservations = (await session.scalars(select(RateLimitReservation))).all()
        submission_count = await session.scalar(select(func.count(AgentSubmission.id)))
    assert len(reservations) == 1
    assert reservations[0].status == "consumed"
    assert submission_count == 1
