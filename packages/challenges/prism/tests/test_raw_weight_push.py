"""Prism raw-weight push client and durable acknowledgement cursor tests.

Covers VAL-SDK-017/VAL-WEIGHT-028..030/101: schema compatibility with Base,
exact ack gate for cursor advance, mismatch rejection, and restart retry of
the same logical snapshot.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from base.challenge_sdk.roles import Role, activate_role
from base.challenge_sdk.schemas import RawWeightPushRequest

from prism_challenge.db import Database
from prism_challenge.raw_weight_push import (
    RawWeightPushClient,
    RawWeightPushStore,
    maybe_build_push_client_from_settings,
)

HOTKEY = "5CkeyABC"
TOKEN = "prism-shared-token"
SLUG = "prism"


class FakeClock:
    def __init__(self) -> None:
        self._now = datetime.now(UTC).replace(microsecond=0)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class TransportQueue:
    """httpx MockTransport with sequential scripted responses."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            return httpx.Response(500, json={"detail": "exhausted"})
        return self.responses.pop(0)


@pytest.fixture
async def database(tmp_path: Path) -> Database:
    db = Database(tmp_path / "prism.sqlite3")
    await db.init()
    return db


@pytest.mark.asyncio
async def test_schema_compatible_payload_accepted_by_base(database: Database) -> None:
    clock = FakeClock()
    store = RawWeightPushStore(database, challenge_slug=SLUG)
    await store.init()
    client = RawWeightPushClient(
        database=database,
        challenge_slug=SLUG,
        master_base_url="http://master.test",
        shared_token=TOKEN,
        now_fn=clock.now,
    )
    payload, raw = client._build_payload(
        weights={HOTKEY: 1.0},
        epoch=7,
        revision=1,
        nonce="n-compat",
        now=clock.now(),
    )
    # Round-trip through Base schema without Prism-specific parsing.
    again = RawWeightPushRequest.model_validate_json(raw)
    assert again.payload_digest == payload.payload_digest
    assert again.weights == {HOTKEY: 1.0}
    assert "uids" not in json.loads(raw)


@pytest.mark.asyncio
async def test_cursor_advances_only_on_exact_ack(database: Database) -> None:
    clock = FakeClock()
    transport = TransportQueue(
        [
            httpx.Response(503, json={"detail": "unavailable"}),
            httpx.Response(
                200,
                json={
                    "protocol_version": "1.0",
                    "challenge_slug": SLUG,
                    "epoch": 10,
                    "revision": 1,
                    "snapshot_id": "snap-wrong",
                    "payload_digest": "0" * 64,
                    "accepted": True,
                    "idempotent": False,
                },
            ),
        ]
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(transport.handler),
        base_url="http://master.test",
    )
    client = RawWeightPushClient(
        database=database,
        challenge_slug=SLUG,
        master_base_url="http://master.test",
        shared_token=TOKEN,
        now_fn=clock.now,
        http_client=http,
        epoch_fn=lambda: 10,
    )
    await client.init()
    with activate_role(Role.CHALLENGE):
        first = await client.push_once(weights={HOTKEY: 1.0}, epoch=10)
    assert first.cursor_advanced is False
    assert first.status in {"server_error", "rejected"}
    assert await client.store.get_cursor() is None
    pending = await client.store.get_pending()
    assert pending is not None
    # Mismatched digest ack must not advance.
    with activate_role(Role.CHALLENGE):
        second = await client.push_once(reuse_pending=True)
    assert second.cursor_advanced is False
    assert second.status == "ack_mismatch"
    assert await client.store.get_cursor() is None
    await http.aclose()


@pytest.mark.asyncio
async def test_valid_ack_advances_cursor_and_restart_reuses_pending(
    database: Database,
) -> None:
    clock = FakeClock()
    payload_holder: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        parsed = RawWeightPushRequest.model_validate_json(body)
        payload_holder["digest"] = parsed.payload_digest
        payload_holder["epoch"] = parsed.epoch
        payload_holder["revision"] = parsed.revision
        if payload_holder.get("fail_once"):
            payload_holder["fail_once"] = False
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": SLUG,
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": "snap-1",
                "payload_digest": parsed.payload_digest,
                "accepted": True,
                "idempotent": False,
            },
        )

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://master.test",
    )
    client = RawWeightPushClient(
        database=database,
        challenge_slug=SLUG,
        master_base_url="http://master.test",
        shared_token=TOKEN,
        now_fn=clock.now,
        http_client=http,
        epoch_fn=lambda: 11,
    )
    await client.init()
    payload_holder["fail_once"] = True
    with activate_role(Role.CHALLENGE):
        failed = await client.push_once(weights={HOTKEY: 0.5}, epoch=11)
    assert failed.cursor_advanced is False
    pending_before = await client.store.get_pending()
    assert pending_before is not None
    digest = pending_before["payload_digest"]

    # Simulate restart with a fresh client sharing the same database.
    client2 = RawWeightPushClient(
        database=database,
        challenge_slug=SLUG,
        master_base_url="http://master.test",
        shared_token=TOKEN,
        now_fn=clock.now,
        http_client=http,
        epoch_fn=lambda: 11,
    )
    await client2.init()
    with activate_role(Role.CHALLENGE):
        ok = await client2.push_once(reuse_pending=True)
    assert ok.cursor_advanced is True
    assert ok.status == "acknowledged"
    assert ok.payload_digest == digest
    cursor = await client2.store.get_cursor()
    assert cursor is not None
    assert cursor.payload_digest == digest
    assert cursor.snapshot_id == "snap-1"
    assert await client2.store.get_pending() is None
    await http.aclose()


@pytest.mark.asyncio
async def test_wrong_role_cannot_push(database: Database) -> None:
    from base.challenge_sdk.roles import RoleContractError

    client = RawWeightPushClient(
        database=database,
        challenge_slug=SLUG,
        master_base_url="http://master.test",
        shared_token=TOKEN,
    )
    await client.init()
    with activate_role(Role.MASTER):
        with pytest.raises(RoleContractError):
            await client.push_once(weights={HOTKEY: 1.0}, epoch=1)


def test_maybe_build_push_client_requires_master_and_token(database: Database) -> None:
    class _Settings:
        raw_weight_push_enabled = True
        master_base_url = None
        worker_plane = type("WP", (), {"master_base_url": None})()
        slug = SLUG
        epoch_seconds = 3600
        architecture_reward_weight = 0.5
        training_reward_weight = 0.5
        raw_weight_push_interval_seconds = 5.0

        def internal_token(self) -> str:
            return TOKEN

    assert (
        maybe_build_push_client_from_settings(
            settings=_Settings(), database=database, repository=object()
        )
        is None
    )

    class _Enabled(_Settings):
        master_base_url = "http://master.test"

    client = maybe_build_push_client_from_settings(
        settings=_Enabled(), database=database, repository=object()
    )
    assert client is not None
    assert client.master_base_url == "http://master.test"
