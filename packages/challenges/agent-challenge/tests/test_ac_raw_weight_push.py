"""Agent-challenge raw-weight push client and durable ack cursor tests.

Mirrors Prism's push contract (VAL-SDK-017 / VAL-WEIGHT-028..030) so master
accepts agent-challenge payloads identically. WTA maps must surface as a single
winner hotkey on the wire.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from base.challenge_sdk.roles import Role, activate_role
from base.challenge_sdk.schemas import RawWeightPushRequest

from agent_challenge.evaluation.raw_weight_push import RawWeightPushClient
from agent_challenge.sdk.db import Database

WINNER = "5CwinnerHK1"
LOSER = "5CloserHK22"
TOKEN = "ac-shared-token-secret"
SLUG = "agent-challenge"


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
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ac-push.sqlite3'}")
    await db.init()
    return db


@pytest.mark.asyncio
async def test_push_payload_is_single_winner_for_wta(database: Database) -> None:
    """WTA map with one hotkey produces a payload with exactly that hotkey."""

    clock = FakeClock()
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        parsed = RawWeightPushRequest.model_validate_json(request.content)
        captured["weights"] = dict(parsed.weights)
        captured["slug"] = parsed.challenge_slug
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": SLUG,
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": "snap-wta",
                "payload_digest": parsed.payload_digest,
                "accepted": True,
                "idempotent": False,
            },
        )

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://master.test",
    )
    # WTA output shape from get_weights: single winner only.
    wta_weights = {WINNER: 0.91}
    client = RawWeightPushClient(
        database=database,
        challenge_slug=SLUG,
        master_base_url="http://master.test",
        shared_token=TOKEN,
        now_fn=clock.now,
        http_client=http,
        epoch_fn=lambda: 42,
        weights_fn=lambda: wta_weights,
    )
    await client.init()
    with activate_role(Role.CHALLENGE):
        result = await client.push_once()
    assert result.cursor_advanced is True
    assert result.status == "acknowledged"
    assert captured["slug"] == SLUG
    assert set(captured["weights"]) == {WINNER}
    assert captured["weights"][WINNER] == pytest.approx(0.91)
    assert LOSER not in captured["weights"]
    await http.aclose()


@pytest.mark.asyncio
async def test_empty_weights_does_not_advance_cursor(database: Database) -> None:
    clock = FakeClock()
    transport = TransportQueue([])
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
        epoch_fn=lambda: 1,
        weights_fn=lambda: {},
    )
    await client.init()
    with activate_role(Role.CHALLENGE):
        result = await client.push_once()
    assert result.cursor_advanced is False
    assert result.status == "skipped_empty"
    assert await client.store.get_cursor() is None
    assert transport.requests == []
    await http.aclose()


@pytest.mark.asyncio
async def test_successful_ack_advances_cursor(database: Database) -> None:
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        parsed = RawWeightPushRequest.model_validate_json(request.content)
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": SLUG,
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": "snap-ok",
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
    with activate_role(Role.CHALLENGE):
        ok = await client.push_once(weights={WINNER: 0.5}, epoch=11)
    assert ok.cursor_advanced is True
    assert ok.status == "acknowledged"
    cursor = await client.store.get_cursor()
    assert cursor is not None
    assert cursor.payload_digest == ok.payload_digest
    assert cursor.snapshot_id == "snap-ok"
    assert cursor.epoch == 11
    assert await client.store.get_pending() is None
    await http.aclose()


@pytest.mark.asyncio
async def test_mismatched_ack_digest_does_not_advance_cursor(database: Database) -> None:
    clock = FakeClock()
    transport = TransportQueue(
        [
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
        result = await client.push_once(weights={WINNER: 1.0}, epoch=10)
    assert result.cursor_advanced is False
    assert result.status == "ack_mismatch"
    assert await client.store.get_cursor() is None
    pending = await client.store.get_pending()
    assert pending is not None
    await http.aclose()


@pytest.mark.asyncio
async def test_push_does_not_log_token(
    database: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        parsed = RawWeightPushRequest.model_validate_json(request.content)
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": SLUG,
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": "snap-log",
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
        epoch_fn=lambda: 3,
    )
    await client.init()
    with (
        caplog.at_level(logging.DEBUG, logger="agent_challenge.evaluation.raw_weight_push"),
        activate_role(Role.CHALLENGE),
    ):
        await client.push_once(weights={WINNER: 1.0}, epoch=3)
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert TOKEN not in joined
    for record in caplog.records:
        assert TOKEN not in str(record.__dict__)
    await http.aclose()


@pytest.mark.asyncio
async def test_expired_pending_is_rebuilt_not_retried(database: Database) -> None:
    clock = FakeClock()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        parsed = RawWeightPushRequest.model_validate_json(request.content)
        seen.append(parsed.nonce)
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": SLUG,
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": "snap-ac-rebuilt",
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
        epoch_fn=lambda: 99,
        freshness_seconds=60,
    )
    await client.init()
    old = clock.now() - timedelta(hours=12)
    stale_payload, stale_raw = client._build_payload(
        weights={WINNER: 1.0},
        epoch=1,
        revision=1,
        nonce="ac-stale-nonce",
        now=old,
    )
    await client.store.record_pending(
        epoch=stale_payload.epoch,
        revision=stale_payload.revision,
        payload_digest=stale_payload.payload_digest,
        canonical_payload=stale_raw.decode("utf-8"),
        nonce=stale_payload.nonce,
        attempted_at=old.isoformat(),
    )
    with activate_role(Role.CHALLENGE):
        result = await client.push_once(weights={WINNER: 1.0}, epoch=99, reuse_pending=True)
    assert result.status == "acknowledged"
    assert len(seen) == 1
    assert seen[0] != "ac-stale-nonce"
    assert await client.store.get_pending() is None
    await http.aclose()


@pytest.mark.asyncio
async def test_all_zero_weights_skipped_locally(database: Database) -> None:
    clock = FakeClock()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

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
        epoch_fn=lambda: 1,
    )
    await client.init()
    with activate_role(Role.CHALLENGE):
        result = await client.push_once(
            weights={"5GziQCcRpN8NCJktX343brnfuVe3w6gUYieeStXPD1Dag2At": 0.0},
            epoch=1,
        )
    assert result.cursor_advanced is False
    assert result.status in {"skipped_empty", "skipped_zero"}
    assert calls == []
    await http.aclose()


@pytest.mark.asyncio
async def test_successful_push_writes_ack_cursor_fields(database: Database) -> None:
    """ACK path must set last_epoch, last_snapshot_id, and acknowledged_at."""

    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "target_epoch": 100,
                    "highest_sealed_epoch": 99,
                    "max_future_epoch_ahead": 2,
                },
            )
        parsed = RawWeightPushRequest.model_validate_json(request.content)
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": SLUG,
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": "snap-ack-fields",
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
        # No epoch_fn: must use master target-epoch.
        epoch_fn=None,
    )
    await client.init()
    with activate_role(Role.CHALLENGE):
        result = await client.push_once(weights={WINNER: 1.0})
    assert result.status == "acknowledged"
    assert result.cursor_advanced is True
    assert result.epoch == 100
    assert result.snapshot_id == "snap-ack-fields"
    cursor = await client.store.get_cursor()
    assert cursor is not None
    assert cursor.epoch == 100
    assert cursor.snapshot_id == "snap-ack-fields"
    assert cursor.acknowledged_at is not None
    assert cursor.acknowledged_at != ""
    await http.aclose()


@pytest.mark.asyncio
async def test_sealed_epoch_409_advances_to_master_target(database: Database) -> None:
    """Sealed-epoch 409 must not infinite-retry the same epoch; advance via master."""

    clock = FakeClock()
    post_epochs: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            # Master says next unsealed is 50 after first sealed rejection path.
            return httpx.Response(
                200,
                json={
                    "target_epoch": 50,
                    "highest_sealed_epoch": 49,
                    "max_future_epoch_ahead": 2,
                },
            )
        parsed = RawWeightPushRequest.model_validate_json(request.content)
        post_epochs.append(parsed.epoch)
        if parsed.epoch == 49:
            return httpx.Response(
                409, json={"detail": "epoch is sealed; revision rejected"}
            )
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": SLUG,
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": f"snap-{parsed.epoch}",
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
        # Stale local clock would stick on sealed 49 forever without master.
        epoch_fn=lambda: 49,
    )
    await client.init()
    with activate_role(Role.CHALLENGE):
        result = await client.push_once(weights={WINNER: 1.0})
    assert result.status == "acknowledged"
    assert result.epoch == 50
    assert 49 in post_epochs or post_epochs == [50]
    # Must not hammer the same sealed epoch repeatedly in one push_once.
    assert post_epochs.count(49) <= 1
    assert await client.store.get_cursor() is not None
    assert (await client.store.get_cursor()).epoch == 50
    await http.aclose()


@pytest.mark.asyncio
async def test_conflict_409_does_not_retry_same_epoch_revision(
    database: Database,
) -> None:
    """Conflicting payload at (epoch, rev) must bump revision, not loop forever."""

    clock = FakeClock()
    seen: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "target_epoch": 7,
                    "highest_sealed_epoch": 6,
                    "max_future_epoch_ahead": 2,
                },
            )
        parsed = RawWeightPushRequest.model_validate_json(request.content)
        seen.append((parsed.epoch, parsed.revision))
        if parsed.revision == 1:
            return httpx.Response(
                409, json={"detail": "conflicting raw weight payload"}
            )
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": SLUG,
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": "snap-rev2",
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
        epoch_fn=None,
    )
    await client.init()
    with activate_role(Role.CHALLENGE):
        result = await client.push_once(weights={WINNER: 1.0})
    assert result.status == "acknowledged"
    assert result.revision == 2
    assert (7, 1) in seen
    assert (7, 2) in seen
    assert seen.count((7, 1)) == 1
    cursor = await client.store.get_cursor()
    assert cursor is not None
    assert cursor.epoch == 7
    assert cursor.revision == 2
    assert cursor.acknowledged_at is not None
    await http.aclose()


@pytest.mark.asyncio
async def test_epoch_source_prefers_master_over_local_clock(database: Database) -> None:
    """When master target-epoch is available, do not use local wall-clock epoch_fn."""

    clock = FakeClock()
    posted: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "target_epoch": 9001,
                    "highest_sealed_epoch": 9000,
                    "max_future_epoch_ahead": 2,
                },
            )
        parsed = RawWeightPushRequest.model_validate_json(request.content)
        posted.append(parsed.epoch)
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": SLUG,
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": "snap-master-epoch",
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
        epoch_fn=lambda: 1,  # would be wrong if preferred
    )
    await client.init()
    with activate_role(Role.CHALLENGE):
        result = await client.push_once(weights={WINNER: 1.0})
    assert result.status == "acknowledged"
    assert posted == [9001]
    await http.aclose()
