"""Behavioral tests for the hotkey-signed assignment coordination endpoints.

Covers VAL-ASSIGN-010..021: signed-request + permit gating on the pull/progress/
result routes, capability-filtered caller-only pull (assigned/running),
pull assigned->running + lease deadline, progress lease refresh + prism
checkpoint ref, ownership/state rejection, result transition + persistence,
idempotent re-post, and non-owner result rejection.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    Validator,
    ValidatorStatus,
    WorkResult,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment, WorkAssignmentStatus
from base.master.app_proxy import create_proxy_app
from base.master.assignment_coordination import AssignmentCoordinationService
from base.security.validator_auth import (
    MetagraphValidatorEligibility,
    SqlAlchemyValidatorNonceStore,
    ValidatorSignedRequestVerifier,
    canonical_validator_request,
)

NOW_EPOCH = 1_750_000_000.0
LEASE_SECONDS = 900


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class FakeNonceStore:
    async def reserve(self, **_: Any) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


class FakeClock:
    def __init__(self, epoch: float) -> None:
        self.epoch = float(epoch)

    def time(self) -> float:
        return self.epoch

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, UTC)


def _sign(hotkey: str, canonical: str) -> str:
    return hashlib.sha256(f"{hotkey}:{canonical}".encode()).hexdigest()


def _signature_verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message.decode())


class Harness:
    def __init__(
        self,
        client: AsyncClient,
        session_factory: Any,
        clock: FakeClock,
    ) -> None:
        self.client = client
        self.session_factory = session_factory
        self.clock = clock
        self._nonce = 0

    def _next_nonce(self) -> str:
        self._nonce += 1
        return f"nonce-{self._nonce}"

    async def signed(
        self,
        *,
        path: str,
        body: bytes = b"",
        hotkey: str = "permitted",
        nonce: str | None = None,
        timestamp: float | None = None,
        sign: bool = True,
        include_headers: bool = True,
    ):
        ts = str(int(self.clock.time() if timestamp is None else timestamp))
        nonce_val = self._next_nonce() if nonce is None else nonce
        headers = {"Content-Type": "application/json"}
        if include_headers:
            canonical = canonical_validator_request(
                method="POST",
                path=path,
                query_string="",
                timestamp=ts,
                nonce=nonce_val,
                body=body,
            )
            signature = _sign(hotkey, canonical) if sign else "deadbeef-not-a-signature"
            headers.update(
                {
                    "X-Hotkey": hotkey,
                    "X-Signature": signature,
                    "X-Nonce": nonce_val,
                    "X-Timestamp": ts,
                }
            )
        return await self.client.post(path, content=body, headers=headers)

    async def pull(self, *, hotkey: str = "permitted", **kwargs: Any):
        return await self.signed(path="/v1/assignments/pull", hotkey=hotkey, **kwargs)

    async def progress(
        self,
        assignment_id: str,
        *,
        hotkey: str = "permitted",
        checkpoint_ref: str | None = None,
    ):
        payload: dict[str, Any] = {}
        if checkpoint_ref is not None:
            payload["checkpoint_ref"] = checkpoint_ref
        body = json.dumps(payload).encode()
        return await self.signed(
            path=f"/v1/assignments/{assignment_id}/progress",
            body=body,
            hotkey=hotkey,
        )

    async def result(
        self,
        assignment_id: str,
        *,
        success: bool,
        hotkey: str = "permitted",
        payload: dict[str, Any] | None = None,
    ):
        body = json.dumps({"success": success, "payload": payload or {}}).encode()
        return await self.signed(
            path=f"/v1/assignments/{assignment_id}/result",
            body=body,
            hotkey=hotkey,
        )

    async def add_validator(
        self,
        hotkey: str,
        capabilities: list[str],
        *,
        status: ValidatorStatus = ValidatorStatus.ONLINE,
    ) -> None:
        async with session_scope(self.session_factory) as session:
            session.add(
                Validator(
                    hotkey=hotkey,
                    uid=None,
                    status=status,
                    capabilities=list(capabilities),
                    version="1.0.0",
                    registered_at=self.clock.now(),
                    last_heartbeat_at=self.clock.now(),
                )
            )

    async def add_assignment(
        self,
        *,
        work_unit_id: str,
        hotkey: str | None,
        status: WorkAssignmentStatus,
        challenge_slug: str = "agent-challenge",
        required_capability: str = "cpu",
        checkpoint_ref: str | None = None,
        deadline_at: datetime | None = None,
        result_ref: str | None = None,
    ) -> str:
        unit_id = uuid.uuid4()
        async with session_scope(self.session_factory) as session:
            session.add(
                WorkAssignment(
                    id=unit_id,
                    challenge_slug=challenge_slug,
                    work_unit_id=work_unit_id,
                    submission_ref="hk-sub",
                    payload={"task_id": work_unit_id},
                    required_capability=required_capability,
                    assigned_validator_hotkey=hotkey,
                    status=status,
                    attempt_count=1,
                    max_attempts=3,
                    deadline_at=deadline_at,
                    checkpoint_ref=checkpoint_ref,
                    result_ref=result_ref,
                    created_at=self.clock.now(),
                    updated_at=self.clock.now(),
                )
            )
        return str(unit_id)

    async def get_assignment(self, assignment_id: str) -> WorkAssignment | None:
        async with self.session_factory() as session:
            return (
                await session.execute(
                    select(WorkAssignment).where(
                        WorkAssignment.id == uuid.UUID(assignment_id)
                    )
                )
            ).scalar_one_or_none()

    async def count_results(self, assignment_id: str) -> int:
        async with self.session_factory() as session:
            return await session.scalar(
                select(func.count(WorkResult.id)).where(
                    WorkResult.assignment_id == uuid.UUID(assignment_id)
                )
            )


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        ["permitted", "permitted2"],
        validator_permits=[True, True],
        stakes=[100.0, 50.0],
    )
    clock = FakeClock(NOW_EPOCH)
    verifier = ValidatorSignedRequestVerifier(
        nonce_store=SqlAlchemyValidatorNonceStore(session_factory),
        eligibility=MetagraphValidatorEligibility(cache),
        signature_verifier=_signature_verifier,
        ttl_seconds=300,
        now_fn=clock.time,
    )
    service = AssignmentCoordinationService(
        session_factory,
        lease_seconds=LEASE_SECONDS,
        now_fn=clock.now,
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        validator_verifier=verifier,
        assignment_coordination_service=service,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    try:
        yield Harness(client, session_factory, clock)
    finally:
        await client.aclose()
        await engine.dispose()


# ---------------------------------------------------------------------------
# Auth + eligibility (VAL-ASSIGN-010/011/012)
# ---------------------------------------------------------------------------


# VAL-ASSIGN-010
async def test_unsigned_and_bad_signature_rejected_valid_accepted(
    harness: Harness,
) -> None:
    await harness.add_validator("permitted", ["cpu"])

    missing = await harness.pull(include_headers=False)
    assert missing.status_code == 401

    forged = await harness.pull(sign=False)
    assert forged.status_code == 401

    ok = await harness.pull()
    assert ok.status_code == 200


# VAL-ASSIGN-011
async def test_stale_timestamp_and_replayed_nonce_rejected(harness: Harness) -> None:
    await harness.add_validator("permitted", ["cpu"])

    stale = await harness.pull(timestamp=NOW_EPOCH - 1_000)
    assert stale.status_code == 401

    first = await harness.pull(nonce="replay-1")
    assert first.status_code == 200
    replay = await harness.pull(nonce="replay-1")
    assert replay.status_code == 409


# VAL-ASSIGN-012
async def test_signed_but_unpermitted_hotkey_rejected(harness: Harness) -> None:
    # "stranger" is correctly signed but absent from the metagraph.
    response = await harness.pull(hotkey="stranger")
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Pull (VAL-ASSIGN-015/016)
# ---------------------------------------------------------------------------


# VAL-ASSIGN-015
async def test_pull_returns_only_callers_non_terminal_capability_matched(
    harness: Harness,
) -> None:
    await harness.add_validator("permitted", ["cpu"])
    await harness.add_validator("permitted2", ["cpu"])

    mine_assigned = await harness.add_assignment(
        work_unit_id="u-assigned",
        hotkey="permitted",
        status=WorkAssignmentStatus.ASSIGNED,
    )
    mine_running = await harness.add_assignment(
        work_unit_id="u-running",
        hotkey="permitted",
        status=WorkAssignmentStatus.RUNNING,
    )
    await harness.add_assignment(
        work_unit_id="u-done",
        hotkey="permitted",
        status=WorkAssignmentStatus.COMPLETED,
    )
    await harness.add_assignment(
        work_unit_id="u-other",
        hotkey="permitted2",
        status=WorkAssignmentStatus.ASSIGNED,
    )
    # A gpu unit owned by a cpu-only caller must be filtered out of the pull.
    await harness.add_assignment(
        work_unit_id="u-gpu",
        hotkey="permitted",
        status=WorkAssignmentStatus.ASSIGNED,
        required_capability="gpu",
        challenge_slug="prism",
    )

    response = await harness.pull(hotkey="permitted")
    assert response.status_code == 200
    returned = {a["id"] for a in response.json()["assignments"]}
    assert returned == {mine_assigned, mine_running}


# VAL-ASSIGN-016
async def test_pull_transitions_assigned_to_running_with_future_deadline(
    harness: Harness,
) -> None:
    await harness.add_validator("permitted", ["cpu"])
    assignment_id = await harness.add_assignment(
        work_unit_id="u-1", hotkey="permitted", status=WorkAssignmentStatus.ASSIGNED
    )

    before = await harness.get_assignment(assignment_id)
    assert before is not None
    assert before.status == WorkAssignmentStatus.ASSIGNED
    assert before.deadline_at is None

    response = await harness.pull(hotkey="permitted")
    assert response.status_code == 200
    view = response.json()["assignments"][0]
    assert view["status"] == "running"

    after = await harness.get_assignment(assignment_id)
    assert after is not None
    assert after.status == WorkAssignmentStatus.RUNNING
    assert after.deadline_at is not None
    assert _as_utc(after.deadline_at) > harness.clock.now()
    assert after.last_progress_at is not None


# ---------------------------------------------------------------------------
# Progress (VAL-ASSIGN-017/018)
# ---------------------------------------------------------------------------


# VAL-ASSIGN-017
async def test_progress_refreshes_deadline_and_stores_checkpoint(
    harness: Harness,
) -> None:
    await harness.add_validator("permitted", ["cpu", "gpu"])
    assignment_id = await harness.add_assignment(
        work_unit_id="psub-1",
        hotkey="permitted",
        status=WorkAssignmentStatus.RUNNING,
        required_capability="gpu",
        challenge_slug="prism",
        deadline_at=harness.clock.now(),
    )

    harness.clock.epoch = NOW_EPOCH + 120
    response = await harness.progress(assignment_id, checkpoint_ref="hf://ckpt/42")
    assert response.status_code == 200
    body = response.json()
    assert body["checkpoint_ref"] == "hf://ckpt/42"

    row = await harness.get_assignment(assignment_id)
    assert row is not None
    assert row.checkpoint_ref == "hf://ckpt/42"
    assert row.last_progress_at is not None
    assert _as_utc(row.last_progress_at) == harness.clock.now()
    assert row.deadline_at is not None
    assert _as_utc(row.deadline_at) > harness.clock.now()
    assert _as_utc(row.deadline_at) > datetime.fromtimestamp(NOW_EPOCH, UTC)


# VAL-ASSIGN-018
async def test_progress_rejected_for_non_owner_and_non_running(
    harness: Harness,
) -> None:
    await harness.add_validator("permitted", ["cpu"])
    await harness.add_validator("permitted2", ["cpu"])

    foreign = await harness.add_assignment(
        work_unit_id="u-foreign",
        hotkey="permitted2",
        status=WorkAssignmentStatus.RUNNING,
        deadline_at=harness.clock.now(),
    )
    not_running = await harness.add_assignment(
        work_unit_id="u-assigned",
        hotkey="permitted",
        status=WorkAssignmentStatus.ASSIGNED,
    )

    foreign_resp = await harness.progress(foreign, hotkey="permitted")
    assert foreign_resp.status_code == 403
    foreign_row = await harness.get_assignment(foreign)
    assert foreign_row is not None
    assert foreign_row.last_progress_at is None

    state_resp = await harness.progress(not_running, hotkey="permitted")
    assert state_resp.status_code == 409
    state_row = await harness.get_assignment(not_running)
    assert state_row is not None
    assert state_row.deadline_at is None
    assert state_row.last_progress_at is None


# ---------------------------------------------------------------------------
# Result (VAL-ASSIGN-019/020/021)
# ---------------------------------------------------------------------------


# VAL-ASSIGN-019
async def test_result_success_completes_and_persists(harness: Harness) -> None:
    await harness.add_validator("permitted", ["cpu"])
    assignment_id = await harness.add_assignment(
        work_unit_id="u-1", hotkey="permitted", status=WorkAssignmentStatus.RUNNING
    )

    response = await harness.result(assignment_id, success=True, payload={"score": 0.9})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["result_ref"]
    assert body["idempotent"] is False

    row = await harness.get_assignment(assignment_id)
    assert row is not None
    assert row.status == WorkAssignmentStatus.COMPLETED
    assert row.result_ref == body["result_ref"]
    assert await harness.count_results(assignment_id) == 1


async def test_result_failure_marks_failed(harness: Harness) -> None:
    await harness.add_validator("permitted", ["cpu"])
    assignment_id = await harness.add_assignment(
        work_unit_id="u-2", hotkey="permitted", status=WorkAssignmentStatus.RUNNING
    )

    response = await harness.result(assignment_id, success=False)
    assert response.status_code == 200
    assert response.json()["status"] == "failed"

    row = await harness.get_assignment(assignment_id)
    assert row is not None
    assert row.status == WorkAssignmentStatus.FAILED
    assert await harness.count_results(assignment_id) == 1


# VAL-ASSIGN-020
async def test_result_post_is_idempotent(harness: Harness) -> None:
    await harness.add_validator("permitted", ["cpu"])
    assignment_id = await harness.add_assignment(
        work_unit_id="u-3", hotkey="permitted", status=WorkAssignmentStatus.RUNNING
    )

    first = await harness.result(assignment_id, success=True, payload={"score": 0.5})
    assert first.status_code == 200
    first_ref = first.json()["result_ref"]

    second = await harness.result(assignment_id, success=True, payload={"score": 0.99})
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["idempotent"] is True
    assert second_body["result_ref"] == first_ref
    assert second_body["status"] == "completed"

    row = await harness.get_assignment(assignment_id)
    assert row is not None
    assert row.status == WorkAssignmentStatus.COMPLETED
    assert row.result_ref == first_ref
    # No duplicate result row created by the repeated post.
    assert await harness.count_results(assignment_id) == 1


# VAL-ASSIGN-021
async def test_result_post_by_non_owner_is_rejected(harness: Harness) -> None:
    await harness.add_validator("permitted", ["cpu"])
    await harness.add_validator("permitted2", ["cpu"])
    assignment_id = await harness.add_assignment(
        work_unit_id="u-4",
        hotkey="permitted2",
        status=WorkAssignmentStatus.RUNNING,
    )

    response = await harness.result(assignment_id, success=True, hotkey="permitted")
    assert response.status_code == 403

    row = await harness.get_assignment(assignment_id)
    assert row is not None
    assert row.status == WorkAssignmentStatus.RUNNING
    assert row.result_ref is None
    assert await harness.count_results(assignment_id) == 0


async def test_result_unknown_assignment_returns_404(harness: Harness) -> None:
    await harness.add_validator("permitted", ["cpu"])
    response = await harness.result(str(uuid.uuid4()), success=True)
    assert response.status_code == 404
