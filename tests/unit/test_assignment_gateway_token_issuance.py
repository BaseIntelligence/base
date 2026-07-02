"""Production issuance of the per-assignment scoped gateway token (VAL-LLM-024).

When a validator pulls an active assignment from the real pull route, the master
server-side issues a fresh scoped gateway token (scope = validator hotkey +
assignment id, source=agent, expiry bounded by the lease deadline) via the
gateway token authority and includes it, plus the master gateway base URL
(``BASE_LLM_GATEWAY_URL={root}/llm/v1``), in the returned assignment payload. The
delivered token authorizes a gateway call for that assignment and is rejected for
a different scope or once the assignment is terminal. No raw provider key is ever
placed in the payload.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    Validator,
    ValidatorStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment, WorkAssignmentStatus
from base.master.app_proxy import create_proxy_app
from base.master.assignment_coordination import (
    GATEWAY_BASE_URL_PAYLOAD_KEY,
    GATEWAY_TOKEN_PAYLOAD_KEY,
    AssignmentCoordinationService,
    GatewayPayloadIssuer,
    WorkAssignmentLifecycleResolver,
)
from base.master.llm_gateway import (
    BASE_LLM_GATEWAY_URL_ENV,
    GATEWAY_ASSIGNMENT_HEADER,
    GATEWAY_TOKEN_HEADER,
    GATEWAY_VALIDATOR_HEADER,
    ProviderConfig,
    SourceRoute,
    build_llm_gateway_service,
)
from base.security.validator_auth import (
    MetagraphValidatorEligibility,
    SqlAlchemyValidatorNonceStore,
    ValidatorSignedRequestVerifier,
    canonical_validator_request,
)

NOW_EPOCH = 1_750_000_000.0
LEASE_SECONDS = 900
GATEWAY_BASE_URL = "http://testserver"
YUNWU_KEY = "yunwu-provider-secret-key"
GATEWAY_TOKEN_SECRET = "gateway-hmac-secret"
MODEL = "claude-opus-4-8"


class FakeNonceStore:
    async def reserve(self, **_: Any) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


class FakeClock:
    def __init__(self, epoch: float) -> None:
        self.epoch = float(epoch)

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
        gateway_service: Any,
        clock: FakeClock,
    ) -> None:
        self.client = client
        self.session_factory = session_factory
        self.gateway_service = gateway_service
        self.clock = clock
        self._nonce = 0

    def _next_nonce(self) -> str:
        self._nonce += 1
        return f"nonce-{self._nonce}"

    async def signed_post(
        self, *, path: str, body: bytes = b"", hotkey: str = "permitted"
    ):
        ts = str(int(self.clock.epoch))
        nonce = self._next_nonce()
        canonical = canonical_validator_request(
            method="POST",
            path=path,
            query_string="",
            timestamp=ts,
            nonce=nonce,
            body=body,
        )
        headers = {
            "Content-Type": "application/json",
            "X-Hotkey": hotkey,
            "X-Signature": _sign(hotkey, canonical),
            "X-Nonce": nonce,
            "X-Timestamp": ts,
        }
        return await self.client.post(path, content=body, headers=headers)

    async def pull(self, *, hotkey: str = "permitted"):
        return await self.signed_post(path="/v1/assignments/pull", hotkey=hotkey)

    async def result(
        self, assignment_id: str, *, success: bool, hotkey: str = "permitted"
    ):
        body = json.dumps({"success": success, "payload": {}}).encode()
        return await self.signed_post(
            path=f"/v1/assignments/{assignment_id}/result",
            body=body,
            hotkey=hotkey,
        )

    async def gateway_call(
        self,
        *,
        token: str | None,
        model: str = "agent-sent-placeholder",
        validator: str | None = None,
        assignment: str | None = None,
    ):
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token is not None:
            headers[GATEWAY_TOKEN_HEADER] = token
        if validator is not None:
            headers[GATEWAY_VALIDATOR_HEADER] = validator
        if assignment is not None:
            headers[GATEWAY_ASSIGNMENT_HEADER] = assignment
        body = json.dumps({"model": model, "messages": []}).encode()
        return await self.client.post(
            "/llm/v1/chat/completions", content=body, headers=headers
        )

    async def add_validator(self, hotkey: str, capabilities: list[str]) -> None:
        async with session_scope(self.session_factory) as session:
            session.add(
                Validator(
                    hotkey=hotkey,
                    uid=None,
                    status=ValidatorStatus.ONLINE,
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
        hotkey: str,
        status: WorkAssignmentStatus = WorkAssignmentStatus.ASSIGNED,
    ) -> str:
        unit_id = uuid.uuid4()
        async with session_scope(self.session_factory) as session:
            session.add(
                WorkAssignment(
                    id=unit_id,
                    challenge_slug="agent-challenge",
                    work_unit_id=work_unit_id,
                    submission_ref="hk-sub",
                    payload={"task_id": work_unit_id},
                    required_capability="cpu",
                    assigned_validator_hotkey=hotkey,
                    status=status,
                    attempt_count=1,
                    max_attempts=3,
                    created_at=self.clock.now(),
                    updated_at=self.clock.now(),
                )
            )
        return str(unit_id)

    async def assignment_payload(self, assignment_id: str) -> dict[str, Any]:
        async with self.session_factory() as session:
            row = await session.get(WorkAssignment, uuid.UUID(assignment_id))
            assert row is not None
            return dict(row.payload or {})


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(["permitted"], validator_permits=[True], stakes=[100.0])
    clock = FakeClock(NOW_EPOCH)
    verifier = ValidatorSignedRequestVerifier(
        nonce_store=SqlAlchemyValidatorNonceStore(session_factory),
        eligibility=MetagraphValidatorEligibility(cache),
        signature_verifier=_signature_verifier,
        ttl_seconds=300,
        now_fn=lambda: clock.epoch,
    )
    gateway_service = build_llm_gateway_service(
        api_keys={"yunwu": YUNWU_KEY},
        token_secret=GATEWAY_TOKEN_SECRET,
        provider_config=ProviderConfig(mode="mock"),
        sources={"agent": SourceRoute(provider="yunwu", model=MODEL)},
        assignment_resolver=WorkAssignmentLifecycleResolver(session_factory),
    )
    service = AssignmentCoordinationService(
        session_factory,
        lease_seconds=LEASE_SECONDS,
        gateway_payload_issuer=GatewayPayloadIssuer(
            issuer=gateway_service,
            gateway_base_url=GATEWAY_BASE_URL,
        ),
        now_fn=clock.now,
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        validator_verifier=verifier,
        assignment_coordination_service=service,
        llm_gateway_service=gateway_service,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url=GATEWAY_BASE_URL)
    try:
        yield Harness(client, session_factory, gateway_service, clock)
    finally:
        await client.aclose()
        await engine.dispose()


# VAL-LLM-024
async def test_pull_stamps_scoped_token_and_gateway_base_urls(
    harness: Harness,
) -> None:
    await harness.add_validator("permitted", ["cpu"])
    assignment_id = await harness.add_assignment(work_unit_id="u-1", hotkey="permitted")

    response = await harness.pull()
    assert response.status_code == 200
    payload = response.json()["assignments"][0]["payload"]

    token = payload[GATEWAY_TOKEN_PAYLOAD_KEY]
    assert isinstance(token, str) and token
    assert payload[BASE_LLM_GATEWAY_URL_ENV] == f"{GATEWAY_BASE_URL}/llm/v1"
    assert payload[GATEWAY_BASE_URL_PAYLOAD_KEY] == GATEWAY_BASE_URL
    # The old provider-path base-url env keys are no longer stamped.
    assert "DEEPSEEK_BASE_URL" not in payload
    assert "OPENROUTER_BASE_URL" not in payload

    # The token carries source=agent so the gateway resolves yunwu + the model.
    claims = harness.gateway_service.token_authority.verify(token)
    assert claims.source == "agent"

    # The payload never carries a raw provider key.
    assert YUNWU_KEY not in json.dumps(payload)

    # The token is ephemeral: it is NOT persisted into the work_assignments row.
    stored = await harness.assignment_payload(assignment_id)
    assert GATEWAY_TOKEN_PAYLOAD_KEY not in stored


# VAL-LLM-024
async def test_delivered_token_authorizes_in_scope_and_rejects_out_of_scope(
    harness: Harness,
) -> None:
    await harness.add_validator("permitted", ["cpu"])
    await harness.add_assignment(work_unit_id="u-1", hotkey="permitted")

    response = await harness.pull()
    token = response.json()["assignments"][0]["payload"][GATEWAY_TOKEN_PAYLOAD_KEY]

    # In-scope: the delivered token authorizes a gateway call -> 200, provider hit.
    ok = await harness.gateway_call(token=token)
    assert ok.status_code == 200
    yunwu = harness.gateway_service.provider("yunwu")
    assert yunwu.call_count == 1
    # The master injected the real provider key server-side (never the caller).
    assert yunwu.requests[-1].header("authorization") == f"Bearer {YUNWU_KEY}"
    # The gateway overwrote the body model with the resolved model.
    assert yunwu.requests[-1].json_body()["model"] == MODEL

    # Out-of-scope: the same token attributed to a different assignment -> 403.
    other = await harness.gateway_call(token=token, assignment=str(uuid.uuid4()))
    assert other.status_code == 403
    assert yunwu.call_count == 1

    # Out-of-scope: the same token attributed to a different validator -> 403.
    cross = await harness.gateway_call(token=token, validator="someone-else")
    assert cross.status_code == 403
    assert yunwu.call_count == 1

    # The raw provider key never appears in any response surface.
    assert YUNWU_KEY not in ok.text
    assert YUNWU_KEY not in other.text


# VAL-LLM-023 (lifecycle binding preserved end-to-end via the issued token)
async def test_token_rejected_once_assignment_terminal(harness: Harness) -> None:
    await harness.add_validator("permitted", ["cpu"])
    assignment_id = await harness.add_assignment(work_unit_id="u-1", hotkey="permitted")

    token = (await harness.pull()).json()["assignments"][0]["payload"][
        GATEWAY_TOKEN_PAYLOAD_KEY
    ]
    # While the assignment is running the token is accepted.
    assert (await harness.gateway_call(token=token)).status_code == 200
    yunwu = harness.gateway_service.provider("yunwu")
    assert yunwu.call_count == 1

    # Complete the assignment; the same token is now rejected (inactive).
    completed = await harness.result(assignment_id, success=True)
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"

    rejected = await harness.gateway_call(token=token)
    assert rejected.status_code == 403
    assert yunwu.call_count == 1


async def test_pull_without_gateway_issuer_stamps_no_token() -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)
    clock = FakeClock(NOW_EPOCH)
    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(["permitted"], validator_permits=[True], stakes=[1.0])
    verifier = ValidatorSignedRequestVerifier(
        nonce_store=SqlAlchemyValidatorNonceStore(session_factory),
        eligibility=MetagraphValidatorEligibility(cache),
        signature_verifier=_signature_verifier,
        ttl_seconds=300,
        now_fn=lambda: clock.epoch,
    )
    service = AssignmentCoordinationService(
        session_factory, lease_seconds=LEASE_SECONDS, now_fn=clock.now
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        validator_verifier=verifier,
        assignment_coordination_service=service,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url=GATEWAY_BASE_URL)
    try:
        harness = Harness(client, session_factory, None, clock)
        await harness.add_validator("permitted", ["cpu"])
        await harness.add_assignment(work_unit_id="u-1", hotkey="permitted")
        payload = (await harness.pull()).json()["assignments"][0]["payload"]
        assert GATEWAY_TOKEN_PAYLOAD_KEY not in payload
        assert BASE_LLM_GATEWAY_URL_ENV not in payload
    finally:
        await client.aclose()
        await engine.dispose()


class _RecordingIssuer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | None, str | None]] = []

    def issue_token(
        self,
        *,
        validator_hotkey: str,
        assignment_id: str,
        ttl_seconds: int | None = None,
        source: str | None = None,
        model: str | None = None,
    ) -> str:
        self.calls.append((validator_hotkey, assignment_id, ttl_seconds, source))
        return f"token-{assignment_id}-{ttl_seconds}"


def _service_with_issuer() -> tuple[AssignmentCoordinationService, _RecordingIssuer]:
    clock = FakeClock(NOW_EPOCH)
    issuer = _RecordingIssuer()
    service = AssignmentCoordinationService(
        object(),  # type: ignore[arg-type]
        lease_seconds=LEASE_SECONDS,
        gateway_payload_issuer=GatewayPayloadIssuer(
            issuer=issuer, gateway_base_url=GATEWAY_BASE_URL + "/"
        ),
        now_fn=clock.now,
    )
    return service, issuer


def _detached_unit(deadline_at: datetime | None) -> WorkAssignment:
    return WorkAssignment(id=uuid.uuid4(), deadline_at=deadline_at)


def test_token_ttl_is_bounded_by_future_deadline() -> None:
    service, issuer = _service_with_issuer()
    unit = _detached_unit(
        datetime.fromtimestamp(NOW_EPOCH, UTC) + timedelta(seconds=120)
    )

    payload = service.gateway_payload(unit, hotkey="permitted")

    assert payload is not None
    assert payload[BASE_LLM_GATEWAY_URL_ENV] == f"{GATEWAY_BASE_URL}/llm/v1"
    # The assignment token is stamped with source=agent + bounded ttl.
    assert issuer.calls == [("permitted", str(unit.id), 120, "agent")]


def test_token_ttl_handles_naive_deadline_from_db() -> None:
    # DB-returned datetimes can be naive (SQLite/Postgres); they must be treated
    # as UTC so the bound is still correct.
    service, issuer = _service_with_issuer()
    naive_deadline = datetime.fromtimestamp(NOW_EPOCH, UTC).replace(
        tzinfo=None
    ) + timedelta(seconds=90)

    service.gateway_payload(_detached_unit(naive_deadline), hotkey="permitted")

    assert issuer.calls[0][2] == 90


def test_token_ttl_falls_back_to_lease_when_no_or_past_deadline() -> None:
    service, issuer = _service_with_issuer()

    service.gateway_payload(_detached_unit(None), hotkey="permitted")
    service.gateway_payload(
        _detached_unit(datetime.fromtimestamp(NOW_EPOCH, UTC) - timedelta(seconds=30)),
        hotkey="permitted",
    )

    assert issuer.calls[0][2] == LEASE_SECONDS
    assert issuer.calls[1][2] == LEASE_SECONDS


def test_gateway_payload_is_none_without_issuer() -> None:
    clock = FakeClock(NOW_EPOCH)
    service = AssignmentCoordinationService(
        object(),  # type: ignore[arg-type]
        lease_seconds=LEASE_SECONDS,
        now_fn=clock.now,
    )
    assert service.gateway_payload(_detached_unit(None), hotkey="permitted") is None
