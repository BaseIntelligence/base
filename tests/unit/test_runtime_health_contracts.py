from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from sqlalchemy import text
from typer.testing import CliRunner

import base.cli_app.main as cli_main
from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.config import ChallengeSettings
from base.challenge_sdk.health import ReadinessProbe, evaluate_readiness
from base.challenge_sdk.roles import Capability, Role, capabilities_for_role
from base.challenge_sdk.schemas import (
    HealthResponse,
    RuntimeStatusResponse,
    VersionResponse,
)
from base.challenge_sdk.version import (
    API_VERSION,
    ARTIFACT_VERSION,
    DISTRIBUTION_NAME,
    RELEASE_ID,
    SDK_CONTRACT_VERSION,
)
from base.cli_app.main import app
from base.db.session import create_engine, create_session_factory
from base.master.app_proxy import create_proxy_app
from base.master.health import postgres_readiness_probe


class _ChallengeDatabase:
    def __init__(self) -> None:
        self.ready = True
        self.health_calls = 0

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def healthcheck(self) -> bool:
        self.health_calls += 1
        return self.ready


class _Registry:
    async def list(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


async def _empty_weights() -> dict[str, float]:
    return {}


def _challenge_app(database: _ChallengeDatabase) -> Any:
    return create_challenge_app(
        settings=ChallengeSettings(
            slug="prism",
            name="Prism",
            version="0.1.0",
            shared_token="test-token",
            shared_token_file=None,
        ),
        database=database,
        public_router=APIRouter(),
        get_weights_fn=_empty_weights,
    )


def test_challenge_health_readiness_version_and_methods_are_truthful() -> None:
    database = _ChallengeDatabase()
    with TestClient(_challenge_app(database)) as client:
        healthy = client.get("/health")
        ready = client.get("/ready")
        version = client.get("/version")

        assert healthy.status_code == 200
        assert healthy.headers["content-type"].startswith("application/json")
        assert HealthResponse.model_validate(healthy.json()).model_dump(
            mode="json"
        ) == {
            "status": "ok",
            "slug": "prism",
            "version": "0.1.0",
            "role": "challenge",
            "ready": True,
            "capabilities": [
                "challenge.scoring",
                "challenge.ordinary_proof",
                "challenge.state",
            ],
            "checks": [
                {"name": "database", "status": "ok", "required": True},
            ],
        }
        assert ready.status_code == 200
        parsed_version = VersionResponse.model_validate(version.json())
        assert parsed_version.challenge_slug == "prism"
        assert parsed_version.challenge_version == "0.1.0"
        assert parsed_version.role == Role.CHALLENGE

        calls_before_version = database.health_calls
        assert client.get("/version").status_code == 200
        assert database.health_calls == calls_before_version

        database.ready = False
        live_while_unready = client.get("/health")
        unavailable = client.get("/ready")
        assert live_while_unready.status_code == 200
        assert live_while_unready.json()["status"] == "unhealthy"
        assert live_while_unready.json()["ready"] is False
        assert unavailable.status_code == 503
        assert unavailable.json()["ready"] is False

        assert client.head("/health").status_code == 200
        assert client.head("/ready").status_code == 503
        assert client.head("/version").status_code == 200
        for path in ("/health", "/ready", "/version"):
            assert client.post(path).status_code == 405


def test_challenge_health_tracks_required_background_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "base.challenge_sdk.app_factory.signal.raise_signal",
        lambda _signal: None,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def worker(_app: Any) -> None:
        started.set()
        await release.wait()

    database = _ChallengeDatabase()
    app_instance = create_challenge_app(
        settings=ChallengeSettings(
            slug="worker-challenge",
            version="1.2.3",
            shared_token="test-token",
            shared_token_file=None,
        ),
        database=database,
        public_router=APIRouter(),
        get_weights_fn=_empty_weights,
        background_tasks=(worker,),
    )
    with TestClient(app_instance) as client:
        assert started.is_set()
        assert client.get("/ready").json()["ready"] is True
        release.set()
        for _ in range(100):
            if client.get("/ready").status_code == 503:
                break
        response = client.get("/health")
        assert response.json()["ready"] is False
        assert {
            "name": "worker",
            "status": "unhealthy",
            "required": True,
        } in response.json()["checks"]


def test_master_health_blocks_mutations_when_postgres_is_unavailable() -> None:
    dependency_ready = True
    probe_calls = 0

    async def postgres() -> bool:
        nonlocal probe_calls
        probe_calls += 1
        return dependency_ready

    master = create_proxy_app(
        registry=_Registry(),
        miner_verifier=object(),  # type: ignore[arg-type]
        readiness_probes=(ReadinessProbe(name="postgresql", check=postgres),),
    )

    @master.post("/test-mutation")
    async def mutation() -> dict[str, bool]:
        return {"mutated": True}

    with TestClient(master) as client:
        health = client.get("/health")
        version = client.get("/version")
        assert health.status_code == 200
        assert health.json()["role"] == "master"
        assert health.json()["ready"] is True
        assert VersionResponse.model_validate(version.json()).challenge_slug is None
        assert client.post("/test-mutation").status_code == 200

        dependency_ready = False
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        rejected = client.post("/test-mutation")
        assert rejected.status_code == 503
        assert rejected.json() == {
            "detail": {
                "code": "runtime_not_ready",
                "detail": "mandatory runtime dependencies are unavailable",
            }
        }
        calls_before_version = probe_calls
        assert client.get("/version").status_code == 200
        assert probe_calls == calls_before_version


async def test_postgres_readiness_requires_expected_migration_revision() -> None:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    session_factory = create_session_factory(engine)
    async with engine.begin() as connection:
        await connection.execute(
            text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        await connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES ('old')")
        )

    probe = postgres_readiness_probe(
        session_factory,
        expected_migration_revision="expected",
    )
    assert (await evaluate_readiness((probe,)))[0].status == "unhealthy"
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE alembic_version SET version_num = 'expected'")
        )
    assert (await evaluate_readiness((probe,)))[0].status == "ok"
    await engine.dispose()


def test_health_and_version_never_disclose_canary_secrets() -> None:
    canaries = (
        "canary-bearer-token",
        "postgresql://canary:secret@database/private",
        "/wallets/canary-hotkey",
        "provider-canary-api-key",
    )
    database = _ChallengeDatabase()
    app_instance = _challenge_app(database)
    app_instance.state.canary_secrets = canaries

    with TestClient(app_instance) as client:
        for path in ("/health", "/ready", "/version"):
            response = client.get(path)
            material = json.dumps(dict(response.headers)) + response.text
            assert all(canary not in material for canary in canaries)


def test_validator_status_is_role_scoped_and_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_health = HealthResponse(
        slug="base-master",
        version=ARTIFACT_VERSION,
        role=Role.MASTER.value,
        capabilities=capabilities_for_role(Role.MASTER),
    )
    master_version = VersionResponse(
        distribution_name=DISTRIBUTION_NAME,
        artifact_version=ARTIFACT_VERSION,
        release_id=RELEASE_ID,
        api_version=API_VERSION,
        challenge_slug=None,
        challenge_version=ARTIFACT_VERSION,
        sdk_contract_version=SDK_CONTRACT_VERSION,
        sdk_version=SDK_CONTRACT_VERSION,
        role=Role.MASTER.value,
        capabilities=capabilities_for_role(Role.MASTER),
    )
    requested: list[str] = []

    @asynccontextmanager
    async def fake_client(
        _base_url: str, _timeout_seconds: float
    ) -> AsyncIterator[Any]:
        class _Client:
            async def get(self, path: str) -> httpx.Response:
                requested.append(path)
                payload = master_health if path == "/health" else master_version
                return httpx.Response(
                    200,
                    json=payload.model_dump(mode="json"),
                    request=httpx.Request("GET", f"http://master{path}"),
                )

        yield _Client()

    settings = SimpleNamespace(
        validator=SimpleNamespace(
            submit_on_chain_enabled=False,
            agent=SimpleNamespace(
                master_url="http://master",
                request_timeout_seconds=1.0,
            ),
            resolved_weights_url="http://master",
        )
    )
    monkeypatch.setattr(cli_main, "load_settings", lambda _path: settings)
    monkeypatch.setattr(cli_main, "_validator_status_client", fake_client)
    monkeypatch.setattr(
        cli_main,
        "create_bittensor_submit_runtime",
        lambda *_args, **_kwargs: pytest.fail("status constructed a chain runtime"),
    )

    result = CliRunner().invoke(app, ["validator", "status"])

    assert result.exit_code == 0, result.output
    status = RuntimeStatusResponse.model_validate_json(result.output)
    assert status.health.role == Role.VALIDATOR
    assert status.health.ready is True
    assert status.version.role == Role.VALIDATOR
    assert Capability.VALIDATOR_OWN_SET_WEIGHTS not in status.version.capabilities
    assert all(
        not capability.startswith("master.")
        for capability in status.version.capabilities
    )
    assert requested == ["/health", "/version"]
