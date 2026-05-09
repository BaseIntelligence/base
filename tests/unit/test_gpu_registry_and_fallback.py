from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from platform_network.gpu.capabilities import ResourceCapabilityChecker
from platform_network.gpu.registry import FileGpuServerRegistry
from platform_network.master.docker_orchestrator import ChallengeResources
from platform_network.master.weight_fallback import (
    FallbackWeightClient,
    LatestWeightsStore,
    SignedWeightsPayload,
    SignedWeightsService,
    WeightFallbackError,
    sign_payload,
)
from platform_network.schemas.gpu_server import GpuServerCreate, GpuServerUpdate
from platform_network.schemas.weight_fallback import SignedWeightsResponse
from platform_network.schemas.weights import FinalWeights


def test_file_gpu_registry_crud_and_token_redaction(tmp_path: Path) -> None:
    registry = FileGpuServerRegistry(
        tmp_path / "gpu.json",
        secret_dir=tmp_path / "secrets",
    )

    created = registry.create(
        GpuServerCreate(
            id="gpu-a",
            base_url="https://gpu-a",
            token="super-secret-token",
            min_gpu_count=2,
        )
    )

    assert created.token_hint == "supe…oken"
    assert registry.get_token("gpu-a") == "super-secret-token"
    assert "super-secret-token" not in registry.state_file.read_text(encoding="utf-8")

    updated = registry.update("gpu-a", GpuServerUpdate(enabled=False))
    assert updated.enabled is False
    assert registry.set_enabled("gpu-a", True).enabled is True

    reloaded = FileGpuServerRegistry(
        tmp_path / "gpu.json",
        secret_dir=tmp_path / "secrets",
    )
    assert reloaded.get("gpu-a").base_url == "https://gpu-a"
    reloaded.delete("gpu-a")
    assert reloaded.list() == []


def test_resource_capability_checker_gpu_decisions(tmp_path: Path) -> None:
    registry = {
        "gpu-a": FileGpuServerRegistry(
            tmp_path / "unused.json", secret_dir=tmp_path
        ).create(GpuServerCreate(id="gpu-a", base_url="http://gpu", min_gpu_count=1))
    }
    checker = ResourceCapabilityChecker(registry)

    assert checker.check(ChallengeResources()).can_run is True
    assert checker.check(ChallengeResources(gpu_server="gpu-a", gpu_count=1)).can_run
    assert (
        checker.check(ChallengeResources(gpu_server="missing", gpu_count=1)).reason
        == "gpu_server_unknown"
    )
    assert (
        checker.check(ChallengeResources(gpu_server="gpu-a", gpu_count=2)).reason
        == "gpu_capacity_insufficient"
    )


def test_signed_weights_store_and_service(tmp_path: Path) -> None:
    store = LatestWeightsStore(tmp_path / "weights.json")
    store.write_final(FinalWeights(uids=[1], weights=[1.0], hotkey_weights={"hk": 1.0}))
    service = SignedWeightsService(store=store, signing_secret="secret")

    signed = service.latest("demo")

    assert signed.payload.challenge_slug == "demo"
    assert signed.payload.weights == {"hk": 1.0}
    assert signed.signature == sign_payload(signed.payload, "secret")


@pytest.mark.asyncio
async def test_fallback_weight_client_validates_signature_and_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = SignedWeightsPayload(weights={"hk": 1.0})
    signed = SignedWeightsResponse(
        payload=payload,
        signature=sign_payload(payload, "secret"),
    )

    class AsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, *args: object, **kwargs: object) -> httpx.Response:
            return httpx.Response(
                200,
                json=signed.model_dump(mode="json"),
                request=httpx.Request("GET", "https://primary/v1/weights/latest"),
            )

    monkeypatch.setattr(
        "platform_network.master.weight_fallback.httpx.AsyncClient",
        AsyncClient,
    )
    result = await FallbackWeightClient(
        primary_url="https://primary",
        token="tok",
        signing_secret="secret",
    ).get_weights(slug="demo", emission_percent=10)
    assert result.ok is True
    assert result.weights == {"hk": 1.0}

    old = SignedWeightsPayload(
        weights={"hk": 1.0},
        computed_at=datetime.now(UTC) - timedelta(seconds=999),
    )
    signed.payload = old
    signed.signature = sign_payload(old, "secret")
    with pytest.raises(WeightFallbackError):
        await FallbackWeightClient(
            primary_url="https://primary",
            token="tok",
            signing_secret="secret",
            max_age_seconds=1,
        ).get_weights(slug="demo", emission_percent=10)
