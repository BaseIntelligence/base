from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import platform_network.gpu.client as gpu_client_module
from platform_network.gpu.agent import GpuAgentService, create_gpu_agent_app
from platform_network.gpu.client import GpuAgentClient
from platform_network.gpu.router import ChallengeOrchestratorRouter
from platform_network.master.docker_orchestrator import (
    ChallengeResources,
    ChallengeRuntime,
    ChallengeSpec,
    DockerOrchestrationError,
)


def _runtime(slug: str = "demo") -> ChallengeRuntime:
    return ChallengeRuntime(
        slug=slug,
        image="ghcr.io/org/demo:1",
        container_id="cid",
        container_name=f"challenge-{slug}",
        internal_base_url=f"http://challenge-{slug}:8000",
        sqlite_volume_name=f"platform_{slug}_sqlite",
        health={"status": "ok"},
        version={
            "api_version": "1.0",
            "challenge_version": "1",
            "capabilities": ["get_weights", "proxy_routes"],
        },
    )


class FakeOrchestrator:
    def __init__(self) -> None:
        self.started: list[tuple[ChallengeSpec, bool]] = []
        self.restarted: list[ChallengeSpec] = []
        self.stopped: list[tuple[str, bool]] = []
        self._runtime: dict[str, ChallengeRuntime] = {}

    @property
    def runtime(self) -> dict[str, ChallengeRuntime]:
        return dict(self._runtime)

    def start_challenge(
        self, spec: ChallengeSpec, *, recreate: bool = False
    ) -> ChallengeRuntime:
        self.started.append((spec, recreate))
        runtime = _runtime(spec.slug)
        self._runtime[spec.slug] = runtime
        return runtime

    def restart_challenge(self, spec: ChallengeSpec) -> ChallengeRuntime:
        self.restarted.append(spec)
        return _runtime(spec.slug)

    def stop_challenge(self, slug: str, *, remove: bool = False) -> None:
        self.stopped.append((slug, remove))


def test_gpu_agent_auth_start_status_restart_and_stop() -> None:
    orchestrator = FakeOrchestrator()
    app = create_gpu_agent_app(
        token_provider=lambda: "tok",
        service=GpuAgentService(orchestrator),  # type: ignore[arg-type]
    )
    client = TestClient(app)
    payload = {
        "slug": "demo",
        "image": "ghcr.io/org/demo:1",
        "version": "1",
        "challenge_token": "secret",
        "resources": {"gpu_count": 1, "gpu_capabilities": ["gpu", "compute"]},
        "recreate": True,
    }

    unauthorized = client.post("/v1/challenges/start", json=payload)
    assert unauthorized.status_code == 401

    response = client.post(
        "/v1/challenges/start",
        headers={"authorization": "Bearer tok"},
        json=payload,
    )
    assert response.status_code == 200, response.text
    assert response.json()["slug"] == "demo"
    spec, recreate = orchestrator.started[0]
    assert recreate is True
    assert spec.resources.gpu_count == 1
    assert spec.resources.gpu_capabilities == ("gpu", "compute")

    status_response = client.get(
        "/v1/challenges/demo/status", headers={"authorization": "Bearer tok"}
    )
    assert status_response.status_code == 200

    restart_response = client.post(
        "/v1/challenges/restart",
        headers={"authorization": "Bearer tok"},
        json=payload,
    )
    assert restart_response.status_code == 200
    assert orchestrator.restarted[0].slug == "demo"

    stop_response = client.post(
        "/v1/challenges/stop",
        headers={"authorization": "Bearer tok"},
        json={"slug": "demo", "remove": True},
    )
    assert stop_response.status_code == 200
    assert orchestrator.stopped == [("demo", True)]


def test_gpu_router_routes_local_and_remote() -> None:
    local = FakeOrchestrator()
    remote_calls: list[tuple[str, ChallengeSpec]] = []

    class RemoteClient:
        def start_challenge(
            self, spec: ChallengeSpec, *, recreate: bool = False
        ) -> ChallengeRuntime:
            remote_calls.append(("start", spec))
            return _runtime(spec.slug)

        def restart_challenge(self, spec: ChallengeSpec) -> ChallengeRuntime:
            remote_calls.append(("restart", spec))
            return _runtime(spec.slug)

        def stop_challenge(self, slug: str, *, remove: bool = False) -> None:
            remote_calls.append(("stop", ChallengeSpec(slug=slug, image="x")))

    router = ChallengeOrchestratorRouter(
        local_orchestrator=local,  # type: ignore[arg-type]
        gpu_clients={"gpu-a": RemoteClient()},  # type: ignore[dict-item]
    )
    local_spec = ChallengeSpec(slug="cpu", image="ghcr.io/org/cpu:1")
    gpu_spec = ChallengeSpec(
        slug="gpu",
        image="ghcr.io/org/gpu:1",
        resources=ChallengeResources(gpu_server="gpu-a", gpu_count=1),
    )

    router.start_challenge(local_spec)
    router.start_challenge(gpu_spec)
    router.restart_challenge(gpu_spec)
    router.stop_challenge("gpu", remove=True)

    assert local.started[0][0].slug == "cpu"
    assert [call[0] for call in remote_calls] == ["start", "restart", "stop"]
    assert "gpu" not in router.runtime


def test_gpu_router_rejects_unknown_server() -> None:
    router = ChallengeOrchestratorRouter(
        local_orchestrator=SimpleNamespace(runtime={}), gpu_clients={}
    )
    with pytest.raises(DockerOrchestrationError):
        router.start_challenge(
            ChallengeSpec(
                slug="gpu",
                image="ghcr.io/org/gpu:1",
                resources=ChallengeResources(gpu_server="missing", gpu_count=1),
            )
        )


def test_gpu_agent_client_serializes_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class Client:
        def __init__(self, **kwargs: object) -> None:
            calls.append({"init": kwargs})

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(
            self, path: str, json: dict[str, object], headers: dict[str, str]
        ) -> httpx.Response:
            calls.append({"path": path, "json": json, "headers": headers})
            return httpx.Response(
                200,
                json=_runtime("gpu").__dict__,
                request=httpx.Request("POST", "https://gpu/v1/challenges/start"),
            )

    monkeypatch.setattr(gpu_client_module.httpx, "Client", Client)
    client = GpuAgentClient(
        server_id="gpu-a",
        base_url="https://gpu",
        token="tok",
        timeout_seconds=5,
        verify_tls=False,
    )
    runtime = client.start_challenge(
        ChallengeSpec(
            slug="gpu",
            image="ghcr.io/org/gpu:1",
            resources=ChallengeResources(gpu_server="gpu-a", gpu_count=1),
        ),
        recreate=True,
    )

    assert runtime.slug == "gpu"
    assert calls[0]["init"] == {
        "base_url": "https://gpu",
        "timeout": 5,
        "verify": False,
    }
    assert calls[1]["path"] == "/v1/challenges/start"
    assert calls[1]["headers"] == {"Authorization": "Bearer tok"}
    assert calls[1]["json"]["resources"]["gpu_count"] == 1  # type: ignore[index]
