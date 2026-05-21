from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from platform_network.master.docker_orchestrator import (
    ChallengeResources,
    ChallengeSpec,
    DockerOrchestrationError,
)
from platform_network.master.kubernetes_orchestrator import (
    KubernetesOrchestrator,
    KubernetesTargetRouter,
)


class FakeKubernetesClient:
    def __init__(self) -> None:
        self.applied: list[dict[str, Any]] = []
        self.deleted: list[tuple[str, str | None]] = []
        self.waits: list[dict[str, Any]] = []

    def apply(self, resource: dict[str, Any]) -> dict[str, Any]:
        self.applied.append(resource)
        return resource

    def delete(self, resource: dict[str, Any] | str, name: str | None = None) -> None:
        if isinstance(resource, dict):
            self.deleted.append((resource["kind"], resource["metadata"]["name"]))
        else:
            self.deleted.append((resource, name))

    def wait_workload_ready(
        self, *, kind: str, name: str, replicas: int, timeout_seconds: int
    ) -> None:
        self.waits.append(
            {
                "kind": kind,
                "name": name,
                "replicas": replicas,
                "timeout_seconds": timeout_seconds,
            }
        )

    def service_json(
        self, service_name: str, path: str, *, port: int | str | None = None
    ) -> dict[str, Any]:
        if path == "health":
            return {"status": "ok", "service": service_name, "port": port}
        return {"api_version": "1.0", "capabilities": ["get_weights", "proxy_routes"]}


def test_deployment_start_applies_service_workload_hpa_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeKubernetesClient()
    orchestrator = KubernetesOrchestrator(
        client=client,
        mode="deployment",
        autoscaling_min_replicas=2,
        autoscaling_max_replicas=4,
        health_retries=1,
        health_retry_delay_seconds=0,
    )
    spec = ChallengeSpec(
        slug="demo",
        image="ghcr.io/org/demo:1",
        challenge_token="token",
        resources=ChallengeResources(cpu=1),
    )
    monkeypatch.setattr(
        orchestrator,
        "_get_json",
        lambda url: (
            {"status": "ok", "slug": "demo"}
            if url.endswith("/health")
            else {"api_version": "1.0", "capabilities": ["get_weights", "proxy_routes"]}
        ),
    )

    runtime = orchestrator.start_challenge(spec)
    kinds = [resource["kind"] for resource in client.applied]

    assert kinds == ["Secret", "Service", "Deployment", "HorizontalPodAutoscaler"]
    assert client.applied[2]["spec"]["replicas"] == 2
    assert client.applied[3]["spec"]["maxReplicas"] == 4
    assert client.waits[0]["replicas"] == 2
    assert runtime.container_name == "challenge-demo"
    assert orchestrator.runtime["demo"] == runtime


def test_statefulset_start_does_not_apply_hpa(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeKubernetesClient()
    orchestrator = KubernetesOrchestrator(
        client=client,
        mode="statefulset",
        autoscaling_max_replicas=4,
        health_retries=1,
        health_retry_delay_seconds=0,
    )
    spec = ChallengeSpec(slug="demo", image="ghcr.io/org/demo:1")
    monkeypatch.setattr(
        orchestrator,
        "_get_json",
        lambda url: (
            {"status": "ok"}
            if url.endswith("/health")
            else {"api_version": "1.0", "capabilities": ["get_weights", "proxy_routes"]}
        ),
    )

    orchestrator.start_challenge(spec)

    assert [resource["kind"] for resource in client.applied] == [
        "Service",
        "StatefulSet",
    ]
    assert client.waits[0]["replicas"] == 1


def test_deployment_start_can_apply_keda_scaled_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeKubernetesClient()
    orchestrator = KubernetesOrchestrator(
        client=client,
        mode="deployment",
        autoscaling_keda_enabled=True,
        autoscaling_min_replicas=1,
        autoscaling_max_replicas=3,
        health_retries=1,
        health_retry_delay_seconds=0,
    )
    spec = ChallengeSpec(
        slug="demo",
        image="ghcr.io/org/demo:1",
        resources=ChallengeResources(cpu=1),
    )
    monkeypatch.setattr(
        orchestrator,
        "_get_json",
        lambda url: (
            {"status": "ok"}
            if url.endswith("/health")
            else {"api_version": "1.0", "capabilities": ["get_weights", "proxy_routes"]}
        ),
    )

    orchestrator.start_challenge(spec)

    assert [resource["kind"] for resource in client.applied] == [
        "Service",
        "Deployment",
        "ScaledObject",
    ]


def test_stop_deletes_workloads_hpa_and_optional_service_secret() -> None:
    client = FakeKubernetesClient()
    orchestrator = KubernetesOrchestrator(client=client)

    orchestrator.stop_challenge("demo", remove=True)

    assert client.deleted == [
        ("Deployment", "challenge-demo"),
        ("StatefulSet", "challenge-demo"),
        ("HorizontalPodAutoscaler", "challenge-demo"),
        ("ScaledObject", "challenge-demo"),
        ("Service", "challenge-demo"),
        ("Secret", "challenge-demo-secrets"),
    ]


def test_pull_rejects_non_ghcr_images() -> None:
    orchestrator = KubernetesOrchestrator(client=FakeKubernetesClient())
    with pytest.raises(DockerOrchestrationError, match="GHCR"):
        orchestrator.pull_image("docker.io/org/demo:1")


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/org/demo@sha256:" + "a" * 64,
        "ghcr.io/org/demo:latest@sha256:" + "a" * 64,
    ],
)
def test_from_settings_production_policy_rejects_unpinned_challenge_images(
    image: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeKubernetesClient()
    monkeypatch.setattr(
        "platform_network.master.kubernetes_orchestrator.KubernetesClient",
        lambda **kwargs: client,
    )
    orchestrator = KubernetesOrchestrator.from_settings(
        _settings(environment="production", runtime_backend="kubernetes")
    )

    assert orchestrator.production_policy is True
    with pytest.raises(ValueError, match="semver-tagged digest-pinned"):
        orchestrator.start_challenge(_production_spec(image))


def test_from_settings_production_policy_rejects_secret_spec_before_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeKubernetesClient()
    monkeypatch.setattr(
        "platform_network.master.kubernetes_orchestrator.KubernetesClient",
        lambda **kwargs: client,
    )
    orchestrator = KubernetesOrchestrator.from_settings(
        _settings(environment="production", runtime_backend="kubernetes")
    )
    spec = ChallengeSpec(
        slug="demo",
        image="ghcr.io/org/demo@sha256:" + "a" * 64,
        challenge_token="challenge-token",
        docker_broker_token="broker-token",
        env={"CHALLENGE_DATABASE_URL": "postgresql://platform@db/platform"},
        resources=ChallengeResources(cpu=1, memory="512Mi"),
    )

    with pytest.raises(ValueError, match="semver-tagged digest-pinned"):
        orchestrator.start_challenge(spec)

    assert client.applied == []


def test_from_settings_production_policy_accepts_semver_digest_challenge_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeKubernetesClient()
    monkeypatch.setattr(
        "platform_network.master.kubernetes_orchestrator.KubernetesClient",
        lambda **kwargs: client,
    )
    orchestrator = KubernetesOrchestrator.from_settings(
        _settings(environment="production", runtime_backend="kubernetes")
    )
    monkeypatch.setattr(
        orchestrator,
        "_get_json",
        lambda url: (
            {"status": "ok"}
            if url.endswith("/health")
            else {"api_version": "1.0", "capabilities": ["get_weights", "proxy_routes"]}
        ),
    )

    runtime = orchestrator.start_challenge(
        _production_spec("ghcr.io/org/demo:1.2.3@sha256:" + "a" * 64)
    )

    assert runtime.slug == "demo"
    assert client.applied[1]["spec"]["template"]["spec"]["containers"][0]["image"] == (
        "ghcr.io/org/demo:1.2.3@sha256:" + "a" * 64
    )


def test_ready_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = KubernetesOrchestrator(
        client=FakeKubernetesClient(),
        health_retries=1,
        health_retry_delay_seconds=0,
    )
    spec = ChallengeSpec(slug="demo", image="ghcr.io/org/demo:1")
    monkeypatch.setattr(orchestrator, "_get_json", lambda url: {"status": "bad"})

    with pytest.raises(DockerOrchestrationError, match="failed Kubernetes"):
        orchestrator.wait_until_ready(spec)


def test_service_proxy_readiness_uses_kubernetes_api() -> None:
    client = FakeKubernetesClient()
    orchestrator = KubernetesOrchestrator(
        client=client,
        health_check_mode="service_proxy",
        health_retries=1,
        health_retry_delay_seconds=0,
    )

    health, version = orchestrator.wait_until_ready(
        ChallengeSpec(slug="demo", image="ghcr.io/org/demo:1")
    )

    assert health["service"] == "challenge-demo"
    assert version["api_version"] == "1.0"


def test_kubernetes_target_router_routes_explicit_and_gpu_targets() -> None:
    default = StubOrchestrator("default")
    gpu = StubOrchestrator("gpu-a")
    router = KubernetesTargetRouter(
        default_orchestrator=default,  # type: ignore[arg-type]
        target_orchestrators={"gpu-a": gpu},  # type: ignore[arg-type]
        target_capacities={"gpu-a": 2},
    )

    explicit = ChallengeSpec(
        slug="explicit",
        image="ghcr.io/org/demo:1",
        resources=ChallengeResources(gpu_server="gpu-a", gpu_count=1),
    )
    automatic = ChallengeSpec(
        slug="automatic",
        image="ghcr.io/org/demo:1",
        resources=ChallengeResources(gpu_count=1),
    )
    local = ChallengeSpec(slug="local", image="ghcr.io/org/demo:1")

    assert router.start_challenge(explicit).container_name == "gpu-a-explicit"
    assert router.start_challenge(automatic).container_name == "gpu-a-automatic"
    assert router.start_challenge(local).container_name == "default-local"
    with pytest.raises(DockerOrchestrationError, match="No valid Kubernetes target"):
        router.start_challenge(
            ChallengeSpec(
                slug="missing",
                image="ghcr.io/org/demo:1",
                resources=ChallengeResources(gpu_server="missing"),
            )
        )


def test_kubernetes_target_router_uses_dynamic_agent_assignments() -> None:
    registry = DynamicTargetRegistry()
    router = DynamicRouter(
        default_orchestrator=StubOrchestrator("default"),  # type: ignore[arg-type]
        settings=SimpleNamespace(),
        target_registry=registry,
    )

    registry.targets["agent-a"] = SimpleNamespace(
        id="agent-a", enabled=True, draining=False, gpu_count=2
    )
    runtime = router.start_challenge(
        ChallengeSpec(
            slug="automatic",
            image="ghcr.io/org/demo:1",
            resources=ChallengeResources(gpu_count=1),
        )
    )

    assert runtime.container_name == "agent-a-automatic"
    assert registry.assignments == {"automatic": "agent-a"}

    registry.targets["agent-a"].enabled = False
    router.stop_challenge("automatic", remove=True)
    assert registry.assignments == {}


def test_kubernetes_target_router_reuses_persisted_healthy_assignment() -> None:
    registry = DynamicTargetRegistry()
    registry.targets["agent-a"] = SimpleNamespace(
        id="agent-a", enabled=True, draining=False, gpu_count=2
    )
    registry.assign_challenge("demo", "agent-a", 1)
    router = DynamicRouter(
        default_orchestrator=StubOrchestrator("default"),  # type: ignore[arg-type]
        settings=SimpleNamespace(),
        target_registry=registry,
    )

    runtime = router.start_challenge(
        ChallengeSpec(
            slug="demo",
            image="ghcr.io/org/demo:1",
            resources=ChallengeResources(gpu_count=1),
        )
    )

    assert runtime.container_name == "agent-a-demo"
    assert registry.assignments == {"demo": "agent-a"}


def test_kubernetes_target_router_reassigns_unavailable_targets() -> None:
    registry = DynamicTargetRegistry()
    registry.targets["agent-a"] = SimpleNamespace(
        id="agent-a", enabled=False, draining=False, gpu_count=2
    )
    registry.targets["agent-b"] = SimpleNamespace(
        id="agent-b", enabled=True, draining=False, gpu_count=3
    )
    registry.assign_challenge("disabled", "agent-a", 1)
    registry.assign_challenge("deleted", "agent-missing", 1)
    registry.targets["agent-c"] = SimpleNamespace(
        id="agent-c", enabled=True, draining=True, gpu_count=2
    )
    registry.assign_challenge("draining", "agent-c", 1)
    router = DynamicRouter(
        default_orchestrator=StubOrchestrator("default"),  # type: ignore[arg-type]
        settings=SimpleNamespace(),
        target_registry=registry,
    )

    for slug in ("disabled", "deleted", "draining"):
        runtime = router.start_challenge(
            ChallengeSpec(
                slug=slug,
                image="ghcr.io/org/demo:1",
                resources=ChallengeResources(gpu_count=1),
            )
        )
        assert runtime.container_name == f"agent-b-{slug}"
        assert registry.assignments[slug] == "agent-b"


def test_kubernetes_target_router_rejects_unhealthy_or_over_capacity_targets() -> None:
    registry = DynamicTargetRegistry()
    registry.targets["agent-a"] = SimpleNamespace(
        id="agent-a", enabled=True, draining=False, gpu_count=1
    )
    registry.health_status["agent-a"] = "error"
    router = DynamicRouter(
        default_orchestrator=StubOrchestrator("default"),  # type: ignore[arg-type]
        settings=SimpleNamespace(),
        target_registry=registry,
    )

    with pytest.raises(DockerOrchestrationError, match="No valid Kubernetes target"):
        router.start_challenge(
            ChallengeSpec(
                slug="unhealthy",
                image="ghcr.io/org/demo:1",
                resources=ChallengeResources(gpu_count=1),
            )
        )

    registry.health_status["agent-a"] = "ok"
    registry.assign_challenge("existing", "agent-a", 1)
    with pytest.raises(DockerOrchestrationError, match="No valid Kubernetes target"):
        router.start_challenge(
            ChallengeSpec(
                slug="over-capacity",
                image="ghcr.io/org/demo:1",
                resources=ChallengeResources(gpu_count=1),
            )
        )


def test_kubernetes_target_router_builds_direct_kubeconfig_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class CapturingOrchestrator(StubOrchestrator):
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__("direct-a")

    registry = DynamicTargetRegistry()
    registry.targets["direct-a"] = SimpleNamespace(
        id="direct-a",
        mode="direct",
        enabled=True,
        draining=False,
        gpu_count=1,
        namespace="platform-gpu",
        storage_class="fast",
        node_selector={"accelerator": "nvidia"},
        tolerations=[{"key": "gpu", "operator": "Exists"}],
        runtime_class_name="nvidia",
        kubeconfig_file="/tmp/direct-a.kubeconfig",
    )
    monkeypatch.setattr(
        "platform_network.master.kubernetes_orchestrator.KubernetesOrchestrator",
        CapturingOrchestrator,
    )
    settings = _settings(environment="production", runtime_backend="kubernetes")
    router = KubernetesTargetRouter(
        default_orchestrator=StubOrchestrator("default"),  # type: ignore[arg-type]
        settings=settings,
        target_registry=registry,
    )

    runtime = router.start_challenge(
        ChallengeSpec(
            slug="demo",
            image="ghcr.io/org/demo:1",
            resources=ChallengeResources(gpu_server="direct-a", gpu_count=1),
        )
    )

    assert runtime.container_name == "direct-a-demo"
    assert captured["namespace"] == "platform-gpu"
    assert captured["storage_class_name"] == "fast"
    assert captured["node_selector"] == {"accelerator": "nvidia"}
    assert captured["tolerations"] == [{"key": "gpu", "operator": "Exists"}]
    assert captured["runtime_class_name"] == "nvidia"
    assert captured["health_check_mode"] == "service_proxy"
    assert captured["kubeconfig"] == "/tmp/direct-a.kubeconfig"
    assert captured["in_cluster"] is False
    assert captured["production_policy"] is True
    assert registry.assignments == {"demo": "direct-a"}


def test_kubernetes_target_router_builds_agent_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class CapturingAgentClient(StubOrchestrator):
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(kwargs["target_id"])

    registry = DynamicTargetRegistry()
    registry.targets["agent-a"] = SimpleNamespace(
        id="agent-a",
        mode="agent",
        enabled=True,
        draining=False,
        gpu_count=1,
        agent_url="https://agent-a",
        timeout_seconds=12.0,
        verify_tls=True,
    )
    registry.agent_tokens["agent-a"] = "test-agent-credential"
    monkeypatch.setattr(
        "platform_network.kubernetes.agent.KubernetesAgentClient",
        CapturingAgentClient,
    )
    settings = _settings()
    router = KubernetesTargetRouter(
        default_orchestrator=StubOrchestrator("default"),  # type: ignore[arg-type]
        settings=settings,
        target_registry=registry,
    )

    runtime = router.start_challenge(
        ChallengeSpec(
            slug="demo",
            image="ghcr.io/org/demo:1",
            resources=ChallengeResources(gpu_server="agent-a", gpu_count=1),
        )
    )

    assert runtime.container_name == "agent-a-demo"
    assert captured == {
        "target_id": "agent-a",
        "base_url": "https://agent-a",
        "token": "test-agent-credential",
        "timeout_seconds": 12.0,
        "verify_tls": True,
        "docker_broker_url": "http://broker:8082",
    }


class DynamicTargetRegistry:
    def __init__(self) -> None:
        self.targets: dict[str, Any] = {}
        self.assignments: dict[str, str] = {}
        self.metadata: dict[str, dict[str, object]] = {}
        self.health_status: dict[str, str] = {}
        self.agent_tokens: dict[str, str] = {}

    def list(self) -> list[Any]:
        return list(self.targets.values())

    def get(self, target_id: str) -> Any:
        return self.targets[target_id]

    def assign_challenge(
        self, slug: str, target_id: str, gpu_count: int | None = None
    ) -> None:
        self.assignments[slug] = target_id
        self.metadata[slug] = {"target_id": target_id, "gpu_count": gpu_count or 0}

    def get_assignment(self, slug: str) -> str | None:
        return self.assignments.get(slug)

    def clear_assignment(self, slug: str) -> None:
        self.assignments.pop(slug, None)
        self.metadata.pop(slug, None)

    def get_assignment_metadata(self, slug: str) -> dict[str, object] | None:
        return self.metadata.get(slug)

    def health(self, target_id: str) -> SimpleNamespace:
        return SimpleNamespace(status=self.health_status.get(target_id, "ok"))

    def get_agent_token(self, target_id: str) -> str:
        return self.agent_tokens.get(target_id, "")


class DynamicRouter(KubernetesTargetRouter):
    def _build_target_orchestrator(self, target: Any) -> Any:
        return StubOrchestrator(target.id)


class StubOrchestrator:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._runtime: dict[str, object] = {}

    @property
    def runtime(self):
        return dict(self._runtime)

    def start_challenge(self, spec: ChallengeSpec, *, recreate: bool = False):
        runtime = type(
            "Runtime",
            (),
            {
                "container_name": f"{self.prefix}-{spec.slug}",
                "slug": spec.slug,
                "image": spec.image,
            },
        )()
        self._runtime[spec.slug] = runtime
        return runtime

    def restart_challenge(self, spec: ChallengeSpec):
        return self.start_challenge(spec, recreate=True)

    def stop_challenge(self, slug: str, *, remove: bool = False) -> None:
        self._runtime.pop(slug, None)

    def pull_image(self, image: str):
        return image

    def pull_challenge(self, spec: ChallengeSpec):
        return self.start_challenge(spec)


def _production_spec(image: str) -> ChallengeSpec:
    return ChallengeSpec(
        slug="demo",
        image=image,
        env={"CHALLENGE_DATABASE_URL": "postgresql://platform@db/platform"},
        resources=ChallengeResources(cpu=1, memory="512Mi"),
    )


def _settings(
    *, environment: str = "development", runtime_backend: str = "docker"
) -> SimpleNamespace:
    return SimpleNamespace(
        environment=environment,
        runtime=SimpleNamespace(backend=runtime_backend),
        docker=SimpleNamespace(broker_url="http://broker:8082"),
        kubernetes=SimpleNamespace(
            namespace="platform",
            challenge_mode="deployment",
            storage_class=None,
            storage_size="10Gi",
            gpu_resource_name="nvidia.com/gpu",
            node_selector={},
            tolerations=[],
            runtime_class_name=None,
            image_pull_secrets=[],
            kubeconfig=None,
            in_cluster=True,
            autoscaling=SimpleNamespace(
                enabled=True,
                keda_enabled=False,
                min_replicas=1,
                max_replicas=3,
                target_cpu_utilization=70,
            ),
            target_defaults=SimpleNamespace(
                gpu_resource_name=None,
                node_selector={},
                tolerations=[],
                runtime_class_name=None,
                image_pull_secrets=[],
            ),
        ),
    )
