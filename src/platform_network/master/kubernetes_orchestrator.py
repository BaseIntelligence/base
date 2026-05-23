from __future__ import annotations

import asyncio
import base64
import secrets
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from platform_network.config.policy import production_policy_enabled_for_settings
from platform_network.kubernetes.client import KubernetesClient
from platform_network.kubernetes.names import (
    POSTGRES_SECRET_KEY_DATABASE_URL,
    POSTGRES_SECRET_KEY_PASSWORD,
    challenge_name,
    challenge_postgres_names,
    challenge_secret_name,
)
from platform_network.kubernetes.resources import (
    build_challenge_hpa,
    build_challenge_postgres_secret,
    build_challenge_postgres_service,
    build_challenge_postgres_statefulset,
    build_challenge_scaled_object,
    build_challenge_secret,
    build_challenge_service,
    build_challenge_workload,
)
from platform_network.master.docker_orchestrator import (
    ChallengeRuntime,
    ChallengeSpec,
    DockerOrchestrationError,
)


def _managed_postgres_resources_from_settings(settings: Any) -> dict[str, Any] | None:
    requests = dict(getattr(settings, "requests", {}) or {})
    limits = dict(getattr(settings, "limits", {}) or {})
    resources: dict[str, Any] = {}
    if requests:
        resources["requests"] = requests
    if limits:
        resources["limits"] = limits
    return resources or None


class KubernetesOrchestrator:
    """Orchestrate challenges as Kubernetes workloads."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        namespace: str = "platform",
        mode: str = "statefulset",
        storage_class_name: str | None = None,
        storage_size: str = "10Gi",
        pull_ghcr_only: bool = True,
        request_timeout_seconds: float = 5.0,
        health_retries: int = 12,
        health_retry_delay_seconds: float = 2.0,
        gpu_resource_name: str = "nvidia.com/gpu",
        node_selector: dict[str, str] | None = None,
        tolerations: list[dict[str, Any]] | None = None,
        runtime_class_name: str | None = None,
        image_pull_secrets: list[str] | None = None,
        autoscaling_enabled: bool = True,
        autoscaling_keda_enabled: bool = False,
        autoscaling_min_replicas: int = 1,
        autoscaling_max_replicas: int = 3,
        autoscaling_target_cpu_utilization: int = 70,
        docker_broker_url: str = "http://platform-broker:8082",
        health_check_mode: str = "direct",
        kubeconfig: str | None = None,
        in_cluster: bool = True,
        production_policy: bool = False,
        managed_postgres_enabled: bool = True,
        managed_postgres_image: str = "postgres:16-alpine",
        managed_postgres_storage_class_name: str | None = None,
        managed_postgres_storage_size: str = "10Gi",
        managed_postgres_retain_pvc: bool = True,
        managed_postgres_retain_secret: bool = True,
        managed_postgres_resources: dict[str, Any] | None = None,
    ) -> None:
        self.client = client or KubernetesClient(
            namespace=namespace, kubeconfig=kubeconfig, in_cluster=in_cluster
        )
        self.namespace = namespace
        self.mode = mode
        self.storage_class_name = storage_class_name
        self.storage_size = storage_size
        self.pull_ghcr_only = pull_ghcr_only
        self.request_timeout_seconds = request_timeout_seconds
        self.health_retries = health_retries
        self.health_retry_delay_seconds = health_retry_delay_seconds
        self.gpu_resource_name = gpu_resource_name
        self.node_selector = node_selector or {}
        self.tolerations = tolerations or []
        self.runtime_class_name = runtime_class_name
        self.image_pull_secrets = image_pull_secrets or []
        self.autoscaling_enabled = autoscaling_enabled
        self.autoscaling_keda_enabled = autoscaling_keda_enabled
        self.autoscaling_min_replicas = autoscaling_min_replicas
        self.autoscaling_max_replicas = autoscaling_max_replicas
        self.autoscaling_target_cpu_utilization = autoscaling_target_cpu_utilization
        self.docker_broker_url = docker_broker_url
        self.health_check_mode = health_check_mode
        self.production_policy = production_policy
        self.managed_postgres_enabled = managed_postgres_enabled
        self.managed_postgres_image = managed_postgres_image
        self.managed_postgres_storage_class_name = managed_postgres_storage_class_name
        self.managed_postgres_storage_size = managed_postgres_storage_size
        self.managed_postgres_retain_pvc = managed_postgres_retain_pvc
        self.managed_postgres_retain_secret = managed_postgres_retain_secret
        self.managed_postgres_resources = managed_postgres_resources
        self._runtime: dict[str, ChallengeRuntime] = {}

    @property
    def runtime(self) -> dict[str, ChallengeRuntime]:
        return dict(self._runtime)

    def pull_image(self, image: str) -> object:
        if self.pull_ghcr_only and not image.startswith("ghcr.io/"):
            raise DockerOrchestrationError("Challenge images must be pulled from GHCR")
        return {"image": image, "status": "scheduled-by-kubernetes"}

    def start_challenge(
        self, spec: ChallengeSpec, *, recreate: bool = False
    ) -> ChallengeRuntime:
        self.pull_image(spec.image)
        workload = build_challenge_workload(
            spec,
            namespace=self.namespace,
            mode=self.mode,
            storage_class_name=self.storage_class_name,
            storage_size=self.storage_size,
            replicas=self._initial_replicas(),
            gpu_resource_name=self.gpu_resource_name,
            node_selector=self.node_selector,
            tolerations=self.tolerations,
            runtime_class_name=self.runtime_class_name,
            image_pull_secrets=self.image_pull_secrets,
            docker_broker_url=self.docker_broker_url,
            production=self.production_policy,
            managed_postgres=self.managed_postgres_enabled,
        )
        if recreate:
            self.stop_challenge(spec.slug, remove=True)
        if self.managed_postgres_enabled:
            self._ensure_managed_postgres(spec.slug)
        secret = build_challenge_secret(spec, namespace=self.namespace)
        if secret is not None:
            self.client.apply(secret)
        service = build_challenge_service(spec, namespace=self.namespace)
        self.client.apply(service)
        self.client.apply(workload)
        if self._should_apply_autoscaler(spec):
            if self.autoscaling_keda_enabled:
                autoscaler = build_challenge_scaled_object(
                    spec,
                    namespace=self.namespace,
                    min_replicas=self.autoscaling_min_replicas,
                    max_replicas=self.autoscaling_max_replicas,
                    target_cpu_utilization=self.autoscaling_target_cpu_utilization,
                )
            else:
                autoscaler = build_challenge_hpa(
                    spec,
                    namespace=self.namespace,
                    min_replicas=self.autoscaling_min_replicas,
                    max_replicas=self.autoscaling_max_replicas,
                    target_cpu_utilization=self.autoscaling_target_cpu_utilization,
                )
            self.client.apply(autoscaler)
        self.client.wait_workload_ready(
            kind=workload["kind"],
            name=workload["metadata"]["name"],
            replicas=workload["spec"]["replicas"],
            timeout_seconds=int(
                self.health_retries * self.health_retry_delay_seconds
                + self.request_timeout_seconds
            ),
        )
        health, version = self.wait_until_ready(spec)
        runtime = ChallengeRuntime(
            slug=spec.slug,
            image=spec.image,
            container_id=challenge_name(spec.slug),
            container_name=challenge_name(spec.slug),
            internal_base_url=spec.internal_base_url,
            sqlite_volume_name=spec.sqlite_volume_name,
            health=health,
            version=version,
        )
        self._runtime[spec.slug] = runtime
        return runtime

    def _ensure_managed_postgres(self, slug: str) -> None:
        names = challenge_postgres_names(slug)
        try:
            database_url = self._reusable_managed_postgres_database_url(slug)
            if database_url is None:
                password = secrets.token_urlsafe(32)
                database_url = self._managed_postgres_database_url(slug, password)
                self.client.apply(
                    build_challenge_postgres_secret(
                        slug,
                        namespace=self.namespace,
                        retain=self.managed_postgres_retain_secret,
                        password=password,
                        database_url=database_url,
                    )
                )
            self.client.apply(
                build_challenge_postgres_service(slug, namespace=self.namespace)
            )
            statefulset = build_challenge_postgres_statefulset(
                slug,
                namespace=self.namespace,
                image=self.managed_postgres_image,
                storage_class_name=(
                    self.managed_postgres_storage_class_name or self.storage_class_name
                ),
                storage_size=self.managed_postgres_storage_size,
                retain_pvc=self.managed_postgres_retain_pvc,
                resources=self.managed_postgres_resources,
            )
            self.client.apply(statefulset)
            self.client.wait_workload_ready(
                kind=statefulset["kind"],
                name=names.statefulset_name,
                replicas=1,
                timeout_seconds=int(
                    self.health_retries * self.health_retry_delay_seconds
                    + self.request_timeout_seconds
                ),
            )
            self._check_managed_postgres_readiness(
                slug=slug,
                service_name=names.service_name,
                database_url=database_url,
            )
        except Exception:
            raise DockerOrchestrationError(
                f"Managed Postgres for challenge {slug!r} "
                f"({names.statefulset_name}) failed readiness/authentication"
            ) from None

    def _reusable_managed_postgres_database_url(self, slug: str) -> str | None:
        names = challenge_postgres_names(slug)
        secret = self.client.get("Secret", names.secret_name)
        if not secret:
            return None
        password = self._secret_value(secret, POSTGRES_SECRET_KEY_PASSWORD)
        database_url = self._secret_value(secret, POSTGRES_SECRET_KEY_DATABASE_URL)
        if not password or not database_url:
            return None
        return database_url

    def _managed_postgres_database_url(self, slug: str, password: str) -> str:
        names = challenge_postgres_names(slug)
        encoded_password = quote(password, safe="")
        return (
            f"postgresql+asyncpg://{names.database_user}:{encoded_password}"
            f"@{names.service_name}:5432/{names.database_name}"
        )

    def _check_managed_postgres_readiness(
        self, *, slug: str, service_name: str, database_url: str
    ) -> None:
        client_hook = getattr(self.client, "check_postgres_ready", None)
        if callable(client_hook):
            client_hook(slug=slug, service_name=service_name, database_url=database_url)
            return
        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import create_async_engine
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise DockerOrchestrationError(
                f"Managed Postgres for challenge {slug!r} cannot authenticate; "
                "SQLAlchemy asyncio support is unavailable"
            ) from exc

        async def probe() -> None:
            engine = create_async_engine(
                database_url,
                connect_args={"timeout": self.request_timeout_seconds},
            )
            try:
                async with engine.connect() as connection:
                    await connection.execute(text("SELECT 1"))
            finally:
                await engine.dispose()

        def run_probe() -> None:
            asyncio.run(probe())

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            run_probe()
            return
        error: list[BaseException] = []

        def threaded_probe() -> None:
            try:
                run_probe()
            except BaseException as exc:  # pragma: no cover - thread handoff
                error.append(exc)

        thread = threading.Thread(target=threaded_probe, daemon=True)
        thread.start()
        thread.join()
        if error:
            raise error[0]

    @staticmethod
    def _secret_value(secret: dict[str, Any], key: str) -> str | None:
        string_data = secret.get("stringData") or secret.get("string_data") or {}
        if key in string_data:
            value = string_data[key]
            return str(value) if value else None
        data = secret.get("data") or {}
        encoded = data.get(key)
        if not encoded:
            return None
        try:
            return base64.b64decode(str(encoded)).decode("utf-8")
        except Exception:
            return None

    @staticmethod
    def _managed_postgres_resources_from_settings(
        settings: Any,
    ) -> dict[str, Any] | None:
        return _managed_postgres_resources_from_settings(settings)

    def restart_challenge(self, spec: ChallengeSpec) -> ChallengeRuntime:
        return self.start_challenge(spec, recreate=True)

    def stop_challenge(self, slug: str, *, remove: bool = False) -> None:
        name = challenge_name(slug)
        self.client.delete("Deployment", name)
        self.client.delete("StatefulSet", name)
        self.client.delete("HorizontalPodAutoscaler", name)
        self.client.delete("ScaledObject", name)
        if remove:
            self.client.delete("Service", name)
            self.client.delete("Secret", challenge_secret_name(slug))
            if self.managed_postgres_enabled:
                names = challenge_postgres_names(slug)
                self.client.delete("StatefulSet", names.statefulset_name)
                self.client.delete("Service", names.service_name)
        self._runtime.pop(slug, None)

    def pull_challenge(self, spec: ChallengeSpec) -> object:
        return self.start_challenge(spec, recreate=False)

    def wait_until_ready(
        self, spec: ChallengeSpec
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        last_error: Exception | None = None
        for _attempt in range(self.health_retries):
            try:
                health = self._challenge_json(spec, "health")
                self._validate_health(spec, health)
                version = self._challenge_json(spec, "version")
                self._validate_version(spec, version)
                return health, version
            except Exception as exc:
                last_error = exc
                time.sleep(self.health_retry_delay_seconds)
        raise DockerOrchestrationError(
            f"Challenge {spec.slug!r} failed Kubernetes health/version checks"
        ) from last_error

    def _get_json(self, url: str) -> dict[str, Any]:
        response = httpx.get(url, timeout=self.request_timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise DockerOrchestrationError(
                f"Challenge endpoint {url} returned non-object JSON"
            )
        return payload

    def _challenge_json(self, spec: ChallengeSpec, path: str) -> dict[str, Any]:
        if self.health_check_mode == "service_proxy":
            return self.client.service_json(
                challenge_name(spec.slug), path, port=spec.port
            )
        return self._get_json(f"{spec.internal_base_url}/{path}")

    def _should_apply_autoscaler(self, spec: ChallengeSpec) -> bool:
        return (
            self.mode == "deployment"
            and self.autoscaling_enabled
            and self.autoscaling_max_replicas > 1
            and spec.resources.cpu is not None
        )

    def _initial_replicas(self) -> int:
        if self.mode == "deployment" and self.autoscaling_enabled:
            return self.autoscaling_min_replicas
        return 1

    def _validate_health(self, spec: ChallengeSpec, payload: dict[str, Any]) -> None:
        status = payload.get("status")
        if status not in {None, "ok", "healthy"}:
            raise DockerOrchestrationError(
                f"Challenge {spec.slug!r} reported unhealthy status: {status!r}"
            )

    def _validate_version(self, spec: ChallengeSpec, payload: dict[str, Any]) -> None:
        api_version = payload.get("api_version") or payload.get("apiVersion")
        if api_version is not None and str(api_version) != spec.expected_api_version:
            raise DockerOrchestrationError(
                f"Challenge {spec.slug!r} API version mismatch: {api_version!r}"
            )
        capabilities = payload.get("capabilities")
        if capabilities is None:
            return
        if not isinstance(capabilities, list):
            raise DockerOrchestrationError(
                f"Challenge {spec.slug!r} returned invalid capabilities"
            )
        missing = set(spec.required_capabilities).difference(
            str(item) for item in capabilities
        )
        if missing:
            raise DockerOrchestrationError(
                f"Challenge {spec.slug!r} missing capabilities: {sorted(missing)}"
            )

    @classmethod
    def from_settings(cls, settings: Any) -> KubernetesOrchestrator:
        production_policy = production_policy_enabled_for_settings(settings)
        return cls(
            namespace=settings.kubernetes.namespace,
            mode=settings.kubernetes.challenge_mode,
            storage_class_name=settings.kubernetes.storage_class,
            storage_size=settings.kubernetes.storage_size,
            gpu_resource_name=settings.kubernetes.gpu_resource_name,
            node_selector=settings.kubernetes.node_selector,
            tolerations=settings.kubernetes.tolerations,
            runtime_class_name=settings.kubernetes.runtime_class_name,
            image_pull_secrets=settings.kubernetes.image_pull_secrets,
            autoscaling_enabled=settings.kubernetes.autoscaling.enabled,
            autoscaling_keda_enabled=settings.kubernetes.autoscaling.keda_enabled,
            autoscaling_min_replicas=settings.kubernetes.autoscaling.min_replicas,
            autoscaling_max_replicas=settings.kubernetes.autoscaling.max_replicas,
            autoscaling_target_cpu_utilization=(
                settings.kubernetes.autoscaling.target_cpu_utilization
            ),
            docker_broker_url=settings.docker.broker_url,
            health_check_mode=(
                "direct" if settings.kubernetes.in_cluster else "service_proxy"
            ),
            kubeconfig=settings.kubernetes.kubeconfig,
            in_cluster=settings.kubernetes.in_cluster,
            production_policy=production_policy,
            managed_postgres_enabled=settings.kubernetes.managed_postgres.enabled,
            managed_postgres_image=settings.kubernetes.managed_postgres.image,
            managed_postgres_storage_class_name=(
                settings.kubernetes.managed_postgres.storage_class
            ),
            managed_postgres_storage_size=settings.kubernetes.managed_postgres.storage_size,
            managed_postgres_retain_pvc=settings.kubernetes.managed_postgres.retain_pvc,
            managed_postgres_retain_secret=settings.kubernetes.managed_postgres.retain_secret,
            managed_postgres_resources=_managed_postgres_resources_from_settings(
                settings.kubernetes.managed_postgres.resources
            ),
        )


class KubernetesTargetRouter:
    def __init__(
        self,
        *,
        default_orchestrator: KubernetesOrchestrator,
        target_orchestrators: dict[str, Any] | None = None,
        target_capacities: dict[str, int] | None = None,
        settings: Any | None = None,
        target_registry: Any | None = None,
    ) -> None:
        self.default_orchestrator = default_orchestrator
        self.target_orchestrators = target_orchestrators or {}
        self.target_capacities = target_capacities or {}
        self.settings = settings
        self.target_registry = target_registry
        self._slug_to_target: dict[str, str] = {}
        self._slug_gpu_counts: dict[str, int] = {}

    @property
    def runtime(self) -> dict[str, ChallengeRuntime]:
        runtime = self.default_orchestrator.runtime
        for orchestrator in self.target_orchestrators.values():
            runtime.update(orchestrator.runtime)
        return runtime

    def start_challenge(
        self, spec: ChallengeSpec, *, recreate: bool = False
    ) -> ChallengeRuntime:
        target_id, orchestrator = self._select(spec)
        runtime = orchestrator.start_challenge(spec, recreate=recreate)
        if target_id:
            self._slug_to_target[spec.slug] = target_id
            self._slug_gpu_counts[spec.slug] = int(spec.resources.gpu_count or 0)
            self._assign(spec.slug, target_id)
        return runtime

    def restart_challenge(self, spec: ChallengeSpec) -> ChallengeRuntime:
        return self.start_challenge(spec, recreate=True)

    def stop_challenge(self, slug: str, *, remove: bool = False) -> None:
        target_id = self._assignment_for(slug) or self._slug_to_target.pop(slug, None)
        if target_id:
            self._orchestrator_for_target(target_id).stop_challenge(slug, remove=remove)
            self._clear_assignment(slug)
            return
        self.default_orchestrator.stop_challenge(slug, remove=remove)

    def pull_image(self, image: str) -> object:
        return self.default_orchestrator.pull_image(image)

    def pull_challenge(self, spec: ChallengeSpec) -> object:
        return self.start_challenge(spec, recreate=False)

    def _select(self, spec: ChallengeSpec) -> tuple[str | None, Any]:
        assigned = self._assignment_for(spec.slug)
        if assigned and self._target_eligible(assigned, spec, exclude_slug=spec.slug):
            return assigned, self._orchestrator_for_target(assigned)
        if assigned:
            self._clear_assignment(spec.slug)
        requested = spec.resources.gpu_server
        if requested:
            if not self._target_eligible(requested, spec, exclude_slug=spec.slug):
                raise DockerOrchestrationError(
                    f"No valid Kubernetes target available for challenge {spec.slug!r}"
                )
            return requested, self._orchestrator_for_target(requested)
        if spec.resources.gpu_count:
            for target_id, orchestrator in self._target_orchestrators().items():
                if self._target_eligible(target_id, spec, exclude_slug=spec.slug):
                    return target_id, orchestrator
            raise DockerOrchestrationError(
                f"No valid Kubernetes target available for challenge {spec.slug!r}"
            )
        return None, self.default_orchestrator

    def _target_orchestrators(self) -> dict[str, Any]:
        if self.target_registry is None or self.settings is None:
            return self.target_orchestrators
        return self._build_targets()[0]

    def _target_capacities(self) -> dict[str, int]:
        if self.target_registry is None or self.settings is None:
            return self.target_capacities
        return self._build_targets()[1]

    def _orchestrator_for_target(self, target_id: str) -> Any:
        orchestrator = self._target_orchestrators().get(target_id)
        if (
            orchestrator is None
            and self.target_registry is not None
            and self.settings is not None
        ):
            orchestrator = self._build_target_orchestrator(
                self.target_registry.get(target_id)
            )
        if orchestrator is None:
            raise DockerOrchestrationError(f"Unknown Kubernetes target: {target_id}")
        return orchestrator

    def _assign(self, slug: str, target_id: str) -> None:
        gpu_count = self._slug_gpu_counts.get(slug, 0)
        if self.target_registry is not None and hasattr(
            self.target_registry, "assign_challenge"
        ):
            try:
                self.target_registry.assign_challenge(slug, target_id, gpu_count)
            except TypeError:
                self.target_registry.assign_challenge(slug, target_id)
        self._slug_gpu_counts[slug] = gpu_count

    def _assignment_for(self, slug: str) -> str | None:
        if self.target_registry is not None and hasattr(
            self.target_registry, "get_assignment"
        ):
            return self.target_registry.get_assignment(slug)
        return self._slug_to_target.get(slug)

    def _clear_assignment(self, slug: str) -> None:
        if self.target_registry is not None and hasattr(
            self.target_registry, "clear_assignment"
        ):
            self.target_registry.clear_assignment(slug)
        self._slug_to_target.pop(slug, None)
        self._slug_gpu_counts.pop(slug, None)

    def _build_targets(self) -> tuple[dict[str, Any], dict[str, int]]:
        assert self.settings is not None
        assert self.target_registry is not None
        target_orchestrators: dict[str, Any] = {}
        target_capacities: dict[str, int] = {}
        for target in self.target_registry.list():
            if not target.enabled or getattr(target, "draining", False):
                continue
            target_orchestrators[target.id] = self._build_target_orchestrator(target)
            target_capacities[target.id] = target.gpu_count
        return target_orchestrators, target_capacities

    def _target_eligible(
        self, target_id: str, spec: ChallengeSpec, *, exclude_slug: str | None = None
    ) -> bool:
        try:
            target = self._target_record(target_id)
        except Exception:
            return False
        if target is not None:
            if not getattr(target, "enabled", True) or getattr(
                target, "draining", False
            ):
                return False
        if self.target_registry is not None and hasattr(self.target_registry, "health"):
            try:
                health = self.target_registry.health(target_id)
            except Exception:
                return False
            if getattr(health, "status", None) != "ok":
                return False
        requested_gpu = int(spec.resources.gpu_count or 0)
        if requested_gpu <= 0:
            return True
        capacity = self._target_capacities().get(target_id, 0)
        return capacity >= requested_gpu + self._assigned_gpu_count(
            target_id, exclude_slug=exclude_slug
        )

    def _target_record(self, target_id: str) -> Any | None:
        if self.target_registry is not None and hasattr(self.target_registry, "get"):
            return self.target_registry.get(target_id)
        if target_id in self._target_orchestrators():
            return None
        raise DockerOrchestrationError(f"Unknown Kubernetes target: {target_id}")

    def _assigned_gpu_count(
        self, target_id: str, *, exclude_slug: str | None = None
    ) -> int:
        total = 0
        for slug, assigned in self._assignments_snapshot().items():
            if slug == exclude_slug or assigned != target_id:
                continue
            total += self._assignment_gpu_count(slug)
        return total

    def _assignments_snapshot(self) -> dict[str, str]:
        if self.target_registry is not None and hasattr(
            self.target_registry, "assignments"
        ):
            assignments = self.target_registry.assignments
            if callable(assignments):
                return assignments()
            return dict(assignments)
        return dict(self._slug_to_target)

    def _assignment_gpu_count(self, slug: str) -> int:
        if self.target_registry is not None and hasattr(
            self.target_registry, "get_assignment_metadata"
        ):
            metadata = self.target_registry.get_assignment_metadata(slug) or {}
            return int(metadata.get("gpu_count") or 0)
        return self._slug_gpu_counts.get(slug, 0)

    def _build_target_orchestrator(self, target: Any) -> Any:
        assert self.settings is not None
        assert self.target_registry is not None
        if target.mode == "agent":
            from platform_network.kubernetes.agent import KubernetesAgentClient

            token = self.target_registry.get_agent_token(target.id)
            if not target.agent_url or not token:
                raise DockerOrchestrationError(
                    f"Kubernetes agent target {target.id!r} is missing URL or token"
                )
            return KubernetesAgentClient(
                target_id=target.id,
                base_url=target.agent_url,
                token=token,
                timeout_seconds=target.timeout_seconds,
                verify_tls=target.verify_tls,
                docker_broker_url=self.settings.docker.broker_url,
            )
        defaults = self.settings.kubernetes.target_defaults
        production_policy = production_policy_enabled_for_settings(self.settings)
        return KubernetesOrchestrator(
            namespace=target.namespace,
            mode=self.settings.kubernetes.challenge_mode,
            storage_class_name=target.storage_class
            or self.settings.kubernetes.storage_class,
            storage_size=self.settings.kubernetes.storage_size,
            gpu_resource_name=defaults.gpu_resource_name
            or self.settings.kubernetes.gpu_resource_name,
            node_selector={
                **self.settings.kubernetes.node_selector,
                **defaults.node_selector,
                **target.node_selector,
            },
            tolerations=target.tolerations
            or defaults.tolerations
            or self.settings.kubernetes.tolerations,
            runtime_class_name=target.runtime_class_name
            or defaults.runtime_class_name
            or self.settings.kubernetes.runtime_class_name,
            image_pull_secrets=defaults.image_pull_secrets
            or self.settings.kubernetes.image_pull_secrets,
            autoscaling_enabled=self.settings.kubernetes.autoscaling.enabled,
            autoscaling_keda_enabled=self.settings.kubernetes.autoscaling.keda_enabled,
            autoscaling_min_replicas=self.settings.kubernetes.autoscaling.min_replicas,
            autoscaling_max_replicas=self.settings.kubernetes.autoscaling.max_replicas,
            autoscaling_target_cpu_utilization=(
                self.settings.kubernetes.autoscaling.target_cpu_utilization
            ),
            docker_broker_url=self.settings.docker.broker_url,
            health_check_mode="service_proxy",
            kubeconfig=target.kubeconfig_file,
            in_cluster=False,
            production_policy=production_policy,
            managed_postgres_enabled=self.settings.kubernetes.managed_postgres.enabled,
            managed_postgres_image=self.settings.kubernetes.managed_postgres.image,
            managed_postgres_storage_class_name=(
                self.settings.kubernetes.managed_postgres.storage_class
                or target.storage_class
                or self.settings.kubernetes.storage_class
            ),
            managed_postgres_storage_size=(
                self.settings.kubernetes.managed_postgres.storage_size
            ),
            managed_postgres_retain_pvc=(
                self.settings.kubernetes.managed_postgres.retain_pvc
            ),
            managed_postgres_retain_secret=(
                self.settings.kubernetes.managed_postgres.retain_secret
            ),
            managed_postgres_resources=(
                _managed_postgres_resources_from_settings(
                    self.settings.kubernetes.managed_postgres.resources
                )
            ),
        )

    @classmethod
    def from_settings(
        cls, settings: Any, target_registry: Any
    ) -> KubernetesTargetRouter:
        return cls(
            default_orchestrator=KubernetesOrchestrator.from_settings(settings),
            settings=settings,
            target_registry=target_registry,
        )


def kubeconfig_from_file(path: str | Path | None) -> str | None:
    return None if path is None else str(path)
