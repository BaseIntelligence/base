from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from platform_network.kubernetes.config_updater import (
    CONFIG_DIGEST_ANNOTATION,
    ConfigSyncSource,
    ConfigSyncUpdater,
    RolloutTarget,
    _runtime_config_payload,
)

ROOT = Path(__file__).resolve().parents[2]


class CoreApi:
    def __init__(self, config_map: SimpleNamespace) -> None:
        self.config_map = config_map
        self.config_map_patches: list[dict[str, Any]] = []
        self.secret_patches: list[dict[str, Any]] = []

    def read_namespaced_config_map(self, name: str, namespace: str) -> SimpleNamespace:
        assert name == "platform-config"
        assert namespace == "platform-test"
        return self.config_map

    def patch_namespaced_config_map(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> None:
        assert name == "platform-config"
        assert namespace == "platform-test"
        self.config_map_patches.append(body)
        annotations = self.config_map.metadata.annotations
        annotations.update(body.get("metadata", {}).get("annotations", {}))
        self.config_map.data.update(body.get("data", {}))

    def patch_namespaced_secret(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> None:
        self.secret_patches.append({"name": name, "namespace": namespace, "body": body})


class AppsApi:
    def __init__(self) -> None:
        self.deployment_patches: list[tuple[str, str, dict[str, Any]]] = []
        self.deployments: dict[str, SimpleNamespace] = {}

    def patch_namespaced_deployment(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> None:
        self.deployment_patches.append((name, namespace, body))
        deployment = self.deployments.setdefault(
            name,
            SimpleNamespace(
                spec=SimpleNamespace(
                    template=SimpleNamespace(metadata=SimpleNamespace(annotations={}))
                )
            ),
        )
        deployment.spec.template.metadata.annotations.update(
            body["spec"]["template"]["metadata"]["annotations"]
        )

    def read_namespaced_deployment(self, name: str, namespace: str) -> SimpleNamespace:
        return self.deployments.get(
            name,
            SimpleNamespace(
                spec=SimpleNamespace(
                    template=SimpleNamespace(metadata=SimpleNamespace(annotations={}))
                )
            ),
        )


class FailingOnceAppsApi(AppsApi):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_patch = True

    def patch_namespaced_deployment(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> None:
        if self.fail_next_patch:
            self.fail_next_patch = False
            raise RuntimeError("rollout patch failed")
        super().patch_namespaced_deployment(name, namespace, body)


def config_map(*, digest: str = "sha256:old") -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(annotations={CONFIG_DIGEST_ANNOTATION: digest}),
        data={"master.yaml": "environment: current\n"},
    )


def test_default_github_source_contract_uses_platform_main_without_secrets() -> None:
    source = ConfigSyncSource.default()

    assert source.repository == "PlatformNetwork/platform"
    assert source.branch == "main"
    assert source.sync_secrets is False
    assert "deploy/helm/platform/values.yaml" in source.paths
    assert "Secret" not in source.allowed_kinds


def test_invalid_github_yaml_preserves_current_state_and_skips_rollout() -> None:
    core = CoreApi(config_map(digest="sha256:current"))
    apps = AppsApi()
    updater = ConfigSyncUpdater(
        core_api=core,
        apps_api=apps,
        source=ConfigSyncSource.default(fetcher=lambda _: "master: [invalid"),
    )

    result = updater.sync_once(
        namespace="platform-test",
        config_map="platform-config",
        rollout_targets=[RolloutTarget(kind="Deployment", name="platform-validator")],
    )

    assert result.changed is False
    assert result.reason == "invalid_config"
    assert result.current_digest == "sha256:current"
    assert core.config_map_patches == []
    assert apps.deployment_patches == []


def test_secret_manifests_are_rejected_and_never_patched() -> None:
    source_text = """
    apiVersion: v1
    kind: Secret
    metadata:
      name: platform-secrets
    stringData:
      token: plaintext
    """
    core = CoreApi(config_map())
    apps = AppsApi()
    updater = ConfigSyncUpdater(
        core_api=core,
        apps_api=apps,
        source=ConfigSyncSource.default(fetcher=lambda _: source_text),
    )

    result = updater.sync_once(
        namespace="platform-test",
        config_map="platform-config",
        rollout_targets=[RolloutTarget(kind="Deployment", name="platform-validator")],
    )

    assert result.changed is False
    assert result.reason == "secret_sync_rejected"
    assert core.config_map_patches == []
    assert core.secret_patches == []
    assert apps.deployment_patches == []


def test_changed_config_patches_configmap_and_shared_rollout_annotation() -> None:
    source_text = """
    environment: production
    network:
      netuid: 42
    validator:
      registry_url: https://registry.example.test
    """
    core = CoreApi(config_map(digest="sha256:old"))
    apps = AppsApi()
    updater = ConfigSyncUpdater(
        core_api=core,
        apps_api=apps,
        source=ConfigSyncSource.default(fetcher=lambda _: source_text),
    )

    result = updater.sync_once(
        namespace="platform-test",
        config_map="platform-config",
        rollout_targets=[
            RolloutTarget(kind="Deployment", name="platform-admin"),
            RolloutTarget(kind="Deployment", name="platform-validator"),
        ],
    )

    assert result.changed is True
    assert result.current_digest == "sha256:old"
    assert result.new_digest is not None
    assert result.new_digest.startswith("sha256:")
    config_patch = core.config_map_patches[0]
    assert config_patch["data"]["master.yaml"] == source_text
    assert (
        config_patch["metadata"]["annotations"][CONFIG_DIGEST_ANNOTATION]
        == result.new_digest
    )
    assert [name for name, _, _ in apps.deployment_patches] == [
        "platform-admin",
        "platform-validator",
    ]
    for _, namespace, patch in apps.deployment_patches:
        assert namespace == "platform-test"
        assert (
            patch["spec"]["template"]["metadata"]["annotations"][
                CONFIG_DIGEST_ANNOTATION
            ]
            == result.new_digest
        )


def test_same_digest_retries_rollout_after_partial_failure() -> None:
    source_text = "environment: production\n"
    core = CoreApi(config_map(digest="sha256:old"))
    apps = FailingOnceAppsApi()
    updater = ConfigSyncUpdater(
        core_api=core,
        apps_api=apps,
        source=ConfigSyncSource.default(fetcher=lambda _: source_text),
    )

    with pytest.raises(RuntimeError, match="rollout patch failed"):
        updater.sync_once(
            namespace="platform-test",
            config_map="platform-config",
            rollout_targets=[RolloutTarget(kind="Deployment", name="platform-admin")],
        )

    assert len(core.config_map_patches) == 1
    digest = core.config_map.metadata.annotations[CONFIG_DIGEST_ANNOTATION]
    assert apps.deployment_patches == []

    result = updater.sync_once(
        namespace="platform-test",
        config_map="platform-config",
        rollout_targets=[RolloutTarget(kind="Deployment", name="platform-admin")],
    )

    assert result.changed is True
    assert result.reason == "rollout_retried"
    assert result.current_digest == digest
    assert result.new_digest == digest
    assert len(core.config_map_patches) == 1
    assert [name for name, _, _ in apps.deployment_patches] == ["platform-admin"]


def test_config_sync_extracts_runtime_config_from_rendered_configmap_manifest() -> None:
    source_text = """
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: platform-config
    data:
      master.yaml: |
        environment: production
        runtime:
          backend: kubernetes
    """
    core = CoreApi(config_map(digest="sha256:old"))
    updater = ConfigSyncUpdater(
        core_api=core,
        apps_api=AppsApi(),
        source=ConfigSyncSource.default(fetcher=lambda _: source_text),
    )

    updater.sync_once(
        namespace="platform-test",
        config_map="platform-config",
        rollout_targets=[],
    )

    assert core.config_map_patches[0]["data"]["master.yaml"] == (
        "environment: production\nruntime:\n  backend: kubernetes\n"
    )


def test_validator_helm_values_runtime_config_uses_validator_service_account() -> None:
    values = (ROOT / "deploy/helm/platform/values.yaml").read_text()

    payload = _runtime_config_payload(
        values,
        config_map="platform-validator-config",
        namespace="platform-validator",
    )

    config = yaml.safe_load(payload)
    assert config["kubernetes"]["namespace"] == "platform-validator"
    assert config["kubernetes"]["service_account"] == "platform-validator"


def test_unknown_rollout_kind_is_rejected_without_patching_configmap() -> None:
    core = CoreApi(config_map())
    updater = ConfigSyncUpdater(
        core_api=core,
        apps_api=AppsApi(),
        source=ConfigSyncSource.default(fetcher=lambda _: "environment: production\n"),
    )

    with pytest.raises(ValueError, match="unsupported rollout kind"):
        updater.sync_once(
            namespace="platform-test",
            config_map="platform-config",
            rollout_targets=[RolloutTarget(kind="StatefulSet", name="platform-db")],
        )

    assert core.config_map_patches == []
