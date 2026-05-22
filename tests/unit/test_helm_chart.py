from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from platform_network.config.settings import Settings

CHART = Path(__file__).resolve().parents[2] / "deploy" / "helm" / "platform"
PRODUCTION_VALUES = CHART / "values.production.example.yaml"


def test_helm_template_renders_secure_kubernetes_control_plane() -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    rendered = subprocess.check_output(
        [helm, "template", "platform", str(CHART)],
        text=True,
    )
    documents = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]
    kinds = {doc["kind"] for doc in documents}

    assert {"Deployment", "Role", "RoleBinding", "HorizontalPodAutoscaler"}.issubset(
        kinds
    )
    assert "NetworkPolicy" in kinds
    assert "docker.sock" not in rendered
    assert "privileged: true" not in rendered
    assert _role_resources(documents, "autoscaling") == {"horizontalpodautoscalers"}
    assert _role_resources(documents, "networking.k8s.io") == {"networkpolicies"}
    assert _role_resources(documents, "keda.sh") == set()

    for container in _containers(documents):
        security = container.get("securityContext", {})
        assert security.get("allowPrivilegeEscalation") is False
        assert security.get("privileged") is False
        assert security.get("capabilities", {}).get("drop") == ["ALL"]
    for pod_spec in _pod_specs(documents):
        security = pod_spec.get("securityContext", {})
        assert security.get("runAsNonRoot") is True
        assert security.get("runAsUser") == 1000
        assert security.get("runAsGroup") == 1000
        assert security.get("fsGroup") == 1000


def test_helm_template_switches_from_hpa_to_keda_scaledobjects() -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    rendered = subprocess.check_output(
        [helm, "template", "platform", str(CHART), "--set", "keda.enabled=true"],
        text=True,
    )
    documents = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]
    kinds = [doc["kind"] for doc in documents]

    assert "ScaledObject" in kinds
    assert "HorizontalPodAutoscaler" not in kinds
    assert _role_resources(documents, "keda.sh") == {"scaledobjects"}


def test_helm_template_renders_validator_registry_url_default_and_override(
    tmp_path: Path,
) -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    override_file = tmp_path / "override.yaml"
    override_file.write_text(
        textwrap.dedent(
            """
            validator:
              registryUrl: https://registry.override.test
            network:
              walletPath: /var/lib/platform/wallets
            """
        ),
        encoding="utf-8",
    )

    cases = [
        ([], "https://chain.platform.network", ""),
        (
            ["-f", str(override_file)],
            "https://registry.override.test",
            "/var/lib/platform/wallets",
        ),
    ]
    for args, expected, expected_wallet_path in cases:
        rendered = subprocess.check_output(
            [helm, "template", "platform", str(CHART), *args],
            text=True,
        )
        documents = [
            doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)
        ]
        config = yaml.safe_load(
            _document(documents, "ConfigMap", "platform-config")["data"]["master.yaml"]
        )

        assert config["validator"]["registry_url"] == expected
        assert config["network"]["wallet_path"] == expected_wallet_path
        assert config["master"]["registry_url"] == "http://platform-admin:8000"


def test_helm_validator_deployment_uses_configured_image_pull_policy() -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    rendered = subprocess.check_output(
        [helm, "template", "platform", str(CHART), "--set", "image.pullPolicy=Always"],
        text=True,
    )
    documents = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]
    validator = _document(documents, "Deployment", "platform-validator")
    container = validator["spec"]["template"]["spec"]["containers"][0]

    assert container["imagePullPolicy"] == "Always"


def test_helm_mutable_auto_update_renders_master_and_validator_latest_images() -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    rendered = subprocess.check_output(
        [helm, "template", "platform", str(CHART)],
        text=True,
    )
    documents = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]

    for name, container_name in (
        ("platform-admin", "admin"),
        ("platform-proxy", "proxy"),
        ("platform-broker", "broker"),
    ):
        deployment = _document(documents, "Deployment", name)
        container = _named_container(_pod_spec(deployment), container_name)
        assert container["image"] == "ghcr.io/platformnetwork/platform-master:latest"
        assert container["imagePullPolicy"] == "Always"

    validator = _document(documents, "Deployment", "platform-validator")
    validator_container = _named_container(_pod_spec(validator), "validator")
    assert validator_container["image"] == "ghcr.io/platformnetwork/platform:latest"
    assert validator_container["imagePullPolicy"] == "Always"

    weights = _document(documents, "CronJob", "platform-weights")
    weights_container = _named_container(_pod_spec(weights), "weights")
    assert (
        weights_container["image"] == "ghcr.io/platformnetwork/platform-master:latest"
    )
    assert weights_container["imagePullPolicy"] == "Always"


def test_helm_renders_one_minute_image_updaters_for_master_and_validator() -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    rendered = subprocess.check_output(
        [helm, "template", "platform", str(CHART)],
        text=True,
    )
    documents = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]
    expected = {
        "platform-admin-image-updater": (
            "deployment",
            "platform-admin",
            "admin",
            "ghcr.io/platformnetwork/platform-master:latest",
        ),
        "platform-proxy-image-updater": (
            "deployment",
            "platform-proxy",
            "proxy",
            "ghcr.io/platformnetwork/platform-master:latest",
        ),
        "platform-broker-image-updater": (
            "deployment",
            "platform-broker",
            "broker",
            "ghcr.io/platformnetwork/platform-master:latest",
        ),
        "platform-weights-image-updater": (
            "cronjob",
            "platform-weights",
            "weights",
            "ghcr.io/platformnetwork/platform-master:latest",
        ),
        "platform-validator-image-updater": (
            "deployment",
            "platform-validator",
            "validator",
            "ghcr.io/platformnetwork/platform:latest",
        ),
    }

    for cronjob_name, (kind, resource, container_name, image) in expected.items():
        cronjob = _document(documents, "CronJob", cronjob_name)
        pod_spec = _pod_spec(cronjob)
        assert pod_spec is not None
        updater = _named_container(pod_spec, "image-updater")

        assert cronjob["spec"]["schedule"] == "*/1 * * * *"
        assert cronjob["spec"]["concurrencyPolicy"] == "Forbid"
        assert pod_spec["serviceAccountName"] == "platform-image-updater"
        assert pod_spec["restartPolicy"] == "OnFailure"
        assert updater["image"] == "ghcr.io/platformnetwork/platform:latest"
        assert updater["imagePullPolicy"] == "Always"
        assert updater["command"] == [
            "platform",
            "validator",
            "refresh-image",
            "--namespace",
            "default",
            "--resource-kind",
            kind,
            "--name",
            resource,
            "--container",
            container_name,
            "--image",
            image,
            "--registry-endpoint",
            "",
        ]


def test_helm_image_updater_rbac_is_namespace_scoped() -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    rendered = subprocess.check_output(
        [helm, "template", "platform", str(CHART)],
        text=True,
    )
    documents = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]
    service_account = _document(documents, "ServiceAccount", "platform-image-updater")
    role = _document(documents, "Role", "platform-image-updater")
    role_binding = _document(documents, "RoleBinding", "platform-image-updater")

    assert service_account["automountServiceAccountToken"] is True
    assert role_binding["subjects"] == [
        {"kind": "ServiceAccount", "name": "platform-image-updater"}
    ]
    assert "ClusterRole" not in {doc["kind"] for doc in documents}
    assert "ClusterRoleBinding" not in {doc["kind"] for doc in documents}
    rules = role["rules"]
    deployment_rule = next(
        rule for rule in rules if rule["resources"] == ["deployments"]
    )
    cronjob_rule = next(rule for rule in rules if rule["resources"] == ["cronjobs"])
    pods_rule = next(rule for rule in rules if rule["resources"] == ["pods"])

    assert set(deployment_rule["resourceNames"]) == {
        "platform-admin",
        "platform-proxy",
        "platform-broker",
        "platform-validator",
    }
    assert deployment_rule["verbs"] == ["get", "patch"]
    assert cronjob_rule["resourceNames"] == ["platform-weights"]
    assert cronjob_rule["verbs"] == ["get", "patch"]
    assert pods_rule["verbs"] == ["get", "list"]


def test_helm_template_renders_target_gpu_and_remote_agent_security_values(
    tmp_path: Path,
) -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        textwrap.dedent(
            """
            image:
              pullSecrets:
                - ghcr-auth
            kubernetes:
              runtimeClassName: nvidia
              nodeSelector:
                accelerator: nvidia
              tolerations:
                - key: nvidia.com/gpu
                  operator: Exists
                  effect: NoSchedule
              targetDefaults:
                imagePullSecrets:
                  - remote-ghcr-auth
                gpuResourceName: nvidia.com/gpu
                runtimeClassName: nvidia
                nodeSelector:
                  accelerator: nvidia
                tolerations:
                  - key: nvidia.com/gpu
                    operator: Exists
            kubernetesTargets:
              enabled: true
              targets:
                - id: gpu-a
                  mode: agent
                  agent_url: https://gpu-a.example.test
                  namespace: platform
                  service_account: platform-target
                  gpu_count: 4
                  agent_token_file: /var/lib/platform/secrets/gpu-a-agent-token
                  labels:
                    pool: gpu
                  enabled: true
                  verify_tls: true
            remoteAgents:
              enabled: true
              networkPolicy:
                egressCIDRs:
                  - 10.10.0.0/24
                ports:
                  - 8443
            networkPolicy:
              egress:
                allowAll: false
            """
        ),
        encoding="utf-8",
    )

    rendered = subprocess.check_output(
        [helm, "template", "platform", str(CHART), "-f", str(values_file)],
        text=True,
    )
    documents = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]
    config = yaml.safe_load(
        _document(documents, "ConfigMap", "platform-config")["data"]["master.yaml"]
    )
    config["database"]["url"] = "postgresql+asyncpg://db.example.test/platform"
    config["docker"]["broker_allowed_images"] = ["ghcr.io/platformnetwork/"]
    config["database"]["url"] = "postgresql+asyncpg://user:pass@postgres/platform"
    Settings.model_validate(config)
    network_policy = _document(documents, "NetworkPolicy", "platform-control-plane")

    assert config["kubernetes"]["runtime_class_name"] == "nvidia"
    assert config["kubernetes"]["node_selector"] == {"accelerator": "nvidia"}
    assert config["kubernetes"]["target_defaults"]["image_pull_secrets"] == [
        "remote-ghcr-auth"
    ]
    assert config["kubernetes_targets"][0]["id"] == "gpu-a"
    assert config["kubernetes_targets"][0]["agent_url"] == "https://gpu-a.example.test"
    assert config["kubernetes_targets"][0]["gpu_count"] == 4

    ingress_peers = network_policy["spec"]["ingress"][0]["from"]
    assert {"namespaceSelector": {}} not in ingress_peers
    assert any("matchExpressions" in peer["podSelector"] for peer in ingress_peers)
    assert network_policy["spec"]["egress"] == [
        {
            "to": [{"ipBlock": {"cidr": "10.10.0.0/24"}}],
            "ports": [{"protocol": "TCP", "port": 8443}],
        }
    ]


def test_helm_production_values_render_safe_control_plane() -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    rendered = subprocess.check_output(
        [helm, "template", "platform", str(CHART), "-f", str(PRODUCTION_VALUES)],
        text=True,
    )
    documents = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]
    rendered_lower = rendered.lower()
    images = [container["image"] for container in _containers(documents)]
    network_policy = _document(documents, "NetworkPolicy", "platform-control-plane")
    service_account = _document(documents, "ServiceAccount", "platform-master")
    role_binding = _document(documents, "RoleBinding", "platform-runtime")
    role = _document(documents, "Role", "platform-runtime")
    pod_specs = _pod_specs(documents)

    assert "sqlite" not in rendered_lower
    assert ":latest" not in rendered_lower
    assert "image-updater" not in rendered
    assert "imagePullSecrets" not in rendered
    assert (
        "sha256:1111111111111111111111111111111111111111111111111111111111111111"
        in rendered
    )
    assert all("@sha256:" in image for image in images)
    assert _database_secret_refs(documents) == {("platform-production-postgres", "url")}
    assert service_account["automountServiceAccountToken"] is True
    assert role_binding["subjects"] == [
        {"kind": "ServiceAccount", "name": "platform-master"}
    ]
    assert "ClusterRole" not in {doc["kind"] for doc in documents}
    assert "ClusterRoleBinding" not in {doc["kind"] for doc in documents}
    assert all("*" not in rule.get("resources", []) for rule in role["rules"])
    assert all("*" not in rule.get("verbs", []) for rule in role["rules"])
    assert network_policy["spec"]["egress"] != [{}]
    assert network_policy["spec"]["egress"]
    assert any(
        {"protocol": "UDP", "port": 53} in rule.get("ports", [])
        for rule in network_policy["spec"]["egress"]
    )
    assert any(
        {"protocol": "TCP", "port": 5432} in rule.get("ports", [])
        for rule in network_policy["spec"]["egress"]
    )
    assert any(
        {"protocol": "TCP", "port": 443} in rule.get("ports", [])
        for rule in network_policy["spec"]["egress"]
    )
    assert {"namespaceSelector": {}} not in network_policy["spec"]["ingress"][0]["from"]
    for pod_spec in pod_specs:
        security = pod_spec.get("securityContext", {})
        assert security.get("runAsNonRoot") is True
        assert security.get("runAsUser") == 1000
        assert security.get("runAsGroup") == 1000
        assert security.get("fsGroup") == 1000
    for container in _containers(documents):
        resources = container.get("resources", {})
        assert resources.get("requests", {}).get("cpu")
        assert resources.get("requests", {}).get("memory")
        assert resources.get("limits", {}).get("cpu")
        assert resources.get("limits", {}).get("memory")


@pytest.mark.parametrize(
    "override, expected",
    [
        (
            """
            image:
              tag: latest
            """,
            "/image/tag",
        ),
        (
            """
            image:
              digest: ""
            """,
            "digest",
        ),
        (
            """
            networkPolicy:
              egress:
                allowAll: true
            """,
            "allowAll",
        ),
        (
            """
            image:
              tag: canary
            """,
            "/image/tag",
        ),
    ],
)
def test_helm_production_schema_rejects_unsafe_values(
    tmp_path: Path, override: str, expected: str
) -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    values_file = tmp_path / "unsafe.yaml"
    values_file.write_text(textwrap.dedent(override), encoding="utf-8")

    result = subprocess.run(
        [
            helm,
            "template",
            "platform",
            str(CHART),
            "-f",
            str(PRODUCTION_VALUES),
            "-f",
            str(values_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert expected in result.stderr


def _containers(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    for doc in documents:
        pod_spec = _pod_spec(doc)
        if pod_spec:
            containers.extend(pod_spec.get("containers", []))
    return containers


def _pod_specs(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [pod_spec for doc in documents if (pod_spec := _pod_spec(doc))]


def _database_secret_refs(documents: list[dict[str, Any]]) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = set()
    for container in _containers(documents):
        for env in container.get("env", []):
            secret_ref = env.get("valueFrom", {}).get("secretKeyRef")
            if secret_ref:
                refs.add((secret_ref["name"], secret_ref["key"]))
    return refs


def _pod_spec(doc: dict[str, Any]) -> dict[str, Any] | None:
    kind = doc.get("kind")
    if kind == "Deployment":
        return doc["spec"]["template"]["spec"]
    if kind == "CronJob":
        return doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    return None


def _named_container(pod_spec: dict[str, Any] | None, name: str) -> dict[str, Any]:
    assert pod_spec is not None
    for container in pod_spec.get("containers", []):
        if container.get("name") == name:
            return container
    raise AssertionError(f"container {name!r} not rendered")


def _document(documents: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    for doc in documents:
        if doc.get("kind") == kind and doc.get("metadata", {}).get("name") == name:
            return doc
    raise AssertionError(f"{kind}/{name} not rendered")


def _role_resources(documents: list[dict[str, Any]], api_group: str) -> set[str]:
    resources: set[str] = set()
    for doc in documents:
        if doc.get("kind") != "Role":
            continue
        for rule in doc.get("rules", []):
            if api_group in rule.get("apiGroups", []):
                resources.update(rule.get("resources", []))
    return resources


def test_helm_production_policy_rejects_unsafe_values(tmp_path: Path) -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    cases = [
        ("latest.yaml", "image:\n  tag: latest\n", "/image/tag"),
        ("missing-digest.yaml", "image:\n  digest: ''\n", "/image/digest"),
        (
            "verify-tls.yaml",
            textwrap.dedent(
                """
                kubernetesTargets:
                  enabled: true
                  targets:
                    - id: gpu-a
                      mode: agent
                      agent_url: https://gpu-a.example.test
                      verify_tls: false
                """
            ),
            "verify_tls=true",
        ),
        (
            "mutable-autoupdate.yaml",
            "imageAutoUpdate:\n  enabled: true\n",
            "imageAutoUpdate.enabled=true",
        ),
        (
            "missing-db-secret.yaml",
            "database:\n  urlSecret:\n    name: ''\n    key: url\n",
            "/database/urlSecret/name",
        ),
    ]

    for filename, values, message in cases:
        values_file = tmp_path / filename
        values_file.write_text(values, encoding="utf-8")
        result = subprocess.run(
            [
                helm,
                "template",
                "platform",
                str(CHART),
                "-f",
                str(PRODUCTION_VALUES),
                "-f",
                str(values_file),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert message in result.stderr


def test_helm_policy_can_be_disabled_for_local_dev_images(tmp_path: Path) -> None:
    helm = shutil.which("helm")
    if helm is None:
        pytest.skip("helm is not installed")

    values_file = tmp_path / "local.yaml"
    values_file.write_text(
        textwrap.dedent(
            """
            policy:
              enforceProduction: false
            image:
              repository: localhost:5000/platform
              tag: latest
              digest: ''
            images:
              master:
                repository: localhost:5000/platform-master
                tag: latest
                digest: ''
                pullPolicy: Always
              validator:
                repository: localhost:5000/platform
                tag: latest
                digest: ''
                pullPolicy: Always
              updater:
                repository: localhost:5000/platform
                tag: latest
                digest: ''
                pullPolicy: Always
            """
        ),
        encoding="utf-8",
    )

    rendered = subprocess.check_output(
        [helm, "template", "platform", str(CHART), "-f", str(values_file)],
        text=True,
    )

    assert "localhost:5000/platform:latest" in rendered
