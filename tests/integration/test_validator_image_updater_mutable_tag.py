from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = str(os.getpid())
REGISTRY_NAME = f"platform-kind-registry-{RUN_ID}"
REGISTRY_PORT = "5001"
CLUSTER = f"platform-validator-mutable-tag-{RUN_ID}"
PUSH_IMAGE = f"localhost:{REGISTRY_PORT}/platform-validator-mutable:latest"
CLUSTER_IMAGE = PUSH_IMAGE
UPDATER_PUSH_IMAGE = f"localhost:{REGISTRY_PORT}/platform-updater:latest"
UPDATER_CLUSTER_IMAGE = UPDATER_PUSH_IMAGE
NAMESPACE = "platform-validator-mutable"


def _run(cmd: list[str], *, input_text: str | None = None, timeout: int = 120) -> str:
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def _tool(name: str) -> None:
    if shutil.which(name) is None:
        pytest.skip(f"{name} is not installed")


def _docker(*args: str, timeout: int = 120) -> str:
    return _run(["docker", *args], timeout=timeout)


def _kubectl(*args: str, timeout: int = 120) -> str:
    return _run(["kubectl", "--context", f"kind-{CLUSTER}", *args], timeout=timeout)


def _ensure_registry() -> None:
    names = _docker("ps", "-a", "--format", "{{.Names}}")
    if REGISTRY_NAME in names.splitlines():
        _docker("rm", "-f", REGISTRY_NAME)
    _docker(
        "run",
        "-d",
        "--restart=always",
        "-p",
        f"{REGISTRY_PORT}:5000",
        "--name",
        REGISTRY_NAME,
        "registry:2",
    )


def _create_cluster() -> None:
    config = f"""kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
  - |-
    [plugins."io.containerd.grpc.v1.cri".registry.mirrors."localhost:{REGISTRY_PORT}"]
      endpoint = ["http://{REGISTRY_NAME}:5000"]
"""
    _run(["kind", "delete", "cluster", "--name", CLUSTER], timeout=120)
    _run(
        ["kind", "create", "cluster", "--name", CLUSTER, "--config", "-"],
        input_text=config,
        timeout=180,
    )
    networks = _docker(
        "inspect",
        "-f",
        "{{json .NetworkSettings.Networks}}",
        REGISTRY_NAME,
    )
    if '"kind"' not in networks:
        _docker("network", "connect", "kind", REGISTRY_NAME)


def _build_and_push(tag_marker: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dockerfile = Path(tmp) / "Dockerfile"
        dockerfile.write_text(
            "FROM busybox:1.36\n"
            f"LABEL platform.mutable-tag-test={tag_marker}\n"
            'CMD ["sh", "-c", "sleep 3600"]\n',
            encoding="utf-8",
        )
        _docker("build", "-t", PUSH_IMAGE, tmp, timeout=180)
    _docker("push", PUSH_IMAGE, timeout=180)


def _build_and_push_updater() -> None:
    _docker(
        "build",
        "-f",
        "docker/Dockerfile.validator",
        "-t",
        UPDATER_PUSH_IMAGE,
        ".",
        timeout=300,
    )
    _docker("push", UPDATER_PUSH_IMAGE, timeout=180)


def _generate_mnemonic() -> str:
    return _run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "import bittensor; print(bittensor.Keypair.generate_mnemonic())",
        ]
    ).strip()


def _validator_image_id(previous_uid: str | None = None) -> tuple[str, str]:
    deadline = time.time() + 180
    while time.time() < deadline:
        pods = json.loads(
            _kubectl(
                "-n",
                NAMESPACE,
                "get",
                "pods",
                "-l",
                "platform.component=validator",
                "-o",
                "json",
            )
        ).get("items", [])
        for pod in pods:
            uid = pod["metadata"]["uid"]
            if previous_uid is not None and uid == previous_uid:
                continue
            statuses = pod.get("status", {}).get("containerStatuses", [])
            if statuses and statuses[0].get("imageID"):
                return uid, statuses[0]["imageID"]
        time.sleep(2)
    raise AssertionError("validator pod imageID was not observed")


def _run_updater_job(name: str) -> None:
    _kubectl(
        "-n",
        NAMESPACE,
        "create",
        "job",
        "--from=cronjob/platform-validator-image-updater",
        name,
    )
    try:
        _kubectl(
            "-n",
            NAMESPACE,
            "wait",
            "--for=condition=complete",
            f"job/{name}",
            "--timeout=180s",
            timeout=240,
        )
    except AssertionError as exc:
        pods = _kubectl("-n", NAMESPACE, "get", "pods", "-o", "wide")
        describe = _kubectl("-n", NAMESPACE, "describe", f"job/{name}")
        logs = _kubectl("-n", NAMESPACE, "logs", f"job/{name}", timeout=60)
        raise AssertionError(
            f"{exc}\nPods:\n{pods}\nDescribe:\n{describe}\nLogs:\n{logs}"
        ) from exc


def _assert_no_replacement_pod(uid: str) -> None:
    time.sleep(10)
    current_uid, _ = _validator_image_id()
    assert current_uid == uid


def _deployment_digest_annotation() -> str:
    return _kubectl(
        "-n",
        NAMESPACE,
        "get",
        "deployment/platform-validator",
        "-o",
        r"jsonpath={.metadata.annotations.platform\.network/image-digest}",
    ).strip()


def test_validator_updater_repulls_changed_mutable_tag() -> None:
    if os.environ.get("PLATFORM_RUN_KIND_MUTABLE_TAG_TEST") != "1":
        pytest.skip(
            "set PLATFORM_RUN_KIND_MUTABLE_TAG_TEST=1 to run kind mutable tag test"
        )
    for tool in ["docker", "kind", "kubectl", "uv"]:
        _tool(tool)

    _ensure_registry()
    try:
        _create_cluster()
        try:
            _build_and_push("a")
            _build_and_push_updater()
            mnemonic = _generate_mnemonic()
            _run(
                [
                    "bash",
                    "scripts/install-validator.sh",
                    "--namespace",
                    NAMESPACE,
                    "--image",
                    CLUSTER_IMAGE,
                    "--auto-update-image",
                    UPDATER_CLUSTER_IMAGE,
                    "--auto-update-registry-endpoint",
                    f"http://{REGISTRY_NAME}:5000",
                    "--auto-update-schedule",
                    "*/1 * * * *",
                ],
                input_text=f"{mnemonic}\n",
                timeout=180,
            )
            first_uid, first_image_id = _validator_image_id()

            _run_updater_job("updater-initial")
            _assert_no_replacement_pod(first_uid)
            first_digest = _deployment_digest_annotation()

            _build_and_push("b")
            _run_updater_job("updater-change")
            second_uid, second_image_id = _validator_image_id(previous_uid=first_uid)

            assert second_uid != first_uid
            assert first_image_id != second_image_id
            assert _deployment_digest_annotation() != first_digest

            _run_updater_job("updater-current")
            _assert_no_replacement_pod(second_uid)
        finally:
            _run(["kind", "delete", "cluster", "--name", CLUSTER], timeout=120)
    finally:
        _docker("rm", "-f", REGISTRY_NAME)
