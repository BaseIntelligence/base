from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from base.challenge_sdk.executors.docker import (
    DockerContainerInfo,
    DockerExecutor,
    DockerExecutorError,
    DockerLimits,
    DockerMount,
    DockerRunSpec,
    _encode_mount,
)


def test_build_run_command_has_security_flags(tmp_path: Path) -> None:
    spec = DockerRunSpec(
        image="baseintelligence/swe-forge:task",
        command=("bash", "/workspace/forge/evaluate.sh"),
        mounts=(DockerMount(tmp_path, "/workspace/forge"),),
        workdir="/workspace/repo",
        labels={"base.job": "job-1", "base.task": "task-1"},
        limits=DockerLimits(cpus=1.5, memory="512m", pids_limit=64),
    )
    executor = DockerExecutor(
        challenge="agent", allowed_images=("baseintelligence/swe-forge:",)
    )

    cmd = executor.build_run_command(spec, "agent-job-task")

    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--network" in cmd and "none" in cmd
    assert "--cpus" in cmd and "1.5" in cmd
    assert "--memory" in cmd and "512m" in cmd
    assert "--pids-limit" in cmd and "64" in cmd
    assert "--cap-drop" in cmd and "ALL" in cmd
    assert "no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--init" in cmd
    assert "--memory-swap" in cmd and "512m" in cmd
    assert "--ulimit" in cmd and "nofile=1024:1024" in cmd
    assert "--label" in cmd and "base.challenge=agent" in cmd
    assert "base.challenge=evil" not in cmd
    assert f"{tmp_path.resolve()}:/workspace/forge:ro" in cmd
    assert cmd[-3:] == [
        "baseintelligence/swe-forge:task",
        "bash",
        "/workspace/forge/evaluate.sh",
    ]


def test_build_run_command_emits_gpus_when_gpu_requested(tmp_path: Path) -> None:
    spec = DockerRunSpec(
        image="baseintelligence/swe-forge:task",
        command=("true",),
        mounts=(DockerMount(tmp_path, "/workspace/forge"),),
        limits=DockerLimits(gpu_count=2),
    )
    cmd = DockerExecutor(
        challenge="agent", allowed_images=("baseintelligence/",)
    ).build_run_command(spec, "name")

    assert "--gpus" in cmd
    assert cmd[cmd.index("--gpus") + 1] == "2"


def test_build_run_command_omits_gpus_without_gpu_request(tmp_path: Path) -> None:
    spec = DockerRunSpec(
        image="baseintelligence/swe-forge:task",
        command=("true",),
        mounts=(DockerMount(tmp_path, "/workspace/forge"),),
    )
    cmd = DockerExecutor(
        challenge="agent", allowed_images=("baseintelligence/",)
    ).build_run_command(spec, "name")

    assert "--gpus" not in cmd


def test_docker_limits_default_to_hardened_runtime_controls() -> None:
    limits = DockerLimits(cpus=1, memory="512m", pids_limit=1)

    assert limits.init is True
    assert limits.read_only is True
    assert limits.cap_drop == ("ALL",)
    assert limits.security_opt == ("no-new-privileges",)


def test_docker_limits_gpu_count_default_and_positive_request() -> None:
    assert DockerLimits().gpu_count is None
    assert DockerLimits(gpu_count=1).gpu_count == 1


@pytest.mark.parametrize("gpu_count", [0, -1, True, "1", 1.5])
def test_docker_limits_reject_invalid_gpu_count(gpu_count: Any) -> None:
    with pytest.raises(DockerExecutorError, match="GPU count"):
        DockerLimits(gpu_count=gpu_count)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpus": 0},
        {"memory": ""},
        {"memory_swap": ""},
        {"pids_limit": 0},
        {"cap_drop": ()},
        {"security_opt": ()},
    ],
)
def test_docker_limits_reject_unsafe_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(DockerExecutorError):
        DockerLimits(**kwargs)


def test_reserved_labels_cannot_be_overridden(tmp_path: Path) -> None:
    spec = DockerRunSpec(
        image="baseintelligence/swe-forge:task",
        command=("true",),
        mounts=(DockerMount(tmp_path, "/workspace/forge"),),
        labels={"base.challenge": "evil", "base.job": "job-1"},
    )
    cmd = DockerExecutor(
        challenge="agent", allowed_images=("baseintelligence/",)
    ).build_run_command(spec, "name")

    assert "base.challenge=agent" in cmd
    assert "base.challenge=evil" not in cmd


@pytest.mark.parametrize(
    "image",
    ["-v", "bad image", "../../bad"],
)
def test_rejects_unsafe_image_refs(tmp_path: Path, image: str) -> None:
    spec = DockerRunSpec(
        image=image,
        command=("true",),
        mounts=(DockerMount(tmp_path, "/x"),),
    )
    with pytest.raises(DockerExecutorError):
        DockerExecutor(challenge="agent").build_run_command(spec, "name")


def test_rejects_images_outside_allowlist(tmp_path: Path) -> None:
    spec = DockerRunSpec(
        image="docker.io/library/python:latest@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        command=("true",),
        mounts=(DockerMount(tmp_path, "/x"),),
    )
    with pytest.raises(DockerExecutorError):
        DockerExecutor(challenge="agent", allowed_images=("baseintelligence/",)).run(
            spec, timeout_seconds=1
        )


def test_rejects_invalid_image_pull_policy_before_broker_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base.challenge_sdk.executors.docker as module

    called = False

    def fake_urlopen(request: object, timeout: int) -> object:
        nonlocal called
        called = True
        raise AssertionError("broker POST should not be attempted")

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    spec = DockerRunSpec(
        image="python:3.12-slim",
        command=("python", "-V"),
        image_pull_policy="Sometimes",
    )

    with pytest.raises(DockerExecutorError, match="image pull policy"):
        DockerExecutor(
            challenge="agent",
            backend="broker",
            broker_url="http://broker",
            broker_token="tok",
            allowed_images=("python:",),
        ).run(spec, timeout_seconds=20)
    assert called is False


def test_allows_default_network_for_broker_compatible_jobs(tmp_path: Path) -> None:
    spec = DockerRunSpec(
        image="baseintelligence/swe-forge:task",
        command=("true",),
        mounts=(DockerMount(tmp_path, "/x"),),
        limits=DockerLimits(network="default"),
    )

    cmd = DockerExecutor(
        challenge="agent", allowed_images=("baseintelligence/",)
    ).build_run_command(spec, "name")

    assert "--network" in cmd and "default" in cmd


def test_cleanup_job_uses_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        if cmd[:3] == ["docker", "ps", "-aq"]:
            return SimpleNamespace(stdout="abc\ndef\n", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    DockerExecutor(challenge="agent").cleanup_job("job-1")

    assert calls[0] == [
        "docker",
        "ps",
        "-aq",
        "--filter",
        "label=base.challenge=agent",
        "--filter",
        "label=base.job=job-1",
    ]
    assert calls[1] == ["docker", "rm", "-f", "abc", "def"]


def test_list_containers_uses_challenge_and_job_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "ID": "abc",
                    "Names": "agent-job",
                    "Image": "python:3.12",
                    "Status": "Up",
                    "CreatedAt": "now",
                    "Labels": "base.challenge=agent,base.job=job-1",
                }
            )
            + "\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    containers = DockerExecutor(challenge="agent").list_containers("job-1")

    assert calls[0] == [
        "docker",
        "ps",
        "-a",
        "--filter",
        "label=base.challenge=agent",
        "--filter",
        "label=base.job=job-1",
        "--format",
        "{{json .}}",
    ]
    assert containers == [
        DockerContainerInfo(
            container_id="abc",
            container_name="agent-job",
            image="python:3.12",
            status="Up",
            job_id="job-1",
            created="now",
            labels={"base.challenge": "agent", "base.job": "job-1"},
        )
    ]


def test_base_sdk_broker_backend_posts_run_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import base.challenge_sdk.executors.docker as module

    (tmp_path / "input.txt").write_text("ok", encoding="utf-8")
    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "container_name": "agent-job",
                    "stdout": "ok",
                    "stderr": "",
                    "returncode": 0,
                    "timed_out": False,
                }
            ).encode()

    def fake_urlopen(request: object, timeout: int) -> Response:
        captured["timeout"] = timeout
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["headers"] = dict(request.headers.items())  # type: ignore[attr-defined]
        captured["payload"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    result = DockerExecutor(
        challenge="agent",
        backend="broker",
        broker_url="http://broker",
        broker_token="tok",
        allowed_images=("python:",),
    ).run(
        DockerRunSpec(
            image="python:3.12-slim",
            command=("python", "-V"),
            workdir="/workspace/task",
            env={
                "BASE_RUNNER_MODE": "controlled",
                "BASE_TOKEN_FILE": "/var/run/secrets/base/token",
            },
            mounts=(DockerMount(tmp_path, "/mnt"),),
            labels={
                "base.job": "job-1",
                "base.task": "terminal-bench-1",
                "custom.label": "survives",
            },
            limits=DockerLimits(
                cpus=1.5,
                memory="768m",
                pids_limit=96,
                gpu_count=1,
            ),
            image_pull_policy="IfNotPresent",
        ),
        timeout_seconds=20,
    )

    assert result.returncode == 0
    assert captured["url"] == "http://broker/v1/docker/run"
    headers = cast(dict[str, str], captured["headers"])
    assert headers["Authorization"] == "Bearer tok"
    payload = cast(dict[str, Any], captured["payload"])
    assert payload["image"] == "python:3.12-slim"
    assert payload["command"] == ["python", "-V"]
    assert payload["workdir"] == "/workspace/task"
    assert payload["image_pull_policy"] == "IfNotPresent"
    assert payload["env"] == {
        "BASE_RUNNER_MODE": "controlled",
        "BASE_TOKEN_FILE": "/var/run/secrets/base/token",
    }
    assert payload["mounts"] == [
        {
            "target": "/mnt",
            "read_only": True,
            "source_type": "directory",
            "source_name": ".",
            "archive_b64": payload["mounts"][0]["archive_b64"],
        }
    ]
    assert payload["labels"] == {
        "base.job": "job-1",
        "base.task": "terminal-bench-1",
        "custom.label": "survives",
    }
    assert payload["limits"]["cpus"] == 1.5
    assert payload["limits"]["memory"] == "768m"
    assert payload["limits"]["pids_limit"] == 96
    assert payload["limits"]["gpu_count"] == 1
    assert payload["job_id"] == "job-1"
    assert payload["task_id"] == "terminal-bench-1"
    assert payload["timeout_seconds"] == 20
    assert "Bearer tok" not in json.dumps(payload)


def test_broker_backend_lists_containers(monkeypatch: pytest.MonkeyPatch) -> None:
    import base.challenge_sdk.executors.docker as module

    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "containers": [
                        {
                            "container_id": "abc",
                            "container_name": "agent-job",
                            "image": "python",
                            "status": "running",
                            "job_id": "job-1",
                            "labels": {"base.challenge": "agent"},
                        }
                    ]
                }
            ).encode()

    def fake_urlopen(request: object, timeout: int) -> Response:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["payload"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    containers = DockerExecutor(
        challenge="agent",
        backend="broker",
        broker_url="http://broker",
        broker_token="tok",
    ).list_containers("job-1")

    assert captured["url"] == "http://broker/v1/docker/list"
    assert cast(dict[str, Any], captured["payload"]) == {"job_id": "job-1"}
    assert containers[0].container_name == "agent-job"


def test_broker_backend_cleanup_posts_job_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base.challenge_sdk.executors.docker as module

    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"status": "ok"}).encode()

    def fake_urlopen(request: object, timeout: int) -> Response:
        captured["timeout"] = timeout
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["headers"] = dict(request.headers.items())  # type: ignore[attr-defined]
        captured["payload"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    DockerExecutor(
        challenge="agent",
        backend="broker",
        broker_url="http://broker",
        broker_token="tok",
    ).cleanup_job("job-1")

    assert captured["url"] == "http://broker/v1/docker/cleanup"
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert cast(dict[str, Any], captured["payload"]) == {"job_id": "job-1"}
    assert captured["timeout"] == 30


def test_template_executor_matches_shared_sdk() -> None:
    root = Path(__file__).resolve().parents[2]
    shared = root / "src/base/challenge_sdk/executors/docker.py"
    template = (
        root
        / "src/base/templates/challenge/src"
        / "__package_name__/sdk/executors/docker.py.j2"
    )
    assert template.read_text(encoding="utf-8") == shared.read_text(encoding="utf-8")


def _decode_members(payload: dict[str, Any]) -> tarfile.TarFile:
    raw = base64.b64decode(cast(str, payload["mounts"][0]["archive_b64"]))
    return tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")


def test_encode_mount_drops_internal_symlink(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
    os.symlink("agent.py", tmp_path / "link.py")

    encoded = _encode_mount(DockerMount(tmp_path, "/workspace/agent"))

    with _decode_members({"mounts": [encoded]}) as tar:
        names = {member.name for member in tar.getmembers()}
        assert not any(member.issym() or member.islnk() for member in tar.getmembers())
    assert "./agent.py" in names
    assert "./link.py" not in names


def test_encode_mount_external_symlink_leaks_no_target_bytes(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("BROKER_TOKEN_SUPERSECRET", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "agent.py").write_text("x = 1\n", encoding="utf-8")
    os.symlink(secret, workspace / "evil")

    encoded = _encode_mount(DockerMount(workspace, "/workspace/agent"))

    raw = base64.b64decode(cast(str, encoded["archive_b64"]))
    assert b"BROKER_TOKEN_SUPERSECRET" not in raw
    with _decode_members({"mounts": [encoded]}) as tar:
        names = {member.name for member in tar.getmembers()}
    assert "./evil" not in names


def test_encode_mount_drops_broken_symlink_and_symlinked_dir(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
    os.symlink("does-not-exist", tmp_path / "broken")
    os.symlink("/", tmp_path / "rootdir")

    encoded = _encode_mount(DockerMount(tmp_path, "/workspace/agent"))

    with _decode_members({"mounts": [encoded]}) as tar:
        members = tar.getmembers()
        assert not any(member.issym() or member.islnk() for member in members)
        assert not any(
            member.name.startswith("/") or ".." in Path(member.name).parts
            for member in members
        )


def test_encode_mount_output_passes_broker_validation(tmp_path: Path) -> None:
    from base.master.docker_broker import _validate_tar_members

    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("y = 2\n", encoding="utf-8")
    os.symlink("agent.py", tmp_path / "alias.py")

    encoded = _encode_mount(DockerMount(tmp_path, "/workspace/agent"))

    with _decode_members({"mounts": [encoded]}) as tar:
        _validate_tar_members(tar)


def test_encode_mount_keeps_regular_files_unchanged(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "mod.py").write_text("y = 2\n", encoding="utf-8")

    encoded = _encode_mount(DockerMount(tmp_path, "/workspace/agent"))

    with _decode_members({"mounts": [encoded]}) as tar:
        names = {member.name for member in tar.getmembers()}
    assert "./agent.py" in names
    assert "./pkg/mod.py" in names
