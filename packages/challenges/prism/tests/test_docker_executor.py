from __future__ import annotations

import json
import subprocess
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest
from base.challenge_sdk.executor import (
    DockerContainerInfo,
    DockerExecutor,
    DockerExecutorError,
    DockerLimits,
    DockerMount,
    DockerRunSpec,
)


def test_build_run_command_has_security_flags(tmp_path: Path) -> None:
    spec = DockerRunSpec(
        image="ghcr.io/baseintelligence/prism-evaluator:latest",
        command=("python", "/workspace/runner.py"),
        mounts=(DockerMount(tmp_path, "/workspace"),),
        workdir="/workspace",
        env={"PRISM_GPU_COUNT": "1"},
        labels={"base.job": "job-1", "base.task": "architecture"},
        limits=DockerLimits(cpus=1.5, memory="512m", pids_limit=64, user="65532:65532"),
    )

    cmd = DockerExecutor(
        challenge="prism", allowed_images=("ghcr.io/baseintelligence/",)
    ).build_run_command(spec, "prism-job-task")

    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--network" in cmd and "none" in cmd
    assert "--cap-drop" in cmd and "ALL" in cmd
    assert "no-new-privileges" in cmd
    assert "--read-only" in cmd
    assert "--init" in cmd
    assert "--memory-swap" in cmd and "512m" in cmd
    assert "--user" in cmd and "65532:65532" in cmd
    assert "--ulimit" in cmd and "nofile=1024:1024" in cmd
    assert "--label" in cmd and "base.challenge=prism" in cmd
    assert f"{tmp_path.resolve()}:/workspace:ro" in cmd
    assert "PRISM_GPU_COUNT=1" in cmd
    assert cmd[-3:] == [
        "ghcr.io/baseintelligence/prism-evaluator:latest",
        "python",
        "/workspace/runner.py",
    ]


def test_platform_v10_resource_security_fields_are_preserved(tmp_path: Path) -> None:
    limits = DockerLimits(
        tmpfs=("/tmp:rw,noexec,nosuid,size=1g", "/run:rw,noexec,nosuid,size=64m"),
        cap_drop=("ALL", "NET_RAW"),
        security_opt=("no-new-privileges", "seccomp=unconfined"),
    )
    spec = DockerRunSpec(
        image="ghcr.io/baseintelligence/prism-evaluator:latest",
        command=("true",),
        mounts=(DockerMount(tmp_path, "/workspace"),),
        limits=limits,
    )

    cmd = DockerExecutor(
        challenge="prism", allowed_images=("ghcr.io/baseintelligence/",)
    ).build_run_command(spec, "prism-job-task")

    assert limits.tmpfs == (
        "/tmp:rw,noexec,nosuid,size=1g",
        "/run:rw,noexec,nosuid,size=64m",
    )
    assert limits.cap_drop == ("ALL", "NET_RAW")
    assert limits.security_opt == ("no-new-privileges", "seccomp=unconfined")
    assert cmd.count("--tmpfs") == 2
    assert "/tmp:rw,noexec,nosuid,size=1g" in cmd
    assert "/run:rw,noexec,nosuid,size=64m" in cmd
    assert cmd.count("--cap-drop") == 2
    assert "ALL" in cmd
    assert "NET_RAW" in cmd
    assert cmd.count("--security-opt") == 2
    assert "no-new-privileges" in cmd
    assert "seccomp=unconfined" in cmd


def test_reserved_labels_cannot_be_overridden(tmp_path: Path) -> None:
    spec = DockerRunSpec(
        image="ghcr.io/baseintelligence/prism-evaluator:latest",
        command=("true",),
        mounts=(DockerMount(tmp_path, "/workspace"),),
        labels={"base.challenge": "evil", "base.job": "job-1"},
    )

    cmd = DockerExecutor(
        challenge="prism", allowed_images=("ghcr.io/baseintelligence/",)
    ).build_run_command(spec, "name")

    assert "base.challenge=prism" in cmd
    assert "base.challenge=evil" not in cmd


@pytest.mark.parametrize("image", ["-v", "bad image", "../../bad"])
def test_rejects_unsafe_image_refs(tmp_path: Path, image: str) -> None:
    spec = DockerRunSpec(
        image=image,
        command=("true",),
        mounts=(DockerMount(tmp_path, "/x"),),
    )
    with pytest.raises(DockerExecutorError):
        DockerExecutor(challenge="prism").build_run_command(spec, "name")


def test_rejects_images_outside_allowlist(tmp_path: Path) -> None:
    spec = DockerRunSpec(
        image="docker.io/library/python:latest",
        command=("true",),
        mounts=(DockerMount(tmp_path, "/x"),),
    )
    with pytest.raises(DockerExecutorError, match="not allowed"):
        DockerExecutor(challenge="prism", allowed_images=("ghcr.io/baseintelligence/",)).run(
            spec, timeout_seconds=1
        )


def test_rejects_empty_command_bad_network_and_bad_mount(tmp_path: Path) -> None:
    executor = DockerExecutor(challenge="prism")
    with pytest.raises(DockerExecutorError, match="cannot be empty"):
        executor.build_run_command(
            DockerRunSpec(image="python:3.12", command=(), mounts=(DockerMount(tmp_path, "/x"),)),
            "name",
        )
    with pytest.raises(DockerExecutorError, match="network"):
        executor.build_run_command(
            DockerRunSpec(
                image="python:3.12",
                command=("true",),
                limits=DockerLimits(network="bridge"),
            ),
            "name",
        )
    with pytest.raises(DockerExecutorError, match="mount source"):
        executor.build_run_command(
            DockerRunSpec(
                image="python:3.12",
                command=("true",),
                mounts=(DockerMount(tmp_path / "missing", "/x"),),
            ),
            "name",
        )
    with pytest.raises(DockerExecutorError, match="mount target"):
        executor.build_run_command(
            DockerRunSpec(
                image="python:3.12",
                command=("true",),
                mounts=(DockerMount(tmp_path, "relative"),),
            ),
            "name",
        )


def test_run_cli_success_and_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(stdout="out", stderr="err", returncode=7)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = DockerExecutor(challenge="prism").run(
        DockerRunSpec(
            image="python:3.12",
            command=("true",),
            mounts=(DockerMount(tmp_path, "/x"),),
            labels={"base.job": "job-1"},
            name="fixed-name",
        ),
        timeout_seconds=3,
    )

    assert result.container_name == "fixed-name"
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.returncode == 7
    assert calls[-1] == ["docker", "rm", "-f", "fixed-name"]


def test_run_cli_timeout_removes_container(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, timeout=1, output=b"out", stderr=b"err")
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = DockerExecutor(challenge="prism").run(
        DockerRunSpec(
            image="python:3.12",
            command=("true",),
            mounts=(DockerMount(tmp_path, "/x"),),
            name="timeout-name",
        ),
        timeout_seconds=1,
    )

    assert result.timed_out is True
    assert result.returncode == 124
    assert result.stdout == "out"
    assert calls.count(["docker", "rm", "-f", "timeout-name"]) == 2


def test_run_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(DockerExecutorError, match="unsupported"):
        DockerExecutor(challenge="prism", backend="remote").run(
            DockerRunSpec(
                image="python:3.12",
                command=("true",),
                mounts=(DockerMount(tmp_path, "/x"),),
            ),
            timeout_seconds=1,
        )


def test_cleanup_job_uses_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        if cmd[:3] == ["docker", "ps", "-aq"]:
            return SimpleNamespace(stdout="abc\ndef\n", stderr="", returncode=0)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    DockerExecutor(challenge="prism").cleanup_job("job-1")

    assert calls[0] == [
        "docker",
        "ps",
        "-aq",
        "--filter",
        "label=base.challenge=prism",
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
                    "Names": "prism-job",
                    "Image": "python:3.12",
                    "Status": "Up",
                    "CreatedAt": "now",
                    "Labels": "base.challenge=prism,base.job=job-1",
                }
            )
            + "\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    containers = DockerExecutor(challenge="prism").list_containers("job-1")

    assert calls[0] == [
        "docker",
        "ps",
        "-a",
        "--filter",
        "label=base.challenge=prism",
        "--filter",
        "label=base.job=job-1",
        "--format",
        "{{json .}}",
    ]
    assert containers == [
        DockerContainerInfo(
            container_id="abc",
            container_name="prism-job",
            image="python:3.12",
            status="Up",
            job_id="job-1",
            created="now",
            labels={"base.challenge": "prism", "base.job": "job-1"},
        )
    ]


def test_list_containers_reports_cli_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="denied", returncode=1),
    )

    with pytest.raises(DockerExecutorError, match="Docker list failed: denied"):
        DockerExecutor(challenge="prism").list_containers()


def test_broker_backend_posts_run_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import base.challenge_sdk.executors.docker as module

    (tmp_path / "input.txt").write_text("ok", encoding="utf-8")
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "container_name": "prism-job",
                    "stdout": "ok",
                    "stderr": "",
                    "returncode": "0",
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
        challenge="prism",
        backend="broker",
        broker_url="http://broker",
        broker_token="tok",
        allowed_images=("python:",),
    ).run(
        DockerRunSpec(
            image="python:3.12-slim",
            command=("python", "-V"),
            mounts=(DockerMount(tmp_path, "/mnt"),),
            labels={"base.job": "job-1"},
        ),
        timeout_seconds=20,
    )

    assert result.returncode == 0
    assert captured["timeout"] == 35
    assert captured["url"] == "http://broker/v1/docker/run"
    assert captured["headers"]["Authorization"] == "Bearer tok"
    payload = captured["payload"]
    assert payload["image"] == "python:3.12-slim"
    assert payload["mounts"][0]["target"] == "/mnt"
    assert payload["timeout_seconds"] == 20


def test_broker_backend_lists_containers(monkeypatch: pytest.MonkeyPatch) -> None:
    import base.challenge_sdk.executors.docker as module

    captured: dict[str, object] = {}

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
                            "container_name": "prism-job",
                            "image": "python",
                            "status": "running",
                            "job_id": "job-1",
                            "labels": {"base.challenge": "prism"},
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
        challenge="prism",
        backend="broker",
        broker_url="http://broker",
        broker_token="tok",
    ).list_containers("job-1")

    assert captured["url"] == "http://broker/v1/docker/list"
    assert captured["payload"] == {"job_id": "job-1"}
    assert containers[0].container_name == "prism-job"


def test_broker_cleanup_posts_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    import base.challenge_sdk.executors.docker as module

    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"status":"ok"}'

    def fake_urlopen(request: object, timeout: int) -> Response:
        captured["url"] = request.full_url  # type: ignore[attr-defined]
        captured["payload"] = json.loads(request.data.decode())  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    DockerExecutor(
        challenge="prism",
        backend="broker",
        broker_url="http://broker",
        broker_token="tok",
    ).cleanup_job("job-1")

    assert captured["url"] == "http://broker/v1/docker/cleanup"
    assert captured["payload"] == {"job_id": "job-1"}


def test_broker_list_rejects_invalid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    import base.challenge_sdk.executors.docker as module

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"containers":{}}'

    monkeypatch.setattr(module, "urlopen", lambda request, timeout: Response())
    with pytest.raises(DockerExecutorError, match="invalid list payload"):
        DockerExecutor(
            challenge="prism",
            backend="broker",
            broker_url="http://broker",
            broker_token="tok",
        ).list_containers()


def test_broker_requires_url_and_token(tmp_path: Path) -> None:
    spec = DockerRunSpec(
        image="python:3.12",
        command=("true",),
        mounts=(DockerMount(tmp_path, "/x"),),
    )
    with pytest.raises(DockerExecutorError, match="URL"):
        DockerExecutor(challenge="prism", backend="broker", broker_token="tok").run(
            spec, timeout_seconds=1
        )
    with pytest.raises(DockerExecutorError, match="token"):
        DockerExecutor(challenge="prism", backend="broker", broker_url="http://broker").run(
            spec, timeout_seconds=1
        )


def test_broker_reads_token_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import base.challenge_sdk.executors.docker as module

    token_file = tmp_path / "token"
    token_file.write_text("file-token\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"containers":[]}'

    def fake_urlopen(request: object, timeout: int) -> Response:
        captured["headers"] = dict(request.headers.items())  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    DockerExecutor(
        challenge="prism",
        backend="broker",
        broker_url="http://broker",
        broker_token_file=str(token_file),
    ).list_containers()

    assert captured["headers"]["Authorization"] == "Bearer file-token"


def test_broker_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    import base.challenge_sdk.executors.docker as module

    class ErrorBody(BytesIO):
        def read(self, *args: object, **kwargs: object) -> bytes:
            return b"bad request"

    def http_error(request: object, timeout: int) -> object:
        raise HTTPError("http://broker", 400, "bad", {}, ErrorBody())

    monkeypatch.setattr(module, "urlopen", http_error)
    with pytest.raises(DockerExecutorError, match="bad request"):
        DockerExecutor(
            challenge="prism",
            backend="broker",
            broker_url="http://broker",
            broker_token="tok",
        ).list_containers()

    def url_error(request: object, timeout: int) -> object:
        raise URLError("down")

    monkeypatch.setattr(module, "urlopen", url_error)
    with pytest.raises(DockerExecutorError, match="unavailable"):
        DockerExecutor(
            challenge="prism",
            backend="broker",
            broker_url="http://broker",
            broker_token="tok",
        ).list_containers()
