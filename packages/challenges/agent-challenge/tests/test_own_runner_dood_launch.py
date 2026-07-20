"""DooD (Docker-outside-of-Docker) launch mechanism for the in-CVM orchestrator (M2).

Black-box tests for :mod:`agent_challenge.evaluation.own_runner.dood` and its
wiring into the task-container launch path, encoding the M2 validation-contract
assertions this feature fulfills:

* VAL-ORCH-011  the orchestrator drives the unix socket exclusively
                (``DOCKER_HOST`` is ``unix:///var/run/docker.sock``); no ``tcp://``
                host is configured or dialed and no inner ``dockerd`` is spawned.
* VAL-ORCH-012  the Docker socket (and ``dstack.sock``) are mounted ONLY on the
                orchestrator, never into a launched Terminal-Bench task container.

The launch path is exercised offline with the Docker client faked/recorded
(no live daemon), so the recorded client target, the launch argument vectors,
and each task container's mount list are asserted directly.
"""

from __future__ import annotations

import subprocess

import pytest

from agent_challenge.evaluation.own_runner import container_builder as cb
from agent_challenge.evaluation.own_runner import exec_bridge as eb
from agent_challenge.evaluation.own_runner.container_builder import TaskContainerBuilder
from agent_challenge.evaluation.own_runner.dood import (
    DOCKER_SOCKET_PATH,
    DOOD_DOCKER_HOST,
    DSTACK_SOCKET_PATH,
    SENSITIVE_SOCKET_PATHS,
    DoodConfigError,
    DoodSocketExposureError,
    assert_no_socket_mounts,
    dood_docker_argv,
    dood_docker_env,
    has_socket_mount,
    is_tcp_docker_host,
    iter_volume_mounts,
    resolve_docker_host,
    socket_mount_specs,
    spawns_inner_dockerd,
)
from agent_challenge.evaluation.own_runner.exec_bridge import DockerExecEnvironment
from agent_challenge.evaluation.own_runner.taskdefs import ResourceLimits


# --------------------------------------------------------------------------- #
# helpers: record docker subprocess launches without a live daemon
# --------------------------------------------------------------------------- #
class _RecordedRun:
    """A ``subprocess.run`` stand-in that records argv + kwargs and returns rc 0."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, argv, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        self.calls.append({"argv": list(argv), "kwargs": kwargs})
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    @property
    def argvs(self) -> list[list[str]]:
        return [c["argv"] for c in self.calls]  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# VAL-ORCH-011: unix-socket-only client target, no tcp, no inner dockerd
# --------------------------------------------------------------------------- #
def test_dood_docker_host_is_the_guest_unix_socket() -> None:
    assert DOOD_DOCKER_HOST == "unix:///var/run/docker.sock"
    assert DOCKER_SOCKET_PATH == "/var/run/docker.sock"
    assert DSTACK_SOCKET_PATH == "/var/run/dstack.sock"
    assert not is_tcp_docker_host(DOOD_DOCKER_HOST)
    assert "tcp://" not in DOOD_DOCKER_HOST


def test_is_tcp_docker_host_discriminates() -> None:
    assert is_tcp_docker_host("tcp://127.0.0.1:2375") is True
    assert is_tcp_docker_host("tcp://0.0.0.0:2376") is True
    assert is_tcp_docker_host("TCP://host:2375") is True
    assert is_tcp_docker_host("unix:///var/run/docker.sock") is False
    assert is_tcp_docker_host(None) is False
    assert is_tcp_docker_host("") is False


def test_resolve_docker_host_defaults_to_unix_socket() -> None:
    assert resolve_docker_host({}) == DOOD_DOCKER_HOST
    assert resolve_docker_host(None) == DOOD_DOCKER_HOST
    assert resolve_docker_host({"DOCKER_HOST": ""}) == DOOD_DOCKER_HOST
    # An explicit unix socket is preserved (still unix-only).
    assert (
        resolve_docker_host({"DOCKER_HOST": "unix:///var/run/docker.sock"})
        == "unix:///var/run/docker.sock"
    )


def test_resolve_docker_host_fails_closed_on_tcp() -> None:
    with pytest.raises(DoodConfigError):
        resolve_docker_host({"DOCKER_HOST": "tcp://1.2.3.4:2375"})
    with pytest.raises(DoodConfigError):
        resolve_docker_host({"DOCKER_HOST": "tcp://0.0.0.0:2376"})


def test_dood_docker_env_pins_unix_socket_and_never_tcp() -> None:
    env = dood_docker_env({"PATH": "/usr/bin", "DOCKER_HOST": "tcp://evil:2375"})
    assert env["DOCKER_HOST"] == DOOD_DOCKER_HOST
    # PATH (and other benign vars) are preserved; no value dials tcp.
    assert env["PATH"] == "/usr/bin"
    assert not any("tcp://" in str(v) for v in env.values())


def test_dood_docker_argv_pins_the_unix_socket_target() -> None:
    argv = dood_docker_argv("docker", "run", "-d", "img")
    assert argv == ["docker", "-H", DOOD_DOCKER_HOST, "run", "-d", "img"]
    assert "tcp://" not in " ".join(argv)


def test_spawns_inner_dockerd_detects_a_daemon_bringup() -> None:
    assert spawns_inner_dockerd(["dockerd", "--host=unix:///var/run/docker.sock"]) is True
    assert spawns_inner_dockerd(["/usr/bin/dockerd"]) is True
    assert spawns_inner_dockerd(["sh", "-c", "start_dockerd; dockerd --data-root=/x &"]) is True
    # A plain sibling launch on the socket never brings up a daemon.
    assert spawns_inner_dockerd(["docker", "run", "-d", "img", "sleep", "infinity"]) is False


def test_exec_environment_records_the_unix_socket_target() -> None:
    env = DockerExecEnvironment("c")
    assert env.docker_host == DOOD_DOCKER_HOST
    assert not is_tcp_docker_host(env.docker_host)
    builder = TaskContainerBuilder()
    assert builder.docker_host == DOOD_DOCKER_HOST
    assert not is_tcp_docker_host(builder.docker_host)


def test_launch_targets_unix_socket_no_tcp_no_dockerd(monkeypatch) -> None:
    rec = _RecordedRun()
    monkeypatch.setattr(eb.subprocess, "run", rec)
    DockerExecEnvironment.launch("some/image:1", container_name="own-runner-t")
    assert rec.calls, "expected a docker run launch"
    call = rec.calls[0]
    # DOCKER_HOST handed to the client is the guest unix socket.
    assert call["kwargs"]["env"]["DOCKER_HOST"] == DOOD_DOCKER_HOST  # type: ignore[index]
    # No tcp dial and no inner dockerd anywhere in the launch.
    for argv in rec.argvs:
        assert "tcp://" not in " ".join(argv)
        assert not spawns_inner_dockerd(argv)


def test_builder_run_container_targets_unix_socket_no_tcp_no_dockerd(monkeypatch) -> None:
    rec = _RecordedRun()
    monkeypatch.setattr(cb.subprocess, "run", rec)
    builder = TaskContainerBuilder()
    env = builder.run_container("some/image:1", resources=ResourceLimits(cpus=1, memory_mb=512))
    assert env.docker_host == DOOD_DOCKER_HOST
    run_calls = [c for c in rec.calls if "run" in c["argv"]]  # type: ignore[operator]
    assert run_calls, "expected a docker run launch"
    for call in run_calls:
        assert call["kwargs"]["env"]["DOCKER_HOST"] == DOOD_DOCKER_HOST  # type: ignore[index]
    for argv in rec.argvs:
        assert "tcp://" not in " ".join(argv)
        assert not spawns_inner_dockerd(argv)


# --------------------------------------------------------------------------- #
# VAL-ORCH-012: docker/dstack socket never mounted into task containers
# --------------------------------------------------------------------------- #
def test_iter_volume_mounts_extracts_all_mount_forms() -> None:
    argv = [
        "docker",
        "run",
        "-v",
        "/data:/data",
        "--volume",
        "/logs:/logs:ro",
        "--volume=/cfg:/cfg",
        "--mount",
        "type=bind,src=/x,dst=/y",
        "img",
    ]
    mounts = list(iter_volume_mounts(argv))
    assert "/data:/data" in mounts
    assert "/logs:/logs:ro" in mounts
    assert "/cfg:/cfg" in mounts
    assert "type=bind,src=/x,dst=/y" in mounts


def test_socket_mount_specs_detects_docker_and_dstack_sockets() -> None:
    docker_v = [
        "docker",
        "run",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
        "img",
    ]
    assert socket_mount_specs(docker_v) == ["/var/run/docker.sock:/var/run/docker.sock"]

    dstack_mount = [
        "docker",
        "run",
        "--mount",
        "type=bind,src=/var/run/dstack.sock,dst=/var/run/dstack.sock",
        "img",
    ]
    assert has_socket_mount(dstack_mount) is True

    ro_volume = ["docker", "run", "--volume=/var/run/docker.sock:/sock:ro", "img"]
    assert has_socket_mount(ro_volume) is True


def test_socket_mount_specs_detects_parent_dir_mount() -> None:
    # A parent-dir mount exposes the socket even though the socket path is NOT a
    # literal substring of the spec (path-aware detection, not substring).
    parent_v = ["docker", "run", "-v", "/var/run:/var/run", "img"]
    assert socket_mount_specs(parent_v) == ["/var/run:/var/run"]
    assert has_socket_mount(parent_v) is True
    with pytest.raises(DoodSocketExposureError):
        assert_no_socket_mounts(parent_v)

    # ``--mount`` form of the same parent-dir exposure.
    parent_mount = ["docker", "run", "--mount", "type=bind,src=/var/run,dst=/var/run", "img"]
    assert has_socket_mount(parent_mount) is True

    # An ancestor further up the tree still exposes the socket.
    var_mount = ["docker", "run", "-v", "/var:/var", "img"]
    assert has_socket_mount(var_mount) is True

    # Mounting the host root exposes everything.
    root_mount = ["docker", "run", "-v", "/:/host", "img"]
    assert has_socket_mount(root_mount) is True


def test_socket_mount_specs_no_false_positive_on_prefix_lookalikes() -> None:
    # A sibling dir that merely shares a textual path prefix is NOT the socket
    # dir (path-aware matching must not fire on ``/var/running``), and a file
    # whose name extends the socket name is not the socket.
    lookalike_dir = ["docker", "run", "-v", "/var/running:/var/running", "img"]
    assert socket_mount_specs(lookalike_dir) == []
    assert has_socket_mount(lookalike_dir) is False
    assert_no_socket_mounts(lookalike_dir)

    lookalike_file = ["docker", "run", "-v", "/var/run/docker.sock.bak:/x", "img"]
    assert socket_mount_specs(lookalike_file) == []
    assert has_socket_mount(lookalike_file) is False


def test_socket_path_constants_are_single_sourced() -> None:
    # DOCKER_SOCKET_PATH / DSTACK_SOCKET_PATH must be single-sourced: the compose
    # generator imports the exact dood constants (same object identity), so the
    # two definitions can never silently diverge.
    from agent_challenge.canonical import compose as compose_mod

    assert compose_mod.DOCKER_SOCKET_PATH is DOCKER_SOCKET_PATH
    assert compose_mod.DSTACK_SOCKET_PATH is DSTACK_SOCKET_PATH


def test_socket_mount_specs_empty_for_a_clean_task_launch() -> None:
    clean = [
        "docker",
        "run",
        "-d",
        "--name",
        "own-runner-task",
        "--network",
        "none",
        "-v",
        "/workspace:/workspace",
        "img",
        "sleep",
        "infinity",
    ]
    assert socket_mount_specs(clean) == []
    assert has_socket_mount(clean) is False


def test_assert_no_socket_mounts_raises_on_exposure_and_passes_when_clean() -> None:
    exposing = ["docker", "run", "-v", f"{DOCKER_SOCKET_PATH}:{DOCKER_SOCKET_PATH}", "img"]
    with pytest.raises(DoodSocketExposureError):
        assert_no_socket_mounts(exposing)
    for sock in SENSITIVE_SOCKET_PATHS:
        with pytest.raises(DoodSocketExposureError):
            assert_no_socket_mounts(["docker", "run", "--mount", f"src={sock},dst={sock}", "img"])
    # A clean launch does not raise.
    assert_no_socket_mounts(["docker", "run", "-d", "img", "sleep", "infinity"])


def test_launched_task_container_has_no_socket_mount(monkeypatch) -> None:
    rec = _RecordedRun()
    monkeypatch.setattr(eb.subprocess, "run", rec)
    DockerExecEnvironment.launch("some/image:1")
    for argv in rec.argvs:
        # Enumerate the task container's mount list: neither socket is present.
        assert socket_mount_specs(argv) == []
        assert DOCKER_SOCKET_PATH not in " ".join(argv)
        assert DSTACK_SOCKET_PATH not in " ".join(argv)


def test_builder_run_container_task_spec_has_no_socket_mount(monkeypatch) -> None:
    rec = _RecordedRun()
    monkeypatch.setattr(cb.subprocess, "run", rec)
    builder = TaskContainerBuilder()
    builder.run_container(
        "some/image:1",
        resources=ResourceLimits(cpus=1, memory_mb=512, allow_internet=True),
    )
    run_calls = [c["argv"] for c in rec.calls if "run" in c["argv"]]  # type: ignore[operator]
    assert run_calls
    for argv in run_calls:
        assert socket_mount_specs(argv) == []
        assert DOCKER_SOCKET_PATH not in " ".join(argv)
        assert DSTACK_SOCKET_PATH not in " ".join(argv)
