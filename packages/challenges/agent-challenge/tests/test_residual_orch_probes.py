"""Offline residual ORCH probe marker formats (VAL-ORCH-009/010/014/022).

Public_logs residual path without guest SSH: secret-free marker lines prove
concurrency bound, concurrent running samples, DooD inspect fields, network-none
seal egress fail, and gateway-only / fail-closed posture.

Discriminators (would fail a wrong implementation):
* marker prefix is always residual_orch kind=...
* secrets / PEM / tcp docker hosts never appear in emitted lines
* concurrency_bound logs the bound used by the job
* ps_sample / inflight encode running vs bound and gt_one flags
* task_inspect encodes NetworkMode, Privileged=false, no tcp 2375/2376
* network_none_seal proves network=none + egress blocked/fail_closed
* gateway_posture fails closed when no gateway is configured

No Phala create; pure offline fakes with injectable docker runner.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_challenge.evaluation.own_runner import residual_orch_probes as probes
from agent_challenge.evaluation.own_runner.dood import DOOD_DOCKER_HOST


def _cp(returncode: int = 0, stdout: str = "", stderr: str = "") -> Any:
    return type(
        "CP",
        (),
        {"returncode": returncode, "stdout": stdout, "stderr": stderr},
    )()


class _RecordingRun:
    def __init__(self, decide: Any) -> None:
        self.decide = decide
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append(list(argv))
        return self.decide(argv, **kwargs)


def test_residual_probes_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(probes.RESIDUAL_ORCH_PROBES_ENV, raising=False)
    assert probes.residual_orch_probes_enabled() is False
    assert probes.maybe_make_probe_controller(bound=2) is None


def test_residual_probes_enabled_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(probes.RESIDUAL_ORCH_PROBES_ENV, "1")
    assert probes.residual_orch_probes_enabled() is True
    ctrl = probes.maybe_make_probe_controller(bound=2)
    assert ctrl is not None
    assert ctrl.enabled is True


def test_format_residual_marker_has_prefix_and_kind() -> None:
    line = probes.format_residual_marker("concurrency_bound", bound=2, nproc=2)
    assert line.startswith(f"{probes.RESIDUAL_ORCH_MARKER} kind=concurrency_bound")
    assert "bound=2" in line
    assert "nproc=2" in line


def test_format_residual_marker_redacts_secrets_and_pems() -> None:
    line = probes.format_residual_marker(
        "task_inspect",
        note="BEGIN CERTIFICATE abc",
        token="FAKESECRET_a2b3c4d5e6f7g8h9i0j1",
        detail="password=hunter2",
    )
    assert "BEGIN CERTIFICATE" not in line
    assert "sk-live" not in line
    assert "hunter2" not in line
    assert "redacted" in line


def test_format_residual_marker_redacts_tcp_docker_host() -> None:
    line = probes.format_residual_marker(
        "task_inspect",
        docker_host="tcp://127.0.0.1:2375",
    )
    assert "tcp://127.0.0.1:2375" not in line
    assert "tcp_docker_redacted" in line


def test_log_concurrency_bound_emits_bound(capsys: pytest.CaptureFixture[str]) -> None:
    line = probes.log_concurrency_bound(4, nproc=8, source="auto")
    out = capsys.readouterr().out
    assert "residual_orch kind=concurrency_bound" in line
    assert "bound=4" in line
    assert "bound=4" in out
    assert "nproc=8" in out


def test_parse_and_count_task_containers() -> None:
    names = probes.parse_docker_ps_names(
        "own-runner-task-aaa\nown-runner-task-bbb\ndstack-orchestrator-1\nresidual-orch-seal-xyz\n"
    )
    assert len(names) == 4
    assert probes.count_task_containers(names) == 2


def test_sample_task_running_count_filters_prefixes() -> None:
    def decide(argv: list[str], **_kwargs: Any) -> Any:
        assert "ps" in argv
        assert any("label=base.own_runner=1" in a for a in argv)
        return _cp(
            0,
            "own-runner-task-1\nown-runner-task-2\nresidual-orch-seal-x\n",
        )

    runner = _RecordingRun(decide)
    running, names = probes.sample_task_running_count(runner=runner)
    assert running == 2
    assert names == ["own-runner-task-1", "own-runner-task-2"]


def test_log_ps_sample_flags(capsys: pytest.CaptureFixture[str]) -> None:
    line = probes.log_ps_sample(
        bound=2,
        running=2,
        names=["own-runner-task-aa", "own-runner-task-bb"],
        sample_index=3,
    )
    out = capsys.readouterr().out
    assert "kind=ps_sample" in line
    assert "running=2" in line
    assert "bound=2" in line
    assert "within_bound=true" in line
    assert "gt_one=true" in line
    assert "i=3" in out


def test_extract_inspect_fields_dood_safe_sibling() -> None:
    payload = {
        "HostConfig": {
            "NetworkMode": "none",
            "Privileged": False,
            "Binds": ["/opt/agent-challenge/task-cache:/opt/agent-challenge/task-cache:ro"],
        },
        "NetworkSettings": {"Networks": {}},
    }
    fields = probes.extract_inspect_fields(payload, docker_host=DOOD_DOCKER_HOST)
    assert fields["network_mode"] == "none"
    assert fields["privileged"] is False
    assert fields["binds_count"] == 1
    assert fields["has_docker_sock_bind"] is False
    assert fields["has_dstack_sock_bind"] is False
    assert fields["docker_host_is_unix"] is True
    assert fields["docker_host_has_tcp_2375_2376"] is False


def test_extract_inspect_fields_detects_tcp_and_privileged() -> None:
    payload = {
        "HostConfig": {
            "NetworkMode": "bridge",
            "Privileged": True,
            "Binds": ["/var/run/docker.sock:/var/run/docker.sock"],
        }
    }
    fields = probes.extract_inspect_fields(
        payload,
        docker_host="tcp://10.0.0.2:2376",
    )
    assert fields["privileged"] is True
    assert fields["has_docker_sock_bind"] is True
    assert fields["docker_host_is_unix"] is False
    assert fields["docker_host_has_tcp_2375_2376"] is True


def test_log_task_inspect_emits_contract_fields(capsys: pytest.CaptureFixture[str]) -> None:
    fields = {
        "container": "own-runner-task-deadbeef",
        "network_mode": "none",
        "privileged": False,
        "binds_count": 1,
        "has_docker_sock_bind": False,
        "has_dstack_sock_bind": False,
        "docker_host_is_unix": True,
        "docker_host_has_tcp_2375_2376": False,
    }
    line = probes.log_task_inspect(fields, task_id="adaptive-rejection-sampler")
    out = capsys.readouterr().out
    assert "kind=task_inspect" in line
    assert "NetworkMode=none" in line
    assert "Privileged=false" in line
    assert "docker_host_has_tcp_2375_2376=false" in line
    assert "task_id=adaptive-rejection-sampler" in out
    assert "BEGIN " not in out


def test_run_network_none_seal_probe_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    container = {"name": None}

    def decide(argv: list[str], **_kwargs: Any) -> Any:
        if len(argv) >= 2 and argv[1] == "run":
            # --name <name>
            idx = argv.index("--name")
            container["name"] = argv[idx + 1]
            assert "--network" in argv
            assert argv[argv.index("--network") + 1] == "none"
            return _cp(0, "cid\n")
        if len(argv) >= 2 and argv[1] == "inspect":
            body = [
                {
                    "HostConfig": {
                        "NetworkMode": "none",
                        "Privileged": False,
                        "Binds": [],
                    }
                }
            ]
            return _cp(0, json.dumps(body))
        if len(argv) >= 2 and argv[1] == "exec":
            return _cp(0, "EGRESS_BLOCKED\n")
        if len(argv) >= 2 and argv[1] == "rm":
            return _cp(0, "")
        return _cp(1, "", "unexpected")

    runner = _RecordingRun(decide)
    result = probes.run_network_none_seal_probe(runner=runner)
    assert result["network_mode"] == "none"
    assert result["egress"] == "blocked"
    assert result["ok"] is True
    assert result["container"].startswith(probes.RESIDUAL_SEAL_NAME_PREFIX)
    line = probes.log_network_none_seal(result)
    out = capsys.readouterr().out
    assert "kind=network_none_seal" in line
    assert "NetworkMode=none" in line
    assert "egress=blocked" in out
    assert "ok=true" in out


def test_gateway_posture_fail_closed_without_gateway(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("BASE_LLM_GATEWAY_URL", raising=False)
    line = probes.log_gateway_posture(env={})
    out = capsys.readouterr().out
    assert "kind=gateway_posture" in line
    assert "gateway_configured=false" in line
    assert "fail_closed_no_nongateway_egress" in out


def test_gateway_posture_with_configured_host(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = {"BASE_LLM_GATEWAY_URL": "https://gateway.example.test/v1"}
    line = probes.log_gateway_posture(env=env, probe_result="reachable")
    out = capsys.readouterr().out
    assert "gateway_configured=true" in line
    assert "gateway_host=gateway.example.test" in out
    assert "allowlist_host=reachable" in out
    assert "sk-" not in out


def test_controller_emits_full_residual_set(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(probes.RESIDUAL_ORCH_PROBES_ENV, "1")

    def decide(argv: list[str], **_kwargs: Any) -> Any:
        if len(argv) >= 2 and argv[1] == "ps":
            return _cp(0, "own-runner-task-a\nown-runner-task-b\n")
        if len(argv) >= 2 and argv[1] == "run":
            return _cp(0, "cid\n")
        if len(argv) >= 2 and argv[1] == "inspect":
            body = [
                {
                    "HostConfig": {
                        "NetworkMode": "none",
                        "Privileged": False,
                        "Binds": [],
                    }
                }
            ]
            return _cp(0, json.dumps(body))
        if len(argv) >= 2 and argv[1] == "exec":
            return _cp(0, "EGRESS_BLOCKED\n")
        if len(argv) >= 2 and argv[1] == "rm":
            return _cp(0, "")
        return _cp(0, "")

    runner = _RecordingRun(decide)
    ctrl = probes.ResidualOrchProbeController(
        bound=2,
        env={probes.RESIDUAL_ORCH_PROBES_ENV: "1"},
        sample_interval_sec=0.2,
        runner=runner,
        nproc=2,
    )
    assert ctrl.enabled
    ctrl.on_job_start()
    # Give the sampler at least one tick.
    import time

    time.sleep(0.45)
    ctrl.on_container_launched("own-runner-task-deadbeef", task_id="demo-task")
    ctrl.on_container_exited()
    ctrl.on_job_done()
    out = capsys.readouterr().out
    assert "kind=concurrency_bound" in out
    assert "bound=2" in out
    assert "kind=ps_sample" in out or "kind=ps_sample_summary" in out
    assert "kind=task_inspect" in out
    assert "NetworkMode=none" in out
    assert "Privileged=false" in out
    assert "kind=network_none_seal" in out
    assert "kind=gateway_posture" in out
    assert "kind=inflight" in out
    # Secret surface must stay clean.
    assert "BEGIN CERTIFICATE" not in out
    assert "OPENROUTER" not in out
    assert "tcp://127.0.0.1:2375" not in out


def test_compose_allows_residual_probe_env() -> None:
    from agent_challenge.canonical.compose import DEFAULT_ALLOWED_ENVS

    assert probes.RESIDUAL_ORCH_PROBES_ENV in DEFAULT_ALLOWED_ENVS


def test_flag_off_byte_identical_no_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(probes.RESIDUAL_ORCH_PROBES_ENV, raising=False)
    assert probes.maybe_make_probe_controller(bound=4, nproc=4) is None


def test_sample_task_running_count_engine_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, path: str, **_kwargs: Any) -> tuple[int, Any]:
        assert method == "GET"
        assert path.startswith("/containers/json")
        return 200, [
            {"Names": ["/own-runner-task-aaa"]},
            {"Names": ["/own-runner-task-bbb"]},
            {"Names": ["/residual-orch-seal-x"]},
        ]

    monkeypatch.setattr(probes, "docker_engine_request", fake_request)
    running, names = probes.sample_task_running_count()
    assert running == 2
    assert names == ["own-runner-task-aaa", "own-runner-task-bbb"]


def test_inspect_task_container_engine_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, path: str, **_kwargs: Any) -> tuple[int, Any]:
        assert method == "GET"
        assert path.startswith("/containers/")
        return 200, {
            "HostConfig": {
                "NetworkMode": "none",
                "Privileged": False,
                "Binds": ["/opt/task-cache:/opt/task-cache:ro"],
            }
        }

    monkeypatch.setattr(probes, "docker_engine_request", fake_request)
    fields = probes.inspect_task_container("own-runner-task-deadbeef")
    assert fields["network_mode"] == "none"
    assert fields["privileged"] is False
    assert fields["docker_host_is_unix"] is True
    assert fields["docker_host_has_tcp_2375_2376"] is False


def test_run_concurrent_loader_probe_engine_api(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    created: list[str] = []

    def fake_request(
        method: str,
        path: str,
        body: Any = None,
        **_kwargs: Any,
    ) -> tuple[int, Any]:
        if method == "GET" and path.startswith("/images/json"):
            return 200, [
                {
                    "Id": "sha256:orch",
                    "RepoTags": ["docker.io/mathiiss/agent-challenge-canonical@sha256:deadbeef"],
                }
            ]
        if method == "POST" and path.startswith("/containers/create"):
            assert isinstance(body, dict)
            assert body.get("HostConfig", {}).get("NetworkMode") == "none"
            assert body.get("HostConfig", {}).get("Privileged") is False
            assert body.get("Labels", {}).get("base.own_runner") == "1"
            created.append(path)
            return 201, {"Id": "cid"}
        if method == "POST" and path.endswith("/start"):
            return 204, None
        if method == "GET" and path.endswith("/json"):
            return 200, {
                "HostConfig": {
                    "NetworkMode": "none",
                    "Privileged": False,
                    "Binds": [],
                }
            }
        if method == "GET" and path.startswith("/containers/json"):
            # Sampler after loaders: report started residual loaders.
            names = [{"Names": [f"/own-runner-task-residual-{i}"]} for i in range(len(created))]
            return 200, names or [{"Names": ["/own-runner-task-residual-a"]}]
        if method == "DELETE":
            return 204, None
        return 500, {"message": f"unexpected {method} {path}"}

    monkeypatch.setattr(probes, "docker_engine_request", fake_request)
    result = probes.run_concurrent_loader_probe(bound=3, count=2, runner=None)
    assert result["ok"] is True
    assert result["started"] == 2
    assert len(result["names"]) == 2
    assert all(n.startswith(probes.RESIDUAL_LOADER_NAME_PREFIX) for n in result["names"])
    line = probes.log_concurrent_loaders(result, bound=3)
    out = capsys.readouterr().out
    assert "kind=concurrent_loaders" in line
    assert "started=2" in out
    assert "bound=3" in out
    assert "ok=true" in out


def test_controller_spawns_loaders_and_emits_gt_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(probes.RESIDUAL_ORCH_PROBES_ENV, "1")
    created = 0

    def fake_request(
        method: str,
        path: str,
        body: Any = None,
        **_kwargs: Any,
    ) -> tuple[int, Any]:
        nonlocal created
        if method == "GET" and path.startswith("/images/json"):
            return 200, [{"Id": "sha256:local", "RepoTags": ["local-orch:latest"]}]
        if method == "POST" and path.startswith("/containers/create"):
            created += 1
            return 201, {"Id": f"c{created}"}
        if method == "POST" and "/start" in path:
            return 204, None
        if method == "POST" and "/exec" in path:
            return 201, {"Id": "exec1"}
        if method == "GET" and path.startswith("/containers/json"):
            # At least two residual loaders for concurrent >1 evidence.
            return 200, [
                {"Names": ["/own-runner-task-residual-aaa"]},
                {"Names": ["/own-runner-task-residual-bbb"]},
            ]
        if method == "GET" and path.endswith("/json"):
            return 200, {
                "HostConfig": {
                    "NetworkMode": "none",
                    "Privileged": False,
                    "Binds": [],
                }
            }
        if method == "DELETE":
            return 204, None
        return 200, None

    monkeypatch.setattr(probes, "docker_engine_request", fake_request)
    ctrl = probes.ResidualOrchProbeController(
        bound=3,
        env={probes.RESIDUAL_ORCH_PROBES_ENV: "1"},
        sample_interval_sec=0.25,
        runner=None,
        nproc=4,
    )
    ctrl.on_job_start()
    import time

    time.sleep(0.35)
    ctrl.on_job_done()
    out = capsys.readouterr().out
    assert "kind=concurrency_bound" in out
    assert "nproc=4" in out
    assert "kind=concurrent_loaders" in out
    assert "kind=task_inspect" in out
    assert "NetworkMode=none" in out
    assert "Privileged=false" in out
    assert "docker_host_has_tcp_2375_2376=false" in out
    assert "kind=network_none_seal" in out
    assert "kind=gateway_posture" in out
    assert "kind=ps_sample" in out
    # Concurrent >1 is required for VAL-ORCH-009 multi-vCPU residual.
    assert "running=2" in out or "gt_one=true" in out
    assert created >= 2
