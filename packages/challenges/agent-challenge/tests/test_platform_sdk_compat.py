from __future__ import annotations

from dataclasses import fields

from base.challenge_sdk.executors import docker as platform_docker

from agent_challenge.sdk import executors as challenge_executors


def test_challenge_executor_shim_reexports_platform_docker_classes() -> None:
    assert challenge_executors.DockerExecutor is platform_docker.DockerExecutor
    assert challenge_executors.DockerLimits is platform_docker.DockerLimits
    assert challenge_executors.DockerMount is platform_docker.DockerMount
    assert challenge_executors.DockerRunResult is platform_docker.DockerRunResult
    assert challenge_executors.DockerRunSpec is platform_docker.DockerRunSpec


def test_docker_limits_contract_fields_match_platform_sdk() -> None:
    required_fields = [
        "cpus",
        "memory",
        "memory_swap",
        "pids_limit",
        "network",
        "read_only",
        "user",
        "tmpfs",
        "ulimits",
        "cap_drop",
        "security_opt",
        "init",
    ]
    field_names = [field.name for field in fields(challenge_executors.DockerLimits)]

    assert field_names[: len(required_fields)] == required_fields
    assert set(field_names) - set(required_fields) <= {"gpu_count", "privileged"}


def test_docker_run_spec_contract_fields_match_platform_sdk() -> None:
    assert [field.name for field in fields(challenge_executors.DockerRunSpec)] == [
        "image",
        "command",
        "mounts",
        "workdir",
        "env",
        "labels",
        "name",
        "limits",
        "image_pull_policy",
    ]
