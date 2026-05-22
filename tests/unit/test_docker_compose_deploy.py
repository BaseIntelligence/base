from __future__ import annotations

from pathlib import Path

import yaml

WATCHTOWER_ENABLE_LABEL = "com.centurylinklabs.watchtower.enable"
WATCHTOWER_IMAGE = "nickfedor/watchtower:1.17.1"
EXPECTED_HARDENING = {
    "init": True,
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges:true"],
    "pids_limit": 512,
    "cpus": 2,
    "mem_limit": "4g",
    "read_only": True,
    "tmpfs": ["/tmp:rw,noexec,nosuid,size=256m"],
}
EXPECTED_WATCHTOWER_HARDENING = {
    **EXPECTED_HARDENING,
    "cpus": 1,
    "mem_limit": "512m",
    "tmpfs": ["/tmp:rw,noexec,nosuid,size=128m"],
}


def _load_compose_file(name: str) -> dict:
    return yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "docker" / name).read_text(
            encoding="utf-8"
        )
    )


def _assert_hardened(
    service: dict, expected: dict[str, object] = EXPECTED_HARDENING
) -> None:
    for key, value in expected.items():
        assert service[key] == value
    assert service.get("privileged") is not True


def _socket_services(services: dict) -> set[str]:
    return {
        name
        for name, service in services.items()
        if any(
            "/var/run/docker.sock" in volume for volume in service.get("volumes", [])
        )
    }


def _assert_socket_services_document_risk(services: dict, expected: set[str]) -> None:
    socket_services = _socket_services(services)
    assert socket_services == expected
    for name in socket_services:
        labels = services[name]["labels"]
        assert labels["platform.security.docker-socket"]
        assert labels["platform.security.docker-socket-risk"] == (
            "host-daemon-access-is-not-production-isolation"
        )


def test_compose_deploy_has_expected_services_and_socket_scope() -> None:
    compose = _load_compose_file("compose.yml")
    services = compose["services"]

    for name in (
        "master-admin",
        "master-proxy",
        "platform-docker-broker",
        "validator",
        "gpu-agent",
    ):
        assert name in services
    assert "postgres" not in services

    for service in services.values():
        _assert_hardened(service)
    _assert_socket_services_document_risk(
        services, {"master-admin", "platform-docker-broker", "gpu-agent"}
    )
    assert "platform_challenges" in services["platform-docker-broker"]["networks"]
    assert compose["networks"]["platform_challenges"]["attachable"] is True
    assert compose["networks"]["platform_challenges"]["internal"] is True
    assert "platform_db" in compose["volumes"]
    assert "platform_state" in compose["volumes"]
    assert "platform_secrets" in compose["volumes"]
    assert "platform_bittensor" in compose["volumes"]
    assert any(
        "platform_db:/var/lib/platform-db" in volume
        for volume in services["master-admin"]["volumes"]
    )
    assert any(
        "platform_db:/var/lib/platform-db" in volume
        for volume in services["master-proxy"]["volumes"]
    )
    assert any(
        "platform_secrets:/var/lib/platform/secrets" in volume
        for volume in services["master-proxy"]["volumes"]
    )
    assert any(
        "platform_bittensor:/root/.bittensor" in volume
        for volume in services["master-proxy"]["volumes"]
    )
    assert services["master-admin"]["environment"]["PLATFORM_DATABASE__URL"].startswith(
        "sqlite+aiosqlite:///"
    )
    assert (
        services["validator"]["environment"]["PLATFORM_VALIDATOR__REGISTRY_URL"]
        == "http://master-admin:8000"
    )
    assert "PLATFORM_DOCKER__BROKER_ALLOWED_IMAGES" not in services[
        "platform-docker-broker"
    ].get("environment", {})
    assert "healthcheck" in services["master-admin"]
    assert "healthcheck" in services["master-proxy"]


def test_dev_compose_keeps_socket_limited_and_documented() -> None:
    compose = _load_compose_file("compose.dev.yml")
    services = compose["services"]

    for service in services.values():
        _assert_hardened(service)
    _assert_socket_services_document_risk(
        services, {"master", "platform-docker-broker"}
    )
    assert compose["networks"]["platform_challenges"]["internal"] is True


def test_watchtower_overlay_uses_label_enable_mode() -> None:
    compose = _load_compose_file("compose.watchtower.yml")
    services = compose["services"]

    watchtower = services["watchtower"]
    assert watchtower["image"] == WATCHTOWER_IMAGE
    assert "--label-enable" in watchtower["command"]
    assert "--cleanup" in watchtower["command"]
    assert any(
        "/var/run/docker.sock:/var/run/docker.sock" == volume
        for volume in watchtower["volumes"]
    )
    _assert_hardened(watchtower, EXPECTED_WATCHTOWER_HARDENING)
    assert watchtower["labels"]["platform.security.docker-socket"] == (
        "watchtower-profile-local-updates-only"
    )
    assert watchtower["labels"]["platform.security.docker-socket-risk"] == (
        "host-daemon-access-is-not-production-isolation"
    )
    assert WATCHTOWER_ENABLE_LABEL not in watchtower.get("labels", {})


def test_watchtower_overlay_opts_in_only_control_plane_services() -> None:
    compose = _load_compose_file("compose.watchtower.yml")
    services = compose["services"]

    enabled_services = {
        name
        for name, service in services.items()
        if service.get("labels", {}).get(WATCHTOWER_ENABLE_LABEL) == "true"
    }
    assert enabled_services == {
        "master-admin",
        "master-proxy",
        "platform-docker-broker",
        "validator",
        "gpu-agent",
    }


def test_watchtower_overlay_does_not_opt_in_challenges_jobs_or_databases() -> None:
    base_compose = _load_compose_file("compose.yml")
    watchtower_compose = _load_compose_file("compose.watchtower.yml")
    forbidden_service_names = {
        "postgres",
        "db",
        "database",
        "challenge",
        "challenges",
        "job",
        "jobs",
        "worker",
        "broker-job",
    }

    for service_name, service in watchtower_compose["services"].items():
        labels = service.get("labels", {})
        if labels.get(WATCHTOWER_ENABLE_LABEL) == "true":
            service_name_parts = set(service_name.replace("-", "_").split("_"))
            assert service_name_parts.isdisjoint(forbidden_service_names)

    assert "postgres" not in base_compose["services"]
    assert "watchtower" not in base_compose["services"]
    assert all(
        WATCHTOWER_ENABLE_LABEL not in service.get("labels", {})
        for service in base_compose["services"].values()
    )
