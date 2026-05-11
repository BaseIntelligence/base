from __future__ import annotations

from pathlib import Path

import yaml


def test_compose_deploy_has_expected_services_and_socket_scope() -> None:
    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "docker/compose.yml").read_text(
            encoding="utf-8"
        )
    )
    services = compose["services"]

    for name in (
        "postgres",
        "master-admin",
        "master-proxy",
        "platform-docker-broker",
        "validator",
        "gpu-agent",
    ):
        assert name in services

    socket_services = {
        name
        for name, service in services.items()
        if any(
            "/var/run/docker.sock" in volume for volume in service.get("volumes", [])
        )
    }
    assert socket_services == {
        "master-admin",
        "platform-docker-broker",
        "gpu-agent",
    }
    assert "platform_challenges" in services["platform-docker-broker"]["networks"]
    assert compose["networks"]["platform_challenges"]["attachable"] is True
    assert "platform_state" in compose["volumes"]
    assert "platform_secrets" in compose["volumes"]
