"""TDD tests for pure Lium pod → sidecar HTTP base URL resolution.

Fail-closed: never jupyter_url, never ssh_connect_cmd, never port guessing.
"""

from __future__ import annotations

from typing import Any

from base.compute.constation_sidecar_endpoint import (
    SidecarEndpointHit,
    SidecarEndpointMiss,
    SidecarEndpointMissReason,
    resolve_sidecar_base_url,
)

_INTERNAL = 8787
_HOST = "1.2.3.4"
_EXTERNAL = 32001
_EXPECTED = f"http://{_HOST}:{_EXTERNAL}"


def _pod(
    *,
    executor_ip: object = _HOST,
    ports_mapping: object = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    default_mapping: dict[str, int] = {"8787": _EXTERNAL}
    raw: dict[str, Any] = {
        "executor": {"executor_ip_address": executor_ip},
        "ports_mapping": (default_mapping if ports_mapping is None else ports_mapping),
        "status": "RUNNING",
        "template": {
            "internal_ports": [_INTERNAL],
            "docker_image_digest": "sha256:" + ("a" * 64),
        },
    }
    if extra:
        raw.update(extra)
    return raw


def test_resolves_from_dict_ports_mapping() -> None:
    """Given dict ports_mapping, When resolve, Then http://host:external."""
    result = resolve_sidecar_base_url(_pod(), internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointHit)
    assert result.base_url == _EXPECTED


def test_ports_mapping_json_string_parsed() -> None:
    """Given ports_mapping as JSON string, When resolve, Then parsed and hit."""
    raw = _pod(ports_mapping='{"8787": 32001}')

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointHit)
    assert result.base_url == _EXPECTED


def test_ports_mapping_empty_string_default_fails() -> None:
    """Given ports_mapping == '{}', When resolve, Then sidecar_port_unpublished."""
    raw = _pod(ports_mapping="{}")

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointMiss)
    assert result.reason is SidecarEndpointMissReason.SIDECAR_PORT_UNPUBLISHED


def test_string_and_int_keys_both_match() -> None:
    """Given int key 8787, When resolve with internal_port 8787, Then hit."""
    raw = _pod(ports_mapping={8787: _EXTERNAL})

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointHit)
    assert result.base_url == _EXPECTED


def test_hostport_dict_value_form() -> None:
    """Given HostPort nested dict value, When resolve, Then external port extracted."""
    raw = _pod(ports_mapping={"8787": {"HostPort": "32001"}})

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointHit)
    assert result.base_url == _EXPECTED


def test_hostport_list_value_form() -> None:
    """Given list of HostPort dicts, When resolve, Then first valid port used."""
    raw = _pod(ports_mapping={"8787": [{"HostPort": "32001"}]})

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointHit)
    assert result.base_url == _EXPECTED


def test_missing_internal_port_fails() -> None:
    """Given mapping without internal_port key, When resolve, Then unpublished."""
    raw = _pod(ports_mapping={"22": 22001})

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointMiss)
    assert result.reason is SidecarEndpointMissReason.SIDECAR_PORT_UNPUBLISHED


def test_missing_executor_ip_fails() -> None:
    """Given blank executor_ip_address, When resolve, Then pod_endpoint_missing."""
    raw = _pod(executor_ip="")

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointMiss)
    assert result.reason is SidecarEndpointMissReason.POD_ENDPOINT_MISSING


def test_unparseable_ports_mapping_fails() -> None:
    """Given invalid JSON ports_mapping string, When resolve, Then unparseable."""
    raw = _pod(ports_mapping="{not-json")

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointMiss)
    assert result.reason is SidecarEndpointMissReason.PORTS_MAPPING_UNPARSEABLE


def test_out_of_range_port_fails() -> None:
    """Given external port outside 1..65535, When resolve, Then unpublished."""
    raw = _pod(ports_mapping={"8787": 70000})

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointMiss)
    assert result.reason is SidecarEndpointMissReason.SIDECAR_PORT_UNPUBLISHED


def test_never_falls_back_to_jupyter_url() -> None:
    """Given jupyter_url present but no mapped sidecar port, When resolve, Then fail."""
    raw = _pod(
        ports_mapping={},
        extra={"jupyter_url": "https://jupyter.example/lab?token=abc"},
    )

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointMiss)
    assert result.reason is SidecarEndpointMissReason.SIDECAR_PORT_UNPUBLISHED


def test_never_uses_ssh_connect_cmd() -> None:
    """Given ssh_connect_cmd but no mapped sidecar port, When resolve, Then fail."""
    raw = _pod(
        ports_mapping={},
        extra={"ssh_connect_cmd": f"ssh root@{_HOST} -p 22001"},
    )

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointMiss)
    assert result.reason is SidecarEndpointMissReason.SIDECAR_PORT_UNPUBLISHED


def test_ipv6_host_is_bracketed() -> None:
    """Given IPv6 executor_ip_address, When resolve, Then bracketed host in URL."""
    ipv6 = "2001:db8::1"
    raw = _pod(executor_ip=ipv6, ports_mapping={"8787": _EXTERNAL})

    result = resolve_sidecar_base_url(raw, internal_port=_INTERNAL)

    assert isinstance(result, SidecarEndpointHit)
    assert result.base_url == f"http://[{ipv6}]:{_EXTERNAL}"
