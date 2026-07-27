"""TDD tests for HTTP SidecarAttestor client (recipe listen POST /v1/sidecar/attest).

Transport only: does not consume nonces. Fail-closed on malformed wire.
"""

from __future__ import annotations

from typing import cast

import httpx
import pytest
import respx

from base.compute.constation_sidecar_client import (
    ATTEST_PATH,
    HttpSidecarAttestor,
    SidecarAttestHit,
    SidecarAttestMiss,
)
from base.compute.constation_types import ConstationFailCode

BASE = "http://10.0.0.5:32001"
DIGEST = "sha256:" + ("ab" * 32)
NONCE = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
POD_ID = "pod-xyz"


def _valid_wire(
    *,
    nonce: str = NONCE,
    digest: str = DIGEST,
    phase: str = "start",
    pod_id: str = POD_ID,
) -> dict[str, object]:
    return {
        "schema_version": "prism_attestation_payload.v1",
        "algorithm": "hmac-sha256",
        "signature": "aa" * 32,
        "payload": {
            "nonce": nonce,
            "digest": digest,
            "pod_id": pod_id,
            "variant": "cpu",
            "sealed_manifest_hashes": {"/app/main.py": "bb" * 32},
            "build_secret_response": "cc" * 32,
        },
        "phase": phase,
        "hardware_root_of_trust": False,
        "sufficient_alone_for_tier_elevation": False,
        "proves": "entity_holding_in_image_secret_responded",
    }


def _client(**overrides: object) -> HttpSidecarAttestor:
    kwargs: dict[str, object] = {"base_url": BASE, "timeout_seconds": 2.0}
    kwargs.update(overrides)
    return HttpSidecarAttestor(**kwargs)  # type: ignore[arg-type]


@respx.mock
async def test_attest_200_valid_extracts_digest() -> None:
    """Given 200 valid wire; When attest; Then hit with payload digest."""
    route = respx.post(f"{BASE}{ATTEST_PATH}").mock(
        return_value=httpx.Response(200, json=_valid_wire(phase="interval"))
    )

    result = await _client().attest(nonce=NONCE, phase="interval")

    assert isinstance(result, SidecarAttestHit)
    assert result.digest == DIGEST
    assert result.nonce == NONCE
    assert result.pod_id == POD_ID
    assert result.phase == "interval"
    assert route.called
    assert route.calls.last.request.method == "POST"
    import json

    assert json.loads(route.calls.last.request.content.decode()) == {
        "nonce": NONCE,
        "phase": "interval",
    }


@respx.mock
async def test_attest_400_maps_to_sidecar_response_invalid() -> None:
    """Given HTTP 400; When attest; Then SIDECAR_RESPONSE_INVALID miss."""
    respx.post(f"{BASE}{ATTEST_PATH}").mock(
        return_value=httpx.Response(400, json={"error": "bad request"})
    )

    result = await _client().attest(nonce=NONCE, phase="start")

    assert isinstance(result, SidecarAttestMiss)
    assert result.reason is ConstationFailCode.SIDECAR_RESPONSE_INVALID


@respx.mock
async def test_attest_timeout_maps_to_connection_failure() -> None:
    """Given request timeout; When attest; Then typed connection/network fail."""
    respx.post(f"{BASE}{ATTEST_PATH}").mock(
        side_effect=httpx.TimeoutException("timed out")
    )

    result = await _client(timeout_seconds=0.5).attest(nonce=NONCE, phase="end")

    assert isinstance(result, SidecarAttestMiss)
    assert result.reason in {
        ConstationFailCode.SIDECAR_ATTEST_FAILED,
        ConstationFailCode.NETWORK_PARTITION,
    }


@respx.mock
async def test_attest_missing_digest_field_fail_closed() -> None:
    """Given 200 wire without payload.digest; When attest; Then invalid."""
    wire = _valid_wire()
    payload = dict(cast(dict[str, object], wire["payload"]))
    del payload["digest"]
    wire["payload"] = payload
    respx.post(f"{BASE}{ATTEST_PATH}").mock(return_value=httpx.Response(200, json=wire))

    result = await _client().attest(nonce=NONCE, phase="start")

    assert isinstance(result, SidecarAttestMiss)
    assert result.reason is ConstationFailCode.SIDECAR_RESPONSE_INVALID


@respx.mock
async def test_attest_blank_digest_fail_closed() -> None:
    """Given 200 wire with blank digest; When attest; Then invalid."""
    respx.post(f"{BASE}{ATTEST_PATH}").mock(
        return_value=httpx.Response(200, json=_valid_wire(digest="   "))
    )

    result = await _client().attest(nonce=NONCE, phase="start")

    assert isinstance(result, SidecarAttestMiss)
    assert result.reason is ConstationFailCode.SIDECAR_RESPONSE_INVALID


@respx.mock
async def test_attest_missing_payload_object_fail_closed() -> None:
    """Given 200 JSON without payload object; When attest; Then invalid."""
    respx.post(f"{BASE}{ATTEST_PATH}").mock(
        return_value=httpx.Response(
            200,
            json={
                "schema_version": "prism_attestation_payload.v1",
                "signature": "aa" * 32,
                "digest": DIGEST,
            },
        )
    )

    result = await _client().attest(nonce=NONCE, phase="start")

    assert isinstance(result, SidecarAttestMiss)
    assert result.reason is ConstationFailCode.SIDECAR_RESPONSE_INVALID


@respx.mock
async def test_attest_transport_error_maps_to_connection_failure() -> None:
    """Given connect error; When attest; Then SIDECAR_ATTEST_FAILED."""
    respx.post(f"{BASE}{ATTEST_PATH}").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    result = await _client().attest(nonce=NONCE, phase="start")

    assert isinstance(result, SidecarAttestMiss)
    assert result.reason is ConstationFailCode.SIDECAR_ATTEST_FAILED


@respx.mock
async def test_attest_does_not_call_nonce_service() -> None:
    """Given success; When attest; Then only HTTP POST (transport only)."""
    respx.post(f"{BASE}{ATTEST_PATH}").mock(
        return_value=httpx.Response(200, json=_valid_wire())
    )
    client = _client()

    result = await client.attest(nonce=NONCE, phase="start")

    assert isinstance(result, SidecarAttestHit)
    # No nonce consume API on client
    assert not hasattr(client, "consume")
    assert not hasattr(client, "nonce_service")


def test_rejects_non_positive_timeout() -> None:
    """Given timeout_seconds <= 0; When construct; Then ValueError."""
    with pytest.raises(ValueError, match="timeout"):
        HttpSidecarAttestor(base_url=BASE, timeout_seconds=0.0)


def test_strips_trailing_slash_on_base_url() -> None:
    """Given base_url with trailing slash; When construct; Then normalized."""
    client = HttpSidecarAttestor(base_url=f"{BASE}/", timeout_seconds=1.0)
    assert client.base_url == BASE
