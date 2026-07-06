"""Secret-hygiene tests across BOTH compute provider clients (VAL-PROV-006).

Constructs each client with a sentinel API key, exercises a successful mocked
call and a failing (HTTP 500) mocked call, then asserts the sentinel never leaks
into ``repr``, ``str``, any log record (root logger, DEBUG level), or the
stringified raised exception.
"""

from __future__ import annotations

import logging

import httpx
import pytest
import respx

from base.compute import LiumClient, LiumError, TargonClient, TargonError

LIUM_BASE = "https://lium.io/api"
TARGON_BASE = "https://api.targon.com/tha/v2"

LIUM_SENTINEL = "SENTINEL-LIUM-KEY-XYZ"
TARGON_SENTINEL = "SENTINEL-TARGON-KEY-XYZ"


@respx.mock
async def test_lium_client_never_leaks_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = LiumClient(LIUM_SENTINEL)
    assert LIUM_SENTINEL not in repr(client)
    assert LIUM_SENTINEL not in str(client)

    with caplog.at_level(logging.DEBUG, logger=""):
        # Successful call.
        respx.get(f"{LIUM_BASE}/users/me").mock(
            return_value=httpx.Response(200, json={"balance": 1.0})
        )
        assert await client.balance() == pytest.approx(1.0)

        # Failing call (HTTP 500).
        respx.get(f"{LIUM_BASE}/pods").mock(return_value=httpx.Response(500))
        with pytest.raises(LiumError) as exc_info:
            await client.list_pods()

    assert LIUM_SENTINEL not in str(exc_info.value)
    assert LIUM_SENTINEL not in repr(exc_info.value)
    assert LIUM_SENTINEL not in caplog.text


@respx.mock
async def test_targon_client_never_leaks_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = TargonClient(TARGON_SENTINEL)
    assert TARGON_SENTINEL not in repr(client)
    assert TARGON_SENTINEL not in str(client)

    with caplog.at_level(logging.DEBUG, logger=""):
        # Successful call.
        respx.get(f"{TARGON_BASE}/workloads").mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        assert await client.list_workloads() == []

        # Failing call (HTTP 500).
        respx.get(f"{TARGON_BASE}/apps").mock(return_value=httpx.Response(500))
        with pytest.raises(TargonError) as exc_info:
            await client.list_apps()

    assert TARGON_SENTINEL not in str(exc_info.value)
    assert TARGON_SENTINEL not in repr(exc_info.value)
    assert TARGON_SENTINEL not in caplog.text


@respx.mock
async def test_targon_deploy_failure_never_leaks_key() -> None:
    client = TargonClient(TARGON_SENTINEL)
    respx.post(f"{TARGON_BASE}/workloads/deploy").mock(
        return_value=httpx.Response(402, json={"error": "payment required"})
    )
    with pytest.raises(TargonError) as exc_info:
        await client.deploy({"name": "x"})
    assert TARGON_SENTINEL not in str(exc_info.value)
    assert TARGON_SENTINEL not in repr(exc_info.value)
