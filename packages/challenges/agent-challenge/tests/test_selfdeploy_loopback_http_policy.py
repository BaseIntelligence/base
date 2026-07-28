"""Loopback-only insecure HTTP opt-in for SelfDeployRouteClient.

``http://`` is allowed only for loopback hosts AND only when
``SELFDEPLOY_ALLOW_INSECURE_LOOPBACK=1``. Everything else keeps raising
exactly as before (``challenge endpoint must use https://``).
"""

from __future__ import annotations

import pytest

from agent_challenge.selfdeploy.client import RouteClientError, SelfDeployRouteClient


def test_https_base_url_always_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SELFDEPLOY_ALLOW_INSECURE_LOOPBACK", raising=False)
    client = SelfDeployRouteClient("https://chain.joinbase.ai/challenges/agent-challenge")
    assert client._base_url.startswith("https://")


def test_http_loopback_rejected_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SELFDEPLOY_ALLOW_INSECURE_LOOPBACK", raising=False)
    with pytest.raises(RouteClientError, match="must use https://"):
        SelfDeployRouteClient("http://127.0.0.1:18082")


@pytest.mark.parametrize(
    "base",
    [
        "http://127.0.0.1:18082",
        "http://localhost:18082/challenges/agent-challenge",
        "http://[::1]:18082",
    ],
)
def test_http_loopback_accepted_with_opt_in(
    monkeypatch: pytest.MonkeyPatch, base: str
) -> None:
    monkeypatch.setenv("SELFDEPLOY_ALLOW_INSECURE_LOOPBACK", "1")
    client = SelfDeployRouteClient(base)
    assert client._base_url.startswith("http://")


@pytest.mark.parametrize(
    "base",
    [
        "http://example.com",
        "http://10.0.0.1:8080",
        "http://192.168.1.1",
        "http://challenge.joinbase.ai",
    ],
)
def test_http_non_loopback_rejected_even_with_opt_in(
    monkeypatch: pytest.MonkeyPatch, base: str
) -> None:
    monkeypatch.setenv("SELFDEPLOY_ALLOW_INSECURE_LOOPBACK", "1")
    with pytest.raises(RouteClientError, match="must use https://"):
        SelfDeployRouteClient(base)


def test_opt_in_requires_exact_value_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELFDEPLOY_ALLOW_INSECURE_LOOPBACK", "true")
    with pytest.raises(RouteClientError, match="must use https://"):
        SelfDeployRouteClient("http://127.0.0.1:18082")
