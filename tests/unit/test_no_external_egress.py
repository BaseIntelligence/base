"""Unit tests for the offline network-egress guard (VAL-CROSS-006 part a).

The guard (``scripts/mission/no_external_egress.py``) is the reusable plugin the
flags-OFF legacy regression uses to prove BOTH repos' default suites run with ZERO real
egress to lium.io / api.targon.com: it blocks any DNS/connect to a non-loopback host
while leaving loopback (the local test PostgreSQL, in-process stubs) and AF_UNIX sockets
working.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "mission"))

import no_external_egress as guard  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_socket():
    guard.uninstall()
    yield
    guard.uninstall()


def test_error_is_oserror_subclass_so_probes_skip_not_error() -> None:
    # A "is the network up?" probe catches OSError and skips; the guard must not surface
    # as a RuntimeError that would error test collection instead.
    assert issubclass(guard.ExternalEgressBlocked, OSError)


def test_blocks_external_dns_resolution() -> None:
    guard.install()
    for host in ("lium.io", "api.targon.com", "huggingface.co", "example.com"):
        with pytest.raises(guard.ExternalEgressBlocked):
            socket.getaddrinfo(host, 443)


def test_allows_loopback_resolution() -> None:
    guard.install()
    assert socket.getaddrinfo("127.0.0.1", 15433)
    assert socket.getaddrinfo("localhost", 80)
    assert socket.getaddrinfo("::1", 80)


def test_blocks_external_socket_connect() -> None:
    guard.install()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(guard.ExternalEgressBlocked):
            sock.connect(("1.1.1.1", 80))
    finally:
        sock.close()


def test_afunix_and_loopback_addresses_are_not_blocked() -> None:
    guard.install()
    # AF_UNIX addresses are filesystem paths (multiprocessing's /tmp/pymp-* sockets),
    # never external, so they must pass the address guard untouched.
    guard._guard_address("/tmp/pymp-abc123/listener-0")
    guard._guard_address(("127.0.0.1", 15433))
    guard._guard_address(("::1", 8080, 0, 0))


def test_httpx_call_to_provider_host_is_blocked() -> None:
    httpx = pytest.importorskip("httpx")
    guard.install()
    # httpx maps the underlying OSError from the transport to its own ConnectError, but
    # the guard's message is preserved -- the real egress never leaves the host.
    with pytest.raises(httpx.ConnectError) as excinfo:
        httpx.get("https://lium.io/api/users/me", timeout=5)
    assert "blocked real network egress" in str(excinfo.value)


def test_install_is_idempotent_and_uninstall_restores() -> None:
    original = socket.getaddrinfo
    guard.install()
    guard.install()
    assert socket.getaddrinfo is not original
    guard.uninstall()
    assert socket.getaddrinfo is original
