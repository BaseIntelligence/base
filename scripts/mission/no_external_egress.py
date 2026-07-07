"""Offline network-egress guard for the flags-OFF legacy regression (VAL-CROSS-006).

Import this module as a pytest plugin (``-p no_external_egress``) to prove a test run
performs ZERO real network egress: any DNS resolution or socket connect to a
non-loopback host raises :class:`ExternalEgressBlocked` instead of touching the
network. Loopback (``127.0.0.0/8`` / ``::1`` / ``localhost``) and AF_UNIX sockets stay
allowed so the local test PostgreSQL (127.0.0.1:15433) and in-process fixtures keep
working, while a stray call to ``lium.io`` / ``api.targon.com`` (or anything else
external) fails loudly.

Usage (run from the base repo, with the test PostgreSQL on 15433 already up)::

    PYTHONPATH="$PWD/scripts/mission" \
      BASE_TEST_DATABASE_URL=postgresql+asyncpg://base:base@localhost:15433/base_test \
      uv run pytest -p no_external_egress -q -p no:cacheprovider

It installs from the ``pytest_configure`` hook, which runs before collection, so even
a test module that probes the network at import time is guarded. Importing the module
by itself has no side effects, and the guard is a no-op to install twice. NOT for
production; a mission verification harness only.
"""

from __future__ import annotations

import ipaddress
import socket

_ALLOWED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


class ExternalEgressBlocked(OSError):
    """A test attempted real network egress to a non-loopback host.

    Subclasses :class:`OSError` (not a bare ``RuntimeError``) so the standard
    "is the network up?" probe -- ``try: socket.create_connection(...) except OSError``
    -- treats an external host as unreachable and SKIPS, exactly as it would in an
    environment with external resolution disabled, instead of erroring collection.
    """


def _host_is_loopback(host: object) -> bool:
    if host is None:
        # AF_UNIX and empty binds carry no host; never external.
        return True
    text = str(host)
    if text == "":
        return True
    if text in _ALLOWED_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        # A DNS name that is not an explicit localhost alias: treat as external.
        return False


def _guard_address(address: object) -> None:
    # Only INET/INET6 addresses are (host, port[, ...]) tuples. AF_UNIX addresses are
    # str/bytes filesystem paths (e.g. multiprocessing's /tmp/pymp-* sockets) and carry
    # no host, so they are never external and must not be blocked.
    if not (isinstance(address, (tuple, list)) and address):
        return
    host = address[0]
    if not _host_is_loopback(host):
        raise ExternalEgressBlocked(
            f"blocked real network egress to {host!r} "
            "(offline legacy regression: only loopback is permitted)"
        )


_REAL_GETADDRINFO = socket.getaddrinfo
_REAL_CREATE_CONNECTION = socket.create_connection
_REAL_SOCKET_CONNECT = socket.socket.connect
_REAL_SOCKET_CONNECT_EX = socket.socket.connect_ex


def _guarded_getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
    if not _host_is_loopback(host):
        raise ExternalEgressBlocked(
            f"blocked DNS resolution for {host!r} "
            "(offline legacy regression: only loopback is permitted)"
        )
    return _REAL_GETADDRINFO(host, *args, **kwargs)


def _guarded_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
    _guard_address(address)
    return _REAL_CREATE_CONNECTION(address, *args, **kwargs)


def _guarded_socket_connect(self, address):  # type: ignore[no-untyped-def]
    _guard_address(address)
    return _REAL_SOCKET_CONNECT(self, address)


def _guarded_socket_connect_ex(self, address):  # type: ignore[no-untyped-def]
    _guard_address(address)
    return _REAL_SOCKET_CONNECT_EX(self, address)


def install() -> None:
    """Block non-loopback egress via stdlib socket monkeypatches (idempotent)."""

    if getattr(socket, "_no_external_egress_installed", False):
        return
    socket.getaddrinfo = _guarded_getaddrinfo  # type: ignore[assignment]
    socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
    socket.socket.connect = _guarded_socket_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_socket_connect_ex  # type: ignore[method-assign]
    socket._no_external_egress_installed = True  # type: ignore[attr-defined]


def uninstall() -> None:
    """Restore the original stdlib socket entry points (idempotent, for teardown)."""

    if not getattr(socket, "_no_external_egress_installed", False):
        return
    socket.getaddrinfo = _REAL_GETADDRINFO  # type: ignore[assignment]
    socket.create_connection = _REAL_CREATE_CONNECTION  # type: ignore[assignment]
    socket.socket.connect = _REAL_SOCKET_CONNECT  # type: ignore[method-assign]
    socket.socket.connect_ex = _REAL_SOCKET_CONNECT_EX  # type: ignore[method-assign]
    socket._no_external_egress_installed = False  # type: ignore[attr-defined]


def pytest_configure(config: object) -> None:  # noqa: ARG001 - pytest hook
    """Install the guard before collection so the session performs zero external egress.

    ``pytest_configure`` runs before collection, so even a module that probes the
    network at import time is guarded. Importing this module has NO side effects on its
    own -- it guards only once loaded as a pytest plugin (``-p no_external_egress``).
    """

    install()
