"""Validator key-release authority forms (plan wire + RA-TLS bake).

Import-light helper shared by :mod:`eval_wire` (lean CVM) and
:mod:`compose` (measure-time bake). Free HTTP(S) KR URLs are never a plan
trust root (VAL-ACLOCK-008/009/010).
"""

from __future__ import annotations

DEFAULT_KEY_RELEASE_RA_TLS_PORT = 8701


def parse_key_release_authority(endpoint: str) -> tuple[str, int] | None:
    """Return ``(host, port)`` for validator KR / RA-TLS authority forms only.

    Accepts:

    * ``host:port`` (any valid TCP port)
    * ``ratls://host:port``, ``tls://host:port``, ``tcp://host:port``

    Rejects free HTTP(S) URLs, path-bearing authorities, userinfo, empty host,
    and unknown schemes.
    """

    if not isinstance(endpoint, str):
        return None
    value = endpoint.strip()
    if not value or len(value) > 16_384:
        return None
    scheme = ""
    authority = value
    if "://" in value:
        scheme, authority = value.split("://", 1)
        scheme = scheme.lower()
        if scheme in {"http", "https"}:
            return None
        if scheme not in {"ratls", "tls", "tcp"}:
            return None
    if any(sep in authority for sep in ("/", "?", "#", "@")):
        return None
    if ":" not in authority:
        return None
    host, port_text = authority.rsplit(":", 1)
    try:
        port = int(port_text)
    except ValueError:
        return None
    host = host.strip().strip("[]")
    if not host or not 1 <= port <= 65535:
        return None
    if any(ch.isspace() for ch in host) or any(ord(ch) < 32 for ch in host):
        return None
    return host, port


def is_validator_key_release_authority(endpoint: str) -> bool:
    """True when ``endpoint`` is an honest validator KR / RA-TLS authority form."""

    return parse_key_release_authority(endpoint) is not None


def parse_raw_key_release_bake(endpoint: str) -> tuple[str, int] | None:
    """Return ``(host, port)`` when endpoint should bake static RA-TLS HOST/PORT.

    Same authority forms as :func:`parse_key_release_authority`, but bare
    ``host:port`` without a raw scheme only bakes when ``port`` is the product
    RA-TLS listener port (:data:`DEFAULT_KEY_RELEASE_RA_TLS_PORT`). HTTP(S)
    measure-time placeholders return ``None`` (static free-URL bake path only).
    """

    parsed = parse_key_release_authority(endpoint)
    if parsed is None:
        return None
    host, port = parsed
    value = endpoint.strip()
    scheme = ""
    if "://" in value:
        scheme = value.split("://", 1)[0].lower()
    if not scheme and port != DEFAULT_KEY_RELEASE_RA_TLS_PORT:
        return None
    return host, port


__all__ = [
    "DEFAULT_KEY_RELEASE_RA_TLS_PORT",
    "is_validator_key_release_authority",
    "parse_key_release_authority",
    "parse_raw_key_release_bake",
]
