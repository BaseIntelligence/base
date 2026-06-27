"""Unit tests for client-side validator request signing."""

from __future__ import annotations

from base.security.validator_auth import (
    canonical_validator_request,
)
from base.validator.agent.signing import (
    KeypairRequestSigner,
    build_signed_headers,
)


class _FakeKeypair:
    def __init__(self, ss58: str) -> None:
        self.ss58_address = ss58

    def sign(self, message: bytes) -> bytes:
        return b"sig:" + message[:4]


def test_build_signed_headers_matches_server_canonical() -> None:
    signed: dict[str, str] = {}

    class _RecordingSigner:
        hotkey = "permitted"

        def sign(self, message: bytes) -> str:
            signed["message"] = message.decode()
            return "deadbeef"

    headers = build_signed_headers(
        _RecordingSigner(),
        method="POST",
        path="/v1/validators/register",
        body=b'{"a":1}',
        nonce="n-1",
        timestamp=1_750_000_000,
    )
    expected_canonical = canonical_validator_request(
        method="POST",
        path="/v1/validators/register",
        query_string="",
        timestamp="1750000000",
        nonce="n-1",
        body=b'{"a":1}',
    )
    assert signed["message"] == expected_canonical
    assert headers["X-Hotkey"] == "permitted"
    assert headers["X-Signature"] == "deadbeef"
    assert headers["X-Nonce"] == "n-1"
    assert headers["X-Timestamp"] == "1750000000"


def test_build_signed_headers_generates_unique_nonce() -> None:
    class _Signer:
        hotkey = "permitted"

        def sign(self, message: bytes) -> str:
            return "x"

    first = build_signed_headers(
        _Signer(), method="POST", path="/v1/validators/heartbeat"
    )
    second = build_signed_headers(
        _Signer(), method="POST", path="/v1/validators/heartbeat"
    )
    assert first["X-Nonce"] != second["X-Nonce"]


def test_keypair_signer_returns_hex_signature() -> None:
    signer = KeypairRequestSigner(_FakeKeypair("hk-1"))
    assert signer.hotkey == "hk-1"
    signature = signer.sign(b"hello world")
    assert signature.startswith("0x")
    assert bytes.fromhex(signature[2:]) == b"sig:" + b"hell"
