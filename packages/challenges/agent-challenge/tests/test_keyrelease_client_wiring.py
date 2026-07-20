"""Client wiring for an end-to-end golden key-release (event_log + RA-TLS key).

The in-CVM client must attach, in its ``/release`` request, the dstack
``cc-eventlog`` (so the validator can replay RTMR3, VAL-KEY-014) and a real
RA-TLS session public key bound into ``report_data`` (VAL-KEY-012/016). These
tests drive the client with a quote provider that returns an event log/vm_config
(as the dstack SDK does) and capture the exact bytes the client sends, so the
request the server already accepts is actually produced end-to-end.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from agent_challenge.keyrelease.client import (
    GoldenKeyReleaseClient,
    KeyReleaseProtocolError,
    key_release_report_data,
)

GOLDEN_KEY = b"golden-decryption-key-0123456789"
FAKE_QUOTE = "ab" * 64
EVENT_LOG = [
    {"imr": 3, "event": "compose-hash", "event_payload": "ab" * 32},
    {"imr": 3, "event": "key-provider", "event_payload": "cd" * 8},
]


class _QuoteWithEventLog:
    """Mirrors a dstack ``get_quote`` response (quote + event_log + vm_config)."""

    def __init__(
        self,
        *,
        quote: str = FAKE_QUOTE,
        event_log: Any = None,
        vm_config: Any = None,
    ) -> None:
        self.quote = quote
        self.event_log = event_log
        self.vm_config = vm_config
        self.report_data_seen: bytes | None = None

    def get_quote(self, report_data: bytes):
        self.report_data_seen = report_data
        return self


class _CapturingRelease:
    def __init__(self, *, key: bytes = GOLDEN_KEY) -> None:
        self.captured: dict[str, Any] | None = None
        self._key = key

    def __call__(self, request):
        self.captured = json.loads(request.data.decode())

        class _Resp:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Resp(json.dumps({"key": base64.b64encode(self._key).decode()}).encode())


def _urlopen(nonce_body: bytes, release_handler):
    def _open(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)

        class _Resp:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        if url.endswith("/nonce"):
            return _Resp(nonce_body)
        return release_handler(request)

    return _open


def _client(provider, release_handler, *, ra_tls_pubkey=b"") -> GoldenKeyReleaseClient:
    return GoldenKeyReleaseClient(
        "https://validator.test:8700",
        quote_provider=provider,
        ra_tls_pubkey=ra_tls_pubkey,
        urlopen=_urlopen(json.dumps({"nonce": "fresh-nonce"}).encode(), release_handler),
    )


def test_release_payload_attaches_event_log_and_ra_tls_pubkey():
    provider = _QuoteWithEventLog(event_log=EVENT_LOG)
    capture = _CapturingRelease()
    key = _client(provider, capture, ra_tls_pubkey=b"enclave-pub").acquire_golden_key()

    assert key == GOLDEN_KEY
    assert capture.captured is not None
    # The dstack cc-eventlog is attached for the server's RTMR3 replay.
    assert capture.captured["event_log"] == EVENT_LOG
    # A real RA-TLS public key is sent and bound into report_data.
    assert capture.captured["ra_tls_pubkey"] == b"enclave-pub".hex()
    assert provider.report_data_seen == key_release_report_data("fresh-nonce", b"enclave-pub")


def test_event_log_json_string_is_coerced_to_list():
    provider = _QuoteWithEventLog(event_log=json.dumps(EVENT_LOG))
    capture = _CapturingRelease()
    _client(provider, capture, ra_tls_pubkey=b"pub").acquire_golden_key()
    assert capture.captured["event_log"] == EVENT_LOG


def test_vm_config_is_forwarded_when_present():
    provider = _QuoteWithEventLog(event_log=EVENT_LOG, vm_config={"os_image_hash": "ab" * 32})
    capture = _CapturingRelease()
    _client(provider, capture, ra_tls_pubkey=b"pub").acquire_golden_key()
    assert capture.captured["vm_config"] == {"os_image_hash": "ab" * 32}


def test_missing_event_log_is_sent_as_empty_list():
    provider = _QuoteWithEventLog(event_log=None)
    capture = _CapturingRelease()
    _client(provider, capture, ra_tls_pubkey=b"pub").acquire_golden_key()
    assert capture.captured["event_log"] == []


def test_malformed_event_log_json_fails_closed():
    provider = _QuoteWithEventLog(event_log="{not json")
    capture = _CapturingRelease()
    with pytest.raises(KeyReleaseProtocolError):
        _client(provider, capture, ra_tls_pubkey=b"pub").acquire_golden_key()


def test_non_list_event_log_fails_closed():
    provider = _QuoteWithEventLog(event_log={"not": "a list"})
    capture = _CapturingRelease()
    with pytest.raises(KeyReleaseProtocolError):
        _client(provider, capture, ra_tls_pubkey=b"pub").acquire_golden_key()
