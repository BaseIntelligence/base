"""Malformed framed KR sub-reasons + dstack GetQuote client normalize (offline).

Live residual (after CA PEM fix): host KR records
``key_release_deny reason=malformed_request`` with null ledger, while framed
RA-TLS is online. Causes collapse into one wire code:

1. host ``validate_framed_request`` / ratls cert ValuePaths all map to
   ``malformed_request`` without a secret-free scrapable sub-token;
2. guest ``GoldenKeyReleaseClient._extract_*`` emits RAW dstack GetQuote shapes
   (``0x`` quote_hex, empty IMR3 digests, base64 digests/payloads) that the host
   schema rejects; review already normalizes these.

No Phala/CVM create; no full suite. Discriminators:

- host detail tokens must appear ONLY on deny logs, never change wire
  ``reason_code`` away from ``malformed_request``;
- denials never embed sentinel key material/payload free-loaders;
- client-normalized dstack frames must pass ``validate_framed_request``;
- un-normalizable residue still raises typed ``KeyReleaseProtocolError``
  pre-send.
"""

from __future__ import annotations

import base64
import json

import pytest

from agent_challenge.keyrelease.client import (
    KeyReleaseProtocolError,
    _extract_event_log,
    _extract_quote_hex,
    _normalize_framed_event_log,
    _normalize_quote_hex,
)
from agent_challenge.keyrelease.quote import (
    COMPOSE_HASH_EVENT,
    DSTACK_RUNTIME_EVENT_TYPE,
    KEY_PROVIDER_EVENT,
    runtime_event_digest,
)
from agent_challenge.keyrelease.server import (
    REASON_MALFORMED_REQUEST,
    _log_key_release_deny,
    build_frame,
    parse_frame,
    validate_framed_request,
)

EVAL_RUN_ID = "eval-malform-1"
NONCE = "nonce-malform-1"
SENTINEL_KEY = "super-secret-golden-key-SENTINEL-xyz"


def _identity_event_log(*, digests: str | None = "filled") -> list[dict[str, object]]:
    compose_payload = "ab" * 32
    provider_payload = "7b226e616d65223a227068616c61227d"  # {"name":"phala"}
    bootstrap_payload = "11" * 16
    entries = [
        ("instance-id", bootstrap_payload),
        (COMPOSE_HASH_EVENT, compose_payload),
        ("boot-mr-done", bootstrap_payload),
        (KEY_PROVIDER_EVENT, provider_payload),
    ]
    out: list[dict[str, object]] = []
    for name, payload in entries:
        digest = runtime_event_digest(name, bytes.fromhex(payload)).hex()
        if digests == "empty":
            digest = ""
        out.append(
            {
                "imr": 3,
                "event_type": DSTACK_RUNTIME_EVENT_TYPE,
                "digest": digest,
                "event": name,
                "event_payload": payload,
            }
        )
    return out


def _valid_request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "schema_version": 1,
        "eval_run_id": EVAL_RUN_ID,
        "nonce": NONCE,
        "quote_hex": "aa" * 1200,
        "event_log": _identity_event_log(),
    }
    request.update(overrides)
    return request


def _frame_bytes(request: dict[str, object]) -> bytes:
    return build_frame(request)[4:]


# --------------------------------------------------------------------------- #
# (A) host sub-reason tokens on key_release_deny only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("maker", "detail"),
    [
        (lambda: b"", "frame_empty"),
        (lambda: b"{not-json", "frame_json"),
        (
            lambda: (
                json.dumps(_valid_request(), separators=(",", ":"), sort_keys=False).encode()
                # non-canonical field order will be wrong unless it reorders; force
                # non-canonical by pretty-printing with spaces
                if False
                else json.dumps(_valid_request(), indent=2).encode()
            ),
            "frame_canonical",
        ),
        (
            lambda: _frame_bytes(
                {
                    "schema_version": 1,
                    "eval_run_id": EVAL_RUN_ID,
                    "nonce": NONCE,
                    "quote_hex": "aa" * 16,
                    # missing event_log / schema-closed field set
                }
            ),
            "frame_fields",
        ),
        (
            lambda: _frame_bytes(_valid_request(eval_run_id="has space")),
            "frame_ids",
        ),
        (
            lambda: _frame_bytes(_valid_request(quote_hex="0x" + "AA" * 32)),
            "quote_hex",
        ),
        (
            lambda: _frame_bytes(
                _valid_request(
                    event_log=[{"imr": 3, "event": "compose-hash"}]  # not closed
                )
            ),
            "event_log",
        ),
    ],
)
def test_validate_framed_request_raises_malformed_with_detail(maker, detail) -> None:
    from agent_challenge.keyrelease.server import MalformedFrameError

    with pytest.raises(MalformedFrameError) as excinfo:
        validate_framed_request(maker())
    # Wire reason stays one code; sub-token is only on the exception detail.
    assert str(excinfo.value) == REASON_MALFORMED_REQUEST
    assert excinfo.value.detail == detail


def test_parse_frame_empty_and_json_details() -> None:
    from agent_challenge.keyrelease.server import MalformedFrameError

    with pytest.raises(MalformedFrameError) as empty:
        parse_frame(b"")
    assert empty.value.detail == "frame_empty"

    with pytest.raises(MalformedFrameError) as bad_json:
        parse_frame(b"not-json")
    assert bad_json.value.detail == "frame_json"


def test_key_release_deny_log_includes_detail_token_secret_free(capsys) -> None:
    _log_key_release_deny(
        reason=REASON_MALFORMED_REQUEST,
        eval_run_id=EVAL_RUN_ID,
        detail="event_log",
    )
    _log_key_release_deny(
        reason=REASON_MALFORMED_REQUEST,
        eval_run_id=f"{EVAL_RUN_ID}/{SENTINEL_KEY}",
        detail=f"quote_hex {SENTINEL_KEY}",
    )
    err = capsys.readouterr().err
    assert (
        f"key_release_deny reason=malformed_request eval_run_id={EVAL_RUN_ID} detail=event_log"
    ) in err
    # Sanitizer keeps reason + detail as token-ish identifiers only.
    assert SENTINEL_KEY not in err
    assert "detail=quote_hex" in err
    # Non-malformed denials still omit a colliding free-form detail.
    _log_key_release_deny(
        reason="measurement_not_allowlisted",
        eval_run_id="eval-ok",
    )
    err2 = capsys.readouterr().err
    assert "detail=" not in err2
    assert "key_release_deny reason=measurement_not_allowlisted" in err2


def test_malformed_detail_tokens_are_closed_set() -> None:
    from agent_challenge.keyrelease.server import MALFORMED_DETAIL_TOKENS

    assert MALFORMED_DETAIL_TOKENS == frozenset(
        {
            "frame_empty",
            "frame_json",
            "frame_canonical",
            "frame_fields",
            "frame_ids",
            "quote_hex",
            "event_log",
            "ratls_cert",
        }
    )


# --------------------------------------------------------------------------- #
# (B) client GetQuote normalize
# --------------------------------------------------------------------------- #


def test_normalize_quote_hex_strips_0x_and_lowercases() -> None:
    assert _normalize_quote_hex("0xDeAdBeEf") == "deadbeef"
    assert _normalize_quote_hex("0XAABB") == "aabb"
    assert _extract_quote_hex({"quote": "0x" + "Aa" * 16}) == "aa" * 16


def test_normalize_quote_hex_rejects_non_hex() -> None:
    with pytest.raises(KeyReleaseProtocolError):
        _normalize_quote_hex("0xzzzz")
    with pytest.raises(KeyReleaseProtocolError):
        _normalize_quote_hex("abc")  # odd length


def test_normalize_event_log_fills_empty_imr3_and_closed_keys() -> None:
    raw = _identity_event_log(digests="empty")
    # dstack extras / casing / 0x prefixes that live guests emit
    raw[0]["digest"] = ""
    raw[1]["digest"] = "0x"  # blank after strip
    raw[1]["event_payload"] = "0x" + str(raw[1]["event_payload"]).upper()
    raw[2]["extra_dstack_field"] = "drop-me"
    normalized = _normalize_framed_event_log(raw)
    closed = {"imr", "event_type", "digest", "event", "event_payload"}
    assert all(set(entry) == closed for entry in normalized)
    assert all(
        entry["digest"] and entry["digest"] == entry["digest"].lower() for entry in normalized
    )
    assert all(not str(entry["digest"]).startswith("0x") for entry in normalized)
    # Recomputed digests match sealed runtime_event_digest formula.
    for entry in normalized:
        expected = runtime_event_digest(
            str(entry["event"]),
            bytes.fromhex(str(entry["event_payload"])),
        ).hex()
        assert entry["digest"] == expected


def test_normalize_event_log_decodes_base64_digests_and_payloads() -> None:
    """dstack-shaped base64 digests/payloads coerce to lowercase even hex."""

    compose_payload = bytes.fromhex("cd" * 32)
    provider_payload = b'{"name":"phala"}'
    compose_digest = runtime_event_digest(COMPOSE_HASH_EVENT, compose_payload)
    provider_digest = runtime_event_digest(KEY_PROVIDER_EVENT, provider_payload)
    raw = [
        {
            "imr": 3,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "digest": base64.b64encode(compose_digest).decode(),
            "event": COMPOSE_HASH_EVENT,
            "event_payload": base64.b64encode(compose_payload).decode(),
        },
        {
            "imr": 3,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "digest": base64.b64encode(provider_digest).decode(),
            "event": KEY_PROVIDER_EVENT,
            "event_payload": base64.b64encode(provider_payload).decode(),
        },
    ]
    normalized = _normalize_framed_event_log(raw)
    assert normalized[0]["digest"] == compose_digest.hex()
    assert normalized[0]["event_payload"] == compose_payload.hex()
    assert normalized[1]["digest"] == provider_digest.hex()
    assert normalized[1]["event_payload"] == provider_payload.hex()


def test_dstack_shaped_getquote_frame_passes_validate_framed_request() -> None:
    """End-to-end offline: client normalize → canonical frame → host validate."""

    compose_payload = bytes.fromhex("ef" * 32)
    provider_payload = b'{"name":"phala"}'
    bootstrap = bytes.fromhex("22" * 8)
    # RAW dstack GetQuote-ish: 0x casing, empty digests, extra keys, Base64 payloads
    raw_events = [
        {
            "imr": "3",  # string imr as some guests emit
            "event_type": str(DSTACK_RUNTIME_EVENT_TYPE),
            "digest": "",
            "event": "instance-id",
            "event_payload": bootstrap.hex().upper(),
            "timestamp": 1,
        },
        {
            "imr": 3,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "digest": "",
            "event": COMPOSE_HASH_EVENT,
            "event_payload": "0x" + compose_payload.hex().upper(),
        },
        {
            "imr": 3,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "digest": "",
            "event": KEY_PROVIDER_EVENT,
            "event_payload": base64.b64encode(provider_payload).decode(),
        },
    ]
    quote = "0x" + ("Ab" * 1200)

    quote_hex = _normalize_quote_hex(quote)
    event_log = _normalize_framed_event_log(raw_events)
    # extract helpers used by acquire_golden_key work the same path
    assert _extract_quote_hex({"quote": quote}) == quote_hex
    assert _extract_event_log({"event_log": raw_events}) == event_log

    frame = _frame_bytes(
        {
            "schema_version": 1,
            "eval_run_id": EVAL_RUN_ID,
            "nonce": NONCE,
            "quote_hex": quote_hex,
            "event_log": event_log,
        }
    )
    validated = validate_framed_request(frame)
    assert validated["quote_hex"] == "ab" * 1200
    assert validated["event_log"][1]["event"] == COMPOSE_HASH_EVENT
    assert validated["event_log"][2]["event"] == KEY_PROVIDER_EVENT


def test_client_fails_closed_pre_send_when_event_log_still_invalid() -> None:
    """Normalization that cannot produce a schema-closed log fails typed."""

    with pytest.raises(KeyReleaseProtocolError):
        _normalize_framed_event_log(
            [
                {
                    "imr": 3,
                    "event_type": DSTACK_RUNTIME_EVENT_TYPE,
                    "digest": "",
                    "event": "only-one",
                    "event_payload": "aa",
                }
            ],
            enforce_schema=True,
        )


def test_extract_event_log_json_string_normalizes() -> None:
    raw = _identity_event_log(digests="empty")
    # as dstack returns JSON string
    normalized = _extract_event_log({"event_log": json.dumps(raw)})
    assert normalized[0]["digest"]
    assert len(normalized) == 4
