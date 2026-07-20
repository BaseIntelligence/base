"""quote_measurement_mismatch field-class diag (message words → closed token).

Live residual after or_outcome_bind tip (7e5497c3): dry-run allowlist was IN-LIST
but measured guest returned durable reason_code=quote_measurement_mismatch with
public_logs that did not say which register/field mismatched.

Product Mode B: map known ValueError message words from review_runtime
``_measurement_from_quote`` onto a closed diag token
{compose,key_provider,os,mrtd,rtmr0,rtmr1,rtmr2,rtmr3,event_log,other}
attached to OpenRouterTransportError + bounded_review_failure_surface.
Never re-emit digests/secrets. No invent allow/KR/MRTD.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agent_challenge.review.openrouter import (
    OpenRouterTransportError,
    infrastructure_failure_reason,
    short_quote_measurement_diag_class,
)

# Exact / known product ValueError strings from review_runtime._measurement_from_quote
# plus residual pattern messages that openrouter already classifies as mismatch.
_MESSAGE_TO_DIAG: tuple[tuple[str, str], ...] = (
    ("quoted compose hash mismatches assignment", "compose"),
    ("quoted key provider mismatches assignment", "key_provider"),
    ("quoted os image hash mismatches assignment", "os"),
    ("quoted mrtd mismatches assignment", "mrtd"),
    ("quoted rtmr0 mismatches assignment", "rtmr0"),
    ("quoted rtmr1 mismatches assignment", "rtmr1"),
    ("quoted rtmr2 mismatches assignment", "rtmr2"),
    ("quoted rtmr3 mismatches assignment", "rtmr3"),
    # Soft residual wording still attributeable by closed keywords:
    ("measurement register rtmr0 mismatches assignment", "rtmr0"),
    ("quote measurement mismatch without field token", "other"),
    ("", "other"),
)


def _load_review_runtime():
    path = Path(__file__).resolve().parents[1] / "docker" / "review" / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_quote_diag_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("message", "expected_diag"), _MESSAGE_TO_DIAG)
def test_message_words_map_to_closed_quote_diag(message: str, expected_diag: str) -> None:
    """TDD: every known mismatch message collapses to one allowlisted token."""

    token = short_quote_measurement_diag_class(ValueError(message))
    assert token == expected_diag
    # Closed set only; never free-form or secret fragments.
    assert token in {
        "compose",
        "key_provider",
        "os",
        "mrtd",
        "rtmr0",
        "rtmr1",
        "rtmr2",
        "rtmr3",
        "event_log",
        "other",
    }
    assert "sha256" not in token
    assert " " not in token


def test_diag_never_reemits_raw_digests_or_secrets() -> None:
    """Token is closed; digest-looking residue in message never becomes diag text."""

    toxic = (
        "quoted compose hash mismatches assignment "
        "got=deadbeefcafebabe00112233445566778899aabbccddeeff0011223344556677 "
        "want=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff "
        "api_key=sk-or-v1-SENSITIVE"
    )
    token = short_quote_measurement_diag_class(ValueError(toxic))
    assert token == "compose"
    assert "deadbeef" not in token
    assert "sk-or" not in token
    assert "SENSITIVE" not in token


def test_openrouter_transport_error_accepts_quote_diag_token() -> None:
    """OpenRouterTransportError must retain quote field-class diag tokens."""

    err = OpenRouterTransportError(
        "quote_measurement_mismatch",
        "quoted compose hash mismatches assignment",
        diag="compose",
    )
    assert err.reason_code == "quote_measurement_mismatch"
    assert err.diag == "compose"
    # Unknown free-form collapses to other (never accepted as surface free text).
    bad = OpenRouterTransportError(
        "quote_measurement_mismatch",
        "quoted compose hash mismatches assignment",
        diag="compose_hash_0xdeadbeef",
    )
    assert bad.diag == "other"


def test_infrastructure_reason_still_quote_measurement_mismatch() -> None:
    """Field diag is additive; reason_code stays quote_measurement_mismatch."""

    for message, _diag in _MESSAGE_TO_DIAG[:7]:
        assert infrastructure_failure_reason(ValueError(message)) == ("quote_measurement_mismatch")
        wrapped = OpenRouterTransportError(
            "quote_measurement_mismatch",
            message,
            diag=short_quote_measurement_diag_class(ValueError(message)),
        )
        assert infrastructure_failure_reason(wrapped) == "quote_measurement_mismatch"


@pytest.mark.parametrize(
    ("message", "expected_diag"),
    [
        ("quoted compose hash mismatches assignment", "compose"),
        ("quoted key provider mismatches assignment", "key_provider"),
        ("quoted os image hash mismatches assignment", "os"),
        ("quoted mrtd mismatches assignment", "mrtd"),
        ("quoted rtmr0 mismatches assignment", "rtmr0"),
        ("quoted rtmr1 mismatches assignment", "rtmr1"),
        ("quoted rtmr2 mismatches assignment", "rtmr2"),
        ("quoted rtmr3 mismatches assignment", "rtmr3"),
    ],
)
def test_bounded_surface_exposes_quote_diag_from_value_error(
    message: str, expected_diag: str
) -> None:
    """public_logs residual must include the field-class diag for mismatch."""

    runtime = _load_review_runtime()
    surface = runtime.bounded_review_failure_surface(ValueError(message))
    assert surface["error"] == "review_failed"
    assert surface["reason"] == "ValueError"
    assert surface["reason_code"] == "quote_measurement_mismatch"
    assert surface.get("diag") == expected_diag
    # Never leak the full mismatch wording or registry digests into the surface.
    assert "mismatches assignment" not in str(surface)
    assert "quoted" not in str(surface)


def test_bounded_surface_exposes_diag_from_transport_error() -> None:
    runtime = _load_review_runtime()
    surface = runtime.bounded_review_failure_surface(
        OpenRouterTransportError(
            "quote_measurement_mismatch",
            "quoted os image hash mismatches assignment",
            diag="os",
        )
    )
    assert surface["reason_code"] == "quote_measurement_mismatch"
    assert surface.get("diag") == "os"
    assert "os image" not in str(surface)


def test_bounded_surface_unknown_measurement_mismatch_uses_other() -> None:
    runtime = _load_review_runtime()
    surface = runtime.bounded_review_failure_surface(
        ValueError("measurement mismatch without a named field")
    )
    assert surface["reason_code"] == "quote_measurement_mismatch"
    assert surface.get("diag") == "other"
