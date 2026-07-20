"""report residual: capture /report HTTP status + prefer detail.code over status-class.

ROOTCASE residual after timeline tip 9c5bd647 (sub26/27):
  guest durable ``{"diag":"http_status","reason_code":"report_envelope_invalid"}``
  while history stuck reviewing_verifying — no HTTP status number and body codes
  collapsed solely to ``http_status`` when non-2xx family matched first.

Product Mode B SPEED:
1. Surface closed ``http_status`` (3-digit) for every /report non-2xx residual so
   residual diagnosis is not opaque http_status alone.
2. Prefer parseable ``detail.code`` (and receipt-shaped reason_code/status) over
   pure status-class collapse when body tokens exist.
3. Service maps ``ReviewReportError`` message classes into finer detail.code so
   timeline/evidence/measurement subclasses survive the guest mapper (no invent
   allow / KR / gateway).

Evidence: eval-1task-timeline-pass + eval-1task SUMMARYs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agent_challenge.review.openrouter import (
    infrastructure_failure_reason,
    short_report_envelope_diag_class,
)
from agent_challenge.review.report import ReviewReportError

REPO_ROOT = Path(__file__).resolve().parents[1]

_REPORT_ENVELOPE_DIAG_ALLOWLIST = frozenset(
    {
        "timeline",
        "evidence",
        "measurement",
        "schema",
        "http_status",
        "other",
    }
)


def _load_review_runtime():
    path = REPO_ROOT / "docker" / "review" / "review_runtime.py"
    spec = importlib.util.spec_from_file_location(
        "review_runtime_http_status_envelope_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1) Guest: surface always captures numeric HTTP status on /report residual
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body"),
    (
        (401, b'{"detail":{"code":"review_capability_invalid"}}'),
        (403, b'{"detail":{"code":"review_capability_invalid"}}'),
        (409, b'{"detail":{"code":"review_report_conflict"}}'),
        (410, b'{"detail":{"code":"review_capability_invalid"}}'),
        (413, b'{"detail":{"code":"review_report_too_large"}}'),
        (422, b'{"detail":{"code":"review_report_invalid"}}'),
        (429, b'{"detail":{"code":"review_rate_limited"}}'),
        (500, b"<html>bad gateway</html>"),
        (502, b""),
        (503, b'{"status":"verifier_unavailable","reason_code":"review_verifier_unavailable"}'),
    ),
)
def test_report_post_error_surface_captures_http_status(status: int, body: bytes) -> None:
    """public_logs must expose closed http_status so residual is diagnosable."""

    runtime = _load_review_runtime()
    exc = runtime._report_post_error(status, body)
    surface = runtime.bounded_review_failure_surface(exc)
    assert surface["error"] == "review_failed"
    assert surface["reason"] in {"ValueError", "ReportEnvelopeError"}
    assert "http_status" in surface, "surface must capture numeric HTTP status"
    assert surface["http_status"] == str(status)
    # never free-form bodies / secrets
    dumped = str(surface).lower()
    assert "html" not in dumped
    assert "secret" not in dumped
    assert "sk-or" not in dumped


def test_opaque_body_still_http_status_diag_but_status_number_present() -> None:
    runtime = _load_review_runtime()
    exc = runtime._report_post_error(502, b"<html>bad gateway</html>")
    assert short_report_envelope_diag_class(exc) == "http_status"
    assert infrastructure_failure_reason(exc) == "report_envelope_invalid"
    surface = runtime.bounded_review_failure_surface(exc)
    assert surface.get("diag") == "http_status"
    assert surface.get("http_status") == "502"


# ---------------------------------------------------------------------------
# 2) Prefer detail.code (and receipt reason_code) over pure status collapse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "status", "expected_diag", "expected_reason"),
    (
        # body codes must survive even when status is in "status-class" set
        (
            b'{"detail":{"code":"attestation_stale_over_24h"}}',
            409,
            "timeline",
            "report_timeline_invalid",
        ),
        (
            b'{"detail":{"code":"report_timeline_invalid"}}',
            409,
            "timeline",
            "report_timeline_invalid",
        ),
        (
            b'{"detail":{"code":"review_evidence_invalid"}}',
            413,
            "evidence",
            "report_evidence_invalid",
        ),
        (
            b'{"detail":{"code":"review_measurement_unallowlisted"}}',
            409,
            "measurement",
            "report_envelope_invalid",
        ),
        (
            b'{"detail":{"code":"review_report_invalid"}}',
            409,
            "schema",
            "report_envelope_invalid",
        ),
        (
            b'{"detail":{"code":"review_or_planned_digest_missing"}}',
            500,
            "schema",
            "report_envelope_invalid",
        ),
        # capability / rate / size status-family codes stay http_status class
        # but still carry http_status number
        (
            b'{"detail":{"code":"review_capability_invalid"}}',
            410,
            "http_status",
            "report_envelope_invalid",
        ),
        (
            b'{"detail":{"code":"review_rate_limited"}}',
            429,
            "http_status",
            "report_envelope_invalid",
        ),
        # receipt-shape 503 from product (not detail envelope)
        (
            b'{"status":"verifier_unavailable","reason_code":"review_verifier_unavailable"}',
            503,
            "http_status",
            "report_envelope_invalid",
        ),
    ),
)
def test_body_code_preferred_over_status_class_collapse(
    body: bytes, status: int, expected_diag: str, expected_reason: str
) -> None:
    runtime = _load_review_runtime()
    exc = runtime._report_post_error(status, body)
    assert short_report_envelope_diag_class(exc) == expected_diag
    assert expected_diag in _REPORT_ENVELOPE_DIAG_ALLOWLIST
    assert infrastructure_failure_reason(exc) == expected_reason
    surface = runtime.bounded_review_failure_surface(exc)
    assert surface.get("diag") == expected_diag
    assert surface.get("http_status") == str(status)
    # body free text never re-emitted
    assert "detail" not in str(surface)
    assert "attestation_stale" not in str(surface)
    assert "planned_digest" not in str(surface)


def test_empty_body_status_only_still_diag_http_status_with_number() -> None:
    runtime = _load_review_runtime()
    for status in (401, 410, 500, 503):
        exc = runtime._report_post_error(status, b"")
        surface = runtime.bounded_review_failure_surface(exc)
        assert surface["diag"] == "http_status"
        assert surface["http_status"] == str(status)
        assert surface["reason_code"] == "report_envelope_invalid"


# ---------------------------------------------------------------------------
# 3) Service: ReviewReportError → finer detail.code for guest mapper
# ---------------------------------------------------------------------------


def test_review_report_error_to_detail_code_maps_timeline_evidence_measurement() -> None:
    from agent_challenge.api.routes import _review_report_error_detail_code

    cases = (
        ("review timeline does not match assignment", "report_timeline_invalid"),
        ("report timeline is future or post-receipt", "report_timeline_invalid"),
        ("report finished time is invalid", "report_timeline_invalid"),
        ("review times are invalid", "report_timeline_invalid"),
        ("receipt precedes assignment issue", "report_timeline_invalid"),
        ("review evidence is required for a new report receipt", "review_evidence_invalid"),
        ("review evidence is malformed", "review_evidence_invalid"),
        ("review evidence does not bind report and marker", "review_evidence_invalid"),
        (
            "quote measurement does not match assignment",
            "review_measurement_mismatch",
        ),
        ("review measurement is incomplete", "review_measurement_mismatch"),
        ("review measurement is invalid", "review_measurement_mismatch"),
        ("review core is invalid", "review_report_invalid"),
        ("review core identity does not match assignment", "review_report_invalid"),
        ("returned model does not match assignment", "review_report_invalid"),
        ("something unknown", "review_report_invalid"),
    )
    for message, expected in cases:
        code = _review_report_error_detail_code(ReviewReportError(message))
        assert code == expected, (message, code, expected)
        # closed snake token only
        assert code.replace("_", "").isalnum()
        assert code == code.lower()


def test_review_report_error_detail_code_never_leaks_free_form() -> None:
    from agent_challenge.api.routes import _review_report_error_detail_code

    toxic = ReviewReportError(
        "quote measurement does not match assignment got="
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff "
        "api_key=sk-or-v1-SENSITIVE"
    )
    code = _review_report_error_detail_code(toxic)
    assert code == "review_measurement_mismatch"
    assert "ffff" not in code
    assert "sk-or" not in code
    assert "sensitive" not in code.lower()


# ---------------------------------------------------------------------------
# 4) Mode B residual: ReviewEvidenceError / missing key → closed code not 500
# ---------------------------------------------------------------------------


def test_review_report_error_missing_evidence_key_maps_crypto_unavailable() -> None:
    """Store-path wrap yields ReviewReportError with closed crypto code."""

    from agent_challenge.api.routes import _review_report_error_detail_code

    code = _review_report_error_detail_code(
        ReviewReportError("review evidence encryption key is unavailable")
    )
    assert code == "review_evidence_crypto_unavailable"
    assert code.replace("_", "").isalnum()
    assert code == code.lower()
    assert "secret" not in code
    assert "sk-or" not in code


def test_review_evidence_error_detail_code_maps_crypto_vs_generic() -> None:
    from agent_challenge.api.routes import _review_evidence_error_detail_code
    from agent_challenge.review.evidence import ReviewEvidenceError

    crypto = ReviewEvidenceError("review evidence encryption key is unavailable")
    assert _review_evidence_error_detail_code(crypto) == "review_evidence_crypto_unavailable"

    not_configured = ReviewEvidenceError("review evidence encryption key is not configured")
    assert (
        _review_evidence_error_detail_code(not_configured) == "review_evidence_crypto_unavailable"
    )

    generic = ReviewEvidenceError("review evidence object kinds are invalid")
    assert _review_evidence_error_detail_code(generic) == "review_evidence_invalid"

    toxic = ReviewEvidenceError(
        "review evidence encryption key is unavailable path=/secrets/evidence.key "
        "token=sk-or-v1-should-not-leak"
    )
    code = _review_evidence_error_detail_code(toxic)
    assert code == "review_evidence_crypto_unavailable"
    assert "sk-or" not in code
    assert "path=" not in code
    assert "/secrets" not in code


def test_guest_mapper_prefers_evidence_crypto_detail_code_over_http_status() -> None:
    """Guest rebuild not required: existing evidence token → diag=evidence.

    review_evidence_crypto_unavailable and review_evidence_invalid both contain
    the evidence token, so diag maps without guest rebuild.
    """

    runtime = _load_review_runtime()
    bodies = (
        b'{"detail":{"code":"review_evidence_crypto_unavailable"}}',
        b'{"detail":{"code":"review_evidence_invalid"}}',
    )
    for body in bodies:
        exc = runtime._report_post_error(422, body)
        assert short_report_envelope_diag_class(exc) == "evidence"
        assert infrastructure_failure_reason(exc) == "report_evidence_invalid"
        surface = runtime.bounded_review_failure_surface(exc)
        assert surface.get("diag") == "evidence"
        assert surface.get("http_status") == "422"
        dumped = str(surface).lower()
        assert "sk-or" not in dumped
        assert "secret" not in dumped
        assert surface.get("diag") != "http_status"


def test_store_path_crypto_unwrap_maps_to_review_report_error() -> None:
    """Mirrors report._store_and_describe_evidence ReviewEvidenceError wrap."""

    import asyncio

    from agent_challenge.api.routes import _review_report_error_detail_code
    from agent_challenge.review.evidence import ReviewEvidenceError
    from agent_challenge.review.report import ReviewReportError

    async def _wrap_once() -> None:
        try:
            raise ReviewEvidenceError("review evidence encryption key is unavailable")
        except ReviewEvidenceError as exc:
            text = str(exc).lower()
            if any(
                token in text
                for token in (
                    "encryption key",
                    "unavailable",
                    "not configured",
                )
            ):
                raise ReviewReportError("review evidence encryption key is unavailable") from exc
            raise ReviewReportError("review evidence is invalid") from exc

    with pytest.raises(ReviewReportError, match="encryption key is unavailable") as caught:
        asyncio.run(_wrap_once())
    assert _review_report_error_detail_code(caught.value) == "review_evidence_crypto_unavailable"


def test_require_review_evidence_encryption_fail_closed_when_attested() -> None:
    from agent_challenge.sdk.config import ChallengeSettings

    settings = ChallengeSettings(
        database_url="sqlite+aiosqlite:///:memory:",
        shared_token="test-shared-token-not-evidence",
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        review_evidence_encryption_key=None,
        review_evidence_encryption_key_file=None,
        eval_result_signer_mnemonic="test mnemonic phrase for offline only",
    )
    with pytest.raises(ValueError, match="review evidence encryption key"):
        settings.require_review_evidence_encryption_for_production()
    try:
        settings.require_review_evidence_encryption_for_production()
    except ValueError as exc:
        msg = str(exc).lower()
        assert "sk-or" not in msg
        assert "test-shared" not in msg

    ok = ChallengeSettings(
        database_url="sqlite+aiosqlite:///:memory:",
        shared_token="test-shared-token-not-evidence",
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        review_evidence_encryption_key="dedicated-evidence-key-material",
        eval_result_signer_mnemonic="test mnemonic phrase for offline only",
    )
    ok.require_review_evidence_encryption_for_production()  # no raise

    legacy = ChallengeSettings(
        database_url="sqlite+aiosqlite:///:memory:",
        shared_token="test-shared-token-not-evidence",
        attested_review_enabled=False,
        phala_attestation_enabled=False,
        review_evidence_encryption_key=None,
    )
    legacy.require_review_evidence_encryption_for_production()  # no raise when flags off
