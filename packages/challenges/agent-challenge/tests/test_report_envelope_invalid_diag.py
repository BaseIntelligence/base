"""report_envelope_invalid subclasses → bounded public_logs diag.

SPEED residual after OS identity tip 85fbaa8b (sub24): guest durable
``{"error":"review_failed","reason":"ValueError","reason_code":"report_envelope_invalid"}``
with no subclass detail after provision+os path GREEN.

Product Mode B:
1. Prefer bounded public_logs ``diag`` for report_envelope_invalid subclasses
   (timeline|evidence|measurement|schema|http_status|other) derived from
   /report 4xx body codes and status — never leak secrets or body free-text.
2. Keep Dockerfile inventory complete for post-OpenRouter
   evidence/or_outcome/report envelope path + fail-closed selfcheck.
3. Unit TDD. No invent allow/KR/gateway.

Evidence residual: eval-1task-litellm-speed-pass (sub24).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from agent_challenge.review import compose as review_compose
from agent_challenge.review.openrouter import (
    OpenRouterTransportError,
    infrastructure_failure_reason,
    short_report_envelope_diag_class,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DOCKERFILE = review_compose.REVIEW_DOCKERFILE

# Feature-required closed diag set for report_envelope_invalid residual surface.
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

# Body code / status → closed diag. Codes only appear as short tokens in surface
# messages; raw HTTP bodies never re-emitted.
_BODY_CODE_TO_DIAG: tuple[tuple[bytes, int, str], ...] = (
    # schema (Pydantic / envelope structural reject)
    (b'{"detail":{"code":"review_report_invalid"}}', 422, "schema"),
    (b'{"detail":{"code":"review_report_json_invalid"}}', 400, "schema"),
    (b'{"detail":{"code":"review_report_data_mismatch"}}', 422, "schema"),
    (b'{"detail":"review_report_invalid"}', 422, "schema"),
    (b'{"code":"review_report_invalid"}', 422, "schema"),
    # timeline (explicit code or attested-time refuse codes bubbled as detail)
    (b'{"detail":{"code":"report_timeline_invalid"}}', 422, "timeline"),
    (b'{"detail":{"code":"review_report_timeline_invalid"}}', 422, "timeline"),
    (b'{"detail":{"code":"attestation_stale_over_24h"}}', 422, "timeline"),
    (b'{"detail":{"code":"attestation_time_order_invalid"}}', 422, "timeline"),
    (b'{"detail":{"code":"attestation_times_invalid"}}', 422, "timeline"),
    # evidence
    (b'{"detail":{"code":"review_evidence_invalid"}}', 422, "evidence"),
    (b'{"detail":{"code":"report_evidence_invalid"}}', 422, "evidence"),
    # measurement / allowlist binds on report admission
    (b'{"detail":{"code":"review_measurement_unallowlisted"}}', 422, "measurement"),
    (b'{"detail":{"code":"review_measurement_mismatch"}}', 422, "measurement"),
    (b'{"detail":{"code":"quote_measurement_mismatch"}}', 422, "measurement"),
    # size / conflict / auth-ish status without finer code → http_status
    (b'{"detail":{"code":"review_report_too_large"}}', 413, "http_status"),
    (b'{"detail":{"code":"review_report_conflict"}}', 409, "http_status"),
    (b'{"detail":{"code":"review_rate_limited"}}', 429, "http_status"),
    # OR-outcome codes that land as 422 detail on /report
    (b'{"detail":{"code":"review_or_planned_digest_missing"}}', 422, "schema"),
    (b'{"detail":{"code":"review_outcome_unbound"}}', 422, "schema"),
    (b'{"detail":{"code":"review_openrouter_unmeasured_host"}}', 422, "schema"),
)


def _load_review_runtime():
    path = REPO_ROOT / "docker" / "review" / "review_runtime.py"
    spec = importlib.util.spec_from_file_location(
        "review_runtime_report_envelope_diag_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("body", "status", "expected_diag"), _BODY_CODE_TO_DIAG)
def test_report_post_error_maps_body_codes_to_closed_diag(
    body: bytes, status: int, expected_diag: str
) -> None:
    """TDD: /report non-2xx body codes collapse to closed diag tokens only."""

    runtime = _load_review_runtime()
    exc = runtime._report_post_error(status, body)
    token = short_report_envelope_diag_class(exc)
    assert token == expected_diag
    assert token in _REPORT_ENVELOPE_DIAG_ALLOWLIST
    # Never re-emit body free text or secret-looking tokens.
    assert "secret" not in str(exc).lower()
    assert b"detail" not in str(exc).encode("utf-8")
    raw_json = body.decode("utf-8", errors="replace")
    # Exception message must not contain the raw JSON body.
    assert raw_json not in str(exc)


def test_report_post_error_opaque_body_uses_http_status_diag() -> None:
    """Unparseable / opaque /report body → diag=http_status (status class only)."""

    runtime = _load_review_runtime()
    for status, body in (
        (500, b"raw provider body with secret=sk-or-v1-abc and api_key=SENSITIVE"),
        (502, b"<html>bad gateway</html>"),
        (401, b"not-json"),
        (422, b""),
    ):
        exc = runtime._report_post_error(status, body)
        assert infrastructure_failure_reason(exc) == "report_envelope_invalid"
        assert short_report_envelope_diag_class(exc) == "http_status"
        text = str(exc).lower()
        assert "secret" not in text
        assert "sk-or" not in text
        assert "sensitive" not in text
        assert "html" not in text
        # Surface must remain secret-free too.
        surface = runtime.bounded_review_failure_surface(exc)
        assert surface["reason_code"] == "report_envelope_invalid"
        assert surface.get("diag") == "http_status"
        assert "secret" not in str(surface).lower()


def test_bounded_surface_attaches_report_envelope_diag_for_each_subclass() -> None:
    """public_logs residual: reason_code + closed diag for every subclass."""

    runtime = _load_review_runtime()
    samples = (
        (b'{"detail":{"code":"review_report_invalid"}}', 422, "schema"),
        (b'{"detail":{"code":"attestation_stale_over_24h"}}', 422, "timeline"),
        (b'{"detail":{"code":"review_evidence_invalid"}}', 422, "evidence"),
        (b'{"detail":{"code":"review_measurement_unallowlisted"}}', 422, "measurement"),
        (b'{"detail":{"code":"review_report_too_large"}}', 413, "http_status"),
    )
    for body, status, expected_diag in samples:
        exc = runtime._report_post_error(status, body)
        surface = runtime.bounded_review_failure_surface(exc)
        assert surface["error"] == "review_failed"
        assert surface["reason"] in {"ValueError", "ReportEnvelopeError"}
        # Parent residual reason stays report_envelope_invalid except when
        # existing separate codes for evidence/timeline still apply.
        reason = surface["reason_code"]
        assert reason in {
            "report_envelope_invalid",
            "report_evidence_invalid",
            "report_timeline_invalid",
        }
        assert surface.get("diag") == expected_diag
        assert surface["diag"] in _REPORT_ENVELOPE_DIAG_ALLOWLIST
        # No free-form body leak.
        assert "detail" not in str(surface)
        assert "attestation_stale" not in str(surface)
        assert "review_measurement_unallowlisted" not in str(surface)


def test_message_classifier_for_local_envelope_valueerrors() -> None:
    """Local guest ValueErrors (envelope build / stage) also map closed diag."""

    cases: tuple[tuple[str, str], ...] = (
        ("report envelope invalid from /report", "schema"),
        ("review envelope digest mismatches review core", "schema"),
        ("unsupported review envelope schema version", "schema"),
        ("review envelope domain is invalid", "schema"),
        ("report evidence invalid from /report", "evidence"),
        ("review evidence field is invalid", "evidence"),
        ("report timeline invalid from /report", "timeline"),
        ("report timeline is future or post-receipt", "timeline"),
        ("report measurement no longer matches sealed assignment", "measurement"),
        ("quoted measurement is not allowlisted for report", "measurement"),
        ("stage envelope failed: TypeError", "other"),
        ("unknown residual wording without keywords", "other"),
        ("", "other"),
    )
    for message, expected in cases:
        token = short_report_envelope_diag_class(ValueError(message))
        assert token == expected, (message, token, expected)
        assert token in _REPORT_ENVELOPE_DIAG_ALLOWLIST


def test_report_envelope_diag_never_accepts_free_form_or_digest() -> None:
    """OpenRouterTransportError / surface must collapse free-form diag to other."""

    runtime = _load_review_runtime()
    err = OpenRouterTransportError(
        "report_envelope_invalid",
        "report envelope invalid from /report",
        diag="review_report_invalid_got_deadbeef",
    )
    assert err.diag == "other"
    surface = runtime.bounded_review_failure_surface(err)
    assert surface["reason_code"] == "report_envelope_invalid"
    assert surface.get("diag") == "other"
    assert "deadbeef" not in str(surface)


def test_toxic_body_never_leaks_into_surface() -> None:
    runtime = _load_review_runtime()
    toxic = (
        b'{"detail":{"code":"review_report_invalid","msg":'
        b'"got=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff '
        b'api_key=sk-or-v1-LEAKED"}}'
    )
    exc = runtime._report_post_error(422, toxic)
    surface = runtime.bounded_review_failure_surface(exc)
    dumped = str(surface).lower()
    assert "ffff" not in dumped
    assert "sk-or" not in dumped
    assert "leak" not in dumped
    assert surface.get("diag") == "schema"


def test_transport_error_accepts_report_envelope_diag_tokens() -> None:
    for token in sorted(_REPORT_ENVELOPE_DIAG_ALLOWLIST):
        err = OpenRouterTransportError(
            "report_envelope_invalid",
            "report envelope invalid from /report",
            diag=token,
        )
        assert err.diag == token
        assert infrastructure_failure_reason(err) == "report_envelope_invalid"


# ---------------------------------------------------------------------------
# Dockerfile inventory (post-OpenRouter envelope build path)
# ---------------------------------------------------------------------------


def _dockerfile_text() -> str:
    assert REVIEW_DOCKERFILE.is_file()
    return REVIEW_DOCKERFILE.read_text(encoding="utf-8")


# Modules used after openrouter success on guest residual path:
# openrouter → policy → report core/envelope (lazy attested_times) +
# or_outcome_bind (OR digests) + keyrelease quote (measure/quote).
_POST_OR_ENVELOPE_MODULES: tuple[str, ...] = (
    "src/agent_challenge/review/openrouter.py",
    "src/agent_challenge/review/or_outcome_bind.py",
    "src/agent_challenge/review/attested_times.py",
    "src/agent_challenge/review/policy.py",
    "src/agent_challenge/review/report.py",
    "src/agent_challenge/review/schemas.py",
    "src/agent_challenge/review/canonical.py",
    "src/agent_challenge/keyrelease/quote.py",
    "src/agent_challenge/keyrelease/__init__.py",
    "docker/review/review_runtime.py",
)


def test_dockerfile_copies_all_post_openrouter_envelope_path_modules() -> None:
    """Fail closed if any post-OR envelope build module is omitted from image."""

    dockerfile = _dockerfile_text()
    for module_path in _POST_OR_ENVELOPE_MODULES:
        assert f"COPY {module_path}" in dockerfile, (
            f"review Dockerfile must COPY {module_path} for guest envelope path"
        )


def test_dockerfile_selfcheck_covers_envelope_bind_path() -> None:
    """Build-time selfcheck imports report_data + OR-bind + envelope helpers."""

    dockerfile = _dockerfile_text()
    assert "agent_challenge.review.attested_times" in dockerfile
    assert "agent_challenge.review.or_outcome_bind" in dockerfile
    assert "review_report_data_hex" in dockerfile or "review_report_data_preimage" in dockerfile
    assert "require_real_or_digests" in dockerfile or "build_openrouter_observation" in dockerfile
    # Fail-closed RUN python must exist.
    assert re.search(r"^\s*RUN\s+python\b", dockerfile, flags=re.MULTILINE | re.IGNORECASE)


def test_dockerfile_selfcheck_imports_build_review_envelope_or_report_path() -> None:
    """Selfcheck should exercise the report envelope construction import surface.

    Prefer explicit build_review_envelope / review_report_data_hex so a future
    COPY omit of a lazy dependency fails the image build before Phala spend.
    """

    dockerfile = _dockerfile_text()
    # Either the envelope builder itself or the report_data binders are enough
    # when paired with attested_times import (already required).
    has_envelope = "build_review_envelope" in dockerfile
    has_report_data = (
        "review_report_data_hex" in dockerfile or "review_report_data_preimage" in dockerfile
    )
    assert has_envelope or has_report_data
    # Prefer hardening: document expected selfcheck imports for residual workers.
    assert "from agent_challenge.review.report import" in dockerfile or (
        "agent_challenge.review.report" in dockerfile
    )


def test_surface_diag_allowlist_includes_report_envelope_tokens() -> None:
    """Runtime surface allowlist must include every report envelope subclass."""

    runtime = _load_review_runtime()
    allow = getattr(runtime, "_SURFACE_DIAG_ALLOWLIST", frozenset())
    missing = sorted(_REPORT_ENVELOPE_DIAG_ALLOWLIST - set(allow))
    assert missing == [], f"surface diag allowlist missing {missing}"
