"""Backend fail-closed wiring for the in-CVM golden key-release (VAL-ORCH-035).

When a validator key-release endpoint is configured (Phala path) but the golden
key cannot be obtained — the endpoint denies, is unreachable, or drops
mid-exchange — the orchestrator must:
  * NOT run the verifier against a missing/placeholder golden (run_own_runner_job
    is never invoked),
  * NOT emit a passing score / attestation envelope,
  * surface exactly one parseable fail-closed ``BASE_BENCHMARK_RESULT=`` line
    (score 0, reason ``phala_key_release_failed``), and return nonzero.

When no endpoint is configured (legacy path) the eval runs unchanged and makes no
key-release call.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from agent_challenge.canonical import attested_result as ar
from agent_challenge.evaluation import own_runner_backend as backend
from agent_challenge.evaluation.own_runner.orchestrator import JobResult
from agent_challenge.evaluation.own_runner.result_schema import (
    RESULT_LINE_PREFIX,
    build_benchmark_result,
)
from agent_challenge.keyrelease.client import (
    KEY_RELEASE_URL_ENV,
    KeyReleaseDenied,
    KeyReleaseError,
    KeyReleaseMidExchangeError,
    KeyReleaseUnreachable,
)


def _canned_result() -> JobResult:
    return JobResult(
        status="completed",
        score=1.0,
        resolved=1,
        total=1,
        reason_code=None,
        pass_at_k={},
        n_total_trials=1,
        n_completed_trials=1,
        n_errored_trials=0,
        trial_outcomes=[],
        benchmark_result=build_benchmark_result(
            status="completed", score=1.0, resolved=1, total=1, reason_code=None
        ),
    )


class _RunRecorder:
    """Stand-in for run_own_runner_job that records whether it was invoked."""

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, **kwargs: Any) -> JobResult:
        self.called = True
        return _canned_result()


def _fake_client_factory(*, exc: Exception | None = None, key: bytes = b"golden-key"):
    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def acquire_golden_key(self, **kwargs: Any) -> bytes:
            if exc is not None:
                raise exc
            return key

    return _FakeClient


def _result_lines(out: str) -> list[dict]:
    return [
        json.loads(ln[len(RESULT_LINE_PREFIX) :])
        for ln in out.splitlines()
        if ln.startswith(RESULT_LINE_PREFIX)
    ]


def _run_main(monkeypatch, tmp_path) -> tuple[int, str, _RunRecorder]:
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)
    rc = backend.main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])
    return rc, recorder


def _enable_phala_key_release(monkeypatch) -> None:
    monkeypatch.setenv(backend.PHALA_ATTESTATION_ENABLED_ENV, "1")
    plan = {
        "eval_run_id": "eval-run-001",
        "key_release_endpoint": "validator.test:8701",
        "key_release_nonce": "key-nonce-001",
        "score_nonce": "score-nonce-001",
        "issued_at_ms": 0,
        "expires_at_ms": 4_102_444_800_000,
        "selected_tasks": [
            {
                "task_id": "hello-world",
                "image_ref": "registry/task@sha256:" + "a" * 64,
            }
        ],
        "k": 1,
        "agent_hash": "f" * 64,
        "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    }
    monkeypatch.setattr(
        backend,
        "_resolve_phala_binding_from_env",
        lambda: {"eval_plan": plan, "rtmr3": "d" * 96},
    )
    monkeypatch.setattr(
        backend,
        "assert_agent_artifact_matches_plan",
        lambda **_: "f" * 64,
    )
    monkeypatch.setattr(
        backend,
        "assert_package_tree_matches_plan",
        lambda **_: "b" * 64,
    )
    monkeypatch.setattr(backend, "_preflight_eval_plan_tasks", lambda **_: {})


@pytest.mark.parametrize(
    "exc",
    [
        KeyReleaseDenied("measurement not allowlisted"),
        KeyReleaseUnreachable("connection refused"),
        KeyReleaseMidExchangeError("dropped after nonce"),
        KeyReleaseError("generic key-release failure"),
    ],
)
def test_key_unavailable_fails_closed_without_scoring(exc, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    _enable_phala_key_release(monkeypatch)
    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _fake_client_factory(exc=exc))
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])

    out = capsys.readouterr().out
    lines = _result_lines(out)
    # Exactly one parseable fail-closed result line.
    assert len(lines) == 1
    payload = lines[0]
    assert payload["status"] == "failed"
    assert payload["score"] == 0.0
    assert payload["reason_code"] == "phala_key_release_failed"
    assert rc != 0
    # The verifier/scoring path (run_own_runner_job) NEVER ran against golden.
    assert recorder.called is False
    # No attestation envelope / passing artifact leaked out.
    assert ar.EXECUTION_PROOF_RESULT_KEY not in payload


def test_flag_off_never_constructs_key_release_client(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    monkeypatch.delenv(backend.PHALA_ATTESTATION_ENABLED_ENV, raising=False)

    def _forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("flag-off must not access dstack key-release")

    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _forbidden)
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])

    out = capsys.readouterr().out
    lines = _result_lines(out)
    assert rc == 0
    assert recorder.called is True
    assert len(lines) == 1
    assert lines[0]["status"] == "completed"


def test_no_key_release_endpoint_uses_legacy_path(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv(KEY_RELEASE_URL_ENV, raising=False)
    monkeypatch.delenv(backend.PHALA_ATTESTATION_ENABLED_ENV, raising=False)

    # A client here would be a bug: legacy path must make no key-release call.
    def _forbidden(*args: Any, **kwargs: Any):  # pragma: no cover - only fails on misuse
        raise AssertionError("key-release client constructed on the legacy path")

    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _forbidden)
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])

    assert rc == 0
    assert recorder.called is True
    assert _result_lines(capsys.readouterr().out)[0]["status"] == "completed"


def test_acquire_helper_returns_none_without_endpoint(monkeypatch):
    monkeypatch.delenv(KEY_RELEASE_URL_ENV, raising=False)
    # Guard residual suite pollution: raw host/port must not silently arm KR path.
    monkeypatch.delenv("KEY_RELEASE_RA_TLS_HOST", raising=False)
    monkeypatch.delenv("KEY_RELEASE_RA_TLS_PORT", raising=False)
    monkeypatch.delenv("CHALLENGE_PHALA_EVAL_PLAN", raising=False)
    assert backend._acquire_golden_key_if_required() is None


def test_acquire_helper_returns_key_on_success(monkeypatch):
    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _fake_client_factory(key=b"abc"))
    assert backend._acquire_golden_key_if_required() == b"abc"


def _write_temp_leaf_pem(tmp_path) -> tuple[Any, str]:
    """Create a throw-away PEM leaf cert; return (path, expected SPKI sha256 hex)."""

    import hashlib
    from datetime import UTC, datetime, timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ra-tls-test")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    spki = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    path = tmp_path / "leaf.pem"
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return path, hashlib.sha256(spki).hexdigest()


def test_v2_acquire_uses_cert_spki_when_pubkey_and_spki_env_unset(monkeypatch, tmp_path):
    """Live residual: empty PUBKEY/SPKI env + real cert must bind cert SPKI.

    Pre-fix force-sha256(empty) failed in GoldenKeyReleaseClient with
    ``ra_tls_spki_digest does not match`` before any framed send.
    """

    import hashlib

    cert_path, expected_spki = _write_temp_leaf_pem(tmp_path)
    monkeypatch.delenv(backend.PHALA_RA_TLS_PUBKEY_ENV, raising=False)
    monkeypatch.delenv(backend.PHALA_RA_TLS_SPKI_SHA256_ENV, raising=False)
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_CERT_FILE", str(cert_path))
    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")

    captured: dict[str, Any] = {}

    class _CapturingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def acquire_golden_key(self, **kwargs: Any) -> bytes:
            captured.update(kwargs)
            # Discriminator vs pre-fix empty digest.
            empty = hashlib.sha256(b"").hexdigest()
            assert kwargs.get("ra_tls_spki_digest") != empty
            assert kwargs.get("ra_tls_spki_digest") == expected_spki
            return b"granted-key"

    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _CapturingClient)
    monkeypatch.setattr(
        "agent_challenge.canonical.attested_result.DstackQuoteProvider",
        lambda *a, **k: object(),
    )

    plan = {
        "eval_run_id": "eval-run-live-spki",
        "key_release_nonce": "key-nonce-live",
    }
    key = backend._acquire_golden_key_if_required(eval_plan=plan)
    assert key == b"granted-key"
    assert captured["ra_tls_spki_digest"] == expected_spki
    # Optionally materializes observability env after resolution.
    assert os.environ.get(backend.PHALA_RA_TLS_SPKI_SHA256_ENV) == expected_spki


def test_resolve_ra_tls_spki_digest_prefers_explicit_env(monkeypatch, tmp_path):
    cert_path, cert_spki = _write_temp_leaf_pem(tmp_path)
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_CERT_FILE", str(cert_path))
    monkeypatch.setenv(backend.PHALA_RA_TLS_SPKI_SHA256_ENV, "ab" * 32)
    assert backend._resolve_ra_tls_spki_digest(ra_tls_pubkey=b"") == "ab" * 32
    assert cert_spki  # ensure helper ran
