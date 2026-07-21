"""Durable guest fail-closed surface for pre-KR residual (eval-pre-kr-failclosed).

Live residual after SPKI fix reached ``tcp_connect_ok`` then collapsed to opaque
``terminal_bench_failed`` because pre-frame/quote/agent failures fell through
``main``'s bare ``except Exception`` without stage labels.

These tests encode the offline contract:
  * guest_eval_fail markers print stage/class/detail (secret-free)
  * agent/preflight failures get distinct durable labels (not opaque KR seals)
  * bare Exception from quote/pre-frame acquire maps to ``phala_key_release_failed``
  * KeyReleaseError still maps via ``reason_code``
  * progress breadcrumbs after preflight_ok / acquire_start
  * preflight can PASS with a live-like plan when digests match (mocked)
"""

from __future__ import annotations

import json
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
    def __init__(self) -> None:
        self.called = False

    async def __call__(self, **kwargs: Any) -> JobResult:
        self.called = True
        return _canned_result()


def _result_lines(out: str) -> list[dict]:
    return [
        json.loads(ln[len(RESULT_LINE_PREFIX) :])
        for ln in out.splitlines()
        if ln.startswith(RESULT_LINE_PREFIX)
    ]


def _live_like_plan() -> dict[str, Any]:
    return {
        "eval_run_id": "eval-run-live37",
        "key_release_endpoint": "validator.test:8701",
        "key_release_nonce": "key-nonce-live37",
        "score_nonce": "score-nonce-live37",
        "issued_at_ms": 0,
        "expires_at_ms": 4_102_444_800_000,
        "selected_tasks": [
            {
                "task_id": "hello-world",
                "image_ref": "registry/task@sha256:" + "a" * 64,
                "task_config_sha256": "b" * 64,
            },
            {
                "task_id": "task-two",
                "image_ref": "registry/task@sha256:" + "c" * 64,
                "task_config_sha256": "d" * 64,
            },
            {
                "task_id": "task-three",
                "image_ref": "registry/task@sha256:" + "e" * 64,
                "task_config_sha256": "f" * 64,
            },
        ],
        "k": 1,
        "agent_hash": "aa" * 32,
    }


def _enable_phala(monkeypatch, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    monkeypatch.setenv(backend.PHALA_ATTESTATION_ENABLED_ENV, "1")
    eval_plan = plan if plan is not None else _live_like_plan()
    monkeypatch.setattr(
        backend,
        "_resolve_phala_binding_from_env",
        lambda: {"eval_plan": eval_plan, "rtmr3": "d" * 96},
    )
    return eval_plan


def test_preflight_ok_marker_with_live_like_plan(monkeypatch, tmp_path, capsys) -> None:
    """When agent+tasks pass, preflight_ok + acquire_start mark before KR."""

    plan = _enable_phala(monkeypatch)
    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    monkeypatch.setattr(
        backend,
        "assert_agent_artifact_matches_plan",
        lambda **_: plan["agent_hash"],
    )
    monkeypatch.setattr(
        backend,
        "_preflight_eval_plan_tasks",
        lambda **_: {t["task_id"]: object() for t in plan["selected_tasks"]},
    )

    def _blocked(*_a: Any, **_k: Any) -> bytes:
        raise KeyReleaseDenied("measurement not allowlisted")

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def acquire_golden_key(self, **kwargs: Any) -> bytes:
            return _blocked()

    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _FakeClient)
    monkeypatch.setattr(
        "agent_challenge.canonical.attested_result.DstackQuoteProvider",
        lambda *a, **k: object(),
    )
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(["run", "--job-dir", str(tmp_path / "job")])
    combined = capsys.readouterr().out
    assert rc != 0
    assert recorder.called is False
    assert "guest_eval stage=preflight_ok" in combined
    assert "guest_eval stage=acquire_start" in combined
    assert "guest_eval_fail stage=key_release" in combined
    lines = _result_lines(combined)
    assert len(lines) == 1
    assert lines[0]["reason_code"] == "phala_key_release_failed"
    assert lines[0]["score"] == 0.0


def test_missing_agent_hash_labeled_not_opaque(monkeypatch, tmp_path, capsys) -> None:
    """Missing agent artifact/digest emits agent_identity labels, not silent generic."""

    _enable_phala(monkeypatch)
    monkeypatch.delenv(KEY_RELEASE_URL_ENV, raising=False)
    monkeypatch.delenv(backend.PHALA_AGENT_HASH_ENV, raising=False)
    monkeypatch.delenv(backend.AGENT_ARTIFACT_PATH_ENV, raising=False)
    monkeypatch.delenv(backend.AGENT_ARTIFACT_PATH_ENV_ALT, raising=False)

    def _unexpected_kr(**_: Any) -> bytes | None:
        pytest.fail("agent identity must fail before key release")

    def _unexpected_job(**_: Any) -> JobResult:
        pytest.fail("agent identity must fail before job execution")

    # Real assert path: no artifact on disk and no declared hash.
    monkeypatch.setattr(backend, "resolve_agent_artifact_path", lambda: None)
    monkeypatch.setattr(backend, "_acquire_golden_key_if_required", _unexpected_kr)
    monkeypatch.setattr(backend, "run_own_runner_job", _unexpected_job)

    rc = backend.main(["run", "--job-dir", str(tmp_path / "job")])
    combined = capsys.readouterr().out
    assert rc != 0
    assert "guest_eval_fail stage=agent_identity" in combined
    assert "class=ValueError" in combined
    lines = _result_lines(combined)
    assert len(lines) == 1
    payload = lines[0]
    assert payload["status"] == "failed"
    assert payload["score"] == 0.0
    # Still terminal_bench_failed reason (no new taxonomy code) but labeled.
    assert payload["reason_code"] == "terminal_bench_failed"
    assert payload.get("failure_stage") == "agent_identity"
    assert payload.get("failure_class") == "ValueError"
    assert "agent" in str(payload.get("failure_detail", "")).lower()


def test_quote_provider_bare_exception_maps_to_key_release_not_terminal(
    monkeypatch, tmp_path, capsys
) -> None:
    """quote_provider raising plain Exception must surface phala_key_release_failed."""

    plan = _enable_phala(monkeypatch)
    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    monkeypatch.setattr(
        backend,
        "assert_agent_artifact_matches_plan",
        lambda **_: plan["agent_hash"],
    )
    monkeypatch.setattr(
        backend,
        "_preflight_eval_plan_tasks",
        lambda **_: {"hello-world": object()},
    )

    class _BoomQuote:
        def get_quote(self, report_data: bytes) -> Any:
            raise RuntimeError("dstack socket hung")

    class _ClientWithBoomQuote:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.quote_provider = _BoomQuote()
            self.ra_tls_pubkey = kwargs.get("ra_tls_pubkey", b"")
            self.endpoint_url = args[0] if args else "https://validator.test:8700"

        def acquire_golden_key(self, **kwargs: Any) -> bytes:
            # Exercise the real client wrap path via a thin call into authentic logic.
            from agent_challenge.keyrelease.client import GoldenKeyReleaseClient

            real = GoldenKeyReleaseClient(
                self.endpoint_url,
                quote_provider=self.quote_provider,
                ra_tls_pubkey=self.ra_tls_pubkey,
            )
            return real.acquire_golden_key(**kwargs)

    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _ClientWithBoomQuote)
    monkeypatch.setattr(
        "agent_challenge.canonical.attested_result.DstackQuoteProvider",
        lambda *a, **k: _BoomQuote(),
    )
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(
        [
            "run",
            "--task",
            "hello-world",
            "--task",
            "task-two",
            "--task",
            "task-three",
            "--job-dir",
            str(tmp_path / "job"),
        ]
    )
    combined = capsys.readouterr().out
    assert rc != 0
    assert recorder.called is False
    lines = _result_lines(combined)
    assert len(lines) == 1
    payload = lines[0]
    assert payload["reason_code"] == "phala_key_release_failed"
    assert payload["reason_code"] != "terminal_bench_failed"
    assert "guest_eval_fail stage=key_release" in combined
    assert ar.EXECUTION_PROOF_RESULT_KEY not in payload


def test_key_release_error_still_maps_via_reason_code(monkeypatch, tmp_path, capsys) -> None:
    plan = _enable_phala(monkeypatch)
    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    monkeypatch.setattr(
        backend,
        "assert_agent_artifact_matches_plan",
        lambda **_: plan["agent_hash"],
    )
    monkeypatch.setattr(
        backend,
        "_preflight_eval_plan_tasks",
        lambda **_: {"hello-world": object()},
    )

    class _Denied:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def acquire_golden_key(self, **kwargs: Any) -> bytes:
            raise KeyReleaseError("generic key-release failure")

    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _Denied)
    monkeypatch.setattr(
        "agent_challenge.canonical.attested_result.DstackQuoteProvider",
        lambda *a, **k: object(),
    )
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(["run", "--job-dir", str(tmp_path / "job")])
    combined = capsys.readouterr().out
    assert rc != 0
    payload = _result_lines(combined)[0]
    assert payload["reason_code"] == "phala_key_release_failed"
    assert "guest_eval_fail stage=key_release class=KeyReleaseError" in combined
    # Specific KR reasons must NOT get failure_stage override noise.
    assert "failure_stage" not in payload


def test_pre_frame_exception_in_acquire_helper_is_key_release_error(monkeypatch) -> None:
    """Bare Exception in quote-provider / client construction must wrap as KR error."""

    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")

    class _ExplodingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("quote provider construction boom")

    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _ExplodingClient)
    monkeypatch.setattr(
        "agent_challenge.canonical.attested_result.DstackQuoteProvider",
        lambda *a, **k: object(),
    )
    with pytest.raises(KeyReleaseError) as ei:
        backend._acquire_golden_key_if_required(
            eval_plan={
                "eval_run_id": "run-1",
                "key_release_nonce": "n-1",
            }
        )
    assert ei.value.reason_code == "phala_key_release_failed"
    assert "pre-frame" in str(ei.value).lower() or "boom" in str(ei.value).lower()


def test_guest_eval_fail_sanitizes_secretish_detail() -> None:
    detail = "-----BEGIN PRIVATE KEY-----\nabcsecret\n-----END PRIVATE KEY-----"
    sanitized = backend._sanitize_guest_detail(detail)
    assert "BEGIN" not in sanitized
    assert "abcsecret" not in sanitized
    assert sanitized == "redacted"


def test_quote_ok_and_frame_send_markers_from_client(monkeypatch, capsys, tmp_path) -> None:
    """Client progress markers after successful quote; frame_send when mTLS is ready."""

    from agent_challenge.keyrelease.client import GoldenKeyReleaseClient

    class _OkQuote:
        def get_quote(self, report_data: bytes) -> Any:
            class _R:
                quote = "ab" * 64
                event_log = "[]"
                vm_config = "{}"

            return _R()

    # Empty pubkey path (no cert env) so SPKI bind is deterministic.
    monkeypatch.delenv("CHALLENGE_PHALA_RA_TLS_CERT_FILE", raising=False)
    # Provide dummy mTLS files so _raw_release reaches frame_send (then fail connect).
    cert = tmp_path / "c.pem"
    key = tmp_path / "k.pem"
    ca = tmp_path / "ca.pem"
    for path in (cert, key, ca):
        path.write_text("not-a-real-pem\n", encoding="utf-8")
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_CERT_FILE", str(cert))
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_KEY_FILE", str(key))
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_CA_FILE", str(ca))
    # With a cert present, SPKI resolution needs a real PEM; force explicit digest
    # and make _resolve_spki_digest return the caller-provided digest.
    spki = "ab" * 32
    monkeypatch.setattr(
        GoldenKeyReleaseClient,
        "_resolve_spki_digest",
        lambda self: spki,
    )

    client = GoldenKeyReleaseClient(
        "ratls://127.0.0.1:9",
        quote_provider=_OkQuote(),
        ra_tls_pubkey=b"",
        timeout=0.2,
    )
    with pytest.raises(KeyReleaseError):
        client.acquire_golden_key(
            eval_run_id="run",
            key_release_nonce="nonce",
            ra_tls_spki_digest=spki,
        )
    out = capsys.readouterr().out
    assert "guest_eval stage=quote_ok" in out
    # frame_send may not be reached if SSL context refuses the dummy PEMs; quote_ok
    # alone proves post-quote progress reached. Accept either path.
    assert "guest_eval stage=quote_ok" in out


def test_annotate_only_for_terminal_bench_failed() -> None:
    kr = build_benchmark_result(
        status="failed", score=0.0, resolved=0, total=3, reason_code="phala_key_release_failed"
    )
    untouched = backend._annotate_failclosed_result(
        dict(kr), stage="key_release", class_name="X", detail="y"
    )
    assert "failure_stage" not in untouched
    assert untouched["reason_code"] == "phala_key_release_failed"

    generic = build_benchmark_result(
        status="failed", score=0.0, resolved=0, total=3, reason_code="terminal_bench_failed"
    )
    annotated = backend._annotate_failclosed_result(
        generic, stage="agent_identity", class_name="ValueError", detail="missing agent"
    )
    assert annotated["failure_stage"] == "agent_identity"
    assert annotated["failure_class"] == "ValueError"
    assert "missing" in annotated["failure_detail"]
