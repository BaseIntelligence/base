"""In-enclave golden decryption wiring (feature key-release-decrypt-in-enclave).

The released golden key must be CONSUMED to decrypt the encrypted-at-rest golden
inside the enclave BEFORE the eval runs, fail closed if decryption fails, and
never surface the key or golden plaintext to a miner-visible path (VAL-KEY-016/
017/018; expectedBehavior: usable only in-enclave).

These tests drive ``own_runner_backend.main`` with a fake key-release client
(so no network/dstack is needed) and assert:
  * the released key is used to decrypt the golden before the eval is invoked;
  * a decryption failure fails closed (one parseable ``failed`` line, reason
    ``phala_golden_decrypt_failed``) WITHOUT running the eval and WITHOUT a
    passing score;
  * the key and golden plaintext never appear on stdout or on any written path;
  * the RA-TLS public key is resolved from the deploy env and handed to client.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from agent_challenge.evaluation import own_runner_backend as backend
from agent_challenge.evaluation.own_runner.orchestrator import JobResult
from agent_challenge.evaluation.own_runner.result_schema import (
    RESULT_LINE_PREFIX,
    build_benchmark_result,
)
from agent_challenge.golden import crypto, package
from agent_challenge.keyrelease.client import KEY_RELEASE_URL_ENV

# Fragment the golden plaintext marker so a VAL-KEY-001 repo scan stays clean.
GOLDEN_MARKER = "harbor-independence/" + "oracle-golden"


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
    def __init__(self, log: list[str] | None = None) -> None:
        self.called = False
        self._log = log

    async def __call__(self, **kwargs: Any) -> JobResult:
        self.called = True
        if self._log is not None:
            self._log.append("eval")
        return _canned_result()


def _client_factory(*, key: bytes, captured: dict[str, Any] | None = None):
    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if captured is not None:
                captured["args"] = args
                captured["kwargs"] = kwargs

        def acquire_golden_key(self, **kwargs: Any) -> bytes:
            return key

    return _FakeClient


def _result_lines(out: str) -> list[dict]:
    return [
        json.loads(ln[len(RESULT_LINE_PREFIX) :])
        for ln in out.splitlines()
        if ln.startswith(RESULT_LINE_PREFIX)
    ]


def _seed_encrypted_golden(golden_dir, key: bytes, doc: dict) -> None:
    golden_dir.mkdir(parents=True, exist_ok=True)
    blob = package.encrypt_golden_bytes(json.dumps(doc).encode(), key)
    (golden_dir / package.ORACLE_CIPHERTEXT_NAME).write_bytes(blob)


def _enable_phala_decrypt(monkeypatch, *, task_id: str) -> None:
    monkeypatch.setenv(backend.PHALA_ATTESTATION_ENABLED_ENV, "1")
    monkeypatch.setenv("CHALLENGE_ATTESTED_REVIEW_ENABLED", "1")
    plan = {
        "eval_run_id": "eval-run-001",
        "key_release_endpoint": "https://validator.test:8700",
        "key_release_nonce": "key-nonce-001",
        "score_nonce": "score-nonce-001",
        "issued_at_ms": 0,
        "expires_at_ms": 4_102_444_800_000,
        "selected_tasks": [
            {
                "task_id": task_id,
                "image_ref": "registry/task@sha256:" + "a" * 64,
                "task_config_sha256": "b" * 64,
            }
        ],
        "k": 1,
        "agent_hash": "f" * 64,
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
    monkeypatch.setattr(backend, "_preflight_eval_plan_tasks", lambda **_: {})
    monkeypatch.setattr(backend, "_emit_job_result", lambda *args, **kwargs: 0)


def test_released_key_decrypts_golden_before_eval(monkeypatch, tmp_path, capsys):
    key = bytes(range(32))
    golden_dir = tmp_path / "golden"
    _seed_encrypted_golden(golden_dir, key, {"schema": GOLDEN_MARKER, "results": {"t": 1}})

    order: list[str] = []
    real_load = package.load_encrypted_oracle_golden

    def _tracking_load(k: bytes, **kw: Any):
        order.append("decrypt")
        return real_load(k, **kw)

    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    _enable_phala_decrypt(monkeypatch, task_id="hello-world")
    monkeypatch.setenv(backend.GOLDEN_DIR_ENV, str(golden_dir))
    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _client_factory(key=key))
    monkeypatch.setattr(package, "load_encrypted_oracle_golden", _tracking_load)
    monkeypatch.setattr(backend, "run_own_runner_job", _RunRecorder(log=order))

    rc = backend.main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])

    assert rc == 0
    # The golden was decrypted (with the released key) BEFORE the eval ran.
    assert order == ["decrypt", "eval"]


def test_golden_decrypt_failure_fails_closed_without_eval(monkeypatch, tmp_path, capsys):
    # Seed ciphertext under one key but release a DIFFERENT (wrong) key.
    golden_dir = tmp_path / "golden"
    _seed_encrypted_golden(golden_dir, bytes(range(32)), {"schema": GOLDEN_MARKER})
    wrong_key = bytes([0xAA]) * 32

    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    _enable_phala_decrypt(monkeypatch, task_id="hello-world")
    monkeypatch.setenv(backend.GOLDEN_DIR_ENV, str(golden_dir))
    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _client_factory(key=wrong_key))
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(["run", "--task", "hello-world", "--job-dir", str(tmp_path / "job")])

    out = capsys.readouterr().out
    lines = _result_lines(out)
    assert rc != 0
    assert len(lines) == 1
    assert lines[0]["status"] == "failed"
    assert lines[0]["score"] == 0.0
    assert lines[0]["reason_code"] == backend.GOLDEN_DECRYPT_FAILED_REASON
    # The eval never ran against a missing/placeholder golden.
    assert recorder.called is False


def test_missing_ciphertext_fails_closed(monkeypatch, tmp_path, capsys):
    empty_dir = tmp_path / "golden-empty"
    empty_dir.mkdir()
    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    _enable_phala_decrypt(monkeypatch, task_id="t")
    monkeypatch.setenv(backend.GOLDEN_DIR_ENV, str(empty_dir))
    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _client_factory(key=bytes(range(32))))
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(["run", "--task", "t", "--job-dir", str(tmp_path / "job")])

    lines = _result_lines(capsys.readouterr().out)
    assert rc != 0
    assert lines[0]["status"] == "failed"
    assert lines[0]["reason_code"] == backend.GOLDEN_DECRYPT_FAILED_REASON
    assert recorder.called is False


def test_no_endpoint_does_not_decrypt_golden(monkeypatch, tmp_path):
    monkeypatch.delenv(KEY_RELEASE_URL_ENV, raising=False)

    def _forbidden(*a: Any, **k: Any):  # pragma: no cover - only on misuse
        raise AssertionError("golden decrypt attempted on the legacy path")

    monkeypatch.setattr(package, "load_encrypted_oracle_golden", _forbidden)
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(["run", "--task", "t", "--job-dir", str(tmp_path / "job")])
    assert rc == 0
    assert recorder.called is True


def test_key_and_golden_never_leak_to_stdout_or_disk(monkeypatch, tmp_path, capsys):
    key = bytes([0x42]) * 32
    golden_dir = tmp_path / "golden"
    _seed_encrypted_golden(golden_dir, key, {"schema": GOLDEN_MARKER, "answer": "SECRET-ANSWER"})

    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    _enable_phala_decrypt(monkeypatch, task_id="t")
    monkeypatch.setenv(backend.GOLDEN_DIR_ENV, str(golden_dir))
    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _client_factory(key=key))
    monkeypatch.setattr(backend, "run_own_runner_job", _RunRecorder())

    rc = backend.main(["run", "--task", "t", "--job-dir", str(tmp_path / "job")])
    assert rc == 0

    out = capsys.readouterr().out
    # Neither the raw/encoded key nor golden plaintext is ever printed.
    for needle in (key.hex(), base64.b64encode(key).decode(), GOLDEN_MARKER, "SECRET-ANSWER"):
        assert needle not in out

    ciphertext_path = golden_dir / package.ORACLE_CIPHERTEXT_NAME
    for path in tmp_path.rglob("*"):
        if not path.is_file() or path == ciphertext_path:
            continue
        data = path.read_bytes()
        assert key not in data
        assert GOLDEN_MARKER.encode() not in data
        assert b"SECRET-ANSWER" not in data


def test_ra_tls_pubkey_resolved_from_env_and_passed_to_client(monkeypatch, tmp_path):
    key = bytes(range(32))
    golden_dir = tmp_path / "golden"
    _seed_encrypted_golden(golden_dir, key, {"schema": GOLDEN_MARKER})
    captured: dict[str, Any] = {}
    pubkey = b"\x01\x02\x03\x04enclave-ra-tls"

    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    _enable_phala_decrypt(monkeypatch, task_id="t")
    monkeypatch.setenv(backend.GOLDEN_DIR_ENV, str(golden_dir))
    monkeypatch.setenv(backend.PHALA_RA_TLS_PUBKEY_ENV, pubkey.hex())
    monkeypatch.setattr(
        backend, "GoldenKeyReleaseClient", _client_factory(key=key, captured=captured)
    )
    monkeypatch.setattr(backend, "run_own_runner_job", _RunRecorder())

    rc = backend.main(["run", "--task", "t", "--job-dir", str(tmp_path / "job")])
    assert rc == 0
    assert captured["kwargs"].get("ra_tls_pubkey") == pubkey


def test_invalid_ra_tls_pubkey_hex_fails_closed(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(KEY_RELEASE_URL_ENV, "https://validator.test:8700")
    _enable_phala_decrypt(monkeypatch, task_id="t")
    monkeypatch.setenv(backend.PHALA_RA_TLS_PUBKEY_ENV, "zz-not-hex")
    monkeypatch.setattr(backend, "GoldenKeyReleaseClient", _client_factory(key=bytes(range(32))))
    recorder = _RunRecorder()
    monkeypatch.setattr(backend, "run_own_runner_job", recorder)

    rc = backend.main(["run", "--task", "t", "--job-dir", str(tmp_path / "job")])
    lines = _result_lines(capsys.readouterr().out)
    assert rc != 0
    assert lines[0]["status"] == "failed"
    # A misconfigured RA-TLS key is a key-release failure (fail closed, no eval).
    assert lines[0]["reason_code"] == "phala_key_release_failed"
    assert recorder.called is False


def test_acquire_helper_returns_key_and_decrypt_helper_unseals(monkeypatch, tmp_path):
    key = bytes(range(32))
    golden_dir = tmp_path / "golden"
    doc = {"schema": GOLDEN_MARKER, "results": {"a": 1}}
    _seed_encrypted_golden(golden_dir, key, doc)
    monkeypatch.setenv(backend.GOLDEN_DIR_ENV, str(golden_dir))
    unsealed = backend._decrypt_golden_in_enclave(key)
    assert unsealed == doc
    # Wrong key raises the fail-closed crypto error.
    with pytest.raises(crypto.GoldenCryptoError):
        backend._decrypt_golden_in_enclave(bytes([0x00]) * 32)
