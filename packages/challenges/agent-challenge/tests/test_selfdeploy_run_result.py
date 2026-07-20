"""Run fail-closed + attested-result surfacing/verification for the miner CLI.

Covers VAL-DEPLOY-011 (wrong/unreachable key-release endpoint → clear failure,
no golden key, NO fabricated attested result), VAL-DEPLOY-013 (a completed run
surfaces a well-formed attested-result envelope), and VAL-DEPLOY-014 (a surfaced
quote's ``report_data`` recomputes to the documented §6 binding for the run).

Run assertions drive the REAL in-CVM backend (`own_runner_backend.main`) with the
endpoint wired in for the unreachable case (it fails closed before touching docker
or golden), and an injected fail-closed backend for the deny case. Live TDX quote
verification (Phala verify / dcap-qvl) is a M6 concern; here the quote-verify hook
is injected.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

from agent_challenge.canonical.attested_result import emit_attested_benchmark_result
from agent_challenge.evaluation.own_runner.result_schema import build_benchmark_result
from agent_challenge.keyrelease.client import KEY_RELEASE_FAILED_REASON
from agent_challenge.selfdeploy import cli
from agent_challenge.selfdeploy import result as result_mod
from agent_challenge.selfdeploy import run as run_mod

URL = "https://validator.example/keyrelease"

# A definitely-closed local port (fail fast, no external network, no spend).
UNREACHABLE_URL = "http://127.0.0.1:9/"


def _failclosed_line(reason: str = KEY_RELEASE_FAILED_REASON) -> str:
    """A fail-closed BASE_BENCHMARK_RESULT= line (score 0, no attestation)."""

    return "BASE_BENCHMARK_RESULT=" + json.dumps(
        {"reason_code": reason, "resolved": 0, "score": 0.0, "status": "failed", "total": 1},
        sort_keys=True,
    )


# --------------------------------------------------------------------------- #
# Fixtures: build a genuine attested-result stdout line
# --------------------------------------------------------------------------- #
class _FakeQuote:
    quote = "deadbeef" * 16
    event_log = [{"event": "compose-hash", "payload": "c" * 64}]
    # schema-v2 Eval keys (or dstack aliases cpu_count/memory_size)
    vm_config = {"vcpu": 1, "memory_mb": 2048}


class _FakeProvider:
    def get_quote(self, report_data):  # noqa: ARG002
        return _FakeQuote()


def _measurement() -> dict:
    return {
        "mrtd": "a" * 96,
        "rtmr0": "a" * 96,
        "rtmr1": "a" * 96,
        "rtmr2": "a" * 96,
        "compose_hash": "b" * 64,
        "os_image_hash": "b" * 64,
    }


def _attested_stdout(scores=None) -> str:
    scores = scores or {"t1": 1.0, "t2": 0.0}
    br = build_benchmark_result(
        status="completed", score=0.5, resolved=1, total=2, reason_code="ok"
    )
    buffer = io.StringIO()
    emit_attested_benchmark_result(
        benchmark_result=br,
        canonical_measurement=_measurement(),
        rtmr3="d" * 96,
        agent_hash="agent-abc",
        task_ids=sorted(scores),
        scores=scores,
        validator_nonce="nonce-123",
        quote_provider=_FakeProvider(),
        manifest_sha256="m" * 64,
        stream=buffer,
    )
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-011: wrong/unreachable endpoint → clear failure, no fabricated result
# --------------------------------------------------------------------------- #
def test_unreachable_endpoint_fails_closed_no_attested_result(tmp_path):
    # A standalone self-deploy invocation has no validator-issued immutable Eval
    # plan, so it fails closed before docker, dstack, or golden handling.
    outcome = run_mod.run_eval(
        job_dir=str(tmp_path),
        task_ids=["hello-world"],
        key_release_url=UNREACHABLE_URL,
    )
    assert outcome.succeeded is False
    assert outcome.attested is False
    assert outcome.exit_code != 0
    assert outcome.surfaced is not None
    assert outcome.surfaced.reason_code == "terminal_bench_failed"
    assert outcome.surfaced.attestation is None  # NO fabricated attestation
    assert outcome.clear_error and "failed closed" in outcome.clear_error.lower()


def test_denying_endpoint_fails_closed_no_attested_result():
    # A wrong endpoint that DENIES the quote: model the backend's fail-closed
    # emission (score 0, key-release reason, no envelope).
    def denying_backend(argv):  # noqa: ARG001
        print(_failclosed_line())
        return 1

    outcome = run_mod.run_eval(
        job_dir="/tmp/job",
        task_ids=["t1"],
        key_release_url="https://wrong.example/kr",
        backend_main=denying_backend,
    )
    assert outcome.succeeded is False
    assert outcome.attested is False
    assert outcome.surfaced is not None
    assert outcome.surfaced.attestation is None
    assert outcome.clear_error


def test_run_cli_reports_clear_failure_and_no_attested_result(tmp_path):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(
            [
                "run",
                "--job-dir",
                str(tmp_path),
                "--task",
                "hello-world",
                "--key-release-url",
                UNREACHABLE_URL,
            ]
        )
    assert code != 0
    # A clear miner-facing error, and no attested envelope in stdout.
    assert "failed closed" in err.getvalue().lower()
    assert "tdx_quote" not in out.getvalue()
    assert "execution_proof" not in out.getvalue()


def test_run_endpoint_wired_into_backend_env():
    # The run wires the operator-supplied endpoint into the backend via the
    # key-release env var (the backend requests golden from exactly that URL).
    seen = {}

    def capturing_backend(argv):  # noqa: ARG001
        import os

        from agent_challenge.keyrelease.client import KEY_RELEASE_URL_ENV

        seen["url"] = os.environ.get(KEY_RELEASE_URL_ENV)
        print(_failclosed_line())
        return 1

    run_mod.run_eval(
        job_dir="/tmp/job",
        task_ids=["t1"],
        key_release_url="https://exact-endpoint.example/kr",
        backend_main=capturing_backend,
    )
    assert seen["url"] == "https://exact-endpoint.example/kr"


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-013: a completed run surfaces a well-formed attested envelope
# --------------------------------------------------------------------------- #
def test_completed_run_surfaces_wellformed_envelope():
    surfaced = result_mod.surface_result(_attested_stdout())
    assert surfaced.attested is True
    att = surfaced.attestation
    assert isinstance(att["tdx_quote"], str) and att["tdx_quote"]
    assert isinstance(att["event_log"], list) and att["event_log"]
    assert isinstance(att["report_data"], str) and len(att["report_data"]) == 128
    measurement = att["measurement"]
    for field in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3", "compose_hash", "os_image_hash"):
        assert isinstance(measurement[field], str) and measurement[field], field
    # Per-task scores are surfaced and well-typed.
    assert surfaced.scores == {"t1": 1.0, "t2": 0.0}
    assert all(isinstance(v, (int, float)) for v in surfaced.scores.values())


def test_result_cli_surfaces_envelope_fields(tmp_path, capsys):
    path = tmp_path / "run.txt"
    path.write_text(_attested_stdout())
    code = cli.main(["result", "--from", str(path)])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["attested"] is True
    assert set(summary["attestation"]) >= {
        "tdx_quote",
        "event_log",
        "report_data",
        "measurement",
        "vm_config",
    }
    assert summary["scores"] == {"t1": 1.0, "t2": 0.0}


def test_fail_closed_run_surfaces_no_attestation():
    surfaced = result_mod.surface_result(_failclosed_line())
    assert surfaced.attested is False
    assert surfaced.attestation is None
    assert surfaced.reason_code == KEY_RELEASE_FAILED_REASON


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-014: surfaced quote's report_data recomputes to the §6 binding
# --------------------------------------------------------------------------- #
def test_surfaced_report_data_recomputes_to_binding():
    surfaced = result_mod.surface_result(_attested_stdout())
    check = surfaced.binding_check
    assert check is not None
    assert check.valid is True
    assert check.report_data_matches is True
    assert check.scores_digest_matches is True
    assert check.measurement_consistent is True
    # The recomputed report_data equals the quote's report_data exactly.
    assert check.recomputed_report_data == surfaced.attestation["report_data"]


def test_tampered_scores_break_the_binding():
    stdout = _attested_stdout()
    envelope = result_mod.extract_envelope(stdout)
    assert envelope is not None
    execution_proof, binding = envelope
    binding["scores"]["t1"] = 0.0  # alter a reported score
    check = result_mod.verify_report_data_binding(execution_proof, binding)
    assert check.valid is False
    assert check.scores_digest_matches is False


def test_tampered_nonce_breaks_the_binding():
    stdout = _attested_stdout()
    execution_proof, binding = result_mod.extract_envelope(stdout)
    binding["validator_nonce"] = "different-nonce"
    check = result_mod.verify_report_data_binding(execution_proof, binding)
    assert check.valid is False
    assert check.report_data_matches is False


def test_quote_verifier_hook_is_surfaced():
    # An injected quote verifier (Phala verify / dcap-qvl at M6) verdict rides
    # along with the surfaced result.
    surfaced = result_mod.surface_result(_attested_stdout(), quote_verifier=lambda q: True)
    assert surfaced.quote_verified is True
    summary = surfaced.summary()
    assert summary["quote_verified"] is True

    tampered = result_mod.surface_result(_attested_stdout(), quote_verifier=lambda q: False)
    assert tampered.quote_verified is False


def test_result_cli_verdict_with_allowlist(tmp_path, capsys):
    path = tmp_path / "run.txt"
    path.write_text(_attested_stdout())
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps([{**_measurement(), "key_provider": "kms"}]))
    code = cli.main(["result", "--from", str(path), "--allowlist", str(allow)])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["allowlist_verdict"]["verdict"] == "IN-LIST"
