"""VAL-DEPLOY-026: a rejected/unaccepted attested result is surfaced to the miner.

When the validator rejects or parks a submission's attested result (attestation
absent, quote fails verification, measurement not on the allowlist, nonce stale),
the miner-facing self-deploy ``result`` surface must report a clear, non-sensitive
NOT-ACCEPTED verdict with a coarse reason -- never reporting success, never
reporting a (fabricated) score, and never leaking golden material, the golden key,
or any quote-embedded secret.

These are discriminators: each rejection cause is driven independently over a
genuinely valid attested envelope (so a naive "attested ⇒ accepted" surface would
FAIL), and the accepted control confirms the surface does not reject a good run.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

from agent_challenge.canonical.attested_result import emit_attested_benchmark_result
from agent_challenge.evaluation.own_runner.result_schema import build_benchmark_result
from agent_challenge.keyrelease.client import KEY_RELEASE_FAILED_REASON
from agent_challenge.keyrelease.nonce import NonceState
from agent_challenge.selfdeploy import cli
from agent_challenge.selfdeploy import result as result_mod

# A sentinel that stands in for any secret/golden material. It never appears in
# the surfaced envelope, so it must never appear in any surfaced verdict either.
GOLDEN_SENTINEL = "GOLDEN-PLAINTEXT-SENTINEL-do-not-leak"


def _measurement() -> dict:
    return {
        "mrtd": "a" * 96,
        "rtmr0": "a" * 96,
        "rtmr1": "a" * 96,
        "rtmr2": "a" * 96,
        "compose_hash": "b" * 64,
        "os_image_hash": "b" * 64,
    }


class _FakeQuote:
    quote = "deadbeef" * 16
    event_log = [{"event": "compose-hash", "payload": "c" * 64}]
    # schema-v2 Eval keys (or dstack aliases cpu_count/memory_size)
    vm_config = {"vcpu": 1, "memory_mb": 2048}


class _FakeProvider:
    def get_quote(self, report_data):  # noqa: ARG002
        return _FakeQuote()


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


def _failclosed_line(reason: str = KEY_RELEASE_FAILED_REASON) -> str:
    return "BASE_BENCHMARK_RESULT=" + json.dumps(
        {"reason_code": reason, "resolved": 0, "score": 0.0, "status": "failed", "total": 1},
        sort_keys=True,
    )


def _run_result_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# Function-level: evaluate_acceptance yields a coarse NOT-ACCEPTED verdict per
# rejection cause and does not reject a fully valid, allowlisted, verified run.
# --------------------------------------------------------------------------- #
def test_attestation_absent_is_not_accepted():
    surfaced = result_mod.surface_result(_failclosed_line())
    verdict = result_mod.evaluate_acceptance(surfaced)
    assert verdict.accepted is False
    assert verdict.reason == result_mod.ACCEPTANCE_ATTESTATION_ABSENT


def test_quote_not_verified_is_not_accepted():
    surfaced = result_mod.surface_result(_attested_stdout(), quote_verifier=lambda _q: False)
    verdict = result_mod.evaluate_acceptance(surfaced)
    assert verdict.accepted is False
    assert verdict.reason == result_mod.ACCEPTANCE_ATTESTATION_NOT_VERIFIED


def test_measurement_not_allowlisted_is_not_accepted():
    surfaced = result_mod.surface_result(_attested_stdout())
    verdict = result_mod.evaluate_acceptance(surfaced, measurement_allowlisted=False)
    assert verdict.accepted is False
    assert verdict.reason == result_mod.ACCEPTANCE_MEASUREMENT_NOT_ALLOWLISTED


def test_stale_nonce_is_not_accepted():
    surfaced = result_mod.surface_result(_attested_stdout())
    verdict = result_mod.evaluate_acceptance(surfaced, nonce_state=NonceState.EXPIRED)
    assert verdict.accepted is False
    assert verdict.reason == result_mod.ACCEPTANCE_NONCE_STALE


def test_fully_valid_verified_allowlisted_fresh_run_is_accepted():
    surfaced = result_mod.surface_result(_attested_stdout(), quote_verifier=lambda _q: True)
    verdict = result_mod.evaluate_acceptance(
        surfaced,
        measurement_allowlisted=True,
        nonce_state=NonceState.OK,
        key_grant_ok=True,
    )
    assert verdict.accepted is True
    assert verdict.reason is None


def test_tampered_binding_is_not_accepted():
    # A surfaced result whose scores were altered breaks the report_data binding.
    stdout = _attested_stdout()
    execution_proof, binding = result_mod.extract_envelope(stdout)
    binding["scores"]["t1"] = 0.0
    check = result_mod.verify_report_data_binding(execution_proof, binding)
    assert check.valid is False
    # A surfaced result carrying an invalid binding is never accepted.
    surfaced = result_mod.surface_result(stdout)
    assert surfaced.binding_check.valid is True  # sanity: the on-wire binding is intact
    forced = result_mod.SurfacedResult(
        attested=True,
        benchmark_result=surfaced.benchmark_result,
        status=surfaced.status,
        reason_code=surfaced.reason_code,
        execution_proof=surfaced.execution_proof,
        binding=surfaced.binding,
        binding_check=check,
        quote_verified=None,
    )
    verdict = result_mod.evaluate_acceptance(forced)
    assert verdict.accepted is False
    assert verdict.reason == result_mod.ACCEPTANCE_BINDING_MISMATCH


def test_no_signal_is_pending_not_a_false_accept():
    surfaced = result_mod.surface_result(_attested_stdout())
    verdict = result_mod.evaluate_acceptance(surfaced)
    # Without any validator signal the verdict is PENDING (None), never a
    # fabricated acceptance.
    assert verdict.accepted is None
    assert verdict.reason is None


# --------------------------------------------------------------------------- #
# CLI-level: each rejection cause surfaces a clear, non-sensitive verdict with a
# non-zero exit, NO score, and no secret material.
# --------------------------------------------------------------------------- #
def _assert_rejection_output(code: int, out: str, expected_reason: str):
    assert code == 1, out
    payload = json.loads(out)
    assert payload["accepted"] is False
    assert payload["reason"] == expected_reason
    # No fabricated score and no quote/secret material in the rejection surface.
    assert "score" not in payload
    assert "scores" not in payload
    assert "attestation" not in payload
    assert "tdx_quote" not in out
    assert _FakeQuote.quote not in out
    assert GOLDEN_SENTINEL not in out
    # The reason is a coarse, non-sensitive label (whitelist).
    assert payload["reason"] in {
        result_mod.ACCEPTANCE_ATTESTATION_ABSENT,
        result_mod.ACCEPTANCE_ATTESTATION_NOT_VERIFIED,
        result_mod.ACCEPTANCE_MEASUREMENT_NOT_ALLOWLISTED,
        result_mod.ACCEPTANCE_NONCE_STALE,
        result_mod.ACCEPTANCE_NONCE_CONSUMED,
        result_mod.ACCEPTANCE_NONCE_UNKNOWN,
        result_mod.ACCEPTANCE_BINDING_MISMATCH,
        result_mod.ACCEPTANCE_KEY_GRANT_MISSING,
    }


def test_cli_attestation_absent_rejection(tmp_path):
    path = tmp_path / "run.txt"
    path.write_text(_failclosed_line())
    code, out, _err = _run_result_cli(["result", "--from", str(path)])
    _assert_rejection_output(code, out, result_mod.ACCEPTANCE_ATTESTATION_ABSENT)


def test_cli_quote_not_verified_rejection(tmp_path):
    path = tmp_path / "run.txt"
    path.write_text(_attested_stdout())
    code, out, _err = _run_result_cli(["result", "--from", str(path), "--quote-verified", "false"])
    _assert_rejection_output(code, out, result_mod.ACCEPTANCE_ATTESTATION_NOT_VERIFIED)


def test_cli_measurement_not_allowlisted_rejection(tmp_path):
    path = tmp_path / "run.txt"
    path.write_text(_attested_stdout())
    allow = tmp_path / "allow.json"
    # An allowlist that does NOT contain the run's measurement.
    other = {**_measurement(), "mrtd": "f" * 96}
    allow.write_text(json.dumps([other]))
    code, out, _err = _run_result_cli(["result", "--from", str(path), "--allowlist", str(allow)])
    _assert_rejection_output(code, out, result_mod.ACCEPTANCE_MEASUREMENT_NOT_ALLOWLISTED)


def test_cli_stale_nonce_rejection(tmp_path):
    path = tmp_path / "run.txt"
    path.write_text(_attested_stdout())
    code, out, _err = _run_result_cli(["result", "--from", str(path), "--nonce-state", "stale"])
    _assert_rejection_output(code, out, result_mod.ACCEPTANCE_NONCE_STALE)


def test_cli_accepted_run_reports_acceptance_and_scores(tmp_path):
    path = tmp_path / "run.txt"
    path.write_text(_attested_stdout())
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps([_measurement()]))
    code, out, _err = _run_result_cli(
        [
            "result",
            "--from",
            str(path),
            "--allowlist",
            str(allow),
            "--quote-verified",
            "true",
            "--nonce-state",
            "ok",
        ]
    )
    assert code == 0, out
    summary = json.loads(out)
    assert summary["acceptance"] == {"accepted": True, "reason": None}
    assert summary["allowlist_verdict"]["verdict"] == "IN-LIST"
    # An accepted run DOES surface its scores/attestation (contrast with reject).
    assert summary["scores"] == {"t1": 1.0, "t2": 0.0}
    assert summary["attested"] is True
