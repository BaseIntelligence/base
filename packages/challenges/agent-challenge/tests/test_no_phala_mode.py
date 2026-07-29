"""NO_PHALA temporary host-local unattested mode.

Tests cover:
- Mode OFF (default): settings.no_phala is False; Phala client constructible
  when credentials present; attested path not relaxed.
- Mode ON: Phala client / deploy seams refuse; results marked unattested;
  guest_artifact_proof enforces expected == download == executed.
- Contradiction: NO_PHALA + either attestation flag fails closed at settings.
- Unforgeable: mark_result_unattested always forces attested=False and strips
  execution_proof even if the caller payload claimed otherwise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_challenge.evaluation import no_phala as np
from agent_challenge.evaluation.no_phala import (
    ATTESTATION_STATUS_UNATTESTED,
    CHALLENGE_NO_PHALA_ENV,
    CONTRADICTION_MESSAGE,
    EXECUTION_MODE_NO_PHALA_HOST,
    NO_PHALA_ENV,
    ArtifactProvenanceError,
    NoPhalaModeError,
    assert_envelope_not_attested,
    assert_no_phala_compatible,
    build_guest_artifact_proof,
    is_no_phala_enabled,
    mark_result_unattested,
    refuse_phala_client,
    resolve_no_phala_from_environ,
)
from agent_challenge.evaluation.own_runner.orchestrator import JobResult, TrialOutcome
from agent_challenge.evaluation.own_runner.result_schema import (
    RESULT_LINE_PREFIX,
    build_benchmark_result,
)
from agent_challenge.evaluation.own_runner_backend import (
    PHALA_ATTESTATION_ENABLED_ENV,
    _emit_job_result,
    agent_artifact_sha256,
)
from agent_challenge.sdk.config import ChallengeSettings


@pytest.fixture(autouse=True)
def _clear_no_phala_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHALLENGE_NO_PHALA_ENV, raising=False)
    monkeypatch.delenv(NO_PHALA_ENV, raising=False)
    monkeypatch.delenv(PHALA_ATTESTATION_ENABLED_ENV, raising=False)
    monkeypatch.delenv("CHALLENGE_ATTESTED_REVIEW_ENABLED", raising=False)
    monkeypatch.delenv("CHALLENGE_PHALA_ATTESTATION_ENABLED", raising=False)


# --------------------------------------------------------------------------- #
# S1 — Mode OFF (default)
# --------------------------------------------------------------------------- #


def test_no_phala_default_off() -> None:
    assert resolve_no_phala_from_environ({}) is False
    settings = ChallengeSettings(
        phala_attestation_enabled=False,
        attested_review_enabled=False,
        no_phala=False,
    )
    assert settings.no_phala is False


def test_mode_off_does_not_mark_legacy_emit(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Attested path untouched: gate off + NO_PHALA off => plain result line."""

    monkeypatch.delenv(CHALLENGE_NO_PHALA_ENV, raising=False)
    monkeypatch.delenv(NO_PHALA_ENV, raising=False)
    monkeypatch.delenv(PHALA_ATTESTATION_ENABLED_ENV, raising=False)

    result = _canned_job_result()
    rc = _emit_job_result(result, ["hello-world"])
    assert rc == 0
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith(RESULT_LINE_PREFIX)][-1]
    payload = json.loads(line[len(RESULT_LINE_PREFIX) :])
    assert "attested" not in payload
    assert "execution_mode" not in payload
    assert "guest_artifact_proof" not in payload
    assert "execution_proof" not in payload


# --------------------------------------------------------------------------- #
# Env precedence
# --------------------------------------------------------------------------- #


def test_challenge_prefix_wins_over_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHALLENGE_NO_PHALA_ENV, "false")
    monkeypatch.setenv(NO_PHALA_ENV, "true")
    assert resolve_no_phala_from_environ() is False


def test_plain_no_phala_accepted_when_prefix_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NO_PHALA_ENV, "true")
    assert resolve_no_phala_from_environ() is True
    settings = ChallengeSettings(
        phala_attestation_enabled=False,
        attested_review_enabled=False,
    )
    assert settings.no_phala is True


def test_never_inferred_from_missing_phala_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHALA_API_KEY", raising=False)
    monkeypatch.delenv("PHALA_CLOUD_API_KEY", raising=False)
    assert resolve_no_phala_from_environ({}) is False


# --------------------------------------------------------------------------- #
# S3 — Contradiction fail-closed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "phala,review",
    [
        (True, True),
        (True, False),  # mixed also fails topology first, but contradiction covers both
        (False, True),
    ],
)
def test_contradiction_with_attestation_flags(phala: bool, review: bool) -> None:
    with pytest.raises(ValueError, match="must both be"):
        # Mixed flags fail topology; both-on + no_phala fails contradiction.
        ChallengeSettings(
            phala_attestation_enabled=phala,
            attested_review_enabled=review,
            no_phala=True,
        )


def test_contradiction_both_attestation_on_and_no_phala() -> None:
    with pytest.raises(ValueError, match="NO_PHALA"):
        assert_no_phala_compatible(
            no_phala=True,
            phala_attestation_enabled=True,
            attested_review_enabled=True,
        )
    assert CONTRADICTION_MESSAGE.startswith("NO_PHALA mode cannot")


def test_settings_contradiction_both_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NO_PHALA_ENV, "true")
    from pydantic import ValidationError

    with pytest.raises((ValueError, ValidationError), match="Phala TEE dual flags|NO_PHALA|unattested"):
        ChallengeSettings(
            phala_attestation_enabled=True,
            attested_review_enabled=True,
        )


# --------------------------------------------------------------------------- #
# S2 — Mode ON: no Phala client; local mark
# --------------------------------------------------------------------------- #


def test_phala_client_refused_when_no_phala(monkeypatch: pytest.MonkeyPatch) -> None:
    """T40: Phala selfdeploy module removed entirely."""

    monkeypatch.setenv(NO_PHALA_ENV, "true")
    with pytest.raises(ModuleNotFoundError):
        __import__("agent_challenge.selfdeploy.phala")



def test_eval_deploy_refused_when_no_phala(monkeypatch: pytest.MonkeyPatch) -> None:
    """T40: Phala selfdeploy module removed entirely."""

    monkeypatch.setenv(NO_PHALA_ENV, "true")
    with pytest.raises(ModuleNotFoundError):
        __import__("agent_challenge.selfdeploy.eval")



def test_review_deploy_refused_when_no_phala(monkeypatch: pytest.MonkeyPatch) -> None:
    """T40: Phala selfdeploy module removed entirely."""

    monkeypatch.setenv(NO_PHALA_ENV, "true")
    with pytest.raises(ModuleNotFoundError):
        __import__("agent_challenge.selfdeploy.review")



def test_mode_on_emit_marks_unattested(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(NO_PHALA_ENV, "true")
    monkeypatch.delenv(PHALA_ATTESTATION_ENABLED_ENV, raising=False)

    zip_bytes = b"PK\x03\x04fake-agent-zip-for-no-phala"
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(zip_bytes)
    digest = hashlib.sha256(zip_bytes).hexdigest()
    monkeypatch.setenv("CHALLENGE_PHALA_AGENT_ARTIFACT", str(zip_path))
    monkeypatch.setenv("CHALLENGE_PHALA_AGENT_HASH", digest)

    result = _canned_job_result()
    rc = _emit_job_result(result, ["hello-world"])
    assert rc == 0
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith(RESULT_LINE_PREFIX)][-1]
    payload = json.loads(line[len(RESULT_LINE_PREFIX) :])
    assert payload["attested"] is False
    assert payload["attestation_status"] == ATTESTATION_STATUS_UNATTESTED
    assert payload["execution_mode"] == EXECUTION_MODE_NO_PHALA_HOST
    assert "execution_proof" not in payload
    proof = payload["guest_artifact_proof"]
    assert proof["expected_hash"] == digest
    assert proof["download_hash"] == digest
    assert proof["executed_hash"] == digest
    assert proof["match"] is True
    assert_envelope_not_attested(payload)


# --------------------------------------------------------------------------- #
# S4 — Unforgeable marking
# --------------------------------------------------------------------------- #


def test_mark_result_unattested_cannot_forge_attested() -> None:
    forged = {
        "status": "completed",
        "score": 1.0,
        "attested": True,
        "attestation_status": "attested",
        "execution_proof": {"tier": "phala-tdx", "attestation": {"tdx_quote": "ab" * 100}},
        "attestation_binding": {"agent_hash": "a" * 64},
    }
    out = mark_result_unattested(forged)
    assert out["attested"] is False
    assert out["attestation_status"] == ATTESTATION_STATUS_UNATTESTED
    assert out["execution_mode"] == EXECUTION_MODE_NO_PHALA_HOST
    assert "execution_proof" not in out
    assert "attestation_binding" not in out
    with pytest.raises(ValueError, match="must not claim attested"):
        assert_envelope_not_attested({**out, "attested": True})


def test_mark_ignores_caller_attested_true_kw() -> None:
    """Even if payload insists attested=True, output is False."""

    out = mark_result_unattested({"attested": True, "score": 0.5})
    assert out[np.RESULT_KEY_ATTESTED] is False


# --------------------------------------------------------------------------- #
# S5 — Artifact provenance
# --------------------------------------------------------------------------- #


def test_artifact_proof_enforces_triple_match() -> None:
    h = "ab" * 32
    proof = build_guest_artifact_proof(
        expected_hash=h, download_hash=h, executed_hash=h
    )
    assert proof["match"] is True
    with pytest.raises(ArtifactProvenanceError, match="mismatch"):
        build_guest_artifact_proof(
            expected_hash=h,
            download_hash=h,
            executed_hash="cd" * 32,
        )


def test_artifact_sha_matches_executed(tmp_path: Path) -> None:
    data = b"PK\x03\x04miner-zip-bytes"
    path = tmp_path / "a.zip"
    path.write_bytes(data)
    digest = agent_artifact_sha256(path)
    assert digest == hashlib.sha256(data).hexdigest()
    proof = build_guest_artifact_proof(
        expected_hash=digest, download_hash=digest, executed_hash=digest
    )
    assert proof["match"] is True


# --------------------------------------------------------------------------- #
# Health / version surface
# --------------------------------------------------------------------------- #


async def test_health_shows_no_phala_off(client) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["no_phala"] is False
    assert body["attestation_mode"] == "standard"


async def test_health_shows_no_phala_on() -> None:
    """health_fields is the source of truth for /health mode visibility."""

    fields = np.health_fields(no_phala=True)
    assert fields == {
        "no_phala": True,
        "attestation_mode": EXECUTION_MODE_NO_PHALA_HOST,
    }
    off = np.health_fields(no_phala=False)
    assert off == {"no_phala": False, "attestation_mode": "standard"}


def test_is_no_phala_enabled_settings_flag() -> None:
    assert is_no_phala_enabled(settings_flag=True) is True
    assert is_no_phala_enabled(settings_flag=False) is False


def test_refuse_phala_client_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NO_PHALA_ENV, raising=False)
    refuse_phala_client()  # no raise


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _canned_job_result() -> JobResult:
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
        trial_outcomes=[
            TrialOutcome(
                task_name="hello-world",
                trial_name="hello-world__attempt-0",
                status="completed",
                rewards={"reward": 1.0},
            )
        ],
        benchmark_result=build_benchmark_result(
            status="completed",
            score=1.0,
            resolved=1,
            total=1,
            reason_code=None,
        ),
    )
