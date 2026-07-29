"""T22: unattested flag ON must NOT become a generic bypass of prism validations.

Complements T20 (narrow missing-bundle admit). Every case below runs with the
unattested / NO_PHALA flag explicitly ON and asserts fail-closed behavior for
a distinct validation other than missing_constation_bundle.

Matrix (validation | still enforced when unattested ON) is asserted by the
test ids and summarized in evidence T22-no-bypass/MATRIX.md.
"""

from __future__ import annotations

import asyncio
import io
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.constation import CheckOutcome, ConstationBundle, miner_fault_reason
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.ingestion import (
    ResultIngestionError,
    _evaluate_constation_gate,
    ingest_work_unit_result,
    parse_execution_proof,
    verify_proof_integrity,
)
from prism_challenge.proof import (
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)
from prism_challenge.unattested_execution import (
    CHALLENGE_NO_PHALA_ENV,
    CHALLENGE_UNATTESTED_EXECUTION_ENV,
    NO_PHALA_ENV,
    is_unattested_execution_enabled,
)

WORKER_KEY = "//WorkerUnattestedNoBypass"
DIGEST = "sha256:" + ("22" * 32)
MISSING = "miner_fault:missing_constation_bundle"

TINY_ARCH = """
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    return TinyLM(ctx.vocab_size)
"""

TINY_TRAIN = """
import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        nv = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, nv), batch.tokens[:, 1:].reshape(-1) % nv
        )
        loss.backward()
        opt.step()
"""

_SHARD = (
    '{{"id": "doc-{i}", "text": "the locked fineweb edu training sample number {i} '
    'has enough bytes to cover several challenge instrument batches deterministically"}}\n'
)


def _stage_train(root: Path) -> Path:
    data_dir = root / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.jsonl").write_text(
        "".join(_SHARD.format(i=i) for i in range(64)), encoding="utf-8"
    )
    return data_dir


def _zip_bytes() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", TINY_ARCH)
        archive.writestr("training.py", TINY_TRAIN)
    return stream.getvalue()


def _settings(tmp_path: Path, **extra: Any) -> PrismSettings:
    kw: dict[str, Any] = dict(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'coord.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=False,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        sequence_length=16,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        base_eval_artifact_root=tmp_path / "artifacts",
        worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY),
        constation_base_url="http://base-constation.test",
        constation_internal_token="constation-tok",
    )
    kw.update(extra)
    return PrismSettings(**kw)


def _manifest(marker: str = "ok") -> dict[str, Any]:
    return {
        "schema_version": "prism_run_manifest.v2",
        "metrics": {
            "token_accuracy": 0.5,
            "loss": 1.0,
            "step": 1,
            "marker": marker,
        },
        "timing": {"wall_seconds": 1.0},
    }


def _clear_unattested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        CHALLENGE_UNATTESTED_EXECUTION_ENV,
        CHALLENGE_NO_PHALA_ENV,
        NO_PHALA_ENV,
    ):
        monkeypatch.delenv(key, raising=False)


def _enable_unattested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn unattested ON via NO_PHALA alias (ChallengeSettings-safe for HTTP)."""
    _clear_unattested_env(monkeypatch)
    monkeypatch.setenv(NO_PHALA_ENV, "true")
    assert is_unattested_execution_enabled() is True


def _seed_client(client: TestClient) -> str:
    seed = client.post(
        "/internal/v1/bridge/submissions",
        content=_zip_bytes(),
        headers={
            "Authorization": "Bearer secret",
            "X-Base-Verified-Hotkey": "hk-owner",
            "X-Submission-Filename": "project.zip",
            "Content-Type": "application/octet-stream",
        },
    )
    assert seed.status_code == 200, seed.text
    return str(seed.json()["id"])


def _score_row(db_path: Path, submission_id: str) -> Any:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()


def _http_post_result(
    client: TestClient,
    *,
    sid: str,
    result: dict[str, Any],
    proof: dict[str, Any] | None = None,
    slug: str,
) -> Any:
    body: dict[str, Any] = {
        "api_version": "1.0",
        "work_unit_id": sid,
        "assignment_id": sid,
        "submission_ref": "hk-owner",
        "challenge_slug": slug,
        "result": result,
    }
    if proof is not None:
        body["proof"] = proof
    return client.post(
        "/internal/v1/work_units/result",
        json=body,
        headers={"Authorization": "Bearer secret"},
    )


def _assert_not_missing_bundle_bypass(resp: Any, *, expected_code: str | None = None) -> str:
    """422 fail-closed with a code that is NOT the missing-bundle unattested path."""
    assert resp.status_code == 422, resp.text
    detail = resp.json().get("detail")
    code = detail.get("code") if isinstance(detail, dict) else str(detail)
    assert code != MISSING, f"must not collapse to missing-bundle: {resp.text}"
    assert "missing_constation" not in str(code), resp.text
    if expected_code is not None:
        assert code == expected_code, resp.text
    return str(code)


def _failing_bundle() -> ConstationBundle:
    """Minimal bundle that fails allowlist (six-check path, not missing-bundle)."""
    return ConstationBundle(
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        variant="default",
        digest=DIGEST,
        work_unit_id="wu-t22",
        miner_hotkey="hk",
        pod_id="pod-1",
        nonce="nonce-1",
        signed_attestation="sig-bytes",
        expected_sealed_manifest_hashes={"a": "1" * 64},
        reported_sealed_manifest_hashes={"a": "1" * 64},
        lium_declared_digest=None,
        constation_gap_budget_seconds=60.0,
        constation_observed_max_gap_seconds=1.0,
    )


# --- Unit: proof shape / integrity still enforced with flag ON --------------------


def test_flag_on_parse_still_rejects_proof_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V1 proof_missing — unattested does not skip ExecutionProof presence."""
    _enable_unattested(monkeypatch)
    with pytest.raises(ResultIngestionError) as exc:
        parse_execution_proof({"executed": 1})
    assert exc.value.reason == "proof_missing"


def test_flag_on_parse_still_rejects_proof_bad_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V2 proof_bad_version."""
    _enable_unattested(monkeypatch)
    signer = worker_signer_from_key(WORKER_KEY)
    manifest = _manifest()
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=compute_manifest_sha256(manifest),
        unit_id="u1",
        image_digest=DIGEST,
        constation_digest=DIGEST,
    ).model_dump(mode="json")
    proof["version"] = 99
    with pytest.raises(ResultIngestionError) as exc:
        parse_execution_proof({PROOF_PAYLOAD_KEY: proof, MANIFEST_PAYLOAD_KEY: manifest})
    assert exc.value.reason == "proof_bad_version"


def test_flag_on_parse_still_rejects_proof_bad_manifest_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V3 proof_bad_manifest_hash."""
    _enable_unattested(monkeypatch)
    signer = worker_signer_from_key(WORKER_KEY)
    manifest = _manifest()
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=compute_manifest_sha256(manifest),
        unit_id="u1",
        image_digest=DIGEST,
        constation_digest=DIGEST,
    ).model_dump(mode="json")
    proof["manifest_sha256"] = "not-a-valid-hex-digest"
    with pytest.raises(ResultIngestionError) as exc:
        parse_execution_proof({PROOF_PAYLOAD_KEY: proof, MANIFEST_PAYLOAD_KEY: manifest})
    assert exc.value.reason == "proof_bad_manifest_hash"


def test_flag_on_parse_still_rejects_proof_missing_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V4 proof_missing_signature."""
    _enable_unattested(monkeypatch)
    signer = worker_signer_from_key(WORKER_KEY)
    manifest = _manifest()
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=compute_manifest_sha256(manifest),
        unit_id="u1",
        image_digest=DIGEST,
        constation_digest=DIGEST,
    ).model_dump(mode="json")
    proof["worker_signature"] = {"worker_pubkey": "only-pubkey"}
    with pytest.raises(ResultIngestionError) as exc:
        parse_execution_proof({PROOF_PAYLOAD_KEY: proof, MANIFEST_PAYLOAD_KEY: manifest})
    assert exc.value.reason == "proof_missing_signature"


def test_flag_on_verify_still_rejects_manifest_tampered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V5 manifest_tampered — hash mismatch after sign."""
    _enable_unattested(monkeypatch)
    signer = worker_signer_from_key(WORKER_KEY)
    unit_id = "u-tamper"
    manifest = _manifest("orig")
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=compute_manifest_sha256(manifest),
        unit_id=unit_id,
        image_digest=DIGEST,
        constation_digest=DIGEST,
    )
    tampered = _manifest("mutated")
    with pytest.raises(ResultIngestionError) as exc:
        verify_proof_integrity(proof, unit_id=unit_id, manifest=tampered)
    assert exc.value.reason == "manifest_tampered"


def test_flag_on_verify_still_rejects_signature_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V6 signature_invalid — corrupt worker sig bytes."""
    _enable_unattested(monkeypatch)
    signer = worker_signer_from_key(WORKER_KEY)
    unit_id = "u-sig"
    manifest = _manifest()
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=compute_manifest_sha256(manifest),
        unit_id=unit_id,
        image_digest=DIGEST,
        constation_digest=DIGEST,
    )
    corrupt = proof.model_copy(
        update={"worker_signature": proof.worker_signature.model_copy(update={"sig": "0x00"})}
    )
    with pytest.raises(ResultIngestionError) as exc:
        verify_proof_integrity(corrupt, unit_id=unit_id, manifest=manifest)
    assert exc.value.reason == "signature_invalid"


def test_flag_on_result_malformed_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V7 result_malformed — non-Mapping result rejected at ingest boundary."""
    _enable_unattested(monkeypatch)

    class _NoWorker:
        settings = type("S", (), {"worker_plane": type("W", (), {"enabled": False})()})()
        repository = None

    async def _run() -> None:
        with pytest.raises(ResultIngestionError) as exc:
            await ingest_work_unit_result(
                worker=_NoWorker(),  # type: ignore[arg-type]
                work_unit_id="wu",
                submission_ref="hk",
                result="not-an-object",  # type: ignore[arg-type]
            )
        assert exc.value.reason == "result_malformed"

    asyncio.run(_run())


# --- Unit: constation gate narrowness with flag ON --------------------------------


def test_flag_on_infra_fault_still_not_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V8 infra_fault — unattested does not swallow infra path (T20 expand)."""
    _enable_unattested(monkeypatch)
    gate = _evaluate_constation_gate(
        bundle=None,
        check_allowlist=None,
        check_nonce=None,
        verify_constation_signature=None,
        constation_infra_fault="constation_unavailable",
        constation_attempt=0,
        max_constation_attempts=3,
    )
    assert gate.admit is False
    assert gate.retryable is True
    assert gate.reason is not None
    assert gate.reason.startswith("infra_fault:")
    assert gate.reason != MISSING


def test_flag_on_infra_fault_retry_exhausted_still_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V9 infra_fault retry exhausted — still no admit under unattested."""
    _enable_unattested(monkeypatch)
    gate = _evaluate_constation_gate(
        bundle=None,
        check_allowlist=None,
        check_nonce=None,
        verify_constation_signature=None,
        constation_infra_fault="constation_unavailable",
        constation_attempt=3,
        max_constation_attempts=3,
    )
    assert gate.admit is False
    assert gate.retryable is False
    assert gate.reason is not None
    assert gate.reason.startswith("infra_fault:")


def test_flag_on_bundle_present_failed_six_check_still_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V10 six-check fail — unattested only applies when bundle is None.

    A present but invalid bundle must still fail closed (miner_fault), not
    fall through to the unattested missing-bundle admit branch.
    """
    _enable_unattested(monkeypatch)
    bundle = _failing_bundle()

    def _deny_allowlist(**_kwargs: Any) -> CheckOutcome:
        return CheckOutcome(ok=False, reason="unknown_digest")

    def _ok_nonce(**_kwargs: Any) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    def _ok_sig(_att: object) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    gate = _evaluate_constation_gate(
        bundle=bundle,
        check_allowlist=_deny_allowlist,
        check_nonce=_ok_nonce,
        verify_constation_signature=_ok_sig,
        constation_infra_fault=None,
        constation_attempt=0,
        max_constation_attempts=3,
    )
    assert gate.admit is False
    assert gate.constation_ok is False
    assert gate.reason is not None
    assert gate.reason.startswith("miner_fault:")
    assert gate.reason != MISSING
    assert "unattested" not in (gate.reason or "")
    assert gate.reason == miner_fault_reason("unknown_digest")


def test_flag_on_bundle_present_checkers_unavailable_is_infra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V11 checkers unavailable with bundle present → infra_fault, not unattested admit."""
    _enable_unattested(monkeypatch)
    gate = _evaluate_constation_gate(
        bundle=_failing_bundle(),
        check_allowlist=None,
        check_nonce=None,
        verify_constation_signature=None,
        constation_infra_fault=None,
        constation_attempt=0,
        max_constation_attempts=3,
    )
    assert gate.admit is False
    assert gate.retryable is True
    assert gate.reason is not None
    assert gate.reason.startswith("infra_fault:")


# --- HTTP surface: flag ON still 422 on non-bundle faults -------------------------


def test_http_flag_on_still_rejects_missing_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H1 missing proof → result_envelope_invalid (expands T20 S3)."""
    _enable_unattested(monkeypatch)
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        sid = _seed_client(client)
        resp = _http_post_result(
            client,
            sid=sid,
            slug=settings.slug,
            result={"executed": 1, MANIFEST_PAYLOAD_KEY: _manifest()},
        )
        # Envelope requires proof: ExecutionProof — fails closed before ingest.
        code = _assert_not_missing_bundle_bypass(
            resp, expected_code="result_envelope_invalid"
        )
        assert code == "result_envelope_invalid"
        assert _score_row(tmp_path / "coord.sqlite3", sid) is None


def test_http_flag_on_still_rejects_bad_proof_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H2 bad proof version → result_envelope_invalid via HTTP."""
    _enable_unattested(monkeypatch)
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path)
    signer = worker_signer_from_key(WORKER_KEY)
    with TestClient(create_app(settings)) as client:
        sid = _seed_client(client)
        manifest = _manifest()
        proof = build_execution_proof(
            signer=signer,
            manifest_sha256=compute_manifest_sha256(manifest),
            unit_id=sid,
            image_digest=DIGEST,
            constation_digest=DIGEST,
        ).model_dump(mode="json")
        proof["version"] = 7
        resp = _http_post_result(
            client,
            sid=sid,
            slug=settings.slug,
            result={
                "executed": 1,
                PROOF_PAYLOAD_KEY: proof,
                MANIFEST_PAYLOAD_KEY: manifest,
            },
            proof=proof,
        )
        # SDK ExecutionProof.version is Literal[1] — envelope rejects before ingest.
        _assert_not_missing_bundle_bypass(resp, expected_code="result_envelope_invalid")
        assert _score_row(tmp_path / "coord.sqlite3", sid) is None


def test_http_flag_on_still_rejects_manifest_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H3 manifest_tampered via HTTP."""
    _enable_unattested(monkeypatch)
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path)
    signer = worker_signer_from_key(WORKER_KEY)
    with TestClient(create_app(settings)) as client:
        sid = _seed_client(client)
        signed_manifest = _manifest("signed")
        proof = build_execution_proof(
            signer=signer,
            manifest_sha256=compute_manifest_sha256(signed_manifest),
            unit_id=sid,
            image_digest=DIGEST,
            constation_digest=DIGEST,
        ).model_dump(mode="json")
        forwarded = _manifest("tampered-after-sign")
        resp = _http_post_result(
            client,
            sid=sid,
            slug=settings.slug,
            result={
                "executed": 1,
                PROOF_PAYLOAD_KEY: proof,
                MANIFEST_PAYLOAD_KEY: forwarded,
            },
            proof=proof,
        )
        _assert_not_missing_bundle_bypass(resp, expected_code="manifest_tampered")
        assert _score_row(tmp_path / "coord.sqlite3", sid) is None


def test_http_flag_on_still_rejects_signature_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H4 signature_invalid via HTTP (corrupt sig)."""
    _enable_unattested(monkeypatch)
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path)
    signer = worker_signer_from_key(WORKER_KEY)
    with TestClient(create_app(settings)) as client:
        sid = _seed_client(client)
        manifest = _manifest()
        proof = build_execution_proof(
            signer=signer,
            manifest_sha256=compute_manifest_sha256(manifest),
            unit_id=sid,
            image_digest=DIGEST,
            constation_digest=DIGEST,
        ).model_dump(mode="json")
        proof["worker_signature"] = {
            **proof["worker_signature"],
            "sig": "0xdeadbeef",
        }
        resp = _http_post_result(
            client,
            sid=sid,
            slug=settings.slug,
            result={
                "executed": 1,
                PROOF_PAYLOAD_KEY: proof,
                MANIFEST_PAYLOAD_KEY: manifest,
            },
            proof=proof,
        )
        _assert_not_missing_bundle_bypass(resp, expected_code="signature_invalid")
        assert _score_row(tmp_path / "coord.sqlite3", sid) is None


def test_http_flag_on_still_rejects_proof_bad_manifest_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H5 bad manifest hash shape → result_envelope_invalid via HTTP."""
    _enable_unattested(monkeypatch)
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path)
    signer = worker_signer_from_key(WORKER_KEY)
    with TestClient(create_app(settings)) as client:
        sid = _seed_client(client)
        manifest = _manifest()
        proof = build_execution_proof(
            signer=signer,
            manifest_sha256=compute_manifest_sha256(manifest),
            unit_id=sid,
            image_digest=DIGEST,
            constation_digest=DIGEST,
        ).model_dump(mode="json")
        proof["manifest_sha256"] = "gg" * 32  # non-hex
        resp = _http_post_result(
            client,
            sid=sid,
            slug=settings.slug,
            result={
                "executed": 1,
                PROOF_PAYLOAD_KEY: proof,
                MANIFEST_PAYLOAD_KEY: manifest,
            },
            proof=proof,
        )
        # SDK ExecutionProof.manifest_sha256 pattern — envelope rejects before ingest.
        _assert_not_missing_bundle_bypass(resp, expected_code="result_envelope_invalid")
        assert _score_row(tmp_path / "coord.sqlite3", sid) is None


def test_http_flag_on_still_rejects_challenge_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H6 result_challenge_mismatch via HTTP — binding still enforced."""
    _enable_unattested(monkeypatch)
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path)
    signer = worker_signer_from_key(WORKER_KEY)
    with TestClient(create_app(settings)) as client:
        sid = _seed_client(client)
        manifest = _manifest()
        proof = build_execution_proof(
            signer=signer,
            manifest_sha256=compute_manifest_sha256(manifest),
            unit_id=sid,
            image_digest=DIGEST,
            constation_digest=DIGEST,
        ).model_dump(mode="json")
        resp = _http_post_result(
            client,
            sid=sid,
            slug="not-the-prism-slug",
            result={
                "executed": 1,
                PROOF_PAYLOAD_KEY: proof,
                MANIFEST_PAYLOAD_KEY: manifest,
            },
            proof=proof,
        )
        _assert_not_missing_bundle_bypass(resp, expected_code="result_challenge_mismatch")
        assert _score_row(tmp_path / "coord.sqlite3", sid) is None


def test_matrix_documents_enforced_validations() -> None:
    """Structural lock: T22 covers ≥4 distinct non-missing-bundle validations."""
    enforced = frozenset(
        {
            "proof_missing",
            "proof_bad_version",
            "proof_bad_manifest_hash",
            "proof_missing_signature",
            "manifest_tampered",
            "signature_invalid",
            "result_malformed",
            "result_envelope_invalid",
            "result_challenge_mismatch",
            "infra_fault",
            "miner_fault:unknown_digest",
        }
    )
    assert MISSING not in enforced
    assert miner_fault_reason("missing_constation_bundle") == MISSING
    assert len(enforced) >= 4
