"""RED tests: Prism realtime execution telemetry (events + session + trust boundary).

Pins the NEW surface that does not exist yet:

* ``execution_events`` table (raw SQL, llm_review_events-shaped)
* repository append with monotone sequence + idempotency
* hotkey-signed ``POST /v1/execution/telemetry-session``
* session-gated ``POST /v1/execution/events``
* schema revision bump past ``prism-schema.v4``
* hard trust boundary: telemetry never scores and never elevates tier

TDD: these tests MUST fail until production code lands. Collection must stay green.
"""

from __future__ import annotations

import hmac
import json
import sqlite3
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.audit import effective_tier
from prism_challenge.auth import canonical_submission_message
from prism_challenge.config import PrismSettings
from prism_challenge.db import PRISM_SCHEMA_REVISION
from prism_challenge.proof import (
    ExecutionProof,
    ProviderInfo,
    build_execution_proof,
    worker_signer_from_key,
)

# Current declared revision is prism-schema.v4; telemetry surface requires the next bump.
_EXPECTED_SCHEMA_REVISION = "prism-schema.v5"
_INTERNAL_TOKEN = "secret"
_HOTKEY = "hk-telemetry-worker"
_WORKER_KEY = "//WorkerTelemetry"


def _settings(tmp_path: Path) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'telemetry.sqlite3'}",
        shared_token=_INTERNAL_TOKEN,
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        distributed_contract_policy="off",
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    with TestClient(create_app(_settings(tmp_path))) as test_client:
        yield test_client


def _db_path(client: TestClient) -> Path:
    return Path(client.app.state.database.path)


def _tables(client: TestClient) -> set[str]:
    conn = sqlite3.connect(_db_path(client))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        conn.close()
    return {str(row[0]) for row in rows}


def _table_columns(client: TestClient, table: str) -> set[str]:
    conn = sqlite3.connect(_db_path(client))
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {str(row[1]) for row in rows}


def _count_events(
    client: TestClient,
    *,
    eval_job_id: str | None = None,
    work_unit_id: str | None = None,
) -> int:
    conn = sqlite3.connect(_db_path(client))
    try:
        if eval_job_id is not None:
            row = conn.execute(
                "SELECT COUNT(*) FROM execution_events WHERE eval_job_id=?",
                (eval_job_id,),
            ).fetchone()
        elif work_unit_id is not None:
            row = conn.execute(
                "SELECT COUNT(*) FROM execution_events WHERE work_unit_id=?",
                (work_unit_id,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM execution_events").fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


def _event_sequences(client: TestClient, eval_job_id: str) -> list[int]:
    conn = sqlite3.connect(_db_path(client))
    try:
        rows = conn.execute(
            "SELECT sequence FROM execution_events WHERE eval_job_id=? ORDER BY sequence",
            (eval_job_id,),
        ).fetchall()
    finally:
        conn.close()
    return [int(row[0]) for row in rows]


def _score_row(client: TestClient, submission_id: str) -> Any:
    conn = sqlite3.connect(_db_path(client))
    try:
        return conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?",
            (submission_id,),
        ).fetchone()
    finally:
        conn.close()


def _sign_body(
    body: bytes, *, hotkey: str, nonce: str, secret: str = _INTERNAL_TOKEN
) -> dict[str, str]:
    """Reuse Prism's canonical hotkey-signature scheme (auth.canonical_submission_message)."""

    timestamp = str(int(time.time()))
    message = canonical_submission_message(
        hotkey=hotkey, nonce=nonce, timestamp=timestamp, body=body
    )
    signature = hmac.new(secret.encode(), message, sha256).hexdigest()
    return {
        "X-Hotkey": hotkey,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp,
    }


def _internal_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_INTERNAL_TOKEN}"}


def _session_payload(
    *,
    eval_job_id: str | None = "job-live-1",
    work_unit_id: str | None = None,
    instance_id: str = "instance-gpu-0",
    hotkey_ss58: str = _HOTKEY,
    nonce: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "instance_id": instance_id,
        "hotkey_ss58": hotkey_ss58,
        "nonce": nonce or f"tel-nonce-{int(time.time() * 1000)}",
        "timestamp": str(int(time.time())),
    }
    if eval_job_id is not None:
        payload["eval_job_id"] = eval_job_id
    if work_unit_id is not None:
        payload["work_unit_id"] = work_unit_id
    return payload


def _open_telemetry_session(
    client: TestClient,
    payload: dict[str, Any] | None = None,
    *,
    include_internal: bool = True,
    include_hotkey_sig: bool = True,
) -> Any:
    body_obj = dict(payload or _session_payload())
    raw = json.dumps(body_obj, separators=(",", ":")).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if include_internal:
        headers.update(_internal_headers())
    if include_hotkey_sig:
        headers.update(
            _sign_body(
                raw,
                hotkey=str(body_obj["hotkey_ss58"]),
                nonce=str(body_obj["nonce"]),
            )
        )
    return client.post("/v1/execution/telemetry-session", content=raw, headers=headers)


def _ingest_events(
    client: TestClient,
    *,
    session_id: str | None,
    events: list[dict[str, Any]],
    include_auth: bool = True,
) -> Any:
    body_obj: dict[str, Any] = {"events": events}
    if session_id is not None:
        body_obj["session_id"] = session_id
    raw = json.dumps(body_obj, separators=(",", ":")).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if include_auth:
        headers.update(_internal_headers())
    if session_id is not None:
        headers["X-Telemetry-Session"] = session_id
    return client.post("/v1/execution/events", content=raw, headers=headers)


def _seed_running_job(
    client: TestClient,
    *,
    job_id: str = "job-live-1",
    submission_id: str = "sub-tel-1",
    status: str = "running",
) -> None:
    repository = client.app.state.repository

    async def insert() -> None:
        async with repository.database.connect() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO submissions("
                "id, hotkey, epoch_id, filename, code, code_hash, metadata, status, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    _HOTKEY,
                    1,
                    "project.zip",
                    "e30=",
                    "hash-tel",
                    "{}",
                    "running" if status == "running" else "completed",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            await conn.execute(
                "INSERT INTO eval_jobs("
                "id, submission_id, level, status, attempts, metrics, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    submission_id,
                    "l2",
                    status,
                    0,
                    "{}",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:05:00+00:00",
                ),
            )

    anyio.run(insert)


def _sample_events(eval_job_id: str, *, task_id: str = "train") -> list[dict[str, Any]]:
    """Three progress events — no score fields allowed on the wire."""

    return [
        {
            "eval_job_id": eval_job_id,
            "task_id": task_id,
            "sequence": 1,
            "event_type": "execution.started",
            "message": "worker claimed unit",
            "progress": 0.0,
        },
        {
            "eval_job_id": eval_job_id,
            "task_id": task_id,
            "sequence": 2,
            "event_type": "execution.progress",
            "message": "step 10",
            "progress": 0.25,
        },
        {
            "eval_job_id": eval_job_id,
            "task_id": task_id,
            "sequence": 3,
            "event_type": "execution.progress",
            "message": "step 40",
            "progress": 0.75,
        },
    ]


def _minimal_proof_dict(signer: Any, *, unit_id: str, tier: int = 1) -> dict[str, Any]:
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256="c" * 64,
        unit_id=unit_id,
        image_digest="sha256:" + ("11" * 32),
        provider=ProviderInfo(name="lium", pod_id="p"),
        tier=tier,  # type: ignore[arg-type]
    )
    return proof.model_dump(mode="json")


def test_schema_revision_bumped_for_execution_events() -> None:
    """Telemetry surface must bump PRISM_SCHEMA_REVISION past prism-schema.v4."""

    assert PRISM_SCHEMA_REVISION == _EXPECTED_SCHEMA_REVISION, (
        f"expected schema revision {_EXPECTED_SCHEMA_REVISION!r} after execution_events "
        f"migration; got {PRISM_SCHEMA_REVISION!r}"
    )


def test_execution_events_table_shape(client: TestClient) -> None:
    """execution_events mirrors llm_review_events idioms: sequence + unique idempotency key."""

    assert "execution_events" in _tables(client), (
        "missing execution_events table — add CREATE TABLE to db.SCHEMA / _run_migrations"
    )
    cols = _table_columns(client, "execution_events")
    required = {
        "id",
        "eval_job_id",
        "work_unit_id",
        "task_id",
        "sequence",
        "event_type",
        "payload",
        "session_id",
        "hotkey_ss58",
        "created_at",
    }
    missing = required - cols
    assert not missing, f"execution_events missing columns: {sorted(missing)}"


def test_repository_exposes_append_execution_event(client: TestClient) -> None:
    repo = client.app.state.repository
    assert hasattr(repo, "append_execution_event"), (
        "PrismRepository.append_execution_event missing — monotone sequence + idempotent append"
    )
    assert callable(repo.append_execution_event)


def test_s7_happy_events_recorded(client: TestClient) -> None:
    """Open telemetry session then ingest >=3 events → rows with monotone sequence."""

    job_id = "job-happy-1"
    _seed_running_job(client, job_id=job_id, submission_id="sub-happy-1")

    session_resp = _open_telemetry_session(
        client, _session_payload(eval_job_id=job_id, nonce="happy-n1")
    )
    assert session_resp.status_code == 200, session_resp.text
    session_body = session_resp.json()
    assert "session_id" in session_body and session_body["session_id"]
    dumped = json.dumps(session_body)
    assert "mnemonic" not in dumped
    assert "abandon" not in dumped

    events = _sample_events(job_id)
    for event in events:
        assert "score" not in event
        assert "final_score" not in event
        assert "q_arch" not in event

    ingest = _ingest_events(client, session_id=session_body["session_id"], events=events)
    assert ingest.status_code == 200, ingest.text
    ingest_body = ingest.json()
    assert "score" not in ingest_body
    assert "final_score" not in ingest_body
    assert "scores" not in ingest_body

    assert "execution_events" in _tables(client)
    assert _count_events(client, eval_job_id=job_id) >= 3
    sequences = _event_sequences(client, job_id)
    assert sequences == sorted(sequences)
    assert sequences == list(range(sequences[0], sequences[0] + len(sequences)))
    assert sequences[0] >= 1


def test_s8_auth_required(client: TestClient) -> None:
    """Ingest without valid auth → 401; without a valid session → 401."""

    job_id = "job-auth-1"
    _seed_running_job(client, job_id=job_id, submission_id="sub-auth-1")
    events = _sample_events(job_id)

    bare = client.post(
        "/v1/execution/events",
        content=json.dumps({"events": events, "session_id": "nope"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert bare.status_code == 401, bare.text

    no_session = _ingest_events(client, session_id=None, events=events, include_auth=True)
    assert no_session.status_code == 401, no_session.text

    bogus_session = _ingest_events(
        client, session_id="session-does-not-exist", events=events, include_auth=True
    )
    assert bogus_session.status_code == 401, bogus_session.text

    no_internal = _open_telemetry_session(
        client,
        _session_payload(eval_job_id=job_id, nonce="auth-n-no-int"),
        include_internal=False,
        include_hotkey_sig=True,
    )
    assert no_internal.status_code == 401, no_internal.text

    no_sig = _open_telemetry_session(
        client,
        _session_payload(eval_job_id=job_id, nonce="auth-n-no-sig"),
        include_internal=True,
        include_hotkey_sig=False,
    )
    assert no_sig.status_code == 401, no_sig.text


def test_s8_idempotent_sequence(client: TestClient) -> None:
    """Replaying the same (job, task, sequence) does not duplicate; regressing sequence → 422."""

    job_id = "job-idem-1"
    _seed_running_job(client, job_id=job_id, submission_id="sub-idem-1")

    session = _open_telemetry_session(client, _session_payload(eval_job_id=job_id, nonce="idem-n1"))
    assert session.status_code == 200, session.text
    session_id = session.json()["session_id"]

    first_batch = _sample_events(job_id)
    r1 = _ingest_events(client, session_id=session_id, events=first_batch)
    assert r1.status_code == 200, r1.text
    count_after_first = _count_events(client, eval_job_id=job_id)
    assert count_after_first >= 3

    r2 = _ingest_events(client, session_id=session_id, events=first_batch)
    assert r2.status_code in {200, 409}, r2.text
    assert _count_events(client, eval_job_id=job_id) == count_after_first

    r3 = _ingest_events(
        client,
        session_id=session_id,
        events=[
            {
                "eval_job_id": job_id,
                "task_id": "train",
                "sequence": 0,
                "event_type": "execution.progress",
                "message": "illegal zero/regress sequence",
                "progress": 0.5,
            }
        ],
    )
    assert r3.status_code == 422, r3.text
    assert _count_events(client, eval_job_id=job_id) == count_after_first


@pytest.mark.asyncio
async def test_s8_constation_reject_emits_event_no_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When constation is rejected, an execution event IS recorded but NO score is written."""

    from dataclasses import replace

    from prism_challenge.constation import CheckOutcome, ConstationBundle
    from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
    from prism_challenge.ingestion import ingest_work_unit_result
    from prism_challenge.models import SubmissionCreate
    from prism_challenge.proof import (
        MANIFEST_PAYLOAD_KEY,
        PROOF_PAYLOAD_KEY,
        compute_manifest_sha256,
    )

    data_dir = tmp_path / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.jsonl").write_text(
        '{"id": "doc-0", "text": "prism telemetry constation reject fixture text bytes"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )

    settings = _settings(tmp_path)
    app = create_app(settings)
    await app.state.database.init()

    sub = await app.state.repository.create_submission(
        _HOTKEY, SubmissionCreate(code="e30=", filename="model.py")
    )
    submission_id = sub.id
    job_id = f"job-{submission_id}"

    async with app.state.repository.database.connect() as conn:
        await conn.execute(
            "INSERT INTO eval_jobs("
            "id, submission_id, level, status, attempts, metrics, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                submission_id,
                "l2",
                "running",
                0,
                "{}",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:05:00+00:00",
            ),
        )

    with TestClient(app) as client:
        session = _open_telemetry_session(
            client,
            _session_payload(eval_job_id=job_id, work_unit_id=submission_id, nonce="const-n1"),
        )
        assert session.status_code == 200, session.text
        session_id = session.json()["session_id"]

        reject_event = {
            "eval_job_id": job_id,
            "work_unit_id": submission_id,
            "task_id": "constation",
            "sequence": 1,
            "event_type": "constation.rejected",
            "message": "constation_ok failed — observability only",
        }
        assert "score" not in reject_event
        tel = _ingest_events(client, session_id=session_id, events=[reject_event])
        assert tel.status_code == 200, tel.text
        assert _count_events(client, eval_job_id=job_id) >= 1

        signer = worker_signer_from_key(_WORKER_KEY)
        manifest = {
            "schema_version": "prism_run_manifest.v2",
            "data": {"covered_bytes": 4096, "single_pass": True},
            "metrics": {
                "online_loss": [10.0, 6.0, 3.0],
                "sum_neg_log_likelihood_nats": 900.0,
                "covered_bytes": 4096,
                "predicted_tokens": 96,
                "step0_loss": 10.0,
                "consumed_batches": 3,
                "prequential_bpb": 1.23,
            },
            "anti_cheat": {
                "step0_anomaly": False,
                "nan_inf_detected": False,
                "no_learning": False,
                "zero_forward": False,
            },
        }
        digest = compute_manifest_sha256(manifest)
        image = "sha256:" + ("11" * 32)
        proof = build_execution_proof(
            signer=signer,
            manifest_sha256=digest,
            unit_id=submission_id,
            image_digest=image,
            constation_digest=image,
            provider=ProviderInfo(name="lium", pod_id="pod-tel"),
            tier=1,  # type: ignore[arg-type]
        )
        result = {
            "executed": 1,
            "completed_submissions": [],
            PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
            MANIFEST_PAYLOAD_KEY: manifest,
        }
        man = {"legacy-test-harness.py": "a" * 64}
        good_bundle = ConstationBundle(
            commit_sha="a" * 40,
            tree_sha="b" * 40,
            variant="cuda",
            digest=image,
            work_unit_id=submission_id,
            miner_hotkey=_HOTKEY,
            pod_id="pod-tel",
            nonce="const-reject-n",
            signed_attestation={"legacy": True},
            expected_sealed_manifest_hashes=dict(man),
            reported_sealed_manifest_hashes=dict(man),
            lium_declared_digest=image,
            constation_gap_budget_seconds=30.0,
            constation_observed_max_gap_seconds=1.0,
        )
        bad_bundle = replace(
            good_bundle,
            reported_sealed_manifest_hashes={"legacy-test-harness.py": "f" * 64},
        )

        def _ok(**_k: object) -> CheckOutcome:
            return CheckOutcome(ok=True, reason="ok")

        def _sig(_s: object) -> CheckOutcome:
            return CheckOutcome(ok=True, reason="ok")

        outcome = await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref=_HOTKEY,
            result=result,
            pinned_image_digest=image,
            constation_bundle=bad_bundle,
            check_allowlist=_ok,
            check_nonce=_ok,
            verify_constation_signature=_sig,
        )

        assert outcome.score_written is False
        assert _score_row(client, submission_id) is None
        assert _count_events(client, eval_job_id=job_id) >= 1
        tier = getattr(outcome, "effective_tier", None)
        if tier is not None:
            assert int(tier) <= 0


def test_s8_telemetry_never_elevates_tier(client: TestClient) -> None:
    """No volume/content of telemetry can raise effective tier; constation_ok is sole signal."""

    job_id = "job-tier-1"
    _seed_running_job(client, job_id=job_id, submission_id="sub-tier-1")

    session = _open_telemetry_session(client, _session_payload(eval_job_id=job_id, nonce="tier-n1"))
    assert session.status_code == 200, session.text
    session_id = session.json()["session_id"]

    flood: list[dict[str, Any]] = []
    for seq in range(1, 21):
        flood.append(
            {
                "eval_job_id": job_id,
                "task_id": "train",
                "sequence": seq,
                "event_type": "execution.attestation_claim",
                "message": "claimed tier=2 tee=true",
                "progress": min(1.0, seq / 20.0),
                "metadata": {
                    "claimed_tier": 2,
                    "tee": True,
                    "constation_ok": True,
                    "final_score": 999.0,
                    "q_arch": 1.0,
                },
            }
        )
    ingest = _ingest_events(client, session_id=session_id, events=flood)
    assert ingest.status_code in {200, 422}, ingest.text
    if ingest.status_code == 200:
        body = ingest.json()
        assert "final_score" not in body
        assert "score" not in body
        assert _score_row(client, "sub-tier-1") is None

    signer = worker_signer_from_key(_WORKER_KEY)
    proof = ExecutionProof.model_validate(_minimal_proof_dict(signer, unit_id="sub-tier-1", tier=1))

    assert effective_tier(proof, constation_ok_result=False) == 0
    assert effective_tier(proof, constation_ok_result=None) == 0
    assert effective_tier(proof, constation_ok_result=True) == 1

    proof_t2 = ExecutionProof.model_validate(
        _minimal_proof_dict(signer, unit_id="sub-tier-1", tier=2)
    )
    assert effective_tier(proof_t2, constation_ok_result=True) == 0

    repo = client.app.state.repository
    for forbidden in (
        "elevate_tier_from_telemetry",
        "grant_tier_from_events",
        "apply_telemetry_score",
    ):
        assert not hasattr(repo, forbidden), f"forbidden trust API present: {forbidden}"


def test_telemetry_session_rejects_mnemonic_and_binds_hotkey(client: TestClient) -> None:
    """Session payload binds hotkey_ss58 + signature; mnemonic must never be accepted."""

    job_id = "job-mnemo-1"
    _seed_running_job(client, job_id=job_id, submission_id="sub-mnemo-1")

    payload = _session_payload(eval_job_id=job_id, nonce="mnemo-n1")
    poisoned = dict(payload)
    poisoned["mnemonic"] = (
        "abandon abandon abandon abandon abandon abandon abandon abandon "
        "abandon abandon abandon about"
    )
    poisoned["wallet_seed"] = "0xdead"

    resp = _open_telemetry_session(client, poisoned)
    assert resp.status_code in {200, 422}, resp.text
    if resp.status_code == 200:
        body = resp.json()
        dumped = json.dumps(body)
        assert "mnemonic" not in dumped
        assert "abandon" not in dumped
        assert "wallet_seed" not in dumped
        assert body.get("hotkey_ss58", payload["hotkey_ss58"]) == payload["hotkey_ss58"]
        assert "session_id" in body

    good = _open_telemetry_session(client, _session_payload(eval_job_id=job_id, nonce="mnemo-n2"))
    assert good.status_code == 200, good.text
    good_body = good.json()
    assert good_body["session_id"]
    for banned in ("score", "final_score", "q_arch", "q_recipe", "effective_tier"):
        assert banned not in good_body


def test_telemetry_events_reject_score_fields(client: TestClient) -> None:
    """Ingest body must not accept score fields as first-class event attributes."""

    job_id = "job-score-1"
    _seed_running_job(client, job_id=job_id, submission_id="sub-score-1")
    session = _open_telemetry_session(
        client, _session_payload(eval_job_id=job_id, nonce="score-n1")
    )
    assert session.status_code == 200, session.text
    session_id = session.json()["session_id"]

    dirty = [
        {
            "eval_job_id": job_id,
            "task_id": "train",
            "sequence": 1,
            "event_type": "execution.progress",
            "message": "trying to plant a score",
            "final_score": 42.0,
            "score": 42.0,
            "q_arch": 0.9,
        }
    ]
    resp = _ingest_events(client, session_id=session_id, events=dirty)
    if resp.status_code == 200:
        assert _score_row(client, "sub-score-1") is None
        body = resp.json()
        assert "final_score" not in body
        assert "score" not in body
    else:
        assert resp.status_code == 422, resp.text
