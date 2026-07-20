"""Offline cross-component backward-compat + nonce-flow + legacy-surface tests.

Companion to ``tests/test_cross_integration_e2e_offline.py`` (the anti-cheat
rejection-path suite). This module pins the OTHER half of the cross-component
contract WITHOUT a live CVM:

* nonce-flow integrity across the key-release endpoint AND the result verifier
  (issued-here == embedded-in-quote == verified-there; one run bound by one nonce);
* flag-OFF byte-identical legacy behaviour (validator pipeline + job score);
* the low-rate replay audit (match = no-op on weights / mismatch = visible flag,
  never a silent weight mutation);
* the unchanged signed-request intake + replay/skew/cap guards, identical whether
  the Phala flag is ON or OFF;
* deterministic task selection preserved and exactly what ``report_data`` binds;
* the status / SSE surfaces functioning over an attested run while leaking no
  sensitive material; and
* the deploy-but-no-result bound (fold to a single failed result, no weight, no
  hang, and the issued nonce non-redeemable afterward).

Every test is a DISCRIMINATOR: it pairs the invariant with a positive control /
mismatch case so a vacuous implementation would fail it.

Fulfils VAL-CROSS-017, 018, 019, 020, 022, 028, 029, 030, 031 (challenge side).
The base leg of the flag-OFF invariant (VAL-CROSS-021) lives in the base repo
(``tests/unit/test_cross_integration_carry_chain.py``).
"""

from __future__ import annotations

import base64
import io
import json
import uuid
import zipfile
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.canonical import report_data as rd
from agent_challenge.canonical.attested_result import (
    ATTESTATION_BINDING_RESULT_KEY,
    EXECUTION_PROOF_RESULT_KEY,
    build_attestation_binding,
    build_execution_proof_envelope,
    build_measurement,
    build_phala_attestation,
)
from agent_challenge.canonical.measurement import CanonicalMeasurement
from agent_challenge.core.config import settings as core_settings
from agent_challenge.evaluation.attestation import (
    AttestationGate,
    AttestationOutcome,
    InMemoryNonceLedger,
    ResultMeasurementAllowlist,
    extract_attestation_envelope,
)
from agent_challenge.evaluation.benchmarks import (
    BenchmarkTask,
    benchmark_tasks_to_json,
    select_benchmark_tasks,
)
from agent_challenge.evaluation.own_runner.keep_policy import keep_good_job_score
from agent_challenge.evaluation.own_runner.result_schema import (
    build_benchmark_result,
    format_benchmark_result_line,
)
from agent_challenge.evaluation.replay_audit import (
    AggregationSpec,
    AuditCandidate,
    audit_submission,
)
from agent_challenge.evaluation.validator_executor import (
    finalize_job_if_complete,
    fold_terminally_failed_work_unit,
    get_task_attestation,
    run_validator_cycle,
)
from agent_challenge.evaluation.weights import get_weights, is_reward_eligible_job
from agent_challenge.keyrelease.allowlist import CanonicalEntry, MeasurementAllowlist
from agent_challenge.keyrelease.client import key_release_report_data
from agent_challenge.keyrelease.nonce import NonceState, NonceStore
from agent_challenge.keyrelease.quote import (
    COMPOSE_HASH_EVENT,
    KEY_PROVIDER_EVENT,
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
    os_image_hash_from_registers,
)
from agent_challenge.keyrelease.server import KeyReleaseService
from agent_challenge.models import AgentSubmission, EvaluationJob, TaskResult
from agent_challenge.sdk.config import effective_evaluation_task_count
from agent_challenge.sdk.executors import DockerRunResult
from agent_challenge.security import build_signed_auth_dependency
from agent_challenge.submissions.state_machine import transition_submission_status

# --------------------------------------------------------------------------- #
# One coherent canonical image: the same measurement pins the key-release
# allowlist (7 registers incl. key_provider) AND the result verifier allowlist.
# --------------------------------------------------------------------------- #
REGS = {"mrtd": "11" * 48, "rtmr0": "22" * 48, "rtmr1": "33" * 48, "rtmr2": "44" * 48}
COMPOSE_PAYLOAD = bytes.fromhex("ab" * 32)
KEY_PROVIDER_PAYLOAD = b'{"name":"kms","id":"kms-1"}'
ENCLAVE_PUBKEY = b"enclave-ra-tls-pubkey-0123456789"  # 32 bytes
SENTINEL_KEY = b"SENTINEL-CROSS-INTEGRATION-KEY!!"  # 32 bytes


def _canonical_measurement() -> dict[str, str]:
    return {
        **REGS,
        "compose_hash": COMPOSE_PAYLOAD.hex(),
        "os_image_hash": os_image_hash_from_registers(REGS["mrtd"], REGS["rtmr1"], REGS["rtmr2"]),
    }


def _event_log() -> tuple[list[dict], str]:
    return build_rtmr3_event_log(
        [
            ("app-id", b"canonical-app"),
            (COMPOSE_HASH_EVENT, COMPOSE_PAYLOAD),
            (KEY_PROVIDER_EVENT, KEY_PROVIDER_PAYLOAD),
            ("instance-id", b"instance-xyz"),
        ]
    )


def _attested_line(task_id: str, *, agent_hash: str, nonce: str, score: float = 1.0) -> str:
    """A ``BASE_BENCHMARK_RESULT=`` line carrying a self-consistent Phala envelope."""

    scores = {task_id: score}
    task_ids = [task_id]
    canonical = CanonicalMeasurement(**_canonical_measurement())
    _event, rtmr3 = _event_log()
    digest = rd.scores_digest(scores)
    report_data_hex = rd.report_data_hex(
        canonical_measurement=canonical,
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores_digest=digest,
        validator_nonce=nonce,
    )
    quote = build_tdx_quote(
        mrtd=REGS["mrtd"],
        rtmr0=REGS["rtmr0"],
        rtmr1=REGS["rtmr1"],
        rtmr2=REGS["rtmr2"],
        rtmr3=rtmr3,
        report_data=report_data_hex,
    )
    attestation = build_phala_attestation(
        tdx_quote=quote,
        event_log=_event_log()[0],
        report_data_hex=report_data_hex,
        measurement=build_measurement(canonical, rtmr3=rtmr3),
        vm_config={},
    )
    envelope = build_execution_proof_envelope(manifest_sha256="ab" * 32, attestation=attestation)
    binding = build_attestation_binding(
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores=scores,
        scores_digest=digest,
        validator_nonce=nonce,
        canonical_measurement=canonical,
    )
    result = build_benchmark_result(
        status="completed" if score >= 1.0 else "failed",
        score=score,
        resolved=round(score),
        total=1,
        reason_code=None,
    )
    result[EXECUTION_PROOF_RESULT_KEY] = envelope
    result[ATTESTATION_BINDING_RESULT_KEY] = binding
    return format_benchmark_result_line(result)


def _plain_line(score: float = 1.0) -> str:
    status = "completed" if score >= 1.0 else "failed"
    payload = {"status": status, "score": score, "resolved": round(score), "total": 1}
    return "BASE_BENCHMARK_RESULT=" + json.dumps(payload, sort_keys=True)


def _make_gate(*, nonces: list[str]) -> AttestationGate:
    ledger = InMemoryNonceLedger()
    for nonce in nonces:
        ledger.issue(nonce)
    return AttestationGate(
        quote_verifier=StaticQuoteVerifier(valid=True),
        allowlist=ResultMeasurementAllowlist.from_measurements([_canonical_measurement()]),
        nonce_validator=ledger,
    )


# --------------------------------------------------------------------------- #
# Key-release endpoint helpers (validator-operated, sentinel golden key).
# --------------------------------------------------------------------------- #
def _canonical_entry() -> CanonicalEntry:
    return CanonicalEntry(
        mrtd=REGS["mrtd"],
        rtmr0=REGS["rtmr0"],
        rtmr1=REGS["rtmr1"],
        rtmr2=REGS["rtmr2"],
        compose_hash=COMPOSE_PAYLOAD.hex(),
        os_image_hash=os_image_hash_from_registers(REGS["mrtd"], REGS["rtmr1"], REGS["rtmr2"]),
        # decode_key_provider collapses live KMS/phala JSON payloads to the pin id "phala"
        key_provider="phala",
    )


def _make_key_release_service() -> KeyReleaseService:
    return KeyReleaseService(
        allowlist=MeasurementAllowlist([_canonical_entry()]),
        verifier=StaticQuoteVerifier(tcb_status="UpToDate"),
        nonce_store=NonceStore(),
        golden_key_loader=lambda: SENTINEL_KEY,
    )


def _key_release_request(service: KeyReleaseService, *, nonce: str | None = None) -> dict:
    if nonce is None:
        nonce = service.issue_nonce()
    event_log, rtmr3 = _event_log()
    quote = build_tdx_quote(
        mrtd=REGS["mrtd"],
        rtmr0=REGS["rtmr0"],
        rtmr1=REGS["rtmr1"],
        rtmr2=REGS["rtmr2"],
        rtmr3=rtmr3,
        report_data=key_release_report_data(nonce, ENCLAVE_PUBKEY),
    )
    return {
        "nonce": nonce,
        "quote_hex": quote,
        "ra_tls_pubkey_hex": ENCLAVE_PUBKEY.hex(),
        "event_log": event_log,
        "session_peer_pubkey": ENCLAVE_PUBKEY,
    }


class _AdvanceableClock:
    """A monotonic-style clock whose time only moves when the test advances it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# --------------------------------------------------------------------------- #
# DB pipeline helpers (validator cycle / finalize / weights).
# --------------------------------------------------------------------------- #
def _configure_runner_broker(monkeypatch, tmp_path, *, task_count: int | None = None) -> None:
    base = "agent_challenge.evaluation.runner.settings"
    monkeypatch.setattr(f"{base}.benchmark_backend", "terminal_bench")
    monkeypatch.setattr(f"{base}.terminal_bench_execution_backend", "own_runner")
    monkeypatch.setattr(f"{base}.evaluation_concurrency", 1)
    if task_count is not None:
        monkeypatch.setattr(f"{base}.evaluation_task_count", task_count)
    monkeypatch.setattr(f"{base}.docker_enabled", True)
    monkeypatch.setattr(f"{base}.docker_backend", "broker")
    monkeypatch.setattr(f"{base}.docker_broker_url", "https://broker.test")
    monkeypatch.setattr(f"{base}.docker_broker_token", "broker-token")
    monkeypatch.setattr(f"{base}.docker_broker_token_file", None)
    harbor = tmp_path / "harbor-runs"
    harbor.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(f"{base}.harbor_output_dir", str(harbor))


def _enable_phala(monkeypatch, enabled: bool) -> None:
    monkeypatch.setattr("agent_challenge.core.config.settings.phala_attestation_enabled", enabled)


def _terminal_bench_tasks(count: int) -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id=f"terminal-bench/task-{index}",
            docker_image=f"ghcr.io/baseintelligence/terminal-bench-runner:{index}",
            prompt=f"task {index}",
            benchmark="terminal_bench",
            metadata={"task_id": f"terminal-bench/task-{index}"},
        )
        for index in range(count)
    ]


async def _create_job(
    session, *, agent_hash, tasks, tmp_path, miner_hotkey=None
) -> tuple[AgentSubmission, EvaluationJob]:
    agent_dir = tmp_path / agent_hash
    agent_dir.mkdir(parents=True, exist_ok=True)
    submission = AgentSubmission(
        miner_hotkey=miner_hotkey or f"hotkey-{agent_hash}",
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=str(agent_dir),
        status="evaluation queued",
        raw_status="tb_queued",
        effective_status="evaluation queued",
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=uuid.uuid4().hex,
        submission_id=submission.id,
        status="queued",
        selected_tasks_json=benchmark_tasks_to_json(tasks),
        total_tasks=len(tasks),
        verdict="valid",
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    return submission, job


class _RecordingBroker:
    """A validator-own broker returning a preset stdout line per task (+ counts runs)."""

    def __init__(self, lines: dict[str, str]) -> None:
        self.runs: list[str] = []
        self.lines = lines

    def run(self, spec, timeout_seconds: int):
        task_id = spec.labels["base.task"]
        self.runs.append(task_id)
        return DockerRunResult(
            container_name="broker-fake",
            stdout=self.lines.get(task_id, _plain_line()),
            stderr="",
            returncode=0,
        )


# =========================================================================== #
# Nonce-flow integrity (VAL-CROSS-017, 018)
# =========================================================================== #
def test_val_cross_017_nonce_issued_here_equals_verified_there() -> None:
    # issued-here: the validator key-release endpoint mints ONE nonce and, for a
    # quote binding that exact nonce, releases the golden key.
    service = _make_key_release_service()
    nonce = service.issue_nonce()
    released = service.authorize_release(**_key_release_request(service, nonce=nonce))
    assert released.released is True
    assert released.key == SENTINEL_KEY

    # embedded-in-quote == verified-there: the result verifier, holding the SAME
    # issued nonce, accepts a result quote whose report_data binds it, and the
    # nonce string is byte-identical end to end.
    line = _attested_line("terminal-bench/task-0", agent_hash="agent-n", nonce=nonce)
    decision = _make_gate(nonces=[nonce]).decide(line, expected_agent_hash="agent-n")
    assert decision.outcome is AttestationOutcome.VERIFIED
    envelope = extract_attestation_envelope(line)
    assert envelope is not None
    assert envelope[1]["validator_nonce"] == nonce

    # A nonce the validator NEVER issued is refused at BOTH surfaces.
    forged = "forged-nonce-never-issued"
    fresh_service = _make_key_release_service()
    denied = fresh_service.authorize_release(**_key_release_request(fresh_service, nonce=forged))
    assert denied.released is False
    assert denied.key is None
    forged_line = _attested_line("terminal-bench/task-0", agent_hash="agent-n", nonce=forged)
    forged_decision = _make_gate(nonces=[nonce]).decide(forged_line, expected_agent_hash="agent-n")
    assert forged_decision.outcome is AttestationOutcome.VERIFICATION_FAILED


def test_val_cross_018_one_run_is_bound_by_one_nonce() -> None:
    service = _make_key_release_service()
    run_nonce = service.issue_nonce()
    # The golden key is obtained under run_nonce.
    out = service.authorize_release(**_key_release_request(service, nonce=run_nonce))
    assert out.released is True
    assert out.key == SENTINEL_KEY

    # The result quote for the SAME run binds the SAME nonce -> accepted.
    same = _attested_line("terminal-bench/task-0", agent_hash="agent-r", nonce=run_nonce)
    assert (
        _make_gate(nonces=[run_nonce]).decide(same, expected_agent_hash="agent-r").outcome
        is AttestationOutcome.VERIFIED
    )

    # A result quote carrying a DIFFERENT validator nonce than the one the golden
    # key was obtained under cannot be paired with that key: this run expects only
    # run_nonce, so a result under another nonce is rejected.
    other_nonce = service.issue_nonce()
    different = _attested_line("terminal-bench/task-0", agent_hash="agent-r", nonce=other_nonce)
    assert (
        _make_gate(nonces=[run_nonce]).decide(different, expected_agent_hash="agent-r").outcome
        is AttestationOutcome.VERIFICATION_FAILED
    )


# =========================================================================== #
# Backward compatibility, flag OFF (VAL-CROSS-019, 020)
# =========================================================================== #
async def test_val_cross_019_flag_off_byte_identical_legacy_pipeline(
    database_session, monkeypatch, tmp_path
) -> None:
    _configure_runner_broker(monkeypatch, tmp_path, task_count=2)
    _enable_phala(monkeypatch, False)  # flag OFF -> legacy own_runner path
    tasks = _terminal_bench_tasks(2)
    async with database_session() as session:
        _sub, job = await _create_job(
            session, agent_hash="legacy-off", tasks=tasks, tmp_path=tmp_path, miner_hotkey="hk-off"
        )
        await session.commit()
        job_pk, job_id = job.id, job.job_id

    broker = _RecordingBroker({t.task_id: _plain_line() for t in tasks})
    # Flag OFF: NO attestation gate is passed or consulted anywhere.
    cycle = await run_validator_cycle(executor=broker)

    assert job_id in cycle.finalized_jobs
    assert sorted(broker.runs) == sorted(t.task_id for t in tasks)  # R=1: each task once
    async with database_session() as session:
        results = (
            (await session.execute(select(TaskResult).where(TaskResult.job_id == job_pk)))
            .scalars()
            .all()
        )
        job_row = await session.get(EvaluationJob, job_pk)
        attestations = [await get_task_attestation(session, job_pk, t.task_id) for t in tasks]
    assert {r.task_id for r in results} == {t.task_id for t in tasks}
    assert job_row.status == "completed"
    assert job_row.score == 1.0
    # No attestation side effects: not a single TaskAttestation row was written.
    assert all(record is None for record in attestations)
    assert (await get_weights()).get("hk-off") == 1.0  # legacy weight earned

    # Re-running the cycle re-pulls nothing and re-runs nothing (idempotent R=1).
    again = await run_validator_cycle(executor=broker)
    assert again.pulled == 0
    assert again.posted == 0
    assert sorted(broker.runs) == sorted(t.task_id for t in tasks)

    # DISCRIMINATOR (non-vacuity): flag ON with the SAME plain results and no gate
    # fails closed -> the attestation path IS consulted (job parked, no weight),
    # proving the flag-off run above genuinely bypassed attestation.
    _enable_phala(monkeypatch, True)
    tasks_on = _terminal_bench_tasks(2)
    async with database_session() as session:
        _sub2, job2 = await _create_job(
            session, agent_hash="legacy-on", tasks=tasks_on, tmp_path=tmp_path, miner_hotkey="hk-on"
        )
        await session.commit()
        job2_pk = job2.id

    broker2 = _RecordingBroker({t.task_id: _plain_line() for t in tasks_on})
    cycle2 = await run_validator_cycle(executor=broker2)
    assert cycle2.finalized_jobs == ()  # parked, not finalized
    async with database_session() as session:
        posted = await session.scalar(
            select(func.count(TaskResult.id)).where(TaskResult.job_id == job2_pk)
        )
        parked = await get_task_attestation(session, job2_pk, tasks_on[0].task_id)
    assert posted == 0
    assert parked is not None
    assert parked.verified is False
    assert "hk-on" not in (await get_weights())


async def test_val_cross_020_flag_off_k1_variance_off_byte_identical_job_score(
    database_session, monkeypatch, tmp_path
) -> None:
    _configure_runner_broker(monkeypatch, tmp_path, task_count=4)
    _enable_phala(monkeypatch, False)
    monkeypatch.setattr("agent_challenge.core.config.settings.keep_good_tasks_policy", "off")
    tasks = _terminal_bench_tasks(4)
    scores = [1.0, 1.0, 0.0, 0.5]
    async with database_session() as session:
        _sub, job = await _create_job(
            session, agent_hash="score-legacy", tasks=tasks, tmp_path=tmp_path, miner_hotkey="hk-sc"
        )
        for task, score in zip(tasks, scores, strict=True):
            session.add(
                TaskResult(
                    job_id=job.id,
                    task_id=task.task_id,
                    docker_image=task.docker_image,
                    status="completed" if score >= 1.0 else "failed",
                    score=score,
                    returncode=0,
                    duration_seconds=0.0,
                )
            )
        await session.commit()
        job_pk, job_id = job.id, job.job_id

    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()

    # Legacy aggregation: score = sum(task scores)/total; passed = count(score>=1).
    expected_score = sum(scores) / len(scores)
    expected_passed = sum(1 for s in scores if s >= 1.0)
    assert summary is not None
    assert summary.score == expected_score
    assert summary.passed_tasks == expected_passed
    assert summary.total_tasks == len(scores)
    # The "off" keep policy IS exactly the legacy mean; a keep policy would differ.
    assert keep_good_job_score(scores, policy="off") == expected_score
    assert keep_good_job_score(scores, policy="drop-lowest-n", drop_lowest_n=1) != expected_score

    # Weight eligibility matches legacy (full task set + at least one pass).
    required = effective_evaluation_task_count(core_settings.evaluation_task_count)
    async with database_session() as session:
        job_row = await session.get(EvaluationJob, job_pk)
    assert is_reward_eligible_job(job_row, required) is (
        job_row.total_tasks >= required and expected_passed >= 1
    )
    assert is_reward_eligible_job(job_row, required) is True


# =========================================================================== #
# Replay audit, master <-> challenge (VAL-CROSS-022)
# =========================================================================== #
class _ReplayBroker:
    """Legacy own_runner broker replaying a fixed per-task score (records dispatch)."""

    def __init__(self, task_scores: list[float]) -> None:
        self._task_scores = task_scores
        self.calls: list[tuple[str, int]] = []

    def __call__(self, submission_id: str, *, k: int):
        self.calls.append((submission_id, k))
        return {f"task-{i}": [score] * k for i, score in enumerate(self._task_scores)}


async def _add_completed_scoring_job(session, *, hotkey, agent_hash, score, total_tasks) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    submission = AgentSubmission(
        miner_hotkey=hotkey,
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=f"/tmp/{agent_hash}.zip",
        status="tb_completed",
        raw_status="tb_completed",
        effective_status="valid",
        submitted_at=now,
        created_at=now,
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=f"job-{agent_hash}",
        submission_id=submission.id,
        status="completed",
        selected_tasks_json="[]",
        score=score,
        passed_tasks=1,
        total_tasks=total_tasks,
        verdict="valid",
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id


async def test_val_cross_022_replay_audit_match_noop_mismatch_flags_no_weight_mutation(
    database_session,
) -> None:
    spec = AggregationSpec(per_task_aggregation="mean", keep_policy="off")
    required = effective_evaluation_task_count(core_settings.evaluation_task_count)
    async with database_session() as session:
        await _add_completed_scoring_job(
            session, hotkey="hk-1", agent_hash="a1", score=0.9, total_tasks=required
        )
        await session.commit()

    weights_before = await get_weights()
    assert weights_before == {"hk-1": 0.9}  # an accepted, weight-bearing submission

    # (a) MATCH: a replay within tolerance is a NO-OP on weights and raises no flag.
    match = audit_submission(
        AuditCandidate("a1", attested_score=0.9, n_attempts=1),
        _ReplayBroker([0.9, 0.9]),
        spec=spec,
        tolerance=0.2,
    )
    assert match.flagged is False
    assert match.flag is None
    assert await get_weights() == weights_before

    # (b) MISMATCH: a replay beyond tolerance raises a visible flag carrying all
    # four dispute fields and STILL does not silently mutate the accepted weights.
    mismatch = audit_submission(
        AuditCandidate("a1", attested_score=0.9, n_attempts=1),
        _ReplayBroker([0.1, 0.1]),
        spec=spec,
        tolerance=0.2,
    )
    assert mismatch.flagged is True
    assert mismatch.flag is not None
    assert mismatch.flag.submission_id == "a1"
    assert mismatch.flag.attested_score == 0.9
    assert mismatch.flag.delta > 0.2
    weights_after = await get_weights()
    assert weights_after == weights_before
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == "job-a1"))
    assert job.score == 0.9  # accepted score untouched by the audit


# =========================================================================== #
# Unchanged signed intake, flag ON vs OFF (VAL-CROSS-028)
# =========================================================================== #
_NOW_028 = datetime(2026, 5, 22, 12, 0, tzinfo=UTC)


def _fake_verifier(hotkey: str, message: str, signature: str) -> bool:
    return signature == "valid-signature"


def _signed_headers(*, hotkey, nonce, signature="valid-signature", timestamp=None) -> dict:
    return {
        "X-Hotkey": hotkey,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp or _NOW_028.isoformat(),
    }


def _zip_payload(name: str, marker: str) -> dict:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("agent.py", f"class Agent:\n    pass\n# {marker}\n")
    return {
        "name": name,
        "artifact_zip_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


async def _run_intake_scenarios(client) -> dict[str, int]:
    ttl = core_settings.signing_ttl_seconds
    stale_ts = (_NOW_028 - timedelta(seconds=ttl + 1)).isoformat()
    outcomes: dict[str, int] = {}

    valid = await client.post(
        "/submissions",
        json=_zip_payload("valid-agent", "v"),
        headers=_signed_headers(hotkey="hk-valid", nonce="n-valid"),
    )
    outcomes["valid"] = valid.status_code

    # Same hotkey, fresh nonce -> one-per-hotkey-per-3h window rejects.
    over_freq = await client.post(
        "/submissions",
        json=_zip_payload("rate-agent-2", "v2"),
        headers=_signed_headers(hotkey="hk-valid", nonce="n-valid-2"),
    )
    outcomes["over_frequency"] = over_freq.status_code

    bad_sig = await client.post(
        "/submissions",
        json=_zip_payload("badsig-agent", "b"),
        headers=_signed_headers(hotkey="hk-badsig", nonce="n-badsig", signature="bad-signature"),
    )
    outcomes["bad_signature"] = bad_sig.status_code

    stale = await client.post(
        "/submissions",
        json=_zip_payload("stale-agent", "s"),
        headers=_signed_headers(hotkey="hk-stale", nonce="n-stale", timestamp=stale_ts),
    )
    outcomes["stale_timestamp"] = stale.status_code

    replay_first = await client.post(
        "/submissions",
        json=_zip_payload("replay-agent", "r"),
        headers=_signed_headers(hotkey="hk-replay", nonce="n-replay"),
    )
    outcomes["replay_first"] = replay_first.status_code
    replayed = await client.post(
        "/submissions",
        json=_zip_payload("replay-agent-2", "r2"),
        headers=_signed_headers(hotkey="hk-replay", nonce="n-replay"),
    )
    outcomes["replayed_nonce"] = replayed.status_code

    oversized = {
        "name": "oversize-agent",
        "artifact_zip_base64": base64.b64encode(b"0" * 1_048_577).decode("ascii"),
    }
    over_cap = await client.post(
        "/submissions",
        json=oversized,
        headers=_signed_headers(hotkey="hk-oversize", nonce="n-oversize"),
    )
    outcomes["over_cap_zip"] = over_cap.status_code
    return outcomes


@pytest.mark.parametrize("phala_flag", [False, True])
async def test_val_cross_028_signed_intake_identical_flag_on_off(
    client, monkeypatch, tmp_path, phala_flag
) -> None:
    monkeypatch.setattr("agent_challenge.api.routes.settings.artifact_root", str(tmp_path / "art"))
    _enable_phala(monkeypatch, phala_flag)
    # Install the REAL signed-auth dependency (fake verifier + fixed clock) so the
    # sr25519/skew/replay guards are genuinely exercised, not stubbed away.
    app.dependency_overrides[routes.signed_submission_auth] = build_signed_auth_dependency(
        core_settings, verifier=_fake_verifier, now_provider=lambda: _NOW_028
    )
    try:
        outcomes = await _run_intake_scenarios(client)
    finally:
        app.dependency_overrides.pop(routes.signed_submission_auth, None)

    # Both flag states assert the SAME expected accept/reject map, so a passing
    # run under each flag proves the intake contract is byte-for-byte identical.
    assert outcomes == {
        "valid": 201,
        "over_frequency": 429,
        "bad_signature": 401,
        "stale_timestamp": 401,
        "replay_first": 201,
        "replayed_nonce": 409,
        "over_cap_zip": 413,
    }


# =========================================================================== #
# Deterministic task selection == report_data binding (VAL-CROSS-029)
# =========================================================================== #
def test_val_cross_029_task_selection_deterministic_and_report_data_bound(monkeypatch) -> None:
    tasks = _terminal_bench_tasks(20)
    agent_hash = "a" * 64

    # Pure function of the agent hash: identical across repeated calls.
    first = [t.task_id for t in select_benchmark_tasks(tasks, agent_hash=agent_hash, count=5)]
    second = [t.task_id for t in select_benchmark_tasks(tasks, agent_hash=agent_hash, count=5)]
    assert first == second

    # Flag state never changes the selection (selection never reads the flag).
    _enable_phala(monkeypatch, False)
    off_sel = [t.task_id for t in select_benchmark_tasks(tasks, agent_hash=agent_hash, count=5)]
    _enable_phala(monkeypatch, True)
    on_sel = [t.task_id for t in select_benchmark_tasks(tasks, agent_hash=agent_hash, count=5)]
    assert off_sel == on_sel

    # Non-vacuous determinism: a different agent hash selects a different subset.
    other = [t.task_id for t in select_benchmark_tasks(tasks, agent_hash="b" * 64, count=5)]
    assert other != off_sel

    # report_data binds EXACTLY the sorted selected task ids (order-independent),
    # and a different task set yields a different report_data (the set is bound).
    canonical = CanonicalMeasurement(**_canonical_measurement())
    common = {
        "canonical_measurement": canonical,
        "agent_hash": agent_hash,
        "scores_digest": rd.scores_digest({tid: 1.0 for tid in first}),
        "validator_nonce": "nonce-sel",
    }
    rd_sorted = rd.report_data_hex(task_ids=sorted(first), **common)
    rd_shuffled = rd.report_data_hex(task_ids=list(reversed(first)), **common)
    assert rd_sorted == rd_shuffled
    different_ids = sorted(first[:-1] + ["terminal-bench/not-selected"])
    assert rd.report_data_hex(task_ids=different_ids, **common) != rd_sorted


# =========================================================================== #
# Status / SSE surfaces under attestation, no leak (VAL-CROSS-030)
# =========================================================================== #
_SENSITIVE_SENTINELS = (
    "GOLDEN-PLAINTEXT-def-solve",
    "GOLDEN-KEY-SENTINEL",
    "gw-token-SENTINEL",
    "miner-secret-SENTINEL",
    "quote-secret-SENTINEL",
    "class SecretLeakAgent",
    "broker-ref-SENTINEL",
    "/tmp/private-golden.py",
)

# A valid attested-run lifecycle: analysis -> eval, an attestation park manifests
# as a retryable eval failure, retried once, then permanently folded to failed.
_ATTESTED_PROGRESSION = (
    ("received", "api", "received"),
    ("upload_verified", "api", "artifact verified"),
    ("rate_limit_reserved", "api", "rate limit reserved"),
    ("analysis_queued", "analysis", "queued"),
    ("ast_running", "worker", "ast started"),
    ("analysis_allowed", "worker", "analysis allowed"),
    ("waiting_miner_env", "worker", "waiting_miner_env"),
    ("tb_queued", "evaluation", "evaluation queued"),
    ("tb_running", "evaluation", "evaluation_job_running"),
    ("tb_failed_retryable", "evaluation", "evaluation_retry_queued"),
    ("tb_queued", "evaluation", "evaluation queued"),
    ("tb_running", "evaluation", "evaluation_job_running"),
    ("tb_failed_final", "evaluation", "evaluation_retry_cap_reached"),
)


def _parse_sse_events(text: str) -> list[dict]:
    events: list[dict] = []
    for frame in text.strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in frame.splitlines():
            name, value = line.split(": ", 1)
            fields[name] = value
        events.append({"id": int(fields["id"]), "data": json.loads(fields["data"])})
    return events


async def _attested_lifecycle_submission(session, *, agent_hash) -> tuple[int, list[int]]:
    submission = AgentSubmission(
        miner_hotkey=f"miner-{agent_hash}",
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=f"/tmp/{agent_hash}.zip",
        status="received",
        raw_status="received",
        effective_status="received",
    )
    session.add(submission)
    await session.flush()
    leaky_metadata = {
        "private_path": "/tmp/private-golden.py",
        "secret": "GOLDEN-KEY-SENTINEL",
        "golden": "GOLDEN-PLAINTEXT-def-solve",
        "token": "gw-token-SENTINEL",
        "miner_env": "miner-secret-SENTINEL",
        "quote": "quote-secret-SENTINEL",
        "source": "class SecretLeakAgent",
        "broker_ref": "broker-ref-SENTINEL",
    }
    event_ids: list[int] = []
    for index, (to_status, actor, reason) in enumerate(_ATTESTED_PROGRESSION):
        kwargs = {"from_status": None} if index == 0 else {}
        event = await transition_submission_status(
            session,
            submission,
            to_status,
            actor=actor,
            reason=reason,
            metadata=dict(leaky_metadata),
            **kwargs,
        )
        event_ids.append(event.id)
    await session.commit()
    return submission.id, event_ids


async def test_val_cross_030_status_and_sse_functional_and_no_leak(
    client, database_session
) -> None:
    async with database_session() as session:
        submission_id, event_ids = await _attested_lifecycle_submission(
            session, agent_hash="attested-surface"
        )

    status = await client.get(f"/submissions/{submission_id}/status")
    events = await client.get(f"/submissions/{submission_id}/events")
    assert status.status_code == 200
    assert events.status_code == 200

    # The surfaces reflect the attested run's progression (all events, terminal).
    status_payload = status.json()
    assert status_payload["last_event_id"] == event_ids[-1]
    assert status_payload["public_state"] == "error"  # tb_failed_final terminal
    parsed = _parse_sse_events(events.text)
    assert [event["id"] for event in parsed] == event_ids
    assert parsed[-1]["data"]["public_state"] == "error"

    # A scan of BOTH surfaces finds zero sensitive/secret material.
    for blob in (status.text, events.text):
        for sentinel in _SENSITIVE_SENTINELS:
            assert sentinel not in blob

    # Durable Last-Event-ID: a stale/unknown id -> 409 + replay_from.
    stale = await client.get(
        f"/submissions/{submission_id}/events",
        headers={"Last-Event-ID": str(event_ids[0] - 1)},
    )
    assert stale.status_code == 409
    assert stale.json() == {"detail": "unknown Last-Event-ID", "replay_from": event_ids[0]}

    # A known Last-Event-ID resumes strictly after it.
    resume = await client.get(
        f"/submissions/{submission_id}/events",
        headers={"Last-Event-ID": str(event_ids[2])},
    )
    assert resume.status_code == 200
    assert [event["id"] for event in _parse_sse_events(resume.text)] == event_ids[3:]


# =========================================================================== #
# Deploy-but-no-result is bounded (VAL-CROSS-031)
# =========================================================================== #
async def test_val_cross_031_deploy_but_no_result_folds_no_weight_no_hang(
    database_session, monkeypatch, tmp_path
) -> None:
    _configure_runner_broker(monkeypatch, tmp_path, task_count=1)
    _enable_phala(monkeypatch, True)  # attested run that never emits a result
    tasks = _terminal_bench_tasks(1)
    task_id = tasks[0].task_id
    async with database_session() as session:
        _sub, job = await _create_job(
            session, agent_hash="stalled", tasks=tasks, tmp_path=tmp_path, miner_hotkey="hk-stalled"
        )
        await session.commit()
        job_pk, job_id = job.id, job.job_id

    # The CVM never produces a result; the coordination plane exhausts max_attempts
    # and folds the stalled unit to a SINGLE terminal failed (score 0) result.
    async with database_session() as session:
        folded = await fold_terminally_failed_work_unit(
            session, job_id=job_id, task_id=task_id, reason="deploy_but_no_result"
        )
        # Folding again is idempotent (never double-counts, never loops).
        await fold_terminally_failed_work_unit(
            session, job_id=job_id, task_id=task_id, reason="deploy_but_no_result"
        )
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()
    assert folded.status == "failed"

    # The job finalizes (no hang) as a single failed, zero-score result.
    assert summary is not None
    assert summary.status == "completed"
    assert summary.score == 0.0
    assert summary.passed_tasks == 0
    async with database_session() as session:
        results = (
            (await session.execute(select(TaskResult).where(TaskResult.job_id == job_pk)))
            .scalars()
            .all()
        )
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].score == 0.0

    # The stalled submission earns NO weight; a passing+attested job WOULD (the
    # eligibility gate is a real discriminator, not vacuously empty).
    required = effective_evaluation_task_count(core_settings.evaluation_task_count)
    async with database_session() as session:
        job_row = await session.get(EvaluationJob, job_pk)
    assert "hk-stalled" not in (await get_weights())
    assert is_reward_eligible_job(job_row, required, attestation_verified=False) is False
    passing = EvaluationJob(
        job_id="probe",
        submission_id=job_row.submission_id,
        status="completed",
        selected_tasks_json="[]",
        score=1.0,
        passed_tasks=required,
        total_tasks=required,
        verdict="valid",
    )
    assert is_reward_eligible_job(passing, required, attestation_verified=True) is True

    # Any nonce issued for that run cannot later be redeemed: past the eval bound
    # it expires and stays single-use (never releases a key afterward).
    bound_seconds = 120.0
    stalled_clock = _AdvanceableClock()
    store = NonceStore(ttl_seconds=bound_seconds, clock=stalled_clock)
    nonce = store.issue()
    assert store.is_outstanding(nonce) is True  # redeemable within the bound
    stalled_clock.now = bound_seconds + 1.0  # the deployed CVM never returned
    assert store.is_outstanding(nonce) is False
    assert store.consume(nonce) is NonceState.EXPIRED
    assert store.consume(nonce) is NonceState.CONSUMED  # burned, still non-redeemable

    # Positive control: within the bound the store DOES release once (single-use).
    fresh_store = NonceStore(ttl_seconds=bound_seconds, clock=_AdvanceableClock())
    fresh = fresh_store.issue()
    assert fresh_store.consume(fresh) is NonceState.OK
    assert fresh_store.consume(fresh) is NonceState.CONSUMED
