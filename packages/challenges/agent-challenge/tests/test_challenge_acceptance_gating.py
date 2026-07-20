"""Challenge-side Phala acceptance gating (VAL-VERIFY-015..019, 026).

With the Phala attestation flag ON, the decentralized validator writes a task's
score ONLY when that task's result carries a VERIFIED Phala attestation:

* VAL-VERIFY-015 -- an unattested result is rejected/parked (no TaskResult row).
* VAL-VERIFY-016 -- an attestation that fails verification is rejected/parked.
* VAL-VERIFY-017 -- a verified result is persisted exactly once (idempotent).
* VAL-VERIFY-018 -- partial attestation persists only the verified tasks; the job
  does not finalize by silently scoring the unverified tasks.
* VAL-VERIFY-019 -- weight eligibility requires verified attestations.
* VAL-VERIFY-026 -- a rejected/parked result records a retrievable, distinguishable
  reason (unattested vs verification-failed vs verifier-unavailable/retryable).

Plus the flag-OFF invariant: legacy scoring is byte-identical and the quote
verifier is never invoked.

The tests are discriminators: they build a genuinely valid attested envelope
(reused across cases) and, for each rejection case, tamper exactly one bound
component (agent_hash, scores, measurement, report_data, nonce, quote signature)
so a naive "accept if an attestation is present" gate would FAIL them.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import func, select

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
from agent_challenge.evaluation.attestation import (
    ATTESTATION_MISSING,
    ATTESTATION_VERIFICATION_FAILED,
    ATTESTATION_VERIFIER_UNAVAILABLE,
    AttestationGate,
    AttestationOutcome,
    AttestationVerifierUnavailable,
    InMemoryNonceLedger,
    ResultMeasurementAllowlist,
)
from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.own_runner.result_schema import (
    build_benchmark_result,
    format_benchmark_result_line,
)
from agent_challenge.evaluation.validator_executor import (
    execute_work_unit,
    finalize_job_if_complete,
    get_task_attestation,
    run_validator_cycle,
)
from agent_challenge.evaluation.weights import get_weights, is_reward_eligible_job
from agent_challenge.evaluation.work_units import list_pending_work_units
from agent_challenge.keyrelease.quote import (
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
    os_image_hash_from_registers,
)
from agent_challenge.models import AgentSubmission, EvaluationJob, TaskResult
from agent_challenge.sdk.executors import DockerRunResult

_REGS = {"mrtd": "11" * 48, "rtmr0": "22" * 48, "rtmr1": "33" * 48, "rtmr2": "44" * 48}
_ALT_REGS = {"mrtd": "aa" * 48, "rtmr0": "bb" * 48, "rtmr1": "cc" * 48, "rtmr2": "dd" * 48}
_COMPOSE_PAYLOAD = bytes(range(32))
_KEY_PROVIDER_PAYLOAD = b"kms-app-key-provider"


def _canonical_measurement(regs: dict[str, str] = _REGS) -> tuple[dict, list, str]:
    event_log, rtmr3 = build_rtmr3_event_log(
        [("compose-hash", _COMPOSE_PAYLOAD), ("key-provider", _KEY_PROVIDER_PAYLOAD)]
    )
    measurement = {
        **regs,
        "compose_hash": _COMPOSE_PAYLOAD.hex(),
        "os_image_hash": os_image_hash_from_registers(regs["mrtd"], regs["rtmr1"], regs["rtmr2"]),
    }
    return measurement, event_log, rtmr3


def _attested_line(
    task_id: str,
    *,
    agent_hash: str,
    nonce: str,
    score: float = 1.0,
    regs: dict[str, str] = _REGS,
    report_data_nonce: str | None = None,
    scores_override: dict | None = None,
) -> str:
    """A ``BASE_BENCHMARK_RESULT=`` line carrying a Phala-tier attested envelope.

    ``report_data_nonce`` (if given) binds the quote's report_data to a DIFFERENT
    nonce than the binding block (report_data tamper); ``scores_override`` reports
    scores that differ from the ones the digest was computed over (scores tamper).
    """

    scores = {task_id: score}
    task_ids = [task_id]
    measurement, event_log, rtmr3 = _canonical_measurement(regs)
    canonical = CanonicalMeasurement(**measurement)
    scores_digest = rd.scores_digest(scores)
    report_data_hex = rd.report_data_hex(
        canonical_measurement=canonical,
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores_digest=scores_digest,
        validator_nonce=report_data_nonce if report_data_nonce is not None else nonce,
    )
    quote = build_tdx_quote(
        mrtd=regs["mrtd"],
        rtmr0=regs["rtmr0"],
        rtmr1=regs["rtmr1"],
        rtmr2=regs["rtmr2"],
        rtmr3=rtmr3,
        report_data=report_data_hex,
    )
    attestation = build_phala_attestation(
        tdx_quote=quote,
        event_log=event_log,
        report_data_hex=report_data_hex,
        measurement=build_measurement(canonical, rtmr3=rtmr3),
        vm_config={},
    )
    envelope = build_execution_proof_envelope(manifest_sha256="ab" * 32, attestation=attestation)
    binding = build_attestation_binding(
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores=scores_override if scores_override is not None else scores,
        scores_digest=scores_digest,
        validator_nonce=nonce,
        canonical_measurement=canonical,
    )
    result = build_benchmark_result(
        status="completed", score=score, resolved=round(score), total=1, reason_code=None
    )
    result[EXECUTION_PROOF_RESULT_KEY] = envelope
    result[ATTESTATION_BINDING_RESULT_KEY] = binding
    return format_benchmark_result_line(result)


def _plain_line(score: float = 1.0) -> str:
    status = "completed" if score >= 1.0 else "failed"
    payload = {"status": status, "score": score, "resolved": round(score), "total": 1}
    return "BASE_BENCHMARK_RESULT=" + json.dumps(payload, sort_keys=True)


def _make_gate(
    *, nonces: list[str], regs: dict[str, str] = _REGS, verifier=None
) -> AttestationGate:
    measurement, _log, _rtmr3 = _canonical_measurement(regs)
    ledger = InMemoryNonceLedger()
    for nonce in nonces:
        ledger.issue(nonce)
    return AttestationGate(
        quote_verifier=verifier if verifier is not None else StaticQuoteVerifier(valid=True),
        allowlist=ResultMeasurementAllowlist.from_measurements([measurement]),
        nonce_validator=ledger,
    )


class _RecordingBroker:
    """A validator-own broker returning a preset stdout line per task."""

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


def _patch_terminal_bench(monkeypatch, tmp_path) -> None:
    base = "agent_challenge.evaluation.runner.settings"
    monkeypatch.setattr(f"{base}.benchmark_backend", "terminal_bench")
    monkeypatch.setattr(f"{base}.terminal_bench_execution_backend", "own_runner")
    monkeypatch.setattr(f"{base}.evaluation_concurrency", 1)
    monkeypatch.setattr(f"{base}.evaluation_task_count", 1)
    monkeypatch.setattr(f"{base}.docker_enabled", True)
    monkeypatch.setattr(f"{base}.docker_backend", "broker")
    monkeypatch.setattr(f"{base}.docker_broker_url", "https://broker.test")
    monkeypatch.setattr(f"{base}.docker_broker_token", "broker-token")
    monkeypatch.setattr(f"{base}.docker_broker_token_file", None)
    harbor = tmp_path / "harbor-runs"
    harbor.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(f"{base}.harbor_output_dir", str(harbor))


def _enable_phala(monkeypatch, enabled: bool = True) -> None:
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
    session,
    *,
    agent_hash: str,
    tasks: list[BenchmarkTask],
    tmp_path,
    miner_hotkey: str | None = None,
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


# --------------------------------------------------------------------------- #
# Gate unit behavior: a fully valid attested result verifies (positive control).
# --------------------------------------------------------------------------- #
def test_valid_attested_result_is_accepted():
    nonce = "nonce-ok"
    gate = _make_gate(nonces=[nonce])
    line = _attested_line("terminal-bench/task-0", agent_hash="agent-x", nonce=nonce)
    decision = gate.decide(line, expected_agent_hash="agent-x")
    assert decision.outcome is AttestationOutcome.VERIFIED
    assert decision.accepted is True
    assert decision.reason is None


# --------------------------------------------------------------------------- #
# VAL-VERIFY-016 discriminators: tamper one bound component -> rejected.
# --------------------------------------------------------------------------- #
def test_wrong_agent_hash_is_rejected():
    nonce = "nonce-agent"
    gate = _make_gate(nonces=[nonce])
    line = _attested_line("terminal-bench/task-0", agent_hash="attacker", nonce=nonce)
    decision = gate.decide(line, expected_agent_hash="real-agent")
    assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED


def test_tampered_scores_is_rejected():
    nonce = "nonce-scores"
    gate = _make_gate(nonces=[nonce])
    line = _attested_line(
        "terminal-bench/task-0",
        agent_hash="agent-x",
        nonce=nonce,
        scores_override={"terminal-bench/task-0": 0.5},
    )
    decision = gate.decide(line, expected_agent_hash="agent-x")
    assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED


def test_measurement_not_allowlisted_is_rejected():
    nonce = "nonce-meas"
    gate = _make_gate(nonces=[nonce])  # allowlist pins _REGS
    line = _attested_line(
        "terminal-bench/task-0", agent_hash="agent-x", nonce=nonce, regs=_ALT_REGS
    )
    decision = gate.decide(line, expected_agent_hash="agent-x")
    assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED


def test_report_data_mismatch_is_rejected():
    nonce = "nonce-rd"
    gate = _make_gate(nonces=[nonce])
    line = _attested_line(
        "terminal-bench/task-0",
        agent_hash="agent-x",
        nonce=nonce,
        report_data_nonce="a-different-nonce",
    )
    decision = gate.decide(line, expected_agent_hash="agent-x")
    assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED


def test_unknown_nonce_is_rejected():
    gate = _make_gate(nonces=[])  # nothing issued
    line = _attested_line("terminal-bench/task-0", agent_hash="agent-x", nonce="never-issued")
    decision = gate.decide(line, expected_agent_hash="agent-x")
    assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED


def test_forged_quote_signature_is_rejected():
    nonce = "nonce-sig"
    gate = _make_gate(nonces=[nonce], verifier=StaticQuoteVerifier(valid=False))
    line = _attested_line("terminal-bench/task-0", agent_hash="agent-x", nonce=nonce)
    decision = gate.decide(line, expected_agent_hash="agent-x")
    assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED


def test_out_of_date_tcb_is_rejected():
    nonce = "nonce-tcb"
    gate = _make_gate(nonces=[nonce], verifier=StaticQuoteVerifier(tcb_status="OutOfDate"))
    line = _attested_line("terminal-bench/task-0", agent_hash="agent-x", nonce=nonce)
    decision = gate.decide(line, expected_agent_hash="agent-x")
    assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED


def test_nonce_is_single_use_across_two_results():
    nonce = "nonce-once"
    gate = _make_gate(nonces=[nonce])
    line = _attested_line("terminal-bench/task-0", agent_hash="agent-x", nonce=nonce)
    first = gate.decide(line, expected_agent_hash="agent-x")
    second = gate.decide(line, expected_agent_hash="agent-x")
    assert first.outcome is AttestationOutcome.VERIFIED
    assert second.outcome is AttestationOutcome.VERIFICATION_FAILED


# --------------------------------------------------------------------------- #
# VAL-VERIFY-026: distinct, retrievable reasons; verifier outage is retryable.
# --------------------------------------------------------------------------- #
def test_unattested_result_reason_is_distinct():
    gate = _make_gate(nonces=[])
    decision = gate.decide(_plain_line(), expected_agent_hash="agent-x")
    assert decision.outcome is AttestationOutcome.UNATTESTED
    assert decision.reason == ATTESTATION_MISSING
    assert decision.retryable is False


def test_verifier_unavailable_is_retryable_park():
    nonce = "nonce-unavail"

    class _Unavailable:
        def verify(self, quote_hex):
            raise AttestationVerifierUnavailable("collateral fetch timed out")

    gate = _make_gate(nonces=[nonce], verifier=_Unavailable())
    line = _attested_line("terminal-bench/task-0", agent_hash="agent-x", nonce=nonce)
    decision = gate.decide(line, expected_agent_hash="agent-x")
    assert decision.outcome is AttestationOutcome.VERIFIER_UNAVAILABLE
    assert decision.reason == ATTESTATION_VERIFIER_UNAVAILABLE
    assert decision.retryable is True


# --------------------------------------------------------------------------- #
# VAL-VERIFY-015: flag ON -- unattested result parked, NO score row.
# --------------------------------------------------------------------------- #
async def test_flag_on_unattested_result_is_parked_not_scored(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch)
    tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="unattested", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_pk = job.id

    async with database_session() as session:
        units = await list_pending_work_units(session)
    broker = _RecordingBroker({tasks[0].task_id: _plain_line()})
    gate = _make_gate(nonces=["some-nonce"])
    async with database_session() as session:
        outcome = await execute_work_unit(session, units[0], executor=broker, attestation_gate=gate)
        await session.commit()

    assert outcome.posted is False
    assert outcome.attestation_reason == ATTESTATION_MISSING
    async with database_session() as session:
        count = await session.scalar(
            select(func.count(TaskResult.id)).where(TaskResult.job_id == job_pk)
        )
        record = await get_task_attestation(session, job_pk, tasks[0].task_id)
    assert count == 0
    assert record is not None
    assert record.verified is False
    assert record.reason == ATTESTATION_MISSING


# --------------------------------------------------------------------------- #
# VAL-VERIFY-016: flag ON -- failing-verification result parked, NO score row.
# --------------------------------------------------------------------------- #
async def test_flag_on_failing_attestation_is_parked_not_scored(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch)
    tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="tampered", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_pk = job.id

    nonce = "nonce-fail"
    # A genuine-looking attested line whose bound agent_hash != the submission's.
    line = _attested_line(tasks[0].task_id, agent_hash="attacker", nonce=nonce)
    async with database_session() as session:
        units = await list_pending_work_units(session)
    broker = _RecordingBroker({tasks[0].task_id: line})
    gate = _make_gate(nonces=[nonce])
    async with database_session() as session:
        outcome = await execute_work_unit(session, units[0], executor=broker, attestation_gate=gate)
        await session.commit()

    assert outcome.posted is False
    assert outcome.attestation_reason == ATTESTATION_VERIFICATION_FAILED
    assert outcome.retryable is False
    async with database_session() as session:
        count = await session.scalar(
            select(func.count(TaskResult.id)).where(TaskResult.job_id == job_pk)
        )
        record = await get_task_attestation(session, job_pk, tasks[0].task_id)
    assert count == 0
    assert record.reason == ATTESTATION_VERIFICATION_FAILED


# --------------------------------------------------------------------------- #
# VAL-VERIFY-017: flag ON -- verified result persisted exactly once (idempotent).
# --------------------------------------------------------------------------- #
async def test_flag_on_verified_result_persisted_once(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch)
    tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        submission, job = await _create_job(
            session, agent_hash="verified", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_pk = job.id
        agent_hash = submission.agent_hash

    nonce = "nonce-verified"
    line = _attested_line(tasks[0].task_id, agent_hash=agent_hash, nonce=nonce)
    gate = _make_gate(nonces=[nonce])
    broker = _RecordingBroker({tasks[0].task_id: line})

    async with database_session() as session:
        units = await list_pending_work_units(session)
    async with database_session() as session:
        first = await execute_work_unit(session, units[0], executor=broker, attestation_gate=gate)
        await session.commit()
    assert first.posted is True
    assert first.score == 1.0

    # Re-post the same unit: the already-terminal result short-circuits (no
    # re-verify, no duplicate row, no second nonce consumption).
    repost_broker = _RecordingBroker({tasks[0].task_id: line})
    async with database_session() as session:
        units2 = await list_pending_work_units(session)
    # The task is now terminal, so it is no longer pending; force a direct re-post.
    async with database_session() as session:
        second = await execute_work_unit(
            session, units[0], executor=repost_broker, attestation_gate=gate
        )
        await session.commit()
    assert second.posted is False
    assert repost_broker.runs == []
    assert units2 == []

    async with database_session() as session:
        count = await session.scalar(
            select(func.count(TaskResult.id)).where(TaskResult.job_id == job_pk)
        )
        record = await get_task_attestation(session, job_pk, tasks[0].task_id)
    assert count == 1
    assert record.verified is True


# --------------------------------------------------------------------------- #
# VAL-VERIFY-018: flag ON -- partial attestation persists only verified tasks.
# --------------------------------------------------------------------------- #
async def test_flag_on_partial_attestation_scores_only_verified(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch)
    tasks = _terminal_bench_tasks(5)
    async with database_session() as session:
        submission, job = await _create_job(
            session, agent_hash="partial", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_id = job.job_id
        job_pk = job.id
        agent_hash = submission.agent_hash

    verified_tasks = tasks[:3]
    nonces = {task.task_id: f"nonce-{index}" for index, task in enumerate(verified_tasks)}
    lines = {
        t.task_id: _attested_line(t.task_id, agent_hash=agent_hash, nonce=nonces[t.task_id])
        for t in verified_tasks
    }
    # The last two tasks report unattested results.
    for task in tasks[3:]:
        lines[task.task_id] = _plain_line()

    gate = _make_gate(nonces=list(nonces.values()))
    broker = _RecordingBroker(lines)
    summary = await run_validator_cycle(executor=broker, attestation_gate=gate)

    # Only the 3 verified tasks were scored; the job is NOT finalized.
    assert summary.finalized_jobs == ()
    async with database_session() as session:
        results = (
            (await session.execute(select(TaskResult).where(TaskResult.job_id == job_pk)))
            .scalars()
            .all()
        )
        job_row = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert {r.task_id for r in results} == {t.task_id for t in verified_tasks}
    assert len(results) == 3
    assert job_row.status != "completed"

    # Finalizing is a safe no-op while the unattested tasks are unscored.
    async with database_session() as session:
        assert await finalize_job_if_complete(session, job_id) is None
        await session.commit()

    # The unattested tasks recorded a retrievable park reason (not scored as 0).
    async with database_session() as session:
        for task in tasks[3:]:
            record = await get_task_attestation(session, job_pk, task.task_id)
            assert record is not None
            assert record.verified is False
            assert record.reason == ATTESTATION_MISSING


# --------------------------------------------------------------------------- #
# VAL-VERIFY-019: flag ON -- weight eligibility requires verified attestations.
# --------------------------------------------------------------------------- #
def test_is_reward_eligible_requires_attestation_when_flagged():
    job = EvaluationJob(
        job_id="j",
        submission_id=1,
        status="completed",
        selected_tasks_json="[]",
        total_tasks=1,
        passed_tasks=1,
        score=1.0,
    )
    assert is_reward_eligible_job(job, 1, attestation_verified=True) is True
    assert is_reward_eligible_job(job, 1, attestation_verified=False) is False
    # Default (flag-off callers) preserves legacy eligibility.
    assert is_reward_eligible_job(job, 1) is True


async def test_flag_on_only_attested_job_earns_weight(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    # Per-hotkey weights (not winner-take-all) so an eligible job's absence is
    # attributable to the attestation gate, not to a single-winner tiebreak.
    monkeypatch.setattr("agent_challenge.core.config.settings.weights_winner_take_all", False)
    tasks_a = _terminal_bench_tasks(1)

    # Job A: run under the flag ON with a verified attestation -> completes with a
    # verified attestation record.
    _enable_phala(monkeypatch, True)
    async with database_session() as session:
        submission_a, _job_a = await _create_job(
            session, agent_hash="attested-A", tasks=tasks_a, tmp_path=tmp_path, miner_hotkey="hk-A"
        )
        await session.commit()
        agent_hash_a = submission_a.agent_hash
    nonce = "nonce-A"
    line = _attested_line(tasks_a[0].task_id, agent_hash=agent_hash_a, nonce=nonce)
    gate = _make_gate(nonces=[nonce])
    summary_a = await run_validator_cycle(
        executor=_RecordingBroker({tasks_a[0].task_id: line}), attestation_gate=gate
    )
    assert summary_a.finalized_jobs != ()

    # Job B: run under the flag OFF (legacy) -> completes with NO attestation record.
    _enable_phala(monkeypatch, False)
    async with database_session() as session:
        await _create_job(
            session,
            agent_hash="unattested-B",
            tasks=_terminal_bench_tasks(1),
            tmp_path=tmp_path,
            miner_hotkey="hk-B",
        )
        await session.commit()
    summary_b = await run_validator_cycle(executor=_RecordingBroker({}))
    assert summary_b.finalized_jobs != ()

    # Flag OFF: both threshold-meeting jobs earn weight (legacy behavior).
    assert set(await get_weights()) == {"hk-A", "hk-B"}

    # Flag ON: only the attestation-verified job A earns weight; B is burned.
    _enable_phala(monkeypatch, True)
    weights = await get_weights()
    assert set(weights) == {"hk-A"}


# --------------------------------------------------------------------------- #
# Flag OFF invariant: legacy scoring is byte-identical; verifier never invoked.
# --------------------------------------------------------------------------- #
async def test_flag_off_scores_unattested_and_never_verifies(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch, False)
    tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="flag-off", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_pk = job.id

    class _ExplodingVerifier:
        def verify(self, quote_hex):  # pragma: no cover - must never be called
            raise AssertionError("verifier invoked while flag OFF")

    gate = AttestationGate(
        quote_verifier=_ExplodingVerifier(),
        allowlist=ResultMeasurementAllowlist.from_measurements([_canonical_measurement()[0]]),
        nonce_validator=InMemoryNonceLedger(),
    )
    # Even with an attestation-bearing line, flag OFF ignores it entirely.
    line = _attested_line(tasks[0].task_id, agent_hash="anything", nonce="unissued")
    async with database_session() as session:
        units = await list_pending_work_units(session)
    async with database_session() as session:
        outcome = await execute_work_unit(
            session,
            units[0],
            executor=_RecordingBroker({tasks[0].task_id: line}),
            attestation_gate=gate,
        )
        await session.commit()

    assert outcome.posted is True
    assert outcome.score == 1.0
    assert outcome.attestation_reason is None
    async with database_session() as session:
        count = await session.scalar(
            select(func.count(TaskResult.id)).where(TaskResult.job_id == job_pk)
        )
        record = await get_task_attestation(session, job_pk, tasks[0].task_id)
    assert count == 1
    assert record is None  # no attestation bookkeeping on the flag-off path


@pytest.mark.parametrize(
    "outcome,expected_reason,retryable",
    [
        (AttestationOutcome.UNATTESTED, ATTESTATION_MISSING, False),
        (AttestationOutcome.VERIFICATION_FAILED, ATTESTATION_VERIFICATION_FAILED, False),
        (AttestationOutcome.VERIFIER_UNAVAILABLE, ATTESTATION_VERIFIER_UNAVAILABLE, True),
    ],
)
def test_decision_reason_taxonomy(outcome, expected_reason, retryable):
    from agent_challenge.evaluation.attestation import AttestationDecision

    decision = AttestationDecision.of(outcome)
    assert decision.reason == expected_reason
    assert decision.retryable is retryable


# --------------------------------------------------------------------------- #
# VAL-VERIFY-022: flag OFF -> legacy validator-run path, R=1, byte-identical
# scoring. A result carrying NO attestation is scored and finalize aggregates
# score = sum(task scores)/total, passed = count(score >= 1.0) -- unchanged.
# --------------------------------------------------------------------------- #
async def test_val_verify_022_flag_off_scores_plain_result_and_aggregates(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch, False)
    tasks = _terminal_bench_tasks(2)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="legacy-plain", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_pk = job.id
        job_id = job.job_id

    # One task resolved (1.0), one unresolved (0.0) -- both plain (no attestation)
    # results on the legacy path.
    lines = {tasks[0].task_id: _plain_line(1.0), tasks[1].task_id: _plain_line(0.0)}
    async with database_session() as session:
        units = await list_pending_work_units(session)
    for unit in units:
        async with database_session() as session:
            outcome = await execute_work_unit(session, unit, executor=_RecordingBroker(lines))
            await session.commit()
        # Legacy path: a score is written with no attestation bookkeeping.
        assert outcome.posted is True
        assert outcome.attestation_reason is None

    async with database_session() as session:
        summary = await finalize_job_if_complete(session, job_id)
        await session.commit()
        n_results = await session.scalar(
            select(func.count(TaskResult.id)).where(TaskResult.job_id == job_pk)
        )
        rec0 = await get_task_attestation(session, job_pk, tasks[0].task_id)
        rec1 = await get_task_attestation(session, job_pk, tasks[1].task_id)

    # score = (1.0 + 0.0) / 2 = 0.5; passed = count(score >= 1.0) = 1.
    assert summary is not None
    assert summary.total_tasks == 2
    assert summary.passed_tasks == 1
    assert summary.score == 0.5
    assert n_results == 2
    # No attestation records on the flag-off path.
    assert rec0 is None and rec1 is None


# --------------------------------------------------------------------------- #
# VAL-VERIFY-023: flag OFF -> the external quote-verify dependency is NEVER
# invoked, even when the result carries an attestation payload (it is ignored
# and scored via the legacy path).
# --------------------------------------------------------------------------- #
async def test_val_verify_023_flag_off_external_verifier_zero_invocations(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch, False)
    tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="flag-off-verify", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_pk = job.id

    class _CountingVerifier:
        def __init__(self) -> None:
            self.calls = 0

        def verify(self, quote_hex):
            self.calls += 1
            return StaticQuoteVerifier(valid=True).verify(quote_hex)

    verifier = _CountingVerifier()
    # A fully wired, would-accept gate: if the flag-off path ever consulted it,
    # the counter would advance. It must stay at zero.
    gate = AttestationGate(
        quote_verifier=verifier,
        allowlist=ResultMeasurementAllowlist.from_measurements([_canonical_measurement()[0]]),
        nonce_validator=InMemoryNonceLedger(),
    )
    line = _attested_line(tasks[0].task_id, agent_hash="anything", nonce="unissued")
    async with database_session() as session:
        units = await list_pending_work_units(session)
    async with database_session() as session:
        outcome = await execute_work_unit(
            session,
            units[0],
            executor=_RecordingBroker({tasks[0].task_id: line}),
            attestation_gate=gate,
        )
        await session.commit()

    # The attestation-bearing result is scored via the legacy path unchanged.
    assert outcome.posted is True
    assert outcome.score == 1.0
    assert outcome.attestation_reason is None
    # The external quote-verify dependency recorded zero invocations.
    assert verifier.calls == 0
    async with database_session() as session:
        count = await session.scalar(
            select(func.count(TaskResult.id)).where(TaskResult.job_id == job_pk)
        )
        record = await get_task_attestation(session, job_pk, tasks[0].task_id)
    assert count == 1
    assert record is None


# --------------------------------------------------------------------------- #
# VAL-VERIFY-024: flag OFF -> weight eligibility is unchanged (thresholds only);
# eligibility is identical with and without an attestation payload.
# --------------------------------------------------------------------------- #
def test_val_verify_024_flag_off_eligibility_thresholds_only():
    job = EvaluationJob(
        job_id="j",
        submission_id=1,
        status="completed",
        selected_tasks_json="[]",
        total_tasks=2,
        passed_tasks=1,
        score=0.5,
    )
    # Flag-off callers never pass an attestation gate; eligibility is the legacy
    # conjunction (total_tasks >= required AND passed_tasks >= 1) only.
    assert is_reward_eligible_job(job, 2) is True
    assert is_reward_eligible_job(job, 3) is False  # below required task count
    # A below-threshold pass count is ineligible regardless.
    job.passed_tasks = 0
    assert is_reward_eligible_job(job, 2) is False


async def test_val_verify_024_flag_off_weights_ignore_attestation_payload(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch, False)
    monkeypatch.setattr("agent_challenge.core.config.settings.weights_winner_take_all", False)

    # Job P: scored from a PLAIN (no attestation) result.
    async with database_session() as session:
        await _create_job(
            session,
            agent_hash="elig-plain",
            tasks=_terminal_bench_tasks(1),
            tmp_path=tmp_path,
            miner_hotkey="hk-P",
        )
        await session.commit()
    # Job Q: scored from an ATTESTED result -- the attestation must be ignored
    # (flag OFF), so Q is treated identically to P for eligibility.
    q_tasks = _terminal_bench_tasks(1)
    async with database_session() as session:
        await _create_job(
            session,
            agent_hash="elig-attested",
            tasks=q_tasks,
            tmp_path=tmp_path,
            miner_hotkey="hk-Q",
        )
        await session.commit()

    attested = _attested_line(q_tasks[0].task_id, agent_hash="elig-attested", nonce="n")
    summary = await run_validator_cycle(executor=_RecordingBroker({q_tasks[0].task_id: attested}))
    assert summary.finalized_jobs != ()

    # Flag OFF: both threshold-meeting jobs earn weight -- the attestation payload
    # on Q makes no difference to eligibility.
    assert set(await get_weights()) == {"hk-P", "hk-Q"}
