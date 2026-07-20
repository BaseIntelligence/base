"""Permanent-park short-circuit for the Phala acceptance gate (misc-hardening).

Follow-up hardening on the M4 acceptance gate. A PERMANENT (non-retryable)
attestation park -- ``UNATTESTED`` or ``VERIFICATION_FAILED`` -- writes no
``TaskResult`` score row, so before this change the unit stayed pending and
``execute_work_unit`` re-ran the broker + re-gated it every validator cycle until
the coordination-plane ``max_attempts`` fold finally wrote a terminal failed
result. That wasted a full re-execution each cycle (a real money cost in the M6
live model, which re-runs the eval on the miner's funded CVM).

The gate now consults the recorded ``TaskAttestation.retryable`` flag: a prior
permanent (``retryable=False``) park short-circuits -- the unit folds directly to
a terminal ``failed`` result WITHOUT re-dispatching the broker; a retryable
(``VERIFIER_UNAVAILABLE``) park MUST still be retried. Flag-off is byte-identical
(the short-circuit never consults attestation bookkeeping).

Discriminators:
* the permanent-park test FAILS against the pre-change code (which re-runs the
  broker and re-parks the unit, writing no result);
* the retryable test FAILS against a naive impl that short-circuits every park;
* the flag-off test FAILS against an impl that short-circuits regardless of flag.
"""

from __future__ import annotations

from sqlalchemy import func, select
from test_challenge_acceptance_gating import (
    _attested_line,
    _create_job,
    _enable_phala,
    _make_gate,
    _patch_terminal_bench,
    _plain_line,
    _RecordingBroker,
    _terminal_bench_tasks,
)

from agent_challenge.evaluation.attestation import (
    ATTESTATION_MISSING,
    AttestationVerifierUnavailable,
)
from agent_challenge.evaluation.validator_executor import (
    execute_work_unit,
    get_task_attestation,
)
from agent_challenge.evaluation.work_units import list_pending_work_units
from agent_challenge.keyrelease.quote import StaticQuoteVerifier
from agent_challenge.models import TaskAttestation, TaskResult


class _UnavailableVerifier:
    """A quote verifier that reports a transient outage (retryable park)."""

    def verify(self, quote_hex):
        raise AttestationVerifierUnavailable("collateral fetch timed out")


# --------------------------------------------------------------------------- #
# A permanent (retryable=False) park short-circuits on the next cycle: the unit
# is NOT re-run through the broker, it folds directly to a terminal failed result.
# --------------------------------------------------------------------------- #
async def test_permanent_park_shortcircuits_no_broker_rerun(
    database_session, monkeypatch, tmp_path
):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch)
    tasks = _terminal_bench_tasks(1)
    task_id = tasks[0].task_id
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="perm-park", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_pk = job.id

    gate = _make_gate(nonces=["some-nonce"])

    # Cycle 1: a plain (unattested) result -> permanent park (UNATTESTED); the
    # broker IS dispatched and NO TaskResult score row is written.
    async with database_session() as session:
        units = await list_pending_work_units(session)
    broker1 = _RecordingBroker({task_id: _plain_line()})
    async with database_session() as session:
        first = await execute_work_unit(session, units[0], executor=broker1, attestation_gate=gate)
        await session.commit()
    assert first.posted is False
    assert first.retryable is False
    assert broker1.runs == [task_id]

    async with database_session() as session:
        record = await get_task_attestation(session, job_pk, task_id)
        count_before = await session.scalar(
            select(func.count(TaskResult.id)).where(TaskResult.job_id == job_pk)
        )
        pending_before = await list_pending_work_units(session)
    assert record is not None and record.verified is False and record.retryable is False
    assert count_before == 0
    # No score row was written, so the unit is STILL pending for the next cycle.
    assert [u.work_unit_id for u in pending_before] == [units[0].work_unit_id]

    # Cycle 2: the still-pending unit MUST short-circuit -- the broker is NOT
    # re-dispatched and the unit folds directly to a terminal failed result.
    broker2 = _RecordingBroker({task_id: _plain_line()})
    async with database_session() as session:
        second = await execute_work_unit(
            session, pending_before[0], executor=broker2, attestation_gate=gate
        )
        await session.commit()
    assert broker2.runs == []
    assert second.executed is False
    assert second.posted is True
    assert second.status == "failed"
    assert second.score == 0.0

    # Exactly one terminal failed result now exists and the unit is no longer
    # pending (no further re-execution/re-gating cycles).
    async with database_session() as session:
        results = (
            (await session.execute(select(TaskResult).where(TaskResult.job_id == job_pk)))
            .scalars()
            .all()
        )
        pending_after = await list_pending_work_units(session)
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].score == 0.0
    assert pending_after == []


# --------------------------------------------------------------------------- #
# A retryable (VERIFIER_UNAVAILABLE) park is NOT short-circuited: the broker is
# re-dispatched on the next cycle and, with the verifier healthy, it verifies.
# --------------------------------------------------------------------------- #
async def test_retryable_park_is_still_retried(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch)
    tasks = _terminal_bench_tasks(1)
    task_id = tasks[0].task_id
    async with database_session() as session:
        submission, job = await _create_job(
            session, agent_hash="retry-park", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_pk = job.id
        agent_hash = submission.agent_hash

    nonce = "nonce-retry"
    line = _attested_line(task_id, agent_hash=agent_hash, nonce=nonce)

    # Cycle 1: the quote verifier is transiently unavailable -> retryable park
    # (no score row). The outage raises BEFORE the nonce is consumed.
    gate1 = _make_gate(nonces=[nonce], verifier=_UnavailableVerifier())
    async with database_session() as session:
        units = await list_pending_work_units(session)
    broker1 = _RecordingBroker({task_id: line})
    async with database_session() as session:
        first = await execute_work_unit(session, units[0], executor=broker1, attestation_gate=gate1)
        await session.commit()
    assert first.posted is False
    assert first.retryable is True
    assert broker1.runs == [task_id]

    async with database_session() as session:
        record = await get_task_attestation(session, job_pk, task_id)
        pending_before = await list_pending_work_units(session)
    assert record is not None and record.verified is False and record.retryable is True
    assert [u.work_unit_id for u in pending_before] == [units[0].work_unit_id]

    # Cycle 2: the retryable park MUST be retried -- the broker is re-dispatched
    # and, with the verifier now healthy, the result verifies and is persisted.
    gate2 = _make_gate(nonces=[nonce], verifier=StaticQuoteVerifier(valid=True))
    broker2 = _RecordingBroker({task_id: line})
    async with database_session() as session:
        second = await execute_work_unit(
            session, pending_before[0], executor=broker2, attestation_gate=gate2
        )
        await session.commit()
    assert broker2.runs == [task_id]
    assert second.executed is True
    assert second.posted is True
    assert second.status != "failed"
    assert second.score == 1.0

    async with database_session() as session:
        record2 = await get_task_attestation(session, job_pk, task_id)
    assert record2.verified is True


# --------------------------------------------------------------------------- #
# Flag OFF: the short-circuit never consults attestation bookkeeping, so even a
# seeded permanent-park record does not divert the legacy path.
# --------------------------------------------------------------------------- #
async def test_flag_off_ignores_permanent_park_record(database_session, monkeypatch, tmp_path):
    _patch_terminal_bench(monkeypatch, tmp_path)
    _enable_phala(monkeypatch, False)
    tasks = _terminal_bench_tasks(1)
    task_id = tasks[0].task_id
    async with database_session() as session:
        _submission, job = await _create_job(
            session, agent_hash="flag-off-park", tasks=tasks, tmp_path=tmp_path
        )
        await session.commit()
        job_pk = job.id
        session.add(
            TaskAttestation(
                job_id=job_pk,
                task_id=task_id,
                verified=False,
                reason=ATTESTATION_MISSING,
                retryable=False,
            )
        )
        await session.commit()

    broker = _RecordingBroker({task_id: _plain_line()})
    async with database_session() as session:
        units = await list_pending_work_units(session)
    async with database_session() as session:
        outcome = await execute_work_unit(session, units[0], executor=broker)
        await session.commit()

    # Byte-identical legacy behavior: the broker runs and a normal score is
    # written -- the seeded park record is ignored entirely.
    assert broker.runs == [task_id]
    assert outcome.executed is True
    assert outcome.posted is True
    assert outcome.status == "completed"
    assert outcome.score == 1.0
    assert outcome.attestation_reason is None

    async with database_session() as session:
        results = (
            (await session.execute(select(TaskResult).where(TaskResult.job_id == job_pk)))
            .scalars()
            .all()
        )
    assert len(results) == 1
    assert results[0].status == "completed"
