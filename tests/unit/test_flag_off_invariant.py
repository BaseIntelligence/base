"""Base-side flag-OFF invariant for the Phala-attested agent-challenge integration.

When the Phala attestation flag is OFF an agent-challenge submission is legacy /
unattested: no Phala-tier ``proof`` rides in the bridged work-unit payload. This
module pins the base-side manifestation of that invariant (architecture.md sec 8;
AGENTS.md "Feature-flagged, byte-identical legacy behavior when OFF"):

* VAL-VERIFY-022 -- the master bridges an unattested agent-challenge submission at
  R=1 exactly like the legacy validator-run path: one cpu work unit per selected
  task assigned to a single validator, no worker-plane replica / reconciliation /
  audit unit. The bridged unit set is byte-identical whether or not an attestation
  payload rides along, so attestation presence never perturbs the flag-off path.
* VAL-VERIFY-023 -- no base code path invokes the Phala quote verifier
  (:func:`base.worker.proof.verify_execution_proof`), the validator-side signature
  rebind (``base.validator.agent.adapters.agent_challenge.rebind_worker_signature``),
  or the external dcap-qvl dependency (``DcapQvlVerifier.verify``) while the flag is
  off -- even when a result carries an attestation payload it is ignored and
  dispatched via the legacy own_runner path.

The verifier surfaces are genuinely wired functions (see
``test_worker_proof_phala_verify``); a positive-control test proves the spy detects
a real invocation, so the zero-invocation assertions are non-vacuous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select

from base.db import (
    Base,
    Validator,
    ValidatorStatus,
    WorkAssignmentStatus,
    WorkerAssignment,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment
from base.master.assignment import (
    EXECUTOR_KIND_PAYLOAD_KEY,
    EXECUTOR_KIND_VALIDATOR,
    AssignmentService,
)
from base.master.orchestration import (
    ChallengePendingWork,
    MasterOrchestrationDriver,
)
from base.master.validator_coordination import ValidatorCoordinationService
from base.master.worker_reconciliation import AUDIT_WORK_UNIT_SUFFIX
from base.schemas.assignment import AssignmentView
from base.validator.agent import AssignmentContext, BrokerConfig
from base.validator.agent.adapters import AgentChallengeCycleExecutor

NOW = datetime(2026, 6, 30, 12, 0, 0, tzinfo=UTC)

#: The three base "Phala verifier" surfaces that MUST stay untouched with the
#: flag off (patched to recorders in the guardrail tests).
_VERIFY_EXECUTION_PROOF = "base.worker.proof.verify_execution_proof"
_REBIND_WORKER_SIGNATURE = (
    "base.validator.agent.adapters.agent_challenge.rebind_worker_signature"
)
_DCAP_QVL_VERIFY = "base.worker.phala_quote.DcapQvlVerifier.verify"

_ACTIVE = (WorkAssignmentStatus.ASSIGNED, WorkAssignmentStatus.RUNNING)


@dataclass
class _FakeWorkSource:
    works: list[ChallengePendingWork] = field(default_factory=list)

    async def fetch_pending_work(self) -> list[ChallengePendingWork]:
        return list(self.works)


@dataclass
class _FakeFoldTrigger:
    calls: list[tuple[str, str, str, str]] = field(default_factory=list)

    async def fold(
        self, *, challenge_slug: str, job_id: str, task_id: str, reason: str
    ) -> None:
        self.calls.append((challenge_slug, job_id, task_id, reason))


async def _setup() -> tuple[Any, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, create_session_factory(engine)


def _build_flag_off_driver(
    factory: Any, works: list[ChallengePendingWork]
) -> MasterOrchestrationDriver:
    """A flag-OFF driver: no worker engine, no reconciler (legacy routing).

    ``worker_plane_capabilities`` defaults empty so every capability (incl. cpu)
    is validator-assigned, and neither the worker assignment engine nor the
    reconciler is constructed -- ``run_once`` leaves ``worker``/``reconciliation``
    ``None`` (byte-identical legacy path).
    """

    assignment_service = AssignmentService(
        factory, now_fn=lambda: NOW, default_max_attempts=3
    )
    validator_service = ValidatorCoordinationService(factory, now_fn=lambda: NOW)
    return MasterOrchestrationDriver(
        assignment_service=assignment_service,
        validator_service=validator_service,
        work_source=_FakeWorkSource(works=works),
        fold_trigger=_FakeFoldTrigger(),
        worker_assignment_engine=None,
        worker_reconciler=None,
        seed=1,
    )


async def _add_validator(factory: Any, hotkey: str, capabilities: list[str]) -> None:
    async with session_scope(factory) as session:
        session.add(
            Validator(
                hotkey=hotkey,
                uid=None,
                status=ValidatorStatus.ONLINE,
                capabilities=list(capabilities),
                version="1.0.0",
                registered_at=NOW,
                last_heartbeat_at=NOW,
            )
        )


async def _units(factory: Any) -> dict[str, WorkAssignment]:
    async with factory() as session:
        rows = (await session.execute(select(WorkAssignment))).scalars().all()
        return {r.work_unit_id: r for r in rows}


async def _replica_count(factory: Any, work_unit_id: str) -> int:
    async with factory() as session:
        return (
            await session.execute(
                select(func.count())
                .select_from(WorkerAssignment)
                .where(
                    WorkerAssignment.work_unit_id == work_unit_id,
                    WorkerAssignment.status.in_(_ACTIVE),
                )
            )
        ).scalar_one()


def _agent_work(
    *,
    task_ids: tuple[str, ...] = ("a", "b", "c"),
    job_id: str | None = "job-1",
    attested: bool = False,
) -> ChallengePendingWork:
    # Flag OFF => unattested (empty payload). ``attested=True`` rides a Phala-tier
    # proof block in the payload to prove attestation presence is INERT on the
    # flag-off base path (the master never reads/verifies it).
    payload: dict[str, Any] = {}
    if attested:
        payload["proof"] = {"tier": "phala-tdx"}
    return ChallengePendingWork(
        challenge_slug="agent-challenge",
        submission_id="sub",
        submission_ref="miner-C",
        task_ids=task_ids,
        job_id=job_id,
        payload=payload,
    )


def _install_verifier_spies(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Patch every base Phala-verifier surface to record (and refuse) invocation."""

    def _spy(name: str) -> Any:
        def _recorder(*_a: Any, **_k: Any) -> Any:
            calls.append(name)
            raise AssertionError(f"{name} invoked while the Phala flag is OFF")

        return _recorder

    monkeypatch.setattr(_VERIFY_EXECUTION_PROOF, _spy("verify_execution_proof"))
    monkeypatch.setattr(_REBIND_WORKER_SIGNATURE, _spy("rebind_worker_signature"))
    monkeypatch.setattr(_DCAP_QVL_VERIFY, _spy("dcap_qvl_verify"))


# --------------------------------------------------------------------------- #
# VAL-VERIFY-022: flag OFF -> legacy validator-run path, R=1 (byte-identical).
# --------------------------------------------------------------------------- #
async def test_val_verify_022_flag_off_bridges_agent_challenge_at_r1_legacy() -> None:
    engine, factory = await _setup()
    try:
        driver = _build_flag_off_driver(factory, works=[_agent_work()])
        await _add_validator(factory, "v1", ["cpu"])

        result = await driver.run_once()

        # Worker plane is OFF -> no engine/reconciler ran this pass (legacy).
        assert result.worker is None
        assert result.reconciliation is None

        units = await _units(factory)
        cpu_ids = ["sub:a", "sub:b", "sub:c"]
        for uid in cpu_ids:
            unit = units[uid]
            assert unit.required_capability == "cpu"
            assert unit.status == WorkAssignmentStatus.ASSIGNED
            assert unit.assigned_validator_hotkey == "v1"  # single executor (R=1)
            assert unit.attempt_count == 1
            assert await _replica_count(factory, uid) == 0  # no worker replica
            # A cpu unit never carries a validator worker-plane executor marker.
            assert (unit.payload or {}).get(EXECUTOR_KIND_PAYLOAD_KEY) != (
                EXECUTOR_KIND_VALIDATOR
            )

        # Exactly the fanned cpu units exist; no audit / replica sibling unit.
        assert set(units) == set(cpu_ids)
        assert all(not uid.endswith(AUDIT_WORK_UNIT_SUFFIX) for uid in units)
    finally:
        await engine.dispose()


async def test_val_verify_022_attestation_payload_is_inert_on_flag_off_path() -> None:
    # The bridged unit shape must be byte-identical whether or not an attestation
    # payload rides along: attestation presence never perturbs the flag-off path.
    async def _bridge(attested: bool) -> dict[str, tuple[str, str | None, int]]:
        engine, factory = await _setup()
        try:
            driver = _build_flag_off_driver(
                factory, works=[_agent_work(attested=attested)]
            )
            await _add_validator(factory, "v1", ["cpu"])
            await driver.run_once()
            units = await _units(factory)
            return {
                uid: (
                    u.required_capability,
                    u.assigned_validator_hotkey,
                    u.attempt_count,
                )
                for uid, u in units.items()
            }
        finally:
            await engine.dispose()

    assert await _bridge(attested=False) == await _bridge(attested=True)


# --------------------------------------------------------------------------- #
# VAL-VERIFY-023: flag OFF -> the Phala quote verifier is NEVER invoked.
# --------------------------------------------------------------------------- #
async def test_val_verify_023_master_never_invokes_phala_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_verifier_spies(monkeypatch, calls)

    engine, factory = await _setup()
    try:
        # Even with an attestation-bearing payload, the flag-off master path must
        # neither read nor verify it.
        driver = _build_flag_off_driver(factory, works=[_agent_work(attested=True)])
        await _add_validator(factory, "v1", ["cpu"])

        result = await driver.run_once()

        assert calls == []  # zero verifier / rebind / dcap-qvl invocations
        units = await _units(factory)
        for uid in ("sub:a", "sub:b", "sub:c"):
            assert units[uid].assigned_validator_hotkey == "v1"
            assert await _replica_count(factory, uid) == 0
        assert result.reconciliation is None
    finally:
        await engine.dispose()


async def test_val_verify_023_adapter_dispatches_legacy_without_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_verifier_spies(monkeypatch, calls)
    dispatched: list[dict[str, Any]] = []

    async def _fake_dispatch(**kwargs: Any) -> dict[str, Any]:
        dispatched.append(kwargs)
        return {"pulled": 1, "executed": 1, "posted": 1, "skipped": 0}

    adapter = AgentChallengeCycleExecutor(dispatch=_fake_dispatch)
    assignment = AssignmentView(
        id="11111111-1111-1111-1111-111111111111",
        challenge_slug="agent-challenge",
        work_unit_id="sub:agent-challenge",
        submission_ref="sub",
        # An attestation-bearing payload the legacy path must ignore (not verify).
        payload={"task_id": "task-1", "proof": {"tier": "phala-tdx"}},
        required_capability="cpu",
        status="running",
        attempt_count=1,
        max_attempts=3,
    )
    context = AssignmentContext(
        assignment=assignment,
        gateway_env={"BASE_GATEWAY_TOKEN": "scoped-token"},
        broker=BrokerConfig(
            broker_url="http://broker-val:8082",
            broker_token="bt",
            broker_token_file="/run/bt",
            allowed_images=("img:1",),
        ),
    )

    async def _noop_progress(**_: Any) -> None:
        return None

    result = await adapter.execute(context, progress=_noop_progress)

    assert result.success is True  # dispatched via the legacy own_runner path
    assert len(dispatched) == 1
    assert calls == []  # the adapter never invoked any Phala verifier surface


def test_verifier_spy_is_not_vacuous(monkeypatch: pytest.MonkeyPatch) -> None:
    # Positive control: the spy DOES detect a real invocation, so the
    # zero-invocation assertions above are meaningful, not vacuous.
    import base.worker.proof as proof_module

    calls: list[str] = []
    _install_verifier_spies(monkeypatch, calls)
    with pytest.raises(AssertionError):
        proof_module.verify_execution_proof(object(), unit_id="u")  # type: ignore[arg-type]
    assert calls == ["verify_execution_proof"]
