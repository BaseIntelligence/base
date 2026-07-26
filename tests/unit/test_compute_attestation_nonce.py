"""TDD tests for BASE-issued single-use attestation nonces (mechanism 1).

Nonce service only: issue UUID bound to work_unit_id + miner_hotkey + pod_id,
TTL from BASE clocks, exactly one successful consume. Guest clocks are never
consulted for security decisions (M3).
"""

from __future__ import annotations

import ast
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from base.compute.attestation_nonce import (
    AttestationNonceService,
    NonceBinding,
    NonceConsumeHit,
    NonceConsumeMiss,
    NonceConsumeReason,
    NonceRecord,
)

WORK_UNIT_A = "wu-aaa"
WORK_UNIT_B = "wu-bbb"
HOTKEY_A = "5HotkeyAAAA"
HOTKEY_B = "5HotkeyBBBB"
POD_A = "pod-111"
POD_B = "pod-222"
T0 = datetime(2026, 7, 26, 12, 0, 0, tzinfo=UTC)
TTL = timedelta(hours=2)


def _binding(
    *,
    work_unit_id: str = WORK_UNIT_A,
    miner_hotkey: str = HOTKEY_A,
    pod_id: str = POD_A,
) -> NonceBinding:
    return NonceBinding(
        work_unit_id=work_unit_id,
        miner_hotkey=miner_hotkey,
        pod_id=pod_id,
    )


def _service(
    *,
    now: datetime = T0,
    ttl: timedelta = TTL,
) -> tuple[AttestationNonceService, list[datetime]]:
    """Return service + mutable clock list (index 0 is current BASE time)."""
    clock = [now]

    def now_fn() -> datetime:
        return clock[0]

    return AttestationNonceService(ttl=ttl, now_fn=now_fn), clock


def test_issue_returns_uuid_bound_to_work_unit_hotkey_pod() -> None:
    """Given binding, When issue, Then UUID nonce + BASE issued_at/expires_at."""
    svc, _ = _service()
    binding = _binding()

    issued = svc.issue(binding)

    uuid.UUID(issued.nonce)  # raises if not a UUID string
    assert issued.binding == binding
    assert issued.issued_at == T0
    assert issued.expires_at == T0 + TTL
    assert isinstance(issued, NonceRecord)


def test_consume_once_accepts_matching_binding() -> None:
    """Happy path: issue then consume once with same binding → hit."""
    svc, clock = _service()
    binding = _binding()
    issued = svc.issue(binding)
    clock[0] = T0 + timedelta(minutes=5)

    result = svc.consume(issued.nonce, binding)

    assert isinstance(result, NonceConsumeHit)
    assert result.record.nonce == issued.nonce
    assert result.record.binding == binding
    assert result.received_at == clock[0]


def test_second_consume_rejected_as_already_consumed() -> None:
    """Replay: second consume of same nonce → already_consumed."""
    svc, clock = _service()
    binding = _binding()
    issued = svc.issue(binding)
    clock[0] = T0 + timedelta(minutes=1)
    first = svc.consume(issued.nonce, binding)
    assert isinstance(first, NonceConsumeHit)

    clock[0] = T0 + timedelta(minutes=2)
    second = svc.consume(issued.nonce, binding)

    assert isinstance(second, NonceConsumeMiss)
    assert second.reason is NonceConsumeReason.ALREADY_CONSUMED


def test_cross_work_unit_rejected_with_work_unit_mismatch() -> None:
    """Nonce for unit A rejected when consumed for unit B."""
    svc, clock = _service()
    issued = svc.issue(_binding(work_unit_id=WORK_UNIT_A))
    clock[0] = T0 + timedelta(minutes=1)

    result = svc.consume(
        issued.nonce,
        _binding(work_unit_id=WORK_UNIT_B),
    )

    assert isinstance(result, NonceConsumeMiss)
    assert result.reason is NonceConsumeReason.WORK_UNIT_MISMATCH


def test_cross_hotkey_rejected_with_miner_hotkey_mismatch() -> None:
    svc, clock = _service()
    issued = svc.issue(_binding(miner_hotkey=HOTKEY_A))
    clock[0] = T0 + timedelta(minutes=1)

    result = svc.consume(
        issued.nonce,
        _binding(miner_hotkey=HOTKEY_B),
    )

    assert isinstance(result, NonceConsumeMiss)
    assert result.reason is NonceConsumeReason.MINER_HOTKEY_MISMATCH


def test_cross_pod_rejected_with_pod_mismatch() -> None:
    svc, clock = _service()
    issued = svc.issue(_binding(pod_id=POD_A))
    clock[0] = T0 + timedelta(minutes=1)

    result = svc.consume(
        issued.nonce,
        _binding(pod_id=POD_B),
    )

    assert isinstance(result, NonceConsumeMiss)
    assert result.reason is NonceConsumeReason.POD_MISMATCH


def test_expired_nonce_rejected() -> None:
    """Past TTL (BASE receive time) → expired; guest clock never consulted."""
    svc, clock = _service(ttl=timedelta(seconds=30))
    issued = svc.issue(_binding())
    clock[0] = T0 + timedelta(seconds=31)

    result = svc.consume(issued.nonce, _binding())

    assert isinstance(result, NonceConsumeMiss)
    assert result.reason is NonceConsumeReason.EXPIRED


def test_unknown_nonce_rejected() -> None:
    svc, _ = _service()
    result = svc.consume(str(uuid.uuid4()), _binding())
    assert isinstance(result, NonceConsumeMiss)
    assert result.reason is NonceConsumeReason.UNKNOWN_NONCE


def test_consume_uses_explicit_received_at_not_guest_clock() -> None:
    """Freshness from BASE receive time only — no guest_timestamp parameter."""
    svc, clock = _service(ttl=timedelta(minutes=10))
    issued = svc.issue(_binding())
    # Service clock advanced past expiry, but explicit BASE received_at is early.
    clock[0] = T0 + timedelta(hours=5)
    base_receive = T0 + timedelta(minutes=1)

    result = svc.consume(
        issued.nonce,
        _binding(),
        received_at=base_receive,
    )

    assert isinstance(result, NonceConsumeHit)
    assert result.received_at == base_receive
    # API must not accept a guest clock for security decisions.
    assert "guest" not in svc.consume.__code__.co_varnames


def test_expired_check_order_before_binding_when_already_past_ttl() -> None:
    """Expired wins over binding mismatch when both would apply."""
    svc, clock = _service(ttl=timedelta(seconds=10))
    issued = svc.issue(_binding(work_unit_id=WORK_UNIT_A))
    clock[0] = T0 + timedelta(seconds=11)

    result = svc.consume(
        issued.nonce,
        _binding(work_unit_id=WORK_UNIT_B),
    )

    assert isinstance(result, NonceConsumeMiss)
    assert result.reason is NonceConsumeReason.EXPIRED


def test_snapshot_roundtrip_for_persistence_boundary() -> None:
    svc, clock = _service()
    issued = svc.issue(_binding())
    clock[0] = T0 + timedelta(minutes=1)
    assert isinstance(svc.consume(issued.nonce, _binding()), NonceConsumeHit)

    snap = svc.snapshot()
    restored = AttestationNonceService.from_snapshot(
        snap,
        ttl=TTL,
        now_fn=lambda: clock[0],
    )
    # Consumed state survives restore → replay still rejected.
    again = restored.consume(issued.nonce, _binding())
    assert isinstance(again, NonceConsumeMiss)
    assert again.reason is NonceConsumeReason.ALREADY_CONSUMED


def test_issue_rejects_blank_binding_fields() -> None:
    svc, _ = _service()
    with pytest.raises(ValueError, match="work_unit_id"):
        svc.issue(_binding(work_unit_id="  "))
    with pytest.raises(ValueError, match="miner_hotkey"):
        svc.issue(_binding(miner_hotkey=""))
    with pytest.raises(ValueError, match="pod_id"):
        svc.issue(_binding(pod_id="\t"))


def test_orm_model_and_metadata_table_exist() -> None:
    """Durable table matches pure NonceRecord (migration 0018)."""
    from base.db import AttestationNonce, Base

    row = AttestationNonce(
        nonce=str(uuid.uuid4()),
        work_unit_id=WORK_UNIT_A,
        miner_hotkey=HOTKEY_A,
        pod_id=POD_A,
        issued_at=T0,
        expires_at=T0 + TTL,
        consumed_at=None,
    )
    assert row.work_unit_id == WORK_UNIT_A
    assert "attestation_nonces" in Base.metadata.tables
    assert AttestationNonce.__tablename__ == "attestation_nonces"


def test_alembic_migration_0018_chained_from_digest_allowlist() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "0018_attestation_nonces.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in {"revision", "down_revision"} and isinstance(
                node.value, ast.Constant
            ):
                values[node.target.id] = node.value.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "revision",
                    "down_revision",
                }:
                    if isinstance(node.value, ast.Constant):
                        values[target.id] = node.value.value
    assert values["revision"] == "0018_attestation_nonces"
    assert values["down_revision"] == "0017_digest_allowlist"
    text = path.read_text(encoding="utf-8")
    assert "attestation_nonces" in text
    assert "work_unit_id" in text
    assert "miner_hotkey" in text
    assert "pod_id" in text
    assert "consumed_at" in text
