"""Phala TDX tier for ExecutionProof: schema, report_data binding, build helper.

Foundational schema shared by the image emitter (M1) and the verifier (M4)
(architecture.md sec 6/9). These tests pin the Phala tier value, the attestation
payload schema, the architecture-sec-6 ``report_data`` derivation, and the
``build_phala_execution_proof`` helper -- while asserting existing tier-0/1/2
behavior is unchanged.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from base.schemas.worker import (
    PHALA_TDX_TIER,
    ExecutionProof,
    PhalaAttestation,
    PhalaMeasurement,
    WorkerSignature,
)
from base.validator.agent.signing import KeypairRequestSigner
from base.worker.proof import (
    PHALA_REPORT_DATA_TAG,
    build_phala_execution_proof,
    phala_report_data,
    phala_report_data_hex,
    verify_execution_proof,
)

MANIFEST = "a" * 64
UNIT_ID = "submission-phala-1"


def _signer(uri: str = "//WorkerPhala") -> KeypairRequestSigner:
    import bittensor as bt

    return KeypairRequestSigner(bt.Keypair.create_from_uri(uri))


def _measurement(rtmr3: str = "d" * 96) -> PhalaMeasurement:
    return PhalaMeasurement(
        mrtd="a" * 96,
        rtmr0="b0" * 48,
        rtmr1="b1" * 48,
        rtmr2="b2" * 48,
        rtmr3=rtmr3,
        compose_hash="c" * 64,
        os_image_hash="e" * 64,
    )


def _attestation() -> PhalaAttestation:
    return PhalaAttestation(
        tdx_quote="0xdeadbeef",
        event_log=[{"event": "compose-hash", "digest": "c" * 64}],
        report_data="ab" * 64,
        measurement=_measurement(),
        vm_config={"vcpu": 1, "memory_mb": 2048},
    )


def _report_data_kwargs() -> dict[str, object]:
    return dict(
        canonical_measurement=_measurement(),
        agent_hash="f" * 64,
        task_ids=["task-b", "task-a", "task-c"],
        scores_digest="9" * 64,
        validator_nonce="nonce-123",
    )


# --- tier constant + schema ------------------------------------------------


def test_phala_tdx_tier_constant_value() -> None:
    assert PHALA_TDX_TIER == "phala-tdx"


def test_phala_measurement_canonical_excludes_rtmr3() -> None:
    canonical = _measurement().canonical()
    assert set(canonical) == {
        "mrtd",
        "rtmr0",
        "rtmr1",
        "rtmr2",
        "compose_hash",
        "os_image_hash",
    }
    assert "rtmr3" not in canonical


def test_phala_attestation_round_trips() -> None:
    att = _attestation()
    dumped = att.model_dump(mode="json")
    assert set(dumped) >= {
        "tdx_quote",
        "event_log",
        "report_data",
        "measurement",
        "vm_config",
    }
    assert set(dumped["measurement"]) == {
        "mrtd",
        "rtmr0",
        "rtmr1",
        "rtmr2",
        "rtmr3",
        "compose_hash",
        "os_image_hash",
    }
    assert PhalaAttestation.model_validate(dumped) == att


def test_phala_attestation_accepts_architecture_aliases() -> None:
    att = PhalaAttestation.model_validate(
        {
            "tdx_quote_b64": "0xcafe",
            "report_data_hex": "ff" * 64,
            "measurement": _measurement().model_dump(),
        }
    )
    assert att.tdx_quote == "0xcafe"
    assert att.report_data == "ff" * 64
    # serialization uses the canonical field names.
    assert "tdx_quote" in att.model_dump()
    assert "report_data" in att.model_dump()


@pytest.mark.parametrize("missing", ["tdx_quote", "report_data", "measurement"])
def test_phala_attestation_requires_core_fields(missing: str) -> None:
    payload = {
        "tdx_quote": "0xdead",
        "report_data": "ab" * 64,
        "measurement": _measurement().model_dump(),
    }
    payload.pop(missing)
    with pytest.raises(ValidationError):
        PhalaAttestation.model_validate(payload)


# --- report_data derivation (architecture sec 6) ---------------------------


def test_report_data_is_deterministic_32_bytes() -> None:
    a = phala_report_data(**_report_data_kwargs())  # type: ignore[arg-type]
    b = phala_report_data(**_report_data_kwargs())  # type: ignore[arg-type]
    assert a == b
    assert isinstance(a, bytes)
    assert len(a) == 32


def test_report_data_tag_is_bound() -> None:
    assert PHALA_REPORT_DATA_TAG == "base-agent-challenge-v1"


def test_report_data_task_ids_order_independent() -> None:
    kwargs = _report_data_kwargs()
    kwargs["task_ids"] = ["task-a", "task-b", "task-c"]
    forward = phala_report_data(**kwargs)  # type: ignore[arg-type]
    kwargs["task_ids"] = ["task-c", "task-a", "task-b"]
    shuffled = phala_report_data(**kwargs)  # type: ignore[arg-type]
    assert forward == shuffled


def test_report_data_ignores_rtmr3_runtime_register() -> None:
    base_kwargs = _report_data_kwargs()
    base_kwargs["canonical_measurement"] = _measurement(rtmr3="d" * 96)
    other = _report_data_kwargs()
    other["canonical_measurement"] = _measurement(rtmr3="7" * 96)
    assert phala_report_data(**base_kwargs) == phala_report_data(**other)  # type: ignore[arg-type]


def test_report_data_sensitive_to_every_bound_component() -> None:
    base_digest = phala_report_data(**_report_data_kwargs())  # type: ignore[arg-type]

    changed_measurement = _report_data_kwargs()
    m = _measurement()
    m.compose_hash = "0" * 64
    changed_measurement["canonical_measurement"] = m

    changed_agent = _report_data_kwargs()
    changed_agent["agent_hash"] = "0" * 64

    changed_tasks = _report_data_kwargs()
    changed_tasks["task_ids"] = ["task-a", "task-b"]

    changed_scores = _report_data_kwargs()
    changed_scores["scores_digest"] = "0" * 64

    changed_nonce = _report_data_kwargs()
    changed_nonce["validator_nonce"] = "nonce-999"

    for perturbed in (
        changed_measurement,
        changed_agent,
        changed_tasks,
        changed_scores,
        changed_nonce,
    ):
        assert phala_report_data(**perturbed) != base_digest  # type: ignore[arg-type]


def test_report_data_nonce_changes_digest() -> None:
    a = _report_data_kwargs()
    a["validator_nonce"] = "nonce-A"
    b = _report_data_kwargs()
    b["validator_nonce"] = "nonce-B"
    assert phala_report_data(**a) != phala_report_data(**b)  # type: ignore[arg-type]


def test_report_data_accepts_measurement_mapping() -> None:
    as_model = phala_report_data(**_report_data_kwargs())  # type: ignore[arg-type]
    kwargs = _report_data_kwargs()
    kwargs["canonical_measurement"] = _measurement().model_dump()
    as_mapping = phala_report_data(**kwargs)  # type: ignore[arg-type]
    assert as_model == as_mapping


def test_report_data_hex_is_64_byte_zero_padded_field() -> None:
    digest = phala_report_data(**_report_data_kwargs())  # type: ignore[arg-type]
    hex_field = phala_report_data_hex(**_report_data_kwargs())  # type: ignore[arg-type]
    assert len(hex_field) == 128
    field_bytes = bytes.fromhex(hex_field)
    assert len(field_bytes) == 64
    assert field_bytes[:32] == digest
    assert field_bytes[32:] == b"\x00" * 32


# --- build_phala_execution_proof -------------------------------------------


def test_build_phala_execution_proof_sets_tier_and_attestation() -> None:
    signer = _signer()
    proof = build_phala_execution_proof(
        signer=signer,
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation=_attestation(),
    )
    assert proof.tier == PHALA_TDX_TIER
    assert proof.attestation is not None
    assert proof.attestation["tdx_quote"] == "0xdeadbeef"
    assert proof.attestation["measurement"]["mrtd"] == "a" * 96
    assert proof.worker_signature.worker_pubkey == signer.hotkey


def test_build_phala_execution_proof_accepts_attestation_mapping() -> None:
    signer = _signer()
    proof = build_phala_execution_proof(
        signer=signer,
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation={
            "tdx_quote_b64": "0xfeed",
            "report_data_hex": "ab" * 64,
            "measurement": _measurement().model_dump(),
        },
    )
    assert proof.attestation is not None
    assert proof.attestation["tdx_quote"] == "0xfeed"


def test_phala_proof_worker_signature_still_verifies() -> None:
    signer = _signer()
    proof = build_phala_execution_proof(
        signer=signer,
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation=_attestation(),
    )
    assert verify_execution_proof(proof, unit_id=UNIT_ID) is True
    assert verify_execution_proof(proof, unit_id="other-unit") is False


def test_phala_proof_round_trips_through_serialization() -> None:
    signer = _signer()
    proof = build_phala_execution_proof(
        signer=signer,
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation=_attestation(),
    )
    restored = ExecutionProof.model_validate(proof.model_dump(mode="json"))
    assert restored == proof
    assert restored.tier == PHALA_TDX_TIER
    att = PhalaAttestation.model_validate(restored.attestation)
    assert att.measurement.compose_hash == "c" * 64


# --- existing tier behavior unchanged --------------------------------------


def test_execution_proof_int_tier_unchanged() -> None:
    proof = ExecutionProof(
        version=1,
        tier=2,
        manifest_sha256=MANIFEST,
        worker_signature=WorkerSignature(worker_pubkey="pk", sig="0x00"),
    )
    assert proof.tier == 2
    assert isinstance(proof.tier, int)


def test_execution_proof_string_tier_supported() -> None:
    proof = ExecutionProof(
        version=1,
        tier=PHALA_TDX_TIER,
        manifest_sha256=MANIFEST,
        worker_signature=WorkerSignature(worker_pubkey="pk", sig="0x00"),
    )
    assert proof.tier == "phala-tdx"
    assert isinstance(proof.tier, str)
