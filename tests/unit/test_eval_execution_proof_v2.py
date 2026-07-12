"""Strict schema-v2 Eval ExecutionProof vectors and helper boundary tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from base.schemas.worker import (
    EVAL_MAX_EVENT_LOG_BYTES,
    EVAL_MAX_EVENT_LOG_ENTRIES,
    EVAL_MAX_INTEGER,
    EVAL_MAX_PAYLOAD_BYTES,
    EVAL_MAX_QUOTE_BYTES,
    EVAL_MAX_STRING_BYTES,
    EVAL_MAX_VM_CONFIG_BYTES,
    EvalExecutionProof,
    WorkerSignature,
    parse_eval_execution_proof_json,
)
from base.validator.agent.adapters.agent_challenge import (
    AssignmentExecutionError,
    rebind_worker_signature,
)
from base.worker.phala_quote import (
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
)
from base.worker.phala_verify import (
    InMemoryNonceValidator,
    MeasurementAllowlist,
    PhalaBinding,
)
from base.worker.proof import (
    PHALA_REPORT_DATA_TAG,
    phala_report_data,
    phala_report_data_hex,
    verify_execution_proof,
)

VECTOR_PATH = Path(__file__).with_name("eval_execution_proof_v2_vectors.json")
VECTORS: dict[str, Any] = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
POSITIVE: dict[str, Any] = VECTORS["positive"]


class _Signer:
    hotkey = "validator-hotkey"

    def sign(self, payload: bytes) -> str:
        return "0x" + hashlib.sha256(self.hotkey.encode() + payload).hexdigest()


def _set_path(value: dict[str, Any], path: str, replacement: Any) -> None:
    *parents, leaf = path.split(".")
    cursor: Any = value
    for part in parents:
        cursor = cursor[int(part)] if part.isdigit() else cursor[part]
    cursor[int(leaf) if leaf.isdigit() else leaf] = replacement


def _rename_path(value: dict[str, Any], source: str, target: str) -> None:
    *source_parents, source_leaf = source.split(".")
    cursor: Any = value
    for part in source_parents:
        cursor = cursor[int(part)] if part.isdigit() else cursor[part]
    moved = cursor.pop(source_leaf)
    _set_path(value, target, moved)


def _malformed_payload(vector: dict[str, str]) -> dict[str, Any]:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    if vector["kind"] == "set":
        _set_path(payload, vector["path"], vector["value"])
    else:
        _rename_path(payload, vector["from"], vector["to"])
    return payload


def _binding() -> PhalaBinding:
    binding = POSITIVE["binding"]
    return PhalaBinding(
        agent_hash=binding["agent_hash"],
        task_ids=tuple(binding["task_ids"]),
        scores_digest=binding["scores_digest"],
        eval_run_id=binding["eval_run_id"],
        score_nonce=binding["score_nonce"],
    )


def _signature_verifier(pubkey: str, payload: bytes, signature: str) -> bool:
    return signature == "0x" + hashlib.sha256(pubkey.encode() + payload).hexdigest()


def test_positive_vector_has_exact_v2_preimage_and_canonical_eval_envelope() -> None:
    binding = POSITIVE["binding"]
    preimage = {
        "agent_hash": binding["agent_hash"],
        "canonical_measurement": binding["canonical_measurement"],
        "domain": "base-agent-challenge-v1",
        "eval_run_id": binding["eval_run_id"],
        "schema_version": 2,
        "score_nonce": binding["score_nonce"],
        "scores_digest": binding["scores_digest"],
        "task_ids": binding["task_ids"],
    }
    encoded = json.dumps(
        preimage, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    assert encoded.hex() == POSITIVE["canonical_json_utf8_hex"]
    assert (
        phala_report_data_hex(
            canonical_measurement=binding["canonical_measurement"],
            agent_hash=binding["agent_hash"],
            task_ids=binding["task_ids"],
            scores_digest=binding["scores_digest"],
            eval_run_id=binding["eval_run_id"],
            score_nonce=binding["score_nonce"],
        )
        == POSITIVE["report_data_hex"]
    )
    parsed = EvalExecutionProof.model_validate(POSITIVE["execution_proof"])
    assert parsed.worker_signature.worker_pubkey == ""
    assert parsed.worker_signature.sig == ""


@pytest.mark.parametrize(
    "task_ids",
    [("task-b", "task-a"), ("task-a", "task-a")],
)
def test_v2_report_data_rejects_unsorted_or_duplicate_task_ids(
    task_ids: tuple[str, ...],
) -> None:
    binding = POSITIVE["binding"]
    with pytest.raises(ValueError):
        phala_report_data_hex(
            canonical_measurement=binding["canonical_measurement"],
            agent_hash=binding["agent_hash"],
            task_ids=task_ids,
            scores_digest=binding["scores_digest"],
            eval_run_id=binding["eval_run_id"],
            score_nonce=binding["score_nonce"],
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent_hash", "A" * 64),
        ("scores_digest", "9" * 63),
        ("eval_run_id", "contains whitespace"),
        ("score_nonce", "é"),
    ],
)
def test_v2_report_data_rejects_invalid_scalar_profiles(field: str, value: str) -> None:
    binding = POSITIVE["binding"]
    kwargs = {
        "canonical_measurement": binding["canonical_measurement"],
        "agent_hash": binding["agent_hash"],
        "task_ids": binding["task_ids"],
        "scores_digest": binding["scores_digest"],
        "eval_run_id": binding["eval_run_id"],
        "score_nonce": binding["score_nonce"],
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        phala_report_data_hex(**kwargs)


def test_legacy_report_data_derivation_remains_byte_identical() -> None:
    binding = POSITIVE["binding"]
    kwargs = {
        "canonical_measurement": binding["canonical_measurement"],
        "agent_hash": "legacy-☃",
        "task_ids": ("task-b", "task-a", "task-a"),
        "scores_digest": "legacy-scores",
        "validator_nonce": "legacy-nonce",
    }
    legacy_preimage = {
        "tag": PHALA_REPORT_DATA_TAG,
        "canonical_measurement": binding["canonical_measurement"],
        "agent_hash": "legacy-☃",
        "task_ids": ["task-a", "task-a", "task-b"],
        "scores_digest": "legacy-scores",
        "validator_nonce": "legacy-nonce",
    }
    expected = hashlib.sha256(
        json.dumps(legacy_preimage, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    assert phala_report_data(**kwargs) == expected


@pytest.mark.parametrize(
    "vector",
    VECTORS["malformed"],
    ids=lambda vector: vector["name"],
)
def test_strict_eval_boundary_rejects_shared_malformed_vectors(
    vector: dict[str, str],
) -> None:
    if vector["kind"] == "raw_json":
        with pytest.raises(ValueError, match="duplicate JSON key"):
            parse_eval_execution_proof_json(vector["value"])
        return
    with pytest.raises(ValidationError):
        EvalExecutionProof.model_validate(_malformed_payload(vector))


def test_v2_binding_includes_eval_run_id_and_distinct_score_nonce() -> None:
    binding = POSITIVE["binding"]
    event_log, rtmr3 = build_rtmr3_event_log(
        [
            (
                "compose-hash",
                bytes.fromhex(binding["canonical_measurement"]["compose_hash"]),
            )
        ]
    )
    report_data = phala_report_data_hex(
        canonical_measurement=binding["canonical_measurement"],
        agent_hash=binding["agent_hash"],
        task_ids=binding["task_ids"],
        scores_digest=binding["scores_digest"],
        eval_run_id=binding["eval_run_id"],
        score_nonce=binding["score_nonce"],
    )
    quote = build_tdx_quote(
        mrtd=binding["canonical_measurement"]["mrtd"],
        rtmr0=binding["canonical_measurement"]["rtmr0"],
        rtmr1=binding["canonical_measurement"]["rtmr1"],
        rtmr2=binding["canonical_measurement"]["rtmr2"],
        rtmr3=rtmr3,
        report_data=report_data,
    )
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    payload["attestation"]["tdx_quote"] = quote
    payload["attestation"]["event_log"] = event_log
    payload["attestation"]["report_data"] = report_data
    payload["attestation"]["measurement"]["rtmr3"] = rtmr3
    placeholder = EvalExecutionProof.model_validate(payload)
    proof = rebind_worker_signature(
        placeholder.to_execution_proof(), signer=_Signer(), unit_id="eval-run-001"
    )
    nonce_validator = InMemoryNonceValidator()
    nonce_validator.issue(binding["score_nonce"])
    assert (
        verify_execution_proof(
            proof,
            unit_id="eval-run-001",
            expected_binding=_binding(),
            quote_verifier=StaticQuoteVerifier(),
            allowlist=MeasurementAllowlist.from_measurements(
                [binding["canonical_measurement"]]
            ),
            nonce_validator=nonce_validator,
            signature_verifier=_signature_verifier,
        )
        is True
    )
    original = _binding()
    for field, value in (
        ("eval_run_id", "other-run"),
        ("score_nonce", "other-score-nonce"),
    ):
        crossed = PhalaBinding(
            agent_hash=original.agent_hash,
            task_ids=original.task_ids,
            scores_digest=original.scores_digest,
            eval_run_id=value if field == "eval_run_id" else original.eval_run_id,
            score_nonce=value if field == "score_nonce" else original.score_nonce,
        )
        crossed_nonces = InMemoryNonceValidator()
        crossed_nonces.issue(crossed.nonce)
        assert (
            verify_execution_proof(
                proof,
                unit_id="eval-run-001",
                expected_binding=crossed,
                quote_verifier=StaticQuoteVerifier(),
                allowlist=MeasurementAllowlist.from_measurements(
                    [binding["canonical_measurement"]]
                ),
                nonce_validator=crossed_nonces,
                signature_verifier=_signature_verifier,
            )
            is False
        )


@pytest.mark.parametrize(
    "signature",
    [
        {"worker_pubkey": "non-empty", "sig": ""},
        {"worker_pubkey": "", "sig": "non-empty"},
        {"worker_pubkey": "non-empty", "sig": "non-empty"},
    ],
)
def test_rebind_only_accepts_exact_empty_placeholder(
    signature: dict[str, str],
) -> None:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    payload["worker_signature"] = signature
    if signature == {"worker_pubkey": "", "sig": ""}:
        placeholder = EvalExecutionProof.model_validate(payload)
        rebound = rebind_worker_signature(
            placeholder.to_execution_proof(), signer=_Signer(), unit_id="eval-run-001"
        )
        assert rebound.attestation == placeholder.to_execution_proof().attestation
    else:
        with pytest.raises(ValidationError):
            EvalExecutionProof.model_validate(payload)
        original = EvalExecutionProof.model_validate(POSITIVE["execution_proof"])
        forged = original.to_execution_proof().model_copy(
            update={"worker_signature": WorkerSignature(**signature)}
        )
        with pytest.raises(AssignmentExecutionError):
            rebind_worker_signature(forged, signer=_Signer(), unit_id="eval-run-001")


@pytest.mark.parametrize(
    "path",
    [
        "provider",
        "worker_signature.worker_pubkey",
        "worker_signature.sig",
        "attestation.tdx_quote",
        "attestation.event_log",
        "attestation.report_data",
        "attestation.measurement",
        "attestation.vm_config",
        "attestation.event_log.0.imr",
        "attestation.event_log.0.event_type",
        "attestation.event_log.0.digest",
        "attestation.event_log.0.event",
        "attestation.event_log.0.event_payload",
        "attestation.vm_config.vcpu",
        "attestation.vm_config.memory_mb",
        "attestation.vm_config.os_image_hash",
    ],
)
def test_eval_execution_proof_requires_every_nested_member(path: str) -> None:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    *parents, leaf = path.split(".")
    cursor: Any = payload
    for parent in parents:
        cursor = cursor[int(parent)] if parent.isdigit() else cursor[parent]
    cursor.pop(int(leaf) if leaf.isdigit() else leaf)
    with pytest.raises(ValidationError):
        EvalExecutionProof.model_validate(payload)


def test_eval_execution_proof_accepts_required_nullable_vm_image_hash() -> None:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    payload["attestation"]["vm_config"]["os_image_hash"] = None
    parsed = EvalExecutionProof.model_validate(payload)
    assert parsed.attestation.vm_config.os_image_hash is None


@pytest.mark.parametrize(
    "path",
    [
        "attestation.measurement.mrtd",
        "attestation.measurement.rtmr0",
        "attestation.measurement.rtmr1",
        "attestation.measurement.rtmr2",
        "attestation.measurement.rtmr3",
        "attestation.measurement.compose_hash",
        "attestation.measurement.os_image_hash",
    ],
)
def test_eval_execution_proof_requires_every_measurement_member(path: str) -> None:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    *parents, leaf = path.split(".")
    cursor: Any = payload
    for parent in parents:
        cursor = cursor[parent]
    cursor.pop(leaf)
    with pytest.raises(ValidationError):
        EvalExecutionProof.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attestation.tdx_quote", "aa" * (EVAL_MAX_QUOTE_BYTES + 1)),
        ("attestation.event_log.0.event_payload", "aa" * (EVAL_MAX_PAYLOAD_BYTES + 1)),
        ("attestation.vm_config.os_image_hash", "a" * (EVAL_MAX_STRING_BYTES + 1)),
        ("image_digest", "registry.example/" + "a" * EVAL_MAX_STRING_BYTES),
    ],
)
def test_eval_execution_proof_rejects_one_over_scalar_bounds(
    field: str, value: str
) -> None:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    _set_path(payload, field, value)
    with pytest.raises(ValidationError):
        EvalExecutionProof.model_validate(payload)


def test_eval_execution_proof_rejects_one_over_event_collection_bound() -> None:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    event = copy.deepcopy(payload["attestation"]["event_log"][0])
    payload["attestation"]["event_log"] = [
        copy.deepcopy(event) for _ in range(EVAL_MAX_EVENT_LOG_ENTRIES + 1)
    ]
    with pytest.raises(ValidationError):
        EvalExecutionProof.model_validate(payload)


def test_eval_execution_proof_rejects_one_over_event_log_byte_bound() -> None:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    event = copy.deepcopy(payload["attestation"]["event_log"][0])
    event["event_payload"] = "aa" * 1024
    payload["attestation"]["event_log"] = [
        copy.deepcopy(event) for _ in range(EVAL_MAX_EVENT_LOG_ENTRIES)
    ]
    assert (
        len(
            json.dumps(
                payload["attestation"]["event_log"],
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        )
        > EVAL_MAX_EVENT_LOG_BYTES
    )
    with pytest.raises(ValidationError):
        EvalExecutionProof.model_validate(payload)


def test_eval_execution_proof_rejects_one_over_vm_config_byte_bound() -> None:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    payload["attestation"]["vm_config"]["os_image_hash"] = None
    padding = "a" * EVAL_MAX_VM_CONFIG_BYTES
    payload["attestation"]["vm_config"]["extra_padding"] = padding
    with pytest.raises(ValidationError):
        EvalExecutionProof.model_validate(payload)


@pytest.mark.parametrize("field", ["imr", "event_type"])
@pytest.mark.parametrize("value", [-1, EVAL_MAX_INTEGER + 1])
def test_eval_execution_proof_rejects_out_of_range_event_integers(
    field: str, value: int
) -> None:
    payload = copy.deepcopy(POSITIVE["execution_proof"])
    payload["attestation"]["event_log"][0][field] = value
    with pytest.raises(ValidationError):
        EvalExecutionProof.model_validate(payload)
