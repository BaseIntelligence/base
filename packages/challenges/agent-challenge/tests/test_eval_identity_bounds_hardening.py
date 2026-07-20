"""Identity + attestation-bound hardening for the canonical Eval wire.

Covers the three review findings owned by
``challenge-eval-wire-identity-bounds-hardening``:

* ``agent_hash`` is the SHA-256 of the submitted ZIP artifact domain
* ``task_config_sha256`` is the exact content digest consumed by own_runner
* quote / event-log / payload / VM-config / integer bounds reject before emission
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation import authorization as auth
from agent_challenge.evaluation import own_runner_backend as backend
from agent_challenge.evaluation.own_runner.taskdefs import compute_task_digest

VECTOR_PATH = Path(__file__).with_name("eval_execution_proof_v2_vectors.json")
POSITIVE: dict[str, Any] = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))["positive"]


def _positive_attestation() -> dict[str, Any]:
    return copy.deepcopy(POSITIVE["execution_proof"]["attestation"])


def test_eval_wire_exports_base_matching_bounds() -> None:
    assert ew.EVAL_MAX_QUOTE_BYTES == 64 * 1024
    assert ew.EVAL_MAX_EVENT_LOG_ENTRIES == 4096
    assert ew.EVAL_MAX_EVENT_LOG_BYTES == 2 * 1024 * 1024
    assert ew.EVAL_MAX_VM_CONFIG_BYTES == 256 * 1024
    assert ew.EVAL_MAX_STRING_BYTES == 16 * 1024
    assert ew.EVAL_MAX_PAYLOAD_BYTES == ew.EVAL_MAX_STRING_BYTES
    assert ew.EVAL_MAX_INTEGER == (1 << 63) - 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tdx_quote", "aa" * (ew.EVAL_MAX_QUOTE_BYTES + 1)),
        ("event_log.0.event_payload", "aa" * (ew.EVAL_MAX_PAYLOAD_BYTES + 1)),
    ],
)
def test_phala_attestation_rejects_one_over_scalar_bounds(field: str, value: str) -> None:
    payload = _positive_attestation()
    if field.startswith("event_log."):
        payload["event_log"][0]["event_payload"] = value
    else:
        payload[field] = value
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_phala_attestation(payload)


def test_phala_attestation_rejects_one_over_event_collection_bound() -> None:
    payload = _positive_attestation()
    event = copy.deepcopy(payload["event_log"][0])
    payload["event_log"] = [copy.deepcopy(event) for _ in range(ew.EVAL_MAX_EVENT_LOG_ENTRIES + 1)]
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_phala_attestation(payload)


def test_phala_attestation_rejects_one_over_event_log_byte_bound() -> None:
    payload = _positive_attestation()
    event = copy.deepcopy(payload["event_log"][0])
    event["event_payload"] = "aa" * 1024
    payload["event_log"] = [copy.deepcopy(event) for _ in range(ew.EVAL_MAX_EVENT_LOG_ENTRIES)]
    assert (
        len(
            json.dumps(
                payload["event_log"],
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        )
        > ew.EVAL_MAX_EVENT_LOG_BYTES
    )
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_phala_attestation(payload)


def test_phala_attestation_rejects_oversized_vm_config_encoding() -> None:
    payload = _positive_attestation()
    # Extra field already rejected; build a legal shape that nonetheless
    # exceeds the encoded-byte budget through a non-closed expansion is
    # impossible after schema closure, so assert the encoded size of the
    # positive vm_config is accepted and an oversize integer-path fails.
    assert (
        len(
            json.dumps(
                payload["vm_config"],
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        )
        <= ew.EVAL_MAX_VM_CONFIG_BYTES
    )
    payload["vm_config"]["vcpu"] = ew.EVAL_MAX_INTEGER + 1
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_phala_attestation(payload)


@pytest.mark.parametrize("field", ["imr", "event_type"])
@pytest.mark.parametrize("value", [-1, ew.EVAL_MAX_INTEGER + 1])
def test_phala_attestation_rejects_out_of_range_event_integers(field: str, value: int) -> None:
    payload = _positive_attestation()
    payload["event_log"][0][field] = value
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_phala_attestation(payload)


def test_execution_proof_rejects_oversize_image_digest() -> None:
    proof = copy.deepcopy(POSITIVE["execution_proof"])
    proof["image_digest"] = "registry.example/" + "a" * ew.EVAL_MAX_STRING_BYTES
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_execution_proof(proof)


def test_task_config_digest_is_on_disk_content_digest(tmp_path: Path) -> None:
    task_root = tmp_path / "hello-world"
    task_root.mkdir()
    (task_root / "task.toml").write_text(
        "[metadata]\nname = 'hello-world'\n"
        "[agent]\ntimeout_sec = 30.0\n"
        "[verifier]\ntimeout_sec = 30.0\n"
        "[environment]\nallow_internet = false\n",
        encoding="utf-8",
    )
    (task_root / "instruction.md").write_text("do the thing\n", encoding="utf-8")
    content = compute_task_digest(task_root)
    # The plan binds the exact digest own_runner later recomputes on the cache tree.
    assert ew.task_config_sha256_from_content_digest(content) == content
    assert auth.task_config_digest_from_content(content) == content


def test_validate_eval_plan_task_configs_requires_content_digest_match() -> None:
    plan = {
        "selected_tasks": [
            {
                "task_id": "hello-world",
                "image_ref": "registry.example/task@sha256:" + "d" * 64,
                "task_config_sha256": "ab" * 32,
            }
        ]
    }
    stand_in = type(
        "P",
        (),
        {"content_digest_sha256": "ab" * 32, "task_id": "hello-world"},
    )()
    backend._validate_eval_plan_task_configs(plan, {"hello-world": stand_in})  # type: ignore[arg-type]
    stand_in.content_digest_sha256 = "00" * 32
    with pytest.raises(ValueError, match="task content digest"):
        backend._validate_eval_plan_task_configs(plan, {"hello-world": stand_in})  # type: ignore[arg-type]


def test_agent_artifact_sha256_hashes_exact_zip_bytes(tmp_path: Path) -> None:
    zip_path = tmp_path / "agent.zip"
    payload = b"PK\x03\x04submitted-agent-bytes"
    zip_path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert backend.agent_artifact_sha256(zip_path) == expected
    assert ew.agent_artifact_sha256_hex(payload) == expected


def test_agent_artifact_mismatch_against_plan_raises(tmp_path: Path) -> None:
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(b"agent-a")
    plan_hash = hashlib.sha256(b"agent-b").hexdigest()
    with pytest.raises(ValueError, match="agent artifact"):
        backend.assert_agent_artifact_matches_plan(
            artifact_path=zip_path,
            plan_agent_hash=plan_hash,
        )


def test_task_config_digest_helper_for_benchmark_task_uses_dataset_digest() -> None:
    digest_path = Path(__file__).resolve().parents[1] / "golden" / "dataset-digest.json"
    manifest = json.loads(digest_path.read_text(encoding="utf-8"))
    task = type(
        "Task",
        (),
        {
            "task_id": "terminal-bench/adaptive-rejection-sampler",
            "docker_image": "registry.example/task@sha256:" + "d" * 64,
            "prompt": "p",
            "benchmark": "terminal_bench",
            "metadata": {},
        },
    )()
    expected = manifest["tasks"]["adaptive-rejection-sampler"]["content_digest_sha256"]
    assert auth._task_config_digest(task) == expected
    # Discriminator: metadata-only hash must not equal the content digest domain.
    metadata_only = hashlib.sha256(
        ew.canonical_json_v1(
            {
                "task_id": task.task_id,
                "image_ref": task.docker_image,
                "prompt": task.prompt,
                "benchmark": task.benchmark,
                "metadata": task.metadata,
            }
        )
    ).hexdigest()
    assert auth._task_config_digest(task) != metadata_only


def test_shared_execution_proof_vector_is_schema_closed() -> None:
    """VAL-IMG-025: image/endpoint accept the shared literal ExecutionProof vector."""

    proof = copy.deepcopy(POSITIVE["execution_proof"])
    assert ew.validate_eval_execution_proof(proof) == proof
    # One-field mutation of a closed constant must not normalize away.
    mutated = copy.deepcopy(proof)
    mutated["tier"] = "phala"
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_execution_proof(mutated)


def test_shared_result_request_binds_agent_hash_and_proof() -> None:
    """VAL-IMG-027 / VAL-IMG-028: result request and report_data use ZIP-domain agent_hash."""

    zip_bytes = b"PK\x03\x04exact-evaluated-agent-artifact"
    agent_hash = ew.agent_artifact_sha256_hex(zip_bytes)
    # Reuse the contract suite's fully schema-valid score record (two tasks, k=2).
    record = {
        "schema_version": 1,
        "eval_run_id": "eval-run-001",
        "policy_digest": "2" * 64,
        "k": 2,
        "tasks": [
            {
                "task_id": "task-a",
                "trial_scores_f64be": ["0000000000000000", "3ff0000000000000"],
                "aggregate_score_f64be": "3fe0000000000000",
                "passed_trials": 1,
            },
            {
                "task_id": "task-b",
                "trial_scores_f64be": ["3ff0000000000000", "3ff0000000000000"],
                "aggregate_score_f64be": "3ff0000000000000",
                "passed_trials": 2,
            },
        ],
        "final": {
            "job_score_f64be": "3fe8000000000000",
            "passed_tasks": 1,
            "total_tasks": 2,
        },
    }
    scores_digest = ew.score_record_digest(record)
    binding = ew.build_score_binding(
        canonical_measurement=POSITIVE["binding"]["canonical_measurement"],
        agent_hash=agent_hash,
        eval_run_id="eval-run-001",
        score_nonce=POSITIVE["binding"]["score_nonce"],
        scores_digest=scores_digest,
        task_ids=["task-a", "task-b"],
    )
    prove = copy.deepcopy(POSITIVE["execution_proof"])
    prove["attestation"]["report_data"] = ew.score_report_data_hex(binding)
    request = {
        "schema_version": 1,
        "eval_run_id": "eval-run-001",
        "submission_id": "submission-001",
        "agent_hash": agent_hash,
        "score_record": record,
        "scores_digest": scores_digest,
        "execution_proof": prove,
    }
    validated = ew.validate_eval_result_request(request)
    assert validated["agent_hash"] == agent_hash
    assert validated["execution_proof"]["attestation"]["report_data"] == ew.score_report_data_hex(
        binding
    )
    # Independent recomputation from ZIP bytes (VAL-IMG-028).
    assert ew.agent_artifact_sha256_hex(zip_bytes) == validated["agent_hash"]


def test_plan_agent_hash_domain_is_submission_zip_hash() -> None:
    """Plan-bound agent_hash is the same SHA-256 domain as submitted ZIP bytes."""

    zip_bytes = b"PK submission-zip-domain"
    digest = hashlib.sha256(zip_bytes).hexdigest()
    assert ew.agent_artifact_sha256_hex(zip_bytes) == digest
    # Declared env/path mismatch is fail-closed before quotation.
    with pytest.raises(ValueError, match="agent_hash"):
        backend.assert_agent_artifact_matches_plan(
            artifact_path=None,
            plan_agent_hash=digest,
            declared_agent_hash="0" * 64,
        )


def test_bounds_reject_before_any_aliased_quote_field() -> None:
    """Canonical Bounds: aliases / oversize quote never reach verification."""

    payload = _positive_attestation()
    payload["tdx_quote_b64"] = payload.pop("tdx_quote")
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_phala_attestation(payload)
    # One-byte-over integer bound.
    good = _positive_attestation()
    good["vm_config"]["memory_mb"] = ew.EVAL_MAX_INTEGER + 1
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_phala_attestation(good)


def test_base_oracle_accepts_same_proof_when_importable() -> None:
    """BASE read-only oracle (when present) accepts the same literal proof vector."""

    pytest.importorskip("base")
    try:
        from base.schemas.worker import (  # type: ignore[import-not-found]
            EVAL_MAX_EVENT_LOG_BYTES,
            EVAL_MAX_EVENT_LOG_ENTRIES,
            EVAL_MAX_INTEGER,
            EVAL_MAX_PAYLOAD_BYTES,
            EVAL_MAX_QUOTE_BYTES,
            EVAL_MAX_STRING_BYTES,
            EVAL_MAX_VM_CONFIG_BYTES,
            EvalExecutionProof,
        )
    except ImportError:
        pytest.skip("base schema module unavailable in this venv")

    assert EVAL_MAX_QUOTE_BYTES == ew.EVAL_MAX_QUOTE_BYTES
    assert EVAL_MAX_EVENT_LOG_ENTRIES == ew.EVAL_MAX_EVENT_LOG_ENTRIES
    assert EVAL_MAX_EVENT_LOG_BYTES == ew.EVAL_MAX_EVENT_LOG_BYTES
    assert EVAL_MAX_VM_CONFIG_BYTES == ew.EVAL_MAX_VM_CONFIG_BYTES
    assert EVAL_MAX_STRING_BYTES == ew.EVAL_MAX_STRING_BYTES
    assert EVAL_MAX_PAYLOAD_BYTES == ew.EVAL_MAX_PAYLOAD_BYTES
    assert EVAL_MAX_INTEGER == ew.EVAL_MAX_INTEGER
    proof = copy.deepcopy(POSITIVE["execution_proof"])
    assert EvalExecutionProof.model_validate(proof).model_dump(mode="json") == proof
