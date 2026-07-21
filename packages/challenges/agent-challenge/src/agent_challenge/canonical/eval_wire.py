"""Strict, canonical wire primitives for attested Eval results.

This module is intentionally dependency-light because the canonical CVM image
imports it while emitting ``BASE_BENCHMARK_RESULT``.  It owns the schema-closed
Eval v1 records and schema-version-2 score/key ``report_data`` bindings that
must remain byte-identical to the completed BASE implementation.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from agent_challenge.canonical.key_release_endpoint import parse_key_release_authority
from agent_challenge.review.canonical import (
    CanonicalJsonError,
)
from agent_challenge.review.canonical import (
    canonical_json_v1 as _canonical_json_v1,
)

SCORE_DOMAIN = "base-agent-challenge-v1"
KEY_RELEASE_DOMAIN = "base-agent-challenge-keyrelease-v1"
REPORT_DATA_BYTES = 64

# Transport / allocation bounds must match BASE ``src/base/schemas/worker.py``
# so the image emitter, challenge endpoint, CLI, and BASE oracle reject the
# same oversized or oversize-field vectors before any verification work.
EVAL_MAX_QUOTE_BYTES = 64 * 1024
EVAL_MAX_EVENT_LOG_ENTRIES = 4096
EVAL_MAX_EVENT_LOG_BYTES = 2 * 1024 * 1024
EVAL_MAX_VM_CONFIG_BYTES = 256 * 1024
EVAL_MAX_STRING_BYTES = 16 * 1024
EVAL_MAX_PAYLOAD_BYTES = EVAL_MAX_STRING_BYTES
EVAL_MAX_INTEGER = (1 << 63) - 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REGISTER_RE = re.compile(r"^[0-9a-f]{96}$")
_REPORT_DATA_RE = re.compile(r"^[0-9a-f]{128}$")
_F64_RE = re.compile(r"^[0-9a-f]{16}$")
_EVEN_HEX_RE = re.compile(r"^(?:[0-9a-f]{2})*$")
_NONEMPTY_EVEN_HEX_RE = re.compile(r"^(?:[0-9a-f]{2})+$")
_IMAGE_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")

_MEASUREMENT_FIELDS = (
    "mrtd",
    "rtmr0",
    "rtmr1",
    "rtmr2",
    "compose_hash",
    "os_image_hash",
)


class EvalWireError(ValueError):
    """Raised when an Eval wire object is not canonical and schema closed."""


def canonical_json_v1(value: Any) -> bytes:
    """Serialize strict Eval data using the shared UTF-8/NFC JSON profile."""

    try:
        return _canonical_json_v1(value)
    except CanonicalJsonError as exc:
        raise EvalWireError(str(exc)) from exc


def _object(value: Any, name: str, fields: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvalWireError(f"{name} must be an object")
    actual = set(value)
    expected = set(fields)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise EvalWireError(f"{name} has invalid fields: missing={missing}, unknown={unknown}")
    return dict(value)


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = EVAL_MAX_INTEGER,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum or value > maximum:
        raise EvalWireError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not (1 <= len(value) <= 128):
        raise EvalWireError(f"{name} must be a 1-128 character visible ASCII id")
    if any(not ("!" <= character <= "~") for character in value):
        raise EvalWireError(f"{name} must be a visible ASCII id")
    return value


def _bounded_string(value: Any, name: str, *, maximum: int = EVAL_MAX_STRING_BYTES) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise EvalWireError(f"{name} exceeds its string bound")
    return value


def _hex(value: Any, name: str, pattern: re.Pattern[str], *, max_chars: int | None = None) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise EvalWireError(f"{name} has an invalid canonical hexadecimal encoding")
    if max_chars is not None and len(value) > max_chars:
        raise EvalWireError(f"{name} exceeds its string bound")
    return value


def _sha256(value: Any, name: str) -> str:
    return _hex(value, name, _SHA256_RE)


def _register(value: Any, name: str) -> str:
    return _hex(value, name, _REGISTER_RE)


def _report_data(value: Any, name: str) -> str:
    return _hex(value, name, _REPORT_DATA_RE)


def _image(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IMAGE_RE.fullmatch(value) is None:
        raise EvalWireError(f"{name} must be a digest-pinned image reference")
    if len(value) > EVAL_MAX_STRING_BYTES:
        raise EvalWireError(f"{name} exceeds its string bound")
    return value


def agent_artifact_sha256_hex(zip_bytes: bytes) -> str:
    """SHA-256 of the exact submitted ZIP artifact bytes (submission agent_hash)."""

    if not isinstance(zip_bytes, (bytes, bytearray)):
        raise EvalWireError("agent artifact must be raw ZIP bytes")
    return hashlib.sha256(bytes(zip_bytes)).hexdigest()


def task_config_sha256_from_content_digest(content_digest_sha256: str) -> str:
    """Identity for the plan-bound task config digest domain.

    Eval plan ``task_config_sha256`` is the frozen on-disk task-tree content digest
    (every regular file under the task root) that own_runner recomputes and
    verifies against ``dataset-digest.json``.
    """

    return _sha256(content_digest_sha256, "task_config_sha256")


def canonical_task_ids(task_ids: Iterable[str]) -> list[str]:
    """Canonicalize producer task IDs, refusing duplicate task identity."""

    values = [_id(task_id, "task_id") for task_id in task_ids]
    ordered = sorted(values)
    if len(ordered) != len(set(ordered)):
        raise EvalWireError("task_ids must be unique")
    return ordered


def _strict_task_ids(task_ids: Any) -> list[str]:
    if not isinstance(task_ids, list):
        raise EvalWireError("task_ids must be an array")
    values = [_id(task_id, "task_ids[]") for task_id in task_ids]
    if values != sorted(values) or len(values) != len(set(values)):
        raise EvalWireError("task_ids must be sorted and unique")
    return values


def _validate_canonical_measurement(value: Any) -> dict[str, str]:
    """Validate the static six-field canonical measurement."""

    data = _object(value, "canonical_measurement", _MEASUREMENT_FIELDS)
    result: dict[str, str] = {}
    for field in ("mrtd", "rtmr0", "rtmr1", "rtmr2"):
        result[field] = _register(data[field], f"canonical_measurement.{field}")
    for field in ("compose_hash", "os_image_hash"):
        result[field] = _sha256(data[field], f"canonical_measurement.{field}")
    return result


def canonical_measurement(value: Any) -> dict[str, str]:
    """Validate the static six-field canonical measurement."""

    return _validate_canonical_measurement(value)


def build_score_binding(
    *,
    canonical_measurement: Mapping[str, Any],
    agent_hash: str,
    eval_run_id: str,
    score_nonce: str,
    scores_digest: str,
    task_ids: Sequence[str],
) -> dict[str, Any]:
    """Build the exact schema-version-2 score quote binding.

    This is a strict wire constructor.  Call :func:`canonical_task_ids` first
    when a producer needs to normalize set-equivalent task inputs.
    """

    tasks = _strict_task_ids(list(task_ids))
    return {
        "agent_hash": _sha256(agent_hash, "agent_hash"),
        "canonical_measurement": _validate_canonical_measurement(canonical_measurement),
        "domain": SCORE_DOMAIN,
        "eval_run_id": _id(eval_run_id, "eval_run_id"),
        "schema_version": 2,
        "score_nonce": _id(score_nonce, "score_nonce"),
        "scores_digest": _sha256(scores_digest, "scores_digest"),
        "task_ids": tasks,
    }


def validate_score_binding(value: Any) -> dict[str, Any]:
    data = _object(
        value,
        "score_binding",
        (
            "agent_hash",
            "canonical_measurement",
            "domain",
            "eval_run_id",
            "schema_version",
            "score_nonce",
            "scores_digest",
            "task_ids",
        ),
    )
    if data["domain"] != SCORE_DOMAIN or data["schema_version"] != 2:
        raise EvalWireError("score_binding has an invalid domain or schema version")
    return build_score_binding(
        canonical_measurement=data["canonical_measurement"],
        agent_hash=data["agent_hash"],
        eval_run_id=data["eval_run_id"],
        score_nonce=data["score_nonce"],
        scores_digest=data["scores_digest"],
        task_ids=data["task_ids"],
    )


def score_report_data_hex(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 score-binding digest in the 64-byte TDX field."""

    binding = validate_score_binding(value)
    return hashlib.sha256(canonical_json_v1(binding)).digest().ljust(REPORT_DATA_BYTES, b"\0").hex()


def key_release_report_data_hex(
    *,
    eval_run_id: str,
    key_release_nonce: str,
    ra_tls_spki_digest: str,
) -> str:
    """Build the separate schema-version-2 key-release report-data field."""

    binding = {
        "domain": KEY_RELEASE_DOMAIN,
        "eval_run_id": _id(eval_run_id, "eval_run_id"),
        "key_release_nonce": _id(key_release_nonce, "key_release_nonce"),
        "ra_tls_spki_digest": _sha256(ra_tls_spki_digest, "ra_tls_spki_digest"),
        "schema_version": 2,
    }
    return hashlib.sha256(canonical_json_v1(binding)).digest().ljust(REPORT_DATA_BYTES, b"\0").hex()


def decode_score_f64be(value: Any) -> float:
    """Decode the only permitted score scalar, finite positive binary64 in [0,1]."""

    encoded = _hex(value, "score_f64be", _F64_RE)
    raw = bytes.fromhex(encoded)
    if raw[0] & 0x80 and raw == b"\x80" + b"\0" * 7:
        raise EvalWireError("negative zero is not a canonical score")
    score = struct.unpack(">d", raw)[0]
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise EvalWireError("score must be finite and in [0, 1]")
    return score


def encode_score_f64be(value: Any) -> str:
    """Encode a finite non-negative score exactly as binary64 big-endian bits."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EvalWireError("score must be a number")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise EvalWireError("score must be finite and in [0, 1]")
    if score == 0.0 and math.copysign(1.0, score) < 0:
        raise EvalWireError("negative zero is not canonical")
    return struct.pack(">d", score).hex()


def _validate_scoring_policy(value: Any) -> dict[str, Any]:
    data = _object(
        value,
        "scoring_policy",
        (
            "schema_version",
            "per_task_aggregation",
            "keep_policy",
            "drop_lowest_n",
            "threshold_f64be",
        ),
    )
    if data["schema_version"] != 1:
        raise EvalWireError("scoring_policy schema_version must be 1")
    if data["per_task_aggregation"] not in {"mean", "best_of_k"}:
        raise EvalWireError("invalid per_task_aggregation")
    if data["keep_policy"] not in {"off", "drop_lowest_n", "threshold_band"}:
        raise EvalWireError("invalid keep_policy")
    drop_lowest_n = _integer(data["drop_lowest_n"], "drop_lowest_n")
    threshold = data["threshold_f64be"]
    if data["keep_policy"] == "threshold_band":
        decode_score_f64be(threshold)
        if drop_lowest_n != 0:
            raise EvalWireError("threshold_band requires neutral drop_lowest_n")
    elif threshold is not None:
        raise EvalWireError("threshold_f64be must be null outside threshold_band")
    if data["keep_policy"] != "drop_lowest_n" and drop_lowest_n != 0:
        raise EvalWireError("drop_lowest_n must be neutral outside drop_lowest_n")
    return {
        "schema_version": 1,
        "per_task_aggregation": data["per_task_aggregation"],
        "keep_policy": data["keep_policy"],
        "drop_lowest_n": drop_lowest_n,
        "threshold_f64be": threshold,
    }


def validate_scoring_policy(value: Any) -> dict[str, Any]:
    return _validate_scoring_policy(value)


def scoring_policy_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_v1(_validate_scoring_policy(value))).hexdigest()


def validate_eval_plan(value: Any) -> dict[str, Any]:
    """Validate the immutable Eval plan v1 consumed by the image and endpoint."""

    data = _object(
        value,
        "eval_plan",
        (
            "schema_version",
            "eval_run_id",
            "submission_id",
            "submission_version",
            "authorizing_review_digest",
            "agent_hash",
            "selected_tasks",
            "k",
            "scoring_policy",
            "scoring_policy_digest",
            "eval_app",
            "key_release_endpoint",
            "result_endpoint",
            "key_release_nonce",
            "score_nonce",
            "run_token_sha256",
            "issued_at_ms",
            "expires_at_ms",
        ),
    )
    if data["schema_version"] != 1:
        raise EvalWireError("eval_plan schema_version must be 1")
    eval_run_id = _id(data["eval_run_id"], "eval_plan.eval_run_id")
    submission_id = _id(data["submission_id"], "eval_plan.submission_id")
    submission_version = _integer(
        data["submission_version"], "eval_plan.submission_version", minimum=1
    )
    review_digest = _sha256(data["authorizing_review_digest"], "authorizing_review_digest")
    agent_hash = _sha256(data["agent_hash"], "agent_hash")
    k = _integer(data["k"], "eval_plan.k", minimum=1)
    policy = _validate_scoring_policy(data["scoring_policy"])
    policy_digest = _sha256(data["scoring_policy_digest"], "scoring_policy_digest")
    if policy_digest != scoring_policy_digest(policy):
        raise EvalWireError("scoring_policy_digest does not match scoring_policy")
    if not isinstance(data["selected_tasks"], list) or not data["selected_tasks"]:
        raise EvalWireError("selected_tasks must be a non-empty array")
    selected_tasks: list[dict[str, str]] = []
    previous_task_id = ""
    for item in data["selected_tasks"]:
        task = _object(item, "selected_tasks[]", ("task_id", "image_ref", "task_config_sha256"))
        task_id = _id(task["task_id"], "selected_tasks[].task_id")
        if previous_task_id >= task_id:
            raise EvalWireError("selected_tasks must be sorted and unique")
        previous_task_id = task_id
        selected_tasks.append(
            {
                "task_id": task_id,
                "image_ref": _image(task["image_ref"], "selected_tasks[].image_ref"),
                "task_config_sha256": _sha256(
                    task["task_config_sha256"], "selected_tasks[].task_config_sha256"
                ),
            }
        )
    app = _object(
        data["eval_app"],
        "eval_app",
        (
            "image_ref",
            "compose_hash",
            "app_identity",
            "kms_key_algorithm",
            "kms_public_key_hex",
            "kms_public_key_sha256",
            "measurement",
        ),
    )
    app_measurement = _object(
        app["measurement"],
        "eval_app.measurement",
        (
            "mrtd",
            "rtmr0",
            "rtmr1",
            "rtmr2",
            "os_image_hash",
            "key_provider",
            "vm_shape",
        ),
    )
    app_measurement_valid = {
        "mrtd": _register(app_measurement["mrtd"], "eval_app.measurement.mrtd"),
        "rtmr0": _register(app_measurement["rtmr0"], "eval_app.measurement.rtmr0"),
        "rtmr1": _register(app_measurement["rtmr1"], "eval_app.measurement.rtmr1"),
        "rtmr2": _register(app_measurement["rtmr2"], "eval_app.measurement.rtmr2"),
        "os_image_hash": _sha256(
            app_measurement["os_image_hash"], "eval_app.measurement.os_image_hash"
        ),
        "key_provider": _id(app_measurement["key_provider"], "eval_app.measurement.key_provider"),
        "vm_shape": _id(app_measurement["vm_shape"], "eval_app.measurement.vm_shape"),
    }
    kms_key_algorithm = _id(app["kms_key_algorithm"], "eval_app.kms_key_algorithm")
    if kms_key_algorithm != "x25519":
        raise EvalWireError("eval_app.kms_key_algorithm must be x25519")
    kms_public_key_hex = _sha256(app["kms_public_key_hex"], "eval_app.kms_public_key_hex")
    kms_public_key_sha256 = _sha256(app["kms_public_key_sha256"], "eval_app.kms_public_key_sha256")
    if hashlib.sha256(bytes.fromhex(kms_public_key_hex)).hexdigest() != kms_public_key_sha256:
        raise EvalWireError("kms_public_key_sha256 does not match kms_public_key_hex")
    key_release_nonce = _id(data["key_release_nonce"], "key_release_nonce")
    score_nonce = _id(data["score_nonce"], "score_nonce")
    if key_release_nonce == score_nonce:
        raise EvalWireError("key_release_nonce and score_nonce must be distinct")
    result_endpoint = f"/evaluation/v1/runs/{eval_run_id}/result"
    if data["result_endpoint"] != result_endpoint:
        raise EvalWireError("result_endpoint does not target this eval_run_id")
    if (
        not isinstance(data["key_release_endpoint"], str)
        or not data["key_release_endpoint"]
        or len(data["key_release_endpoint"]) > 16_384
    ):
        raise EvalWireError("key_release_endpoint must be a bounded non-empty string")
    # VAL-ACLOCK-008/010: plan KR is validator RA-TLS / host:port authority only.
    # Free HTTP(S) URLs (including measure-time compose pin placeholders) are not
    # accepted as the signed plan trust root.
    key_release_endpoint = data["key_release_endpoint"].strip()
    if parse_key_release_authority(key_release_endpoint) is None:
        raise EvalWireError(
            "key_release_endpoint must be a validator RA-TLS/authority form "
            "(host:port or ratls|tls|tcp://host:port); free HTTP(S) KR URLs are rejected"
        )
    issued_at_ms = _integer(data["issued_at_ms"], "issued_at_ms")
    expires_at_ms = _integer(data["expires_at_ms"], "expires_at_ms")
    if expires_at_ms <= issued_at_ms:
        raise EvalWireError("eval_plan expiry must be after issue time")
    return {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "submission_id": submission_id,
        "submission_version": submission_version,
        "authorizing_review_digest": review_digest,
        "agent_hash": agent_hash,
        "selected_tasks": selected_tasks,
        "k": k,
        "scoring_policy": policy,
        "scoring_policy_digest": policy_digest,
        "eval_app": {
            "image_ref": _image(app["image_ref"], "eval_app.image_ref"),
            "compose_hash": _sha256(app["compose_hash"], "eval_app.compose_hash"),
            "app_identity": _id(app["app_identity"], "eval_app.app_identity"),
            "kms_key_algorithm": kms_key_algorithm,
            "kms_public_key_hex": kms_public_key_hex,
            "kms_public_key_sha256": kms_public_key_sha256,
            "measurement": app_measurement_valid,
        },
        "key_release_endpoint": key_release_endpoint,
        "result_endpoint": result_endpoint,
        "key_release_nonce": key_release_nonce,
        "score_nonce": score_nonce,
        "run_token_sha256": _sha256(data["run_token_sha256"], "run_token_sha256"),
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
    }


def _aggregate(trials: list[float], policy: Mapping[str, Any]) -> float:
    if policy["per_task_aggregation"] == "best_of_k":
        return max(trials)
    return sum(trials) / len(trials)


def _job_score(aggregates: list[float], policy: Mapping[str, Any]) -> float:
    kept = list(aggregates)
    if policy["keep_policy"] == "drop_lowest_n":
        count = min(policy["drop_lowest_n"], max(0, len(kept) - 1))
        dropped = set(sorted(range(len(kept)), key=lambda index: (kept[index], index))[:count])
        kept = [score for index, score in enumerate(kept) if index not in dropped]
    elif policy["keep_policy"] == "threshold_band":
        threshold = decode_score_f64be(policy["threshold_f64be"])
        kept = [score for score in kept if score >= threshold]
    return sum(kept) / len(kept) if kept else 0.0


def _score_record_shape(value: Any) -> dict[str, Any]:
    data = _object(
        value,
        "score_record",
        ("schema_version", "eval_run_id", "policy_digest", "k", "tasks", "final"),
    )
    if data["schema_version"] != 1:
        raise EvalWireError("score_record schema_version must be 1")
    _id(data["eval_run_id"], "score_record.eval_run_id")
    _sha256(data["policy_digest"], "score_record.policy_digest")
    _integer(data["k"], "score_record.k", minimum=1)
    if not isinstance(data["tasks"], list) or not data["tasks"]:
        raise EvalWireError("score_record.tasks must be a non-empty array")
    previous = ""
    tasks: list[dict[str, Any]] = []
    for item in data["tasks"]:
        task = _object(
            item,
            "score_record.tasks[]",
            ("task_id", "trial_scores_f64be", "aggregate_score_f64be", "passed_trials"),
        )
        task_id = _id(task["task_id"], "score_record.tasks[].task_id")
        if previous >= task_id:
            raise EvalWireError("score_record.tasks must be sorted and unique")
        previous = task_id
        if not isinstance(task["trial_scores_f64be"], list):
            raise EvalWireError("trial_scores_f64be must be an array")
        trials = [decode_score_f64be(entry) for entry in task["trial_scores_f64be"]]
        aggregate = decode_score_f64be(task["aggregate_score_f64be"])
        passed_trials = _integer(task["passed_trials"], "passed_trials")
        tasks.append(
            {
                "task_id": task_id,
                "trial_scores_f64be": list(task["trial_scores_f64be"]),
                "aggregate_score_f64be": task["aggregate_score_f64be"],
                "passed_trials": passed_trials,
                "_trials": trials,
                "_aggregate": aggregate,
            }
        )
    final = _object(
        data["final"],
        "score_record.final",
        ("job_score_f64be", "passed_tasks", "total_tasks"),
    )
    output = dict(data)
    output["tasks"] = tasks
    output["final"] = {
        "job_score_f64be": _hex(final["job_score_f64be"], "job_score_f64be", _F64_RE),
        "passed_tasks": _integer(final["passed_tasks"], "passed_tasks"),
        "total_tasks": _integer(final["total_tasks"], "total_tasks"),
    }
    decode_score_f64be(output["final"]["job_score_f64be"])
    return output


def validate_canonical_score_record(
    value: Any,
    *,
    scoring_policy: Any,
    expected_eval_run_id: str,
    expected_task_ids: Sequence[str],
    expected_k: int,
) -> dict[str, Any]:
    """Validate every score derivation against immutable policy and plan inputs."""

    record = _score_record_shape(value)
    policy = _validate_scoring_policy(scoring_policy)
    if record["eval_run_id"] != _id(expected_eval_run_id, "expected_eval_run_id"):
        raise EvalWireError("score_record eval_run_id does not match the immutable plan")
    if record["policy_digest"] != scoring_policy_digest(policy):
        raise EvalWireError("score_record policy_digest does not match immutable policy")
    if record["k"] != _integer(expected_k, "expected_k", minimum=1):
        raise EvalWireError("score_record k does not match immutable plan")
    expected_tasks = _strict_task_ids(list(expected_task_ids))
    if [task["task_id"] for task in record["tasks"]] != expected_tasks:
        raise EvalWireError("score_record tasks do not match immutable selected tasks")
    aggregates: list[float] = []
    for task in record["tasks"]:
        trials = task["_trials"]
        if len(trials) != record["k"]:
            raise EvalWireError("task trial count does not match k")
        aggregate = _aggregate(trials, policy)
        if task["aggregate_score_f64be"] != encode_score_f64be(aggregate):
            raise EvalWireError("task aggregate does not match ordered trial scores")
        if task["passed_trials"] != sum(score == 1.0 for score in trials):
            raise EvalWireError("passed_trials does not match ordered trial scores")
        aggregates.append(aggregate)
    final = record["final"]
    if final["total_tasks"] != len(expected_tasks):
        raise EvalWireError("final total_tasks must cover the full selected task set")
    if final["passed_tasks"] != sum(score == 1.0 for score in aggregates):
        raise EvalWireError("final passed_tasks must cover the full selected task set")
    if final["job_score_f64be"] != encode_score_f64be(_job_score(aggregates, policy)):
        raise EvalWireError("final job score does not match immutable policy")
    return _public_score_record(record)


def build_canonical_score_record(
    *,
    eval_run_id: str,
    policy: Any,
    trial_scores_by_task: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Build the only canonical score record from ordered evaluated trial scores."""

    normalized_policy = _validate_scoring_policy(policy)
    if not isinstance(trial_scores_by_task, Mapping) or not trial_scores_by_task:
        raise EvalWireError("trial_scores_by_task must contain at least one selected task")
    task_ids = canonical_task_ids(trial_scores_by_task)
    task_trials: dict[str, list[float]] = {}
    expected_k: int | None = None
    for task_id in task_ids:
        raw_trials = trial_scores_by_task[task_id]
        if not isinstance(raw_trials, Sequence) or isinstance(raw_trials, str) or not raw_trials:
            raise EvalWireError("each task must have at least one ordered trial score")
        trials = [decode_score_f64be(encode_score_f64be(value)) for value in raw_trials]
        if expected_k is None:
            expected_k = len(trials)
        elif len(trials) != expected_k:
            raise EvalWireError("every selected task must have exactly the same k trials")
        task_trials[task_id] = trials

    assert expected_k is not None
    tasks: list[dict[str, Any]] = []
    aggregates: list[float] = []
    for task_id in task_ids:
        trials = task_trials[task_id]
        aggregate = _aggregate(trials, normalized_policy)
        aggregates.append(aggregate)
        tasks.append(
            {
                "task_id": task_id,
                "trial_scores_f64be": [encode_score_f64be(score) for score in trials],
                "aggregate_score_f64be": encode_score_f64be(aggregate),
                "passed_trials": sum(score == 1.0 for score in trials),
            }
        )
    return {
        "schema_version": 1,
        "eval_run_id": _id(eval_run_id, "eval_run_id"),
        "policy_digest": scoring_policy_digest(normalized_policy),
        "k": expected_k,
        "tasks": tasks,
        "final": {
            "job_score_f64be": encode_score_f64be(_job_score(aggregates, normalized_policy)),
            "passed_tasks": sum(score == 1.0 for score in aggregates),
            "total_tasks": len(task_ids),
        },
    }


def _public_score_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": record["schema_version"],
        "eval_run_id": record["eval_run_id"],
        "policy_digest": record["policy_digest"],
        "k": record["k"],
        "tasks": [
            {
                "task_id": task["task_id"],
                "trial_scores_f64be": task["trial_scores_f64be"],
                "aggregate_score_f64be": task["aggregate_score_f64be"],
                "passed_trials": task["passed_trials"],
            }
            for task in record["tasks"]
        ],
        "final": dict(record["final"]),
    }


def score_record_digest(value: Any) -> str:
    """Hash a closed canonical score record after structural validation."""

    record = _public_score_record(_score_record_shape(value))
    return hashlib.sha256(canonical_json_v1(record)).hexdigest()


def validate_eval_phala_attestation(value: Any) -> dict[str, Any]:
    data = _object(
        value,
        "attestation",
        ("tdx_quote", "event_log", "report_data", "measurement", "vm_config"),
    )
    # Bound nested transports before retaining attacker-controlled data or any
    # later verification/allocation (matches BASE EvalPhalaAttestation).
    if not isinstance(data["event_log"], list):
        raise EvalWireError("attestation.event_log must be an array")
    if len(data["event_log"]) > EVAL_MAX_EVENT_LOG_ENTRIES:
        raise EvalWireError("event_log exceeds its entry bound")
    encoded_event_bytes = 2 + max(0, len(data["event_log"]) - 1)
    for event in data["event_log"]:
        if not isinstance(event, Mapping):
            raise EvalWireError("event_log[] must be an object")
        if set(event) != {"imr", "event_type", "digest", "event", "event_payload"}:
            raise EvalWireError("event_log entry has invalid fields")
        for field, limit in (
            ("digest", 96),
            ("event", EVAL_MAX_STRING_BYTES),
            ("event_payload", EVAL_MAX_PAYLOAD_BYTES),
        ):
            field_value = event.get(field)
            if isinstance(field_value, str) and len(field_value) > limit:
                raise EvalWireError(f"event_log.{field} exceeds its string bound")
        try:
            import json

            encoded_event_bytes += len(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise EvalWireError("event_log is not encodable") from exc
        if encoded_event_bytes > EVAL_MAX_EVENT_LOG_BYTES:
            raise EvalWireError("event_log exceeds its byte bound")
    if not isinstance(data["vm_config"], Mapping):
        raise EvalWireError("attestation.vm_config must be an object")
    if set(data["vm_config"]) != {"vcpu", "memory_mb", "os_image_hash"}:
        raise EvalWireError("vm_config has invalid fields")
    try:
        import json

        encoded_vm = json.dumps(
            data["vm_config"],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EvalWireError("vm_config is not encodable") from exc
    if len(encoded_vm) > EVAL_MAX_VM_CONFIG_BYTES:
        raise EvalWireError("vm_config exceeds its byte bound")

    quote = _hex(
        data["tdx_quote"],
        "attestation.tdx_quote",
        _NONEMPTY_EVEN_HEX_RE,
        max_chars=2 * EVAL_MAX_QUOTE_BYTES,
    )
    report_data = _report_data(data["report_data"], "attestation.report_data")
    event_log: list[dict[str, Any]] = []
    for event in data["event_log"]:
        entry = _object(
            event,
            "event_log[]",
            ("imr", "event_type", "digest", "event", "event_payload"),
        )
        event_log.append(
            {
                "imr": _integer(entry["imr"], "event_log[].imr"),
                "event_type": _integer(entry["event_type"], "event_log[].event_type"),
                "digest": _register(entry["digest"], "event_log[].digest"),
                "event": _id(entry["event"], "event_log[].event"),
                "event_payload": _hex(
                    entry["event_payload"],
                    "event_log[].event_payload",
                    _EVEN_HEX_RE,
                    max_chars=EVAL_MAX_PAYLOAD_BYTES,
                ),
            }
        )
    measurement_data = _object(
        data["measurement"],
        "attestation.measurement",
        ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3", "compose_hash", "os_image_hash"),
    )
    measurement = {
        **canonical_measurement({field: measurement_data[field] for field in _MEASUREMENT_FIELDS}),
        "rtmr3": _register(measurement_data["rtmr3"], "attestation.measurement.rtmr3"),
    }
    vm = _object(
        data["vm_config"],
        "attestation.vm_config",
        ("vcpu", "memory_mb", "os_image_hash"),
    )
    os_image_hash = vm["os_image_hash"]
    if os_image_hash is not None:
        _sha256(os_image_hash, "attestation.vm_config.os_image_hash")
    return {
        "tdx_quote": quote,
        "event_log": event_log,
        "report_data": report_data,
        "measurement": measurement,
        "vm_config": {
            "vcpu": _integer(vm["vcpu"], "attestation.vm_config.vcpu", minimum=1),
            "memory_mb": _integer(vm["memory_mb"], "attestation.vm_config.memory_mb", minimum=1),
            "os_image_hash": os_image_hash,
        },
    }


def validate_eval_execution_proof(value: Any) -> dict[str, Any]:
    data = _object(
        value,
        "execution_proof",
        (
            "version",
            "tier",
            "manifest_sha256",
            "image_digest",
            "provider",
            "worker_signature",
            "attestation",
        ),
    )
    if data["version"] != 1 or data["tier"] != "phala-tdx" or data["provider"] is not None:
        raise EvalWireError("execution_proof has invalid fixed fields")
    signature = _object(data["worker_signature"], "worker_signature", ("worker_pubkey", "sig"))
    if signature["worker_pubkey"] != "" or signature["sig"] != "":
        raise EvalWireError("Eval wire accepts only the empty worker signature placeholder")
    return {
        "version": 1,
        "tier": "phala-tdx",
        "manifest_sha256": _sha256(data["manifest_sha256"], "manifest_sha256"),
        "image_digest": _image(data["image_digest"], "image_digest"),
        "provider": None,
        "worker_signature": {"worker_pubkey": "", "sig": ""},
        "attestation": validate_eval_phala_attestation(data["attestation"]),
    }


def parse_eval_execution_proof_json(data: bytes | str) -> dict[str, Any]:
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvalWireError("execution proof JSON must be UTF-8") from exc
    if not isinstance(data, str):
        raise EvalWireError("execution proof JSON must be text")

    def _reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise EvalWireError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        import json

        parsed = json.loads(data, object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, ValueError) as exc:
        raise EvalWireError("malformed execution proof JSON") from exc
    return validate_eval_execution_proof(parsed)


def validate_eval_result_request(value: Any) -> dict[str, Any]:
    data = _object(
        value,
        "eval_result_request",
        (
            "schema_version",
            "eval_run_id",
            "submission_id",
            "agent_hash",
            "score_record",
            "scores_digest",
            "execution_proof",
        ),
    )
    if data["schema_version"] != 1:
        raise EvalWireError("eval_result_request schema_version must be 1")
    score_record = _public_score_record(_score_record_shape(data["score_record"]))
    digest = _sha256(data["scores_digest"], "scores_digest")
    if digest != score_record_digest(score_record):
        raise EvalWireError("scores_digest does not match score_record")
    if score_record["eval_run_id"] != _id(data["eval_run_id"], "eval_run_id"):
        raise EvalWireError("score_record eval_run_id does not match result request")
    return {
        "schema_version": 1,
        "eval_run_id": data["eval_run_id"],
        "submission_id": _id(data["submission_id"], "submission_id"),
        "agent_hash": _sha256(data["agent_hash"], "agent_hash"),
        "score_record": score_record,
        "scores_digest": digest,
        "execution_proof": validate_eval_execution_proof(data["execution_proof"]),
    }


def validate_eval_receipt(value: Any) -> dict[str, Any]:
    data = _object(
        value,
        "eval_receipt",
        (
            "schema_version",
            "eval_run_id",
            "receipt_id",
            "body_sha256",
            "received_at_ms",
            "phase",
            "terminal",
            "verified",
            "retryable",
            "reason_code",
            "result_available",
            "finalized_at_ms",
        ),
    )
    if data["schema_version"] != 1 or data["phase"] not in {
        "received",
        "verifying",
        "verified",
        "rejected",
        "verifier_unavailable",
    }:
        raise EvalWireError("eval_receipt has an invalid schema version or phase")
    for field in ("terminal", "verified", "retryable", "result_available"):
        if not isinstance(data[field], bool):
            raise EvalWireError(f"eval_receipt.{field} must be boolean")
    if data["reason_code"] is not None:
        _id(data["reason_code"], "reason_code")
    if data["finalized_at_ms"] is not None:
        _integer(data["finalized_at_ms"], "finalized_at_ms")
    return {
        **data,
        "eval_run_id": _id(data["eval_run_id"], "eval_run_id"),
        "receipt_id": _id(data["receipt_id"], "receipt_id"),
        "body_sha256": _sha256(data["body_sha256"], "body_sha256"),
        "received_at_ms": _integer(data["received_at_ms"], "received_at_ms"),
    }


__all__ = [
    "EVAL_MAX_EVENT_LOG_BYTES",
    "EVAL_MAX_EVENT_LOG_ENTRIES",
    "EVAL_MAX_INTEGER",
    "EVAL_MAX_PAYLOAD_BYTES",
    "EVAL_MAX_QUOTE_BYTES",
    "EVAL_MAX_STRING_BYTES",
    "EVAL_MAX_VM_CONFIG_BYTES",
    "EvalWireError",
    "KEY_RELEASE_DOMAIN",
    "REPORT_DATA_BYTES",
    "SCORE_DOMAIN",
    "agent_artifact_sha256_hex",
    "build_score_binding",
    "build_canonical_score_record",
    "canonical_json_v1",
    "canonical_measurement",
    "canonical_task_ids",
    "decode_score_f64be",
    "encode_score_f64be",
    "key_release_report_data_hex",
    "parse_eval_execution_proof_json",
    "score_record_digest",
    "score_report_data_hex",
    "scoring_policy_digest",
    "task_config_sha256_from_content_digest",
    "validate_canonical_score_record",
    "validate_eval_execution_proof",
    "validate_eval_plan",
    "validate_eval_phala_attestation",
    "validate_eval_receipt",
    "validate_eval_result_request",
    "validate_score_binding",
    "validate_scoring_policy",
]
