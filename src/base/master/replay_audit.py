"""BASE transport and validation for the labelled replay-audit seam.

Replay audits are deliberately not ordinary challenge work.  A request is
eligible only when the challenge labels it with the replay-audit protocol and
includes the complete immutable Eval plan.  This module keeps that discriminator
and the plan bytes together while the request crosses BASE's assignment plane.
It also validates raw ordered trial scores before a result can be forwarded back
to the challenge comparator.
"""

from __future__ import annotations

import json
import math
import re
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

REPLAY_AUDIT_LABEL = "agent-challenge.replay-audit.v1"
REPLAY_AUDIT_REQUEST_KIND = "replay_audit_request"
REPLAY_AUDIT_RESULT_KIND = "replay_audit_result"
REPLAY_AUDIT_ASSIGNMENT_KIND = "replay_audit"

REPLAY_AUDIT_LABEL_KEY = "replay_audit_label"
REPLAY_AUDIT_REQUEST_KEY = "replay_audit_request"
REPLAY_AUDIT_RESULT_KEY = "replay_audit_result"
REPLAY_AUDIT_FORWARDED_KEY = "replay_audit_result_forwarded"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REGISTER_RE = re.compile(r"^[0-9a-f]{96}$")
_F64_RE = re.compile(r"^[0-9a-f]{16}$")
_IMAGE_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


class ReplayAuditWireError(ValueError):
    """A replay payload is malformed or fails its immutable identity checks."""


def _require_id(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not (1 <= len(value) <= 128)
        or any(not ("!" <= char <= "~") for char in value)
    ):
        raise ReplayAuditWireError(f"{name} must be a 1-128 character visible ASCII id")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReplayAuditWireError(f"{name} must be lowercase sha256 hex")
    return value


def _require_register(value: Any, name: str) -> str:
    if not isinstance(value, str) or _REGISTER_RE.fullmatch(value) is None:
        raise ReplayAuditWireError(f"{name} must be lowercase 48-byte hex")
    return value


def _require_f64(value: Any, name: str) -> float:
    if not isinstance(value, str) or _F64_RE.fullmatch(value) is None:
        raise ReplayAuditWireError(f"{name} must be lowercase binary64 hex")
    try:
        raw = bytes.fromhex(value)
        score = struct.unpack(">d", raw)[0]
    except (ValueError, TypeError):
        raise ReplayAuditWireError(f"{name} must be lowercase binary64 hex") from None
    if raw == b"\x80" + b"\0" * 7:
        raise ReplayAuditWireError(f"{name} must not be negative zero")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ReplayAuditWireError(f"{name} must be finite and in [0, 1]")
    return score


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ReplayAuditWireError(f"{name} must be a positive integer")
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            _normalize_canonical(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ReplayAuditWireError("immutable Eval plan is not canonical JSON") from exc


def _validate_raw_body(
    raw_body: bytes, value: Mapping[str, Any], *, kind: str = "replay request"
) -> None:
    """Reject duplicate/alternate JSON object representations at the HTTP seam."""

    try:
        parsed = json.loads(
            raw_body.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ReplayAuditWireError(f"unsupported JSON constant: {constant}")
            ),
        )
    except ReplayAuditWireError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayAuditWireError(f"{kind} body is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, Mapping) or dict(parsed) != dict(value):
        raise ReplayAuditWireError(f"{kind} body differs from parsed request")


def _reject_duplicate_json_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in items:
        if key in result:
            raise ReplayAuditWireError(f"duplicate JSON key: {key!r}")
        result[key] = item
    return result


def parse_replay_json(raw_body: bytes | str) -> Any:
    """Parse replay JSON while rejecting duplicate object keys."""

    try:
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")
        if not isinstance(raw_body, str):
            raise TypeError("replay JSON must be bytes or text")
        return json.loads(
            raw_body,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ReplayAuditWireError(f"unsupported JSON constant: {constant}")
            ),
        )
    except ReplayAuditWireError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayAuditWireError("replay JSON is not valid UTF-8 JSON") from exc


def _normalize_canonical(value: Any) -> Any:
    """Match agent-challenge's canonical_json_v1 profile exactly."""

    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        raise ReplayAuditWireError("canonical Eval plan JSON forbids floats")
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ReplayAuditWireError("canonical Eval plan JSON forbids surrogates")
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReplayAuditWireError(
                    "canonical Eval plan object keys must be strings"
                )
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ReplayAuditWireError("canonical Eval plan has duplicate keys")
            normalized[normalized_key] = _normalize_canonical(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_canonical(item) for item in value]
    raise ReplayAuditWireError("canonical Eval plan contains unsupported JSON value")


def _object(value: Any, name: str, fields: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayAuditWireError(f"{name} must be an object")
    actual = set(value)
    expected = set(fields)
    if actual != expected:
        raise ReplayAuditWireError(
            f"{name} has invalid fields: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return dict(value)


def _require_image(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IMAGE_RE.fullmatch(value) is None:
        raise ReplayAuditWireError(f"{name} must be digest-pinned image reference")
    return value


def _validate_scoring_policy(value: Any) -> dict[str, Any]:
    policy = _object(
        value,
        "eval_plan.scoring_policy",
        (
            "schema_version",
            "per_task_aggregation",
            "keep_policy",
            "drop_lowest_n",
            "threshold_f64be",
        ),
    )
    if policy["schema_version"] != 1:
        raise ReplayAuditWireError("scoring policy schema_version must be 1")
    if policy["per_task_aggregation"] not in {"mean", "best_of_k"}:
        raise ReplayAuditWireError("invalid scoring policy aggregation")
    if policy["keep_policy"] not in {"off", "drop_lowest_n", "threshold_band"}:
        raise ReplayAuditWireError("invalid scoring policy keep_policy")
    drop_lowest_n = policy["drop_lowest_n"]
    if (
        not isinstance(drop_lowest_n, int)
        or isinstance(drop_lowest_n, bool)
        or drop_lowest_n < 0
    ):
        raise ReplayAuditWireError(
            "scoring policy drop_lowest_n must be non-negative integer"
        )
    threshold = policy["threshold_f64be"]
    if policy["keep_policy"] == "threshold_band":
        _require_f64(threshold, "scoring policy threshold_f64be")
        if drop_lowest_n != 0:
            raise ReplayAuditWireError("threshold_band requires neutral drop_lowest_n")
    elif threshold is not None:
        raise ReplayAuditWireError(
            "threshold_f64be must be null outside threshold_band"
        )
    if policy["keep_policy"] != "drop_lowest_n" and drop_lowest_n != 0:
        raise ReplayAuditWireError(
            "drop_lowest_n must be neutral outside drop_lowest_n"
        )
    return {
        "schema_version": 1,
        "per_task_aggregation": policy["per_task_aggregation"],
        "keep_policy": policy["keep_policy"],
        "drop_lowest_n": drop_lowest_n,
        "threshold_f64be": threshold,
    }


def scoring_policy_digest(policy: Mapping[str, Any]) -> str:
    """Compute the digest over the complete canonical Scoring policy v1 bytes."""

    return sha256(_canonical_json(_validate_scoring_policy(policy))).hexdigest()


def _validate_plan(value: Any) -> dict[str, Any]:
    """Validate the complete, schema-closed Eval plan v1 independently."""

    plan = _object(
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
    if plan["schema_version"] != 1:
        raise ReplayAuditWireError("eval_plan schema_version must be 1")
    eval_run_id = _require_id(plan["eval_run_id"], "eval_plan.eval_run_id")
    submission_id = _require_id(plan["submission_id"], "eval_plan.submission_id")
    submission_version = plan["submission_version"]
    if (
        not isinstance(submission_version, int)
        or isinstance(submission_version, bool)
        or submission_version < 1
    ):
        raise ReplayAuditWireError(
            "eval_plan.submission_version must be positive integer"
        )
    review_digest = _require_sha256(
        plan["authorizing_review_digest"], "eval_plan.authorizing_review_digest"
    )
    agent_hash = _require_sha256(plan["agent_hash"], "eval_plan.agent_hash")
    k = _require_positive_int(plan["k"], "eval_plan.k")
    policy = _validate_scoring_policy(plan["scoring_policy"])
    policy_digest = _require_sha256(
        plan["scoring_policy_digest"], "eval_plan.scoring_policy_digest"
    )
    if policy_digest != scoring_policy_digest(policy):
        raise ReplayAuditWireError("scoring_policy_digest does not match policy bytes")

    selected_raw = plan["selected_tasks"]
    if not isinstance(selected_raw, list) or not selected_raw:
        raise ReplayAuditWireError("eval_plan.selected_tasks must be non-empty array")
    selected: list[dict[str, str]] = []
    previous = ""
    for item in selected_raw:
        task = _object(
            item,
            "eval_plan.selected_tasks[]",
            ("task_id", "image_ref", "task_config_sha256"),
        )
        task_id = _require_id(task["task_id"], "selected_tasks[].task_id")
        if previous >= task_id:
            raise ReplayAuditWireError("selected_tasks must be sorted and unique")
        previous = task_id
        selected.append(
            {
                "task_id": task_id,
                "image_ref": _require_image(
                    task["image_ref"], "selected_tasks[].image_ref"
                ),
                "task_config_sha256": _require_sha256(
                    task["task_config_sha256"], "selected_tasks[].task_config_sha256"
                ),
            }
        )

    app = _object(
        plan["eval_app"],
        "eval_plan.eval_app",
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
    measurement = _object(
        app["measurement"],
        "eval_plan.eval_app.measurement",
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
    public_key = _require_sha256(
        app["kms_public_key_hex"], "eval_plan.eval_app.kms_public_key_hex"
    )
    public_key_digest = _require_sha256(
        app["kms_public_key_sha256"], "eval_plan.eval_app.kms_public_key_sha256"
    )
    if sha256(bytes.fromhex(public_key)).hexdigest() != public_key_digest:
        raise ReplayAuditWireError("kms_public_key_sha256 does not match public key")
    if app["kms_key_algorithm"] != "x25519":
        raise ReplayAuditWireError("eval_app.kms_key_algorithm must be x25519")
    valid_measurement = {
        "mrtd": _require_register(measurement["mrtd"], "measurement.mrtd"),
        "rtmr0": _require_register(measurement["rtmr0"], "measurement.rtmr0"),
        "rtmr1": _require_register(measurement["rtmr1"], "measurement.rtmr1"),
        "rtmr2": _require_register(measurement["rtmr2"], "measurement.rtmr2"),
        "os_image_hash": _require_sha256(
            measurement["os_image_hash"], "measurement.os_image_hash"
        ),
        "key_provider": _require_id(
            measurement["key_provider"], "measurement.key_provider"
        ),
        "vm_shape": _require_id(measurement["vm_shape"], "measurement.vm_shape"),
    }

    key_nonce = _require_id(plan["key_release_nonce"], "eval_plan.key_release_nonce")
    score_nonce = _require_id(plan["score_nonce"], "eval_plan.score_nonce")
    if key_nonce == score_nonce:
        raise ReplayAuditWireError("Eval plan nonces must be distinct")
    expected_result_endpoint = f"/evaluation/v1/runs/{eval_run_id}/result"
    if plan["result_endpoint"] != expected_result_endpoint:
        raise ReplayAuditWireError("eval_plan.result_endpoint does not target run")
    endpoint = plan["key_release_endpoint"]
    if not isinstance(endpoint, str) or not endpoint or len(endpoint) > 16_384:
        raise ReplayAuditWireError("eval_plan.key_release_endpoint is invalid")
    for plan_field in ("issued_at_ms", "expires_at_ms"):
        timestamp = plan[plan_field]
        if (
            not isinstance(timestamp, int)
            or isinstance(timestamp, bool)
            or timestamp < 0
        ):
            raise ReplayAuditWireError(
                f"eval_plan.{plan_field} must be non-negative integer"
            )
    if plan["expires_at_ms"] <= plan["issued_at_ms"]:
        raise ReplayAuditWireError("eval_plan expiry must be after issue time")

    return {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "submission_id": submission_id,
        "submission_version": submission_version,
        "authorizing_review_digest": review_digest,
        "agent_hash": agent_hash,
        "selected_tasks": selected,
        "k": k,
        "scoring_policy": policy,
        "scoring_policy_digest": policy_digest,
        "eval_app": {
            "image_ref": _require_image(app["image_ref"], "eval_app.image_ref"),
            "compose_hash": _require_sha256(
                app["compose_hash"], "eval_app.compose_hash"
            ),
            "app_identity": _require_id(app["app_identity"], "eval_app.app_identity"),
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": public_key,
            "kms_public_key_sha256": public_key_digest,
            "measurement": valid_measurement,
        },
        "key_release_endpoint": endpoint,
        "result_endpoint": expected_result_endpoint,
        "key_release_nonce": key_nonce,
        "score_nonce": score_nonce,
        "run_token_sha256": _require_sha256(
            plan["run_token_sha256"], "eval_plan.run_token_sha256"
        ),
        "issued_at_ms": plan["issued_at_ms"],
        "expires_at_ms": plan["expires_at_ms"],
    }


def plan_sha256(plan: Mapping[str, Any]) -> str:
    """Digest the exact canonical bytes of a complete Eval plan v1."""

    return sha256(_canonical_json(_validate_plan(plan))).hexdigest()


def _require_plan(value: Any) -> dict[str, Any]:
    return _validate_plan(value)


def _replay_audit_id(eval_run_id: str, replay_attempt: int) -> str:
    return f"replay:{eval_run_id}:{replay_attempt}"


@dataclass(frozen=True)
class ReplayAuditRequest:
    """One labelled request for a full-plan legacy own_runner replay."""

    audit_id: str
    submission_id: str
    eval_run_id: str
    replay_attempt: int
    plan_sha256: str
    eval_plan: Mapping[str, Any]
    k: int
    selected_tasks: tuple[Mapping[str, Any], ...]
    scoring_policy: Mapping[str, Any]
    scoring_policy_digest: str
    attested_score: float
    raw_body: bytes | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, raw_body: bytes | None = None
    ) -> ReplayAuditRequest:
        expected = {
            "schema_version",
            "audit_label",
            "kind",
            "audit_id",
            "submission_id",
            "eval_run_id",
            "replay_attempt",
            "plan_sha256",
            "eval_plan",
            "k",
            "selected_tasks",
            "scoring_policy",
            "scoring_policy_digest",
            "attested_score",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ReplayAuditWireError("replay request has unknown or missing fields")
        if (
            value["schema_version"] != 1
            or value["audit_label"] != REPLAY_AUDIT_LABEL
            or value["kind"] != REPLAY_AUDIT_REQUEST_KIND
        ):
            raise ReplayAuditWireError("replay request is not separately labelled")
        audit_id = _require_id(value["audit_id"], "audit_id")
        submission_id = _require_id(value["submission_id"], "submission_id")
        eval_run_id = _require_id(value["eval_run_id"], "eval_run_id")
        replay_attempt = _require_positive_int(
            value["replay_attempt"], "replay_attempt"
        )
        if audit_id != _replay_audit_id(eval_run_id, replay_attempt):
            raise ReplayAuditWireError("audit_id does not match replay identity")
        digest = _require_sha256(value["plan_sha256"], "plan_sha256")
        plan = _require_plan(value["eval_plan"])
        if raw_body is not None:
            _validate_raw_body(raw_body, value)
        if plan["eval_run_id"] != eval_run_id:
            raise ReplayAuditWireError("request/run Eval plan identity mismatch")
        if plan["submission_id"] != submission_id:
            raise ReplayAuditWireError("request/submission Eval plan identity mismatch")
        if digest != plan_sha256(plan):
            raise ReplayAuditWireError(
                "plan_sha256 does not match canonical Eval plan bytes"
            )
        k = _require_positive_int(value["k"], "k")
        if k != plan["k"]:
            raise ReplayAuditWireError("request k differs from immutable Eval plan")
        selected = value["selected_tasks"]
        if not isinstance(selected, list) or selected != plan["selected_tasks"]:
            raise ReplayAuditWireError(
                "selected task bytes differ from immutable Eval plan"
            )
        policy = value["scoring_policy"]
        if not isinstance(policy, Mapping) or dict(policy) != plan["scoring_policy"]:
            raise ReplayAuditWireError(
                "scoring policy bytes differ from immutable Eval plan"
            )
        policy_digest = _require_sha256(
            value["scoring_policy_digest"], "scoring_policy_digest"
        )
        if policy_digest != scoring_policy_digest(policy):
            raise ReplayAuditWireError(
                "scoring_policy_digest does not match canonical policy bytes"
            )
        if not isinstance(value["attested_score"], (int, float)) or isinstance(
            value["attested_score"], bool
        ):
            raise ReplayAuditWireError("attested_score must be numeric")
        if (
            not math.isfinite(float(value["attested_score"]))
            or not 0.0 <= float(value["attested_score"]) <= 1.0
        ):
            raise ReplayAuditWireError("attested_score must be finite in [0, 1]")
        return cls(
            audit_id=audit_id,
            submission_id=submission_id,
            eval_run_id=eval_run_id,
            replay_attempt=replay_attempt,
            plan_sha256=digest,
            eval_plan=plan,
            k=k,
            selected_tasks=tuple(dict(item) for item in selected),
            scoring_policy=dict(policy),
            scoring_policy_digest=policy_digest,
            attested_score=float(value["attested_score"]),
            raw_body=raw_body,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the labelled wire object without dropping plan/policy fields."""

        result = {
            "schema_version": 1,
            "audit_label": REPLAY_AUDIT_LABEL,
            "kind": REPLAY_AUDIT_REQUEST_KIND,
            "audit_id": self.audit_id,
            "submission_id": self.submission_id,
            "eval_run_id": self.eval_run_id,
            "replay_attempt": self.replay_attempt,
            "plan_sha256": self.plan_sha256,
            "eval_plan": dict(self.eval_plan),
            "k": self.k,
            "selected_tasks": [dict(item) for item in self.selected_tasks],
            "scoring_policy": dict(self.scoring_policy),
            "scoring_policy_digest": self.scoring_policy_digest,
            "attested_score": self.attested_score,
        }
        return result

    @property
    def work_unit_id(self) -> str:
        return self.audit_id


@dataclass(frozen=True)
class ReplayAuditResult:
    """Raw ordered replay trials returned to the challenge comparator."""

    audit_id: str
    submission_id: str
    eval_run_id: str
    replay_attempt: int
    plan_sha256: str
    trial_scores_by_task: Mapping[str, tuple[float, ...]]
    raw_body: bytes | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, raw_body: bytes | None = None
    ) -> ReplayAuditResult:
        expected = {
            "schema_version",
            "audit_label",
            "kind",
            "audit_id",
            "submission_id",
            "eval_run_id",
            "replay_attempt",
            "plan_sha256",
            "trial_scores_by_task",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ReplayAuditWireError("replay result has unknown or missing fields")
        if (
            value["schema_version"] != 1
            or value["audit_label"] != REPLAY_AUDIT_LABEL
            or value["kind"] != REPLAY_AUDIT_RESULT_KIND
        ):
            raise ReplayAuditWireError("replay result is not separately labelled")
        if raw_body is not None:
            _validate_raw_body(raw_body, value, kind="replay result")
        scores_by_task = value["trial_scores_by_task"]
        if not isinstance(scores_by_task, Mapping) or not scores_by_task:
            raise ReplayAuditWireError("replay result requires raw task trial scores")
        parsed: dict[str, tuple[float, ...]] = {}
        for task_id, scores in scores_by_task.items():
            _require_id(task_id, "trial task_id")
            if (
                not isinstance(scores, Sequence)
                or isinstance(scores, (str, bytes, bytearray))
                or not scores
            ):
                raise ReplayAuditWireError("each replay task requires ordered trials")
            values: list[float] = []
            for score in scores:
                if (
                    not isinstance(score, (int, float))
                    or isinstance(score, bool)
                    or not math.isfinite(float(score))
                    or not 0.0 <= float(score) <= 1.0
                ):
                    raise ReplayAuditWireError(
                        "replay trial scores must be finite numbers"
                    )
                values.append(float(score))
            parsed[task_id] = tuple(values)
        return cls(
            audit_id=_require_id(value["audit_id"], "audit_id"),
            submission_id=_require_id(value["submission_id"], "submission_id"),
            eval_run_id=_require_id(value["eval_run_id"], "eval_run_id"),
            replay_attempt=_require_positive_int(
                value["replay_attempt"], "replay_attempt"
            ),
            plan_sha256=_require_sha256(value["plan_sha256"], "plan_sha256"),
            trial_scores_by_task=parsed,
            raw_body=raw_body,
        )

    def validate_against(self, request: ReplayAuditRequest) -> None:
        # Revalidate the immutable request before comparing result identity.
        ReplayAuditRequest.from_mapping(request.to_dict())
        if (
            self.audit_id != request.audit_id
            or self.submission_id != request.submission_id
            or self.eval_run_id != request.eval_run_id
            or self.replay_attempt != request.replay_attempt
            or self.plan_sha256 != request.plan_sha256
        ):
            raise ReplayAuditWireError("replay result identity differs from request")
        task_ids = [str(item["task_id"]) for item in request.selected_tasks]
        if list(self.trial_scores_by_task) != task_ids:
            raise ReplayAuditWireError(
                "replay result task order differs from immutable plan"
            )
        if any(
            len(scores) != request.k for scores in self.trial_scores_by_task.values()
        ):
            raise ReplayAuditWireError(
                "replay result trial count differs from immutable k"
            )
        if set(self.trial_scores_by_task) != {
            str(item["task_id"]) for item in request.selected_tasks
        }:
            raise ReplayAuditWireError(
                "replay result task set differs from immutable plan"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "audit_label": REPLAY_AUDIT_LABEL,
            "kind": REPLAY_AUDIT_RESULT_KIND,
            "audit_id": self.audit_id,
            "submission_id": self.submission_id,
            "eval_run_id": self.eval_run_id,
            "replay_attempt": self.replay_attempt,
            "plan_sha256": self.plan_sha256,
            "trial_scores_by_task": {
                task_id: list(scores)
                for task_id, scores in self.trial_scores_by_task.items()
            },
        }


def replay_assignment_payload(request: ReplayAuditRequest) -> dict[str, Any]:
    """Build the only assignment payload that may invoke replay dispatch."""

    # Validate at the assignment boundary as well as at HTTP ingestion. This
    # prevents a forged/mutated in-memory request from becoming broker work.
    validated = ReplayAuditRequest.from_mapping(request.to_dict())
    return {
        "assignment_kind": REPLAY_AUDIT_ASSIGNMENT_KIND,
        REPLAY_AUDIT_LABEL_KEY: REPLAY_AUDIT_LABEL,
        REPLAY_AUDIT_REQUEST_KEY: validated.to_dict(),
    }


def is_replay_assignment_payload(payload: Mapping[str, Any] | None) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("assignment_kind") == REPLAY_AUDIT_ASSIGNMENT_KIND
        and payload.get(REPLAY_AUDIT_LABEL_KEY) == REPLAY_AUDIT_LABEL
        and isinstance(payload.get(REPLAY_AUDIT_REQUEST_KEY), Mapping)
    )


__all__ = [
    "REPLAY_AUDIT_ASSIGNMENT_KIND",
    "REPLAY_AUDIT_FORWARDED_KEY",
    "REPLAY_AUDIT_LABEL",
    "REPLAY_AUDIT_LABEL_KEY",
    "REPLAY_AUDIT_REQUEST_KEY",
    "REPLAY_AUDIT_RESULT_KEY",
    "REPLAY_AUDIT_RESULT_KIND",
    "REPLAY_AUDIT_REQUEST_KIND",
    "ReplayAuditRequest",
    "ReplayAuditResult",
    "ReplayAuditWireError",
    "is_replay_assignment_payload",
    "parse_replay_json",
    "plan_sha256",
    "replay_assignment_payload",
    "scoring_policy_digest",
]
