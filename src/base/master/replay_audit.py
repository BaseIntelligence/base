"""BASE transport and validation for the labelled replay-audit seam.

Replay audits are deliberately not ordinary challenge work.  A request is
eligible only when the challenge labels it with the replay-audit protocol and
includes the complete immutable Eval plan.  This module keeps that discriminator
and the plan bytes together while the request crosses BASE's assignment plane.
It also validates raw ordered trial scores before a result can be forwarded back
to the challenge comparator.
"""

from __future__ import annotations

import math
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


class ReplayAuditWireError(ValueError):
    """A replay payload is malformed or fails its immutable identity checks."""


def _require_id(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not ("!" <= char <= "~") for char in value)
    ):
        raise ReplayAuditWireError(f"{name} must be a visible ASCII id")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ReplayAuditWireError(f"{name} must be lowercase sha256 hex")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ReplayAuditWireError(f"{name} must be a positive integer")
    return value


def _canonical_json(value: Any) -> bytes:
    """Serialize plan data with the stable object encoding used by BASE.

    BASE does not become a second score consumer, but it must bind the exact
    request identity used to create an assignment.  The challenge remains the
    authority for Eval-plan schema validation and score comparison.
    """

    try:
        import json

        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ReplayAuditWireError("immutable Eval plan is not canonical JSON") from exc


def plan_sha256(plan: Mapping[str, Any]) -> str:
    """Return the deterministic digest used for BASE request identity checks."""

    return sha256(_canonical_json(dict(plan))).hexdigest()


def _require_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayAuditWireError("eval_plan must be an object")
    plan = dict(value)
    required = {"eval_run_id", "selected_tasks", "k", "scoring_policy"}
    if not required <= set(plan):
        raise ReplayAuditWireError("eval_plan is missing immutable replay fields")
    _require_id(plan["eval_run_id"], "eval_plan.eval_run_id")
    k = _require_positive_int(plan["k"], "eval_plan.k")
    selected = plan["selected_tasks"]
    if not isinstance(selected, list) or not selected:
        raise ReplayAuditWireError("eval_plan.selected_tasks must be a non-empty list")
    task_ids: list[str] = []
    for item in selected:
        if not isinstance(item, Mapping):
            raise ReplayAuditWireError(
                "eval_plan.selected_tasks entries must be objects"
            )
        task_id = _require_id(item.get("task_id"), "selected task_id")
        task_ids.append(task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ReplayAuditWireError("eval_plan.selected_tasks must be unique")
    policy = plan["scoring_policy"]
    if not isinstance(policy, Mapping) or not policy:
        raise ReplayAuditWireError(
            "eval_plan.scoring_policy must be complete object bytes"
        )
    # Keep this check close to the wire, without normalizing or rewriting policy.
    plan["k"] = k
    return plan


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
        digest = _require_sha256(value["plan_sha256"], "plan_sha256")
        plan = _require_plan(value["eval_plan"])
        if plan["eval_run_id"] != eval_run_id:
            raise ReplayAuditWireError("request/run Eval plan identity mismatch")
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
        if not math.isfinite(float(value["attested_score"])):
            raise ReplayAuditWireError("attested_score must be finite")
        if not isinstance(value["attested_score"], (int, float)) or isinstance(
            value["attested_score"], bool
        ):
            raise ReplayAuditWireError("attested_score must be numeric")
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

        return {
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

    return {
        "assignment_kind": REPLAY_AUDIT_ASSIGNMENT_KIND,
        REPLAY_AUDIT_LABEL_KEY: REPLAY_AUDIT_LABEL,
        REPLAY_AUDIT_REQUEST_KEY: request.to_dict(),
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
    "plan_sha256",
    "replay_assignment_payload",
]
