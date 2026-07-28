"""Plan-authoritative scoring for attested Eval runs.

The complete Eval plan is the single authority for the attested path.  This
module owns the small persistence and reconstruction seam shared by result
ingestion, job finalizers, and replay.  Legacy jobs deliberately have no plan
and continue to use their existing arithmetic unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.models import EvaluationJob

#: Stable refuse code when production attestation sees a legacy validator_nonce
#: report_data preimage instead of the schema-v2 score binding.
#: Note: this gates the SCORE BINDING inside report_data (schema_version: 2),
#: not the outer wire score_record.schema_version (which may remain 1).
LEGACY_REPORT_DATA_REJECTED = "legacy_report_data_rejected"

#: Host refuse codes for guest dual-hash execution proof (success path).
#: Distinct from score-0 burns and generic attestation_verification_failed.
GUEST_ARTIFACT_PROOF_MISSING = "guest_artifact_proof_missing"
GUEST_ARTIFACT_PROOF_HASH_MISMATCH = "guest_artifact_proof_hash_mismatch"
GUEST_ARTIFACT_PROOF_AGENT_HASH_MISMATCH = "guest_artifact_proof_agent_hash_mismatch"


class CanonicalPlanScoringError(ValueError):
    """Raised when immutable plan-backed scoring cannot be reconstructed.

    ``reason_code`` is a stable machine-readable code when the failure is a
    known production refuse (e.g. ``legacy_report_data_rejected``).  Callers
    that only inspect the exception message remain compatible.
    """

    def __init__(self, message: str, *, reason_code: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class PlanFinalScore:
    """Full-set score/counts reconstructed from one validated Eval plan."""

    score: float
    passed_tasks: int
    total_tasks: int
    score_record: dict[str, Any]


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise CanonicalPlanScoringError(f"duplicate persisted JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: str, *, field: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (TypeError, ValueError) as exc:
        raise CanonicalPlanScoringError(f"{field} is not canonical JSON") from exc


def _validated_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return ew.validate_eval_plan(value)
    except ew.EvalWireError as exc:
        raise CanonicalPlanScoringError(f"invalid immutable Eval plan: {exc}") from exc


def canonical_eval_plan_json(value: Mapping[str, Any]) -> str:
    """Validate and serialize one immutable Eval plan with canonical bytes."""

    return ew.canonical_json_v1(_validated_plan(value)).decode("utf-8")


def persist_eval_plan(job: EvaluationJob, value: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the exact validated plan bytes to an attested evaluation job."""

    plan = _validated_plan(value)
    job.eval_plan_json = ew.canonical_json_v1(plan).decode("utf-8")
    return plan


def load_eval_plan(job: EvaluationJob) -> dict[str, Any] | None:
    """Load the exact plan bytes for ``job`` or ``None`` for a legacy job."""

    raw = job.eval_plan_json
    if raw is None:
        return None
    parsed = _parse_json(raw, field="eval_plan_json")
    plan = _validated_plan(parsed)
    if ew.canonical_json_v1(plan).decode("utf-8") != raw:
        raise CanonicalPlanScoringError("persisted Eval plan is not canonical")
    return plan


def _validated_record(
    plan: Mapping[str, Any],
    score_record: Mapping[str, Any],
) -> dict[str, Any]:
    task_ids = [task["task_id"] for task in plan["selected_tasks"]]
    try:
        return ew.validate_canonical_score_record(
            score_record,
            scoring_policy=plan["scoring_policy"],
            expected_eval_run_id=plan["eval_run_id"],
            expected_task_ids=task_ids,
            expected_k=plan["k"],
        )
    except ew.EvalWireError as exc:
        raise CanonicalPlanScoringError(
            f"score record does not match immutable Eval plan: {exc}"
        ) from exc


def validate_score_record_from_eval_plan(
    eval_plan: Mapping[str, Any],
    score_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct and validate a score record against exact plan bytes."""

    return _validated_record(_validated_plan(eval_plan), score_record)


def build_score_record_from_eval_plan(
    eval_plan: Mapping[str, Any],
    trial_scores_by_task: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Build the sole score record permitted by a plan's policy and selected set."""

    plan = _validated_plan(eval_plan)
    try:
        record = ew.build_canonical_score_record(
            eval_run_id=plan["eval_run_id"],
            policy=plan["scoring_policy"],
            trial_scores_by_task=trial_scores_by_task,
        )
    except ew.EvalWireError as exc:
        raise CanonicalPlanScoringError(f"invalid ordered trial scores: {exc}") from exc
    return _validated_record(plan, record)


def scoring_policy_from_settings(settings: Any) -> dict[str, Any]:
    """Materialize a strict Scoring policy v1 object from issuance settings.

    Runtime configuration may retain legacy hyphenated spellings for existing
    planless jobs.  They are converted exactly once at plan issuance.  Consumers
    accept only the immutable underscore-based wire vocabulary thereafter.
    """

    aggregation = str(settings.per_task_aggregation)
    if aggregation == "best-of-k":
        aggregation = "best_of_k"
    keep_policy = str(settings.keep_good_tasks_policy)
    if keep_policy == "drop-lowest-n":
        keep_policy = "drop_lowest_n"
    elif keep_policy == "threshold-band":
        keep_policy = "threshold_band"
    elif keep_policy == "best-of-k":
        # A job-level best-of-k was a legacy-only setting.  It has no canonical
        # Scoring policy v1 spelling and cannot silently enter an Eval plan.
        raise CanonicalPlanScoringError("legacy best-of-k is not a canonical keep policy")

    policy = {
        "schema_version": 1,
        "per_task_aggregation": aggregation,
        "keep_policy": keep_policy,
        "drop_lowest_n": (
            int(settings.keep_good_tasks_drop_lowest) if keep_policy == "drop_lowest_n" else 0
        ),
        "threshold_f64be": (
            ew.encode_score_f64be(settings.keep_good_tasks_threshold)
            if keep_policy == "threshold_band"
            else None
        ),
    }
    try:
        return ew.validate_scoring_policy(policy)
    except ew.EvalWireError as exc:
        raise CanonicalPlanScoringError(
            f"settings cannot produce a Scoring policy v1: {exc}"
        ) from exc


def require_schema_v2_score_report_data(
    *,
    reported_report_data: str,
    canonical_measurement: Mapping[str, Any],
    agent_hash: str,
    task_ids: Sequence[str],
    scores_digest: str,
    eval_run_id: str,
    score_nonce: str,
    key_release_nonce: str | None = None,
    phala_attestation_enabled: bool = True,
) -> str:
    """Require schema-v2 score-binding ``report_data`` on the production path.

    When ``phala_attestation_enabled`` is true, a ``report_data`` that only
    validates under the legacy ``validator_nonce`` construction is fail-closed
    rejected with :data:`LEGACY_REPORT_DATA_REJECTED`.  There is no warning or
    downgrade-and-accept path.

    Naming subtlety: the outer wire ``score_record.schema_version`` may remain
    1; this function gates the SCORE BINDING hashed into TDX ``report_data``
    (``schema_version: 2`` via :func:`eval_wire.build_score_binding`).
    """

    from agent_challenge.canonical import report_data as rd

    if not isinstance(reported_report_data, str) or not reported_report_data:
        raise CanonicalPlanScoringError(
            "result report_data does not match immutable Eval plan",
            reason_code="result_report_data_mismatch",
        )

    expected_binding = ew.build_score_binding(
        canonical_measurement=canonical_measurement,
        agent_hash=agent_hash,
        eval_run_id=eval_run_id,
        score_nonce=score_nonce,
        scores_digest=scores_digest,
        task_ids=list(task_ids),
    )
    expected_hex = ew.score_report_data_hex(expected_binding)
    reported = reported_report_data.lower()
    if reported == expected_hex.lower():
        return expected_hex

    if phala_attestation_enabled:
        # Detect legacy validator_nonce constructions that production must never
        # accept.  The live bug bound key_release_nonce (and sometimes score_nonce)
        # as validator_nonce; check both candidates.
        legacy_nonces: list[str] = []
        if isinstance(key_release_nonce, str) and key_release_nonce:
            legacy_nonces.append(key_release_nonce)
        if isinstance(score_nonce, str) and score_nonce and score_nonce not in legacy_nonces:
            legacy_nonces.append(score_nonce)
        for nonce in legacy_nonces:
            try:
                legacy_hex = rd.report_data_hex(
                    canonical_measurement=canonical_measurement,
                    agent_hash=agent_hash,
                    task_ids=list(task_ids),
                    scores_digest=scores_digest,
                    validator_nonce=nonce,
                )
            except (TypeError, ValueError):
                continue
            if reported == legacy_hex.lower():
                raise CanonicalPlanScoringError(
                    "legacy validator_nonce report_data rejected; "
                    "production requires schema-v2 score binding",
                    reason_code=LEGACY_REPORT_DATA_REJECTED,
                )

    raise CanonicalPlanScoringError(
        "result report_data does not match immutable Eval plan",
        reason_code="result_report_data_mismatch",
    )


def require_host_guest_artifact_proof(
    result_request: Mapping[str, Any],
    *,
    expected_agent_hash: str,
) -> dict[str, Any]:
    """Require and bind guest_artifact_proof to the submission agent hash.

    Success/completed Eval results must carry a dual-hash proof that the guest
    executed the same artifact the host already knows (plan/submission
    ``agent_hash``). Fail-closed with greppable reason codes — never a silent
    score-0 that looks like a legitimate bad run.

    Wire-level ``validate_eval_result_request`` keeps the field optional for
    legacy stored bodies; this host policy is the scoring/admission chokepoint.
    """

    if not isinstance(result_request, Mapping):
        raise CanonicalPlanScoringError(
            "success Eval result requires guest_artifact_proof; result request is not an object",
            reason_code=GUEST_ARTIFACT_PROOF_MISSING,
        )
    if "guest_artifact_proof" not in result_request:
        raise CanonicalPlanScoringError(
            "success Eval result requires guest_artifact_proof"
            " (missing on host admission/scoring path)",
            reason_code=GUEST_ARTIFACT_PROOF_MISSING,
        )
    try:
        proof = ew.validate_guest_artifact_proof(result_request["guest_artifact_proof"])
    except ew.EvalWireError as exc:
        raise CanonicalPlanScoringError(
            f"guest_artifact_proof rejected: {exc}",
            reason_code=GUEST_ARTIFACT_PROOF_HASH_MISMATCH,
        ) from exc

    expected = str(expected_agent_hash).lower()
    expected_hash = str(proof["expected_hash"]).lower()
    download_hash = str(proof["download_hash"]).lower()
    executed_hash = str(proof["executed_hash"]).lower()
    if not (expected_hash == download_hash == executed_hash):
        raise CanonicalPlanScoringError(
            "guest_artifact_proof hashes must be equal"
            " (expected_hash == download_hash == executed_hash)",
            reason_code=GUEST_ARTIFACT_PROOF_HASH_MISMATCH,
        )
    if expected_hash != expected:
        raise CanonicalPlanScoringError(
            "guest_artifact_proof does not match submission agent_hash"
            " (proof describes a different artifact than the immutable plan)",
            reason_code=GUEST_ARTIFACT_PROOF_AGENT_HASH_MISMATCH,
        )
    return proof


def validate_eval_result_from_plan(
    eval_plan: Mapping[str, Any],
    result_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate direct-ingestion bytes against one immutable Eval plan.

    This does not verify the TDX quote.  It establishes the deterministic
    contract the direct endpoint supplies to its quote verifier: run identity,
    submission/agent identity, selected set, ordered trial record, plan policy,
    expected image/measurement, guest artifact execution proof, and report-data
    binding are all reconstructed from the same persisted plan bytes.

    Production always requires the schema-v2 score binding inside report_data
    (see :func:`require_schema_v2_score_report_data`).  Outer wire
    ``score_record.schema_version`` may remain 1.
    """

    plan = _validated_plan(eval_plan)
    try:
        request = ew.validate_eval_result_request(result_request)
    except ew.EvalWireError as exc:
        message = str(exc)
        lower = message.lower()
        if "guest_artifact_proof" in lower:
            if "match" in lower or "hash" in lower or "equal" in lower:
                raise CanonicalPlanScoringError(
                    f"invalid Eval result request: {exc}",
                    reason_code=GUEST_ARTIFACT_PROOF_HASH_MISMATCH,
                ) from exc
            raise CanonicalPlanScoringError(
                f"invalid Eval result request: {exc}",
                reason_code=GUEST_ARTIFACT_PROOF_HASH_MISMATCH,
            ) from exc
        raise CanonicalPlanScoringError(f"invalid Eval result request: {exc}") from exc
    if request["eval_run_id"] != plan["eval_run_id"]:
        raise CanonicalPlanScoringError("result eval_run_id does not match immutable Eval plan")
    if request["submission_id"] != plan["submission_id"]:
        raise CanonicalPlanScoringError("result submission_id does not match immutable Eval plan")
    if request["agent_hash"] != plan["agent_hash"]:
        raise CanonicalPlanScoringError("result agent_hash does not match immutable Eval plan")
    # Host enforcement: success path must prove the guest executed the submitted
    # artifact (plan agent_hash). Wire keeps the field optional for legacy bodies.
    guest_proof = require_host_guest_artifact_proof(
        request,
        expected_agent_hash=plan["agent_hash"],
    )
    record = _validated_record(plan, request["score_record"])
    proof = request["execution_proof"]
    if proof["image_digest"] != plan["eval_app"]["image_ref"]:
        raise CanonicalPlanScoringError("result image digest does not match immutable Eval plan")
    expected_measurement = {
        "mrtd": plan["eval_app"]["measurement"]["mrtd"],
        "rtmr0": plan["eval_app"]["measurement"]["rtmr0"],
        "rtmr1": plan["eval_app"]["measurement"]["rtmr1"],
        "rtmr2": plan["eval_app"]["measurement"]["rtmr2"],
        "compose_hash": plan["eval_app"]["compose_hash"],
        "os_image_hash": plan["eval_app"]["measurement"]["os_image_hash"],
    }
    if {
        field: proof["attestation"]["measurement"][field] for field in expected_measurement
    } != expected_measurement:
        raise CanonicalPlanScoringError("result measurement does not match immutable Eval plan")
    task_ids = [task["task_id"] for task in plan["selected_tasks"]]
    # Production path: require schema-v2 score binding; hard-reject legacy
    # validator_nonce report_data with LEGACY_REPORT_DATA_REJECTED.
    require_schema_v2_score_report_data(
        reported_report_data=str(proof["attestation"]["report_data"]),
        canonical_measurement=expected_measurement,
        agent_hash=plan["agent_hash"],
        task_ids=task_ids,
        scores_digest=request["scores_digest"],
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        key_release_nonce=plan.get("key_release_nonce")
        if isinstance(plan.get("key_release_nonce"), str)
        else None,
        phala_attestation_enabled=True,
    )
    return {**request, "score_record": record, "guest_artifact_proof": guest_proof}


def persist_direct_eval_result(
    job: EvaluationJob,
    result_request: Mapping[str, Any],
) -> PlanFinalScore:
    """Validate and retain a direct result only against the persisted plan.

    The result endpoint owns authentication, receipt state, quote verification,
    nonce consumption, and transactional task-result writes.  This narrow
    deterministic step gives it one plan-backed source for score-record
    validation and final score/count persistence before those operations.
    """

    plan = load_eval_plan(job)
    if plan is None:
        raise CanonicalPlanScoringError("direct Eval result requires a persisted Eval plan")
    final = persist_direct_eval_result_from_plan(plan, result_request)
    persist_canonical_score_record(job, final.score_record)
    job.score = final.score
    job.passed_tasks = final.passed_tasks
    job.total_tasks = final.total_tasks
    return final


def persist_direct_eval_result_from_plan(
    eval_plan: Mapping[str, Any],
    result_request: Mapping[str, Any],
) -> PlanFinalScore:
    """Reconstruct an accepted direct score without creating a validator job."""

    plan = _validated_plan(eval_plan)
    request = validate_eval_result_from_plan(plan, result_request)
    record = request["score_record"]
    final = PlanFinalScore(
        score=ew.decode_score_f64be(record["final"]["job_score_f64be"]),
        passed_tasks=record["final"]["passed_tasks"],
        total_tasks=record["final"]["total_tasks"],
        score_record=record,
    )
    return final


def persist_canonical_score_record(
    job: EvaluationJob,
    score_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist a fully reconstructed score record only after plan validation."""

    plan = load_eval_plan(job)
    if plan is None:
        raise CanonicalPlanScoringError("legacy jobs cannot persist an attested score record")
    record = _validated_record(plan, score_record)
    job.canonical_score_record_json = ew.canonical_json_v1(record).decode("utf-8")
    job.canonical_score_record_sha256 = ew.score_record_digest(record)
    return record


def load_canonical_score_record(job: EvaluationJob) -> dict[str, Any] | None:
    """Return a plan-validated persisted score record, if the job has one."""

    raw = job.canonical_score_record_json
    if raw is None:
        return None
    plan = load_eval_plan(job)
    if plan is None:
        raise CanonicalPlanScoringError("canonical score record exists without an Eval plan")
    record = _validated_record(plan, _parse_json(raw, field="canonical_score_record_json"))
    if ew.canonical_json_v1(record).decode("utf-8") != raw:
        raise CanonicalPlanScoringError("persisted canonical score record is not canonical")
    if job.canonical_score_record_sha256 != ew.score_record_digest(record):
        raise CanonicalPlanScoringError("persisted canonical score record digest does not match")
    return record


def _selected_task_ids(plan: Mapping[str, Any]) -> list[str]:
    return [task["task_id"] for task in plan["selected_tasks"]]


def final_score_from_eval_plan(
    job: EvaluationJob,
    *,
    selected_task_ids: Sequence[str],
    task_scores: Mapping[str, float],
) -> PlanFinalScore | None:
    """Derive final scoring fields from a persisted plan or return ``None``.

    A persisted canonical score record is authoritative for an attested result.
    Before direct ingestion stores that record, a plan with ``k=1`` can still be
    finalized from one terminal score per selected task.  For ``k>1`` this refuses
    to collapse already-aggregated task rows, because that would lose the ordered
    trial evidence the plan requires.
    """

    plan = load_eval_plan(job)
    if plan is None:
        return None
    expected_task_ids = _selected_task_ids(plan)
    if sorted(selected_task_ids) != expected_task_ids or len(selected_task_ids) != len(
        set(selected_task_ids)
    ):
        raise CanonicalPlanScoringError("job selected tasks do not match immutable Eval plan")

    record = load_canonical_score_record(job)
    if record is None:
        if plan["k"] != 1:
            raise CanonicalPlanScoringError(
                "k>1 Eval plan requires a persisted canonical score record"
            )
        if set(task_scores) != set(expected_task_ids):
            raise CanonicalPlanScoringError(
                "terminal task scores do not cover immutable selected tasks"
            )
        record = build_score_record_from_eval_plan(
            plan,
            {task_id: [task_scores[task_id]] for task_id in expected_task_ids},
        )
    for task in record["tasks"]:
        try:
            persisted_score = ew.encode_score_f64be(task_scores[task["task_id"]])
        except (KeyError, ew.EvalWireError) as exc:
            raise CanonicalPlanScoringError(
                "terminal task score does not cover immutable selected tasks"
            ) from exc
        if persisted_score != task["aggregate_score_f64be"]:
            raise CanonicalPlanScoringError(
                "persisted task score does not match canonical Eval score record"
            )

    return PlanFinalScore(
        score=ew.decode_score_f64be(record["final"]["job_score_f64be"]),
        passed_tasks=record["final"]["passed_tasks"],
        total_tasks=record["final"]["total_tasks"],
        score_record=record,
    )


def aggregate_trial_scores_from_eval_plan(
    eval_plan: Mapping[str, Any],
    trial_scores: Sequence[float],
) -> float:
    """Aggregate one exact ordered task trial list under the plan's policy."""

    plan = _validated_plan(eval_plan)
    if len(trial_scores) != plan["k"]:
        raise CanonicalPlanScoringError("trial count does not match immutable Eval plan k")
    try:
        values = [ew.decode_score_f64be(ew.encode_score_f64be(score)) for score in trial_scores]
    except ew.EvalWireError as exc:
        raise CanonicalPlanScoringError(f"non-canonical trial score: {exc}") from exc
    if plan["scoring_policy"]["per_task_aggregation"] == "best_of_k":
        return max(values)
    return sum(values) / len(values)


def replay_score_from_eval_plan(
    eval_plan: Mapping[str, Any],
    trial_scores_by_task: Mapping[str, Sequence[float]],
) -> float:
    """Reconstruct the replay score from the exact immutable plan bytes."""

    record = build_score_record_from_eval_plan(eval_plan, trial_scores_by_task)
    return ew.decode_score_f64be(record["final"]["job_score_f64be"])


def plan_backed_job_is_consistent(job: EvaluationJob) -> bool:
    """Return whether a plan-backed job still matches canonical score evidence.

    Weight calculation is deliberately read-only.  It must never trust mutable
    ``score``/count columns for an attested job unless they reproduce the
    immutable plan-derived score record exactly.  Planless legacy jobs remain
    valid here, preserving their historical eligibility behavior.
    """

    try:
        record = load_canonical_score_record(job)
        if record is None:
            return job.eval_plan_json is None
        return (
            ew.encode_score_f64be(job.score) == record["final"]["job_score_f64be"]
            and job.passed_tasks == record["final"]["passed_tasks"]
            and job.total_tasks == record["final"]["total_tasks"]
        )
    except (CanonicalPlanScoringError, ew.EvalWireError):
        return False


__all__ = [
    "GUEST_ARTIFACT_PROOF_AGENT_HASH_MISMATCH",
    "GUEST_ARTIFACT_PROOF_HASH_MISMATCH",
    "GUEST_ARTIFACT_PROOF_MISSING",
    "LEGACY_REPORT_DATA_REJECTED",
    "CanonicalPlanScoringError",
    "PlanFinalScore",
    "aggregate_trial_scores_from_eval_plan",
    "build_score_record_from_eval_plan",
    "canonical_eval_plan_json",
    "final_score_from_eval_plan",
    "load_canonical_score_record",
    "load_eval_plan",
    "persist_canonical_score_record",
    "persist_direct_eval_result",
    "persist_direct_eval_result_from_plan",
    "persist_eval_plan",
    "plan_backed_job_is_consistent",
    "replay_score_from_eval_plan",
    "require_host_guest_artifact_proof",
    "require_schema_v2_score_report_data",
    "scoring_policy_from_settings",
    "validate_eval_result_from_plan",
    "validate_score_record_from_eval_plan",
]
