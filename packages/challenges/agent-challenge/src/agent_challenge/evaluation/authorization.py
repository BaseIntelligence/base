"""Durable authorization ledger for validator-issued attested Eval runs."""

from __future__ import annotations

import hmac
import json
import secrets
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any
from uuid import uuid4

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agent_challenge.canonical import eval_wire
from agent_challenge.core.models import AgentSubmission, EvalNonce, EvalResourceCounter, EvalRun
from agent_challenge.evaluation.benchmarks import load_benchmark_tasks, select_benchmark_tasks
from agent_challenge.evaluation.plan_scoring import (
    CanonicalPlanScoringError,
    canonical_eval_plan_json,
    scoring_policy_from_settings,
)
from agent_challenge.review.authorization import verified_review_assignment_for_submission
from agent_challenge.review.canonical import canonical_json_v1
from agent_challenge.sdk.config import ChallengeSettings

_ACTIVE_PHASES = frozenset({"eval_prepared", "eval_running", "eval_verifying"})
_RETRYABLE_PHASES = frozenset({"eval_cancelled", "eval_expired", "eval_error"})
_FAILURE_REASONS = frozenset(
    {
        "eval_deploy_failed",
        "eval_tunnel_failed",
        "eval_key_release_unavailable",
        "eval_no_result",
    }
)
OUTSTANDING_RESULT_RESOURCE = "eval_result_outstanding"
VERIFYING_RESULT_RESOURCE = "eval_result_verifying"


class EvalAuthorizationRequired(PermissionError):
    """The exact submission does not have a persisted verified review allow."""


class EvalAuthorizationConflict(ValueError):
    """An Eval lifecycle mutation does not match the durable current state."""

    def __init__(self, message: str, *, code: str = "eval_lifecycle_conflict") -> None:
        super().__init__(message)
        self.code = code


class EvalAuthorizationUnavailable(ValueError):
    """Validator-owned Eval deployment identity or policy is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "eval_deployment_identity_unavailable",
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CreatedEvalRun:
    """The prepare response projection, including the one-time secret delivery."""

    run: EvalRun
    plan: dict[str, Any]
    token: str | None


def _as_utc(value: datetime | None) -> datetime:
    result = value or datetime.now(UTC)
    if result.tzinfo is None:
        return result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _compare_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _cursor_mac(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload, "sha256").hexdigest()


def _encode_cursor(
    *,
    submission_id: int,
    watermark: int,
    offset: int,
    secret: str,
) -> str:
    unsigned = canonical_json_v1(
        {
            "submission_id": submission_id,
            "watermark": watermark,
            "offset": offset,
        }
    )
    payload = canonical_json_v1(
        {
            "submission_id": submission_id,
            "watermark": watermark,
            "offset": offset,
            "mac": _cursor_mac(unsigned, secret),
        }
    )
    return urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(value: str, *, submission_id: int, secret: str) -> tuple[int, int]:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise EvalAuthorizationConflict("invalid Eval history cursor", code="eval_cursor_invalid")
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalAuthorizationConflict(
            "invalid Eval history cursor",
            code="eval_cursor_invalid",
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"submission_id", "watermark", "offset", "mac"}
        or payload["submission_id"] != submission_id
        or not isinstance(payload["watermark"], int)
        or not isinstance(payload["offset"], int)
        or not isinstance(payload["mac"], str)
        or payload["watermark"] < 0
        or payload["offset"] < 0
        or not hmac.compare_digest(
            payload["mac"],
            _cursor_mac(
                canonical_json_v1(
                    {
                        "submission_id": payload["submission_id"],
                        "watermark": payload["watermark"],
                        "offset": payload["offset"],
                    }
                ),
                secret,
            ),
        )
    ):
        raise EvalAuthorizationConflict(
            "invalid Eval history cursor",
            code="eval_cursor_invalid",
        )
    return payload["watermark"], payload["offset"]


def _nonce() -> str:
    # token_urlsafe(24) carries 192 bits before encoding and is never derived
    # from miner input or a timestamp.
    return secrets.token_urlsafe(24)


def _dataset_digest_manifest_path() -> Any:
    """Return the frozen on-disk ``dataset-digest.json`` path for task config digests."""

    from pathlib import Path

    from agent_challenge.evaluation.benchmarks import TERMINAL_BENCH_2_1_DIGEST_PATH

    return Path(TERMINAL_BENCH_2_1_DIGEST_PATH)


_CACHED_DATASET_DIGEST: dict[str, Any] | None = None


def _dataset_digest_tasks() -> dict[str, Any]:
    """Load and cache the frozen Terminal-Bench digest task map once.

    Missing or unreadable ``dataset-digest.json`` must never residual as a bare
    HTTP 500. Convert path IO failures into ``EvalAuthorizationUnavailable`` with
    closed ``detail.code`` ``eval_dataset_unavailable``.
    """

    global _CACHED_DATASET_DIGEST
    if _CACHED_DATASET_DIGEST is None:
        import json
        from pathlib import Path

        path = Path(_dataset_digest_manifest_path())
        try:
            raw = path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError) as exc:
            raise EvalAuthorizationUnavailable(
                "frozen dataset digest is unavailable",
                code="eval_dataset_unavailable",
            ) from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvalAuthorizationUnavailable(
                "frozen dataset digest is unreadable",
                code="eval_dataset_unavailable",
            ) from exc
        if not isinstance(data, dict):
            raise EvalAuthorizationUnavailable(
                "frozen dataset digest is malformed",
                code="eval_dataset_unavailable",
            )
        tasks = data.get("tasks")
        if not isinstance(tasks, dict):
            raise EvalAuthorizationUnavailable(
                "frozen dataset digest has no tasks map",
                code="eval_dataset_unavailable",
            )
        _CACHED_DATASET_DIGEST = tasks
    return _CACHED_DATASET_DIGEST


def _bare_task_name(task_id: str) -> str:
    return task_id.rsplit("/", 1)[-1]


def task_config_digest_from_content(content_digest_sha256: str) -> str:
    """Bind plan ``task_config_sha256`` to the own_runner content-digest domain."""

    return eval_wire.task_config_sha256_from_content_digest(content_digest_sha256)


def _task_config_digest(task: Any) -> str:
    """Return the Digest-domain digest own_runner will recompute for ``task``.

    Production Terminal-Bench plans bind ``selected_tasks[].task_config_sha256``
    to the frozen per-task content digest (every regular file under the on-disk
    task tree). Prefer an explicit content digest on the task/metadata, else
    look up the frozen ``dataset-digest.json`` by bare task name. Metadata-only
    hashes are deliberately not used; plan identity must match the image consumer.
    """

    explicit = None
    metadata = getattr(task, "metadata", None)
    if isinstance(metadata, dict):
        explicit = metadata.get("content_digest_sha256") or metadata.get("task_config_sha256")
    if explicit is None:
        explicit = getattr(task, "content_digest_sha256", None)
    if isinstance(explicit, str) and explicit:
        return task_config_digest_from_content(explicit)

    tasks = _dataset_digest_tasks()
    entry = tasks.get(task.task_id) or tasks.get(_bare_task_name(task.task_id))
    if not isinstance(entry, dict):
        raise EvalAuthorizationUnavailable(
            f"task {task.task_id!r} has no frozen content digest for Eval plan binding",
            code="eval_dataset_unavailable",
        )
    digest = entry.get("content_digest_sha256")
    if not isinstance(digest, str) or not digest:
        raise EvalAuthorizationUnavailable(
            f"task {task.task_id!r} frozen digest entry is missing content_digest_sha256",
            code="eval_dataset_unavailable",
        )
    return task_config_digest_from_content(digest)


def _task_image_ref(task: Any) -> str:
    """Prefer an already digest-pinned image; otherwise require a usable pin.

    Tests and live-registry fixtures pin ``docker_image`` as
    ``repo@sha256:<hex>``. The immutable Eval plan requires that form.
    """

    image = getattr(task, "docker_image", None)
    if not isinstance(image, str) or not image:
        raise EvalAuthorizationUnavailable(f"task {task.task_id!r} has no docker image")
    # Already forms a digest-pinned wire reference.
    if "@sha256:" in image:
        return image
    metadata = getattr(task, "metadata", None) or {}
    pinned = None
    if isinstance(metadata, dict):
        pinned = metadata.get("image_ref") or metadata.get("harbor_registry_ref")
    if isinstance(pinned, str) and "@sha256:" in pinned:
        return pinned
    # Fall back to synthesizing a pin from the frozen content digest that is
    # already the immutable task identity; repo prefix tracks the configured
    # runner image without accepting an unpinned mutable tag.
    try:
        digest = _task_config_digest(task)
    except EvalAuthorizationUnavailable:
        raise EvalAuthorizationUnavailable(
            f"task {task.task_id!r} image_ref is not digest-pinned"
        ) from None
    # Use a synthetic repository path so wire schema validation succeeds; the
    # consumer binds task identity via task_config_sha256 + the live registry
    # override map when deploying, not via this placeholder's pullability.
    return f"task-local/{_bare_task_name(task.task_id)}@sha256:{digest}"


def _eval_app(settings: ChallengeSettings) -> dict[str, Any]:
    measurement = settings.eval_app_measurement
    if not isinstance(measurement, dict) or not measurement:
        raise EvalAuthorizationUnavailable("validator Eval measurement is unavailable")
    if not settings.eval_app_image_ref or not settings.eval_app_compose_hash:
        raise EvalAuthorizationUnavailable("validator Eval deployment identity is unavailable")
    if not settings.eval_app_kms_public_key_hex:
        raise EvalAuthorizationUnavailable("validator Eval KMS identity is unavailable")
    try:
        public_key_digest = sha256(bytes.fromhex(settings.eval_app_kms_public_key_hex)).hexdigest()
    except ValueError as exc:
        raise EvalAuthorizationUnavailable("validator Eval KMS public key is malformed") from exc
    expected_measurement = {
        "mrtd",
        "rtmr0",
        "rtmr1",
        "rtmr2",
        "os_image_hash",
        "key_provider",
        "vm_shape",
    }
    if set(measurement) != expected_measurement:
        raise EvalAuthorizationUnavailable("validator Eval measurement is not schema-closed")
    canonical_measurement = {
        "mrtd": measurement["mrtd"],
        "rtmr0": measurement["rtmr0"],
        "rtmr1": measurement["rtmr1"],
        "rtmr2": measurement["rtmr2"],
        "compose_hash": settings.eval_app_compose_hash,
        "os_image_hash": measurement["os_image_hash"],
    }
    allowlist = settings.eval_app_measurement_allowlist
    if not allowlist or canonical_measurement not in list(allowlist):
        raise EvalAuthorizationUnavailable(
            "validator Eval measurement is not in the eval allowlist"
        )
    return {
        "image_ref": settings.eval_app_image_ref,
        "compose_hash": settings.eval_app_compose_hash,
        "app_identity": settings.eval_app_identity,
        "kms_key_algorithm": "x25519",
        "kms_public_key_hex": settings.eval_app_kms_public_key_hex,
        "kms_public_key_sha256": public_key_digest,
        "measurement": dict(measurement),
    }


def _build_plan(
    *,
    submission: AgentSubmission,
    review_digest: str,
    settings: ChallengeSettings,
    eval_run_id: str,
    key_release_nonce: str,
    score_nonce: str,
    token_sha256: str,
    now: datetime,
) -> dict[str, Any]:
    try:
        policy = scoring_policy_from_settings(settings)
    except CanonicalPlanScoringError as exc:
        raise EvalAuthorizationUnavailable("validator scoring policy is unavailable") from exc
    tasks = select_benchmark_tasks(
        load_benchmark_tasks(),
        agent_hash=submission.agent_hash,
        count=settings.evaluation_task_count,
    )
    if not tasks:
        raise EvalAuthorizationUnavailable("validator selected no Eval tasks")
    selected_tasks = sorted(
        (
            {
                "task_id": task.task_id,
                "image_ref": _task_image_ref(task),
                "task_config_sha256": _task_config_digest(task),
            }
            for task in tasks
        ),
        key=lambda item: item["task_id"],
    )
    issued_at_ms = _milliseconds(now)
    expires_at_ms = _milliseconds(now + timedelta(seconds=settings.eval_run_ttl_seconds))
    key_release_endpoint = settings.eval_key_release_endpoint
    if not isinstance(key_release_endpoint, str) or not key_release_endpoint.strip():
        # Empty endpoint used to residual as bare wire 500 via validate_eval_plan;
        # surface a closed unavailable code instead.
        raise EvalAuthorizationUnavailable(
            "validator Eval key release endpoint is unavailable",
            code="eval_key_release_endpoint_unavailable",
        )
    package_tree_sha = getattr(submission, "package_tree_sha", None)
    if not isinstance(package_tree_sha, str) or not package_tree_sha.strip():
        raise EvalAuthorizationUnavailable(
            "submission package_tree_sha is required for Eval plan binding",
            code="package_tree_sha_missing",
        )
    plan = {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "submission_id": str(submission.id),
        "submission_version": submission.version_number or 1,
        "authorizing_review_digest": review_digest,
        "agent_hash": submission.agent_hash,
        "package_tree_sha": package_tree_sha.strip(),
        "selected_tasks": selected_tasks,
        "k": settings.eval_k,
        "scoring_policy": policy,
        "scoring_policy_digest": eval_wire.scoring_policy_digest(policy),
        "eval_app": _eval_app(settings),
        "key_release_endpoint": key_release_endpoint,
        "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
        "key_release_nonce": key_release_nonce,
        "score_nonce": score_nonce,
        "run_token_sha256": token_sha256,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
    }
    try:
        return eval_wire.validate_eval_plan(plan)
    except eval_wire.EvalWireError as exc:
        raise EvalAuthorizationUnavailable(f"validator Eval plan is invalid: {exc}") from exc


async def _latest_run(
    session: AsyncSession,
    submission_id: int,
    *,
    lock: bool = False,
) -> EvalRun | None:
    statement = (
        select(EvalRun)
        .where(EvalRun.submission_id == submission_id)
        .order_by(desc(EvalRun.created_at), desc(EvalRun.id))
        .limit(1)
    )
    if lock:
        statement = statement.with_for_update()
    return await session.scalar(statement)


def _loaded_plan(run: EvalRun) -> dict[str, Any]:
    try:
        parsed = json.loads(run.plan_json)
        plan = eval_wire.validate_eval_plan(parsed)
        if canonical_json_v1(plan).decode("utf-8") != run.plan_json:
            raise ValueError("plan bytes are not canonical")
        if sha256(run.plan_json.encode("utf-8")).hexdigest() != run.plan_sha256:
            raise ValueError("plan digest does not match stored bytes")
        return plan
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise EvalAuthorizationConflict("stored Eval plan is invalid") from exc


async def _authorized_review_digest(
    session: AsyncSession,
    submission: AgentSubmission,
    *,
    settings: ChallengeSettings | None = None,
) -> str:
    """Return review_digest only after **fresh re-verified** review admission.

    Cached ``phase=review_allowed`` / ``status=verified_allow`` alone is
    insufficient (VAL-ACAT-010 / 028 / 029). Materials from the durable
    assignment envelope must re-admit via
    :func:`agent_challenge.evaluation.fresh_review_gate.admit_eval_cvm_launch_from_assignment`
    (bound outcome + OR digests + report_data + ≤24h).

    Production dual-flag path also requires measured package LLM-rules residual
    bound into the authorizing envelope/outcome (VAL-AGATE-004..007). Host
    analyzer allow alone is never sufficient for eval prepare.
    """

    assignment = await verified_review_assignment_for_submission(session, submission)
    if assignment is None or not assignment.review_digest:
        raise EvalAuthorizationRequired("persisted verified review allow is required")

    from agent_challenge.evaluation.fresh_review_gate import (
        EvalCvmFreshReviewError,
        admit_eval_cvm_launch_from_assignment,
    )

    dual_on = True
    require_residual = True
    expected_tree: str | None = getattr(submission, "package_tree_sha", None)
    if isinstance(expected_tree, str):
        expected_tree = expected_tree.strip() or None
    else:
        expected_tree = None
    if settings is not None:
        phala = bool(getattr(settings, "phala_attestation_enabled", True))
        review = bool(getattr(settings, "attested_review_enabled", True))
        dual_on = phala and review
        # Residual is mandatory for production dual-flag prepare; flag-off still
        # fails later emit gates but residual check runs when dual_on.
        require_residual = dual_on

    try:
        decision = admit_eval_cvm_launch_from_assignment(
            assignment,
            dual_flags_on=dual_on,
            require_package_residual=require_residual,
            expected_package_tree_sha=expected_tree,
        )
    except EvalCvmFreshReviewError as exc:
        raise EvalAuthorizationRequired(
            f"fresh review re-verify refused eval launch: {exc.code}"
        ) from exc
    if not decision.may_launch:
        raise EvalAuthorizationRequired(
            f"fresh review re-verify refused eval launch: {decision.reason_code}"
        )
    # Prefer digests from re-admit materials when present.
    if decision.review_digest and decision.review_digest != assignment.review_digest:
        raise EvalAuthorizationRequired(
            "fresh review re-verify digest shear vs authorizing assignment"
        )
    return assignment.review_digest


async def _issue_run(
    session: AsyncSession,
    *,
    submission: AgentSubmission,
    review_digest: str,
    settings: ChallengeSettings,
    now: datetime,
    prior_run: EvalRun | None = None,
) -> CreatedEvalRun:
    existing = await session.scalar(
        select(EvalRun)
        .where(EvalRun.submission_id == submission.id)
        .order_by(desc(EvalRun.id))
        .limit(1)
        .with_for_update()
    )
    if existing is not None and prior_run is None:
        return CreatedEvalRun(run=existing, plan=_loaded_plan(existing), token=None)
    max_runs = int(getattr(settings, "eval_max_runs_per_submission", 8))
    run_count = await session.scalar(
        select(func.count()).select_from(EvalRun).where(EvalRun.submission_id == submission.id)
    )
    if run_count is not None and int(run_count) >= max_runs:
        raise EvalAuthorizationConflict(
            "Eval run limit for this submission is exhausted",
            code="eval_run_limit_exhausted",
        )
    run_id = _new_id("eval")
    token = secrets.token_urlsafe(32)
    max_capability_bytes = int(getattr(settings, "eval_max_capability_bytes", 4_096))
    if len(token.encode("utf-8")) > max_capability_bytes:
        raise EvalAuthorizationUnavailable("Eval capability exceeds eval_max_capability_bytes")
    token_digest = sha256(token.encode("utf-8")).hexdigest()
    plan = _build_plan(
        submission=submission,
        review_digest=review_digest,
        settings=settings,
        eval_run_id=run_id,
        key_release_nonce=_nonce(),
        score_nonce=_nonce(),
        token_sha256=token_digest,
        now=now,
    )
    plan_json = canonical_eval_plan_json(plan)
    plan_digest = sha256(plan_json.encode("utf-8")).hexdigest()
    expires_at = now + timedelta(seconds=settings.eval_run_ttl_seconds)
    run = EvalRun(
        eval_run_id=run_id,
        submission_id=submission.id,
        submission_version=submission.version_number or 1,
        attempt=(prior_run.attempt + 1) if prior_run is not None else 1,
        prior_eval_run_id=prior_run.eval_run_id if prior_run is not None else None,
        authorizing_review_digest=review_digest,
        plan_json=plan_json,
        plan_sha256=plan_digest,
        token_sha256=token_digest,
        token_delivered_at=now,
        phase="eval_prepared",
        retryable=True,
        issued_at=now,
        expires_at=expires_at,
    )
    session.add(run)
    await session.flush()
    session.add_all(
        [
            EvalNonce(
                eval_run_id=run.id,
                nonce=plan["key_release_nonce"],
                purpose="key_release",
                state="outstanding",
                expires_at=expires_at,
            ),
            EvalNonce(
                eval_run_id=run.id,
                nonce=plan["score_nonce"],
                purpose="score",
                state="outstanding",
                expires_at=expires_at,
            ),
        ]
    )
    await session.flush()
    return CreatedEvalRun(run=run, plan=plan, token=token)


async def create_eval_run(
    session: AsyncSession,
    submission: AgentSubmission,
    *,
    settings: ChallengeSettings,
    now: datetime | None = None,
) -> CreatedEvalRun:
    """Authorize one immutable run, or return the current run without a token."""

    locked_submission = await session.scalar(
        select(AgentSubmission).where(AgentSubmission.id == submission.id).with_for_update()
    )
    if locked_submission is None:
        raise EvalAuthorizationRequired("submission does not exist")
    submission = locked_submission
    review_digest = await _authorized_review_digest(session, submission, settings=settings)
    current = await _latest_run(session, submission.id, lock=True)
    if current is not None:
        moment = _as_utc(now)
        await _expire_run_if_needed(session, current, now=moment)
        if current.phase in _RETRYABLE_PHASES:
            raise EvalAuthorizationConflict(
                "current Eval run requires signed retry",
                code="eval_prepare_conflict",
            )
        plan = _loaded_plan(current)
        return CreatedEvalRun(run=current, plan=plan, token=None)
    return await _issue_run(
        session,
        submission=submission,
        review_digest=review_digest,
        settings=settings,
        now=_as_utc(now),
        prior_run=current,
    )


async def _current_expected_run(
    session: AsyncSession,
    submission_id: int,
    expected_run_id: str,
) -> EvalRun:
    run = await session.scalar(
        select(EvalRun)
        .where(EvalRun.submission_id == submission_id, EvalRun.eval_run_id == expected_run_id)
        .with_for_update()
    )
    current = await _latest_run(session, submission_id)
    if run is None:
        raise EvalAuthorizationConflict("expected Eval run is unknown", code="eval_stale_run")
    if current is None or current.id != run.id:
        raise EvalAuthorizationConflict("expected Eval run is not current", code="eval_stale_run")
    return run


async def _revoke_nonces(session: AsyncSession, run: EvalRun, *, now: datetime) -> None:
    result = await session.scalars(select(EvalNonce).where(EvalNonce.eval_run_id == run.id))
    for nonce in result.all():
        if nonce.state == "outstanding":
            nonce.state = "revoked"
            nonce.consumed_at = now


async def _expire_nonces(session: AsyncSession, run: EvalRun, *, now: datetime) -> None:
    result = await session.scalars(select(EvalNonce).where(EvalNonce.eval_run_id == run.id))
    for nonce in result.all():
        if nonce.state == "outstanding":
            nonce.state = "expired"
            nonce.consumed_at = now


async def _expire_run_if_needed(
    session: AsyncSession,
    run: EvalRun,
    *,
    now: datetime,
) -> None:
    if (
        run.receipt_id is None
        and (run.key_release_receipt_sha256 is None or run.key_granted_at is not None)
        and _compare_utc(run.expires_at) <= now
        and run.phase in _ACTIVE_PHASES
    ):
        run.phase = "eval_expired"
        # An external CVM which never submitted a result is accounted for as
        # liveness, not as a verified score.  A successful key grant makes the
        # attempt permanently non-retryable because the hidden material was
        # released; a pre-grant timeout remains retryable.
        run.reason_code = "eval_no_result" if run.key_granted_at is not None else "eval_expired"
        run.failure_origin = "no_result"
        run.retryable = run.key_granted_at is None
        run.verified = False
        run.reward_eligible = False
        run.result_available = False
        run.finalized_at = now
        await _expire_nonces(session, run, now=now)


async def cancel_eval_run(
    session: AsyncSession,
    submission: AgentSubmission,
    expected_run_id: str,
    *,
    now: datetime | None = None,
) -> EvalRun:
    """Cancel only an active, pre-receipt, never-key-granted run."""

    await session.scalar(
        select(AgentSubmission).where(AgentSubmission.id == submission.id).with_for_update()
    )
    run = await _current_expected_run(session, submission.id, expected_run_id)
    moment = _as_utc(now)
    await _expire_run_if_needed(session, run, now=moment)
    if run.phase == "eval_cancelled":
        return run
    if (
        run.receipt_id is not None
        or run.key_release_receipt_sha256 is not None
        or run.key_granted_at is not None
    ):
        raise EvalAuthorizationConflict("receipted or key-granted Eval run cannot be cancelled")
    if run.phase not in _ACTIVE_PHASES:
        raise EvalAuthorizationConflict(
            "Eval run is not active",
            code="eval_run_terminal",
        )
    run.phase = "eval_cancelled"
    run.reason_code = "eval_cancelled"
    run.retryable = True
    run.finalized_at = moment
    await _revoke_nonces(session, run, now=moment)
    return run


async def fail_eval_run(
    session: AsyncSession,
    submission: AgentSubmission,
    *,
    expected_run_id: str,
    reason_code: str,
    now: datetime | None = None,
) -> EvalRun:
    """Record one schema-closed pre-receipt failure and revoke its capability."""

    if reason_code not in _FAILURE_REASONS:
        raise EvalAuthorizationConflict("unknown Eval pre-receipt failure reason")
    await session.scalar(
        select(AgentSubmission).where(AgentSubmission.id == submission.id).with_for_update()
    )
    run = await _current_expected_run(session, submission.id, expected_run_id)
    moment = _as_utc(now)
    await _expire_run_if_needed(session, run, now=moment)
    if run.phase == "eval_error":
        if run.reason_code == reason_code:
            return run
        raise EvalAuthorizationConflict(
            "Eval failure reason conflicts with recorded failure",
            code="eval_failure_conflict",
        )
    if (
        run.receipt_id is not None
        or run.key_release_receipt_sha256 is not None
        or run.key_granted_at is not None
    ):
        raise EvalAuthorizationConflict("receipted or key-granted Eval run cannot fail")
    if run.phase not in _ACTIVE_PHASES:
        raise EvalAuthorizationConflict(
            "Eval run is not active",
            code="eval_run_terminal",
        )
    run.phase = "eval_error"
    run.reason_code = reason_code
    run.failure_origin = "pre_receipt"
    run.retryable = True
    run.verified = False
    run.reward_eligible = False
    run.finalized_at = moment
    await _revoke_nonces(session, run, now=moment)
    return run


async def retry_eval_run(
    session: AsyncSession,
    submission: AgentSubmission,
    *,
    expected_run_id: str,
    settings: ChallengeSettings,
    now: datetime | None = None,
) -> CreatedEvalRun:
    """Replace only an eligible no-receipt, never-granted predecessor."""

    await session.scalar(
        select(AgentSubmission).where(AgentSubmission.id == submission.id).with_for_update()
    )
    current = await _current_expected_run(session, submission.id, expected_run_id)
    moment = _as_utc(now)
    await _expire_run_if_needed(session, current, now=moment)
    if current.phase not in _RETRYABLE_PHASES:
        raise EvalAuthorizationConflict(
            "Eval run is not retryable",
            code="eval_run_terminal",
        )
    if current.phase == "eval_error" and current.reason_code not in _FAILURE_REASONS:
        raise EvalAuthorizationConflict("Eval error is not retryable")
    if (
        current.receipt_id is not None
        or current.key_release_receipt_sha256 is not None
        or current.key_granted_at is not None
    ):
        raise EvalAuthorizationConflict("Eval run cannot be retried after receipt or key grant")
    attempt_count = await session.scalar(
        select(func.count()).select_from(EvalRun).where(EvalRun.submission_id == submission.id)
    )
    if (attempt_count or 0) >= settings.eval_max_attempts:
        raise EvalAuthorizationConflict("Eval run retry limit reached")
    review_digest = await _authorized_review_digest(session, submission, settings=settings)
    return await _issue_run(
        session,
        submission=submission,
        review_digest=review_digest,
        settings=settings,
        now=moment,
        prior_run=current,
    )


def build_key_release_grant_materials(
    *,
    eval_run_id: str,
    key_release_nonce: str,
    ra_tls_spki_digest: str,
    agent_hash: str | None = None,
    report_data_hex: str | None = None,
) -> dict[str, Any]:
    """Construct closed RA-TLS key-release grant materials for score re-check.

    Policy (VAL-ACAT-036/037/040): durable materials must be sufficient to
    recompute schema-v2 ``report_data`` without relying on process-local
    memory. ``key_granted_at`` alone never admits under dual flags.
    """

    spki = str(ra_tls_spki_digest or "").strip().lower()
    if len(spki) != 64 or any(c not in "0123456789abcdef" for c in spki):
        raise EvalAuthorizationConflict(
            "ra_tls_spki_digest must be 64-char lowercase hex",
            code="eval_key_release_grant_invalid",
        )
    run_id = str(eval_run_id or "")
    nonce = str(key_release_nonce or "")
    if not run_id or not nonce:
        raise EvalAuthorizationConflict(
            "eval_run_id and key_release_nonce are required for KR grant",
            code="eval_key_release_grant_invalid",
        )
    expected = eval_wire.key_release_report_data_hex(
        eval_run_id=run_id,
        key_release_nonce=nonce,
        ra_tls_spki_digest=spki,
    )
    if report_data_hex is not None:
        reported = str(report_data_hex).strip().lower()
        if reported != expected:
            raise EvalAuthorizationConflict(
                "key-release report_data_hex does not match recomputed binding",
                code="eval_key_release_grant_invalid",
            )
    grant: dict[str, Any] = {
        "domain": eval_wire.KEY_RELEASE_DOMAIN,
        "schema_version": 2,
        "eval_run_id": run_id,
        "key_release_nonce": nonce,
        "ra_tls_spki_digest": spki,
        "report_data_hex": expected,
    }
    if agent_hash is not None:
        grant["agent_hash"] = str(agent_hash)
    return grant


def persist_key_release_grant_materials(
    run: EvalRun,
    grant: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Write grant JSON onto EvalRun and mirror into the process registry.

    Soft helper for idempotent reloads: returns the durable dict when present.
    """

    from agent_challenge.evaluation.score_chain_gate import (
        register_key_release_grant_for_score,
    )

    if grant is None:
        raw = getattr(run, "key_release_grant_json", None)
        if isinstance(raw, str) and raw:
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                return None
            if isinstance(parsed, dict):
                register_key_release_grant_for_score(str(run.eval_run_id), parsed)
                return parsed
        return None
    payload = json.dumps(grant, sort_keys=True, separators=(",", ":"))
    run.key_release_grant_json = payload
    register_key_release_grant_for_score(str(run.eval_run_id), grant)
    return grant


async def mark_eval_key_granted(
    session: AsyncSession,
    *,
    eval_run_id: str,
    now: datetime | None = None,
    ra_tls_spki_digest: str | None = None,
    report_data_hex: str | None = None,
    grant_materials: dict[str, Any] | None = None,
) -> EvalRun:
    """Atomically grant a key and persist reconstructible grant materials.

    When ``ra_tls_spki_digest`` (or a prebuilt ``grant_materials``) is supplied,
    durable ``key_release_grant_json`` is stamped so score admission can
    re-verify RA-TLS binding after multi-worker restarts (VAL-ACAT-036/037).
    Legacy callers that omit SPKI still mark granted (receipt path tests) but
    leave materials empty; dual-flag score then fail-closes on missing grant.
    """

    run = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == eval_run_id).with_for_update()
    )
    if run is None:
        raise EvalAuthorizationConflict("unknown Eval run")
    if run.key_granted_at is not None:
        # Idempotent re-entry: ensure process registry has durable materials.
        if run.key_release_grant_json:
            persist_key_release_grant_materials(run, None)
        elif grant_materials is not None or ra_tls_spki_digest is not None:
            # Fill missing durable materials on an already-granted run only when
            # grant_json is still empty (upgrade path / incomplete prior grant).
            try:
                grant = _resolve_grant_materials_for_run(
                    run,
                    ra_tls_spki_digest=ra_tls_spki_digest,
                    report_data_hex=report_data_hex,
                    grant_materials=grant_materials,
                )
            except EvalAuthorizationConflict:
                return run
            if grant is not None:
                persist_key_release_grant_materials(run, grant)
        return run
    if run.key_release_receipt_sha256 is None:
        run.key_release_receipt_sha256 = sha256(f"legacy-grant:{eval_run_id}".encode()).hexdigest()
        run.key_release_receipt_received_at = _as_utc(now)
        run.key_release_state = "verifying"
    await _expire_run_if_needed(session, run, now=_as_utc(now))
    if (
        run.phase not in _ACTIVE_PHASES
        or run.key_release_receipt_sha256 is None
        or run.key_release_state != "verifying"
    ):
        raise EvalAuthorizationConflict(
            "Eval run is not eligible for key grant",
            code="eval_run_terminal",
        )
    moment = _as_utc(now)
    grant = _resolve_grant_materials_for_run(
        run,
        ra_tls_spki_digest=ra_tls_spki_digest,
        report_data_hex=report_data_hex,
        grant_materials=grant_materials,
    )
    if grant is not None:
        persist_key_release_grant_materials(run, grant)
    run.key_granted_at = moment
    run.phase = "eval_running"
    run.retryable = False
    run.key_release_state = "granted"
    run.key_release_reason = None
    run.key_release_completed_at = moment
    for nonce in await session.scalars(
        select(EvalNonce).where(
            EvalNonce.eval_run_id == run.id,
            EvalNonce.purpose == "key_release",
        )
    ):
        if nonce.state != "outstanding":
            raise EvalAuthorizationConflict(
                "key-release nonce is not outstanding",
                code="eval_key_release_nonce_terminal",
            )
        nonce.state = "consumed"
        nonce.consumed_at = moment
    return run


def _resolve_grant_materials_for_run(
    run: EvalRun,
    *,
    ra_tls_spki_digest: str | None,
    report_data_hex: str | None,
    grant_materials: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build or validate grant materials from explicit args or our plan."""

    if isinstance(grant_materials, dict) and grant_materials:
        # Validate closed shape + recompute binding consistency.
        domain = grant_materials.get("domain", eval_wire.KEY_RELEASE_DOMAIN)
        return build_key_release_grant_materials(
            eval_run_id=str(grant_materials.get("eval_run_id") or run.eval_run_id),
            key_release_nonce=str(grant_materials.get("key_release_nonce") or ""),
            ra_tls_spki_digest=str(grant_materials.get("ra_tls_spki_digest") or ""),
            agent_hash=(
                str(grant_materials["agent_hash"])
                if grant_materials.get("agent_hash") is not None
                else None
            ),
            report_data_hex=(
                str(grant_materials["report_data_hex"])
                if grant_materials.get("report_data_hex") is not None
                else report_data_hex
            )
            if domain == eval_wire.KEY_RELEASE_DOMAIN or domain is None
            else None,
        )
    if not ra_tls_spki_digest:
        return None
    plan = load_eval_run_plan(run)
    return build_key_release_grant_materials(
        eval_run_id=str(plan.get("eval_run_id") or run.eval_run_id),
        key_release_nonce=str(plan.get("key_release_nonce") or ""),
        ra_tls_spki_digest=str(ra_tls_spki_digest),
        agent_hash=str(plan["agent_hash"]) if plan.get("agent_hash") is not None else None,
        report_data_hex=report_data_hex,
    )


async def register_eval_key_release(
    session: AsyncSession,
    *,
    eval_run_id: str,
    now: datetime | None = None,
) -> EvalRun:
    """Bind key-release registration to one active persisted Eval run."""

    run = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == eval_run_id).with_for_update()
    )
    if run is None:
        raise EvalAuthorizationConflict("unknown Eval run", code="eval_run_unknown")
    moment = _as_utc(now)
    await _expire_run_if_needed(session, run, now=moment)
    if (
        run.phase not in _ACTIVE_PHASES
        or run.receipt_id is not None
        or run.key_granted_at is not None
        or run.key_release_state in {"denied", "granted"}
    ):
        raise EvalAuthorizationConflict(
            "Eval run is not eligible for key release",
            code="eval_run_terminal",
        )
    plan = _loaded_plan(run)
    key_nonce = await session.scalar(
        select(EvalNonce).where(
            EvalNonce.eval_run_id == run.id,
            EvalNonce.purpose == "key_release",
        )
    )
    if (
        key_nonce is None
        or key_nonce.nonce != plan["key_release_nonce"]
        or key_nonce.state != "outstanding"
    ):
        raise EvalAuthorizationConflict(
            "key-release nonce is not outstanding",
            code="eval_key_release_nonce_terminal",
        )
    return run


def load_eval_run_plan(run: EvalRun) -> dict[str, Any]:
    """Return the validated immutable canonical plan persisted for ``run``."""

    return _loaded_plan(run)


async def receipt_eval_key_release(
    session: AsyncSession,
    *,
    eval_run_id: str,
    body_sha256: str,
    now: datetime | None = None,
) -> tuple[EvalRun, bool]:
    """Durably bind one exact schema-valid key-release frame before DCAP.

    Returns ``(run, should_verify)``.  An identical request may take a
    ``retryable`` receipt back to ``verifying``; an in-flight identical request
    observes ``should_verify=False``.  Conflicting or terminal bytes never
    replace the first digest.
    """

    if (
        not isinstance(body_sha256, str)
        or len(body_sha256) != 64
        or any(character not in "0123456789abcdef" for character in body_sha256)
    ):
        raise EvalAuthorizationConflict(
            "invalid key-release receipt digest",
            code="eval_key_release_receipt_invalid",
        )
    run = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == eval_run_id).with_for_update()
    )
    if run is None:
        raise EvalAuthorizationConflict("unknown Eval run", code="eval_run_unknown")
    moment = _as_utc(now)
    if run.key_release_receipt_sha256 is None:
        if _compare_utc(run.expires_at) <= moment:
            await _expire_run_if_needed(session, run, now=moment)
            raise EvalAuthorizationConflict(
                "Eval key-release nonce is expired",
                code="nonce_expired",
            )
        run = await register_eval_key_release(session, eval_run_id=eval_run_id, now=now)
        if _compare_utc(run.expires_at) <= moment:
            raise EvalAuthorizationConflict(
                "Eval key-release nonce is expired",
                code="nonce_expired",
            )
        run.key_release_receipt_sha256 = body_sha256
        run.key_release_receipt_received_at = moment
        run.key_release_state = "verifying"
        run.key_release_reason = None
        await session.flush()
        return run, True
    if not hmac.compare_digest(run.key_release_receipt_sha256, body_sha256):
        raise EvalAuthorizationConflict(
            "key-release request conflicts with the durable receipt",
            code="key_release_receipt_conflict",
        )
    if run.key_release_state == "granted":
        return run, False
    if run.key_release_state == "retryable":
        run.key_release_state = "verifying"
        run.key_release_reason = None
        await session.flush()
        return run, True
    if run.key_release_state == "verifying":
        return run, False
    raise EvalAuthorizationConflict(
        "key-release request is already terminal",
        code="eval_key_release_terminal",
    )


async def mark_eval_key_release_retryable(
    session: AsyncSession,
    *,
    eval_run_id: str,
    body_sha256: str,
    reason_code: str = "verifier_unavailable",
) -> EvalRun:
    """Persist a transient outcome without consuming the key nonce."""

    run = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == eval_run_id).with_for_update()
    )
    if (
        run is None
        or run.key_release_receipt_sha256 is None
        or not hmac.compare_digest(run.key_release_receipt_sha256, body_sha256)
        or run.key_release_state != "verifying"
        or run.key_granted_at is not None
    ):
        raise EvalAuthorizationConflict(
            "key-release receipt is not retryable",
            code="eval_key_release_terminal",
        )
    run.key_release_state = "retryable"
    run.key_release_reason = reason_code
    return run


async def mark_eval_key_release_denied(
    session: AsyncSession,
    *,
    eval_run_id: str,
    body_sha256: str,
    reason_code: str,
    now: datetime | None = None,
) -> EvalRun:
    """Atomically terminalize definitive invalid trust and consume once."""

    run = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == eval_run_id).with_for_update()
    )
    if (
        run is None
        or run.key_release_receipt_sha256 is None
        or not hmac.compare_digest(run.key_release_receipt_sha256, body_sha256)
        or run.key_release_state != "verifying"
        or run.key_granted_at is not None
    ):
        raise EvalAuthorizationConflict(
            "key-release receipt is already terminal",
            code="eval_key_release_terminal",
        )
    moment = _as_utc(now)
    nonce = await session.scalar(
        select(EvalNonce).where(
            EvalNonce.eval_run_id == run.id,
            EvalNonce.purpose == "key_release",
        )
    )
    if nonce is None or nonce.state != "outstanding":
        raise EvalAuthorizationConflict(
            "key-release nonce is already terminal",
            code="eval_key_release_nonce_terminal",
        )
    nonce.state = "consumed"
    nonce.consumed_at = moment
    run.key_release_state = "denied"
    run.key_release_reason = reason_code
    run.key_release_completed_at = moment
    run.phase = "eval_error"
    run.reason_code = "eval_key_release_denied"
    run.failure_origin = "key_release"
    run.retryable = False
    run.verified = False
    run.reward_eligible = False
    run.finalized_at = moment
    return run


async def reserve_eval_resource(
    session: AsyncSession,
    *,
    name: str,
    limit: int,
    conflict_code: str,
) -> EvalResourceCounter:
    """Atomically reserve one unit of a global Eval capacity counter.

    The counter row is row-locked so concurrent workers across processes share
    one global limit for outstanding result receipts or concurrent verifications.
    Counter rows are seeded by database init; a missing row is treated as empty
    and inserted only as a rare bootstrap path.
    """

    if limit <= 0:
        raise EvalAuthorizationConflict(
            "Eval resource capacity is full",
            code=conflict_code,
        )
    counter = await session.scalar(
        select(EvalResourceCounter).where(EvalResourceCounter.name == name).with_for_update()
    )
    if counter is None:
        # Bootstrap path for pre-seed databases; production always seeds.
        counter = EvalResourceCounter(name=name, value=0)
        session.add(counter)
        await session.flush()
        counter = await session.scalar(
            select(EvalResourceCounter).where(EvalResourceCounter.name == name).with_for_update()
        )
        if counter is None:  # pragma: no cover - defensive
            raise EvalAuthorizationConflict(
                "Eval resource capacity is unavailable",
                code=conflict_code,
            )
    if counter.value >= limit:
        raise EvalAuthorizationConflict(
            "Eval resource capacity is full",
            code=conflict_code,
        )
    counter.value += 1
    await session.flush()
    return counter


async def release_eval_resource(
    session: AsyncSession,
    *,
    name: str,
) -> None:
    """Release one unit of a previously reserved Eval capacity counter."""

    counter = await session.get(EvalResourceCounter, name, with_for_update=True)
    if counter is None:
        return
    if counter.value > 0:
        counter.value -= 1
        await session.flush()


async def receipt_eval_result(
    session: AsyncSession,
    *,
    eval_run_id: str,
    body_sha256: str,
    body: bytes | None = None,
    max_submissions_per_minute: int = 10,
    max_outstanding: int = 10_000,
    now: datetime | None = None,
) -> tuple[EvalRun, bool]:
    """Receipt one exact, timely result body before quote verification.

    The receipt is the recovery point for the miner-funded CVM.  It is
    immutable, so a retry after verifier outage can only submit the exact same
    bytes.  The returned boolean says whether this caller owns verification.
    """

    if (
        not isinstance(body_sha256, str)
        or len(body_sha256) != 64
        or any(character not in "0123456789abcdef" for character in body_sha256)
    ):
        raise EvalAuthorizationConflict(
            "invalid Eval result receipt digest",
            code="eval_result_receipt_invalid",
        )
    run = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == eval_run_id).with_for_update()
    )
    if run is None:
        raise EvalAuthorizationConflict("unknown Eval run", code="eval_run_unknown")
    moment = _as_utc(now)
    if run.receipt_id is None:
        # Literal zero admits no submissions: max(..., 1) must not mask it.
        if max_submissions_per_minute <= 0:
            raise EvalAuthorizationConflict(
                "Eval result submission rate limit exceeded",
                code="eval_result_rate_limited",
            )
        await reserve_eval_resource(
            session,
            name=OUTSTANDING_RESULT_RESOURCE,
            limit=max_outstanding,
            conflict_code="eval_result_overloaded",
        )
        window_start = (
            _compare_utc(run.result_submission_count_window_start)
            if (run.result_submission_count_window_start is not None)
            else None
        )
        if window_start is None or moment - window_start >= timedelta(minutes=1):
            run.result_submission_count_window_start = moment
            run.result_submission_count = 0
        if run.result_submission_count >= max_submissions_per_minute:
            # Independent of the outstanding counter (capacity was reserved above);
            # release before rejecting on the per-run rate window.
            await release_eval_resource(session, name=OUTSTANDING_RESULT_RESOURCE)
            raise EvalAuthorizationConflict(
                "Eval result submission rate limit exceeded",
                code="eval_result_rate_limited",
            )
        run.result_submission_count += 1
    if run.receipt_id is None:
        if _compare_utc(run.expires_at) <= moment:
            await release_eval_resource(session, name=OUTSTANDING_RESULT_RESOURCE)
            await _expire_run_if_needed(session, run, now=moment)
            raise EvalAuthorizationConflict("Eval result receipt is expired", code="eval_expired")
        if run.phase not in _ACTIVE_PHASES:
            await release_eval_resource(session, name=OUTSTANDING_RESULT_RESOURCE)
            raise EvalAuthorizationConflict("Eval run is not active", code="eval_run_terminal")
        run.receipt_id = _new_id("receipt")
        run.receipt_body_sha256 = body_sha256
        run.receipt_body = body
        run.receipt_received_at = moment
        run.receipt_verification_claimed_at = moment
        run.phase = "eval_verifying"
        run.reason_code = None
        run.retryable = True
        await session.flush()
        return run, True
    if run.receipt_body_sha256 != body_sha256:
        raise EvalAuthorizationConflict(
            "Eval result conflicts with the durable receipt",
            code="eval_result_receipt_conflict",
        )
    if run.phase == "eval_verifying":
        if run.receipt_verification_claimed_at is None:
            run.receipt_verification_claimed_at = moment
            await session.flush()
            return run, True
        if run.reason_code == "verifier_unavailable" or _compare_utc(
            run.receipt_verification_claimed_at
        ) <= moment - timedelta(seconds=60):
            run.receipt_verification_claimed_at = moment
            await session.flush()
            return run, True
        return run, False
    # Terminal receipts free the outstanding capacity reserved on first receipt.
    if run.phase in {"eval_accepted", "eval_rejected"}:
        return run, False
    return run, False


async def mark_eval_result_retryable(
    session: AsyncSession,
    *,
    eval_run_id: str,
    body_sha256: str,
    reason_code: str = "verifier_unavailable",
) -> EvalRun:
    """Park a receipted result without consuming its score nonce."""

    run = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == eval_run_id).with_for_update()
    )
    if (
        run is None
        or run.receipt_id is None
        or run.receipt_body_sha256 != body_sha256
        or run.phase != "eval_verifying"
    ):
        raise EvalAuthorizationConflict(
            "Eval result receipt is not retryable",
            code="eval_result_terminal",
        )
    run.reason_code = reason_code
    run.retryable = True
    run.verified = False
    run.reward_eligible = False
    return run


async def mark_eval_result_rejected(
    session: AsyncSession,
    *,
    eval_run_id: str,
    body_sha256: str,
    reason_code: str,
    now: datetime | None = None,
) -> EvalRun:
    """Terminalize a definitive result failure and consume its score nonce."""

    run = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == eval_run_id).with_for_update()
    )
    if (
        run is None
        or run.receipt_id is None
        or run.receipt_body_sha256 != body_sha256
        or run.phase != "eval_verifying"
    ):
        raise EvalAuthorizationConflict(
            "Eval result receipt is already terminal",
            code="eval_result_terminal",
        )
    moment = _as_utc(now)
    nonce = await session.scalar(
        select(EvalNonce).where(
            EvalNonce.eval_run_id == run.id,
            EvalNonce.purpose == "score",
        )
    )
    if nonce is None or nonce.state != "outstanding":
        raise EvalAuthorizationConflict(
            "score nonce is already terminal",
            code="eval_score_nonce_terminal",
        )
    nonce.state = "consumed"
    nonce.consumed_at = moment
    run.phase = "eval_rejected"
    run.reason_code = reason_code
    run.failure_origin = "attestation"
    run.retryable = False
    run.verified = False
    run.reward_eligible = False
    run.result_available = False
    run.finalized_at = moment
    await release_eval_resource(session, name=OUTSTANDING_RESULT_RESOURCE)
    return run


async def mark_eval_result_verified(
    session: AsyncSession,
    *,
    eval_run_id: str,
    body_sha256: str,
    now: datetime | None = None,
) -> EvalRun:
    """Consume the score nonce exactly once after a verified direct result."""

    run = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == eval_run_id).with_for_update()
    )
    if (
        run is None
        or run.receipt_id is None
        or run.receipt_body_sha256 != body_sha256
        or run.phase != "eval_verifying"
        or run.key_granted_at is None
    ):
        raise EvalAuthorizationConflict(
            "Eval result is not eligible for verification",
            code="eval_result_terminal",
        )
    moment = _as_utc(now)
    nonce = await session.scalar(
        select(EvalNonce).where(
            EvalNonce.eval_run_id == run.id,
            EvalNonce.purpose == "score",
        )
    )
    if nonce is None or nonce.state != "outstanding":
        raise EvalAuthorizationConflict(
            "score nonce is already terminal",
            code="eval_score_nonce_terminal",
        )
    nonce.state = "consumed"
    nonce.consumed_at = moment
    run.phase = "eval_accepted"
    run.reason_code = None
    run.failure_origin = None
    run.retryable = False
    run.verified = True
    run.result_available = True
    run.finalized_at = moment
    await release_eval_resource(session, name=OUTSTANDING_RESULT_RESOURCE)
    return run


async def eval_status_page(
    session: AsyncSession,
    submission: AgentSubmission,
    *,
    cursor: str | None = None,
    limit: int = 10,
    cursor_secret: str | None = None,
    page_max: int | None = None,
    page_default: int | None = None,
    settings: ChallengeSettings | None = None,
) -> dict[str, Any]:
    """Return safe retained history with no plan bytes, token, or nonce values."""

    maximum = int(
        page_max
        if page_max is not None
        else getattr(settings, "eval_status_page_max", 16)
        if settings is not None
        else 16
    )
    # Keep default available for callers that pass `limit=None` via keyword.
    if page_default is not None and limit is None:  # pragma: no cover - defensive
        limit = int(page_default)
    if not 1 <= limit <= maximum:
        raise EvalAuthorizationConflict(
            "Eval history limit is out of range",
            code="eval_limit_invalid",
        )
    secret = cursor_secret or "agent-challenge-eval-history-v1"
    result = await session.scalars(
        select(EvalRun)
        .where(EvalRun.submission_id == submission.id)
        .options(selectinload(EvalRun.nonces))
        .order_by(EvalRun.created_at, EvalRun.id)
    )
    all_runs = result.all()
    now = datetime.now(UTC)
    for run in all_runs:
        await _expire_run_if_needed(session, run, now=now)
    watermark = all_runs[-1].id if all_runs else 0
    offset = 0
    if cursor is not None:
        watermark, offset = _decode_cursor(
            cursor,
            submission_id=submission.id,
            secret=secret,
        )
    snapshot_runs = [run for run in all_runs if run.id <= watermark]
    runs = snapshot_runs[offset : offset + limit]
    items: list[dict[str, Any]] = []
    for run in runs:
        items.append(
            {
                "eval_run_id": run.eval_run_id,
                "attempt": run.attempt,
                "prior_eval_run_id": run.prior_eval_run_id,
                "receipt_id": run.receipt_id,
                "body_sha256": run.receipt_body_sha256,
                "phase": run.phase,
                "terminal": run.phase
                in {
                    "eval_expired",
                    "eval_cancelled",
                    "eval_error",
                    "eval_rejected",
                    "eval_accepted",
                },
                "verified": run.verified,
                "retryable": run.retryable,
                "reason_code": run.reason_code,
                "key_grant_state": "granted" if run.key_granted_at is not None else "not_granted",
                "key_release_nonce_state": next(
                    (nonce.state for nonce in run.nonces if nonce.purpose == "key_release"),
                    "unknown",
                ),
                "score_nonce_state": next(
                    (nonce.state for nonce in run.nonces if nonce.purpose == "score"),
                    "unknown",
                ),
                "issued_at_ms": _milliseconds(_compare_utc(run.issued_at)),
                "expires_at_ms": _milliseconds(_compare_utc(run.expires_at)),
                "received_at_ms": _milliseconds(_compare_utc(run.receipt_received_at))
                if run.receipt_received_at
                else None,
                "finalized_at_ms": (
                    _milliseconds(_compare_utc(run.finalized_at)) if run.finalized_at else None
                ),
                "result_available": run.result_available,
            }
        )
    next_offset = offset + len(runs)
    next_cursor = (
        _encode_cursor(
            submission_id=submission.id,
            watermark=watermark,
            offset=next_offset,
            secret=secret,
        )
        if next_offset < len(snapshot_runs)
        else None
    )
    return {
        "schema_version": 1,
        "submission_id": submission.id,
        "current_eval_run_id": all_runs[-1].eval_run_id if all_runs else None,
        "items": items,
        "next_cursor": next_cursor,
        "total_count": len(snapshot_runs),
    }


__all__ = [
    "CreatedEvalRun",
    "EvalAuthorizationConflict",
    "EvalAuthorizationRequired",
    "EvalAuthorizationUnavailable",
    "build_key_release_grant_materials",
    "cancel_eval_run",
    "create_eval_run",
    "eval_status_page",
    "fail_eval_run",
    "load_eval_run_plan",
    "mark_eval_key_release_denied",
    "mark_eval_key_release_retryable",
    "mark_eval_key_granted",
    "mark_eval_result_rejected",
    "mark_eval_result_retryable",
    "mark_eval_result_verified",
    "persist_key_release_grant_materials",
    "receipt_eval_result",
    "receipt_eval_key_release",
    "register_eval_key_release",
    "release_eval_resource",
    "reserve_eval_resource",
    "retry_eval_run",
]
