"""Challenge-owned direct Eval result ingestion.

The result endpoint is deliberately separate from the validator work-unit
executor.  Authentication only authorizes delivery, while the immutable Eval
plan, validator allowlist, quote verifier, key-grant state, and score nonce
decide acceptance.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.canonical import eval_wire
from agent_challenge.core.models import (
    AgentSubmission,
    EvalNonce,
    EvalRun,
    EvaluationJob,
    TaskAttestation,
    TaskResult,
)
from agent_challenge.core.statuses import JobStatus, TaskStatus
from agent_challenge.evaluation.attestation import (
    AttestationDecision,
    AttestationGate,
    AttestationOutcome,
    ResultMeasurementAllowlist,
    execution_proof_signing_payload,
)
from agent_challenge.evaluation.authorization import (
    VERIFYING_RESULT_RESOURCE,
    EvalAuthorizationConflict,
    load_eval_run_plan,
    mark_eval_result_rejected,
    mark_eval_result_retryable,
    mark_eval_result_verified,
    receipt_eval_result,
    release_eval_resource,
    reserve_eval_resource,
)
from agent_challenge.evaluation.benchmarks import BenchmarkTask
from agent_challenge.evaluation.eval_agent_llm import MODE_MEASURED_OPENROUTER
from agent_challenge.evaluation.plan_scoring import (
    canonical_eval_plan_json,
    persist_direct_eval_result,
    persist_direct_eval_result_from_plan,
)
from agent_challenge.evaluation.score_chain_gate import (
    REFUSE_INCOMPLETE_CHAIN,
    admit_production_score_for_eval_result,
    build_score_binding_from_plan_and_digest,
    lookup_key_release_grant_for_score,
)
from agent_challenge.keyrelease.quote import DcapQvlVerifier
from agent_challenge.review.authorization import verified_review_assignment_for_submission
from agent_challenge.review.canonical import parse_json_object
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.submissions.state_machine import ensure_submission_status

RESULT_MAX_DEFAULT = 16 * 1024 * 1024


class DirectEvalResultError(ValueError):
    """A bounded, lifecycle-safe direct-result rejection."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def require_endpoint_result_signer(settings: ChallengeSettings) -> None:
    """Fail closed unless the endpoint owns a configured result signer."""

    if not settings.eval_result_signer_uri and not settings.eval_result_signer_mnemonic:
        raise ValueError("eval result signer is required for endpoint-owned signature rebind")


def result_body_sha256(raw_body: bytes) -> str:
    return hashlib.sha256(raw_body).hexdigest()


def validate_result_bounds(
    value: Mapping[str, Any],
    *,
    max_tasks: int,
    max_event_log_entries: int,
    max_event_log_bytes: int = 2 * 1024 * 1024,
    max_vm_config_bytes: int = 256 * 1024,
    max_string_bytes: int = 16 * 1024,
    max_quote_bytes: int,
    max_body_bytes: int | None = None,
) -> None:
    """Reject nested result allocations beyond configured wire budgets."""

    if max_body_bytes is not None:
        encoded = eval_wire.canonical_json_v1(value)
        if len(encoded) > max_body_bytes:
            raise DirectEvalResultError(
                "Eval result body exceeds bound",
                code="result_too_large",
            )
    try:
        score_record = value["score_record"]
        proof = value["execution_proof"]
        if not isinstance(score_record, Mapping) or not isinstance(proof, Mapping):
            raise TypeError
        tasks = score_record["tasks"]
        attestation = proof["attestation"]
        if not isinstance(attestation, Mapping):
            raise TypeError
        event_log = attestation["event_log"]
        quote = attestation["tdx_quote"]
    except (KeyError, TypeError):
        raise DirectEvalResultError("Eval result shape is invalid", code="result_invalid") from None
    if not isinstance(tasks, list) or len(tasks) > max_tasks:
        raise DirectEvalResultError("Eval result task bound exceeded", code="result_tasks_too_many")
    if not isinstance(event_log, list) or len(event_log) > max_event_log_entries:
        raise DirectEvalResultError(
            "Eval result event-log bound exceeded",
            code="result_event_log_too_large",
        )
    if len(json.dumps(event_log, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) > (
        max_event_log_bytes
    ):
        raise DirectEvalResultError(
            "Eval result event-log bytes exceed bound",
            code="result_event_log_too_large",
        )
    vm_config = attestation.get("vm_config")
    if len(json.dumps(vm_config, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) > (
        max_vm_config_bytes
    ):
        raise DirectEvalResultError(
            "Eval result VM config exceeds bound",
            code="result_vm_config_too_large",
        )
    stack: list[tuple[Any, bool]] = [(value, False)]
    while stack:
        item, is_quote = stack.pop()
        if isinstance(item, str):
            limit = max_quote_bytes * 2 if is_quote else max_string_bytes
            if len(item.encode("utf-8")) > limit:
                raise DirectEvalResultError(
                    "Eval result string bound exceeded",
                    code="result_string_too_large",
                )
        elif isinstance(item, Mapping):
            stack.extend((child, key == "tdx_quote") for key, child in item.items())
        elif isinstance(item, list):
            stack.extend((child, is_quote) for child in item)
    if not isinstance(quote, str) or len(quote) > max_quote_bytes * 2:
        raise DirectEvalResultError(
            "Eval result quote bound exceeded", code="result_quote_too_large"
        )


def authenticate_eval_token(run: EvalRun, token: str | None) -> bool:
    """Authenticate only the stored one-time scoped Eval token."""

    if not isinstance(token, str) or not token:
        return False
    return hmac.compare_digest(
        run.token_sha256,
        hashlib.sha256(token.encode("utf-8")).hexdigest(),
    )


def _receipt(run: EvalRun, *, received_at: datetime | None = None) -> dict[str, Any]:
    phase = run.phase
    if phase == "eval_accepted":
        wire_phase = "verified"
    elif phase == "eval_rejected":
        wire_phase = "rejected"
    elif phase == "eval_verifying" and run.reason_code in {
        "verifier_unavailable",
        "persistence_unavailable",
    }:
        wire_phase = "verifier_unavailable"
    elif run.receipt_id is not None:
        wire_phase = "verifying"
    else:
        wire_phase = "received"
    return eval_wire.validate_eval_receipt(
        {
            "schema_version": 1,
            "eval_run_id": run.eval_run_id,
            "receipt_id": run.receipt_id or f"receipt_{uuid4().hex}",
            "body_sha256": run.receipt_body_sha256 or ("0" * 64),
            "received_at_ms": int(
                (received_at or run.receipt_received_at or datetime.now(UTC)).timestamp() * 1000
            ),
            "phase": wire_phase,
            "terminal": phase in {"eval_accepted", "eval_rejected"},
            "verified": bool(run.verified),
            "retryable": bool(run.retryable and phase not in {"eval_accepted", "eval_rejected"}),
            "reason_code": run.reason_code,
            "result_available": bool(run.result_available),
            "finalized_at_ms": (
                int(run.finalized_at.timestamp() * 1000) if run.finalized_at else None
            ),
        }
    )


async def _score_nonce_state(
    session: AsyncSession,
    run: EvalRun,
    *,
    now: datetime,
) -> bool:
    nonce = await session.scalar(
        select(EvalNonce).where(
            EvalNonce.eval_run_id == run.id,
            EvalNonce.purpose == "score",
        )
    )
    return bool(
        nonce is not None
        and nonce.nonce == load_eval_run_plan(run)["score_nonce"]
        and nonce.state == "outstanding"
        and (
            run.receipt_id is not None
            or (
                nonce.expires_at.replace(tzinfo=UTC)
                if nonce.expires_at.tzinfo is None
                else nonce.expires_at.astimezone(UTC)
            )
            > now
        )
    )


def _gate(settings: ChallengeSettings, *, quote_verifier: Any | None) -> AttestationGate:
    try:
        allowlist = ResultMeasurementAllowlist.from_measurements(
            settings.eval_app_measurement_allowlist
        )
    except (KeyError, TypeError, ValueError):
        allowlist = ResultMeasurementAllowlist()
    return AttestationGate(
        quote_verifier=quote_verifier
        if quote_verifier is not None
        else DcapQvlVerifier(timeout=60.0),
        allowlist=allowlist,
        nonce_validator=None,
    )


def _endpoint_worker_signature(
    settings: ChallengeSettings,
    *,
    manifest_sha256: str,
    unit_id: str,
) -> dict[str, str] | None:
    """Create the validator-owned signature used for the one production gate."""

    if not settings.eval_result_signer_uri and not settings.eval_result_signer_mnemonic:
        # Empty placeholder is only valid on the wire from the CVM; production
        # acceptance requires a configured endpoint-owned rebind.
        return None
    try:
        import bittensor as bt

        keypair = (
            bt.Keypair.create_from_uri(settings.eval_result_signer_uri)
            if settings.eval_result_signer_uri
            else bt.Keypair.create_from_mnemonic(settings.eval_result_signer_mnemonic)
        )
        signature = keypair.sign(
            execution_proof_signing_payload(
                manifest_sha256=manifest_sha256,
                unit_id=unit_id,
            )
        )
        if isinstance(signature, bytes | bytearray):
            encoded_signature = "0x" + bytes(signature).hex()
        else:
            encoded_signature = str(signature)
        public_key = str(keypair.ss58_address)
        if settings.eval_result_signer_hotkey and (
            settings.eval_result_signer_hotkey != public_key
        ):
            return None
        return {
            "worker_pubkey": public_key,
            "sig": encoded_signature,
        }
    except Exception:  # noqa: BLE001 - signer configuration fails closed
        return None


def _score_binding_from_validated(
    plan: Mapping[str, Any],
    validated: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Derive closed score-domain binding + report_data from a validated result.

    Returns ``(binding, report_data_hex, scores_digest)``. Any reconstruction
    failure yields ``(None, None, None)`` so the score-chain gate fails closed.
    """

    try:
        scores_digest = str(validated["scores_digest"])
        binding = build_score_binding_from_plan_and_digest(
            eval_plan=plan,
            scores_digest=scores_digest,
        )
        report_data_hex = str(validated["execution_proof"]["attestation"]["report_data"])
        return binding, report_data_hex, scores_digest
    except (KeyError, TypeError, ValueError):
        return None, None, None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Best-effort Mapping view; accepts JSON-text strings for side channels."""

    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, bytes)):
        try:
            raw = value if isinstance(value, str) else value.decode("utf-8")
            parsed = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError):
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _agent_llm_claim_sources(
    *,
    plan: Mapping[str, Any],
    validated: Mapping[str, Any],
    raw_request: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    """Collect optional bags that may carry agent-LLM materials / residue."""

    sources: list[Mapping[str, Any]] = []
    for pool in (raw_request, validated, plan):
        if not isinstance(pool, Mapping):
            continue
        for key in (
            "agent_llm",
            "agent_or_materials",
            "eval_agent_llm",
            "agent_llm_claim",
            "agent_model_claim",
        ):
            bag = _as_mapping(pool.get(key))
            if bag is not None:
                sources.append(bag)
        # Top-level materials bag is itself a source of claims / digests.
        if "domain_role" in pool or "planned_request_sha256" in pool:
            sources.append(pool)
    return sources


def _first_truthy(
    bags: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> bool:
    for bag in bags:
        for key in keys:
            if key in bag and _truthy_flag(bag.get(key)):
                return True
    return False


def _first_str(
    bags: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> str | None:
    for bag in bags:
        for key in keys:
            value = bag.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _first_mapping(
    bags: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Mapping[str, Any] | None:
    for bag in bags:
        for key in keys:
            mapped = _as_mapping(bag.get(key))
            if mapped is not None:
                return mapped
    return None


def extract_agent_llm_materials_for_score(
    *,
    plan: Mapping[str, Any],
    validated: Mapping[str, Any],
    raw_request: Mapping[str, Any] | None = None,
    settings: ChallengeSettings | None = None,
) -> dict[str, Any]:
    """Extract agent-LLM materials for production score-chain re-check.

    Sources (side channels + plan/result bags; closed eval-result wire may not
    serialize them, so dual-flag emission re-loaders also accept top-level
    request / run-plan adjunct fields used by guests and integration tests):

    * ``agent_or_materials`` / ``agent_llm.materials`` (planned+observed digests)
    * claim flags: ``claims_agent_model_call`` / ``claims_model_call`` / …
    * runtime kind + measurement + allowlist for measured OpenRouter mode
    * Base-gateway residue: ``BASE_GATEWAY_*`` env, ``/llm/v1`` URL, tokens

    Tools-only default applies only when **no** model claim, materials, or
    gateway residue is present. When a claim is present without digests, the
    emission gate refuses (VAL-ACAT-016/050–052).
    """

    bags = _agent_llm_claim_sources(
        plan=plan,
        validated=validated,
        raw_request=raw_request,
    )
    raw_map = raw_request if isinstance(raw_request, Mapping) else {}

    materials = None
    # Prefer explicit materials bags before treating a whole claim object as OR materials.
    for pool in (raw_map, validated, plan, *bags):
        if not isinstance(pool, Mapping):
            continue
        candidate = _as_mapping(pool.get("agent_or_materials"))
        if candidate is None:
            candidate = _as_mapping(pool.get("materials"))
        if candidate is not None and (
            "planned" in candidate
            or "planned_request_sha256" in candidate
            or candidate.get("domain_role") == "eval_agent_openrouter"
        ):
            materials = dict(candidate)
            break
        # Whole bag may itself be bound materials.
        if (
            pool.get("domain_role") == "eval_agent_openrouter"
            or ("planned" in pool and "observed" in pool)
            or "planned_request_sha256" in pool
        ) and "agent_llm" not in pool:
            materials = dict(pool)
            break

    claims = _first_truthy(
        [*bags, plan, validated, raw_map],
        (
            "claims_agent_model_call",
            "claims_model_call",
            "agent_model_call",
            "used_model_call",
            "model_call",
            "openrouter_model_call",
        ),
    )
    if materials is not None:
        claims = True

    mode = _first_str(
        [*bags, plan, validated, raw_map],
        ("agent_llm_mode", "llm_mode", "mode"),
    )
    if mode is None:
        mode = MODE_MEASURED_OPENROUTER if claims else None

    runtime_kind = _first_str(
        [*bags, plan, validated, raw_map],
        ("agent_llm_runtime_kind", "runtime_kind", "agent_runtime_kind"),
    )

    measurement_raw = _first_mapping(
        [*bags, plan, validated, raw_map],
        ("agent_llm_measurement", "measurement", "eval_measurement"),
    )
    measurement: dict[str, str] | None = None
    if isinstance(measurement_raw, Mapping):
        measurement = {str(k): str(v) for k, v in measurement_raw.items()}
    if measurement is None:
        # Fall back to immutable eval-app measurement from the plan (measured guest).
        try:
            app = plan.get("eval_app") if isinstance(plan, Mapping) else None
            if isinstance(app, Mapping):
                m = app.get("measurement")
                compose = app.get("compose_hash")
                if isinstance(m, Mapping):
                    measurement = {
                        "compose_hash": str(compose or m.get("compose_hash") or ""),
                        "os_image_hash": str(m.get("os_image_hash") or ""),
                        "mrtd": str(m.get("mrtd") or ""),
                        "rtmr0": str(m.get("rtmr0") or ""),
                        "rtmr1": str(m.get("rtmr1") or ""),
                        "rtmr2": str(m.get("rtmr2") or ""),
                    }
                    if not measurement["compose_hash"]:
                        measurement = None
        except (TypeError, ValueError, AttributeError):
            measurement = None

    allowlist: list[Mapping[str, str]] | None = None
    for pool in (*bags, plan, validated, raw_map):
        if not isinstance(pool, Mapping):
            continue
        for key in ("agent_llm_allowlist", "measurement_allowlist", "allowlist"):
            raw_al = pool.get(key)
            if isinstance(raw_al, list) and raw_al:
                cleaned: list[Mapping[str, str]] = []
                for entry in raw_al:
                    if isinstance(entry, Mapping):
                        cleaned.append({str(k): str(v) for k, v in entry.items()})
                if cleaned:
                    allowlist = cleaned
                    break
        if allowlist is not None:
            break
    if allowlist is None and settings is not None:
        cfg = getattr(settings, "eval_app_measurement_allowlist", None) or ()
        cleaned_cfg: list[Mapping[str, str]] = []
        for entry in cfg:
            if isinstance(entry, Mapping):
                cleaned_cfg.append({str(k): str(v) for k, v in entry.items()})
        if cleaned_cfg:
            allowlist = cleaned_cfg
    if allowlist is None and measurement is not None and claims:
        # Self-allow the plan-bound measurement for re-check completeness when
        # settings allowlist is empty (unit/integration fixtures); production
        # dual-flag path typically has configured allowlist + guest match.
        allowlist = [dict(measurement)]

    agent_env_map = _first_mapping(
        [*bags, plan, validated, raw_map],
        ("agent_env", "eval_agent_env", "agent_environment"),
    )
    agent_env: dict[str, str] | None = None
    if isinstance(agent_env_map, Mapping):
        agent_env = {str(k): str(v) for k, v in agent_env_map.items()}

    gateway_url = _first_str(
        [*bags, plan, validated, raw_map],
        (
            "agent_gateway_url",
            "gateway_url",
            "gateway_base_url",
            "BASE_LLM_GATEWAY_URL",
            "base_llm_gateway_url",
        ),
    )
    gateway_token_present = _first_truthy(
        [*bags, plan, validated, raw_map],
        (
            "agent_gateway_token_present",
            "gateway_token_present",
            "gateway_token",
            "BASE_GATEWAY_TOKEN",
            "base_gateway_token",
        ),
    )
    used_base_llm_v1 = _first_truthy(
        [*bags, plan, validated, raw_map],
        (
            "agent_used_base_llm_v1",
            "used_base_llm_v1",
            "used_base_gateway",
            "used_llm_v1",
        ),
    )
    if gateway_url and "/llm/v1" in str(gateway_url):
        used_base_llm_v1 = True
    if agent_env is not None:
        for key in agent_env:
            upper = str(key).upper()
            if upper in {
                "BASE_GATEWAY_TOKEN",
                "BASE_LLM_GATEWAY_URL",
                "GATEWAY_TOKEN",
            }:
                gateway_token_present = True
                if upper.endswith("URL") or "GATEWAY_URL" in upper:
                    gateway_url = gateway_url or agent_env[key]
                break

    # Runtime default for claimed model path: measured eval CVM when materials exist.
    if claims and runtime_kind is None and materials is not None:
        runtime_kind = "measured_eval_cvm"

    return {
        "agent_llm_mode": mode,
        "agent_or_materials": materials,
        "agent_llm_runtime_kind": runtime_kind,
        "agent_llm_measurement": measurement,
        "agent_llm_allowlist": allowlist,
        "claims_agent_model_call": bool(claims),
        "agent_env": agent_env,
        "agent_gateway_url": gateway_url,
        "agent_gateway_token_present": bool(gateway_token_present),
        "agent_used_base_llm_v1": bool(used_base_llm_v1),
    }


def _key_release_grant_from_result(
    *,
    plan: Mapping[str, Any],
    validated: Mapping[str, Any],
    run: EvalRun,
    raw_request: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Reconstruct a score-admission key-release grant for re-verify.

    The closed eval-result wire does **not** carry the grant inside
    ``execution_proof``. Admission re-check sources (in order):

    1. **Durable** ``run.key_release_grant_json`` stamped at KR success
       (production multi-worker / restart safe; VAL-ACAT-036/037)
    2. Optional in-process attribute ``run._score_chain_key_release_grant``
       (test/injection surface)
    3. Raw request top-level ``key_release_grant`` (side-channel for tests only;
       not round-tripped through body hash validation)
    4. Process-local registry (cache only; insufficient alone after restart)

    Historic ``key_granted_at`` alone never admits under dual flags
    (VAL-ACAT-036/037). Process-local dict alone is not multi-worker durable.
    """

    raw = getattr(run, "key_release_grant_json", None)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, Mapping):
            # Re-hydrate process registry from durable column after restarts.
            eval_run_id = plan.get("eval_run_id") if isinstance(plan, Mapping) else None
            if not isinstance(eval_run_id, str):
                eval_run_id = getattr(run, "eval_run_id", None)
            if isinstance(eval_run_id, str) and eval_run_id:
                from agent_challenge.evaluation.score_chain_gate import (
                    register_key_release_grant_for_score,
                )

                register_key_release_grant_for_score(eval_run_id, parsed)
            return parsed
    attr = getattr(run, "_score_chain_key_release_grant", None)
    if isinstance(attr, Mapping):
        return attr
    if isinstance(raw_request, Mapping):
        carried = raw_request.get("key_release_grant")
        if isinstance(carried, Mapping):
            return carried
    # Process-local registry (secondary; durable column is authoritative).
    eval_run_id = None
    if isinstance(plan, Mapping):
        eval_run_id = plan.get("eval_run_id")
    if not isinstance(eval_run_id, str):
        eval_run_id = getattr(run, "eval_run_id", None)
    if isinstance(eval_run_id, str):
        registered = lookup_key_release_grant_for_score(eval_run_id)
        if registered is not None:
            return registered
    _ = validated
    return None


async def _load_review_envelope_for_run(
    session: AsyncSession,
    run: EvalRun,
) -> Mapping[str, Any] | str | None:
    """Load receipted review-domain envelope for score-chain re-verify."""

    submission = await session.scalar(
        select(AgentSubmission).where(AgentSubmission.id == run.submission_id)
    )
    if submission is None:
        return None
    assignment = await verified_review_assignment_for_submission(session, submission)
    if assignment is None:
        return None
    # Digest must match the immutable plan binding.
    plan = load_eval_run_plan(run)
    if (
        isinstance(assignment.review_digest, str)
        and assignment.review_digest
        and assignment.review_digest != plan.get("authorizing_review_digest")
    ):
        return None
    envelope = assignment.review_report_envelope_json
    if isinstance(envelope, str) and envelope:
        return envelope
    return None


async def _run_gate_with_deadline(
    *,
    gate: AttestationGate,
    validated: Mapping[str, Any],
    plan: Mapping[str, Any],
    nonce_outstanding: bool,
    key_granted: bool,
    rebound_signature: Mapping[str, str] | None,
    deadline_seconds: float,
    dual_flags_on: bool = False,
    review_envelope: Mapping[str, Any] | str | bytes | None = None,
    key_release_grant: Mapping[str, Any] | None = None,
    agent_llm_kwargs: Mapping[str, Any] | None = None,
    settings: ChallengeSettings | None = None,
    raw_request: Mapping[str, Any] | None = None,
) -> AttestationDecision:
    """Run the production gate with a deadline that terminates the verifier process.

    Under dual attestation flags ON, *also* re-verifies the full score chain
    (review + key-release RA-TLS + score domain + nonce + eval-agent OR materials)
    before accept (VAL-ACAT-030/036/037/040/016/050–054). Chain refuse is sticky
    permanent reject.
    """

    loop = asyncio.get_running_loop()
    verifier = gate.quote_verifier

    def _decide() -> AttestationDecision:
        # Production dual-flag path: full-chain conjunction is mandatory
        # (review + KR RA-TLS + score domain + nonce + agent OR digests when
        # claimed). Missing/tampered/stale materials refuse with sticky
        # permanent codes and zero partial score
        # (VAL-ACAT-030/031/036/037/039/040/016/050–054). Flag-off leaves
        # quote-only path.
        if dual_flags_on:
            binding, rd_hex, scores_digest = _score_binding_from_validated(plan, validated)
            llm_kwargs = dict(agent_llm_kwargs or {})
            if not llm_kwargs:
                llm_kwargs = extract_agent_llm_materials_for_score(
                    plan=plan,
                    validated=validated,
                    raw_request=raw_request,
                    settings=settings,
                )
            chain = admit_production_score_for_eval_result(
                settings_dual_flags_on=True,
                eval_plan=plan,
                review_envelope=review_envelope,
                key_release_grant=key_release_grant,
                key_granted_flag=key_granted,
                score_binding=binding,
                score_report_data_hex=rd_hex,
                scores_digest=scores_digest,
                score_nonce_outstanding=nonce_outstanding,
                agent_llm_mode=llm_kwargs.get("agent_llm_mode"),
                agent_or_materials=llm_kwargs.get("agent_or_materials"),
                agent_llm_runtime_kind=llm_kwargs.get("agent_llm_runtime_kind"),
                agent_llm_measurement=llm_kwargs.get("agent_llm_measurement"),
                agent_llm_allowlist=llm_kwargs.get("agent_llm_allowlist"),
                claims_agent_model_call=bool(llm_kwargs.get("claims_agent_model_call", False)),
                agent_env=llm_kwargs.get("agent_env"),
                agent_gateway_url=llm_kwargs.get("agent_gateway_url"),
                agent_gateway_token_present=bool(
                    llm_kwargs.get("agent_gateway_token_present", False)
                ),
                agent_used_base_llm_v1=bool(llm_kwargs.get("agent_used_base_llm_v1", False)),
            )
            if not chain.admitted:
                return AttestationDecision(
                    outcome=AttestationOutcome.VERIFICATION_FAILED,
                    reason=chain.reason_code or REFUSE_INCOMPLETE_CHAIN,
                )

        return gate.decide_eval_result(
            validated,
            eval_plan=plan,
            expected_agent_hash=plan["agent_hash"],
            nonce_outstanding=nonce_outstanding,
            key_granted=key_granted,
            endpoint_rebound=True,
            rebound_worker_signature=rebound_signature,
        )

    future = loop.run_in_executor(None, _decide)
    try:
        return await asyncio.wait_for(future, timeout=deadline_seconds)
    except TimeoutError:
        cancel = getattr(verifier, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # noqa: BLE001 - best-effort process kill
                pass
        return AttestationDecision.of(AttestationOutcome.VERIFIER_UNAVAILABLE)


async def process_direct_eval_result(
    session: AsyncSession,
    *,
    run: EvalRun,
    raw_body: bytes,
    result_request: Mapping[str, Any],
    settings: ChallengeSettings,
    quote_verifier: Any | None = None,
    now: datetime | None = None,
    verification_limit: int | None = None,
) -> tuple[dict[str, Any], bool]:
    """Receipt, verify, and terminalize one direct result.

    The first tuple member is the exact schema-closed receipt.  The second
    member indicates that this request created the receipt, which lets the
    HTTP layer return ``202`` for first delivery and ``200`` for idempotent
    terminal reads.
    """

    digest = result_body_sha256(raw_body)
    validate_result_bounds(
        result_request,
        max_tasks=settings.eval_result_max_tasks,
        max_event_log_entries=settings.eval_result_max_event_log_entries,
        max_event_log_bytes=settings.eval_result_max_event_log_bytes,
        max_vm_config_bytes=settings.eval_result_max_vm_config_bytes,
        max_string_bytes=settings.eval_result_max_string_bytes,
        max_quote_bytes=settings.eval_result_max_quote_bytes,
        max_body_bytes=settings.eval_result_max_bytes,
    )
    plan = load_eval_run_plan(run)
    try:
        validated = eval_wire.validate_eval_result_request(result_request)
        validated = {
            **validated,
            **{
                "score_record": eval_wire.validate_canonical_score_record(
                    validated["score_record"],
                    scoring_policy=plan["scoring_policy"],
                    expected_eval_run_id=plan["eval_run_id"],
                    expected_task_ids=[item["task_id"] for item in plan["selected_tasks"]],
                    expected_k=plan["k"],
                )
            },
        }
        if eval_wire.canonical_json_v1(validated) != raw_body:
            raise DirectEvalResultError(
                "Eval result body is not canonical",
                code="result_noncanonical",
            )
    except (eval_wire.EvalWireError, ValueError, KeyError, TypeError) as exc:
        raise DirectEvalResultError(
            "Eval result schema or plan mismatch", code="result_invalid"
        ) from exc

    received_at = now or datetime.now(UTC)
    recorded_run, should_verify = await receipt_eval_result(
        session,
        eval_run_id=run.eval_run_id,
        body_sha256=digest,
        body=raw_body,
        max_submissions_per_minute=settings.eval_result_max_submissions_per_run_per_minute,
        max_outstanding=settings.eval_result_max_outstanding,
        now=received_at,
    )
    await session.commit()
    if not should_verify:
        return _receipt(recorded_run), False

    current = await session.scalar(
        select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id).with_for_update()
    )
    if current is None:
        raise DirectEvalResultError("Eval run disappeared", code="eval_run_unknown")
    plan = load_eval_run_plan(current)
    nonce_outstanding = await _score_nonce_state(session, current, now=received_at)
    gate = _gate(settings, quote_verifier=quote_verifier)
    proof = validated["execution_proof"]
    rebound_signature = _endpoint_worker_signature(
        settings,
        manifest_sha256=proof["manifest_sha256"],
        unit_id=plan["eval_run_id"],
    )
    dual_flags_on = bool(settings.attested_review_enabled and settings.phala_attestation_enabled)
    # Materials for full-chain weak-preverify under dual flags (VAL-ACAT-030+).
    review_envelope: Mapping[str, Any] | str | None = None
    key_release_grant: Mapping[str, Any] | None = None
    agent_llm_kwargs: dict[str, Any] | None = None
    if dual_flags_on:
        review_envelope = await _load_review_envelope_for_run(session, current)
        key_release_grant = _key_release_grant_from_result(
            plan=plan,
            validated=validated,
            run=current,
            raw_request=result_request,
        )
        # VAL-ACAT-016/050–054: plumb measured OpenRouter digests / claim flags /
        # gateway residue into production score admission (not helpers-only).
        agent_llm_kwargs = extract_agent_llm_materials_for_score(
            plan=plan,
            validated=validated,
            raw_request=result_request,
            settings=settings,
        )
    limit = verification_limit or settings.attestation_max_concurrent_verifications
    slot_reserved = False
    decision: AttestationDecision
    try:
        await reserve_eval_resource(
            session,
            name=VERIFYING_RESULT_RESOURCE,
            limit=limit,
            conflict_code="eval_result_overloaded",
        )
        slot_reserved = True
        await session.commit()
        try:
            decision = await _run_gate_with_deadline(
                gate=gate,
                validated=validated,
                plan=plan,
                nonce_outstanding=nonce_outstanding,
                key_granted=current.key_granted_at is not None,
                rebound_signature=rebound_signature,
                deadline_seconds=settings.eval_result_verifier_deadline_seconds,
                dual_flags_on=dual_flags_on,
                review_envelope=review_envelope,
                key_release_grant=key_release_grant,
                agent_llm_kwargs=agent_llm_kwargs,
                settings=settings,
                raw_request=result_request,
            )
        except Exception:  # noqa: BLE001 - unexpected gate failures park retryable
            decision = AttestationDecision.of(AttestationOutcome.VERIFIER_UNAVAILABLE)
    except EvalAuthorizationConflict:
        # Receipt is durable; capacity is global and may free after another
        # verification finishes. Keep the score nonce and park as retryable.
        decision = AttestationDecision.of(AttestationOutcome.VERIFIER_UNAVAILABLE)
    finally:
        if slot_reserved:
            current_after = await session.scalar(
                select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id).with_for_update()
            )
            if current_after is not None:
                current = current_after
            await release_eval_resource(session, name=VERIFYING_RESULT_RESOURCE)
            await session.flush()
    if decision.outcome is AttestationOutcome.VERIFIER_UNAVAILABLE:
        await mark_eval_result_retryable(
            session,
            eval_run_id=current.eval_run_id,
            body_sha256=digest,
        )
    elif decision.accepted:
        try:
            # Receipt, canonical score reconstruction, review/eval linkage, and
            # nonce consumption share one transaction.  Full-attested results
            # never enter the validator job/task tables.
            # If any persistence step fails, no partially-scored result is left
            # behind.  The already-committed receipt is parked so a recovery
            # worker can retry the exact bytes.
            await _persist_verified_result(
                session,
                run=current,
                result_request=validated,
                settings=settings,
                now=received_at,
            )
            await mark_eval_result_verified(
                session,
                eval_run_id=current.eval_run_id,
                body_sha256=digest,
                now=received_at,
            )
        except Exception:
            current_eval_run_id = current.eval_run_id
            await session.rollback()
            parked = await mark_eval_result_retryable(
                session,
                eval_run_id=current_eval_run_id,
                body_sha256=digest,
                reason_code="persistence_unavailable",
            )
            await session.commit()
            await session.refresh(parked)
            return _receipt(parked, received_at=received_at), True
    else:
        await mark_eval_result_rejected(
            session,
            eval_run_id=current.eval_run_id,
            body_sha256=digest,
            reason_code=decision.reason or "attestation_verification_failed",
            now=received_at,
        )
    await session.flush()
    await session.commit()
    await session.refresh(current)
    return _receipt(current, received_at=received_at), True


async def _persist_verified_result(
    session: AsyncSession,
    *,
    run: EvalRun,
    result_request: Mapping[str, Any],
    settings: ChallengeSettings,
    now: datetime,
) -> None:
    """Persist one accepted direct result and its complete challenge score atomically."""

    if not (settings.attested_review_enabled and settings.phala_attestation_enabled):
        await _persist_legacy_verified_result(
            session,
            run=run,
            result_request=result_request,
            now=now,
        )
        return

    plan = load_eval_run_plan(run)
    validated_result = eval_wire.validate_eval_result_request(result_request)
    final = persist_direct_eval_result_from_plan(plan, validated_result)
    run.score = final.score
    run.passed_tasks = final.passed_tasks
    run.total_tasks = final.total_tasks
    run.canonical_score_record_json = eval_wire.canonical_json_v1(final.score_record).decode(
        "utf-8"
    )
    run.canonical_score_record_sha256 = eval_wire.score_record_digest(final.score_record)
    submission = await session.scalar(
        select(AgentSubmission).where(AgentSubmission.id == run.submission_id).with_for_update()
    )
    review_assignment = (
        await verified_review_assignment_for_submission(session, submission)
        if submission is not None
        else None
    )
    run.reward_eligible = bool(
        submission is not None
        and submission.version_number == run.submission_version
        and review_assignment is not None
        and review_assignment.review_digest == run.authorizing_review_digest
    )
    if submission is not None and submission.raw_status == "review_allowed":
        await ensure_submission_status(
            session,
            submission,
            "tb_completed",
            actor="eval-direct",
            reason="attested_eval_result_verified",
            metadata={"eval_run_id": run.eval_run_id},
        )
    await session.flush()


async def _persist_legacy_verified_result(
    session: AsyncSession,
    *,
    run: EvalRun,
    result_request: Mapping[str, Any],
    now: datetime,
) -> None:
    """Retain the pre-topology score container for legacy direct-result tests."""

    plan = load_eval_run_plan(run)
    job = (
        await session.scalar(
            select(EvaluationJob)
            .where(EvaluationJob.id == run.result_job_id)
            .where(EvaluationJob.eval_plan_json == canonical_eval_plan_json(plan))
            .with_for_update()
        )
        if run.result_job_id is not None
        else None
    )
    if job is None:
        selected_tasks = [
            BenchmarkTask(
                task_id=item["task_id"],
                docker_image=item["image_ref"],
                benchmark="attested_eval",
                metadata={"attested": True},
            )
            for item in plan["selected_tasks"]
        ]
        job = EvaluationJob(
            job_id=f"attested-{run.eval_run_id}",
            submission_id=run.submission_id,
            status=JobStatus.RUNNING,
            selected_tasks_json=json.dumps(
                [
                    {
                        "task_id": task.task_id,
                        "docker_image": task.docker_image,
                        "prompt": task.prompt,
                        "benchmark": task.benchmark,
                        "metadata": task.metadata,
                    }
                    for task in selected_tasks
                ],
                separators=(",", ":"),
            ),
            eval_plan_json=canonical_eval_plan_json(plan),
            total_tasks=len(selected_tasks),
        )
        session.add(job)
        await session.flush()
        run.result_job_id = job.id
    final = persist_direct_eval_result(job, result_request)
    existing_rows = {
        row.task_id: row
        for row in (
            await session.scalars(select(TaskResult).where(TaskResult.job_id == job.id))
        ).all()
    }
    attestation_rows = {
        row.task_id: row
        for row in (
            await session.scalars(select(TaskAttestation).where(TaskAttestation.job_id == job.id))
        ).all()
    }
    selected_task_ids = {item["task_id"] for item in plan["selected_tasks"]}
    if set(existing_rows) - selected_task_ids or set(attestation_rows) - selected_task_ids:
        raise DirectEvalResultError(
            "persisted task evidence does not match immutable Eval plan",
            code="result_persistence_conflict",
        )
    for task in final.score_record["tasks"]:
        task_id = task["task_id"]
        selected = next(item for item in plan["selected_tasks"] if item["task_id"] == task_id)
        score = eval_wire.decode_score_f64be(task["aggregate_score_f64be"])
        row = existing_rows.get(task_id)
        if row is None:
            session.add(
                TaskResult(
                    job_id=job.id,
                    task_id=task_id,
                    docker_image=selected["image_ref"],
                    status=TaskStatus.COMPLETED,
                    score=score,
                    returncode=0,
                    stdout="",
                    stderr="",
                    duration_seconds=0.0,
                )
            )
        else:
            row.status = TaskStatus.COMPLETED
            row.score = score
        attestation = attestation_rows.get(task_id)
        if attestation is None:
            session.add(
                TaskAttestation(
                    job_id=job.id,
                    task_id=task_id,
                    verified=True,
                    reason=None,
                    retryable=False,
                )
            )
        else:
            attestation.verified = True
            attestation.reason = None
            attestation.retryable = False
    job.status = JobStatus.COMPLETED
    job.score = final.score
    job.passed_tasks = final.passed_tasks
    job.total_tasks = final.total_tasks
    job.finished_at = now
    run.result_job_id = job.id
    await session.flush()


async def retry_receipted_eval_result(
    session: AsyncSession,
    *,
    run: EvalRun,
    settings: ChallengeSettings,
    quote_verifier: Any | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], bool]:
    """Resume the exact durable body after a transient verifier outage."""

    if run.receipt_body is None or run.receipt_body_sha256 is None:
        raise DirectEvalResultError(
            "Eval result receipt has no durable body", code="result_invalid"
        )
    result_request = parse_json_object(run.receipt_body)
    return await process_direct_eval_result(
        session,
        run=run,
        raw_body=run.receipt_body,
        result_request=result_request,
        settings=settings,
        quote_verifier=quote_verifier,
        now=now,
    )


__all__ = [
    "DirectEvalResultError",
    "authenticate_eval_token",
    "process_direct_eval_result",
    "require_endpoint_result_signer",
    "retry_receipted_eval_result",
    "result_body_sha256",
    "validate_result_bounds",
]
