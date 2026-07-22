"""Eval-time agent LLM: OpenRouter only inside measured eval CVM.

Product freeze (VAL-ACAT-016 / 050–054, library/ac-attestation.md):

- If scored agents call models, those calls **must** be OpenRouter inside a
  measurement-allowlisted **eval** CVM with planned + observed digests bound
  into eval/score attestation materials.
- Base master ``/llm/v1`` and ``BASE_GATEWAY_TOKEN`` / ``BASE_LLM_GATEWAY_URL``
  are **never** a production success path (``GatewayConfigError`` is residual
  only; never the scored route).
- Tools-only mode is legal: any outbound model call / claim then fail-closes.
- Flag-off residual (either dual production flag OFF) cannot emit production
  scores even when LLM digests and metrics look perfect.

Does **not** restore Base LLM gateway. Does **not** invent REAL-PROVIDER PASS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_challenge.review.or_outcome_bind import (
    OPENROUTER_ORIGIN,
    OPENROUTER_PATH,
    OPENROUTER_TLS_HOSTNAME,
    REFUSE_MISSING_OBSERVED,
    REFUSE_MISSING_PLANNED,
    REFUSE_PLANNED_OBSERVED_MISMATCH,
    REFUSE_TLS_HOST,
    ReviewOrOutcomeError,
    build_observed_openrouter_transport,
    build_planned_openrouter_request,
    planned_request_sha256,
    require_real_or_digests,
    transport_observation_sha256,
)

# ---------------------------------------------------------------------------
# Modes + refuse codes (stable wire)
# ---------------------------------------------------------------------------

MODE_TOOLS_ONLY = "tools_only"
MODE_MEASURED_OPENROUTER = "measured_openrouter_eval_cvm"
LEGAL_LLM_MODES = frozenset({MODE_TOOLS_ONLY, MODE_MEASURED_OPENROUTER})

MEASURED_EVAL_CVM_KIND = "measured_eval_cvm"
BASE_MASTER_KIND = "base_master"
UNMEASURED_HOST_KIND = "unmeasured_host_python"
MINER_LAPTOP_KIND = "miner_laptop"
SIDECAR_PROXY_KIND = "unmeasured_sidecar_proxy"

REFUSE_BASE_GATEWAY = "agent_llm_base_gateway_forbidden"
REFUSE_UNMEASURED_OR = "agent_llm_unmeasured_openrouter"
REFUSE_BASE_GATEWAY_URL = "base_gateway_forbidden"
REFUSE_FLAGS_OFF = "score_refused_flags_off"
REFUSE_TOOLS_ONLY_EGRESS = "agent_llm_tools_only_model_egress"
REFUSE_MISSING_PLANNED_DIGEST = "eval_agent_or_planned_digest_missing"
REFUSE_MISSING_OBSERVED_DIGEST = "eval_agent_or_observed_digest_missing"
REFUSE_DIGEST_MISMATCH = "eval_agent_or_planned_observed_mismatch"
REFUSE_DIGEST_UNBOUND = "eval_agent_or_digests_unbound_score"
REFUSE_MEASUREMENT = "eval_agent_measurement_unallowlisted"
REFUSE_MODE_UNKNOWN = "eval_agent_llm_mode_unknown"
REFUSE_CLAIM_LLM_WITHOUT_DIGESTs = "eval_agent_llm_claim_missing_digests"

# Env / URL surface that must never drive production agent LLM success:
BASE_GATEWAY_ENV_NAMES = frozenset(
    {
        "BASE_GATEWAY_TOKEN",
        "BASE_LLM_GATEWAY_URL",
        "GATEWAY_TOKEN",
        "CENTRAL_GATEWAY_TOKEN",
        "CHALLENGE_LLM_GATEWAY_TOKEN",
        "CHALLENGE_LLM_GATEWAY_BASE_URL",
        "CHALLENGE_LLM_GATEWAY_TOKEN_FILE",
        "CHALLENGE_AGENT_GATEWAY_TOKEN",
    }
)
BASE_GATEWAY_URL_MARKERS = (
    "/llm/v1",
    "BASE_LLM_GATEWAY",
    "BASE_GATEWAY_TOKEN",
)


class EvalAgentLlmError(PermissionError):
    """Fail-closed eval-agent LLM refuse with a stable reason code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True)
class EvalAgentLlmAdmission:
    """Decision for whether an agent LLM claim may contribute to score credit."""

    admitted: bool
    reason_code: str
    mode: str
    production_emit_eligible: bool = False
    planned_request_sha256: str | None = None
    transport_observation_sha256: str | None = None
    runtime_kind: str | None = None
    digests_bound: bool = False
    base_gateway_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reason_code": self.reason_code,
            "mode": self.mode,
            "production_emit_eligible": self.production_emit_eligible,
            "planned_request_sha256": self.planned_request_sha256,
            "transport_observation_sha256": self.transport_observation_sha256,
            "runtime_kind": self.runtime_kind,
            "digests_bound": self.digests_bound,
            "base_gateway_used": self.base_gateway_used,
        }


def _refuse(
    code: str,
    *,
    mode: str,
    runtime_kind: str | None = None,
    base_gateway_used: bool = False,
    planned: str | None = None,
    observed: str | None = None,
) -> EvalAgentLlmAdmission:
    return EvalAgentLlmAdmission(
        admitted=False,
        reason_code=code,
        mode=mode,
        production_emit_eligible=False,
        planned_request_sha256=planned,
        transport_observation_sha256=observed,
        runtime_kind=runtime_kind,
        digests_bound=False,
        base_gateway_used=base_gateway_used,
    )


def normalize_llm_mode(mode: str | None) -> str:
    """Normalize product mode; unknown values refuse at admission."""

    if mode is None or mode == "":
        # Default product: tools-only is always legal; opt into measured OR.
        return MODE_TOOLS_ONLY
    value = str(mode).strip()
    if value not in LEGAL_LLM_MODES:
        raise EvalAgentLlmError(REFUSE_MODE_UNKNOWN, f"unknown llm mode {value!r}")
    return value


def assert_no_base_gateway_agent_env(env: Mapping[str, str] | None) -> None:
    """Refuse residual Base gateway env names in eval-agent material."""

    if not env:
        return
    for key in env:
        upper = str(key).strip().upper()
        if upper in BASE_GATEWAY_ENV_NAMES:
            raise EvalAgentLlmError(
                REFUSE_BASE_GATEWAY,
                f"agent path must not carry Base gateway env {upper}",
            )


def assert_no_base_gateway_url(url_or_base: str | None) -> None:
    """Refuse Base ``/llm/v1`` (and residual gateway token/URL shapes)."""

    if not url_or_base:
        return
    text = str(url_or_base)
    for marker in BASE_GATEWAY_URL_MARKERS:
        if marker in text:
            raise EvalAgentLlmError(
                REFUSE_BASE_GATEWAY_URL,
                f"Base gateway URL/token surface forbidden: marker {marker}",
            )
    lower = text.lower()
    if "/llm/v1" in lower or "base_gateway_token" in lower:
        raise EvalAgentLlmError(REFUSE_BASE_GATEWAY_URL, "Base /llm/v1 forbidden")


def flag_matrix_production_emit(
    *,
    phala_attestation_enabled: bool,
    attested_review_enabled: bool,
) -> dict[str, Any]:
    """Dual-flag matrix: only ON/ON may emit production score ingredients.

    Any OFF refuses emission (VAL-ACAT-054). Residual flag-off eval loops must
    not finalize scores into weight/emission payloads.
    """

    dual_on = bool(phala_attestation_enabled) and bool(attested_review_enabled)
    rows = []
    for phala in (False, True):
        for review in (False, True):
            emit = bool(phala and review)
            rows.append(
                {
                    "phala_attestation_enabled": phala,
                    "attested_review_enabled": review,
                    "production_emit": emit,
                    "refuse_code": None if emit else REFUSE_FLAGS_OFF,
                }
            )
    return {
        "dual_flags_on": dual_on,
        "production_emit": dual_on,
        "refuse_code": None if dual_on else REFUSE_FLAGS_OFF,
        "matrix": rows,
    }


def build_eval_agent_planned_request(
    *,
    body_sha256: str,
    body_length: int,
    routing_sha256: str,
    model: str,
) -> dict[str, Any]:
    """Closed Planned OpenRouter Request for eval-agent transport (reuse v1 keys).

    Model pin for **review** remains product REVIEW_MODEL; eval agents may use
    a challenge-documented model name but still require openrouter.ai origin
    and coherent digests. This constructor reuses the review planned schema when
    model matches REVIEW_MODEL; otherwise it builds the same closed key set and
    still pins origin/path/TLS via observed path checks.

    VAL-AGATE-012/013: no closed agent model catalog; personal finetunes refuse.
    """

    from agent_challenge.evaluation.llm_rules_residual import (
        REFUSE_FINETUNE as _REFUSE_FT,
    )
    from agent_challenge.evaluation.llm_rules_residual import (
        is_personal_finetune_model,
    )
    from agent_challenge.review.schemas import REVIEW_MODEL

    if is_personal_finetune_model(model):
        raise EvalAgentLlmError(_REFUSE_FT, f"personal finetune model refused: {model!r}")

    if model == REVIEW_MODEL:
        return build_planned_openrouter_request(
            body_sha256=body_sha256,
            body_length=body_length,
            routing_sha256=routing_sha256,
            model=model,
        )
    # Non-review model for scored agents: same closed keys/schema, openrouter origin.
    from agent_challenge.review.schemas import (
        OPENROUTER_HEADERS,
        REVIEW_TRANSPORT_SCHEMA_VERSION,
    )

    planned = {
        "schema_version": REVIEW_TRANSPORT_SCHEMA_VERSION,
        "method": "POST",
        "origin": OPENROUTER_ORIGIN,
        "path": OPENROUTER_PATH,
        "headers": dict(OPENROUTER_HEADERS),
        "body_sha256": body_sha256,
        "body_length": int(body_length),
        "model": model,
        "routing_sha256": routing_sha256,
    }
    if planned["body_length"] <= 0:
        raise EvalAgentLlmError(REFUSE_MISSING_PLANNED_DIGEST, "body_length must be positive")
    if not isinstance(body_sha256, str) or len(body_sha256) != 64:
        raise EvalAgentLlmError(REFUSE_MISSING_PLANNED_DIGEST, "body_sha256 required")
    if not isinstance(routing_sha256, str) or len(routing_sha256) != 64:
        raise EvalAgentLlmError(REFUSE_MISSING_PLANNED_DIGEST, "routing_sha256 required")
    return planned


def build_eval_agent_observed_transport(
    *,
    planned: Mapping[str, Any],
    response_body_sha256: str,
    response_body_length: int,
    metadata_sha256: str,
    response_status: int = 200,
) -> dict[str, Any]:
    """Observed OpenRouter transport for eval-agent calls (TLS openrouter.ai)."""

    p_digest = planned_request_sha256(planned)
    try:
        return build_observed_openrouter_transport(
            planned_request_sha256_=p_digest,
            response_body_sha256=response_body_sha256,
            response_body_length=response_body_length,
            metadata_sha256=metadata_sha256,
            response_status=response_status,
        )
    except ReviewOrOutcomeError as exc:
        code = getattr(exc, "code", REFUSE_TLS_HOST)
        raise EvalAgentLlmError(code, str(exc)) from exc


def bind_eval_agent_or_digests_into_score_materials(
    *,
    planned: Mapping[str, Any],
    observed: Mapping[str, Any],
    openrouter_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require planned+observed digests and return score-bound OR materials.

    Materials are carried alongside score-domain binding under dual flags; they
    do **not** mutate the closed score_binding schema v2 field set. Live score
    admission re-checks digests via :func:`require_eval_agent_or_digests`.
    """

    observation = openrouter_observation
    if observation is None:
        # Minimal closed observation from planned/observed digests alone.
        observation = {
            "planned_request_sha256": planned_request_sha256(planned),
            "transport_observation_sha256": transport_observation_sha256(observed),
            "request_body_sha256": planned.get("body_sha256"),
            "request_body_length": planned.get("body_length"),
            "response_status": observed.get("response_status", 200),
            "response_content_encoding": "identity",
            "response_body_sha256": observed.get("response_body_sha256"),
            "response_body_length": observed.get("response_body_length"),
            "response_id": "eval-agent-obs",
            "returned_model": planned.get("model"),
            "metadata_sha256": observed.get("metadata_sha256"),
            "observed_provider": "openrouter",
            "provider_provenance": "openrouter_metadata",
            "cache_hit": False,
        }
    try:
        digests = require_real_or_digests(
            planned=planned,
            observed=observed,
            openrouter_observation=observation,
        )
    except ReviewOrOutcomeError as exc:
        code = getattr(exc, "code", REFUSE_DIGEST_MISMATCH)
        if code == REFUSE_MISSING_PLANNED:
            code = REFUSE_MISSING_PLANNED_DIGEST
        elif code == REFUSE_MISSING_OBSERVED:
            code = REFUSE_MISSING_OBSERVED_DIGEST
        elif code == REFUSE_PLANNED_OBSERVED_MISMATCH:
            code = REFUSE_DIGEST_MISMATCH
        raise EvalAgentLlmError(code, str(exc)) from exc

    return {
        "domain_role": "eval_agent_openrouter",
        "schema_version": 1,
        "planned_request_sha256": digests["planned_request_sha256"],
        "transport_observation_sha256": digests["transport_observation_sha256"],
        "tls_hostname": OPENROUTER_TLS_HOSTNAME,
        "origin": OPENROUTER_ORIGIN,
        "path": OPENROUTER_PATH,
        "planned": dict(planned),
        "observed": dict(observed),
        "openrouter_observation": dict(observation),
    }


def require_eval_agent_or_digests(
    materials: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Re-check planned/observed bind at score admission (fail closed)."""

    if not materials:
        raise EvalAgentLlmError(REFUSE_DIGEST_UNBOUND, "eval agent OR materials missing")
    planned = materials.get("planned")
    observed = materials.get("observed")
    observation = materials.get("openrouter_observation")
    if not isinstance(planned, Mapping) or not isinstance(observed, Mapping):
        raise EvalAgentLlmError(REFUSE_DIGEST_UNBOUND, "planned/observed closed objects required")
    try:
        digests = require_real_or_digests(
            planned=planned,
            observed=observed,
            openrouter_observation=observation if isinstance(observation, Mapping) else None,
        )
    except ReviewOrOutcomeError as exc:
        code = getattr(exc, "code", REFUSE_DIGEST_MISMATCH)
        if code == REFUSE_MISSING_PLANNED:
            code = REFUSE_MISSING_PLANNED_DIGEST
        elif code == REFUSE_MISSING_OBSERVED:
            code = REFUSE_MISSING_OBSERVED_DIGEST
        elif code == REFUSE_PLANNED_OBSERVED_MISMATCH:
            code = REFUSE_DIGEST_MISMATCH
        raise EvalAgentLlmError(code, str(exc)) from exc

    if materials.get("planned_request_sha256") not in (None, digests["planned_request_sha256"]):
        raise EvalAgentLlmError(REFUSE_DIGEST_MISMATCH, "stored planned digest drifted")
    if materials.get("transport_observation_sha256") not in (
        None,
        digests["transport_observation_sha256"],
    ):
        raise EvalAgentLlmError(REFUSE_DIGEST_MISMATCH, "stored observed digest drifted")
    return digests


def admit_eval_agent_llm_for_score(
    *,
    mode: str | None,
    dual_flags_on: bool,
    # Runtime / measurement
    runtime_kind: str | None = None,
    measurement: Mapping[str, str] | None = None,
    allowlist: Sequence[Mapping[str, str]] | None = None,
    # Claims
    claims_model_call: bool = False,
    # Materials (required when mode is measured and claims / requires digests)
    agent_or_materials: Mapping[str, Any] | None = None,
    # Residual Base gateway surfaces (must be empty/absent for success)
    agent_env: Mapping[str, str] | None = None,
    gateway_url: str | None = None,
    gateway_token_present: bool = False,
    used_base_llm_v1: bool = False,
) -> EvalAgentLlmAdmission:
    """Admit agent LLM contribution under production dual flags.

    Tools-only: admits with no digests when no model claim/egress.
    Measured OpenRouter: requires measured_eval_cvm + allowlist + planned/observed.
    Any Base gateway residue refuses. Flag-off refuses production emission.
    """

    try:
        llm_mode = normalize_llm_mode(mode)
    except EvalAgentLlmError as exc:
        return _refuse(exc.code, mode=str(mode or ""))

    # Dual flags: production emission only when ON/ON (VAL-ACAT-054).
    if not dual_flags_on:
        return _refuse(REFUSE_FLAGS_OFF, mode=llm_mode, runtime_kind=runtime_kind)

    # Residual Base gateway never success path (VAL-ACAT-050/053).
    base_gateway_used = bool(
        gateway_token_present
        or used_base_llm_v1
        or (gateway_url is not None and str(gateway_url).strip() != "")
    )
    try:
        assert_no_base_gateway_agent_env(agent_env)
        assert_no_base_gateway_url(gateway_url)
    except EvalAgentLlmError as exc:
        return _refuse(
            exc.code,
            mode=llm_mode,
            runtime_kind=runtime_kind,
            base_gateway_used=True,
        )
    if base_gateway_used or used_base_llm_v1:
        return _refuse(
            REFUSE_BASE_GATEWAY,
            mode=llm_mode,
            runtime_kind=runtime_kind,
            base_gateway_used=True,
        )

    if llm_mode == MODE_TOOLS_ONLY:
        if claims_model_call or agent_or_materials is not None:
            return _refuse(
                REFUSE_TOOLS_ONLY_EGRESS,
                mode=llm_mode,
                runtime_kind=runtime_kind,
            )
        return EvalAgentLlmAdmission(
            admitted=True,
            reason_code="eval_agent_tools_only",
            mode=llm_mode,
            production_emit_eligible=True,
            runtime_kind=runtime_kind,
            digests_bound=False,
            base_gateway_used=False,
        )

    # MODE_MEASURED_OPENROUTER requires measured eval guest + digests.
    kind = (runtime_kind or "").strip()
    if kind in {BASE_MASTER_KIND, UNMEASURED_HOST_KIND, MINER_LAPTOP_KIND, SIDECAR_PROXY_KIND}:
        return _refuse(REFUSE_UNMEASURED_OR, mode=llm_mode, runtime_kind=kind)
    if kind != MEASURED_EVAL_CVM_KIND:
        return _refuse(REFUSE_UNMEASURED_OR, mode=llm_mode, runtime_kind=kind or None)

    if measurement is None:
        return _refuse(REFUSE_MEASUREMENT, mode=llm_mode, runtime_kind=kind)
    if not allowlist:
        return _refuse(REFUSE_MEASUREMENT, mode=llm_mode, runtime_kind=kind)
    closed = {
        "compose_hash": str(measurement.get("compose_hash", "")),
        "os_image_hash": str(measurement.get("os_image_hash", "")),
        "mrtd": str(measurement.get("mrtd", "")),
    }
    matched = False
    for entry in allowlist:
        if (
            str(entry.get("compose_hash", "")) == closed["compose_hash"]
            and str(entry.get("os_image_hash", "")) == closed["os_image_hash"]
            and str(entry.get("mrtd", "")) == closed["mrtd"]
        ):
            matched = True
            break
    if not matched:
        return _refuse(REFUSE_MEASUREMENT, mode=llm_mode, runtime_kind=kind)

    if agent_or_materials is None:
        return _refuse(
            REFUSE_CLAIM_LLM_WITHOUT_DIGESTs if claims_model_call else REFUSE_DIGEST_UNBOUND,
            mode=llm_mode,
            runtime_kind=kind,
        )

    # VAL-AGATE-013: personal finetunes refuse (no closed catalog of allowed models).
    from agent_challenge.evaluation.llm_rules_residual import (
        REFUSE_FINETUNE as _REFUSE_FT,
    )
    from agent_challenge.evaluation.llm_rules_residual import (
        is_personal_finetune_model,
    )

    claimed_model = agent_or_materials.get("model")
    if claimed_model is None and isinstance(agent_or_materials.get("planned"), Mapping):
        claimed_model = agent_or_materials["planned"].get("model")
    if is_personal_finetune_model(str(claimed_model) if claimed_model is not None else None):
        return _refuse(_REFUSE_FT, mode=llm_mode, runtime_kind=kind)

    try:
        digests = require_eval_agent_or_digests(agent_or_materials)
    except EvalAgentLlmError as exc:
        return _refuse(exc.code, mode=llm_mode, runtime_kind=kind)

    return EvalAgentLlmAdmission(
        admitted=True,
        reason_code="eval_agent_measured_openrouter",
        mode=llm_mode,
        production_emit_eligible=True,
        planned_request_sha256=digests["planned_request_sha256"],
        transport_observation_sha256=digests["transport_observation_sha256"],
        runtime_kind=kind,
        digests_bound=True,
        base_gateway_used=False,
    )


def require_eval_agent_llm_for_score(**kwargs: Any) -> EvalAgentLlmAdmission:
    """Fail closed wrapper for live admission call sites."""

    decision = admit_eval_agent_llm_for_score(**kwargs)
    if not decision.admitted:
        raise EvalAgentLlmError(decision.reason_code)
    return decision


# Production residual: GatewayConfigError must never be the production success path.
def refuse_base_gateway_assignment_payload(payload: Mapping[str, Any] | None) -> None:
    """Production score path never builds success from Base gateway payload keys."""

    data = dict(payload or {})
    for key in ("gateway_token", "BASE_GATEWAY_TOKEN", "gateway_url", "gateway_base_url"):
        if data.get(key):
            raise EvalAgentLlmError(
                REFUSE_BASE_GATEWAY,
                f"assignment residual gateway key {key} forbidden for production score",
            )


__all__ = [
    "BASE_GATEWAY_ENV_NAMES",
    "BASE_MASTER_KIND",
    "LEGAL_LLM_MODES",
    "MEASURED_EVAL_CVM_KIND",
    "MINER_LAPTOP_KIND",
    "MODE_MEASURED_OPENROUTER",
    "MODE_TOOLS_ONLY",
    "REFUSE_BASE_GATEWAY",
    "REFUSE_BASE_GATEWAY_URL",
    "REFUSE_CLAIM_LLM_WITHOUT_DIGESTs",
    "REFUSE_DIGEST_MISMATCH",
    "REFUSE_DIGEST_UNBOUND",
    "REFUSE_FLAGS_OFF",
    "REFUSE_MEASUREMENT",
    "REFUSE_MISSING_OBSERVED_DIGEST",
    "REFUSE_MISSING_PLANNED_DIGEST",
    "REFUSE_MODE_UNKNOWN",
    "REFUSE_TOOLS_ONLY_EGRESS",
    "REFUSE_UNMEASURED_OR",
    "SIDECAR_PROXY_KIND",
    "UNMEASURED_HOST_KIND",
    "EvalAgentLlmAdmission",
    "EvalAgentLlmError",
    "admit_eval_agent_llm_for_score",
    "assert_no_base_gateway_agent_env",
    "assert_no_base_gateway_url",
    "bind_eval_agent_or_digests_into_score_materials",
    "build_eval_agent_observed_transport",
    "build_eval_agent_planned_request",
    "flag_matrix_production_emit",
    "normalize_llm_mode",
    "refuse_base_gateway_assignment_payload",
    "require_eval_agent_llm_for_score",
    "require_eval_agent_or_digests",
]
