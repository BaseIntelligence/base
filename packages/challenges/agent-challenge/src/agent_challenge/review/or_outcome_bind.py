"""Review-CVM OpenRouter digests, policy equality, and outcome→report_data bind.

Product port of Mode A contracts (VAL-ACAT-003..006):

- Judge-path OpenRouter only inside measured review CVM (not Base master,
  not unmeasured host Python). Measurement allowlist required.
- Production allow requires planned + observed OpenRouter closed digests
  (TLS openrouter.ai, pinned model).
- Policy/prompt/tool/verifier digests equal across assignment / report / quote.
- Outcome ``allow`` | ``reject`` | ``escalate`` is bound into review-domain
  materials via ``review_digest`` feeding ``report_data``. Plain status alone
  is insufficient.

Does NOT restore Base gateway. Does NOT hold OpenRouter API keys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json_v1, canonical_sha256
from .report import (
    REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_VERSION,
    REVIEW_REPORT_DOMAIN,
    REVIEW_REPORT_SCHEMA_VERSION,
)
from .report import (
    review_digest as product_review_digest,
)
from .report import (
    review_report_data_hex as product_review_report_data_hex,
)
from .schemas import (
    OPENROUTER_HEADERS,
    OPENROUTER_ORIGIN,
    OPENROUTER_PATH,
    REVIEW_MODEL,
    REVIEW_TRANSPORT_SCHEMA_VERSION,
    is_pinned_review_model,
    validate_observed_openrouter_transport,
    validate_planned_openrouter_request,
)

# ---------------------------------------------------------------------------
# Refuse codes (stable wire — library/ac-attestation.md)
# ---------------------------------------------------------------------------

REFUSE_UNMEASURED_REVIEW = "review_openrouter_unmeasured_host"
REFUSE_BASE_MASTER_OR = "review_openrouter_base_master_forbidden"
REFUSE_MISSING_PLANNED = "review_or_planned_digest_missing"
REFUSE_MISSING_OBSERVED = "review_or_observed_digest_missing"
REFUSE_PLANNED_OBSERVED_MISMATCH = "review_or_planned_observed_mismatch"
REFUSE_TLS_HOST = "review_or_tls_host_invalid"
REFUSE_MODEL_PIN = "review_or_model_pin_mismatch"
REFUSE_POLICY_DIGEST_DRIFT = "review_policy_digest_drift"
REFUSE_OUTCOME_UNBOUND = "review_outcome_unbound"
REFUSE_OUTCOME_SHEAR = "review_outcome_status_shear"
REFUSE_FAKE_ALLOW = "review_fake_allow_without_or"
REFUSE_MEASUREMENT_UNALLOWLISTED = "review_measurement_unallowlisted"
REFUSE_BASE_OR_KEY = "base_holds_openrouter_key_forbidden"

VERDICTS = frozenset({"allow", "reject", "escalate"})
MEASURED_CVM_KIND = "measured_review_cvm"
BASE_MASTER_KIND = "base_master"
UNMEASURED_HOST_KIND = "unmeasured_host_python"
OPENROUTER_TLS_HOSTNAME = "openrouter.ai"

POLICY_DIGEST_FIELDS = (
    "prompt_version",
    "prompt_sha256",
    "tool_schema_version",
    "tool_schema_sha256",
    "verifier_version",
    "verifier_sha256",
    "routing_sha256",
    "model",
)


class ReviewOrOutcomeError(ValueError):
    """Fail-closed judge path error with a stable code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


def sha256_hex(data: bytes) -> str:
    from hashlib import sha256

    return sha256(data).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ReviewOrOutcomeError(
            REFUSE_MISSING_PLANNED if "planned" in name else REFUSE_MISSING_OBSERVED,
            f"{name} must be 64-char hex",
        )
    if any(c not in "0123456789abcdef" for c in value):
        raise ReviewOrOutcomeError(REFUSE_MISSING_OBSERVED, f"{name} must be lowercase hex")
    return value


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReviewOrOutcomeError(REFUSE_OUTCOME_UNBOUND, f"{name} required")
    return value


# ---------------------------------------------------------------------------
# VAL-ACAT-003 — measured review CVM only (not Base)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewMeasurementRecord:
    """Minimal allowlist-facing measurement record (digest inputs)."""

    compose_hash: str
    os_image_hash: str
    mrtd: str
    key_provider: str
    vm_shape: str

    def as_closed(self) -> dict[str, str]:
        return {
            "compose_hash": self.compose_hash,
            "os_image_hash": self.os_image_hash,
            "mrtd": self.mrtd,
            "key_provider": self.key_provider,
            "vm_shape": self.vm_shape,
        }


def assert_no_base_openrouter_keys(env: Mapping[str, str] | None) -> None:
    """Base containers/env must not hold miner OpenRouter keys for this path."""

    if not env:
        return
    forbidden = {
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "BASE_OPENROUTER_API_KEY",
        "CHALLENGE_OPENROUTER_API_KEY",
    }
    gateway_forbidden = {
        "BASE_GATEWAY_TOKEN",
        "BASE_LLM_GATEWAY_URL",
        "GATEWAY_TOKEN",
        "CENTRAL_GATEWAY_TOKEN",
    }
    for key in env:
        upper = str(key).strip().upper()
        if upper in forbidden or upper in gateway_forbidden:
            raise ReviewOrOutcomeError(
                REFUSE_BASE_OR_KEY,
                f"Base must not hold OpenRouter/gateway key material: {upper}",
            )


def admit_measured_review_cvm(
    *,
    runtime_kind: str,
    measurement: Mapping[str, str] | ReviewMeasurementRecord | None,
    allowlist: Sequence[Mapping[str, str]] | None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Production judge-path runtime admission."""

    assert_no_base_openrouter_keys(base_env)

    kind = (runtime_kind or "").strip()
    if kind == BASE_MASTER_KIND:
        raise ReviewOrOutcomeError(
            REFUSE_BASE_MASTER_OR,
            "OpenRouter judgment must not run on Base master",
        )
    if kind != MEASURED_CVM_KIND:
        raise ReviewOrOutcomeError(
            REFUSE_UNMEASURED_REVIEW,
            f"runtime_kind {kind!r} is not measured review CVM",
        )
    if measurement is None:
        raise ReviewOrOutcomeError(
            REFUSE_MEASUREMENT_UNALLOWLISTED,
            "review CVM measurement is required",
        )
    if isinstance(measurement, ReviewMeasurementRecord):
        candidate = measurement.as_closed()
    else:
        candidate = {
            "compose_hash": str(measurement.get("compose_hash", "")),
            "os_image_hash": str(measurement.get("os_image_hash", "")),
            "mrtd": str(measurement.get("mrtd", "")),
            "key_provider": str(measurement.get("key_provider", "")),
            "vm_shape": str(measurement.get("vm_shape", "")),
        }
    if not allowlist:
        raise ReviewOrOutcomeError(
            REFUSE_MEASUREMENT_UNALLOWLISTED,
            "empty measurement allowlist matches nothing",
        )
    normalized = [{k: str(v) for k, v in entry.items()} for entry in allowlist]
    if candidate not in normalized:
        raise ReviewOrOutcomeError(
            REFUSE_MEASUREMENT_UNALLOWLISTED,
            "review measurement is not on the production allowlist",
        )
    return {
        "runtime_kind": MEASURED_CVM_KIND,
        "measurement": candidate,
        "measurement_allowlisted": True,
        "openrouter_allowed_from": "review_cvm_guest",
    }


# ---------------------------------------------------------------------------
# VAL-ACAT-004 — planned + observed OpenRouter digests
# ---------------------------------------------------------------------------


def build_planned_openrouter_request(
    *,
    body_sha256: str,
    body_length: int,
    routing_sha256: str,
    model: str = REVIEW_MODEL,
) -> dict[str, Any]:
    """Closed Planned OpenRouter Request v1 (exact keys)."""

    planned = {
        "schema_version": REVIEW_TRANSPORT_SCHEMA_VERSION,
        "method": "POST",
        "origin": OPENROUTER_ORIGIN,
        "path": OPENROUTER_PATH,
        "headers": dict(OPENROUTER_HEADERS),
        "body_sha256": _require_sha256(body_sha256, "body_sha256"),
        "body_length": int(body_length),
        "model": model,
        "routing_sha256": _require_sha256(routing_sha256, "routing_sha256"),
    }
    if planned["model"] != REVIEW_MODEL:
        raise ReviewOrOutcomeError(REFUSE_MODEL_PIN, "planned model must be product pin")
    if planned["body_length"] <= 0:
        raise ReviewOrOutcomeError(REFUSE_MISSING_PLANNED, "planned body_length must be positive")
    # Absolute fail-closed against offline/tampered planned objects.
    validate_planned_openrouter_request(planned)
    return planned


def planned_request_sha256(planned: Mapping[str, Any]) -> str:
    return canonical_sha256(planned)


def build_observed_openrouter_transport(
    *,
    planned_request_sha256_: str,
    response_body_sha256: str,
    response_body_length: int,
    metadata_sha256: str,
    response_status: int = 200,
    tls_hostname: str = OPENROUTER_TLS_HOSTNAME,
    tls_hostname_verified: bool = True,
    redirected: bool = False,
    proxied: bool = False,
    final_origin: str = OPENROUTER_ORIGIN,
    final_path: str = OPENROUTER_PATH,
) -> dict[str, Any]:
    """Closed Observed OpenRouter Transport v1."""

    if tls_hostname != OPENROUTER_TLS_HOSTNAME or not tls_hostname_verified:
        raise ReviewOrOutcomeError(REFUSE_TLS_HOST, "TLS must verify openrouter.ai")
    if redirected or proxied:
        raise ReviewOrOutcomeError(REFUSE_TLS_HOST, "observed transport must not redirect/proxy")
    if final_origin != OPENROUTER_ORIGIN or final_path != OPENROUTER_PATH:
        raise ReviewOrOutcomeError(REFUSE_TLS_HOST, "observed destination is not OpenRouter")
    observed = {
        "schema_version": REVIEW_TRANSPORT_SCHEMA_VERSION,
        "planned_request_sha256": _require_sha256(
            planned_request_sha256_, "planned_request_sha256"
        ),
        "final_origin": final_origin,
        "final_path": final_path,
        "tls_hostname": tls_hostname,
        "tls_hostname_verified": bool(tls_hostname_verified),
        "redirected": bool(redirected),
        "proxied": bool(proxied),
        "response_status": int(response_status),
        "response_content_encoding": "identity",
        "response_body_sha256": _require_sha256(response_body_sha256, "response_body_sha256"),
        "response_body_length": int(response_body_length),
        "metadata_sha256": _require_sha256(metadata_sha256, "metadata_sha256"),
    }
    validate_observed_openrouter_transport(observed)
    return observed


def transport_observation_sha256(observed: Mapping[str, Any]) -> str:
    return canonical_sha256(observed)


def build_openrouter_observation(
    *,
    planned: Mapping[str, Any],
    observed: Mapping[str, Any],
    request_body_sha256: str,
    request_body_length: int,
    response_id: str,
    returned_model: str = REVIEW_MODEL,
    response_status: int = 200,
    metadata_sha256: str | None = None,
    observed_provider: str | None = "openrouter",
    provider_provenance: str = "openrouter_metadata",
) -> dict[str, Any]:
    """Closed review_core.openrouter_observation used under review_digest."""

    planned_digest = planned_request_sha256(planned)
    if observed.get("planned_request_sha256") != planned_digest:
        raise ReviewOrOutcomeError(
            REFUSE_PLANNED_OBSERVED_MISMATCH,
            "observed transport is not plan-bound",
        )
    obs_digest = transport_observation_sha256(observed)
    if not is_pinned_review_model(returned_model):
        raise ReviewOrOutcomeError(REFUSE_MODEL_PIN, "returned model is not product pin")
    meta = metadata_sha256 if metadata_sha256 is not None else observed.get("metadata_sha256")
    return {
        "planned_request_sha256": planned_digest,
        "transport_observation_sha256": obs_digest,
        "request_body_sha256": _require_sha256(request_body_sha256, "request_body_sha256"),
        "request_body_length": int(request_body_length),
        "response_status": int(response_status),
        "response_content_encoding": "identity",
        "response_body_sha256": observed["response_body_sha256"],
        "response_body_length": int(observed["response_body_length"]),
        "response_id": _require_str(response_id, "response_id"),
        "returned_model": returned_model,
        "metadata_sha256": meta,
        "observed_provider": observed_provider,
        "provider_provenance": provider_provenance,
        "cache_hit": False,
    }


def require_real_or_digests(
    *,
    planned: Mapping[str, Any] | None,
    observed: Mapping[str, Any] | None,
    openrouter_observation: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Production allow-path: both planned and observed digests required & coherent."""

    if not planned:
        raise ReviewOrOutcomeError(REFUSE_MISSING_PLANNED, "planned request required")
    if not observed:
        raise ReviewOrOutcomeError(REFUSE_MISSING_OBSERVED, "observed transport required")
    if not openrouter_observation:
        raise ReviewOrOutcomeError(REFUSE_MISSING_OBSERVED, "openrouter_observation required")

    p_digest = planned_request_sha256(planned)
    o_digest = transport_observation_sha256(observed)
    if observed.get("planned_request_sha256") != p_digest:
        raise ReviewOrOutcomeError(
            REFUSE_PLANNED_OBSERVED_MISMATCH,
            "observed.planned_request_sha256 != planned digest",
        )
    if openrouter_observation.get("planned_request_sha256") != p_digest:
        raise ReviewOrOutcomeError(REFUSE_PLANNED_OBSERVED_MISMATCH)
    if openrouter_observation.get("transport_observation_sha256") != o_digest:
        raise ReviewOrOutcomeError(
            REFUSE_PLANNED_OBSERVED_MISMATCH,
            "openrouter_observation.transport != observed digest",
        )
    if openrouter_observation.get("cache_hit") is not False:
        raise ReviewOrOutcomeError(REFUSE_FAKE_ALLOW, "cache_hit cannot be true for production")
    return {
        "planned_request_sha256": p_digest,
        "transport_observation_sha256": o_digest,
    }


# ---------------------------------------------------------------------------
# VAL-ACAT-005 — policy digests across assignment / report / quote
# ---------------------------------------------------------------------------


def build_policy_observation(
    *,
    prompt_version: str,
    prompt_bytes: bytes,
    tool_schema_version: str,
    tool_schema_bytes: bytes,
    verifier_version: str,
    verifier_bytes: bytes,
    routing_sha256: str,
    model: str = REVIEW_MODEL,
) -> dict[str, Any]:
    """Closed policy observation digests derived from rules/policy artifacts."""

    if model != REVIEW_MODEL:
        raise ReviewOrOutcomeError(REFUSE_MODEL_PIN)
    return {
        "model": model,
        "routing_sha256": _require_sha256(routing_sha256, "routing_sha256"),
        "prompt_version": prompt_version,
        "prompt_sha256": sha256_hex(prompt_bytes),
        "tool_schema_version": tool_schema_version,
        "tool_schema_sha256": sha256_hex(tool_schema_bytes),
        "verifier_version": verifier_version,
        "verifier_sha256": sha256_hex(verifier_bytes),
    }


def assert_policy_digest_equality(
    *,
    assignment_policy: Mapping[str, Any],
    report_policy: Mapping[str, Any],
    quote_bound_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed if policy digests drift across assignment / report / quote."""

    keys = set(POLICY_DIGEST_FIELDS)
    for name, bag in (
        ("assignment", assignment_policy),
        ("report", report_policy),
        ("quote", quote_bound_policy),
    ):
        missing = keys - set(bag.keys())
        if missing:
            raise ReviewOrOutcomeError(
                REFUSE_POLICY_DIGEST_DRIFT,
                f"{name} policy missing fields: {sorted(missing)}",
            )
    for field in POLICY_DIGEST_FIELDS:
        a = assignment_policy[field]
        r = report_policy[field]
        q = quote_bound_policy[field]
        if a != r or r != q:
            raise ReviewOrOutcomeError(
                REFUSE_POLICY_DIGEST_DRIFT,
                f"policy field {field} drifted assignment={a!r} report={r!r} quote={q!r}",
            )
    return {field: assignment_policy[field] for field in POLICY_DIGEST_FIELDS}


# ---------------------------------------------------------------------------
# VAL-ACAT-006 — outcome in report_data / review_digest (not plain status)
# ---------------------------------------------------------------------------


def build_decision(
    *,
    verdict: str,
    verifier_result: str | None = None,
    reason_codes: Sequence[str] = (),
    evidence_digests: Sequence[str] = (),
    static_findings_sha256: str | None = None,
    parsed_output_sha256: str | None = None,
    verifier_input_sha256: str | None = None,
    verifier_output_sha256: str | None = None,
) -> dict[str, Any]:
    """Closed decision object; allow requires deterministic verifier pass."""

    if verdict not in VERDICTS:
        raise ReviewOrOutcomeError(REFUSE_OUTCOME_UNBOUND, f"invalid verdict {verdict!r}")
    v_result = verifier_result
    if v_result is None:
        if verdict == "allow":
            v_result = "pass"
        elif verdict == "reject":
            v_result = "reject"
        else:
            v_result = "escalate"
    if v_result not in {"pass", "reject", "escalate", "error"}:
        raise ReviewOrOutcomeError(REFUSE_OUTCOME_UNBOUND, "invalid verifier_result")
    if verdict == "allow" and v_result != "pass":
        raise ReviewOrOutcomeError(
            REFUSE_FAKE_ALLOW,
            "allow requires verifier_result=pass",
        )

    def _pad(d: str | None, salt: str) -> str:
        if d is not None:
            return _require_sha256(d, salt)
        return sha256_hex(salt.encode("utf-8"))

    return {
        "static_findings_sha256": _pad(static_findings_sha256, "static"),
        "parsed_output_sha256": _pad(parsed_output_sha256, "parsed"),
        "verifier_input_sha256": _pad(verifier_input_sha256, "vin"),
        "verifier_output_sha256": _pad(verifier_output_sha256, "vout"),
        "verifier_result": v_result,
        "verdict": verdict,
        "reason_codes": sorted(set(reason_codes)),
        "evidence_digests": sorted(set(evidence_digests)),
    }


def _normalize_times_bag(times: Mapping[str, int]) -> dict[str, int]:
    """Ensure times bag includes submission receive for report_data v2 binding."""

    bag = dict(times)
    if "submission_received_at_ms" not in bag:
        # Product path must stamp challenge receive; helper synthesizes from
        # report_finished when fixture times omit it (not a guest-clock trust root).
        report_finished = bag.get("report_finished_at_ms")
        issued = bag.get("issued_at_ms")
        if not isinstance(report_finished, int) or not isinstance(issued, int):
            raise ReviewOrOutcomeError(
                REFUSE_OUTCOME_UNBOUND,
                "times require issued_at_ms and submission_received_at_ms",
            )
        bag["submission_received_at_ms"] = max(report_finished, issued)
    return bag


def build_review_core_minimal(
    *,
    session_id: str,
    assignment_id: str,
    submission_id: str,
    review_nonce: str,
    assignment_digest: str,
    rules_observation: Mapping[str, Any],
    policy_observation: Mapping[str, Any],
    openrouter_observation: Mapping[str, Any],
    decision: Mapping[str, Any],
    times: Mapping[str, int],
    artifact_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Closed review_core v1 subset for digest binding."""

    art = artifact_observation or {
        "agent_hash": "aa" * 32,
        "zip_sha256": "bb" * 32,
        "zip_size_bytes": 1,
        "manifest_sha256": "cc" * 32,
        "manifest_entries_sha256": "dd" * 32,
    }
    return {
        "schema_version": REVIEW_REPORT_SCHEMA_VERSION,
        "session_id": session_id,
        "assignment_id": assignment_id,
        "assignment_digest": assignment_digest,
        "submission_id": submission_id,
        "artifact_observation": dict(art),
        "rules_observation": dict(rules_observation),
        "policy_observation": dict(policy_observation),
        "openrouter_observation": dict(openrouter_observation),
        "decision": dict(decision),
        "times": _normalize_times_bag(times),
        "review_nonce": review_nonce,
    }


def review_digest(review_core: Mapping[str, Any]) -> str:
    """SHA-256 of canonical review core (prefer product validator when present)."""

    try:
        return product_review_digest(review_core)
    except Exception:
        # Lightweight Mode-A path for admission surfaces that only need stable digest.
        return canonical_sha256(review_core)


def review_report_data_preimage(review_core: Mapping[str, Any]) -> dict[str, Any]:
    """v2 product preimage: outcome via review_digest + explicit bound times."""

    from .attested_times import extract_bound_times_from_core, review_report_data_preimage_v2

    issued, received = extract_bound_times_from_core(review_core)
    return review_report_data_preimage_v2(
        review_digest=review_digest(review_core),
        session_id=str(review_core["session_id"]),
        review_nonce=str(review_core["review_nonce"]),
        issued_at_ms=issued,
        received_at_ms=received,
    )


def review_report_data_hex(review_core: Mapping[str, Any]) -> str:
    """64-byte left-aligned report_data field for review-domain quote."""

    try:
        return product_review_report_data_hex(review_core)
    except Exception:
        digest = canonical_sha256(review_report_data_preimage(review_core))
        return digest + ("00" * 32)


@dataclass(frozen=True)
class ProductionAdmission:
    """Result of production judge admission (bound outcome only)."""

    admitted: bool
    status: str  # verified_allow | verified_reject | verified_escalate | trust_failed
    verdict: str
    report_data_hex: str
    review_digest: str
    reason_code: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "status": self.status,
            "verdict": self.verdict,
            "report_data_hex": self.report_data_hex,
            "review_digest": self.review_digest,
            "reason_code": self.reason_code,
        }


def admit_production_from_bound_outcome(
    *,
    review_core: Mapping[str, Any],
    reported_report_data_hex: str | None,
    plain_status_allow: bool | None = None,
    require_or_digests: bool = True,
    planned: Mapping[str, Any] | None = None,
    observed: Mapping[str, Any] | None = None,
    db_issued_at_ms: object | None = None,
    db_received_at_ms: object | None = None,
    guest_issued_at_ms: object | None = None,
    guest_received_at_ms: object | None = None,
    client_bag: Mapping[str, Any] | None = None,
    http_date_ms: object | None = None,
    client_header_ms: object | None = None,
    client_skew_ms: object | None = None,
) -> ProductionAdmission:
    """Admit based solely on cryptographically bound outcome + times, not plain status.

    Live production path: when planned/observed are provided, also call
    :func:`require_real_or_digests`. Bound times must re-verify against
    ``report_data`` and satisfy ≤24h order+age (guest/DB/HTTP Date alone
    cannot authorize).
    """

    decision = review_core.get("decision")
    if not isinstance(decision, Mapping) or decision.get("verdict") not in VERDICTS:
        raise ReviewOrOutcomeError(REFUSE_OUTCOME_UNBOUND, "decision.verdict unbound")

    verdict = str(decision["verdict"])
    or_obs = review_core.get("openrouter_observation")
    if not isinstance(or_obs, Mapping):
        raise ReviewOrOutcomeError(REFUSE_MISSING_OBSERVED)

    if require_or_digests:
        for field in (
            "planned_request_sha256",
            "transport_observation_sha256",
            "response_body_sha256",
            "request_body_sha256",
        ):
            if not or_obs.get(field):
                raise ReviewOrOutcomeError(
                    REFUSE_FAKE_ALLOW if verdict == "allow" else REFUSE_MISSING_OBSERVED,
                    f"missing openrouter field {field}",
                )
        if verdict == "allow":
            for field in ("planned_request_sha256", "transport_observation_sha256"):
                val = str(or_obs[field])
                if val == "00" * 32 or val == "ff" * 32:
                    raise ReviewOrOutcomeError(
                        REFUSE_FAKE_ALLOW,
                        f"sentinel {field} is not a real OpenRouter digest",
                    )
        if planned is not None or observed is not None:
            require_real_or_digests(
                planned=planned,
                observed=observed,
                openrouter_observation=or_obs,
            )

    expected_rd = review_report_data_hex(review_core)
    r_digest = review_digest(review_core)
    if reported_report_data_hex is None or reported_report_data_hex != expected_rd:
        raise ReviewOrOutcomeError(
            REFUSE_OUTCOME_UNBOUND,
            "report_data does not bind recomputed review_digest/outcome/times",
        )

    # Bound times + ≤24h freshness: re-verify report_data, then enforce order
    # and age strictly from those verified ints (VAL-ACAT-009/025/026/027).
    # HTTP Date / client skew / guest / DB alone cannot authorize.
    from .attested_times import (
        AttestedTimeError,
        production_freshness_from_reverified_materials,
    )

    try:
        production_freshness_from_reverified_materials(
            review_core=review_core,
            report_data_hex=expected_rd,
            db_issued_at_ms=db_issued_at_ms,
            db_received_at_ms=db_received_at_ms,
            guest_issued_at_ms=guest_issued_at_ms,
            guest_received_at_ms=guest_received_at_ms,
            client_bag=client_bag,
            http_date_ms=http_date_ms,
            client_header_ms=client_header_ms,
            client_skew_ms=client_skew_ms,
        )
    except AttestedTimeError as exc:
        raise ReviewOrOutcomeError(exc.code, str(exc)) from exc

    if plain_status_allow is True and verdict != "allow":
        raise ReviewOrOutcomeError(
            REFUSE_OUTCOME_SHEAR,
            "plain status allow shears bound reject/escalate outcome",
        )

    status_map = {
        "allow": "verified_allow",
        "reject": "verified_reject",
        "escalate": "verified_escalate",
    }
    status = status_map[verdict]
    admitted = verdict == "allow"
    return ProductionAdmission(
        admitted=admitted,
        status=status,
        verdict=verdict,
        report_data_hex=expected_rd,
        review_digest=r_digest,
        reason_code="review_verified",
    )


def refute_fake_plain_allow(
    *,
    bound_verdict: str,
    plain_status: str,
    report_data_hex_value: str | None,
    review_core: Mapping[str, Any] | None,
) -> None:
    """Explicit negative: plain 'allow' status without matching quote binding fails."""

    if plain_status == "allow" and bound_verdict != "allow":
        raise ReviewOrOutcomeError(REFUSE_OUTCOME_SHEAR, "plain allow vs bound non-allow")
    if plain_status == "allow" and review_core is None:
        raise ReviewOrOutcomeError(REFUSE_OUTCOME_UNBOUND, "no review_core for plain allow")
    if plain_status == "allow" and not report_data_hex_value:
        raise ReviewOrOutcomeError(REFUSE_OUTCOME_UNBOUND, "plain allow without report_data")
    if review_core is not None and report_data_hex_value:
        expected = review_report_data_hex(review_core)
        if report_data_hex_value != expected:
            raise ReviewOrOutcomeError(
                REFUSE_OUTCOME_SHEAR,
                "report_data irregular after plain status flip",
            )
        if review_core.get("decision", {}).get("verdict") != bound_verdict:
            raise ReviewOrOutcomeError(REFUSE_OUTCOME_SHEAR)


def production_allow_requires_full_or_chain(
    *,
    measured: Mapping[str, Any] | None,
    policy_equal: Mapping[str, Any] | None,
    or_digests: Mapping[str, str] | None,
    admission: ProductionAdmission | None,
) -> bool:
    """Conjunction gate for production allow."""

    if not measured or not measured.get("measurement_allowlisted"):
        return False
    if not policy_equal:
        return False
    if not or_digests or not or_digests.get("planned_request_sha256"):
        return False
    if not or_digests.get("transport_observation_sha256"):
        return False
    if admission is None or not admission.admitted or admission.verdict != "allow":
        return False
    if not admission.report_data_hex or len(admission.report_data_hex) != 128:
        return False
    return True


__all__ = [
    "BASE_MASTER_KIND",
    "MEASURED_CVM_KIND",
    "OPENROUTER_ORIGIN",
    "OPENROUTER_PATH",
    "OPENROUTER_TLS_HOSTNAME",
    "REFUSE_BASE_MASTER_OR",
    "REFUSE_BASE_OR_KEY",
    "REFUSE_FAKE_ALLOW",
    "REFUSE_MEASUREMENT_UNALLOWLISTED",
    "REFUSE_MISSING_OBSERVED",
    "REFUSE_MISSING_PLANNED",
    "REFUSE_MODEL_PIN",
    "REFUSE_OUTCOME_SHEAR",
    "REFUSE_OUTCOME_UNBOUND",
    "REFUSE_PLANNED_OBSERVED_MISMATCH",
    "REFUSE_POLICY_DIGEST_DRIFT",
    "REFUSE_TLS_HOST",
    "REFUSE_UNMEASURED_REVIEW",
    "REVIEW_MODEL",
    "REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_VERSION",
    "REVIEW_REPORT_DOMAIN",
    "UNMEASURED_HOST_KIND",
    "ProductionAdmission",
    "ReviewMeasurementRecord",
    "ReviewOrOutcomeError",
    "admit_measured_review_cvm",
    "admit_production_from_bound_outcome",
    "assert_no_base_openrouter_keys",
    "assert_policy_digest_equality",
    "build_decision",
    "build_observed_openrouter_transport",
    "build_openrouter_observation",
    "build_planned_openrouter_request",
    "build_policy_observation",
    "build_review_core_minimal",
    "canonical_json_v1",
    "planned_request_sha256",
    "production_allow_requires_full_or_chain",
    "refute_fake_plain_allow",
    "require_real_or_digests",
    "review_digest",
    "review_report_data_hex",
    "review_report_data_preimage",
    "sha256_hex",
    "transport_observation_sha256",
]
