"""Production score admission: full attestation chain re-verify, fail closed.

Product freeze (library/ac-attestation.md, VAL-ACAT-011/012/017/018/030–037/039/040):

- Score admission re-verifies the **conjunction** of:
  1. review-domain quote materials (bound times + ≤24h + OR digests + allow),
  2. key-release domain RA-TLS grant (when dual flags ON),
  3. score-domain attestation binding (``base-agent-challenge-v1``) with
     measurements / agent_hash / task_ids / scores_digest / eval_run_id /
     score_nonce domain separation.
- Missing / tampered / stale / replay / domain confusion → refuse with **no**
  partial score, no silent downgrade to AST/local/flag-off emission.
- Offline AST / “tutte” green paths **never** alone produce production scores.
- Nonces are single-use and domain-separated (review ≠ key-release ≠ score).
- Cached “ok” bits (master status, DB key_granted_at alone, prior green) never
  substitute for cryptographic re-verify under dual flags ON.

Does **not** restore Base LLM gateway. Does **not** invent REAL-PROVIDER PASS.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.eval_agent_llm import (
    MODE_TOOLS_ONLY,
    admit_eval_agent_llm_for_score,
)
from agent_challenge.evaluation.eval_agent_llm import (
    REFUSE_FLAGS_OFF as AGENT_LLM_FLAGS_OFF,
)
from agent_challenge.evaluation.fresh_review_gate import (
    REFUSE_STALE as REVIEW_REFUSE_STALE,
)
from agent_challenge.evaluation.fresh_review_gate import (
    admit_eval_cvm_fresh_review,
)
from agent_challenge.evaluation.llm_rules_residual import (
    REFUSE_HOST_ONLY as PACKAGE_REFUSE_HOST_ONLY,
)
from agent_challenge.evaluation.llm_rules_residual import (
    REFUSE_PACKAGE_TREE_MISSING as PACKAGE_REFUSE_TREE_MISSING,
)
from agent_challenge.evaluation.llm_rules_residual import (
    REFUSE_RESIDUAL_FAIL as PACKAGE_REFUSE_RESIDUAL_FAIL,
)
from agent_challenge.evaluation.llm_rules_residual import (
    REFUSE_RESIDUAL_MISSING as PACKAGE_REFUSE_RESIDUAL_MISSING,
)
from agent_challenge.evaluation.llm_rules_residual import (
    REFUSE_RESIDUAL_UNBOUND as PACKAGE_REFUSE_RESIDUAL_UNBOUND,
)
from agent_challenge.review.or_outcome_bind import (
    REVIEW_REPORT_DOMAIN,
    ReviewOrOutcomeError,
    admit_production_from_bound_outcome,
)

# AGATE package residual refuse codes preserved through score admission.
_PACKAGE_PROOF_REFUSE_CODES = frozenset(
    {
        PACKAGE_REFUSE_RESIDUAL_MISSING,
        PACKAGE_REFUSE_RESIDUAL_FAIL,
        PACKAGE_REFUSE_RESIDUAL_UNBOUND,
        PACKAGE_REFUSE_HOST_ONLY,
        PACKAGE_REFUSE_TREE_MISSING,
        "package_residual_kind_invalid",
        "rules_digests_missing",
    }
)

# ---------------------------------------------------------------------------
# Stable refuse codes (library/ac-attestation.md + score-chain wire)
# ---------------------------------------------------------------------------

REFUSE_INCOMPLETE_CHAIN = "score_refused_incomplete_chain"
REFUSE_AST_ONLY = "score_refused_ast_only"
REFUSE_FLAGS_OFF = "score_refused_flags_off"
REFUSE_MISSING_KEY_RELEASE = "score_refused_missing_key_release"
REFUSE_KEY_RELEASE_MISMATCH = "score_refused_key_release_mismatch"
REFUSE_KEY_RELEASE_DOMAIN = "score_refused_key_release_domain"
REFUSE_NONCE_REPLAY = "score_refused_nonce_replay"
REFUSE_NONCE_STALE = "score_refused_nonce_stale"
REFUSE_DOMAIN_CONFUSION = "score_refused_domain_confusion"
REFUSE_TAMPERED = "score_refused_tampered"
REFUSE_STICKY = "score_refused_sticky_reverify"
REFUSE_REVIEW = "score_refused_review_chain"
REFUSE_REVIEW_STALE = "attestation_stale_over_24h"
REFUSE_SCORE_DOMAIN = "score_refused_score_domain"
REFUSE_PARTIAL_FORBIDDEN = "score_refused_partial_forbidden"
REFUSE_MISSING_REVIEW = "review_attestation_missing"
REFUSE_PACKAGE_TREE_MISSING = "score_refused_package_tree_sha_missing"
REFUSE_PACKAGE_PROOF = "score_refused_package_proof"

# Domains (byte-identical to eval_wire / keyrelease client)
SCORE_DOMAIN = ew.SCORE_DOMAIN
KEY_RELEASE_DOMAIN = ew.KEY_RELEASE_DOMAIN
REVIEW_DOMAIN = REVIEW_REPORT_DOMAIN

# Process-local cache of KR grant materials. Production authority is the durable
# EvalRun.key_release_grant_json column stamped at KR success. This dict is a
# same-process fast path only; multi-worker / restart must reload from DB
# (VAL-ACAT-036/037). Empty lookup still fails closed.
_KEY_RELEASE_GRANT_BY_RUN: dict[str, dict[str, Any]] = {}


def register_key_release_grant_for_score(
    eval_run_id: str,
    grant: Mapping[str, Any],
) -> None:
    """Cache KR grant materials in-process for same-worker score re-verify.

    Production writers must also persist ``EvalRun.key_release_grant_json`` via
    ``authorization.mark_eval_key_granted`` / ``persist_key_release_grant_materials``.
    """

    if not isinstance(eval_run_id, str) or not eval_run_id:
        return
    if not isinstance(grant, Mapping):
        return
    _KEY_RELEASE_GRANT_BY_RUN[eval_run_id] = dict(grant)


def lookup_key_release_grant_for_score(eval_run_id: str) -> dict[str, Any] | None:
    """Return process-local KR grant materials, or None.

    Prefer durable ``run.key_release_grant_json`` via
    ``direct_result._key_release_grant_from_result`` for multi-worker paths.
    """

    if not isinstance(eval_run_id, str) or not eval_run_id:
        return None
    grant = _KEY_RELEASE_GRANT_BY_RUN.get(eval_run_id)
    return dict(grant) if isinstance(grant, Mapping) else None


def clear_key_release_grant_for_score(eval_run_id: str | None = None) -> None:
    """Test helper: clear one or all process-local KR grants (simulates restart)."""

    if eval_run_id is None:
        _KEY_RELEASE_GRANT_BY_RUN.clear()
        return
    _KEY_RELEASE_GRANT_BY_RUN.pop(eval_run_id, None)


def load_durable_key_release_grant(run: Any) -> dict[str, Any] | None:
    """Load KR grant materials from a durable EvalRun column and rehydrate cache.

    Returns ``None`` when materials are missing (score path must then refuse
    under dual flags). Does not invent grants from ``key_granted_at`` alone.
    """

    raw = getattr(run, "key_release_grant_json", None)
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, Mapping):
        return None
    grant = dict(parsed)
    eval_run_id = getattr(run, "eval_run_id", None)
    if isinstance(eval_run_id, str) and eval_run_id:
        register_key_release_grant_for_score(eval_run_id, grant)
    return grant


class ScoreChainAdmissionError(PermissionError):
    """Fail-closed production score refuse with a stable reason code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True)
class ScoreChainAdmission:
    """Decision for whether a production score may be emitted / weighted."""

    admitted: bool
    reason_code: str
    reverify_exercised: bool = False
    score: float | None = None  # always None on refuse (zero partial)
    production_emit: bool = False
    partial_score: bool = False  # always False — API surface for ablation asserts
    domains_checked: tuple[str, ...] = ()
    review_digest: str | None = None
    key_release_binding_hex: str | None = None
    score_report_data_hex: str | None = None
    sticky: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reason_code": self.reason_code,
            "reverify_exercised": self.reverify_exercised,
            "score": self.score,
            "production_emit": self.production_emit,
            "partial_score": self.partial_score,
            "domains_checked": list(self.domains_checked),
            "review_digest": self.review_digest,
            "key_release_binding_hex": self.key_release_binding_hex,
            "score_report_data_hex": self.score_report_data_hex,
            "sticky": self.sticky,
        }


def _refuse(
    code: str,
    *,
    reverify_exercised: bool = False,
    domains_checked: tuple[str, ...] = (),
    review_digest: str | None = None,
    sticky: bool = False,
) -> ScoreChainAdmission:
    return ScoreChainAdmission(
        admitted=False,
        reason_code=code,
        reverify_exercised=reverify_exercised,
        score=None,
        production_emit=False,
        partial_score=False,
        domains_checked=domains_checked,
        review_digest=review_digest,
        sticky=sticky,
    )


def _parse_mapping(value: Mapping[str, Any] | str | bytes | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        try:
            raw = value if isinstance(value, str) else value.decode("utf-8")
            parsed = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError):
            return None
        return parsed if isinstance(parsed, Mapping) else None
    if isinstance(value, Mapping):
        return value
    return None


def recompute_key_release_report_data_hex(
    *,
    eval_run_id: str,
    key_release_nonce: str,
    ra_tls_spki_digest: str,
) -> str:
    """Recompute schema-v2 key-release report_data (admission-time re-check)."""

    return ew.key_release_report_data_hex(
        eval_run_id=eval_run_id,
        key_release_nonce=key_release_nonce,
        ra_tls_spki_digest=ra_tls_spki_digest,
    )


def verify_key_release_grant(
    *,
    grant: Mapping[str, Any] | None,
    eval_plan: Mapping[str, Any],
    key_granted_flag: bool = False,
) -> tuple[str | None, str | None]:
    """Re-verify key-release RA-TLS grant at score admission.

    Returns ``(None, report_data_hex)`` on success or ``(refuse_code, None)``
    on failure. DB ``key_granted_at`` alone is **not** sufficient (VAL-ACAT-037).
    """

    plan_eval_run_id = str(eval_plan.get("eval_run_id") or "")
    plan_kr_nonce = str(eval_plan.get("key_release_nonce") or "")
    if not plan_eval_run_id or not plan_kr_nonce:
        return REFUSE_INCOMPLETE_CHAIN, None

    # AGATE VAL-AGATE-009: KR grant path requires plan package_tree_sha proof.
    plan_tree = eval_plan.get("package_tree_sha")
    if not isinstance(plan_tree, str) or len(plan_tree.strip()) != 64:
        return REFUSE_PACKAGE_TREE_MISSING, None
    try:
        int(plan_tree.strip(), 16)
    except ValueError:
        return REFUSE_PACKAGE_TREE_MISSING, None

    if grant is None:
        # Historic env-inject flag without admission-time materials → refuse.
        if key_granted_flag:
            return REFUSE_MISSING_KEY_RELEASE, None
        return REFUSE_MISSING_KEY_RELEASE, None

    domain = grant.get("domain")
    if domain is not None and str(domain) != KEY_RELEASE_DOMAIN:
        # Cross-domain transplant (e.g. review or score materials as "grant").
        if str(domain) in {SCORE_DOMAIN, REVIEW_DOMAIN}:
            return REFUSE_DOMAIN_CONFUSION, None
        return REFUSE_KEY_RELEASE_DOMAIN, None

    grant_eval = grant.get("eval_run_id")
    grant_nonce = grant.get("key_release_nonce")
    grant_spki = grant.get("ra_tls_spki_digest")
    if not (
        isinstance(grant_eval, str)
        and grant_eval
        and isinstance(grant_nonce, str)
        and grant_nonce
        and isinstance(grant_spki, str)
        and len(grant_spki) == 64
    ):
        return REFUSE_MISSING_KEY_RELEASE, None

    if grant_eval != plan_eval_run_id:
        # Borrowed grant from another eval_run / session.
        return REFUSE_KEY_RELEASE_MISMATCH, None
    if grant_nonce != plan_kr_nonce:
        return REFUSE_KEY_RELEASE_MISMATCH, None

    # Domain separation: score_nonce must never equal key_release_nonce.
    score_nonce = eval_plan.get("score_nonce")
    if isinstance(score_nonce, str) and score_nonce and score_nonce == grant_nonce:
        return REFUSE_DOMAIN_CONFUSION, None

    try:
        expected = recompute_key_release_report_data_hex(
            eval_run_id=grant_eval,
            key_release_nonce=grant_nonce,
            ra_tls_spki_digest=grant_spki,
        )
    except (ew.EvalWireError, ValueError, TypeError, KeyError):
        return REFUSE_KEY_RELEASE_MISMATCH, None

    reported = grant.get("report_data_hex")
    if reported is not None:
        if not isinstance(reported, str) or reported.lower() != expected:
            return REFUSE_TAMPERED, None

    # Optional agent_hash binding if grant carries it (cross-check).
    grant_agent = grant.get("agent_hash")
    plan_agent = eval_plan.get("agent_hash")
    if grant_agent is not None and plan_agent is not None and str(grant_agent) != str(plan_agent):
        return REFUSE_KEY_RELEASE_MISMATCH, None

    # Optional package_tree_sha binding if grant carries it (AGATE proof chain).
    grant_tree = grant.get("package_tree_sha")
    if grant_tree is not None:
        if str(grant_tree).strip().lower() != plan_tree.strip().lower():
            return REFUSE_KEY_RELEASE_MISMATCH, None

    return None, expected


def verify_score_domain_binding(
    *,
    score_binding: Mapping[str, Any] | None,
    reported_report_data_hex: str | None,
    eval_plan: Mapping[str, Any],
    scores_digest: str | None = None,
) -> tuple[str | None, str | None]:
    """Re-verify score-domain report_data binding (domain separation enforced)."""

    if score_binding is None:
        return REFUSE_INCOMPLETE_CHAIN, None

    domain = score_binding.get("domain")
    if domain is not None and str(domain) != SCORE_DOMAIN:
        if str(domain) in {KEY_RELEASE_DOMAIN, REVIEW_DOMAIN}:
            return REFUSE_DOMAIN_CONFUSION, None
        return REFUSE_SCORE_DOMAIN, None

    # Must equal plan identity.
    for field in ("eval_run_id", "score_nonce", "agent_hash"):
        plan_val = eval_plan.get(field)
        bind_val = score_binding.get(field)
        if plan_val is not None and bind_val is not None and str(plan_val) != str(bind_val):
            return REFUSE_TAMPERED, None

    # Domain separation: score nonce ≠ key-release nonce.
    if str(score_binding.get("score_nonce") or "") == str(eval_plan.get("key_release_nonce") or ""):
        return REFUSE_DOMAIN_CONFUSION, None

    if scores_digest is not None:
        if str(score_binding.get("scores_digest") or "") != scores_digest:
            return REFUSE_TAMPERED, None

    try:
        expected_hex = ew.score_report_data_hex(score_binding)
    except (ew.EvalWireError, ValueError, TypeError, KeyError):
        return REFUSE_SCORE_DOMAIN, None

    if reported_report_data_hex is not None:
        if not isinstance(reported_report_data_hex, str):
            return REFUSE_TAMPERED, None
        if reported_report_data_hex.lower() != expected_hex:
            return REFUSE_TAMPERED, None

    return None, expected_hex


def admit_production_score_from_chain(
    *,
    dual_flags_on: bool,
    # Review leg
    review_envelope: Mapping[str, Any] | str | bytes | None = None,
    review_core: Mapping[str, Any] | None = None,
    review_report_data_hex: str | None = None,
    review_domain: str | None = None,
    cached_review_allow: bool = False,
    # Key-release leg (RA-TLS)
    key_release_grant: Mapping[str, Any] | None = None,
    key_granted_flag: bool = False,
    # Score leg
    eval_plan: Mapping[str, Any] | None = None,
    score_binding: Mapping[str, Any] | None = None,
    score_report_data_hex: str | None = None,
    scores_digest: str | None = None,
    # Nonce / replay
    score_nonce_state: str = "outstanding",  # outstanding | consumed | expired | unknown
    # Offline consolation / sticky / partial
    offline_ast_pass: bool = False,
    prior_reverify_failed: bool = False,
    proposed_partial_score: float | None = None,
    # Master / orchestrator markers must never alone authorize
    master_status_green: bool = False,
    cached_score_ok: bool = False,
    # VAL-ACAT-016/050–054: eval-agent LLM materials (optional; tools-only default)
    agent_llm_mode: str | None = None,
    agent_or_materials: Mapping[str, Any] | None = None,
    agent_llm_runtime_kind: str | None = None,
    agent_llm_measurement: Mapping[str, str] | None = None,
    agent_llm_allowlist: Sequence[Mapping[str, str]] | None = None,
    claims_agent_model_call: bool = False,
    agent_env: Mapping[str, str] | None = None,
    agent_gateway_url: str | None = None,
    agent_gateway_token_present: bool = False,
    agent_used_base_llm_v1: bool = False,
) -> ScoreChainAdmission:
    """Admit a production score only when the full chain re-verifies.

    Any single failing conjunct refuses the **whole** unit with
    ``score=None`` and ``partial_score=False`` (VAL-ACAT-031/040).

    When the unit claims eval-agent model usage (or supplies OR materials /
    measured mode), re-runs :func:`admit_eval_agent_llm_for_score` so Base
    gateway residue and unmeasured OpenRouter cannot emit production scores
    (VAL-ACAT-016/050–054).
    """

    checked: list[str] = []

    # Sticky refuse: no silent downgrade after a prior re-verify failure.
    if prior_reverify_failed:
        return _refuse(REFUSE_STICKY, reverify_exercised=True, sticky=True)

    # Flag-off / legacy paths cannot emit **production** scores (VAL-ACAT-018/054).
    if not dual_flags_on:
        # Even with perfect metrics/AST, not production emission ingredients.
        return _refuse(
            REFUSE_FLAGS_OFF,
            reverify_exercised=False,
            domains_checked=tuple(checked),
        )

    # Eval-agent LLM gate (VAL-ACAT-016/050–054). Default tools-only when no
    # claim / materials (existing score paths remain valid).
    llm_gate_needed = bool(
        claims_agent_model_call
        or agent_or_materials is not None
        or (agent_llm_mode is not None and agent_llm_mode != MODE_TOOLS_ONLY)
        or agent_gateway_url
        or agent_gateway_token_present
        or agent_used_base_llm_v1
        or (agent_env is not None and any(str(k).upper().startswith("BASE_") for k in agent_env))
    )
    if llm_gate_needed or agent_llm_mode is not None:
        agent_decision = admit_eval_agent_llm_for_score(
            mode=agent_llm_mode if agent_llm_mode is not None else MODE_TOOLS_ONLY,
            dual_flags_on=dual_flags_on,
            runtime_kind=agent_llm_runtime_kind,
            measurement=agent_llm_measurement,
            allowlist=agent_llm_allowlist,
            claims_model_call=claims_agent_model_call,
            agent_or_materials=agent_or_materials,
            agent_env=agent_env,
            gateway_url=agent_gateway_url,
            gateway_token_present=agent_gateway_token_present,
            used_base_llm_v1=agent_used_base_llm_v1,
        )
        if not agent_decision.admitted:
            code = agent_decision.reason_code
            if code == AGENT_LLM_FLAGS_OFF:
                code = REFUSE_FLAGS_OFF
            # Preserve agent_llm_* / base_gateway_forbidden refuse codes for
            # Base gateway and unmeasured OR paths (VAL-ACAT-053/050).
            return _refuse(
                code,
                reverify_exercised=True,
                domains_checked=tuple(checked),
            )

    if eval_plan is None:
        return _refuse(REFUSE_INCOMPLETE_CHAIN, reverify_exercised=False)

    # Master green / cached ok alone is never enough (VAL-ACAT-012).
    # Fall through; missing materials will refuse complicated by cache claim.
    _ = master_status_green
    _ = cached_score_ok

    # AGATE: dual-flag production score requires plan package_tree_sha proof.
    plan_tree_sha: str | None = None
    raw_tree = eval_plan.get("package_tree_sha")
    if isinstance(raw_tree, str) and raw_tree.strip():
        plan_tree_sha = raw_tree.strip()
    if dual_flags_on and (
        plan_tree_sha is None
        or len(plan_tree_sha) != 64
        or any(c not in "0123456789abcdef" for c in plan_tree_sha.lower())
    ):
        return _refuse(
            REFUSE_PACKAGE_TREE_MISSING,
            reverify_exercised=False,
            domains_checked=tuple(checked),
        )

    # ----- 1) Review domain re-verify (times + 24h + OR + allow) -----
    # AGATE VAL-AGATE-008/009/011: dual-flag production score also requires
    # measured package residual allow + package_tree_sha match on the plan.
    checked.append(REVIEW_DOMAIN)
    review_decision = admit_eval_cvm_fresh_review(
        envelope=review_envelope,
        review_core=review_core,
        report_data_hex=review_report_data_hex,
        domain=review_domain,
        cached_outcome_status="verified_allow" if cached_review_allow else None,
        cached_phase="review_allowed" if cached_review_allow else None,
        dual_flags_on=dual_flags_on,
        require_package_residual=bool(dual_flags_on),
        expected_package_tree_sha=plan_tree_sha if dual_flags_on else None,
    )
    if not review_decision.may_launch:
        code = review_decision.reason_code
        # Preserve AGATE package residual refuse codes (no free score attestation).
        if code in _PACKAGE_PROOF_REFUSE_CODES:
            return _refuse(
                code,
                reverify_exercised=True,
                domains_checked=tuple(checked),
                review_digest=review_decision.review_digest,
            )
        # AST-only consolation path: AST green + missing review materials.
        if offline_ast_pass and code in {
            REFUSE_MISSING_REVIEW,
            "review_attestation_missing",
            "eval_cvm_refused_cached_allow_only",
        }:
            return _refuse(
                REFUSE_AST_ONLY,
                reverify_exercised=True,
                domains_checked=tuple(checked),
            )
        if code == REVIEW_REFUSE_STALE or code == "attestation_stale_over_24h":
            return _refuse(
                REFUSE_REVIEW_STALE,
                reverify_exercised=True,
                domains_checked=tuple(checked),
                review_digest=review_decision.review_digest,
            )
        if code in {
            "eval_cvm_refused_cached_allow_only",
            "review_attestation_missing",
        }:
            if offline_ast_pass:
                return _refuse(
                    REFUSE_AST_ONLY,
                    reverify_exercised=True,
                    domains_checked=tuple(checked),
                )
            return _refuse(
                REFUSE_INCOMPLETE_CHAIN,
                reverify_exercised=True,
                domains_checked=tuple(checked),
            )
        return _refuse(
            REFUSE_REVIEW if code not in {REFUSE_TAMPERED, REFUSE_DOMAIN_CONFUSION} else code,
            reverify_exercised=True,
            domains_checked=tuple(checked),
            review_digest=review_decision.review_digest,
        )

    # Authorizing review digest must match plan when both present.
    plan_review_digest = eval_plan.get("authorizing_review_digest")
    if (
        isinstance(plan_review_digest, str)
        and plan_review_digest
        and review_decision.review_digest
        and plan_review_digest != review_decision.review_digest
    ):
        return _refuse(
            REFUSE_TAMPERED,
            reverify_exercised=True,
            domains_checked=tuple(checked),
            review_digest=review_decision.review_digest,
        )

    # ----- 2) Key-release RA-TLS re-check at admission (VAL-ACAT-036/037) -----
    checked.append(KEY_RELEASE_DOMAIN)
    kr_err, kr_hex = verify_key_release_grant(
        grant=key_release_grant,
        eval_plan=eval_plan,
        key_granted_flag=key_granted_flag,
    )
    if kr_err is not None:
        if offline_ast_pass and kr_err in {
            REFUSE_MISSING_KEY_RELEASE,
            REFUSE_INCOMPLETE_CHAIN,
        }:
            return _refuse(
                REFUSE_AST_ONLY,
                reverify_exercised=True,
                domains_checked=tuple(checked),
                review_digest=review_decision.review_digest,
            )
        return _refuse(
            kr_err,
            reverify_exercised=True,
            domains_checked=tuple(checked),
            review_digest=review_decision.review_digest,
        )

    # ----- 3) Score-domain binding re-verify -----
    checked.append(SCORE_DOMAIN)
    score_err, score_rd = verify_score_domain_binding(
        score_binding=score_binding,
        reported_report_data_hex=score_report_data_hex,
        eval_plan=eval_plan,
        scores_digest=scores_digest,
    )
    if score_err is not None:
        if offline_ast_pass and score_err == REFUSE_INCOMPLETE_CHAIN:
            return _refuse(
                REFUSE_AST_ONLY,
                reverify_exercised=True,
                domains_checked=tuple(checked),
                review_digest=review_decision.review_digest,
            )
        return _refuse(
            score_err,
            reverify_exercised=True,
            domains_checked=tuple(checked),
            review_digest=review_decision.review_digest,
        )

    # ----- 4) Score nonce freshness / single-use (VAL-ACAT-017/033) -----
    state = (score_nonce_state or "unknown").strip().lower()
    if state == "consumed":
        return _refuse(
            REFUSE_NONCE_REPLAY,
            reverify_exercised=True,
            domains_checked=tuple(checked),
            review_digest=review_decision.review_digest,
        )
    if state == "expired":
        return _refuse(
            REFUSE_NONCE_STALE,
            reverify_exercised=True,
            domains_checked=tuple(checked),
            review_digest=review_decision.review_digest,
        )
    if state not in {"outstanding", "fresh"}:
        return _refuse(
            REFUSE_INCOMPLETE_CHAIN,
            reverify_exercised=True,
            domains_checked=tuple(checked),
            review_digest=review_decision.review_digest,
        )

    # Never emit fractional / partial consolation scores.
    if proposed_partial_score is not None:
        # Callers that try to carry a partial lead-in with the chain must still
        # only get full admission — we ignore partial and accept only complete
        # chain. Document that partial_score remains False.
        pass

    return ScoreChainAdmission(
        admitted=True,
        reason_code="score_chain_verified",
        reverify_exercised=True,
        score=None,  # score value is not this gate's job; emission binary only
        production_emit=True,
        partial_score=False,
        domains_checked=tuple(checked),
        review_digest=review_decision.review_digest,
        key_release_binding_hex=kr_hex,
        score_report_data_hex=score_rd,
        sticky=False,
    )


def require_production_score_from_chain(**kwargs: Any) -> ScoreChainAdmission:
    """Fail closed on refuse; return admission when production emit may proceed."""

    decision = admit_production_score_from_chain(**kwargs)
    if not decision.admitted:
        raise ScoreChainAdmissionError(decision.reason_code)
    return decision


def admit_production_score_for_eval_result(
    *,
    settings_dual_flags_on: bool,
    eval_plan: Mapping[str, Any],
    review_envelope: Mapping[str, Any] | str | bytes | None,
    key_release_grant: Mapping[str, Any] | None,
    key_granted_flag: bool,
    score_binding: Mapping[str, Any] | None,
    score_report_data_hex: str | None,
    scores_digest: str | None,
    score_nonce_outstanding: bool,
    score_nonce_expired: bool = False,
    offline_ast_pass: bool = False,
    prior_reverify_failed: bool = False,
    master_status_green: bool = False,
    # VAL-ACAT-016/050–054: forward eval-agent LLM materials to from_chain.
    # Live dual-flag ON emission re-checks measured OpenRouter digests when a
    # model call is claimed. Tools-only default applies only when no claim and
    # no materials/gateway residue are present.
    agent_llm_mode: str | None = None,
    agent_or_materials: Mapping[str, Any] | None = None,
    agent_llm_runtime_kind: str | None = None,
    agent_llm_measurement: Mapping[str, str] | None = None,
    agent_llm_allowlist: Sequence[Mapping[str, str]] | None = None,
    claims_agent_model_call: bool = False,
    agent_env: Mapping[str, str] | None = None,
    agent_gateway_url: str | None = None,
    agent_gateway_token_present: bool = False,
    agent_used_base_llm_v1: bool = False,
) -> ScoreChainAdmission:
    """Convenience wrapper for the direct-result / validator admission path.

    Forwards agent OpenRouter materials and Base-gateway residue into
    :func:`admit_production_score_from_chain` so production emission re-checks
    digests / measurement / residual gateway flags (not helper-only).
    """

    if score_nonce_expired:
        nonce_state = "expired"
    elif score_nonce_outstanding:
        nonce_state = "outstanding"
    else:
        # Not outstanding → treat as consumed/replay for fail-closed production.
        nonce_state = "consumed"

    return admit_production_score_from_chain(
        dual_flags_on=settings_dual_flags_on,
        review_envelope=review_envelope,
        key_release_grant=key_release_grant,
        key_granted_flag=key_granted_flag,
        eval_plan=eval_plan,
        score_binding=score_binding,
        score_report_data_hex=score_report_data_hex,
        scores_digest=scores_digest,
        score_nonce_state=nonce_state,
        offline_ast_pass=offline_ast_pass,
        prior_reverify_failed=prior_reverify_failed,
        master_status_green=master_status_green,
        agent_llm_mode=agent_llm_mode,
        agent_or_materials=agent_or_materials,
        agent_llm_runtime_kind=agent_llm_runtime_kind,
        agent_llm_measurement=agent_llm_measurement,
        agent_llm_allowlist=agent_llm_allowlist,
        claims_agent_model_call=claims_agent_model_call,
        agent_env=agent_env,
        agent_gateway_url=agent_gateway_url,
        agent_gateway_token_present=agent_gateway_token_present,
        agent_used_base_llm_v1=agent_used_base_llm_v1,
    )


def build_score_binding_from_plan_and_digest(
    *,
    eval_plan: Mapping[str, Any],
    scores_digest: str,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Helper for tests / live paths to construct a closed score binding."""

    measurement = eval_plan["eval_app"]["measurement"]
    compose_hash = eval_plan["eval_app"]["compose_hash"]
    canonical = {
        "mrtd": measurement["mrtd"],
        "rtmr0": measurement["rtmr0"],
        "rtmr1": measurement["rtmr1"],
        "rtmr2": measurement["rtmr2"],
        "compose_hash": compose_hash,
        "os_image_hash": measurement["os_image_hash"],
    }
    tasks = task_ids or [t["task_id"] for t in eval_plan["selected_tasks"]]
    return ew.build_score_binding(
        canonical_measurement=canonical,
        agent_hash=str(eval_plan["agent_hash"]),
        eval_run_id=str(eval_plan["eval_run_id"]),
        score_nonce=str(eval_plan["score_nonce"]),
        scores_digest=scores_digest,
        task_ids=tasks,
    )


# Re-export review outcome admit for call-site completeness (library wire note).
__all__ = [
    "KEY_RELEASE_DOMAIN",
    "REFUSE_AST_ONLY",
    "REFUSE_DOMAIN_CONFUSION",
    "REFUSE_FLAGS_OFF",
    "REFUSE_INCOMPLETE_CHAIN",
    "REFUSE_KEY_RELEASE_DOMAIN",
    "REFUSE_KEY_RELEASE_MISMATCH",
    "REFUSE_MISSING_KEY_RELEASE",
    "REFUSE_MISSING_REVIEW",
    "REFUSE_NONCE_REPLAY",
    "REFUSE_NONCE_STALE",
    "REFUSE_PACKAGE_PROOF",
    "REFUSE_PACKAGE_TREE_MISSING",
    "REFUSE_PARTIAL_FORBIDDEN",
    "REFUSE_REVIEW",
    "REFUSE_REVIEW_STALE",
    "REFUSE_SCORE_DOMAIN",
    "REFUSE_STICKY",
    "REFUSE_TAMPERED",
    "REVIEW_DOMAIN",
    "SCORE_DOMAIN",
    "ScoreChainAdmission",
    "ScoreChainAdmissionError",
    "admit_eval_agent_llm_for_score",
    "admit_production_from_bound_outcome",
    "admit_production_score_for_eval_result",
    "admit_production_score_from_chain",
    "build_score_binding_from_plan_and_digest",
    "clear_key_release_grant_for_score",
    "load_durable_key_release_grant",
    "lookup_key_release_grant_for_score",
    "recompute_key_release_report_data_hex",
    "register_key_release_grant_for_score",
    "require_production_score_from_chain",
    "verify_key_release_grant",
    "verify_score_domain_binding",
    "ReviewOrOutcomeError",
]
