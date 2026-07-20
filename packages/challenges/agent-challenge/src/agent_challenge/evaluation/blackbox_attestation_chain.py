"""Black-box re-verifiable end-to-end attestation chain (VAL-ACAT-020, VAL-ACAT-X001..X012).

Product freeze (library/ac-attestation.md):

Miner script+ZIP → measured review under ``.rules`` with OpenRouter planned/observed
digests → attested times ≤24h → Eval CVM fresh re-verify → RA-TLS key-release →
optional measured eval-agent OR digests → score domain → raw weights into Base
aggregate path.

This module **composes** existing fail-closed gates into one chronological package
that:

1. Maps every cross-area hop with stable hop ids and codes.
2. Produces an independent re-verify surface from package artifacts alone.
3. Fails closed on mutilation (report_data, digests, times, domain mixups).
4. Asserts zero Base gateway critical-path dependency.
5. Routes weights via challenge raw-weight / master aggregate only (never
   ``set_weights`` / burn recipe).

Lab/disposable or fixture chain materials are acceptable when re-verifiable.
Production PATH honesty stays dual-label: REAL-PROVIDER TEE remains BLOCKED
without hard provider gates. Does **not** restore Base LLM gateway.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.eval_agent_llm import (
    MEASURED_EVAL_CVM_KIND,
    MODE_MEASURED_OPENROUTER,
    MODE_TOOLS_ONLY,
    REFUSE_BASE_GATEWAY,
    admit_eval_agent_llm_for_score,
    bind_eval_agent_or_digests_into_score_materials,
    build_eval_agent_observed_transport,
    build_eval_agent_planned_request,
)
from agent_challenge.evaluation.fresh_review_gate import (
    admit_eval_cvm_fresh_review,
)
from agent_challenge.evaluation.score_chain_gate import (
    KEY_RELEASE_DOMAIN,
    REVIEW_DOMAIN,
    SCORE_DOMAIN,
    admit_production_score_from_chain,
    build_score_binding_from_plan_and_digest,
    recompute_key_release_report_data_hex,
)
from agent_challenge.review.attested_times import FRESHNESS_WINDOW_MS
from agent_challenge.review.harness_entry import (
    PRODUCT_HARNESS_KIND,
    ProductHarnessAdmissionError,
    admit_product_review_entry,
)
from agent_challenge.review.or_outcome_bind import (
    REVIEW_REPORT_DOMAIN,
    build_decision,
    build_observed_openrouter_transport,
    build_openrouter_observation,
    build_planned_openrouter_request,
    build_policy_observation,
    build_review_core_minimal,
    planned_request_sha256,
    review_digest,
    review_report_data_hex,
    sha256_hex,
)

# ---------------------------------------------------------------------------
# Stable hop ids (chronological index — VAL-ACAT-020 / X001..X012)
# ---------------------------------------------------------------------------

HOP_MINER_SUBMIT = "hop.miner_submit_no_gateway"  # X001
HOP_RULES_LOAD = "hop.review_rules_measured"  # X002
HOP_OR_PLANNED = "hop.review_or_planned"  # X003
HOP_OR_OBSERVED = "hop.review_or_observed"  # X004
HOP_FRESHNESS_24H = "hop.freshness_24h"  # X005
HOP_EVAL_CONJUNCTION = "hop.eval_launch_conjunction"  # X006
HOP_KEY_RELEASE = "hop.eval_ra_tls_key_release"  # X007
HOP_EVAL_AGENT_OR = "hop.eval_agent_or_measured"  # X008
HOP_SCORE_DOMAIN = "hop.score_domain_attestation"  # X009
HOP_RAW_WEIGHTS = "hop.raw_weights_aggregate"  # X010
HOP_GATEWAY_FREE = "hop.zero_base_gateway"  # X011
HOP_HONESTY = "hop.fairness_honesty"  # X012

HOP_ORDER: tuple[str, ...] = (
    HOP_MINER_SUBMIT,
    HOP_RULES_LOAD,
    HOP_OR_PLANNED,
    HOP_OR_OBSERVED,
    HOP_FRESHNESS_24H,
    HOP_EVAL_CONJUNCTION,
    HOP_KEY_RELEASE,
    HOP_EVAL_AGENT_OR,
    HOP_SCORE_DOMAIN,
    HOP_RAW_WEIGHTS,
    HOP_GATEWAY_FREE,
    HOP_HONESTY,
)

# Assertion map for evidence index
HOP_ASSERTIONS: dict[str, str] = {
    HOP_MINER_SUBMIT: "VAL-ACAT-X001",
    HOP_RULES_LOAD: "VAL-ACAT-X002",
    HOP_OR_PLANNED: "VAL-ACAT-X003",
    HOP_OR_OBSERVED: "VAL-ACAT-X004",
    HOP_FRESHNESS_24H: "VAL-ACAT-X005",
    HOP_EVAL_CONJUNCTION: "VAL-ACAT-X006",
    HOP_KEY_RELEASE: "VAL-ACAT-X007",
    HOP_EVAL_AGENT_OR: "VAL-ACAT-X008",
    HOP_SCORE_DOMAIN: "VAL-ACAT-X009",
    HOP_RAW_WEIGHTS: "VAL-ACAT-X010",
    HOP_GATEWAY_FREE: "VAL-ACAT-X011",
    HOP_HONESTY: "VAL-ACAT-X012",
}

# Refuse codes unique to this composition surface
REFUSE_CHAIN_INCOMPLETE = "e2e_chain_incomplete"
REFUSE_CHAIN_MUTILATED = "e2e_chain_mutilated_reverify_failed"
REFUSE_GATEWAY_RESIDUAL = "e2e_base_gateway_residual"
REFUSE_SET_WEIGHTS = "e2e_set_weights_forbidden"
REFUSE_BURN_TRACKED = "e2e_burn_weights_tracked_forbidden"
REFUSE_HOP_ORDER = "e2e_hop_order_invalid"
REFUSE_PACKAGE_TAMPER = "e2e_package_tamper"

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
BASE_GATEWAY_URL_MARKERS = ("/llm/v1", "BASE_LLM_GATEWAY", "BASE_GATEWAY_TOKEN")
FORBIDDEN_SIDE_EFFECTS = frozenset({"set_weights", "burn_weights_24h", "WeightSetter"})

# Fixture-stable seeds (deterministic package digests across same inputs)
T0_DEFAULT = 1_700_000_000_000
MS_23H = FRESHNESS_WINDOW_MS - 60_000
PRODUCT_ENTRY = "python -m agent_challenge.selfdeploy"
SAMPLE_ZIP = b"PK\x03\x04agent-challenge-e2e-fixture-agent-v1"
ENTRY_BYTES = b'#!/usr/bin/env python3\n"""selfdeploy entry for e2e chain"""\n'
SAMPLE_RULES: dict[str, bytes] = {
    ".rules/acceptance.md": b"# acceptance\nMeasured review path only.\n",
    ".rules/anti-cheat.md": b"# anti-cheat\nNo unmeasured shortcuts.\n",
    ".rules/hardcoding.md": b"# hardcoding\nNo Base gateway hardcoding.\n",
    ".rules/security.md": b"# security\nFail closed on missing attestation.\n",
}
ROUTING_DEFAULT = sha256_hex(b'{"order":["e2e-blackbox"]}')
BODY_DEFAULT = b'{"model":"x-ai/grok-4.5","messages":[{"role":"user","content":"review"}]}'
BODY_SHA_DEFAULT = sha256_hex(BODY_DEFAULT)
RESP_DEFAULT = (
    b'{"id":"gen-e2e","model":"x-ai/grok-4.5","choices":[{"message":{"content":"allow"}}]}'
)
RESP_SHA_DEFAULT = sha256_hex(RESP_DEFAULT)
META_DEFAULT = sha256_hex(b"meta-e2e-blackbox")
SPKI_DEFAULT = "aa" * 32
REGS_DEFAULT = {
    "mrtd": "11" * 48,
    "rtmr0": "22" * 48,
    "rtmr1": "33" * 48,
    "rtmr2": "44" * 48,
}
COMPOSE_HASH_DEFAULT = "ab" * 32
AGENT_HASH_DEFAULT = "55" * 32
MINER_HOTKEY_DEFAULT = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HopResult:
    """One chronological hop outcome in the e2e evidence index."""

    hop_id: str
    assertion_id: str
    status: str  # pass | refuse | skip
    reason_code: str
    detail: str = ""
    digests: Mapping[str, str] = field(default_factory=dict)
    times_ms: Mapping[str, int] = field(default_factory=dict)
    gateway_dependency: bool = False
    set_weights_invoked: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "hop_id": self.hop_id,
            "assertion_id": self.assertion_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "digests": dict(self.digests),
            "times_ms": dict(self.times_ms),
            "gateway_dependency": self.gateway_dependency,
            "set_weights_invoked": self.set_weights_invoked,
        }


@dataclass(frozen=True)
class ChainAdmission:
    """Overall E2E chain decision."""

    admitted: bool
    reason_code: str
    hops: tuple[HopResult, ...]
    package: Mapping[str, Any]
    package_digest: str
    production_emit: bool
    raw_weights: Mapping[str, float]
    gateway_free: bool
    reverify_ok: bool
    tee_labels: Mapping[str, str]
    side_effects: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reason_code": self.reason_code,
            "hops": [h.as_dict() for h in self.hops],
            "package_digest": self.package_digest,
            "production_emit": self.production_emit,
            "raw_weights": dict(self.raw_weights),
            "gateway_free": self.gateway_free,
            "reverify_ok": self.reverify_ok,
            "tee_labels": dict(self.tee_labels),
            "side_effects": dict(self.side_effects),
            "chronological_hop_ids": [h.hop_id for h in self.hops],
            "assertions_mapped": {h.assertion_id: h.status for h in self.hops if h.assertion_id},
        }


class BlackboxChainError(PermissionError):
    """Fail-closed e2e chain refuse."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def canonical_json_v1(obj: Mapping[str, Any] | Sequence[Any] | Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def package_digest(package: Mapping[str, Any]) -> str:
    """Digest of re-verifiable materials only (exclude derived hop statuses)."""

    materials = {
        "schema_version": package.get("schema_version"),
        "identity": package.get("identity"),
        "review_envelope": package.get("review_envelope"),
        "key_release_grant": package.get("key_release_grant"),
        "eval_plan": package.get("eval_plan"),
        "score_binding": package.get("score_binding"),
        "score_report_data_hex": package.get("score_report_data_hex"),
        "scores_digest": package.get("scores_digest"),
        "agent_or_materials": package.get("agent_or_materials"),
        "raw_weights_payload": package.get("raw_weights_payload"),
        "gateway_inventory": package.get("gateway_inventory"),
        "honesty": package.get("honesty"),
        "issued_at_ms": package.get("issued_at_ms"),
        "received_at_ms": package.get("received_at_ms"),
        "dual_flags_on": package.get("dual_flags_on"),
        "agent_llm_mode": package.get("agent_llm_mode"),
    }
    return hashlib.sha256(canonical_json_v1(materials)).hexdigest()


def scan_gateway_residuals(
    *,
    env: Mapping[str, str] | None = None,
    urls: Sequence[str] | None = None,
    master_routes: Sequence[str] | None = None,
    secrets: Mapping[str, str] | None = None,
) -> list[str]:
    """Return residual Base gateway markers (empty => gateway-free)."""

    hits: list[str] = []
    for name in env or {}:
        upper = str(name).upper()
        if upper in BASE_GATEWAY_ENV_NAMES or "LLM_GATEWAY" in upper:
            hits.append(f"env:{upper}")
    for url in urls or ():
        u = str(url)
        for marker in BASE_GATEWAY_URL_MARKERS:
            if marker.lower() in u.lower():
                hits.append(f"url:{marker}")
    for route in master_routes or ():
        r = str(route)
        if "/llm/v1" in r:
            hits.append(f"route:{r}")
    for name in secrets or {}:
        upper = str(name).upper()
        if upper in BASE_GATEWAY_ENV_NAMES or "OPENROUTER" in upper and "MASTER" in upper:
            hits.append(f"secret:{upper}")
    return hits


def build_raw_weights_payload(
    *,
    hotkey_weights: Mapping[str, float],
    challenge_slug: str = "agent-challenge",
    epoch: int = 1,
    revision: int = 1,
    computed_at: datetime | None = None,
    expires_at: datetime | None = None,
    nonce: str = "e2e-raw-weight-nonce-1",
) -> dict[str, Any]:
    """Closed raw-weight payload for master aggregate path (no set_weights).

    Uses the same canonical digest rule as Base ``RawWeightPushRequest`` so the
    package is independently re-checkable against Base aggregation ingress
    without invoking ops ``set_weights``.
    """

    if not hotkey_weights:
        raise BlackboxChainError(REFUSE_CHAIN_INCOMPLETE, "empty hotkey weight map")
    for hotkey, weight in hotkey_weights.items():
        if not isinstance(weight, float) or weight < 0 or weight != weight:
            raise BlackboxChainError(REFUSE_CHAIN_INCOMPLETE, f"invalid weight for {hotkey!r}")
    now = computed_at or datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
    exp = expires_at or (now + timedelta(hours=1))
    if exp <= now:
        raise BlackboxChainError(REFUSE_CHAIN_INCOMPLETE, "expires_at must be after computed_at")

    def _iso(dt: datetime) -> str:
        return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")

    body: dict[str, Any] = {
        "protocol_version": "1.0",
        "challenge_slug": challenge_slug,
        "epoch": int(epoch),
        "revision": int(revision),
        "computed_at": _iso(now),
        "expires_at": _iso(exp),
        "nonce": nonce,
        "weights": {k: float(v) for k, v in hotkey_weights.items()},
    }
    digest = hashlib.sha256(canonical_json_v1(body)).hexdigest()
    body["payload_digest"] = digest
    body["path"] = f"/internal/v1/challenges/{challenge_slug}/raw-weights"
    body["master_set_weights"] = False
    body["ops_set_weights"] = False
    body["burn_weights_invoked"] = False
    return body


def _times(*, issued: int, received: int) -> dict[str, int]:
    base = min(issued, received)
    return {
        "issued_at_ms": issued,
        "started_at_ms": base,
        "model_call_marked_at_ms": base + 1,
        "request_started_at_ms": base + 2,
        "request_finished_at_ms": base + 3,
        "verifier_finished_at_ms": base + 4,
        "report_finished_at_ms": base + 5,
        "expires_at_ms": max(issued, received) + 3_600_000,
        "submission_received_at_ms": received,
    }


def _rules_observation_from_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rules_version": identity["rules_version"],
        "rules_bundle_sha256": identity["rules_bundle_sha256"],
        "rules_files": list(identity["rules_files"]),
        "rules_file_digests": dict(identity["rules_file_digests"]),
        "rules_policy_text_sha256": identity["rules_policy_text_sha256"],
    }


def build_fixture_review_envelope(
    *,
    identity: Mapping[str, Any],
    issued_at_ms: int,
    received_at_ms: int,
    session_id: str = "rs-e2e-blackbox",
    assignment_id: str = "ra-e2e-blackbox",
    submission_id: str = "sub-e2e-blackbox",
    review_nonce: str = "nonce-e2e-review",
    verdict: str = "allow",
) -> dict[str, Any]:
    """Closed allow review envelope with OR digests + bound times."""

    planned = build_planned_openrouter_request(
        body_sha256=BODY_SHA_DEFAULT,
        body_length=len(BODY_DEFAULT),
        routing_sha256=ROUTING_DEFAULT,
    )
    p_digest = planned_request_sha256(planned)
    observed = build_observed_openrouter_transport(
        planned_request_sha256_=p_digest,
        response_body_sha256=RESP_SHA_DEFAULT,
        response_body_length=len(RESP_DEFAULT),
        metadata_sha256=META_DEFAULT,
    )
    or_obs = build_openrouter_observation(
        planned=planned,
        observed=observed,
        request_body_sha256=BODY_SHA_DEFAULT,
        request_body_length=len(BODY_DEFAULT),
        response_id="gen-e2e",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-e2e",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-e2e",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-e2e",
        routing_sha256=ROUTING_DEFAULT,
    )
    core = build_review_core_minimal(
        session_id=session_id,
        assignment_id=assignment_id,
        submission_id=submission_id,
        review_nonce=review_nonce,
        assignment_digest="aa" * 32,
        rules_observation=_rules_observation_from_identity(identity),
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict=verdict),
        times=_times(issued=issued_at_ms, received=received_at_ms),
    )
    return {
        "schema_version": 1,
        "domain": REVIEW_REPORT_DOMAIN,
        "review_digest": review_digest(core),
        "report_data_hex": review_report_data_hex(core),
        "review_core": core,
        "planned_openrouter": planned,
        "observed_openrouter": observed,
    }


def _os_image_hash() -> str:
    from agent_challenge.keyrelease.quote import os_image_hash_from_registers

    return os_image_hash_from_registers(
        REGS_DEFAULT["mrtd"], REGS_DEFAULT["rtmr1"], REGS_DEFAULT["rtmr2"]
    )


def build_fixture_eval_plan(
    *,
    authorizing_review_digest: str,
    agent_hash: str = AGENT_HASH_DEFAULT,
    eval_run_id: str = "eval-e2e-blackbox-1",
) -> dict[str, Any]:
    os_hash = _os_image_hash()
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    return ew.validate_eval_plan(
        {
            "schema_version": 1,
            "eval_run_id": eval_run_id,
            "submission_id": "submission-e2e-blackbox-1",
            "submission_version": 1,
            "authorizing_review_digest": authorizing_review_digest,
            "agent_hash": agent_hash,
            "selected_tasks": [
                {
                    "task_id": "task-a",
                    "image_ref": "registry.example/task@sha256:" + "77" * 32,
                    "task_config_sha256": "88" * 32,
                }
            ],
            "k": 1,
            "scoring_policy": policy,
            "scoring_policy_digest": ew.scoring_policy_digest(policy),
            "eval_app": {
                "image_ref": "registry.example/eval@sha256:" + "99" * 32,
                "compose_hash": COMPOSE_HASH_DEFAULT,
                "app_identity": "agent-challenge-eval-v1",
                "kms_key_algorithm": "x25519",
                "kms_public_key_hex": "aa" * 32,
                "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("aa" * 32)).hexdigest(),
                "measurement": {
                    **REGS_DEFAULT,
                    "os_image_hash": os_hash,
                    "key_provider": "phala",
                    "vm_shape": "tdx-small",
                },
            },
            "key_release_endpoint": "validator.example:8701",
            "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
            "key_release_nonce": "key-release-e2e-1",
            "score_nonce": "score-nonce-e2e-1",
            "run_token_sha256": "bb" * 32,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )


def build_key_release_grant(plan: Mapping[str, Any], *, spki: str = SPKI_DEFAULT) -> dict[str, Any]:
    rd = recompute_key_release_report_data_hex(
        eval_run_id=str(plan["eval_run_id"]),
        key_release_nonce=str(plan["key_release_nonce"]),
        ra_tls_spki_digest=spki,
    )
    return {
        "domain": KEY_RELEASE_DOMAIN,
        "schema_version": 2,
        "eval_run_id": plan["eval_run_id"],
        "key_release_nonce": plan["key_release_nonce"],
        "ra_tls_spki_digest": spki,
        "report_data_hex": rd,
        "agent_hash": plan.get("agent_hash"),
        "base_gateway_token_minted": False,
    }


def build_scores_digest(plan: Mapping[str, Any]) -> str:
    from agent_challenge.evaluation.plan_scoring import build_score_record_from_eval_plan

    record = build_score_record_from_eval_plan(dict(plan), {"task-a": [1.0]})
    return ew.score_record_digest(record)


def build_eval_agent_materials(
    *,
    body: bytes = b'{"model":"openrouter/auto","messages":[]}',
    model: str = "x-ai/grok-4.5",
) -> dict[str, Any]:
    routing = sha256_hex(b'{"order":["eval-agent-e2e"]}')
    body_sha = sha256_hex(body)
    planned = build_eval_agent_planned_request(
        body_sha256=body_sha,
        body_length=len(body),
        routing_sha256=routing,
        model=model,
    )
    resp = b'{"id":"gen-agent-e2e","choices":[]}'
    observed = build_eval_agent_observed_transport(
        planned=planned,
        response_body_sha256=sha256_hex(resp),
        response_body_length=len(resp),
        metadata_sha256=sha256_hex(b"eval-agent-meta"),
    )
    return bind_eval_agent_or_digests_into_score_materials(
        planned=planned,
        observed=observed,
    )


def build_complete_package(
    *,
    dual_flags_on: bool = True,
    issued_at_ms: int = T0_DEFAULT,
    received_at_ms: int | None = None,
    agent_zip_bytes: bytes = SAMPLE_ZIP,
    entry_script_identity: str = PRODUCT_ENTRY,
    entry_script_bytes: bytes = ENTRY_BYTES,
    rules_files: Mapping[str, bytes] | None = None,
    agent_llm_mode: str = MODE_TOOLS_ONLY,
    include_measured_agent_or: bool = False,
    miner_hotkey: str = MINER_HOTKEY_DEFAULT,
    raw_score: float = 1.0,
    master_env: Mapping[str, str] | None = None,
    master_routes: Sequence[str] | None = None,
    set_weights_count: int = 0,
    burn_invoked: bool = False,
    real_provider_pass_claimed: bool = False,
    verdict: str = "allow",
    skip_review: bool = False,
    skip_key_release: bool = False,
    skip_score: bool = False,
) -> dict[str, Any]:
    """Build a full re-verifiable artifact package for the e2e chain."""

    received = T0_DEFAULT + MS_23H if received_at_ms is None else received_at_ms
    try:
        identity = admit_product_review_entry(
            agent_zip_bytes=agent_zip_bytes,
            entry_script_identity=entry_script_identity,
            entry_script_bytes=entry_script_bytes,
            rules_files=rules_files if rules_files is not None else SAMPLE_RULES,
        )
        identity_dict = identity.as_dict()
        identity_error: str | None = None
    except ProductHarnessAdmissionError as exc:
        identity_dict = {
            "schema_version": 1,
            "harness_kind": PRODUCT_HARNESS_KIND,
            "error": exc.code,
        }
        identity_error = exc.code

    review_envelope: dict[str, Any] | None = None
    if not skip_review and identity_error is None:
        review_envelope = build_fixture_review_envelope(
            identity=identity_dict,
            issued_at_ms=issued_at_ms,
            received_at_ms=received,
            verdict=verdict,
        )

    eval_plan: dict[str, Any] | None = None
    key_release_grant: dict[str, Any] | None = None
    score_binding: dict[str, Any] | None = None
    score_report_data_hex: str | None = None
    scores_digest: str | None = None

    if review_envelope is not None and not skip_score:
        eval_plan = build_fixture_eval_plan(
            authorizing_review_digest=str(review_envelope["review_digest"]),
        )
        if not skip_key_release:
            key_release_grant = build_key_release_grant(eval_plan)
        scores_digest = build_scores_digest(eval_plan)
        score_binding = build_score_binding_from_plan_and_digest(
            eval_plan=eval_plan,
            scores_digest=scores_digest,
        )
        score_report_data_hex = ew.score_report_data_hex(score_binding)

    agent_or_materials: dict[str, Any] | None = None
    llm_mode = agent_llm_mode
    if include_measured_agent_or:
        llm_mode = MODE_MEASURED_OPENROUTER
        agent_or_materials = build_eval_agent_materials()

    raw_payload = build_raw_weights_payload(
        hotkey_weights={miner_hotkey: float(raw_score)},
    )
    if set_weights_count:
        raw_payload = dict(raw_payload)
        raw_payload["master_set_weights"] = True
        raw_payload["ops_set_weights_count"] = set_weights_count
    if burn_invoked:
        raw_payload = dict(raw_payload)
        raw_payload["burn_weights_invoked"] = True

    env = dict(master_env or {})
    residuals = scan_gateway_residuals(
        env=env,
        master_routes=master_routes
        or (
            "/health",
            "/version",
            "/v1/registry",
            "/challenges/agent-challenge/openapi.json",
            "/internal/v1/challenges/agent-challenge/raw-weights",
        ),
        secrets={},
    )

    honesty = {
        "tee_local_fixture": "LOCAL-FIXTURE PASS" if dual_flags_on else "FAIL",
        "tee_real_provider": "REAL-PROVIDER PASS" if real_provider_pass_claimed else "BLOCKED",
        "real_provider_pass_is_possible": False,
        "real_provider_claimed": bool(real_provider_pass_claimed),
        "secrets_redacted": True,
        "secondary_platform_network_502_ok": True,
        "joinbase_authoritative": True,
        "no_false_score_on_refuse": True,
    }

    package: dict[str, Any] = {
        "schema_version": 1,
        "domain": "base-agent-challenge-e2e-blackbox-v1",
        "dual_flags_on": dual_flags_on,
        "issued_at_ms": issued_at_ms,
        "received_at_ms": received,
        "identity": identity_dict,
        "identity_error": identity_error,
        "review_envelope": review_envelope,
        "key_release_grant": key_release_grant,
        "eval_plan": eval_plan,
        "score_binding": score_binding,
        "score_report_data_hex": score_report_data_hex,
        "scores_digest": scores_digest,
        "score_nonce_state": "outstanding",
        "agent_llm_mode": llm_mode,
        "agent_or_materials": agent_or_materials,
        "agent_llm_runtime_kind": MEASURED_EVAL_CVM_KIND if include_measured_agent_or else None,
        "agent_llm_measurement": {
            "compose_hash": COMPOSE_HASH_DEFAULT,
            "os_image_hash": _os_image_hash(),
            "mrtd": REGS_DEFAULT["mrtd"],
        }
        if include_measured_agent_or
        else None,
        "agent_llm_allowlist": [
            {
                "compose_hash": COMPOSE_HASH_DEFAULT,
                "os_image_hash": _os_image_hash(),
                "mrtd": REGS_DEFAULT["mrtd"],
            }
        ]
        if include_measured_agent_or
        else None,
        "claims_agent_model_call": include_measured_agent_or,
        "raw_weights_payload": raw_payload,
        "gateway_inventory": {
            "master_env_keys": sorted(env.keys()),
            "residuals": residuals,
            "master_routes": list(
                master_routes
                or (
                    "/health",
                    "/version",
                    "/v1/registry",
                    "/challenges/agent-challenge/openapi.json",
                    "/internal/v1/challenges/agent-challenge/raw-weights",
                )
            ),
            "openrouter_destination": "https://openrouter.ai:443",
            "base_llm_v1_present": any("/llm/v1" in str(r) for r in (master_routes or ())),
        },
        "side_effects": {
            "set_weights": int(set_weights_count),
            "burn_weights_24h": 1 if burn_invoked else 0,
            "master_set_weights": 1 if set_weights_count else 0,
        },
        "honesty": honesty,
        "miner_hotkey": miner_hotkey,
        "submit_path": "/challenges/agent-challenge/submissions",
        "submit_requires_base_gateway_token": False,
    }
    package["package_digest"] = package_digest(package)
    return package


def _pass_hop(
    hop_id: str,
    *,
    reason_code: str = "ok",
    detail: str = "",
    digests: Mapping[str, str] | None = None,
    times_ms: Mapping[str, int] | None = None,
) -> HopResult:
    return HopResult(
        hop_id=hop_id,
        assertion_id=HOP_ASSERTIONS[hop_id],
        status="pass",
        reason_code=reason_code,
        detail=detail,
        digests=dict(digests or {}),
        times_ms=dict(times_ms or {}),
        gateway_dependency=False,
        set_weights_invoked=False,
    )


def _refuse_hop(
    hop_id: str,
    reason_code: str,
    *,
    detail: str = "",
    gateway_dependency: bool = False,
    set_weights_invoked: bool = False,
) -> HopResult:
    return HopResult(
        hop_id=hop_id,
        assertion_id=HOP_ASSERTIONS[hop_id],
        status="refuse",
        reason_code=reason_code,
        detail=detail,
        gateway_dependency=gateway_dependency,
        set_weights_invoked=set_weights_invoked,
    )


def reverify_package(package: Mapping[str, Any]) -> ChainAdmission:
    """Independently re-verify a previously built package from artifacts alone.

    This is the black-box surface for VAL-ACAT-020: given only the immutable
    package materials + digests, re-run each product gate in chronology and
    refuse on any hop failure or package digest mismatch.
    """

    hops: list[HopResult] = []
    dual = bool(package.get("dual_flags_on"))
    reason_terminal = "e2e_chain_verified"
    admitted = True

    # Digest integrity first
    expected_digest = package.get("package_digest")
    recomputed = package_digest(package)
    if not isinstance(expected_digest, str) or expected_digest != recomputed:
        hop = _refuse_hop(HOP_MINER_SUBMIT, REFUSE_PACKAGE_TAMPER, detail="package_digest mismatch")
        hops.append(hop)
        return ChainAdmission(
            admitted=False,
            reason_code=REFUSE_PACKAGE_TAMPER,
            hops=tuple(hops),
            package=dict(package),
            package_digest=recomputed,
            production_emit=False,
            raw_weights={},
            gateway_free=False,
            reverify_ok=False,
            tee_labels={"REAL-PROVIDER": "BLOCKED"},
            side_effects={"set_weights": 0, "burn_weights_24h": 0},
        )

    # --- X001 miner submit without gateway ---
    identity_err = package.get("identity_error")
    submit_requires_gw = bool(package.get("submit_requires_base_gateway_token"))
    residuals = list((package.get("gateway_inventory") or {}).get("residuals") or [])
    if submit_requires_gw or residuals:
        hops.append(
            _refuse_hop(
                HOP_MINER_SUBMIT,
                REFUSE_GATEWAY_RESIDUAL,
                detail="submit depends on Base gateway residuals",
                gateway_dependency=True,
            )
        )
        admitted = False
        reason_terminal = REFUSE_GATEWAY_RESIDUAL
    elif identity_err:
        hops.append(_refuse_hop(HOP_MINER_SUBMIT, str(identity_err)))
        admitted = False
        reason_terminal = str(identity_err)
    else:
        identity = package.get("identity") or {}
        hops.append(
            _pass_hop(
                HOP_MINER_SUBMIT,
                reason_code="submit_no_gateway",
                digests={
                    "zip_sha256": str(identity.get("zip_sha256", "")),
                    "session_identity_sha256": sha256_hex(
                        canonical_json_v1(identity) if isinstance(identity, Mapping) else b"{}"
                    ),
                },
            )
        )

    # --- X002 rules measured ---
    identity = package.get("identity") or {}
    if admitted and isinstance(identity, Mapping) and identity.get("rules_version"):
        hops.append(
            _pass_hop(
                HOP_RULES_LOAD,
                digests={
                    "rules_version": str(identity["rules_version"]),
                    "rules_bundle_sha256": str(identity.get("rules_bundle_sha256", "")),
                },
            )
        )
    elif admitted:
        hops.append(_refuse_hop(HOP_RULES_LOAD, REFUSE_CHAIN_INCOMPLETE, detail="rules missing"))
        admitted = False
        reason_terminal = REFUSE_CHAIN_INCOMPLETE
    else:
        hops.append(_refuse_hop(HOP_RULES_LOAD, reason_terminal, detail="prior hop failed"))

    # --- X003 / X004 OR planned + observed ---
    envelope = package.get("review_envelope")
    if admitted and isinstance(envelope, Mapping):
        core = envelope.get("review_core")
        if not isinstance(core, Mapping):
            hops.append(_refuse_hop(HOP_OR_PLANNED, REFUSE_CHAIN_INCOMPLETE))
            hops.append(_refuse_hop(HOP_OR_OBSERVED, REFUSE_CHAIN_INCOMPLETE))
            admitted = False
            reason_terminal = REFUSE_CHAIN_INCOMPLETE
        else:
            or_obs = core.get("openrouter_observation") or {}
            planned_sha = str(or_obs.get("planned_request_sha256", ""))
            observed_sha = str(or_obs.get("transport_observation_sha256", ""))
            if not planned_sha or planned_sha in {"00" * 32, "ff" * 32}:
                hops.append(
                    _refuse_hop(
                        HOP_OR_PLANNED,
                        "review_or_planned_digest_missing",
                    )
                )
                admitted = False
                reason_terminal = "review_or_planned_digest_missing"
            else:
                hops.append(
                    _pass_hop(
                        HOP_OR_PLANNED,
                        digests={"planned_request_sha256": planned_sha},
                    )
                )
            if admitted:
                if not observed_sha or observed_sha in {"00" * 32, "ff" * 32}:
                    hops.append(
                        _refuse_hop(
                            HOP_OR_OBSERVED,
                            "review_or_observed_digest_missing",
                        )
                    )
                    admitted = False
                    reason_terminal = "review_or_observed_digest_missing"
                else:
                    hops.append(
                        _pass_hop(
                            HOP_OR_OBSERVED,
                            digests={"transport_observation_sha256": observed_sha},
                        )
                    )
    elif admitted:
        hops.append(_refuse_hop(HOP_OR_PLANNED, REFUSE_CHAIN_INCOMPLETE))
        hops.append(_refuse_hop(HOP_OR_OBSERVED, REFUSE_CHAIN_INCOMPLETE))
        admitted = False
        reason_terminal = REFUSE_CHAIN_INCOMPLETE
    else:
        hops.append(_refuse_hop(HOP_OR_PLANNED, reason_terminal))
        hops.append(_refuse_hop(HOP_OR_OBSERVED, reason_terminal))

    # --- X005 freshness 24h ---
    if admitted and isinstance(envelope, Mapping):
        eval_decision = admit_eval_cvm_fresh_review(envelope=envelope)
        times = {
            "issued_at_ms": int(eval_decision.bound_issued_at_ms or 0),
            "received_at_ms": int(eval_decision.bound_received_at_ms or 0),
        }
        if not eval_decision.may_launch and eval_decision.reason_code in {
            "attestation_stale_over_24h",
            "attestation_time_order_invalid",
            "attestation_times_missing",
            "attestation_times_invalid",
        }:
            hops.append(
                _refuse_hop(
                    HOP_FRESHNESS_24H,
                    eval_decision.reason_code,
                    detail="age window fail-closed",
                )
            )
            admitted = False
            reason_terminal = eval_decision.reason_code
        elif not eval_decision.may_launch:
            # defer conjunct failures to X006; age itself can still be ok
            age_ok = True
            issued = package.get("issued_at_ms")
            received = package.get("received_at_ms")
            if (
                isinstance(issued, int)
                and isinstance(received, int)
                and (received - issued) > FRESHNESS_WINDOW_MS
            ):
                age_ok = False
            if not age_ok:
                hops.append(_refuse_hop(HOP_FRESHNESS_24H, "attestation_stale_over_24h"))
                admitted = False
                reason_terminal = "attestation_stale_over_24h"
            else:
                hops.append(
                    _pass_hop(
                        HOP_FRESHNESS_24H,
                        reason_code="age_within_24h",
                        times_ms=times,
                    )
                )
        else:
            hops.append(
                _pass_hop(
                    HOP_FRESHNESS_24H,
                    reason_code="age_within_24h",
                    times_ms=times,
                )
            )
    elif admitted:
        hops.append(_refuse_hop(HOP_FRESHNESS_24H, REFUSE_CHAIN_INCOMPLETE))
        admitted = False
        reason_terminal = REFUSE_CHAIN_INCOMPLETE
    else:
        hops.append(_refuse_hop(HOP_FRESHNESS_24H, reason_terminal))

    # --- X006 conjunction (rules + OR + age + allow) ---
    if admitted and isinstance(envelope, Mapping):
        eval_decision = admit_eval_cvm_fresh_review(envelope=envelope)
        if not eval_decision.may_launch:
            hops.append(
                _refuse_hop(
                    HOP_EVAL_CONJUNCTION,
                    eval_decision.reason_code,
                    detail="fresh review conjunction failed",
                )
            )
            admitted = False
            reason_terminal = eval_decision.reason_code
        else:
            hops.append(
                _pass_hop(
                    HOP_EVAL_CONJUNCTION,
                    reason_code="eval_launch_eligible",
                    digests={"review_digest": str(eval_decision.review_digest or "")},
                )
            )
    elif admitted:
        hops.append(_refuse_hop(HOP_EVAL_CONJUNCTION, REFUSE_CHAIN_INCOMPLETE))
        admitted = False
        reason_terminal = REFUSE_CHAIN_INCOMPLETE
    else:
        hops.append(_refuse_hop(HOP_EVAL_CONJUNCTION, reason_terminal))

    # --- X007 RA-TLS key-release ---
    grant = package.get("key_release_grant")
    if admitted:
        if not isinstance(grant, Mapping):
            hops.append(_refuse_hop(HOP_KEY_RELEASE, "score_refused_missing_key_release"))
            admitted = False
            reason_terminal = "score_refused_missing_key_release"
        elif grant.get("base_gateway_token_minted"):
            hops.append(
                _refuse_hop(
                    HOP_KEY_RELEASE,
                    REFUSE_GATEWAY_RESIDUAL,
                    gateway_dependency=True,
                    detail="Base gateway token mint on KR path",
                )
            )
            admitted = False
            reason_terminal = REFUSE_GATEWAY_RESIDUAL
        else:
            hops.append(
                _pass_hop(
                    HOP_KEY_RELEASE,
                    digests={
                        "ra_tls_spki_digest": str(grant.get("ra_tls_spki_digest", "")),
                        "report_data_hex": str(grant.get("report_data_hex", ""))[:64],
                    },
                )
            )
    else:
        hops.append(_refuse_hop(HOP_KEY_RELEASE, reason_terminal))

    # --- X008 eval agent OR optional ---
    mode = str(package.get("agent_llm_mode") or MODE_TOOLS_ONLY)
    if admitted:
        agent_decision = admit_eval_agent_llm_for_score(
            mode=mode,
            dual_flags_on=dual,
            runtime_kind=package.get("agent_llm_runtime_kind"),
            measurement=package.get("agent_llm_measurement"),
            allowlist=package.get("agent_llm_allowlist"),
            claims_model_call=bool(package.get("claims_agent_model_call")),
            agent_or_materials=package.get("agent_or_materials"),
            agent_env=None,
            gateway_url=None,
            gateway_token_present=False,
            used_base_llm_v1=False,
        )
        if not agent_decision.admitted:
            hops.append(
                _refuse_hop(
                    HOP_EVAL_AGENT_OR,
                    agent_decision.reason_code,
                    gateway_dependency=agent_decision.reason_code
                    in {REFUSE_BASE_GATEWAY, "base_gateway_forbidden"},
                )
            )
            admitted = False
            reason_terminal = agent_decision.reason_code
        else:
            digests: dict[str, str] = {"mode": mode}
            if agent_decision.planned_request_sha256:
                digests["planned_request_sha256"] = agent_decision.planned_request_sha256
            if agent_decision.transport_observation_sha256:
                digests["transport_observation_sha256"] = (
                    agent_decision.transport_observation_sha256
                )
            hops.append(
                _pass_hop(
                    HOP_EVAL_AGENT_OR,
                    reason_code=agent_decision.reason_code,
                    digests=digests,
                )
            )
    else:
        hops.append(_refuse_hop(HOP_EVAL_AGENT_OR, reason_terminal))

    # --- X009 score domain full chain ---
    if admitted:
        score_decision = admit_production_score_from_chain(
            dual_flags_on=dual,
            review_envelope=package.get("review_envelope"),
            key_release_grant=package.get("key_release_grant"),
            key_granted_flag=package.get("key_release_grant") is not None,
            eval_plan=package.get("eval_plan"),
            score_binding=package.get("score_binding"),
            score_report_data_hex=package.get("score_report_data_hex"),
            scores_digest=package.get("scores_digest"),
            score_nonce_state=str(package.get("score_nonce_state") or "outstanding"),
            agent_llm_mode=mode,
            agent_or_materials=package.get("agent_or_materials"),
            agent_llm_runtime_kind=package.get("agent_llm_runtime_kind"),
            agent_llm_measurement=package.get("agent_llm_measurement"),
            agent_llm_allowlist=package.get("agent_llm_allowlist"),
            claims_agent_model_call=bool(package.get("claims_agent_model_call")),
        )
        if not score_decision.admitted:
            hops.append(
                _refuse_hop(
                    HOP_SCORE_DOMAIN,
                    score_decision.reason_code,
                )
            )
            admitted = False
            reason_terminal = score_decision.reason_code
        else:
            hops.append(
                _pass_hop(
                    HOP_SCORE_DOMAIN,
                    reason_code=score_decision.reason_code,
                    digests={
                        "review_digest": str(score_decision.review_digest or ""),
                        "domains": ",".join(score_decision.domains_checked),
                    },
                )
            )
    else:
        hops.append(_refuse_hop(HOP_SCORE_DOMAIN, reason_terminal))

    # --- X010 raw weights only; no set_weights ---
    side = dict(package.get("side_effects") or {})
    raw_payload = package.get("raw_weights_payload") or {}
    set_w = int(side.get("set_weights") or 0)
    burn = int(side.get("burn_weights_24h") or 0)
    if admitted:
        if set_w > 0 or raw_payload.get("master_set_weights") or raw_payload.get("ops_set_weights"):
            hops.append(
                _refuse_hop(
                    HOP_RAW_WEIGHTS,
                    REFUSE_SET_WEIGHTS,
                    set_weights_invoked=True,
                )
            )
            admitted = False
            reason_terminal = REFUSE_SET_WEIGHTS
        elif burn > 0 or raw_payload.get("burn_weights_invoked"):
            hops.append(_refuse_hop(HOP_RAW_WEIGHTS, REFUSE_BURN_TRACKED))
            admitted = False
            reason_terminal = REFUSE_BURN_TRACKED
        elif not isinstance(raw_payload, Mapping) or not raw_payload.get("weights"):
            hops.append(_refuse_hop(HOP_RAW_WEIGHTS, REFUSE_CHAIN_INCOMPLETE))
            admitted = False
            reason_terminal = REFUSE_CHAIN_INCOMPLETE
        else:
            # Re-check payload_digest
            body = {
                k: v
                for k, v in raw_payload.items()
                if k
                not in {
                    "payload_digest",
                    "path",
                    "master_set_weights",
                    "ops_set_weights",
                    "burn_weights_invoked",
                    "ops_set_weights_count",
                }
            }
            recomputed_pw = hashlib.sha256(canonical_json_v1(body)).hexdigest()
            if recomputed_pw != raw_payload.get("payload_digest"):
                hops.append(
                    _refuse_hop(HOP_RAW_WEIGHTS, REFUSE_CHAIN_MUTILATED, detail="weights digest")
                )
                admitted = False
                reason_terminal = REFUSE_CHAIN_MUTILATED
            else:
                hops.append(
                    _pass_hop(
                        HOP_RAW_WEIGHTS,
                        digests={
                            "payload_digest": str(raw_payload["payload_digest"]),
                            "path": str(
                                raw_payload.get(
                                    "path",
                                    "/internal/v1/challenges/agent-challenge/raw-weights",
                                )
                            ),
                        },
                    )
                )
    else:
        hops.append(
            _refuse_hop(
                HOP_RAW_WEIGHTS,
                reason_terminal,
                set_weights_invoked=set_w > 0,
            )
        )

    # --- X011 zero Base gateway dependency ---
    gw_hits = list((package.get("gateway_inventory") or {}).get("residuals") or [])
    if any(
        "/llm/v1" in str(r)
        for r in (package.get("gateway_inventory") or {}).get("master_routes", [])
    ):
        gw_hits.append("route:/llm/v1")
    if admitted and gw_hits:
        hops.append(
            _refuse_hop(
                HOP_GATEWAY_FREE,
                REFUSE_GATEWAY_RESIDUAL,
                detail=",".join(gw_hits),
                gateway_dependency=True,
            )
        )
        admitted = False
        reason_terminal = REFUSE_GATEWAY_RESIDUAL
    elif admitted:
        hops.append(
            _pass_hop(
                HOP_GATEWAY_FREE,
                reason_code="zero_base_gateway",
            )
        )
    else:
        hops.append(
            _refuse_hop(
                HOP_GATEWAY_FREE,
                reason_terminal,
                gateway_dependency=bool(gw_hits),
            )
        )

    # --- X012 fairness honesty ---
    honesty = dict(package.get("honesty") or {})
    real_claim = bool(honesty.get("real_provider_claimed"))
    tee_real = str(honesty.get("tee_real_provider") or "BLOCKED")
    if real_claim and not honesty.get("real_provider_pass_is_possible"):
        hops.append(
            _refuse_hop(
                HOP_HONESTY,
                "e2e_real_provider_pass_invented",
                detail="REAL-PROVIDER PASS invented without hard gates",
            )
        )
        admitted = False
        reason_terminal = "e2e_real_provider_pass_invented"
    elif admitted and tee_real not in {"BLOCKED", "REAL-PROVIDER PASS"}:
        hops.append(_refuse_hop(HOP_HONESTY, "e2e_tee_label_invalid"))
        admitted = False
        reason_terminal = "e2e_tee_label_invalid"
    elif admitted:
        # For fixture chain honesty: REAL must stay BLOCKED
        if tee_real == "REAL-PROVIDER PASS" and not honesty.get("real_provider_pass_is_possible"):
            hops.append(_refuse_hop(HOP_HONESTY, "e2e_real_provider_pass_invented"))
            admitted = False
            reason_terminal = "e2e_real_provider_pass_invented"
        else:
            hops.append(
                _pass_hop(
                    HOP_HONESTY,
                    reason_code="honesty_ok",
                    digests={
                        "tee_local_fixture": str(
                            honesty.get("tee_local_fixture", "LOCAL-FIXTURE PASS")
                        ),
                        "tee_real_provider": tee_real,
                    },
                )
            )
    else:
        hops.append(_refuse_hop(HOP_HONESTY, reason_terminal))

    # hop order sanity
    hop_ids = [h.hop_id for h in hops]
    if hop_ids != list(HOP_ORDER):
        admitted = False
        reason_terminal = REFUSE_HOP_ORDER

    raw_weights: dict[str, float] = {}
    if admitted and isinstance(raw_payload, Mapping):
        weights = raw_payload.get("weights") or {}
        if isinstance(weights, Mapping):
            raw_weights = {str(k): float(v) for k, v in weights.items()}

    return ChainAdmission(
        admitted=admitted,
        reason_code=reason_terminal if not admitted else "e2e_chain_verified",
        hops=tuple(hops),
        package=dict(package),
        package_digest=recomputed,
        production_emit=admitted and dual,
        raw_weights=raw_weights,
        gateway_free=not bool(gw_hits),
        reverify_ok=admitted,
        tee_labels={
            "LOCAL-FIXTURE": str(honesty.get("tee_local_fixture", "LOCAL-FIXTURE PASS")),
            "REAL-PROVIDER": tee_real,
        },
        side_effects={
            "set_weights": set_w,
            "burn_weights_24h": burn,
        },
    )


def run_attestation_chain(
    *,
    dual_flags_on: bool = True,
    issued_at_ms: int = T0_DEFAULT,
    received_at_ms: int | None = None,
    agent_llm_mode: str = MODE_TOOLS_ONLY,
    include_measured_agent_or: bool = False,
    master_env: Mapping[str, str] | None = None,
    master_routes: Sequence[str] | None = None,
    set_weights_count: int = 0,
    burn_invoked: bool = False,
    real_provider_pass_claimed: bool = False,
    **package_kwargs: Any,
) -> ChainAdmission:
    """Build a fresh complete package and re-verify it end-to-end."""

    package = build_complete_package(
        dual_flags_on=dual_flags_on,
        issued_at_ms=issued_at_ms,
        received_at_ms=received_at_ms,
        agent_llm_mode=agent_llm_mode,
        include_measured_agent_or=include_measured_agent_or,
        master_env=master_env,
        master_routes=master_routes,
        set_weights_count=set_weights_count,
        burn_invoked=burn_invoked,
        real_provider_pass_claimed=real_provider_pass_claimed,
        **package_kwargs,
    )
    return reverify_package(package)


def mutilate_package(
    package: Mapping[str, Any],
    *,
    path: Sequence[str],
    value: Any = None,
    delete: bool = False,
) -> dict[str, Any]:
    """Return a deep-copied package with one path mutated (or deleted).

    Preserves the *original* ``package_digest`` so independent re-verify forms
    refuse on digest mismatch or gate failure — never soft-accept.
    """

    out: MutableMapping[str, Any] = copy.deepcopy(dict(package))
    cursor: Any = out
    for key in path[:-1]:
        if not isinstance(cursor, MutableMapping) or key not in cursor:
            return dict(out)
        cursor = cursor[key]
    if not isinstance(cursor, MutableMapping):
        return dict(out)
    last = path[-1]
    if delete:
        cursor.pop(last, None)
    else:
        cursor[last] = value
    # Keep original package_digest so reverify detects material mutation.
    return dict(out)


def chronological_evidence_index(admission: ChainAdmission) -> dict[str, Any]:
    """Machine-readable chronological index for mission evidence (VAL-ACAT-020)."""

    assertion_table: dict[str, dict[str, Any]] = {}
    for hop in admission.hops:
        assertion_table[hop.assertion_id] = {
            "status": "PASS" if hop.status == "pass" else "FAIL",
            "hop_id": hop.hop_id,
            "reason_code": hop.reason_code,
            "detail": hop.detail,
        }
    assertion_table["VAL-ACAT-020"] = {
        "status": "PASS" if admission.admitted else "FAIL",
        "reason_code": admission.reason_code,
        "package_digest": admission.package_digest,
        "gateway_free": admission.gateway_free,
        "reverify_ok": admission.reverify_ok,
        "production_emit": admission.production_emit,
    }
    return {
        "schema_version": 1,
        "title": "agent-challenge black-box attestation chain",
        "admitted": admission.admitted,
        "reason_code": admission.reason_code,
        "package_digest": admission.package_digest,
        "chronological_hops": [h.as_dict() for h in admission.hops],
        "assertion_table": assertion_table,
        "raw_weights": dict(admission.raw_weights),
        "tee_labels": dict(admission.tee_labels),
        "side_effects": dict(admission.side_effects),
        "gateway_free": admission.gateway_free,
        "no_set_weights": admission.side_effects.get("set_weights", 0) == 0,
        "weights_path": "challenge-raw-weights → master raw-weight ingress → aggregate",
    }


def require_attestation_chain(**kwargs: Any) -> ChainAdmission:
    """Fail closed: raise if the chain does not fully admit."""

    decision = run_attestation_chain(**kwargs)
    if not decision.admitted:
        raise BlackboxChainError(decision.reason_code)
    return decision


__all__ = [
    "BASE_GATEWAY_ENV_NAMES",
    "BlackboxChainError",
    "ChainAdmission",
    "FRESHNESS_WINDOW_MS",
    "HOP_ASSERTIONS",
    "HOP_ORDER",
    "HopResult",
    "MINER_HOTKEY_DEFAULT",
    "MS_23H",
    "REFUSE_BURN_TRACKED",
    "REFUSE_CHAIN_INCOMPLETE",
    "REFUSE_CHAIN_MUTILATED",
    "REFUSE_GATEWAY_RESIDUAL",
    "REFUSE_HOP_ORDER",
    "REFUSE_PACKAGE_TAMPER",
    "REFUSE_SET_WEIGHTS",
    "REVIEW_DOMAIN",
    "SCORE_DOMAIN",
    "KEY_RELEASE_DOMAIN",
    "T0_DEFAULT",
    "build_complete_package",
    "build_fixture_eval_plan",
    "build_fixture_review_envelope",
    "build_key_release_grant",
    "build_raw_weights_payload",
    "canonical_json_v1",
    "chronological_evidence_index",
    "mutilate_package",
    "package_digest",
    "require_attestation_chain",
    "reverify_package",
    "run_attestation_chain",
    "scan_gateway_residuals",
]
