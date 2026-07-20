"""Score AttestationGate key-provider decode + schema-v2 canonical emit.

Live residual after image@sha256:5b190365 / compose 36e5c63e:
  * Full path preflight→KR grant→decrypt_ok→job_done→emit_start→**score_quote_ok**
    with schema-v1 attested result; dcap-qvl UpToDate; measurements match offline.
  * ``AttestationGate.decide_eval_result`` still rejects: ``replay.key_provider`` is
    raw dstack JSON hex ``{"name":"kms","id":...}`` while the plan pin is ``phala``
    (hex ``7068616c61``). KR already decodes via ``decode_key_provider`` (32ed505b);
    the score gate does not.
  * Secondary: guest ``BASE_BENCHMARK_RESULT`` / result body may use spaced
    ``json.dumps`` (default separators). Host ``process_direct_eval_result`` requires
    ``raw_body == eval_wire.canonical_json_v1(validated)`` so non-compact bodies are
    ``result_noncanonical``.

Offline fix (no Phala create):
  (1) ``AttestationGate.decide_eval_result``: ``decode_key_provider(replay.key_provider)``
      before pin compare.
  (2) schema-v2 / attested emit writes compact ``canonical_json_v1`` bytes.

Discriminators (would fail a wrong implementation):
  * Live-shaped JSON-hex key-provider + pin ``phala`` is VERIFIED (not VERIFICATION_FAILED).
  * Pin ``phala`` still rejects a non-phala provider payload.
  * Emission suffix bytes equal ``canonical_json_v1`` of the validated request
    (no spaced separators; revalidate equals raw).
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any

from agent_challenge.canonical import attested_result as ar
from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.attestation import (
    AttestationGate,
    AttestationOutcome,
    ResultMeasurementAllowlist,
)
from agent_challenge.keyrelease.quote import (
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
    decode_key_provider,
    os_image_hash_from_registers,
    replay_rtmr3,
)

REGS = {
    "mrtd": "11" * 48,
    "rtmr0": "22" * 48,
    "rtmr1": "33" * 48,
    "rtmr2": "44" * 48,
}
COMPOSE_HASH = "ab" * 32
OS_IMAGE_HASH = os_image_hash_from_registers(REGS["mrtd"], REGS["rtmr1"], REGS["rtmr2"])
AGENT_HASH = "55" * 32
LIVE_KEY_PROVIDER_JSON = b'{"name":"kms","id":"kms-live-score-1"}'


def _plan(*, key_provider: str = "phala") -> dict[str, Any]:
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
            "eval_run_id": "eval-score-gate-kp-1",
            "submission_id": "submission-score-gate-kp-1",
            "submission_version": 1,
            "authorizing_review_digest": "66" * 32,
            "agent_hash": AGENT_HASH,
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
                "compose_hash": COMPOSE_HASH,
                "app_identity": "agent-challenge-eval-v1",
                "kms_key_algorithm": "x25519",
                "kms_public_key_hex": "aa" * 32,
                "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("aa" * 32)).hexdigest(),
                "measurement": {
                    **REGS,
                    "os_image_hash": OS_IMAGE_HASH,
                    "key_provider": key_provider,
                    "vm_shape": "tdx-small",
                },
            },
            "key_release_endpoint": "validator.example:8701",
            "result_endpoint": "/evaluation/v1/runs/eval-score-gate-kp-1/result",
            "key_release_nonce": "key-release-score-gate-1",
            "score_nonce": "score-score-gate-1",
            "run_token_sha256": "bb" * 32,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )


def _request(
    plan: dict[str, Any],
    *,
    provider_payload: bytes = LIVE_KEY_PROVIDER_JSON,
) -> dict[str, Any]:
    from agent_challenge.evaluation.plan_scoring import build_score_record_from_eval_plan

    event_log, rtmr3 = build_rtmr3_event_log(
        [
            ("compose-hash", bytes.fromhex(COMPOSE_HASH)),
            ("key-provider", provider_payload),
        ]
    )
    # Discriminator: raw RTMR3 payload stays JSON hex; not the pin.
    assert replay_rtmr3(event_log).key_provider == provider_payload.hex()
    record = build_score_record_from_eval_plan(plan, {"task-a": [1.0]})
    scores_digest = ew.score_record_digest(record)
    binding = ew.build_score_binding(
        canonical_measurement={
            "mrtd": REGS["mrtd"],
            "rtmr0": REGS["rtmr0"],
            "rtmr1": REGS["rtmr1"],
            "rtmr2": REGS["rtmr2"],
            "compose_hash": COMPOSE_HASH,
            "os_image_hash": OS_IMAGE_HASH,
        },
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=["task-a"],
    )
    report_data = ew.score_report_data_hex(binding)
    quote = build_tdx_quote(
        mrtd=REGS["mrtd"],
        rtmr0=REGS["rtmr0"],
        rtmr1=REGS["rtmr1"],
        rtmr2=REGS["rtmr2"],
        rtmr3=rtmr3,
        report_data=report_data,
    )
    return {
        "schema_version": 1,
        "eval_run_id": plan["eval_run_id"],
        "submission_id": plan["submission_id"],
        "agent_hash": AGENT_HASH,
        "score_record": record,
        "scores_digest": scores_digest,
        "execution_proof": {
            "version": 1,
            "tier": "phala-tdx",
            "manifest_sha256": "cc" * 32,
            "image_digest": plan["eval_app"]["image_ref"],
            "provider": None,
            "worker_signature": {"worker_pubkey": "", "sig": ""},
            "attestation": {
                "tdx_quote": quote,
                "event_log": event_log,
                "report_data": report_data,
                "measurement": {
                    **REGS,
                    "rtmr3": rtmr3,
                    "compose_hash": COMPOSE_HASH,
                    "os_image_hash": OS_IMAGE_HASH,
                },
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": OS_IMAGE_HASH,
                },
            },
        },
    }


def _gate() -> AttestationGate:
    return AttestationGate(
        quote_verifier=StaticQuoteVerifier(),
        allowlist=ResultMeasurementAllowlist.from_measurements(
            [
                {
                    "mrtd": REGS["mrtd"],
                    "rtmr0": REGS["rtmr0"],
                    "rtmr1": REGS["rtmr1"],
                    "rtmr2": REGS["rtmr2"],
                    "compose_hash": COMPOSE_HASH,
                    "os_image_hash": OS_IMAGE_HASH,
                }
            ]
        ),
    )


# --------------------------------------------------------------------------- #
# (1) Score gate decodes live-shaped key-provider JSON hex like KR
# --------------------------------------------------------------------------- #


def test_decode_key_provider_maps_live_json_to_phala() -> None:
    """Shared KR helper (precondition) maps live dstack JSON hex → ``phala``."""

    assert decode_key_provider(LIVE_KEY_PROVIDER_JSON.hex()) == "phala"
    assert LIVE_KEY_PROVIDER_JSON.hex() != b"phala".hex()


def test_score_gate_accepts_live_shaped_key_provider_json_hex_with_pin_phala() -> None:
    """Live residual: JSON-hex key-provider + plan pin ``phala`` must VERIFIED.

    Without ``decode_key_provider`` the gate compares raw RTMR3 hex to
    ``phala``.encode().hex() and fails closed even though KR/allowlist pin is
    correct. Discriminator against that missed decode.
    """

    plan = _plan(key_provider="phala")
    request = _request(plan, provider_payload=LIVE_KEY_PROVIDER_JSON)
    # Document the residual comparison that used to fail.
    raw_hex = LIVE_KEY_PROVIDER_JSON.hex()
    pin = str(plan["eval_app"]["measurement"]["key_provider"])
    assert raw_hex != pin.encode("ascii").hex()
    assert decode_key_provider(raw_hex) == pin

    decision = _gate().decide_eval_result(
        request,
        eval_plan=plan,
        expected_agent_hash=AGENT_HASH,
        nonce_outstanding=True,
        key_granted=True,
    )
    assert decision.outcome is AttestationOutcome.VERIFIED
    assert decision.accepted is True


def test_score_gate_still_rejects_mismatched_key_provider_pin() -> None:
    """Pin ``phala`` still rejects non-phala provider after decode."""

    plan = _plan(key_provider="phala")
    request = _request(plan, provider_payload=b"evil-kms")
    decision = _gate().decide_eval_result(
        request,
        eval_plan=plan,
        expected_agent_hash=AGENT_HASH,
        nonce_outstanding=True,
        key_granted=True,
    )
    assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED
    assert decision.accepted is False


def test_score_gate_accepts_plain_pin_hex_unchanged() -> None:
    """Offline fixtures that already emit plain ``phala`` hex still verify."""

    plan = _plan(key_provider="phala")
    request = _request(plan, provider_payload=b"phala")
    decision = _gate().decide_eval_result(
        request,
        eval_plan=plan,
        expected_agent_hash=AGENT_HASH,
        nonce_outstanding=True,
        key_granted=True,
    )
    assert decision.outcome is AttestationOutcome.VERIFIED


# --------------------------------------------------------------------------- #
# (2) Attested schema-v2 emit: compact canonical_json_v1 body bytes
# --------------------------------------------------------------------------- #


class _QuoteProvider:
    def __init__(self, quote_hex: str, event_log: list[dict[str, Any]]) -> None:
        self._quote = quote_hex
        self._event_log = event_log

    def get_quote(self, report_data: bytes) -> Any:
        del report_data
        return type(
            "Q",
            (),
            {
                "quote": self._quote,
                "event_log": self._event_log,
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": OS_IMAGE_HASH,
                },
            },
        )()


def test_schema_v2_emit_bytes_equal_canonical_json_v1_of_validated() -> None:
    """Emission body after BASE_BENCHMARK_RESULT= must be canonical (no spaces).

    Spaced ``json.dumps`` (default separators) would fail
    ``process_direct_eval_result``'s exact ``raw_body == canonical_json_v1`` check.
    This is the live-shaped non-canonical residual for the score POST path.
    """

    plan = _plan(key_provider="phala")
    event_log, rtmr3 = build_rtmr3_event_log(
        [
            ("compose-hash", bytes.fromhex(COMPOSE_HASH)),
            ("key-provider", LIVE_KEY_PROVIDER_JSON),
        ]
    )
    # Precomputed score record via plan helper so emission validates.
    from agent_challenge.evaluation.plan_scoring import build_score_record_from_eval_plan

    record = build_score_record_from_eval_plan(plan, {"task-a": [1.0]})
    # Binding that matches what emit will recompute.
    scores_digest = ew.score_record_digest(record)
    binding = ew.build_score_binding(
        canonical_measurement={
            "mrtd": REGS["mrtd"],
            "rtmr0": REGS["rtmr0"],
            "rtmr1": REGS["rtmr1"],
            "rtmr2": REGS["rtmr2"],
            "compose_hash": COMPOSE_HASH,
            "os_image_hash": OS_IMAGE_HASH,
        },
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=["task-a"],
    )
    report_data = ew.score_report_data_hex(binding)
    quote = build_tdx_quote(
        mrtd=REGS["mrtd"],
        rtmr0=REGS["rtmr0"],
        rtmr1=REGS["rtmr1"],
        rtmr2=REGS["rtmr2"],
        rtmr3=rtmr3,
        report_data=report_data,
    )
    provider = _QuoteProvider(quote, event_log)
    buf = io.StringIO()
    line = ar.emit_attested_eval_result_from_plan(
        eval_plan=plan,
        score_record=record,
        rtmr3=rtmr3,
        quote_provider=provider,
        manifest_sha256="cc" * 32,
        stream=buf,
    )
    assert line.startswith(ar.RESULT_LINE_PREFIX)
    body = line[len(ar.RESULT_LINE_PREFIX) :].encode("utf-8")
    # Discriminator vs spaced dumps: default separators insert spaces.
    spaced = json.dumps(json.loads(body), ensure_ascii=False, sort_keys=True).encode("utf-8")
    assert b", " in spaced or b": " in spaced
    assert body != spaced

    validated = ew.validate_eval_result_request(json.loads(body))
    canonical = ew.canonical_json_v1(validated)
    assert body == canonical
    # Host process_direct_eval_result exact-byte gate.
    assert ew.canonical_json_v1(validated) == body


def test_schema_v2_emit_stream_output_matches_return_line() -> None:
    plan = _plan(key_provider="phala")
    event_log, rtmr3 = build_rtmr3_event_log(
        [
            ("compose-hash", bytes.fromhex(COMPOSE_HASH)),
            ("key-provider", b"phala"),
        ]
    )
    from agent_challenge.evaluation.plan_scoring import build_score_record_from_eval_plan

    record = build_score_record_from_eval_plan(plan, {"task-a": [1.0]})
    scores_digest = ew.score_record_digest(record)
    binding = ew.build_score_binding(
        canonical_measurement={
            "mrtd": REGS["mrtd"],
            "rtmr0": REGS["rtmr0"],
            "rtmr1": REGS["rtmr1"],
            "rtmr2": REGS["rtmr2"],
            "compose_hash": COMPOSE_HASH,
            "os_image_hash": OS_IMAGE_HASH,
        },
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=["task-a"],
    )
    report_data = ew.score_report_data_hex(binding)
    quote = build_tdx_quote(
        mrtd=REGS["mrtd"],
        rtmr0=REGS["rtmr0"],
        rtmr1=REGS["rtmr1"],
        rtmr2=REGS["rtmr2"],
        rtmr3=rtmr3,
        report_data=report_data,
    )
    buf = io.StringIO()
    line = ar.emit_attested_eval_result_from_plan(
        eval_plan=plan,
        score_record=record,
        rtmr3=rtmr3,
        quote_provider=_QuoteProvider(quote, event_log),
        manifest_sha256="cc" * 32,
        stream=buf,
    )
    assert buf.getvalue() == line + "\n"
