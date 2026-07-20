"""Strict, deterministic policy verification for attested review.

The OpenRouter response is only an advisory input. This module has no transport,
filesystem, clock, or mutable-rule dependency, so the same immutable facts always
produce the same canonical decision bytes.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

from .canonical import CanonicalJsonError, canonical_json_v1, parse_json_object
from .schemas import MAX_OPENROUTER_RESPONSE_BYTES

_MAX_REASON_CODES = 32
# Model tool-call citation list remains bounded (matches submit_verdict schema).
_MAX_EVIDENCE_PATHS = 64
# Assignment allowlist holds one relative path per package file. Must cover the
# submissions MAX_FILES bound (200). Prior review_runtime tripled each name as
# {path, artifact/path, submission/path}, so N>22 always hit the old multi-set
# check against _MAX_EVIDENCE_PATHS and forced policy_output_malformed before
# allow.
_MAX_ASSIGNED_EVIDENCE_PATHS = 200
_MAX_REASON_CODE_LENGTH = 64
_MAX_EVIDENCE_PATH_LENGTH = 512
_MAX_FINDINGS_PER_SOURCE = 256
# Architecture/oracle bound: final decision reason_codes and evidence_digests
# lists are each capped at 256 entries before any report serialization. The
# live production path prefers ChallengeSettings.review_max_reason_evidence_items.
MAX_REVIEW_DECISION_ENTRIES = 256


def reason_evidence_limit_from_settings(settings: object | None = None) -> int:
    """Resolve the aggregate reason/evidence cap from live config when present."""

    if settings is None:
        return MAX_REVIEW_DECISION_ENTRIES
    return int(getattr(settings, "review_max_reason_evidence_items", MAX_REVIEW_DECISION_ENTRIES))


_MAX_TOOL_ARGUMENT_BYTES = 16 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MODEL_VERDICTS = frozenset({"allow", "reject", "escalate"})
_FINDING_DISPOSITIONS = frozenset({"reject", "escalate"})
_FINDING_SOURCES = frozenset({"static", "dynamic_rule", "prompt"})
_SIMILARITY_BANDS = frozenset({"low", "medium", "high"})


class ReviewPolicyError(ValueError):
    """A model policy response or immutable verifier input is not exact."""


@dataclass(frozen=True)
class ModelPolicyOutput:
    """The sole bounded advisory final tool output accepted from the model."""

    verdict: Literal["allow", "reject", "escalate"]
    reason_codes: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    canonical_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        if self.verdict not in _MODEL_VERDICTS:
            raise ReviewPolicyError("model verdict is not allowed")
        _require_canonical_string_tuple(
            self.reason_codes,
            field_name="model reason codes",
            maximum_items=_MAX_REASON_CODES,
            maximum_length=_MAX_REASON_CODE_LENGTH,
            validator=_require_reason_code,
        )
        _require_canonical_string_tuple(
            self.evidence_paths,
            field_name="model evidence paths",
            maximum_items=_MAX_EVIDENCE_PATHS,
            maximum_length=_MAX_EVIDENCE_PATH_LENGTH,
            validator=_require_evidence_path,
        )
        expected_bytes = canonical_json_v1(
            {
                "schema_version": 1,
                "verdict": self.verdict,
                "reason_codes": list(self.reason_codes),
                "evidence_paths": list(self.evidence_paths),
            }
        )
        if (
            self.canonical_bytes != expected_bytes
            or self.sha256 != sha256(expected_bytes).hexdigest()
        ):
            raise ReviewPolicyError("model policy output binding is invalid")


@dataclass(frozen=True)
class PolicyFinding:
    """A precomputed immutable static, rules, or prompt policy finding."""

    source: Literal["static", "dynamic_rule", "prompt"]
    reason_code: str
    disposition: Literal["reject", "escalate"]
    evidence_sha256: str

    def __post_init__(self) -> None:
        _require_finding_source(self.source)
        _require_reason_code(self.reason_code)
        if self.disposition not in _FINDING_DISPOSITIONS:
            raise ReviewPolicyError("policy finding disposition is invalid")
        _require_sha256(self.evidence_sha256, "policy finding evidence digest")


@dataclass(frozen=True)
class SimilarityFinding:
    """Immutable similarity metadata with source-free evidence binding."""

    risk_band: Literal["low", "medium", "high"]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.risk_band not in _SIMILARITY_BANDS:
            raise ReviewPolicyError("similarity risk band is invalid")
        _require_sha256(self.evidence_sha256, "similarity evidence digest")


@dataclass(frozen=True)
class ReviewPolicyInput:
    """All immutable evidence consumed by the final network-free verifier."""

    static_findings: tuple[PolicyFinding, ...] = ()
    similarity_findings: tuple[SimilarityFinding, ...] = ()
    dynamic_rule_findings: tuple[PolicyFinding, ...] = ()
    prompt_findings: tuple[PolicyFinding, ...] = ()
    model_output: ModelPolicyOutput | None = None

    def __post_init__(self) -> None:
        _require_findings(self.static_findings, "static")
        _require_findings(self.dynamic_rule_findings, "dynamic_rule")
        _require_findings(self.prompt_findings, "prompt")
        if (
            not isinstance(self.similarity_findings, tuple)
            or len(self.similarity_findings) > _MAX_FINDINGS_PER_SOURCE
        ):
            raise ReviewPolicyError("similarity findings must be an immutable tuple")
        if not all(isinstance(item, SimilarityFinding) for item in self.similarity_findings):
            raise ReviewPolicyError("similarity findings are invalid")
        if self.model_output is not None and not isinstance(self.model_output, ModelPolicyOutput):
            raise ReviewPolicyError("model output is invalid")


@dataclass(frozen=True)
class ReviewPolicyDecision:
    """The canonical final authority decision consumed by later report binding."""

    verdict: Literal["allow", "reject", "escalate"]
    reason_codes: tuple[str, ...]
    evidence_digests: tuple[str, ...]
    canonical_bytes: bytes
    sha256: str

    def public_projection(self) -> dict[str, Any]:
        """Return the safe deterministic component of a later public report."""

        return {
            "schema_version": 1,
            "verdict": self.verdict,
            "reason_codes": list(self.reason_codes),
            "evidence_digests": list(self.evidence_digests),
            "verifier_output_sha256": self.sha256,
        }


def parse_model_policy_output(
    response_body: bytes,
    *,
    allowed_evidence_paths: set[str] | frozenset[str],
) -> ModelPolicyOutput:
    """Parse exactly one bounded ``submit_verdict`` call from OpenRouter bytes.

    The parser intentionally accepts no content fallback, no legacy function-call
    aliases, no duplicate final calls, and no open payload fields.
    """

    if not isinstance(response_body, bytes) or not (
        1 <= len(response_body) <= MAX_OPENROUTER_RESPONSE_BYTES
    ):
        raise ReviewPolicyError("model response exceeds policy parser bound")
    if not isinstance(allowed_evidence_paths, (set, frozenset)):
        raise ReviewPolicyError("allowed evidence paths must be a set")
    allowed_paths = _validated_allowed_paths(allowed_evidence_paths)
    # Outer OpenRouter envelopes may include usage floats; accept ordinary JSON
    # maps here. The tool-argument object below still rejects floats via the
    # canonical parser.
    try:
        response = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewPolicyError("model response is not a duplicate-free JSON object") from exc
    if not isinstance(response, Mapping):
        raise ReviewPolicyError("model response is not a duplicate-free JSON object")
    from .schemas import is_pinned_review_model

    if not is_pinned_review_model(response.get("model")):
        raise ReviewPolicyError("model response does not match pinned model")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ReviewPolicyError("model response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
        raise ReviewPolicyError("model response choice is malformed")
    message = choice["message"]
    if message.get("role") != "assistant":
        raise ReviewPolicyError("model prose is not a policy verdict")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        # Without the required tool call, any assistant prose is not a verdict.
        if message.get("content") not in {None, ""}:
            raise ReviewPolicyError("model prose is not a policy verdict")
        raise ReviewPolicyError("model response must contain exactly one final tool call")
    # Provider models (including x-ai/grok-* with thinking enabled) may fill
    # freeform content/reasoning fields beside the required tool call. The sole
    # acceptance channel remains the tool payload; freeform content is ignored.
    call = calls[0]
    # Require the three OpenAI-compat tool-call fields; tolerate provider extras
    # such as ``index`` from Moonshot/OpenRouter snapshots.
    if (
        not isinstance(call, Mapping)
        or not {"id", "type", "function"}.issubset(set(call))
        or not isinstance(call.get("id"), str)
        or not 1 <= len(call["id"]) <= 128
        or call.get("type") != "function"
    ):
        raise ReviewPolicyError("model tool call is malformed")
    function = call.get("function")
    if not isinstance(function, Mapping) or not {"name", "arguments"}.issubset(set(function)):
        raise ReviewPolicyError("model function shape is not exact")
    if (
        function["name"] != "submit_verdict"
        or not isinstance(function["arguments"], str)
        or len(function["arguments"].encode("utf-8")) > _MAX_TOOL_ARGUMENT_BYTES
    ):
        raise ReviewPolicyError("model used an unassigned policy tool")
    try:
        # reject_duplicate_keys=true behavior of parse_json_object; OpenRouter
        # argument strings often contain optional whitespace so normalize with
        # ordinary loads first only when the strict path rejects solely for
        # whitespace (forbids floats still enforced below on extracted fields).
        try:
            arguments = parse_json_object(function["arguments"].encode("utf-8"))
        except CanonicalJsonError as strict_exc:
            # Re-check forTrue duplicate keys using raw scan before softening.
            if (
                '"verdict"' in function["arguments"]
                and function["arguments"].count('"verdict"') > 1
            ):
                raise ReviewPolicyError("model policy arguments are malformed") from strict_exc
            try:
                arguments = json.loads(function["arguments"])
            except (UnicodeDecodeError, json.JSONDecodeError) as soft_exc:
                raise ReviewPolicyError("model policy arguments are malformed") from soft_exc
            if not isinstance(arguments, Mapping):
                raise ReviewPolicyError("model policy arguments are malformed") from strict_exc
    except ReviewPolicyError:
        raise
    if set(arguments) != {"verdict", "reason_codes", "evidence_paths"}:
        raise ReviewPolicyError("model policy arguments contain unknown or missing fields")
    verdict = arguments["verdict"]
    if verdict not in _MODEL_VERDICTS:
        raise ReviewPolicyError("model verdict is not allowed")
    reason_codes = _validated_string_set(
        arguments["reason_codes"],
        field_name="model reason codes",
        maximum_items=_MAX_REASON_CODES,
        maximum_length=_MAX_REASON_CODE_LENGTH,
        validator=_require_reason_code,
    )
    evidence_paths = _validated_string_set(
        arguments["evidence_paths"],
        field_name="model evidence paths",
        maximum_items=_MAX_EVIDENCE_PATHS,
        maximum_length=_MAX_EVIDENCE_PATH_LENGTH,
        validator=_require_evidence_path,
    )
    # Evidence paths are advisory only. Keep only allowlisted package-relative
    # paths (and legacy artifact/submission/ mounts mapped onto them) and drop
    # freeform provider citations. Deterministic verifier remains final.
    evidence_paths = _filter_model_evidence_paths(evidence_paths, allowed_paths)
    canonical = canonical_json_v1(
        {
            "schema_version": 1,
            "verdict": verdict,
            "reason_codes": list(reason_codes),
            "evidence_paths": list(evidence_paths),
        }
    )
    return ModelPolicyOutput(
        verdict=verdict,
        reason_codes=reason_codes,
        evidence_paths=evidence_paths,
        canonical_bytes=canonical,
        sha256=sha256(canonical).hexdigest(),
    )


def verify_review_policy(policy_input: ReviewPolicyInput) -> ReviewPolicyDecision:
    """Apply deterministic policy precedence without any I/O or mutable lookup."""

    if not isinstance(policy_input, ReviewPolicyInput):
        raise ReviewPolicyError("review policy input is invalid")

    reject_reasons: set[str] = set()
    escalate_reasons: set[str] = set()
    evidence_digests: set[str] = set()

    for finding in policy_input.static_findings:
        # Static scanner findings identify already-known cheat classes and are
        # never softened by a model, rules, similarity, or prompt judgement.
        reject_reasons.add(finding.reason_code)
        evidence_digests.add(finding.evidence_sha256)
    for finding in (*policy_input.dynamic_rule_findings, *policy_input.prompt_findings):
        (reject_reasons if finding.disposition == "reject" else escalate_reasons).add(
            finding.reason_code
        )
        evidence_digests.add(finding.evidence_sha256)
    for finding in policy_input.similarity_findings:
        evidence_digests.add(finding.evidence_sha256)
        if finding.risk_band == "high":
            escalate_reasons.add("similarity_high_risk")
        elif finding.risk_band == "medium":
            escalate_reasons.add("similarity_medium_risk")

    model = policy_input.model_output
    if model is None:
        escalate_reasons.add("model_output_malformed")
    else:
        evidence_digests.add(model.sha256)
        if model.verdict == "reject":
            reject_reasons.add("model_reject")
        elif model.verdict == "escalate":
            escalate_reasons.add("model_escalate")
        reject_reasons.update(model.reason_codes if model.verdict == "reject" else ())
        escalate_reasons.update(model.reason_codes if model.verdict == "escalate" else ())

    if reject_reasons:
        verdict: Literal["allow", "reject", "escalate"] = "reject"
        reason_codes = tuple(sorted(reject_reasons | escalate_reasons))
    elif escalate_reasons:
        verdict = "escalate"
        reason_codes = tuple(sorted(escalate_reasons))
    else:
        verdict = "allow"
        reason_codes = ("policy_passed",)

    ordered_evidence = tuple(sorted(evidence_digests))
    if (
        len(reason_codes) > MAX_REVIEW_DECISION_ENTRIES
        or len(ordered_evidence) > MAX_REVIEW_DECISION_ENTRIES
    ):
        raise ReviewPolicyError(
            "review policy decision exceeds aggregate 256 reason or evidence-entry bound"
        )

    canonical = canonical_json_v1(
        {
            "schema_version": 1,
            "verdict": verdict,
            "reason_codes": list(reason_codes),
            "evidence_digests": list(ordered_evidence),
        }
    )
    return ReviewPolicyDecision(
        verdict=verdict,
        reason_codes=reason_codes,
        evidence_digests=ordered_evidence,
        canonical_bytes=canonical,
        sha256=sha256(canonical).hexdigest(),
    )


def _validated_allowed_paths(value: set[str] | frozenset[str]) -> frozenset[str]:
    if len(value) > _MAX_ASSIGNED_EVIDENCE_PATHS:
        raise ReviewPolicyError("too many assigned evidence paths")
    paths: set[str] = set()
    for path in value:
        _require_evidence_path(path)
        paths.add(path)
    return frozenset(paths)


def _filter_model_evidence_paths(
    evidence_paths: tuple[str, ...],
    allowed_paths: frozenset[str],
) -> tuple[str, ...]:
    """Keep advisory citations that resolve to package-relative allowed paths.

    Allowed sets store single relative ZIP members only. Models may still cite
    legacy ``artifact/`` or ``submission/`` mount prefixes; map those onto the
    relative member when present so visibility is preserved without a 3N
    allowlist blow-up.
    """

    kept: set[str] = set()
    for path in evidence_paths:
        resolved = _resolve_allowed_evidence_path(path, allowed_paths)
        if resolved is not None:
            kept.add(resolved)
    return tuple(sorted(kept))


def _resolve_allowed_evidence_path(
    path: str,
    allowed_paths: frozenset[str],
) -> str | None:
    if path in allowed_paths:
        return path
    for prefix in ("artifact/", "submission/"):
        if path.startswith(prefix):
            relative = path[len(prefix) :]
            if relative in allowed_paths:
                return relative
    return None


def _validated_string_set(
    value: object,
    *,
    field_name: str,
    maximum_items: int,
    maximum_length: int,
    validator: Any,
) -> tuple[str, ...]:
    """Normalize an advisory model string array into a sorted unique tuple.

    Measured OpenRouter policy arguments are untrusted and frequently return
    unsorted or duplicate reason codes. Canonicalization is deliberation-safe
    and required so a valid tool call is not rejected for presentation order.
    """

    if not isinstance(value, list) or len(value) > maximum_items:
        raise ReviewPolicyError(f"{field_name} must be a bounded array")
    seen: set[str] = set()
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or len(item) > maximum_length:
            raise ReviewPolicyError(f"{field_name} item is invalid")
        if unicodedata.normalize("NFC", item) != item:
            raise ReviewPolicyError(f"{field_name} items must be NFC-normalized")
        validator(item)
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(sorted(normalized))


def _require_findings(value: object, source: str) -> None:
    if not isinstance(value, tuple) or len(value) > _MAX_FINDINGS_PER_SOURCE:
        raise ReviewPolicyError(f"{source} findings must be an immutable tuple")
    if not all(isinstance(item, PolicyFinding) and item.source == source for item in value):
        raise ReviewPolicyError(f"{source} findings are invalid")


def _require_finding_source(value: str) -> None:
    if value not in _FINDING_SOURCES:
        raise ReviewPolicyError("policy finding source is invalid")


def _require_canonical_string_tuple(
    value: object,
    *,
    field_name: str,
    maximum_items: int,
    maximum_length: int,
    validator: Any,
) -> None:
    if not isinstance(value, tuple) or len(value) > maximum_items:
        raise ReviewPolicyError(f"{field_name} must be a bounded immutable tuple")
    for item in value:
        if not isinstance(item, str) or len(item) > maximum_length:
            raise ReviewPolicyError(f"{field_name} item is invalid")
        if unicodedata.normalize("NFC", item) != item:
            raise ReviewPolicyError(f"{field_name} items must be NFC-normalized")
        validator(item)
    if list(value) != sorted(value) or len(set(value)) != len(value):
        raise ReviewPolicyError(f"{field_name} must be sorted and unique")


def _require_reason_code(value: object) -> None:
    if not isinstance(value, str) or not _REASON_CODE_RE.fullmatch(value):
        raise ReviewPolicyError("reason code is invalid")


def _require_evidence_path(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ReviewPolicyError("evidence path is invalid")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ReviewPolicyError(f"{name} is invalid")


__all__ = [
    "MAX_REVIEW_DECISION_ENTRIES",
    "ModelPolicyOutput",
    "PolicyFinding",
    "ReviewPolicyDecision",
    "ReviewPolicyError",
    "ReviewPolicyInput",
    "SimilarityFinding",
    "parse_model_policy_output",
    "verify_review_policy",
]
