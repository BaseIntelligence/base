"""Rules reviewer for the pre-evaluation analyzer pipeline.

Historically this module routed a bounded prompt through the **Base master LLM
gateway** (``GatewayReviewProvider`` + ``/llm/v1``). That path is **removed**
(VAL-ACAT-015): Base gateway is gone and must not be resurrected.

Production rules trust for scoring comes from the challenge-owned **measured
review harness** (Phala review CVM + real OpenRouter under ``.rules`` with
digest-bound attestation). This module may still implement the
``analyzer.pipeline.AnalyzerReviewer`` protocol for offline/static flows when
injected with a non-gateway provider (tests, local fakes). Building a configured
rules reviewer never wires Base gateway tokens.

When no non-gateway provider is available, lifecycle skips this reviewer rather
than parking on ``missing_llm_gateway_token``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_challenge.analyzer.llm_reviewer import (
    LlmProviderResponse,
    LlmProviderUnavailable,
    LlmReviewProvider,
)
from agent_challenge.analyzer.schemas import (
    EvidenceItem,
    OverallVerdict,
    ReviewerRequest,
    ReviewerResult,
)
from agent_challenge.core.config import settings

RULES_REVIEWER_PROMPT_VERSION = "rules-reviewer-v2-no-base-gateway"
RULES_REVIEW_TOOL = "submit_rules_review"
#: Reason code stamped on the fail-safe ``suspicious`` result when the injected
#: provider is unavailable (infra failure, not a model judgement). Legacy alias
#: preserved for existing tests; does not imply Base gateway consumption.
RULES_REVIEWER_INFRA_REASON = "rules_reviewer_provider_unavailable"
#: Back-compat alias for the historical gateway outage reason string.
RULES_REVIEWER_GATEWAY_UNAVAILABLE_LEGACY = "rules_reviewer_gateway_unavailable"
_ALLOWED_VERDICTS: tuple[str, ...] = ("valid", "invalid", "suspicious", "error")


class _EvidenceInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = ""
    line_start: int = 1
    line_end: int = 1
    snippet: str = ""
    reason_code: str = ""
    description: str = ""


class _RulesReviewPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: str
    reason_codes: list[str] = []
    evidence: list[_EvidenceInput] = []
    notes: str = ""


class GatewayRulesReviewer:
    """Analyzer protocol adapter for an injected non-gateway review provider.

    The class name is retained for import stability; it does **not** require or
    consume Base ``llm_gateway_*`` Settings. Callers that previously built a
    ``GatewayReviewProvider`` for this class must use measured OR or a test
    fake instead (see :func:`build_configured_rules_reviewer`).
    """

    provider_name = "rules_reviewer"

    def __init__(
        self,
        *,
        provider: LlmReviewProvider,
        timeout_seconds: float | None = None,
        per_read_max_bytes: int | None = None,
        total_read_budget: int | None = None,
    ) -> None:
        # Refuse residual Base gateway providers so this type cannot re-consume
        # master /llm/v1 tokens through accidental wiring.
        provider_name = getattr(provider, "provider_name", "") or ""
        base_url = str(getattr(provider, "base_url", "") or "")
        if provider_name == "gateway" or "/llm/v1" in base_url:
            raise ValueError(
                "GatewayRulesReviewer must not consume Base LLM gateway providers "
                "(provider_name=gateway or /llm/v1 base); use measured review "
                "OpenRouter or a non-gateway test provider"
            )
        self.provider = provider
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else settings.llm_reviewer_timeout_seconds
        )
        self.per_read_max_bytes = int(
            per_read_max_bytes
            if per_read_max_bytes is not None
            else settings.llm_reviewer_read_max_bytes
        )
        self.total_read_budget = int(
            total_read_budget
            if total_read_budget is not None
            else settings.llm_reviewer_read_total_budget_bytes
        )

    def review(self, request: ReviewerRequest) -> ReviewerResult:
        try:
            response = self.provider.complete(
                messages=self._messages(request),
                tools=_tool_schemas(),
                tool_choice={"type": "function", "function": {"name": RULES_REVIEW_TOOL}},
                timeout_seconds=self.timeout_seconds,
            )
        except LlmProviderUnavailable:
            return _fail_safe(RULES_REVIEWER_INFRA_REASON)
        payload = _extract_payload(response)
        if payload is None:
            return _fail_safe("rules_reviewer_unparseable")
        result = _to_reviewer_result(payload)
        if result is None:
            return _fail_safe("rules_reviewer_unparseable")
        return result

    def _messages(self, request: ReviewerRequest) -> list[dict[str, Any]]:
        payload = {
            "prompt_version": RULES_REVIEWER_PROMPT_VERSION,
            "rules_version": request.rules_version,
            "rule_files": request.rule_files,
            "policy_excerpt": request.policy_excerpt,
            "workspace_files": request.workspace_files,
            "file_contents": self._bounded_file_contents(request),
            "static_findings": [
                finding.model_dump(mode="json") for finding in request.static_findings
            ],
            "instructions": [
                "Evaluate the submission against EACH policy in policy_excerpt: "
                "acceptance, anti-cheat, hardcoding, and security.",
                "file_contents entries are line-numbered ('N: text'); cite the exact "
                "path and line numbers in evidence.",
                "There is no shell, python, or network tool; never execute the submitted code.",
                "Respond with exactly one submit_rules_review call.",
                "verdict=invalid when any policy is clearly violated (reject-worthy); "
                "verdict=valid when all policies pass; verdict=suspicious when a violation "
                "is plausible but unproven or the evidence is insufficient.",
                "Anti-cheat: reject reading hidden test/solution/oracle/task-metadata files "
                "to discover the expected result (reads_hidden_tests), output tailored to "
                "specific tests (tailors_to_tests), or branching on task identity "
                "(branches_on_task_identity).",
                "Shared similarity to the common baseagent skeleton is expected and is not "
                "a violation on its own.",
            ],
        }
        system_text = (
            "You are a strict, bounded Agent Challenge rules reviewer. You judge a "
            "submission against the provided .rules policies using only the supplied "
            "file contents. You cannot run tools or execute code. Respond only via the "
            "submit_rules_review tool."
        )
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ]

    def _bounded_file_contents(self, request: ReviewerRequest) -> list[dict[str, Any]]:
        bounded: list[dict[str, Any]] = []
        remaining = self.total_read_budget
        for item in request.file_contents:
            if remaining <= 0:
                break
            content = item.content
            truncated = item.truncated
            encoded = content.encode("utf-8")
            if len(encoded) > self.per_read_max_bytes:
                content = encoded[: self.per_read_max_bytes].decode("utf-8", errors="ignore")
                truncated = True
                encoded = content.encode("utf-8")
            if len(encoded) > remaining:
                content = encoded[:remaining].decode("utf-8", errors="ignore")
                truncated = True
                encoded = content.encode("utf-8")
            remaining -= len(encoded)
            bounded.append({"path": item.path, "content": content, "truncated": truncated})
        return bounded


def _tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": RULES_REVIEW_TOOL,
                "description": "Submit the structured rules-policy verdict for the submission.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": list(_ALLOWED_VERDICTS),
                            "description": "valid|invalid|suspicious|error",
                        },
                        "reason_codes": {"type": "array", "items": {"type": "string"}},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "path": {"type": "string"},
                                    "line_start": {"type": "integer", "minimum": 1},
                                    "line_end": {"type": "integer", "minimum": 1},
                                    "snippet": {"type": "string"},
                                    "reason_code": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["path", "line_start", "line_end", "reason_code"],
                            },
                        },
                        "notes": {"type": "string"},
                    },
                    "required": ["verdict"],
                },
            },
        }
    ]


def _extract_payload(response: LlmProviderResponse) -> dict[str, Any] | None:
    for call in response.tool_calls:
        if call.name == RULES_REVIEW_TOOL and isinstance(call.arguments, Mapping):
            return dict(call.arguments)
    return _json_from_content(response.content)


def _json_from_content(content: str) -> dict[str, Any] | None:
    stripped = content.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _to_reviewer_result(payload: Mapping[str, Any]) -> ReviewerResult | None:
    try:
        parsed = _RulesReviewPayload.model_validate(payload)
    except ValidationError:
        return None
    verdict = parsed.verdict.strip().lower()
    if verdict not in _ALLOWED_VERDICTS:
        return None
    reason_codes = [code for code in parsed.reason_codes if code] or [f"rules_reviewer_{verdict}"]
    return ReviewerResult(
        verdict=cast(OverallVerdict, verdict),
        reason_codes=reason_codes,
        evidence=[_to_evidence(item) for item in parsed.evidence],
        notes=parsed.notes[:4000],
    )


def _to_evidence(item: _EvidenceInput) -> EvidenceItem:
    line_start = item.line_start if item.line_start >= 1 else 1
    line_end = item.line_end if item.line_end >= line_start else line_start
    return EvidenceItem(
        path=item.path,
        line_start=line_start,
        line_end=line_end,
        snippet=item.snippet,
        reason_code=item.reason_code or "rules_violation",
        description=item.description,
    )


def _fail_safe(reason_code: str) -> ReviewerResult:
    return ReviewerResult(
        verdict="suspicious",
        reason_codes=[reason_code],
        evidence=[],
        notes="rules reviewer failed safe to needs_review",
    )
