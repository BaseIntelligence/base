from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_challenge.analyzer.gateway_rules_reviewer import (
    RULES_REVIEW_TOOL,
    RULES_REVIEWER_INFRA_REASON,
    GatewayRulesReviewer,
)
from agent_challenge.analyzer.llm_reviewer import (
    GATEWAY_PLACEHOLDER_MODEL,
    LlmProviderResponse,
    LlmProviderTimeout,
    LlmProviderUnavailable,
    LlmToolCall,
)
from agent_challenge.analyzer.pipeline import run_rules_analyzer
from agent_challenge.analyzer.schemas import ReviewerRequest, WorkspaceFileContent


class FakeGatewayProvider:
    # VAL-ACAT-015: must not present as Base master gateway (provider_name=gateway
    # or /llm/v1 is rejected by GatewayRulesReviewer).
    provider_name = "test_fake_provider"
    model_name = GATEWAY_PLACEHOLDER_MODEL

    def __init__(
        self,
        *,
        response: LlmProviderResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str | Mapping[str, Any],
        timeout_seconds: float,
    ) -> LlmProviderResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "tool_choice": tool_choice,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _tool_response(arguments: dict[str, Any]) -> LlmProviderResponse:
    return LlmProviderResponse(
        tool_calls=(LlmToolCall(id="rc-1", name=RULES_REVIEW_TOOL, arguments=arguments),)
    )


def _benign_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "agent.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
    return workspace


def _request() -> ReviewerRequest:
    return ReviewerRequest(
        rules_version="rules-test",
        rule_files=[".rules/anti-cheat.md"],
        policy_excerpt="reject reads_hidden_tests",
        workspace_files=["agent.py"],
        static_findings=[],
        file_contents=[WorkspaceFileContent(path="agent.py", content="1: def solve(): ...")],
    )


def test_gateway_rules_reviewer_invalid_with_evidence(tmp_path: Path) -> None:
    provider = FakeGatewayProvider(
        response=_tool_response(
            {
                "verdict": "invalid",
                "reason_codes": ["reads_hidden_tests"],
                "evidence": [
                    {
                        "path": "agent.py",
                        "line_start": 2,
                        "line_end": 2,
                        "snippet": "open('/app/tests/test_x.py')",
                        "reason_code": "reads_hidden_tests",
                        "description": "reads hidden benchmark tests",
                    }
                ],
                "notes": "clear anti-cheat violation",
            }
        )
    )
    reviewer = GatewayRulesReviewer(provider=provider)

    report = run_rules_analyzer(_benign_workspace(tmp_path), reviewer=reviewer)

    assert report.overall_verdict == "invalid"
    assert report.recommended_status == "rejected"
    assert report.reviewer_used is True
    assert "reads_hidden_tests" in report.reason_codes
    assert report.evidence
    assert report.evidence[0].reason_code == "reads_hidden_tests"
    assert report.evidence[0].path == "agent.py"
    # The gateway was actually invoked with a forced rules-review tool call.
    assert provider.calls
    tool_choice = provider.calls[0]["tool_choice"]
    assert isinstance(tool_choice, dict)
    assert tool_choice["function"]["name"] == RULES_REVIEW_TOOL


def test_gateway_rules_reviewer_valid(tmp_path: Path) -> None:
    provider = FakeGatewayProvider(
        response=_tool_response({"verdict": "valid", "reason_codes": ["rules_passed"]})
    )
    reviewer = GatewayRulesReviewer(provider=provider)

    report = run_rules_analyzer(_benign_workspace(tmp_path), reviewer=reviewer)

    assert report.overall_verdict == "valid"
    assert report.recommended_status == "accepted"
    assert report.reviewer_used is True
    assert {result.status for result in report.rule_results} == {"pass"}


def test_gateway_rules_reviewer_unparseable_is_suspicious(tmp_path: Path) -> None:
    provider = FakeGatewayProvider(response=LlmProviderResponse(content="not-json-at-all"))
    reviewer = GatewayRulesReviewer(provider=provider)

    report = run_rules_analyzer(_benign_workspace(tmp_path), reviewer=reviewer)

    assert report.overall_verdict == "suspicious"
    assert report.recommended_status == "needs_review"


def test_gateway_rules_reviewer_unavailable_is_suspicious_infra(tmp_path: Path) -> None:
    provider = FakeGatewayProvider(error=LlmProviderUnavailable("provider down"))
    reviewer = GatewayRulesReviewer(provider=provider)

    report = run_rules_analyzer(_benign_workspace(tmp_path), reviewer=reviewer)

    assert report.overall_verdict == "suspicious"
    assert RULES_REVIEWER_INFRA_REASON in report.reason_codes


def test_gateway_rules_reviewer_timeout_is_suspicious_infra(tmp_path: Path) -> None:
    provider = FakeGatewayProvider(error=LlmProviderTimeout("provider timeout"))
    reviewer = GatewayRulesReviewer(provider=provider)

    result = reviewer.review(_request())

    assert result.verdict == "suspicious"
    assert result.reason_codes == [RULES_REVIEWER_INFRA_REASON]


def test_gateway_rules_reviewer_refuses_base_gateway_provider() -> None:
    class _BaseGatewayProvider:
        provider_name = "gateway"
        model_name = GATEWAY_PLACEHOLDER_MODEL
        base_url = "https://master.example/llm/v1"

        def complete(self, **kwargs):  # noqa: ANN003
            raise AssertionError("must not be called")

    try:
        GatewayRulesReviewer(provider=_BaseGatewayProvider())  # type: ignore[arg-type]
    except ValueError as exc:
        assert "must not consume Base LLM gateway" in str(exc)
    else:
        raise AssertionError("expected ValueError for Base gateway provider")


def test_gateway_rules_reviewer_parses_fenced_content_json() -> None:
    provider = FakeGatewayProvider(
        response=LlmProviderResponse(
            content='```json\n{"verdict":"invalid","reason_codes":["tailors_to_tests"]}\n```'
        )
    )
    reviewer = GatewayRulesReviewer(provider=provider)

    result = reviewer.review(_request())

    assert result.verdict == "invalid"
    assert result.reason_codes == ["tailors_to_tests"]


def test_gateway_rules_reviewer_rejects_unknown_verdict() -> None:
    provider = FakeGatewayProvider(response=_tool_response({"verdict": "definitely-cheating"}))
    reviewer = GatewayRulesReviewer(provider=provider)

    result = reviewer.review(_request())

    assert result.verdict == "suspicious"


def test_gateway_rules_reviewer_clamps_evidence_line_numbers() -> None:
    provider = FakeGatewayProvider(
        response=_tool_response(
            {
                "verdict": "invalid",
                "evidence": [
                    {"path": "agent.py", "line_start": 0, "line_end": 0, "reason_code": ""}
                ],
            }
        )
    )
    reviewer = GatewayRulesReviewer(provider=provider)

    result = reviewer.review(_request())

    assert result.verdict == "invalid"
    assert result.reason_codes == ["rules_reviewer_invalid"]
    assert result.evidence[0].line_start == 1
    assert result.evidence[0].line_end == 1
    assert result.evidence[0].reason_code == "rules_violation"


def test_gateway_rules_reviewer_missing_verdict_is_suspicious() -> None:
    provider = FakeGatewayProvider(response=_tool_response({"reason_codes": ["oops"]}))
    reviewer = GatewayRulesReviewer(provider=provider)

    result = reviewer.review(_request())

    assert result.verdict == "suspicious"
    assert result.reason_codes == ["rules_reviewer_unparseable"]


def test_gateway_rules_reviewer_stops_reading_when_total_budget_exhausted() -> None:
    reviewer = GatewayRulesReviewer(
        provider=FakeGatewayProvider(response=_tool_response({"verdict": "valid"})),
        per_read_max_bytes=8,
        total_read_budget=8,
    )
    request = ReviewerRequest(
        rules_version="rules-test",
        rule_files=[".rules/anti-cheat.md"],
        policy_excerpt="policy",
        workspace_files=["a.py", "b.py"],
        static_findings=[],
        file_contents=[
            WorkspaceFileContent(path="a.py", content="01234567", truncated=False),
            WorkspaceFileContent(path="b.py", content="89abcdef", truncated=False),
        ],
    )

    bounded = reviewer._bounded_file_contents(request)

    # The first file consumes the whole total budget; the second is dropped.
    assert [entry["path"] for entry in bounded] == ["a.py"]


def test_gateway_rules_reviewer_respects_read_budgets() -> None:
    reviewer = GatewayRulesReviewer(
        provider=FakeGatewayProvider(response=_tool_response({"verdict": "valid"})),
        per_read_max_bytes=8,
        total_read_budget=8,
    )
    request = ReviewerRequest(
        rules_version="rules-test",
        rule_files=[".rules/anti-cheat.md"],
        policy_excerpt="policy",
        workspace_files=["agent.py"],
        static_findings=[],
        file_contents=[
            WorkspaceFileContent(path="agent.py", content="0123456789abcdef", truncated=False)
        ],
    )

    bounded = reviewer._bounded_file_contents(request)

    assert bounded[0]["truncated"] is True
    assert len(bounded[0]["content"].encode("utf-8")) <= 8
