from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from agent_challenge.analyzer import base_skeleton
from agent_challenge.analyzer.llm_reviewer import (
    GATEWAY_PLACEHOLDER_MODEL,
    GatewayReviewProvider,
    KimiLlmReviewer,
    LlmProviderResponse,
    LlmProviderUnavailable,
    LlmToolCall,
    _initial_messages,
    _retry_message,
    _tool_schemas,
)
from agent_challenge.submissions.artifacts import ArtifactReadSession, store_zip_bytes

ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"


def agent_source(contents: str | bytes) -> str | bytes:
    if isinstance(contents, bytes):
        return contents
    if "class Agent" in contents:
        return contents
    return f"{ENTRYPOINT_SOURCE}\n{contents}"


class MockProvider:
    provider_name = "mock"
    model_name = GATEWAY_PLACEHOLDER_MODEL

    def __init__(self, responses: Sequence[LlmProviderResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str | Mapping[str, Any],
        timeout_seconds: int,
    ) -> LlmProviderResponse:
        self.requests.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "tool_choice": tool_choice,
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.responses.pop(0)


def test_gateway_provider_parses_legacy_function_call(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "function_call": {
                                "name": "submit_verdict",
                                "arguments": json.dumps(
                                    {
                                        "verdict": "allow",
                                        "confidence": 0.8,
                                        "rationale": "No issue found.",
                                    }
                                ),
                            },
                        }
                    }
                ]
            }

    def fake_post(url, *, headers, json, timeout):  # noqa: A002 - mirror httpx.post
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return Response()

    monkeypatch.setattr("agent_challenge.analyzer.llm_reviewer.httpx.post", fake_post)
    provider = GatewayReviewProvider(gateway_token="scoped-token", base_url="http://master/llm/v1")

    response = provider.complete(
        messages=[],
        tools=[],
        tool_choice={"type": "function", "function": {"name": "submit_verdict"}},
        timeout_seconds=1,
    )

    # POSTs to the gateway /llm/v1 route with the scoped token, no provider key.
    assert captured["url"] == "http://master/llm/v1/chat/completions"
    assert captured["headers"]["X-Gateway-Token"] == "scoped-token"
    assert "Authorization" not in captured["headers"]
    # A placeholder model is sent; the gateway overwrites it from the token source.
    assert captured["json"]["model"] == GATEWAY_PLACEHOLDER_MODEL
    assert captured["json"]["parallel_tool_calls"] is False
    assert response.tool_calls[0].name == "submit_verdict"
    assert response.tool_calls[0].arguments["verdict"] == "allow"


def test_mock_provider_allow_constructs_auditable_llm_verdict(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    LlmToolCall(
                        id="read-1",
                        name="read_file",
                        arguments={"path": "agent.py", "offset": 0, "limit": 24},
                    ),
                ),
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            ),
            LlmProviderResponse(
                tool_calls=(
                    _submit_call(
                        "allow",
                        confidence=0.91,
                        rationale="No policy issue found.",
                        evidence_paths=["agent.py"],
                    ),
                ),
                cost={"total_cost": 0.001},
            ),
        ]
    )

    outcome = KimiLlmReviewer(provider=provider).review(
        analysis_run_id=42,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
        similarity_evidence=[{"risk_band": "low", "score_percent": 12.5}],
    )

    row = outcome.llm_verdict_row
    assert outcome.verdict.verdict == "allow"
    assert row.analysis_run_id == 42
    assert row.verdict == "allow"
    assert row.model_name == GATEWAY_PLACEHOLDER_MODEL
    assert row.prompt_ref == "llm-reviewer-manifest-tools-v1"
    request = json.loads(row.raw_request_json)
    response = json.loads(row.raw_response_json)
    assert request["manifest"]["entries"][0]["path"] == "agent.py"
    assert response["tool_calls"][0]["content_sha256"]
    assert "def solve" not in row.raw_response_json
    assert provider.requests[0]["messages"][1]["content"].find("agent.py") >= 0


def test_reviewer_prompt_distinguishes_prompt_templates_from_injection(tmp_path: Path) -> None:
    metadata = _stored_artifact(
        tmp_path,
        {
            "prompt-templates/agent.txt": (
                "Task Description:\n{instruction}\n\n"
                "Current terminal state:\n{terminal_state}\n"
                "Before completion, verify the task requirements.\n"
            )
        },
    )
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    _submit_call(
                        "allow",
                        confidence=0.86,
                        rationale="Benign task prompt template without policy bypass.",
                    ),
                )
            )
        ]
    )

    outcome = KimiLlmReviewer(provider=provider).review(
        analysis_run_id=43,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    prompt_payload = json.loads(provider.requests[0]["messages"][1]["content"])
    instructions = " ".join(prompt_payload["instructions"])
    assert outcome.verdict.verdict == "allow"
    assert "ordinary agent prompt templates" in instructions
    assert "concrete bypass or policy-override instruction" in instructions
    delta_files = {item["path"]: item for item in prompt_payload["non_base_delta"]["files"]}
    template = delta_files["prompt-templates/agent.txt"]
    assert template["kind"] == "prompt/config"
    assert "Current terminal state" in template["text"]


def test_reviewer_prompt_includes_non_base_delta_and_config(tmp_path: Path, monkeypatch) -> None:
    metadata = _stored_artifact(
        tmp_path, {"prompt.yaml": "system: You are a careful agent\ntemperature: 0.2\n"}
    )
    # Register agent.py (the base entrypoint) so it is subtracted from the delta.
    agent_entry = next(
        entry for entry in metadata.manifest.entries if entry.normalized_path == "agent.py"
    )
    monkeypatch.setattr(
        "agent_challenge.analyzer.base_skeleton.base_skeleton_fingerprint",
        lambda: base_skeleton.BaseSkeletonFingerprint(
            ast_hashes=frozenset(), file_hashes=frozenset({agent_entry.sha256})
        ),
    )
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    _submit_call(
                        "allow",
                        confidence=0.9,
                        rationale="Only the prompt/config differs from the base; not a clone.",
                    ),
                )
            )
        ]
    )

    outcome = KimiLlmReviewer(provider=provider).review(
        analysis_run_id=201,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    payload = json.loads(provider.requests[0]["messages"][1]["content"])
    delta_files = {item["path"]: item for item in payload["non_base_delta"]["files"]}
    assert "agent.py" not in delta_files  # base skeleton subtracted
    assert delta_files["prompt.yaml"]["kind"] == "prompt/config"
    assert "You are a careful agent" in delta_files["prompt.yaml"]["text"]
    assert outcome.verdict.verdict == "allow"


def test_reviewer_instructions_allow_shared_base(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})

    messages = _initial_messages(metadata.manifest, [])
    instructions = " ".join(json.loads(messages[1]["content"])["instructions"])

    assert "Shared similarity to the common baseagent skeleton is EXPECTED" in instructions
    assert "FULL clone" in instructions
    assert "prefer allow" in instructions
    assert "DELTA versus the baseagent skeleton" in instructions


def test_mock_provider_reject_verdict_is_preserved(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "TASK_ID = 'known'\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    _submit_call(
                        "reject",
                        confidence=0.88,
                        rationale="Benchmark-specific constant detected.",
                        evidence_paths=["agent.py"],
                        policy_flags=["benchmark_task_id_literal"],
                    ),
                )
            )
        ]
    )

    outcome = KimiLlmReviewer(provider=provider).review(
        analysis_run_id=7,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    row = outcome.llm_verdict_row
    assert outcome.verdict.verdict == "reject"
    assert json.loads(row.reason_codes_json) == ["benchmark_task_id_literal"]


def test_mock_provider_escalate_verdict_is_preserved(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    _submit_call(
                        "escalate",
                        confidence=0.35,
                        rationale="Similarity evidence needs human review.",
                        evidence_paths=["agent.py"],
                        policy_flags=["similarity_high"],
                    ),
                )
            )
        ]
    )

    outcome = KimiLlmReviewer(provider=provider).review(
        analysis_run_id=8,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
        similarity_evidence=[{"risk_band": "high", "score_percent": 98}],
    )

    row = outcome.llm_verdict_row
    assert outcome.verdict.verdict == "escalate"
    assert outcome.verdict.evidence_paths == ["agent.py"]
    assert outcome.verdict.policy_flags == ["similarity_high"]
    assert row.verdict == "escalate"
    assert json.loads(row.reason_codes_json) == ["similarity_high"]


def test_incomplete_empty_escalate_verdict_retries_then_fails_closed(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"terminus_kira.py": "def solve():\n    return 1\n"})
    incomplete_call = _submit_call(
        "escalate",
        confidence=0.8,
        rationale="I still need to review the full terminus_kira.py file before finalizing.",
    )
    provider = MockProvider(
        [
            LlmProviderResponse(tool_calls=(incomplete_call,)),
            LlmProviderResponse(tool_calls=(incomplete_call,)),
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=2).review(
        analysis_run_id=9,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    row = outcome.llm_verdict_row
    response = json.loads(row.raw_response_json)
    assert outcome.verdict.verdict == "escalate"
    assert response["fail_closed_reason"] == "incomplete_submit_verdict"
    assert json.loads(row.reason_codes_json) == ["incomplete_submit_verdict"]
    assert len(provider.requests) == 2
    first_event = outcome.transcript["attempts"][0]["events"][0]
    assert first_event["reason_code"] == "incomplete_submit_verdict"


def test_disallowed_path_and_tool_violation_fail_closed(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    LlmToolCall(
                        id="read-bad",
                        name="read_file",
                        arguments={"path": "../agent.py", "offset": 0, "limit": 10},
                    ),
                )
            ),
            LlmProviderResponse(
                tool_calls=(
                    LlmToolCall(id="shell-1", name="run_shell", arguments={"command": "ls"}),
                )
            ),
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=2).review(
        analysis_run_id=10,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    row = outcome.llm_verdict_row
    response = json.loads(row.raw_response_json)
    assert outcome.verdict.verdict == "escalate"
    assert response["tool_calls"][0]["error_code"] == "unsafe_path"
    assert response["fail_closed_reason"] == "disallowed_tool"
    # disallowed_tool is a recoverable tool-miss: the reviewer reports it as a
    # fail_closed disposition so the lifecycle routes it to standby (Work Item 5),
    # never a terminal model escalate.
    assert outcome.disposition == "fail_closed"
    assert outcome.fail_closed_reason == "disallowed_tool"


def test_gateway_provider_is_inert_without_token() -> None:
    provider = GatewayReviewProvider(gateway_token=None, base_url="http://master/llm/v1")

    try:
        provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=1)
    except LlmProviderUnavailable as exc:
        assert "gateway token" in str(exc)
    else:
        raise AssertionError("provider should require a gateway token")


def test_minimal_final_submit_verdict_is_accepted_and_defaults_are_applied(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    LlmToolCall(
                        id="final-minimal",
                        name="submit_verdict",
                        arguments={
                            "verdict": "allow",
                            "confidence": 0.77,
                            "rationale": "Artifact is acceptable.",
                        },
                    ),
                )
            )
        ]
    )

    outcome = KimiLlmReviewer(provider=provider).review(
        analysis_run_id=11,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    assert outcome.verdict.verdict == "allow"
    assert outcome.verdict.evidence_paths == []
    assert outcome.verdict.similarity_assessment == ""
    assert outcome.verdict.policy_flags == []
    assert outcome.llm_verdict_row.verdict == "allow"


def test_final_attempt_forces_submit_verdict_tool_choice_after_non_final_call(
    tmp_path: Path,
) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    LlmToolCall(
                        id="read-1",
                        name="read_file",
                        arguments={"path": "agent.py", "offset": 0, "limit": 10},
                    ),
                    _submit_call(
                        "allow",
                        confidence=0.82,
                        rationale="This verdict was not final.",
                    ),
                )
            ),
            LlmProviderResponse(
                tool_calls=(
                    _submit_call(
                        "allow",
                        confidence=0.83,
                        rationale="Final single verdict.",
                    ),
                )
            ),
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=2).review(
        analysis_run_id=12,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    assert outcome.verdict.verdict == "allow"
    assert provider.requests[0]["tool_choice"] == "auto"
    assert provider.requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_verdict"},
    }
    first_attempt_events = outcome.transcript["attempts"][0]["events"]
    assert first_attempt_events[0]["reason_code"] == "submit_verdict_not_final"


def test_non_final_submit_verdict_fails_closed_when_no_valid_final_call(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    LlmToolCall(
                        id="read-1",
                        name="read_file",
                        arguments={"path": "agent.py", "offset": 0, "limit": 10},
                    ),
                    _submit_call(
                        "allow",
                        confidence=0.8,
                        rationale="This verdict was not final.",
                    ),
                )
            )
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=1).review(
        analysis_run_id=13,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    row = outcome.llm_verdict_row
    response = json.loads(row.raw_response_json)
    assert outcome.verdict.verdict == "escalate"
    assert response["fail_closed_reason"] == "submit_verdict_not_final"
    assert "submit_verdict_not_final" in json.loads(row.reason_codes_json)


def test_malformed_submit_verdict_fails_closed_when_no_valid_final_call(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    LlmToolCall(
                        id="bad-final",
                        name="submit_verdict",
                        # Out-of-range confidence is now clamped, so use a genuine
                        # malformation (invalid verdict enum) to exercise the path.
                        arguments={"rationale": "Looks fine.", "verdict": "maybe"},
                    ),
                )
            )
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=1).review(
        analysis_run_id=14,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    row = outcome.llm_verdict_row
    response = json.loads(row.raw_response_json)
    assert outcome.verdict.verdict == "escalate"
    assert response["fail_closed_reason"] == "malformed_submit_verdict"
    assert "malformed_submit_verdict" in json.loads(row.reason_codes_json)


def test_no_valid_final_submit_verdict_still_fails_closed(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(content="plain text without tool"),
            LlmProviderResponse(content="still no tool"),
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=2).review(
        analysis_run_id=15,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    row = outcome.llm_verdict_row
    response = json.loads(row.raw_response_json)
    assert outcome.verdict.verdict == "escalate"
    assert response["fail_closed_reason"] == "missing_tool_call"
    assert "missing_tool_call" in json.loads(row.reason_codes_json)
    assert provider.requests[0]["tool_choice"] == "auto"
    assert provider.requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_verdict"},
    }


def test_forced_final_attempt_accepts_strict_json_content_verdict(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                content=json.dumps(
                    {
                        "verdict": "allow",
                        "confidence": 0.81,
                        "rationale": "No policy issue found.",
                    }
                )
            )
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=1).review(
        analysis_run_id=16,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    assert outcome.verdict.verdict == "allow"
    assert outcome.llm_verdict_row.verdict == "allow"
    events = outcome.transcript["attempts"][0]["events"]
    assert events[0]["event"] == "content_submit_verdict_fallback"


def test_content_verdict_fallback_only_when_submit_verdict_is_forced(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                content=json.dumps(
                    {
                        "verdict": "allow",
                        "confidence": 0.81,
                        "rationale": "No policy issue found.",
                    }
                )
            ),
            LlmProviderResponse(content="still no final tool"),
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=2).review(
        analysis_run_id=17,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    response = json.loads(outcome.llm_verdict_row.raw_response_json)
    assert outcome.verdict.verdict == "escalate"
    assert response["fail_closed_reason"] == "missing_tool_call"
    assert provider.requests[0]["tool_choice"] == "auto"


def test_content_verdict_fallback_rejects_prose_and_extra_fields(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(content="I think this artifact is safe."),
            LlmProviderResponse(
                content=json.dumps(
                    {
                        "verdict": "allow",
                        "confidence": 0.81,
                        "rationale": "No policy issue found.",
                        "unexpected": True,
                    }
                )
            ),
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=2).review(
        analysis_run_id=18,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    response = json.loads(outcome.llm_verdict_row.raw_response_json)
    assert outcome.verdict.verdict == "escalate"
    assert response["fail_closed_reason"] == "missing_tool_call"


def test_submit_verdict_schema_lists_reason_before_verdict() -> None:
    schemas = {tool["function"]["name"]: tool["function"] for tool in _tool_schemas()}
    submit = schemas["submit_verdict"]["parameters"]

    assert submit["required"] == ["rationale", "verdict"]
    assert list(submit["properties"])[0] == "rationale"
    assert "reasoning FIRST" in submit["properties"]["rationale"]["description"]
    assert submit["properties"]["verdict"]["description"] == "allow|reject|escalate"


def test_read_file_schema_makes_offset_and_limit_optional() -> None:
    schemas = {tool["function"]["name"]: tool["function"] for tool in _tool_schemas()}
    read_file = schemas["read_file"]["parameters"]

    assert read_file["required"] == ["path"]
    assert read_file["additionalProperties"] is False
    assert set(read_file["properties"]) == {"path", "offset", "limit"}


def test_read_file_defaults_offset_and_limit_when_absent(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    LlmToolCall(id="read-1", name="read_file", arguments={"path": "agent.py"}),
                )
            ),
            LlmProviderResponse(
                tool_calls=(_submit_call("allow", confidence=0.9, rationale="Read the file."),)
            ),
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=2).review(
        analysis_run_id=101,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    assert outcome.verdict.verdict == "allow"
    read_metadata = outcome.transcript["tool_calls"][0]
    assert read_metadata["ok"] is True
    assert read_metadata["offset"] == 0
    assert read_metadata["limit"] > 0


def test_confidence_out_of_range_is_clamped_not_rejected(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    LlmToolCall(
                        id="final",
                        name="submit_verdict",
                        arguments={
                            "rationale": "All good.",
                            "verdict": "allow",
                            "confidence": 2.0,
                        },
                    ),
                )
            )
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=1).review(
        analysis_run_id=102,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    assert outcome.verdict.verdict == "allow"
    assert outcome.verdict.confidence == 1.0
    assert outcome.disposition == "verdict"


def test_forced_final_submit_accepted_despite_prior_failed_read(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(
                    LlmToolCall(
                        id="read-miss",
                        name="read_file",
                        arguments={"path": "nonexistent.py"},
                    ),
                )
            ),
            LlmProviderResponse(
                tool_calls=(_submit_call("allow", confidence=0.9, rationale="Reviewed via list."),)
            ),
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=2).review(
        analysis_run_id=103,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    assert outcome.verdict.verdict == "allow"
    assert outcome.disposition == "verdict"
    assert outcome.transcript["tool_calls"][0]["ok"] is False
    assert outcome.transcript["tool_calls"][0]["error_code"] == "unknown_path"


def test_reads_without_submit_on_final_attempt_reports_no_submit_after_reads(
    tmp_path: Path,
) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    read_call = LlmToolCall(id="r", name="read_file", arguments={"path": "agent.py"})
    provider = MockProvider(
        [
            LlmProviderResponse(tool_calls=(read_call,)),
            LlmProviderResponse(tool_calls=(read_call,)),
        ]
    )

    outcome = KimiLlmReviewer(provider=provider, max_attempts=2).review(
        analysis_run_id=105,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    assert outcome.disposition == "fail_closed"
    assert outcome.fail_closed_reason == "no_submit_after_reads"
    response = json.loads(outcome.llm_verdict_row.raw_response_json)
    assert response["fail_closed_reason"] == "no_submit_after_reads"


def test_retry_message_names_allowed_tools_and_forbids_others() -> None:
    content = _retry_message("disallowed_tool")["content"]

    assert "read_file" in content
    assert "submit_verdict" in content
    assert "no shell" in content.lower()


def test_initial_messages_declare_only_two_tools_and_no_shell(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})

    messages = _initial_messages(metadata.manifest, [])
    instructions = " ".join(json.loads(messages[1]["content"])["instructions"])

    assert "ONLY available tools are read_file and submit_verdict" in instructions
    assert "NO shell" in instructions


def test_parses_anthropic_tool_use_content_blocks(monkeypatch) -> None:
    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {
                "model": "claude-opus-4-8",
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "done"},
                                {
                                    "type": "tool_use",
                                    "id": "tu-1",
                                    "name": "submit_verdict",
                                    "input": {"rationale": "No issue.", "verdict": "allow"},
                                },
                            ]
                        }
                    }
                ],
            }

    def fake_post(url, *, headers, json, timeout):  # noqa: A002 - mirror httpx.post
        return Response()

    monkeypatch.setattr("agent_challenge.analyzer.llm_reviewer.httpx.post", fake_post)
    provider = GatewayReviewProvider(gateway_token="scoped", base_url="http://master/llm/v1")

    response = provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=1)

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "submit_verdict"
    assert response.tool_calls[0].arguments["verdict"] == "allow"


def test_empty_tool_name_is_ignored_not_disallowed(monkeypatch) -> None:
    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "   ", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            }

    def fake_post(url, *, headers, json, timeout):  # noqa: A002 - mirror httpx.post
        return Response()

    monkeypatch.setattr("agent_challenge.analyzer.llm_reviewer.httpx.post", fake_post)
    provider = GatewayReviewProvider(gateway_token="scoped", base_url="http://master/llm/v1")

    response = provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=1)

    assert response.tool_calls == ()


def test_review_flags_unexpected_model(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})
    provider = MockProvider(
        [
            LlmProviderResponse(
                tool_calls=(_submit_call("allow", confidence=0.9, rationale="Fine."),),
                raw_response={"model": "gpt-4o"},
            )
        ]
    )

    outcome = KimiLlmReviewer(
        provider=provider, max_attempts=1, expected_model="claude-opus-4-8"
    ).review(
        analysis_run_id=104,
        manifest=metadata.manifest,
        read_session=ArtifactReadSession.from_artifact_metadata(metadata),
    )

    assert outcome.verdict.verdict == "allow"
    assert outcome.transcript["unexpected_model"] is True
    assert json.loads(outcome.llm_verdict_row.raw_response_json)["unexpected_model"] is True


def test_expected_model_is_not_sent_on_the_wire(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"model": "claude-opus-4-8", "choices": [{"message": {"content": ""}}]}

    def fake_post(url, *, headers, json, timeout):  # noqa: A002 - mirror httpx.post
        captured["json"] = json
        return Response()

    monkeypatch.setattr("agent_challenge.analyzer.llm_reviewer.httpx.post", fake_post)
    provider = GatewayReviewProvider(gateway_token="scoped", base_url="http://master/llm/v1")

    provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=1)

    assert captured["json"]["model"] == GATEWAY_PLACEHOLDER_MODEL


def test_prompt_cache_blocks_present_when_enabled(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})

    messages = _initial_messages(metadata.manifest, [], prompt_cache_enabled=True)

    assert isinstance(messages[0]["content"], list)
    user_parts = messages[1]["content"]
    assert isinstance(user_parts, list)
    assert user_parts[-1]["cache_control"] == {"type": "ephemeral"}
    assert "agent.py" in user_parts[-1]["text"]


def test_no_cache_control_when_prompt_cache_disabled(tmp_path: Path) -> None:
    metadata = _stored_artifact(tmp_path, {"agent.py": "def solve():\n    return 1\n"})

    messages = _initial_messages(metadata.manifest, [], prompt_cache_enabled=False)

    assert isinstance(messages[0]["content"], str)
    assert isinstance(messages[1]["content"], str)
    assert "cache_control" not in json.dumps(messages)


def test_gateway_provider_uses_httpx_timeout_object(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": ""}}]}

    def fake_post(url, *, headers, json, timeout):  # noqa: A002 - mirror httpx.post
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("agent_challenge.analyzer.llm_reviewer.httpx.post", fake_post)
    provider = GatewayReviewProvider(gateway_token="scoped", base_url="http://master/llm/v1")

    provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=240)

    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 240
    assert timeout.connect == 10


def _submit_call(
    verdict: str,
    *,
    confidence: float,
    rationale: str,
    evidence_paths: list[str] | None = None,
    policy_flags: list[str] | None = None,
) -> LlmToolCall:
    return LlmToolCall(
        id="final-1",
        name="submit_verdict",
        arguments={
            "verdict": verdict,
            "confidence": confidence,
            "rationale": rationale,
            "evidence_paths": evidence_paths or [],
            "similarity_assessment": "",
            "policy_flags": policy_flags or [],
        },
    )


def _stored_artifact(tmp_path: Path, entries: dict[str, str | bytes]):
    return store_zip_bytes(zip_bytes=_zip_bytes(entries), artifact_root=str(tmp_path))


def _zip_bytes(entries: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    archive_entries = {"agent.py": ENTRYPOINT_SOURCE, **entries}
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in archive_entries.items():
            if filename == "agent.py":
                contents = agent_source(contents)
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()
