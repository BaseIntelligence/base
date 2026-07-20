"""Offline contract tests for the deterministic attested-review policy core."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import httpx
import pytest

from agent_challenge.analyzer.pipeline import run_rules_analyzer
from agent_challenge.review.canonical import canonical_json_v1
from agent_challenge.review.compose import review_build_definition
from agent_challenge.review.openrouter import (
    DirectOpenRouterClient,
    build_openrouter_request_body,
)
from agent_challenge.review.policy import (
    MAX_REVIEW_DECISION_ENTRIES,
    ModelPolicyOutput,
    PolicyFinding,
    ReviewPolicyError,
    ReviewPolicyInput,
    SimilarityFinding,
    parse_model_policy_output,
    verify_review_policy,
)


def _model_response(
    *,
    verdict: str = "allow",
    reason_codes: list[str] | None = None,
    evidence_paths: list[str] | None = None,
    raw_arguments: str | None = None,
) -> bytes:
    return json.dumps(
        {
            "id": "offline-model-response",
            "model": "x-ai/grok-4.5",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "submit_verdict",
                                    "arguments": raw_arguments
                                    or json.dumps(
                                        {
                                            "verdict": verdict,
                                            "reason_codes": reason_codes or [],
                                            "evidence_paths": evidence_paths or [],
                                        },
                                        separators=(",", ":"),
                                    ),
                                },
                            }
                        ],
                    },
                }
            ],
        },
        separators=(",", ":"),
    ).encode()


def _model_output(verdict: str = "allow") -> ModelPolicyOutput:
    return parse_model_policy_output(
        _model_response(verdict=verdict, evidence_paths=["artifact/agent.py"]),
        allowed_evidence_paths={"artifact/agent.py"},
    )


def _finding(source: str, reason_code: str, disposition: str = "reject") -> PolicyFinding:
    return PolicyFinding(
        source=source,
        reason_code=reason_code,
        disposition=disposition,
        evidence_sha256=sha256(f"{source}:{reason_code}".encode()).hexdigest(),
    )


def test_model_output_requires_one_exact_assigned_final_tool() -> None:
    parsed = _model_output()

    assert parsed.verdict == "allow"
    assert parsed.reason_codes == ()
    assert parsed.evidence_paths == ("artifact/agent.py",)
    assert len(parsed.canonical_bytes) > 0


def test_model_output_value_object_rejects_unbound_tampering() -> None:
    parsed = _model_output()

    with pytest.raises(ReviewPolicyError):
        ModelPolicyOutput(
            verdict=parsed.verdict,
            reason_codes=parsed.reason_codes,
            evidence_paths=parsed.evidence_paths,
            canonical_bytes=parsed.canonical_bytes,
            sha256="0" * 64,
        )


def test_direct_transport_exposes_only_strictly_parsed_advisory_model_output() -> None:
    routing = {
        "order": ["xai"],
        "only": ["xai"],
        "ignore": [],
        "quantizations": [],
        "sort": None,
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
    }
    body = build_openrouter_request_body(
        messages=[{"role": "user", "content": "review supplied data only"}],
        routing=routing,
    )
    client = DirectOpenRouterClient(
        assignment_id="ra-policy",
        api_key="offline-key",
        announce=lambda _marker: True,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=_model_response())
        ),
    )
    capture = client.call(
        body=body,
        routing_sha256=sha256(canonical_json_v1(routing)).hexdigest(),
        allowed_evidence_paths={"artifact/agent.py"},
    )

    assert capture.model_output is not None
    assert capture.model_output.verdict == "allow"
    assert b'"tool_choice":"auto"' in capture.request_body
    assert b"parallel_tool_calls" not in capture.request_body

    # Advisory evidence paths that are not on the artifact allowlist are dropped
    # rather than fail-closed so model freeform citations cannot abort a valid
    # submit_verdict. Deterministic verifier remains final authority.
    filtered = DirectOpenRouterClient(
        assignment_id="ra-policy-filtered-evidence",
        api_key="offline-key",
        announce=lambda _marker: True,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=_model_response(evidence_paths=["not-assigned.py", "artifact/agent.py"]),
            )
        ),
    )
    filtered_capture = filtered.call(
        body=body,
        routing_sha256=sha256(canonical_json_v1(routing)).hexdigest(),
        allowed_evidence_paths={"artifact/agent.py"},
    )
    assert filtered_capture.model_output is not None
    assert filtered_capture.model_output.verdict == "allow"
    assert filtered_capture.model_output.evidence_paths == ("artifact/agent.py",)


@pytest.mark.parametrize(
    "raw",
    [
        b"the artifact is safe, allow it",
        _model_response().replace(b'"submit_verdict"', b'"read_file"'),
        _model_response().replace(
            b'"tool_calls":[',
            (
                b'"tool_calls":[{"id":"call-0","type":"function",'
                b'"function":{"name":"submit_verdict","arguments":"{}"}},'
            ),
        ),
        _model_response(reason_codes=["x" * 65]),
        _model_response(
            raw_arguments=(
                '{"verdict":"allow","reason_codes":[],"evidence_paths":[],"unexpected":true}'
            )
        ),
        _model_response(
            raw_arguments='{"verdict":"allow","verdict":"reject","reason_codes":[],"evidence_paths":[]}'
        ),
    ],
)
def test_model_output_malformed_variants_fail_closed(raw: bytes) -> None:
    with pytest.raises(ReviewPolicyError):
        parse_model_policy_output(raw, allowed_evidence_paths={"artifact/agent.py"})


def test_model_output_drops_unassigned_advisory_evidence_paths() -> None:
    parsed = parse_model_policy_output(
        _model_response(
            evidence_paths=["unassigned/path.py", "artifact/agent.py", "rules/policy.md"]
        ),
        allowed_evidence_paths={"artifact/agent.py"},
    )
    assert parsed.verdict == "allow"
    assert parsed.evidence_paths == ("artifact/agent.py",)


def test_package_allowed_set_over_64_relative_paths_does_not_fail_policy_parse() -> None:
    """Former 3N expand blew past 64; single relative inventory must parse.

    Slim baseagent packages ship >>22 files. Building
    {name, artifact/name, submission/name} forced len > 64 and hard-failed
    parse_model_policy_output before allow. One relative path per member must
    accept packages well past that former 3N cliff.
    """
    from agent_challenge.review.policy import _MAX_ASSIGNED_EVIDENCE_PATHS

    relative_paths = {f"pkg/file_{i:03d}.py" for i in range(70)}
    assert len(relative_paths) > 64
    assert len(relative_paths) <= _MAX_ASSIGNED_EVIDENCE_PATHS
    # Former review_runtime expand would be 3*70 = 210 entries.
    former_3n = {
        p for name in relative_paths for p in (name, f"artifact/{name}", f"submission/{name}")
    }
    assert len(former_3n) > 64

    parsed = parse_model_policy_output(
        _model_response(
            evidence_paths=[
                "pkg/file_000.py",
                "artifact/pkg/file_001.py",
                "submission/pkg/file_002.py",
            ]
        ),
        allowed_evidence_paths=relative_paths,
    )
    assert parsed.verdict == "allow"
    assert parsed.evidence_paths == (
        "pkg/file_000.py",
        "pkg/file_001.py",
        "pkg/file_002.py",
    )


def test_former_3n_allowed_set_still_over_assigned_bound_is_allowed_cap() -> None:
    """If callers still build tripled sets past the assigned bound, fail closed."""
    relative_paths = {f"pkg/file_{i:03d}.py" for i in range(70)}
    former_3n = {
        p for name in relative_paths for p in (name, f"artifact/{name}", f"submission/{name}")
    }
    with pytest.raises(ReviewPolicyError, match="too many assigned evidence paths"):
        parse_model_policy_output(
            _model_response(evidence_paths=["pkg/file_000.py"]),
            allowed_evidence_paths=former_3n,
        )


def test_model_legacy_mount_aliases_resolve_to_relative_allowed_paths() -> None:
    parsed = parse_model_policy_output(
        _model_response(evidence_paths=["artifact/agent.py", "submission/src/core.py", "stray.md"]),
        allowed_evidence_paths={"agent.py", "src/core.py"},
    )
    assert parsed.verdict == "allow"
    assert parsed.evidence_paths == ("agent.py", "src/core.py")


@pytest.mark.parametrize(
    ("model_verdict", "expected"),
    [
        ("allow", "allow"),
        ("escalate", "escalate"),
        ("reject", "reject"),
    ],
)
def test_model_verdict_is_a_monotonic_advisory_input(
    model_verdict: str,
    expected: str,
) -> None:
    decision = verify_review_policy(ReviewPolicyInput(model_output=_model_output(model_verdict)))

    assert decision.verdict == expected
    if model_verdict == "allow":
        assert decision.reason_codes == ("policy_passed",)
    else:
        assert decision.reason_codes == (f"model_{model_verdict}",)


@pytest.mark.parametrize("model_verdict", ["allow", "escalate", "reject"])
def test_deterministic_reject_cannot_be_bypassed_by_any_model_verdict(model_verdict: str) -> None:
    decision = verify_review_policy(
        ReviewPolicyInput(
            static_findings=(_finding("static", "reads_hidden_tests"),),
            model_output=_model_output(model_verdict),
        )
    )

    assert decision.verdict == "reject"
    assert "reads_hidden_tests" in decision.reason_codes


def test_malformed_model_output_never_allows_and_static_reject_still_wins() -> None:
    clean = verify_review_policy(ReviewPolicyInput(model_output=None))
    blocked = verify_review_policy(
        ReviewPolicyInput(
            static_findings=(_finding("static", "hardcoded_llm_model"),),
            model_output=None,
        )
    )

    assert clean.verdict == "escalate"
    assert clean.reason_codes == ("model_output_malformed",)
    assert blocked.verdict == "reject"
    assert "hardcoded_llm_model" in blocked.reason_codes


def test_dynamic_prompt_and_similarity_findings_have_deterministic_precedence() -> None:
    escalation = verify_review_policy(
        ReviewPolicyInput(
            dynamic_rule_findings=(
                _finding("dynamic_rule", "ambiguous_security_rule", "escalate"),
            ),
            prompt_findings=(_finding("prompt", "prompt_policy_override", "escalate"),),
            similarity_findings=(
                SimilarityFinding(
                    risk_band="high",
                    evidence_sha256=sha256(b"full-non-base-clone").hexdigest(),
                ),
            ),
            model_output=_model_output("allow"),
        )
    )
    rejection = verify_review_policy(
        ReviewPolicyInput(
            dynamic_rule_findings=(_finding("dynamic_rule", "rules_invalid"),),
            model_output=_model_output("allow"),
        )
    )

    assert escalation.verdict == "escalate"
    assert escalation.reason_codes == (
        "ambiguous_security_rule",
        "prompt_policy_override",
        "similarity_high_risk",
    )
    assert rejection.verdict == "reject"
    assert rejection.reason_codes == ("rules_invalid",)


def test_existing_cheat_discriminators_remain_rejected_and_benign_controls_allow(
    tmp_path: Path,
) -> None:
    # VAL-ACAT-015: measured OpenRouter host alone is legal; use non-measured
    # providers / Base gateway cheat classes for residual unauthorized/model flags.
    cheat_sources = {
        "reads_hidden_tests": (
            "def solve():\n"
            "    with open('/app/tests/test_outputs.py') as handle:\n"
            "        return handle.read()\n"
        ),
        "branches_on_task_identity": (
            "def solve(task_id):\n"
            "    if task_id == 'hidden-benchmark-task':\n"
            "        return 'precomputed'\n"
        ),
        "unauthorized_llm_provider": (
            "import os\n"
            "API_KEY = os.environ['ANTHROPIC_API_KEY']\n"
            "BASE_URL = 'https://api.anthropic.com'\n"
        ),
        "hardcoded_llm_model": "MODEL = 'deepseek-v4-flash'\n",
    }
    for expected_reason, source in cheat_sources.items():
        workspace = tmp_path / expected_reason
        workspace.mkdir()
        (workspace / "agent.py").write_text(source, encoding="utf-8")
        report = run_rules_analyzer(workspace)
        findings = tuple(
            PolicyFinding(
                source="static",
                reason_code=finding.reason_code,
                disposition="reject",
                evidence_sha256=sha256(
                    f"{finding.path}:{finding.line_start}:{finding.snippet}".encode()
                ).hexdigest(),
            )
            for finding in report.hardcoding_findings
        )
        decision = verify_review_policy(
            ReviewPolicyInput(static_findings=findings, model_output=_model_output("allow"))
        )
        assert decision.verdict == "reject"
        assert expected_reason in decision.reason_codes

    benign = tmp_path / "benign"
    benign.mkdir()
    (benign / "agent.py").write_text(
        "def solve(value):\n"
        "    prompt = 'Complete the requested task carefully.'\n"
        "    return value + 1\n",
        encoding="utf-8",
    )
    report = run_rules_analyzer(benign)
    decision = verify_review_policy(
        ReviewPolicyInput(
            static_findings=(),
            similarity_findings=(
                SimilarityFinding(
                    risk_band="low",
                    evidence_sha256=sha256(b"shared-base-only").hexdigest(),
                ),
            ),
            model_output=_model_output("allow"),
        )
    )
    assert report.hardcoding_findings == []
    assert decision.verdict == "allow"


def test_adversarial_submitted_source_is_scanned_as_data_never_executed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = tmp_path / "submitted-source-was-executed"
    (workspace / "agent.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "def solve():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    report = run_rules_analyzer(workspace)
    decision = verify_review_policy(ReviewPolicyInput(model_output=_model_output("allow")))

    assert marker.exists() is False
    assert report.hardcoding_findings == []
    assert decision.verdict == "allow"


def test_verifier_is_byte_deterministic_and_performs_no_external_work() -> None:
    policy_input = ReviewPolicyInput(
        static_findings=(_finding("static", "reads_hidden_tests"),),
        dynamic_rule_findings=(_finding("dynamic_rule", "ambiguous_rule", "escalate"),),
        model_output=_model_output("allow"),
    )

    first = verify_review_policy(policy_input)
    repeated = [verify_review_policy(policy_input) for _ in range(4)]

    assert all(item.canonical_bytes == first.canonical_bytes for item in repeated)
    assert all(item.public_projection() == first.public_projection() for item in repeated)
    assert first.verdict == "reject"


def test_verifier_is_cross_process_deterministic_and_network_free(monkeypatch) -> None:
    attempted_network: list[tuple[object, ...]] = []

    def network_forbidden(*args: object, **kwargs: object) -> None:
        attempted_network.append((*args, *kwargs.values()))
        raise AssertionError("deterministic verifier attempted network I/O")

    monkeypatch.setattr("socket.create_connection", network_forbidden)
    policy_input = ReviewPolicyInput(
        static_findings=(_finding("static", "reads_hidden_tests"),),
        model_output=_model_output("allow"),
    )
    current = verify_review_policy(policy_input).canonical_bytes.hex()
    repository = Path(__file__).resolve().parents[1]
    script = """
from hashlib import sha256
from agent_challenge.review.policy import (
    ModelPolicyOutput,
    PolicyFinding,
    ReviewPolicyInput,
    verify_review_policy,
)
model_bytes = (
    b'{"evidence_paths":["artifact/agent.py"],"reason_codes":[],'
    b'"schema_version":1,"verdict":"allow"}'
)
model = ModelPolicyOutput(
    verdict="allow",
    reason_codes=(),
    evidence_paths=("artifact/agent.py",),
    canonical_bytes=model_bytes,
    sha256=sha256(model_bytes).hexdigest(),
)
finding = PolicyFinding(
    source="static",
    reason_code="reads_hidden_tests",
    disposition="reject",
    evidence_sha256=sha256(b"static:reads_hidden_tests").hexdigest(),
)
decision = verify_review_policy(
    ReviewPolicyInput(static_findings=(finding,), model_output=model)
)
print(decision.canonical_bytes.hex())
"""
    child = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        text=True,
    )

    assert child.stdout.strip() == current
    assert attempted_network == []


def test_review_runtime_invokes_verify_review_policy_with_all_finding_sources() -> None:
    """Production runtime must construct ReviewPolicyInput and call the verifier.

    Model allow remains advisory: static reject detours final verdict to reject.
    Offline fixture only — no real OpenRouter or TDX transport is claimed.
    """

    runtime_path = review_build_definition().dockerfile.parent / "review_runtime.py"
    source = runtime_path.read_text(encoding="utf-8")
    assert "verify_review_policy" in source
    assert "ReviewPolicyInput" in source
    assert "run_review_policy" in source or "apply_review_policy" in source

    spec = importlib.util.spec_from_file_location("review_runtime_policy", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    assert hasattr(runtime, "run_review_policy")
    assert MAX_REVIEW_DECISION_ENTRIES == 256

    static = (
        PolicyFinding(
            source="static",
            reason_code="reads_hidden_tests",
            disposition="reject",
            evidence_sha256=sha256(b"static-ev").hexdigest(),
        ),
    )
    similarity = (
        SimilarityFinding(
            risk_band="low",
            evidence_sha256=sha256(b"sim-ev").hexdigest(),
        ),
    )
    dynamic = (
        PolicyFinding(
            source="dynamic_rule",
            reason_code="ambiguous_security_rule",
            disposition="escalate",
            evidence_sha256=sha256(b"dyn-ev").hexdigest(),
        ),
    )
    prompt = (
        PolicyFinding(
            source="prompt",
            reason_code="prompt_policy_override",
            disposition="escalate",
            evidence_sha256=sha256(b"prompt-ev").hexdigest(),
        ),
    )
    advisory = _model_output("allow")

    result = runtime.run_review_policy(
        static_findings=static,
        similarity_findings=similarity,
        dynamic_rule_findings=dynamic,
        prompt_findings=prompt,
        model_output=advisory,
    )
    assert result["verdict"] == "reject"
    assert "reads_hidden_tests" in result["reason_codes"]
    assert result["verifier_result"] == "reject"
    assert len(result["reason_codes"]) <= 256
    assert len(result["evidence_digests"]) <= 256
    # Runnable path never mutates model allow into the gate opening.
    clean = runtime.run_review_policy(model_output=_model_output("allow"))
    assert clean["verdict"] == "allow"
    assert clean["verifier_result"] == "pass"
    # Malformed model stays escalate; never becomes allow.
    malformed = runtime.run_review_policy(model_output=None)
    assert malformed["verdict"] == "escalate"


def test_aggregate_reason_and_evidence_entries_never_exceed_256() -> None:
    """Architecture bound: combined reason or evidence decision lists ≤ 256 entries."""

    assert MAX_REVIEW_DECISION_ENTRIES == 256

    # 255 static reject findings + advisory model allow:
    # reasons = 255, digests = 255 findings + model digest = 256 → admitted at the cap.
    findings_255 = tuple(
        PolicyFinding(
            source="static",
            reason_code=f"r{i:03d}",
            disposition="reject",
            evidence_sha256=sha256(f"ev-{i}".encode()).hexdigest(),
        )
        for i in range(255)
    )
    at_cap = verify_review_policy(
        ReviewPolicyInput(static_findings=findings_255, model_output=_model_output("allow"))
    )
    assert at_cap.verdict == "reject"
    assert len(at_cap.reason_codes) == 255
    assert len(at_cap.evidence_digests) == 256

    # Adding one more finding evidence digest (256 findings + model) exceeds the evidence cap.
    findings_256 = findings_255 + (
        PolicyFinding(
            source="static",
            reason_code="r255",
            disposition="reject",
            evidence_sha256=sha256(b"ev-255").hexdigest(),
        ),
    )
    with pytest.raises(ReviewPolicyError, match="256|bound|aggregate"):
        verify_review_policy(
            ReviewPolicyInput(
                static_findings=findings_256,
                model_output=_model_output("allow"),
            )
        )

    # Per-source input stays within 256; cross-source reason union may still exceed 256.
    static_flood = tuple(
        PolicyFinding(
            source="static",
            reason_code=f"s{i:03d}",
            disposition="reject",
            evidence_sha256=sha256(b"shared-reason-flood").hexdigest(),
        )
        for i in range(200)
    )
    dynamic_flood = tuple(
        PolicyFinding(
            source="dynamic_rule",
            reason_code=f"d{i:03d}",
            disposition="reject",
            evidence_sha256=sha256(b"shared-reason-flood").hexdigest(),
        )
        for i in range(57)
    )
    with pytest.raises(ReviewPolicyError, match="256|bound|aggregate"):
        verify_review_policy(
            ReviewPolicyInput(
                static_findings=static_flood,
                dynamic_rule_findings=dynamic_flood,
                model_output=_model_output("allow"),
            )
        )
