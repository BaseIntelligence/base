"""VAL-ACAT-013/014/015: gateway-free eval secrets, policy, static analyzer matrix.

Product Mode B: Base LLM gateway is not a required secret and not a legal
agent LLM path. Measured OpenRouter under harness is not auto-cheated solely
for avoiding Base gateway. GatewayRulesReviewer must not consume Base gateway.
"""

from __future__ import annotations

from pathlib import Path

from agent_challenge.analyzer.lifecycle import (
    build_configured_rules_reviewer,
    gateway_llm_base_url,
)
from agent_challenge.analyzer.pipeline import run_rules_analyzer
from agent_challenge.canonical.compose import DEFAULT_ALLOWED_ENVS
from agent_challenge.evaluation.gateway import agent_gateway_config_from_settings
from agent_challenge.review.sessions import create_review_session
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.selfdeploy.eval import (
    EVAL_REQUIRED_SECRET_ENVS,
)

FORBIDDEN_GATEWAY_ENV_NAMES = frozenset(
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


class _SilentReviewer:
    def review(self, request):  # noqa: ANN001, ARG002
        from agent_challenge.analyzer.schemas import ReviewerResult

        return ReviewerResult(verdict="valid", reason_codes=["rules_passed"], notes="ok")


def test_eval_required_secrets_exclude_base_gateway() -> None:
    """VAL-ACAT-013: EVAL_REQUIRED_SECRET_ENVS must not mandate Base gateway vars."""

    assert "BASE_GATEWAY_TOKEN" not in EVAL_REQUIRED_SECRET_ENVS
    assert "BASE_LLM_GATEWAY_URL" not in EVAL_REQUIRED_SECRET_ENVS
    for name in FORBIDDEN_GATEWAY_ENV_NAMES:
        assert name not in EVAL_REQUIRED_SECRET_ENVS
    # Still require the attested run capability material.
    assert "EVAL_RUN_TOKEN" in EVAL_REQUIRED_SECRET_ENVS
    assert "CHALLENGE_PHALA_EVAL_PLAN" in EVAL_REQUIRED_SECRET_ENVS


def test_default_allowed_envs_exclude_base_gateway_vars() -> None:
    """VAL-ACAT-013: compose encrypted_env allowlist omits Base gateway names."""

    allowed = set(DEFAULT_ALLOWED_ENVS)
    for name in ("BASE_GATEWAY_TOKEN", "BASE_LLM_GATEWAY_URL"):
        assert name not in allowed


def test_settings_instantiate_without_gateway_vars(monkeypatch) -> None:
    """VAL-ACAT-013: unit surfaces start healthy with gateway env unset."""

    for name in (
        "CHALLENGE_LLM_GATEWAY_BASE_URL",
        "CHALLENGE_LLM_GATEWAY_TOKEN",
        "CHALLENGE_LLM_GATEWAY_TOKEN_FILE",
        "CHALLENGE_AGENT_GATEWAY_TOKEN",
        "CHALLENGE_AGENT_GATEWAY_TOKEN_FILE",
        "BASE_GATEWAY_TOKEN",
        "BASE_LLM_GATEWAY_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = ChallengeSettings()
    assert settings.llm_gateway_base_url is None
    assert settings.llm_gateway_token is None
    assert settings.agent_gateway_token is None
    # Residual gateway config must never inject agent sandbox env.
    assert agent_gateway_config_from_settings(settings) is None


def test_agent_gateway_config_never_injects_even_when_residual_url_set() -> None:
    """VAL-ACAT-013: residual Settings URL is inert — no Base gateway routing."""

    settings = ChallengeSettings(
        llm_gateway_base_url="https://master.example",
        agent_gateway_token="should-not-inject",
        llm_gateway_token="analyzer-should-not-inject",
    )
    assert agent_gateway_config_from_settings(settings) is None


def test_gateway_rules_reviewer_does_not_consume_base_gateway() -> None:
    """VAL-ACAT-015: configured rules reviewer must not wire Base gateway."""

    reviewer = build_configured_rules_reviewer()
    # Production path returns None (no Base gateway consumer).
    assert reviewer is None


def test_gateway_llm_base_url_helper_pure_path_only() -> None:
    """Helper may join path for residual tests but is not a product agent injector."""

    url = gateway_llm_base_url("https://master.example")
    assert url.endswith("/llm/v1")


def test_static_analyzer_flags_base_gateway_client(tmp_path: Path) -> None:
    """VAL-ACAT-015: residual Base gateway client patterns fail offline."""

    workspace = tmp_path / "gateway-agent"
    workspace.mkdir()
    (workspace / "agent.py").write_text(
        "import os\n"
        "import httpx\n"
        "\n"
        "BASE = os.environ['BASE_LLM_GATEWAY_URL']\n"
        "TOKEN = os.environ['BASE_GATEWAY_TOKEN']\n"
        "httpx.post(BASE + '/chat/completions', headers={'X-Gateway-Token': TOKEN})\n"
        "url = 'https://master.example/llm/v1/chat/completions'\n",
        encoding="utf-8",
    )
    report = run_rules_analyzer(workspace, reviewer=_SilentReviewer())
    assert report.overall_verdict == "invalid"
    codes = {f.reason_code for f in report.hardcoding_findings}
    assert "base_gateway_forbidden" in codes or "unauthorized_llm_provider" in codes
    # Message semantics: Base gateway is the forbidden residue.
    joined = " ".join(f.description for f in report.hardcoding_findings)
    assert "BASE_LLM_GATEWAY" in joined or "Base gateway" in joined or "/llm/v1" in joined


def test_static_analyzer_measured_openrouter_not_auto_cheated(tmp_path: Path) -> None:
    """VAL-ACAT-015: measured OpenRouter path is not spoiled for avoiding Base gateway."""

    workspace = tmp_path / "measured-or-agent"
    workspace.mkdir()
    (workspace / "agent.py").write_text(
        '"""Agent that relies on measured OpenRouter inside review/eval CVM."""\n'
        "\n"
        "def solve(task: str) -> str:\n"
        "    # tools-only score path; LLM judge is measured review under .rules\n"
        "    return task.upper()\n",
        encoding="utf-8",
    )
    report = run_rules_analyzer(workspace, reviewer=_SilentReviewer())
    codes = {f.reason_code for f in report.hardcoding_findings}
    # Pure tools-only / mention of openrouter.ai in docs-context comments for
    # the measured path must not be auto-flagged solely for avoiding gateway.
    assert "unauthorized_llm_provider" not in codes or "openrouter" not in " ".join(
        f.description.lower() for f in report.hardcoding_findings
    )
    assert report.overall_verdict in {"valid", "suspicious"}  # not forced invalid by gateway avoids


def test_static_analyzer_does_not_false_flag_openrouter_host_alone(tmp_path: Path) -> None:
    """Mention of openrouter.ai host for measured path is not auto-cheat."""

    workspace = tmp_path / "or-mention"
    workspace.mkdir()
    (workspace / "agent.py").write_text(
        "ORIGIN = 'https://openrouter.ai'\ndef solve():\n    return ORIGIN\n",
        encoding="utf-8",
    )
    report = run_rules_analyzer(workspace, reviewer=_SilentReviewer())
    codes = {f.reason_code for f in report.hardcoding_findings}
    assert "unauthorized_llm_provider" not in codes
    assert "base_gateway_forbidden" not in codes


def test_create_review_session_retains_harness_identity_export() -> None:
    """Feature wire check: create_review_session export retains harness materials."""

    # Import surface must keep create_review_session for API/product intake.
    assert callable(create_review_session)
    # Document that harness_identity is part of the CreatedReviewSession contract.
    from dataclasses import fields

    from agent_challenge.review.sessions import CreatedReviewSession

    names = {f.name for f in fields(CreatedReviewSession)}
    assert "harness_identity" in names


def test_policy_rules_forbid_base_gateway_text() -> None:
    """VAL-ACAT-014: .rules forbid Base gateway; allow measured OR / tools-only."""

    root = Path(__file__).resolve().parents[1]
    rules = "\n".join(p.read_text(encoding="utf-8") for p in sorted((root / ".rules").glob("*.md")))
    # Residual Base gateway names may appear only as forbidden material.
    assert "BASE_LLM_GATEWAY_URL" in rules
    assert "BASE_GATEWAY_TOKEN" in rules
    assert "must route all LLM traffic through the platform gateway" not in rules
    assert "measured OpenRouter" in rules
    assert "tools-only" in rules
    assert "/llm/v1" in rules  # named as forbidden
