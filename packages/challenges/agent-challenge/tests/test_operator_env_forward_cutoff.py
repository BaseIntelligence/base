"""Operator harbor_forward_env_vars grandfathering by submission.created_at.

Pre-cutoff submissions keep receiving the operator-injected OPENROUTER_API_KEY
(and any other harbor_forward_env_vars). Post-cutoff submissions must supply
their own key via miner env; the operator key is not forwarded and must not
shadow the miner value. Missing keys fail with an attributable error.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_challenge.evaluation.runner import _terminal_bench_env
from agent_challenge.sdk.config import ChallengeSettings

CUTOFF = datetime(2026, 7, 30, 12, 0, 0, tzinfo=UTC)
PRE_CUTOFF = CUTOFF - timedelta(hours=1)
POST_CUTOFF = CUTOFF + timedelta(hours=1)
OPERATOR_KEY = "sk-or-v1-operator-forward-test-value"
MINER_KEY = "sk-or-v1-miner-supplied-test-value"


def _patch_forward_settings(monkeypatch, *, cutoff: datetime | None = CUTOFF) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.settings.harbor_forward_env_vars",
        ("OPENROUTER_API_KEY",),
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.settings.operator_env_forward_cutoff_at",
        cutoff,
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", OPERATOR_KEY)


def test_pre_cutoff_submission_forwards_operator_key(monkeypatch) -> None:
    """Given: submission created before cutoff, operator has OPENROUTER_API_KEY.
    When: _terminal_bench_env builds job env
    Then: operator key is present (grandfathered).
    """
    _patch_forward_settings(monkeypatch)

    env = _terminal_bench_env(
        {"LLM_COST_LIMIT": "1.0"},
        submission_created_at=PRE_CUTOFF,
    )

    assert env["OPENROUTER_API_KEY"] == OPERATOR_KEY


def test_post_cutoff_with_miner_key_uses_miner_not_operator(monkeypatch) -> None:
    """Given: post-cutoff submission with miner OPENROUTER_API_KEY
    When: _terminal_bench_env builds job env
    Then: miner key is present, operator key is absent (not shadowed).
    """
    _patch_forward_settings(monkeypatch)

    env = _terminal_bench_env(
        {"OPENROUTER_API_KEY": MINER_KEY},
        submission_created_at=POST_CUTOFF,
    )

    assert env["OPENROUTER_API_KEY"] == MINER_KEY
    assert env["OPENROUTER_API_KEY"] != OPERATOR_KEY


def test_post_cutoff_without_key_raises_attributable_error(monkeypatch) -> None:
    """Given: post-cutoff submission with no OPENROUTER_API_KEY
    When: _terminal_bench_env builds job env
    Then: ValueError names OPENROUTER_API_KEY (not silent zero / crash elsewhere).
    """
    _patch_forward_settings(monkeypatch)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY") as exc_info:
        _terminal_bench_env(
            {"LLM_COST_LIMIT": "1.0"},
            submission_created_at=POST_CUTOFF,
        )

    message = str(exc_info.value)
    assert "OPENROUTER_API_KEY" in message
    assert "cutoff" in message.lower() or "operator" in message.lower()


def test_cutoff_boundary_instant_is_post_cutoff(monkeypatch) -> None:
    """Given: submission.created_at exactly equal to cutoff
    When: _terminal_bench_env builds job env without miner key
    Then: boundary is post-cutoff (operator not forwarded; missing key fails).
    """
    _patch_forward_settings(monkeypatch)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        _terminal_bench_env(
            {},
            submission_created_at=CUTOFF,
        )

    env = _terminal_bench_env(
        {"OPENROUTER_API_KEY": MINER_KEY},
        submission_created_at=CUTOFF,
    )
    assert env["OPENROUTER_API_KEY"] == MINER_KEY


def test_no_cutoff_configured_keeps_legacy_operator_forward(monkeypatch) -> None:
    """Given: operator_env_forward_cutoff_at is None (unset)
    When: any submission builds env
    Then: operator key is still forwarded (safe default until ops sets cutoff).
    """
    _patch_forward_settings(monkeypatch, cutoff=None)

    env = _terminal_bench_env(
        {},
        submission_created_at=POST_CUTOFF,
    )

    assert env["OPENROUTER_API_KEY"] == OPERATOR_KEY


def test_operator_env_forward_cutoff_at_env_override(monkeypatch) -> None:
    """Given: CHALLENGE_OPERATOR_ENV_FORWARD_CUTOFF_AT is set
    When: ChallengeSettings loads
    Then: operator_env_forward_cutoff_at parses as aware UTC datetime.
    """
    monkeypatch.setenv(
        "CHALLENGE_OPERATOR_ENV_FORWARD_CUTOFF_AT",
        "2026-07-30T12:00:00+00:00",
    )
    settings = ChallengeSettings()
    assert settings.operator_env_forward_cutoff_at == CUTOFF
