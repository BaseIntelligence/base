"""Public task-event text redaction: keep TB task ids, still scrub secrets."""

from __future__ import annotations

from agent_challenge.api.routes import _public_task_event_text

# Shapes asserted by the existing frontend / submission-status contract suites.
_GENUINE_SECRET_SHAPES = (
    "sk-test-secret",
    "Bearer raw-provider-token",
    "broker-token",
    "tb21-platform-sdk-secret",
    "lease-worker-secret",
    "broker-ref-secret",
    "raw-ref-abc123",
)


def test_public_task_event_text_preserves_count_dataset_tokens() -> None:
    """Terminal-Bench task id containing 'token' must not be swallowed whole."""

    message = "task terminal-bench/count-dataset-tokens assigned"
    out = _public_task_event_text(message)
    assert "count-dataset-tokens" in out
    assert "terminal-bench/count-dataset-tokens" in out
    assert "[REDACTED_SECRET]" not in out or "count-dataset-tokens" in out
    assert out == message or "count-dataset-tokens" in out


def test_public_task_event_text_still_redacts_genuine_secrets() -> None:
    """Credential-shaped tokens must remain absent from public event text."""

    payload = (
        "leak sk-test-secret and Bearer raw-provider-token and broker-token "
        "and tb21-platform-sdk-secret and lease-worker-secret and "
        "broker-ref-secret and raw-ref-abc123 end"
    )
    out = _public_task_event_text(payload)
    for secret in _GENUINE_SECRET_SHAPES:
        assert secret not in out, f"secret still visible: {secret!r} in {out!r}"
    assert "[REDACTED_SECRET]" in out
