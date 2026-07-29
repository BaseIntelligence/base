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


def test_secret_redaction_is_linear_on_dotted_runs() -> None:
    """A public log line must never wedge the event loop.

    ``_PUBLIC_SECRET_SEGMENT_RE`` nested ``(?:[A-Za-z0-9_.]+[-_.])*`` over a class
    that also contains the separators, so a dotted/underscored run that never
    reaches ``secret``/``token`` backtracked exponentially: 37 chars took 23ms and
    61 chars took 91s. pip output carries exactly that shape
    ("Successfully installed agent-challenge-1.0.1 aiohappyeyeballs-2.7.1 ..."),
    and the endpoint serving it is public, so one log line pinned the validator
    at 100% CPU and starved every evaluation.
    """
    import time

    from agent_challenge.api.routes import _public_task_event_text

    payload = "a.b_c." * 40 + "!"
    started = time.perf_counter()
    _public_task_event_text(payload)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"redaction took {elapsed:.1f}s on {len(payload)} chars"


def test_realistic_pip_output_is_fast_and_unredacted() -> None:
    from agent_challenge.api.routes import _public_task_event_text

    line = (
        "Successfully installed agent-challenge-1.0.1 aiohappyeyeballs-2.7.1 "
        "aiohttp-3.14.3 aiosignal-1.4.0 async-substrate-interface-2.2.1 "
    ) * 6
    import time

    started = time.perf_counter()
    out = _public_task_event_text(line)
    assert time.perf_counter() - started < 1.0
    assert "agent-challenge-1.0.1" in out
