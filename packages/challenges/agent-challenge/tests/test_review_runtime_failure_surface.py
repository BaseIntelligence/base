"""Guest residual stderr must carry allowlisted reason codes (not class only)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_review_runtime():
    path = Path(__file__).resolve().parents[1] / "docker" / "review" / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_failure_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_failure_surface_includes_allowlisted_reason_code() -> None:
    runtime = _load_review_runtime()
    surface = runtime.bounded_review_failure_surface(
        RuntimeError("assignment fetch failed status=401")
    )
    assert surface["error"] == "review_failed"
    assert surface["reason"] == "RuntimeError"
    # Must map to allowlisted infrastructure code (never raw exception message).
    assert surface["reason_code"] == "report_generation_failed"
    # Closed 3-digit http_status is intentional residual diag surface;
    # free-form body words from the exception must still be refused.
    assert surface.get("http_status") == "401"
    assert "assignment fetch" not in str(surface).lower()


def test_bounded_failure_surface_maps_openrouter_transport() -> None:
    runtime = _load_review_runtime()
    from agent_challenge.review.openrouter import OpenRouterTransportError

    surface = runtime.bounded_review_failure_surface(
        OpenRouterTransportError("dns_failed", "name resolution exploded")
    )
    assert surface["reason_code"] == "dns_failed"
    assert "exploded" not in str(surface).lower()


def test_bounded_failure_surface_exposes_allowed_cap_diag() -> None:
    runtime = _load_review_runtime()
    from agent_challenge.review.openrouter import OpenRouterTransportError

    surface = runtime.bounded_review_failure_surface(
        OpenRouterTransportError(
            "policy_output_malformed",
            "OpenRouter policy output is malformed",
            diag="allowed_cap",
        )
    )
    assert surface["reason_code"] == "policy_output_malformed"
    assert surface.get("diag") == "allowed_cap"


def test_bounded_failure_surface_module_not_found_diag_or_outcome_bind() -> None:
    """ModuleNotFound residuals must surface a short allowlisted module token.

    Post-attested_times guest residual collapsed to anonymous:
    ``{"reason":"ModuleNotFoundError","reason_code":"report_generation_failed"}``
    without the missing module name, thrashing next diagnosis. Emit short
    allowlisted basename (e.g. ``or_outcome_bind``) on ``diag``.
    """

    runtime = _load_review_runtime()
    exc = ModuleNotFoundError("No module named 'agent_challenge.review.or_outcome_bind'")
    exc.name = "agent_challenge.review.or_outcome_bind"
    surface = runtime.bounded_review_failure_surface(exc)
    assert surface["error"] == "review_failed"
    assert surface["reason"] == "ModuleNotFoundError"
    assert surface["reason_code"] == "report_generation_failed"
    assert surface.get("diag") == "or_outcome_bind"
    # Never re-emit free-form import paths or bottoms-up stack wording.
    assert "No module named" not in str(surface)
    assert "agent_challenge.review" not in str(surface)


def test_bounded_failure_surface_module_not_found_diag_attested_times() -> None:
    runtime = _load_review_runtime()
    exc = ModuleNotFoundError("No module named 'agent_challenge.review.attested_times'")
    exc.name = "agent_challenge.review.attested_times"
    surface = runtime.bounded_review_failure_surface(exc)
    assert surface.get("diag") == "attested_times"
    assert "agent_challenge" not in str(surface)


def test_bounded_failure_surface_module_not_found_unknown_stays_silent() -> None:
    """Unknown missing modules must not mint free-form diag tokens."""

    runtime = _load_review_runtime()
    exc = ModuleNotFoundError("No module named 'evil_secret_helper'")
    exc.name = "evil_secret_helper"
    surface = runtime.bounded_review_failure_surface(exc)
    assert surface["reason"] == "ModuleNotFoundError"
    assert "diag" not in surface or surface.get("diag") not in {"evil_secret_helper"}
    assert "evil_secret_helper" not in str(surface)


def test_allowed_evidence_paths_from_artifact_zip_is_single_relative_not_3n() -> None:
    """Packager allowlist must stay 1N so packages with >22 files still parse."""
    import io
    import json
    import zipfile

    from agent_challenge.review.policy import parse_model_policy_output

    runtime = _load_review_runtime()
    buf = io.BytesIO()
    # >22 files: historical 3N expand would push past the old assigned bound of 64.
    file_count = 70
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(file_count):
            zf.writestr(f"src/mod_{i:03d}.py", f"# member {i}\n")
        zf.writestr("agent.py", "def main():\n    return 0\n")
    artifact_bytes = buf.getvalue()

    allowed = runtime.allowed_evidence_paths_from_artifact_zip(artifact_bytes)
    assert len(allowed) == file_count + 1
    assert "agent.py" in allowed
    assert "src/mod_000.py" in allowed
    # No prefix-duplicated aliases.
    assert not any(p.startswith("artifact/") for p in allowed)
    assert not any(p.startswith("submission/") for p in allowed)
    assert len(allowed) * 3 > 64  # documents the former explode cliff

    # Policy parse must accept this package allowlist (not raise assigned cap).
    model_body = json.dumps(
        {
            "id": "offline",
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
                                    "arguments": json.dumps(
                                        {
                                            "verdict": "allow",
                                            "reason_codes": [],
                                            "evidence_paths": [
                                                "agent.py",
                                                "artifact/src/mod_000.py",
                                            ],
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
    parsed = parse_model_policy_output(model_body, allowed_evidence_paths=allowed)
    assert parsed.verdict == "allow"
    assert parsed.evidence_paths == ("agent.py", "src/mod_000.py")
