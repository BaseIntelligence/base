"""TDD: response_malformed subtypes must be distinctly allowlisted.

Product residual after xai routing pin (a98638d1+): host OpenRouter paths return
HTTP 200, but measured guest collapses several fail classes into a single
``response_malformed`` reason_code. These tests pin the expanded allowlist and
transport mappings without inventing allow/KR and without a product OR throttle.

See research/openrouter-response-malformed-xai.md (missionDir).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from agent_challenge.review.canonical import canonical_json_v1
from agent_challenge.review.openrouter import (
    DirectOpenRouterClient,
    OpenRouterTransportError,
    build_openrouter_request_body,
    build_review_infrastructure_failure,
    infrastructure_failure_reason,
    short_policy_error_class,
)
from agent_challenge.review.policy import ReviewPolicyError
from agent_challenge.review.schemas import (
    MAX_OPENROUTER_METADATA_BYTES,
    REVIEW_INFRASTRUCTURE_FAILURE_REASONS,
    REVIEW_MODEL,
)

SENTINEL_KEY = "review-or-subreason-sentinel"
_ROUTING = {
    "order": ["xai"],
    "only": ["xai"],
    "ignore": [],
    "quantizations": [],
    "sort": None,
    "allow_fallbacks": False,
    "require_parameters": True,
    "data_collection": "deny",
}

# Feature-required distinct subclasses (plus residual response_malformed kept).
_SUBTYPES = (
    "compressed_response_forbidden",
    "openrouter_body_not_json",
    "model_pin_mismatch",
    "policy_output_malformed",
    "metadata_bounds",
    "planned_digest_unbound",
)


def _body() -> bytes:
    return build_openrouter_request_body(
        messages=[{"content": "review only supplied bytes", "role": "user"}],
        routing=_ROUTING,
    )


def _routing_sha256() -> str:
    return hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest()


def _client(handler) -> DirectOpenRouterClient:
    return DirectOpenRouterClient(
        assignment_id="ra-subreason",
        api_key=SENTINEL_KEY,
        announce=lambda _: True,
        transport=httpx.MockTransport(handler),
    )


def _tool_payload(*, model: str = REVIEW_MODEL, arguments: dict | None = None) -> dict:
    args = (
        arguments
        if arguments is not None
        else {
            "verdict": "allow",
            "reason_codes": [],
            "evidence_paths": ["artifact/agent.py"],
        }
    )
    return {
        "id": "offline-response",
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "submit_verdict",
                                "arguments": json.dumps(args, separators=(",", ":")),
                            },
                        }
                    ],
                }
            }
        ],
    }


def test_allowlist_includes_response_malformed_subtypes() -> None:
    """Closed allowlist must list every product residual subclass name."""
    missing = [code for code in _SUBTYPES if code not in REVIEW_INFRASTRUCTURE_FAILURE_REASONS]
    assert missing == []
    # Residual parent token retained for any future generic collapse path.
    assert "response_malformed" in REVIEW_INFRASTRUCTURE_FAILURE_REASONS
    # Existing transport outcome mapping stays single-token for 429/5xx class.
    assert "openrouter_rate_limited" in REVIEW_INFRASTRUCTURE_FAILURE_REASONS
    assert "openrouter_unavailable" in REVIEW_INFRASTRUCTURE_FAILURE_REASONS


def test_infrastructure_failure_builder_accepts_each_subtype() -> None:
    for code in _SUBTYPES:
        failure = build_review_infrastructure_failure(
            assignment_id="ra-subreason",
            planned_request_sha256="ab" * 32,
            reason_code=code,
        )
        assert failure["reason_code"] == code


@pytest.mark.parametrize(
    "content_encoding",
    [
        "gzip",
        "br",
        "deflate",
        "GZIP",
    ],
)
def test_compressed_response_maps_to_compressed_response_forbidden(
    content_encoding: str,
) -> None:
    # Headers-only residual: product refuses any non-identity Content-Encoding
    # before body parse. Deliver via stream (never content=) so httpx does not
    # auto-decompress in MockTransport and collapse into openrouter_unavailable.
    raw = json.dumps(_tool_payload(), separators=(",", ":")).encode("utf-8")

    class _ByteStream(httpx.SyncByteStream):
        def __iter__(self):  # type: ignore[override]
            yield raw

        def close(self) -> None:
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "content-encoding": content_encoding,
            },
            stream=_ByteStream(),
            request=request,
        )

    with pytest.raises(OpenRouterTransportError) as info:
        _client(handler).call(
            body=_body(),
            routing_sha256=_routing_sha256(),
            allowed_evidence_paths={"artifact/agent.py"},
        )
    assert info.value.reason_code == "compressed_response_forbidden"
    assert infrastructure_failure_reason(info.value) == "compressed_response_forbidden"


def test_non_json_body_maps_to_openrouter_body_not_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-json<<<",
            headers={
                "content-type": "application/json",
                "content-encoding": "identity",
            },
        )

    with pytest.raises(OpenRouterTransportError) as info:
        _client(handler).call(
            body=_body(),
            routing_sha256=_routing_sha256(),
            allowed_evidence_paths={"artifact/agent.py"},
        )
    assert info.value.reason_code == "openrouter_body_not_json"
    assert infrastructure_failure_reason(info.value) == "openrouter_body_not_json"


def test_wrong_model_id_maps_to_model_pin_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_tool_payload(model="openai/gpt-4o"),
            headers={
                "content-type": "application/json",
                "content-encoding": "identity",
            },
        )

    with pytest.raises(OpenRouterTransportError) as info:
        _client(handler).call(
            body=_body(),
            routing_sha256=_routing_sha256(),
            allowed_evidence_paths={"artifact/agent.py"},
        )
    assert info.value.reason_code == "model_pin_mismatch"
    assert infrastructure_failure_reason(info.value) == "model_pin_mismatch"


def test_policy_output_malformed_maps_with_short_policy_error_class() -> None:
    """Malformed tool arguments → policy_output_malformed + short diag class."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_tool_payload(arguments={"verdict": "not-a-verdict", "reason_codes": []}),
            headers={
                "content-type": "application/json",
                "content-encoding": "identity",
            },
        )

    with pytest.raises(OpenRouterTransportError) as info:
        _client(handler).call(
            body=_body(),
            routing_sha256=_routing_sha256(),
            allowed_evidence_paths={"artifact/agent.py"},
        )
    assert info.value.reason_code == "policy_output_malformed"
    assert infrastructure_failure_reason(info.value) == "policy_output_malformed"
    # Short redacted class only — never full exception messages/bodies.
    assert getattr(info.value, "diag", None) in {
        "verdict",
        "args",
        "shape",
        "tool_shape",
        "tool_count",
        "other",
    }
    assert "not-a-verdict" not in str(getattr(info.value, "diag", ""))


def test_short_policy_error_class_is_closed_and_secret_free() -> None:
    assert short_policy_error_class(ReviewPolicyError("model verdict is not allowed")) == "verdict"
    assert (
        short_policy_error_class(
            ReviewPolicyError("model response must contain exactly one final tool call")
        )
        == "tool_count"
    )
    assert (
        short_policy_error_class(ReviewPolicyError("model policy arguments are malformed"))
        == "args"
    )
    assert (
        short_policy_error_class(ReviewPolicyError("too many assigned evidence paths"))
        == "allowed_cap"
    )
    # Unknown residual collapses without echoing raw text.
    assert short_policy_error_class(ReviewPolicyError("secret=sk-live-xyz")) == "other"


def test_metadata_exceeds_bound_maps_to_metadata_bounds() -> None:
    huge = "m" * (MAX_OPENROUTER_METADATA_BYTES + 8)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_tool_payload(),
            headers={
                "content-type": "application/json",
                "content-encoding": "identity",
                "x-openrouter-metadata": huge,
            },
        )

    with pytest.raises(OpenRouterTransportError) as info:
        _client(handler).call(
            body=_body(),
            routing_sha256=_routing_sha256(),
            allowed_evidence_paths={"artifact/agent.py"},
        )
    assert info.value.reason_code == "metadata_bounds"
    assert infrastructure_failure_reason(info.value) == "metadata_bounds"


def test_rate_limited_and_unavailable_mapping_unchanged() -> None:
    """Regression: 429/5xx must not be reclassified into response_malformed subtypes."""

    for status, expected in (
        (429, "openrouter_rate_limited"),
        (503, "openrouter_unavailable"),
        (500, "openrouter_unavailable"),
    ):

        def handler(request: httpx.Request, sc: int = status) -> httpx.Response:
            return httpx.Response(sc)

        with pytest.raises(OpenRouterTransportError) as info:
            _client(handler).call(
                body=_body(),
                routing_sha256=_routing_sha256(),
                allowed_evidence_paths={"artifact/agent.py"},
            )
        assert info.value.reason_code == expected


def test_guest_public_logs_surface_exposes_subtype_reason_code_and_diag() -> None:
    path = Path(__file__).resolve().parents[1] / "docker" / "review" / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_subreasons", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    surface = module.bounded_review_failure_surface(
        OpenRouterTransportError(
            "policy_output_malformed",
            "OpenRouter policy output is malformed",
            diag="verdict",
        )
    )
    assert surface["error"] == "review_failed"
    assert surface["reason"] == "OpenRouterTransportError"
    assert surface["reason_code"] == "policy_output_malformed"
    assert surface.get("diag") == "verdict"
    # Never leak longer free text via the residual public_logs envelope.
    assert "malformed" not in surface.get("diag", "")

    gzip_surface = module.bounded_review_failure_surface(
        OpenRouterTransportError(
            "compressed_response_forbidden",
            "compressed response is forbidden",
        )
    )
    assert gzip_surface["reason_code"] == "compressed_response_forbidden"
    assert "diag" not in gzip_surface or gzip_surface.get("diag") in {None, ""}


def test_response_byte_cap_stays_fail_closed_residual() -> None:
    """Body-size exceed lingers as response_malformed residual; transport fails closed."""
    from agent_challenge.review.schemas import MAX_OPENROUTER_RESPONSE_BYTES

    max_bytes = MAX_OPENROUTER_RESPONSE_BYTES
    total_to_emit = max_bytes + 64_000
    emitted = {"bytes": 0}

    class OversizedStream(httpx.SyncByteStream):
        def __iter__(self):  # type: ignore[override]
            remaining = total_to_emit
            chunk = b"x" * 8192
            while remaining > 0:
                piece = chunk if remaining >= len(chunk) else chunk[:remaining]
                emitted["bytes"] += len(piece)
                remaining -= len(piece)
                yield piece

        def close(self) -> None:
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "content-encoding": "identity"},
            stream=OversizedStream(),
            request=request,
        )

    with pytest.raises(OpenRouterTransportError) as info:
        _client(handler).call(
            body=_body(),
            routing_sha256=_routing_sha256(),
            allowed_evidence_paths={"artifact/agent.py"},
        )
    # Residual parent token (size bound), distinct from content-encoding/policy/pin.
    assert info.value.reason_code == "response_malformed"
    assert emitted["bytes"] < total_to_emit


def test_no_product_openrouter_rps_limiter_symbols() -> None:
    """Sanity: this residual must not introduce a product RPS limiter surface."""
    import agent_challenge.review.openrouter as openrouter_mod

    source = Path(openrouter_mod.__file__).read_text(encoding="utf-8")
    banned = (
        "rate_limit_rps",
        "openrouter_rps",
        "tokens_per_second",
        "rpm_limiter",
        "Throttle",
        "sleep_between_or",
    )
    for token in banned:
        assert token not in source
