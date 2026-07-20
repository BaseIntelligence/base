"""Product REVIEW_MODEL pin is x-ai/grok-4.5 (not moonshotai/kimi).

TDD contract for the live review model switch:
- constant pin identity
- dated OpenRouter snapshot acceptance via is_pinned_review_model
- default OpenRouter provider order/only is **xai** (OpenRouter slug; not
  model-id prefix x-ai, and not moonshotai)
- no product OpenRouter RPS limiter is introduced by the pin flip
"""

from __future__ import annotations

import re
from pathlib import Path

from agent_challenge.review.openrouter import build_openrouter_request_body
from agent_challenge.review.schemas import (
    REVIEW_MODEL,
    ReviewInputConfig,
    is_pinned_review_model,
)

PIN = "x-ai/grok-4.5"
# Literal legacy pin string (must remain rejected after the product flip).
LEGACY_KIMI = "moonshot" + "ai/kimi-k2.7-code"


def test_review_model_constant_is_grok_45() -> None:
    assert REVIEW_MODEL == PIN
    assert REVIEW_MODEL != LEGACY_KIMI
    assert "kimi" not in REVIEW_MODEL
    assert "moonshotai" not in REVIEW_MODEL


def test_is_pinned_review_model_accepts_exact_and_dated_suffix() -> None:
    assert is_pinned_review_model(PIN) is True
    assert is_pinned_review_model(f"{PIN}-20260717") is True
    # OpenRouter dated snapshot shape used for prior pin families.
    assert is_pinned_review_model(f"{PIN}-20260101") is True


def test_is_pinned_review_model_rejects_legacy_kimi_and_aliases() -> None:
    assert is_pinned_review_model(LEGACY_KIMI) is False
    assert is_pinned_review_model(f"{LEGACY_KIMI}-20260612") is False
    assert is_pinned_review_model(f"{PIN}:free") is False
    assert is_pinned_review_model(f"{PIN}-notadate") is False
    assert is_pinned_review_model(f"{PIN}-2026071") is False  # not 8 digits
    assert is_pinned_review_model("") is False
    assert is_pinned_review_model(None) is False
    assert is_pinned_review_model(123) is False


def test_default_routing_provider_order_only_is_xai() -> None:
    """OpenRouter provider slug is xai; model id stays x-ai/grok-4.5."""

    routing = ReviewInputConfig().resolved_routing()
    assert routing["order"] == ["xai"]
    assert routing["only"] == ["xai"]
    # Guard against the wrong product default that yields OR 404
    # "No allowed providers" with available_providers=["xai"].
    assert routing["order"] != ["x-ai"]
    assert routing["only"] != ["x-ai"]
    assert "x-ai" not in routing["order"]
    assert "x-ai" not in routing["only"]
    assert "moonshotai" not in routing["order"]
    assert "moonshotai" not in routing["only"]
    assert routing["allow_fallbacks"] is False
    assert routing["require_parameters"] is True
    assert routing["data_collection"] == "deny"


def test_openrouter_body_pins_model_and_xai_routing() -> None:
    body = build_openrouter_request_body(
        messages=[{"role": "user", "content": "review under .rules"}],
        routing=ReviewInputConfig().resolved_routing(),
    )
    text = body.decode("utf-8")
    assert f'"model":"{PIN}"' in text
    assert "moonshotai" not in text
    assert "kimi" not in text
    assert '"order":["xai"]' in text
    assert '"only":["xai"]' in text
    # Model id still uses x-ai/ org prefix; provider body must not.
    assert '"provider":' in text
    assert '"order":["x-ai"]' not in text
    assert '"only":["x-ai"]' not in text
    assert "tool_choice" in text
    assert "submit_verdict" in text
    assert "Accept-Encoding" not in text  # header, not body
    # Identity encoding is an OPENROUTER_HEADERS concern, checked below.


def test_openrouter_headers_keep_accept_encoding_identity() -> None:
    from agent_challenge.review.schemas import OPENROUTER_HEADERS

    assert OPENROUTER_HEADERS["accept-encoding"] == "identity"


def test_no_product_openrouter_rps_limiter_in_review_package() -> None:
    """Pin flip must not smuggle a product-side OpenRouter RPS/RPM throttle."""

    review_src = Path(__file__).resolve().parents[1] / "src" / "agent_challenge" / "review"
    banned = re.compile(
        r"(rps_limit|rpm_limit|rate_limit_openrouter|OpenRouterRate|openrouter_rps|"
        r"OR_RPS|tokens_per_minute_limit|sleep_between_openrouter)",
        re.IGNORECASE,
    )
    offenders: list[str] = []
    for path in sorted(review_src.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if banned.search(text):
            offenders.append(str(path.relative_to(review_src.parent.parent.parent)))
    assert not offenders, f"product OpenRouter RPS limiter markers found: {offenders}"
