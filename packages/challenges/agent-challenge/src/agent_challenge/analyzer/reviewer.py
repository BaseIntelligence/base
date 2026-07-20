from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from agent_challenge.analyzer.schemas import ReviewerRequest, ReviewerResult
from agent_challenge.core.config import settings

DISABLED_LANGCHAIN_PROVIDERS = frozenset({"", "0", "false", "none", "off", "disabled"})


def build_configured_analyzer_reviewer() -> LangChainAnalyzerReviewer | None:
    provider = (settings.langchain_provider or "").strip()
    if provider.lower() in DISABLED_LANGCHAIN_PROVIDERS:
        return None
    try:
        from langchain.chat_models import init_chat_model  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        model = init_chat_model(
            settings.langchain_model,
            model_provider=provider,
            temperature=settings.langchain_temperature,
            timeout=settings.langchain_timeout_seconds,
            max_tokens=settings.langchain_max_tokens,
        )
    except Exception:
        return None
    return LangChainAnalyzerReviewer(model)


class LangChainAnalyzerReviewer:
    def __init__(self, model: Any) -> None:
        self._model = model

    def review(self, request: ReviewerRequest) -> ReviewerResult | None:
        try:
            response = self._model.invoke(_review_messages(request))
        except Exception:
            return None
        content = getattr(response, "content", response)
        parsed = _parse_reviewer_content(content)
        if parsed is None:
            return None
        try:
            return ReviewerResult.model_validate(parsed)
        except ValidationError:
            return None


def _review_messages(request: ReviewerRequest) -> list[tuple[str, str]]:
    payload = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    return [
        (
            "system",
            "You are a bounded analyzer reviewer for Agent Challenge submissions. "
            "Return only JSON matching this schema: "
            '{"verdict":"valid|invalid|suspicious|error",'
            '"reason_codes":["short_reason"],"evidence":[],"notes":"brief rationale"}.',
        ),
        (
            "human",
            "Review this bounded analyzer request. Do not assume access to files outside "
            f"the payload. Payload JSON: {payload}",
        ),
    ]


def _parse_reviewer_content(content: Any) -> Mapping[str, Any] | None:
    if isinstance(content, Mapping):
        return content
    if isinstance(content, list):
        text = "".join(
            str(item.get("text", "")) if isinstance(item, Mapping) else str(item)
            for item in content
        )
    else:
        text = str(content)
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None
