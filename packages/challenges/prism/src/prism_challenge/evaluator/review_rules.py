from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReviewRule:
    id: str
    text: str


def load_review_rules(
    *,
    defaults: tuple[ReviewRule, ...] = (),
    rules_json: str | None = None,
    rules_file: str | Path | None = None,
) -> tuple[ReviewRule, ...]:
    rules = list(defaults)
    if rules_file:
        path = Path(rules_file)
        if path.exists():
            rules.extend(_parse_rules(path.read_text(encoding="utf-8"), source=str(path)))
    if rules_json:
        rules.extend(_parse_rules(rules_json, source="rules_json"))
    deduped: dict[str, ReviewRule] = {}
    for rule in rules:
        deduped[rule.id] = rule
    return tuple(deduped.values())


def rules_prompt(rules: tuple[ReviewRule, ...]) -> str:
    if not rules:
        return "No additional subnet rules were provided."
    return "\n".join(f"- {rule.id}: {rule.text}" for rule in rules)


def _parse_rules(raw: str, *, source: str) -> list[ReviewRule]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dynamic review rules in {source}: {exc.msg}") from exc
    if isinstance(payload, dict):
        payload = payload.get("rules", payload.get("subnet_rules", []))
    if not isinstance(payload, list):
        raise ValueError(f"dynamic review rules in {source} must be a list")
    return [_coerce_rule(item, index, source) for index, item in enumerate(payload, start=1)]


def _coerce_rule(item: Any, index: int, source: str) -> ReviewRule:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            raise ValueError(f"empty dynamic review rule #{index} in {source}")
        return ReviewRule(id=f"{source}:{index}", text=text)
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("rule") or item.get("description") or "").strip()
        if not text:
            raise ValueError(f"dynamic review rule #{index} in {source} is missing text")
        rule_id = str(item.get("id") or item.get("rule_id") or f"{source}:{index}").strip()
        return ReviewRule(id=rule_id, text=text)
    raise ValueError(f"dynamic review rule #{index} in {source} must be a string or object")
