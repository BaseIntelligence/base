from __future__ import annotations

import json

import pytest

from prism_challenge.evaluator.review_rules import (
    ReviewRule,
    load_review_rules,
    rules_prompt,
)


def test_load_review_rules_returns_defaults_when_nothing_else():
    defaults = (ReviewRule(id="d1", text="default rule"),)
    rules = load_review_rules(defaults=defaults)
    assert rules == defaults


def test_load_review_rules_from_json_string_list():
    rules = load_review_rules(rules_json=json.dumps(["no eval", "no exec"]))
    assert len(rules) == 2
    assert rules[0].text == "no eval"
    assert rules[0].id == "rules_json:1"
    assert rules[1].id == "rules_json:2"


def test_load_review_rules_from_json_object_with_rules_key():
    payload = {"rules": [{"id": "r1", "text": "be deterministic"}]}
    rules = load_review_rules(rules_json=json.dumps(payload))
    assert rules == (ReviewRule(id="r1", text="be deterministic"),)


def test_load_review_rules_from_json_object_with_subnet_rules_key():
    payload = {"subnet_rules": ["alpha rule"]}
    rules = load_review_rules(rules_json=json.dumps(payload))
    assert len(rules) == 1
    assert rules[0].text == "alpha rule"


def test_load_review_rules_object_rule_alt_text_keys():
    payload = [
        {"rule": "via rule key"},
        {"description": "via description key"},
    ]
    rules = load_review_rules(rules_json=json.dumps(payload))
    assert rules[0].text == "via rule key"
    assert rules[1].text == "via description key"


def test_load_review_rules_object_custom_id_keys():
    payload = [{"rule_id": "custom-1", "text": "t"}]
    rules = load_review_rules(rules_json=json.dumps(payload))
    assert rules[0].id == "custom-1"


def test_load_review_rules_dedupes_by_id():
    defaults = (ReviewRule(id="x", text="first"),)
    rules = load_review_rules(
        defaults=defaults,
        rules_json=json.dumps([{"id": "x", "text": "second wins"}]),
    )
    assert len(rules) == 1
    assert rules[0].text == "second wins"


def test_load_review_rules_from_file(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(["from file"]), encoding="utf-8")
    rules = load_review_rules(rules_file=path)
    assert rules[0].text == "from file"
    assert rules[0].id == f"{path}:1"


def test_load_review_rules_missing_file_is_ignored():
    rules = load_review_rules(rules_file="/nonexistent/path/rules.json")
    assert rules == ()


def test_load_review_rules_file_then_json_both_merged(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(["file rule"]), encoding="utf-8")
    rules = load_review_rules(rules_file=path, rules_json=json.dumps(["json rule"]))
    texts = {rule.text for rule in rules}
    assert texts == {"file rule", "json rule"}


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_invalid_json_raises_value_error():
    with pytest.raises(ValueError, match="invalid dynamic review rules"):
        load_review_rules(rules_json="{not valid json")


def test_non_list_non_dict_payload_raises():
    with pytest.raises(ValueError, match="must be a list"):
        load_review_rules(rules_json=json.dumps("just a string"))


def test_empty_string_rule_raises():
    with pytest.raises(ValueError, match="empty dynamic review rule"):
        load_review_rules(rules_json=json.dumps(["   "]))


def test_object_rule_missing_text_raises():
    with pytest.raises(ValueError, match="missing text"):
        load_review_rules(rules_json=json.dumps([{"id": "x"}]))


def test_rule_wrong_type_raises():
    with pytest.raises(ValueError, match="must be a string or object"):
        load_review_rules(rules_json=json.dumps([123]))


# ---------------------------------------------------------------------------
# rules_prompt
# ---------------------------------------------------------------------------


def test_rules_prompt_empty():
    assert rules_prompt(()) == "No additional subnet rules were provided."


def test_rules_prompt_formats_each_rule():
    rules = (ReviewRule(id="a", text="alpha"), ReviewRule(id="b", text="beta"))
    prompt = rules_prompt(rules)
    assert prompt == "- a: alpha\n- b: beta"
