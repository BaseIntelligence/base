"""Deterministic ARCHITECTURE_NAME parse + moderation (architecture-lab API contract).

The miner declares a module-level ``ARCHITECTURE_NAME = "..."`` constant in ``architecture.py``.
The evaluator extracts it via AST (no code execution) and moderates it deterministically; the
moderation MUST be identical across validators for consensus, so these tests pin the exact rules.
"""

from __future__ import annotations

from prism_challenge.evaluator.components import (
    ARCHITECTURE_NAME_MAX_LENGTH,
    moderate_architecture_name,
    parse_architecture_name,
)


def test_parse_plain_string_literal() -> None:
    src = 'ARCHITECTURE_NAME = "Rotary MoE v3"\n\ndef build_model(ctx):\n    return None\n'
    assert parse_architecture_name(src) == "Rotary MoE v3"


def test_parse_annotated_assignment() -> None:
    src = 'ARCHITECTURE_NAME: str = "Annotated Net"\n'
    assert parse_architecture_name(src) == "Annotated Net"


def test_parse_missing_constant_is_none() -> None:
    assert parse_architecture_name("def build_model(ctx):\n    return None\n") is None


def test_parse_non_string_constant_is_none() -> None:
    assert parse_architecture_name("ARCHITECTURE_NAME = 123\n") is None
    assert parse_architecture_name("ARCHITECTURE_NAME = ['x']\n") is None


def test_parse_computed_expression_is_none() -> None:
    # Only a plain string literal is honored; a runtime-computed value is ignored (no execution).
    assert parse_architecture_name('ARCHITECTURE_NAME = "a" + "b"\n') is None


def test_parse_ignores_non_module_scope() -> None:
    src = 'def build_model(ctx):\n    ARCHITECTURE_NAME = "Local"\n    return None\n'
    assert parse_architecture_name(src) is None


def test_parse_syntax_error_is_none() -> None:
    assert parse_architecture_name("def build_model(:\n") is None


def test_moderate_trims_and_collapses_whitespace() -> None:
    assert moderate_architecture_name("  Rotary   MoE\tv3 \n") == "Rotary MoE v3"


def test_moderate_drops_illegal_characters() -> None:
    # Letters/digits/space and -_.()[]/+& are kept; everything else is dropped.
    assert moderate_architecture_name("Rotary MoE v3!! 🚀") == "Rotary MoE v3"
    assert moderate_architecture_name("a*b@c#d") == "abcd"
    assert moderate_architecture_name("ok-_.()[]/+&end") == "ok-_.()[]/+&end"


def test_moderate_strips_control_characters() -> None:
    assert moderate_architecture_name("clean\x00name\x07") == "cleanname"


def test_moderate_truncates_to_max_length() -> None:
    raw = "A" * 100
    moderated = moderate_architecture_name(raw)
    assert moderated is not None
    assert len(moderated) == ARCHITECTURE_NAME_MAX_LENGTH == 48


def test_moderate_empty_after_moderation_is_none() -> None:
    assert moderate_architecture_name("") is None
    assert moderate_architecture_name("   ") is None
    assert moderate_architecture_name("!!!@@@###") is None
    assert moderate_architecture_name(None) is None


def test_moderate_collapses_whitespace_left_by_dropped_chars() -> None:
    # Dropping an interior illegal char must not leave a double space in the output.
    assert moderate_architecture_name("a ! b") == "a b"


def test_moderate_is_deterministic_idempotent() -> None:
    raw = "  Weird\t Name!! with  🚀 stuff  "
    first = moderate_architecture_name(raw)
    assert first == moderate_architecture_name(raw)
    # Moderating an already-moderated value is a fixed point.
    assert first is not None
    assert moderate_architecture_name(first) == first
