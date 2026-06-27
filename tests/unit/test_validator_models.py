from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

from sqlalchemy import Enum as SQLAlchemyEnum

from base.db import (
    Base,
    Validator,
    ValidatorHealthEvent,
    ValidatorHealthEventType,
    ValidatorRequestNonce,
    ValidatorStatus,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
VALIDATOR_MIGRATION = ROOT_DIR / "alembic/versions/0003_create_validator_registry.py"


def test_validator_models_construct_and_register_metadata() -> None:
    validator = Validator(
        hotkey="5FvalidatorHotkey",
        uid=3,
        status=ValidatorStatus.ONLINE,
        capabilities=["cpu", "gpu"],
        version="1.2.3",
        last_seen_meta={"broker": "ok", "concurrency": 1},
    )
    event = ValidatorHealthEvent(
        validator_hotkey="5FvalidatorHotkey",
        event=ValidatorHealthEventType.REGISTERED,
        message="first registration",
    )
    nonce = ValidatorRequestNonce(
        hotkey="5FvalidatorHotkey", nonce="n-1", body_hash="abc"
    )

    assert validator.hotkey == "5FvalidatorHotkey"
    assert validator.capabilities == ["cpu", "gpu"]
    assert validator.status == ValidatorStatus.ONLINE
    assert event.event == ValidatorHealthEventType.REGISTERED
    assert nonce.nonce == "n-1"

    assert "validators" in Base.metadata.tables
    assert "validator_health_events" in Base.metadata.tables
    assert "validator_request_nonces" in Base.metadata.tables


def test_validator_hotkey_is_unique() -> None:
    table = Base.metadata.tables["validators"]
    hotkey_column = table.c.hotkey
    assert hotkey_column.unique is True
    assert hotkey_column.nullable is False


def test_validator_status_enum_is_non_native_varchar() -> None:
    status_column = Validator.__table__.c.status
    status_type = cast(SQLAlchemyEnum, status_column.type)
    assert status_type.name == "validator_status"
    assert status_type.native_enum is False
    assert status_type.enums == [status.value for status in ValidatorStatus]


def test_validator_event_enum_matches_model() -> None:
    event_column = ValidatorHealthEvent.__table__.c.event
    event_type = cast(SQLAlchemyEnum, event_column.type)
    assert event_type.name == "validator_health_event_type"
    assert event_type.native_enum is False
    assert event_type.enums == [event.value for event in ValidatorHealthEventType]


def _migration_enum_literals(name: str) -> list[str]:
    migration_ast = ast.parse(VALIDATOR_MIGRATION.read_text(encoding="utf-8"))
    for node in migration_ast.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            continue
        enum_call = node.value
        assert isinstance(enum_call, ast.Call)
        return [
            arg.value
            for arg in enum_call.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
    raise AssertionError(f"{name} enum declaration not found in migration")


def test_migration_enum_literals_match_model_enums() -> None:
    assert _migration_enum_literals("validator_status") == [
        status.value for status in ValidatorStatus
    ]
    assert _migration_enum_literals("validator_health_event_type") == [
        event.value for event in ValidatorHealthEventType
    ]
