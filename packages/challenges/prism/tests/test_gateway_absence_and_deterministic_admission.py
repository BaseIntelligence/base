"""Gateway absence + deterministic Prism admission / migration tests."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

import pytest
import yaml

from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.evaluator import source_similarity
from prism_challenge.models import SubmissionStatus


def test_llm_review_and_report_modules_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("prism_challenge.evaluator.llm_review")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("prism_challenge.evaluator.architecture_report")


def test_langchain_openai_not_installed() -> None:
    names = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
    assert "langchain-openai" not in names
    assert "openai" not in names


def test_clean_deterministic_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PRISM_LLM_GATEWAY_URL",
        "PRISM_GATEWAY_TOKEN",
        "BASE_LLM_GATEWAY_URL",
        "BASE_GATEWAY_TOKEN",
        "PRISM_LLM_REVIEW_ENABLED",
        "PRISM_COMPONENT_AGENT_ENABLED",
        "PRISM_COMPONENT_HOLD_LOW_CONFIDENCE",
    ):
        monkeypatch.delenv(name, raising=False)
    PrismSettings(database_path="/tmp/prism-absence.sqlite3", shared_token="test-token")
    assert "llm_gateway_url" not in PrismSettings.model_fields
    assert "llm_review_enabled" not in PrismSettings.model_fields
    # VAL-GATE-007: nondeterministic component-agent knobs are gone.
    for field_name in (
        "component_agent_enabled",
        "component_agent_required",
        "component_agent_model",
        "component_agent_min_confidence",
        "component_agent_transfer_confidence",
        "component_agent_same_threshold",
        "component_agent_hold_threshold",
        "component_agent_candidate_top_k",
        "component_agent_mermaid_enabled",
        "component_hold_low_confidence",
    ):
        assert field_name not in PrismSettings.model_fields


@pytest.mark.parametrize(
    "kwargs",
    [
        {"llm_review_enabled": True},
        {"llm_gateway_url": "http://gateway/llm/v1"},
        {"llm_gateway_token": "tok"},
        {"component_agent_enabled": True},
        {"component_agent_model": "gpt-4o"},
        {"component_hold_low_confidence": True},
        {"component_agent_same_threshold": 0.9},
    ],
)
def test_legacy_llm_settings_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="removed Prism LLM"):
        PrismSettings(shared_token="x", **kwargs)


def test_legacy_component_agent_env_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRISM_COMPONENT_AGENT_ENABLED", "true")
    with pytest.raises(ValueError, match="removed Prism LLM"):
        PrismSettings(shared_token="x")


def test_legacy_llm_env_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRISM_LLM_GATEWAY_URL", "http://example/llm/v1")
    with pytest.raises(ValueError, match="removed Prism LLM"):
        PrismSettings(shared_token="x")


@pytest.mark.asyncio
async def test_openapi_has_no_report_or_held(tmp_path: Path) -> None:
    from httpx import ASGITransport, AsyncClient

    from prism_challenge.app import create_app

    db_path = tmp_path / "openapi.sqlite3"
    settings = PrismSettings(
        database_path=db_path,
        shared_token="token",
        allow_insecure_signatures=True,
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")
    assert response.status_code == 200
    openapi = response.json()
    paths = "\n".join(openapi.get("paths", {}))
    assert "/architectures/{architecture_id}/report" not in paths
    body = yaml.safe_dump(openapi)
    assert "llm_review" not in body
    assert "gateway_token" not in body
    # held status must not be exposed as a current submission enum value
    assert '"held"' not in body and "held" not in body.lower().split("enum")[-1][:200]


def test_submission_status_has_no_held() -> None:
    assert not hasattr(SubmissionStatus, "HELD")
    assert "held" not in {item.value for item in SubmissionStatus}


@pytest.mark.asyncio
async def test_legacy_held_rows_migrate_to_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "migrate.sqlite3"
    db = Database(db_path)
    await db.init()
    async with db.connect() as conn:
        await conn.execute(
            "INSERT INTO miners(hotkey, first_seen, last_seen) VALUES ('hk','t','t')"
        )
        await conn.execute(
            "INSERT INTO epochs(id, starts_at, ends_at, status) VALUES (1,'t','t','active')"
        )
        await conn.execute(
            "INSERT INTO submissions(id, hotkey, epoch_id, filename, code, code_hash, metadata, "
            "status, error, created_at, updated_at) "
            "VALUES ('s-held','hk',1,'architecture.py','print(1)','h', '{}', "
            "'held', 'awaiting review', 't', 't')"
        )
        await conn.execute(
            "INSERT INTO submissions(id, hotkey, epoch_id, filename, code, code_hash, metadata, "
            "status, error, created_at, updated_at) "
            "VALUES ('s-ok','hk',1,'architecture.py','print(2)','h2', '{}', "
            "'pending', NULL, 't', 't')"
        )
    # re-run migrations explicitly
    async with db.connect() as conn:
        from prism_challenge.db import _migrate_legacy_llm_state

        await _migrate_legacy_llm_state(conn)
        rows = await conn.execute_fetchall("SELECT id, status, error FROM submissions ORDER BY id")
    statuses = {row["id"]: (row["status"], row["error"]) for row in rows}
    assert statuses["s-held"][0] == "rejected"
    # Preserve prior operator error text if present; never silently approve.
    assert statuses["s-held"][1] is not None
    assert statuses["s-ok"][0] == "pending"


def test_similarity_quarantine_band_is_deterministic_reject_surface() -> None:
    # classify_duplicate still exposes outcome='quarantine' for the threshold band,
    # but admission maps held/quarantine to terminal reject (never SubmissionStatus.held).
    matrix = source_similarity.DuplicateThresholdMatrix(
        quarantine_source_similarity=0.1,
        same_architecture_similarity=0.99,
        static_reject_similarity=0.96,
        exact_source_similarity=0.98,
    )
    left = source_similarity.snapshot_from_named_sources(
        [("architecture.py", "x = 1\n" * 20), ("training.py", "y = 2\n" * 20)]
    )
    right_payload = left.to_payload()
    decision = source_similarity.classify_duplicate(
        submission_id="a",
        code_hash="different",
        snapshot=left,
        architecture_graph={"nodes": ["n1"]},
        rows=[
            {
                "submission_id": "b",
                "hotkey": "hk",
                "code_hash": "other",
                "architecture_id": "arch-b",
                "architecture_graph_hash": "different-graph",
                "architecture_graph": {"nodes": ["n2"]},
                **right_payload,
            }
        ],
        thresholds=matrix,
    )
    assert decision.held or decision.rejected
    assert decision.outcome in {"quarantine", "reject", "attach", "allow"}
