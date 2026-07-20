"""Dual param ladder admission (VAL-RESLAB-003 / VAL-RESLAB-004 hooks).

Locks explore=124M + promote=350M across settings, PrismContext, static
instantiation, container recheck defaults, Official pin honesty, and stage
labels. Legacy single 150M-only emission default is gone from the dual-ladder
path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.official_comparison import (
    OFFICIAL_EXPLORE_PARAM_CAP,
    OFFICIAL_PARAM_CAP,
    OFFICIAL_PROMOTE_PARAM_CAP,
    ProtocolPin,
    protocol_budget_constants,
)
from prism_challenge.evaluator.param_ladder import (
    EXPLORE_MAX_PARAMETERS,
    LEGACY_SINGLE_PARAM_CAP,
    PARAM_LADDER_CAP_FIELD,
    PARAM_LADDER_PROVISIONAL_CROWN_ELIGIBLE_FIELD,
    PARAM_LADDER_STAGE_FIELD,
    PROMOTE_MAX_PARAMETERS,
    STAGE_EXPLORE,
    STAGE_PROMOTE,
    dual_ladder_summary,
    is_within_stage_cap,
    ladder_labels,
    max_parameters_for_stage,
    normalize_param_ladder_stage,
    promote_path_decision,
    provisional_crown_eligible,
    resolve_max_parameters,
    stage_for_param_count,
)
from prism_challenge.evaluator.sandbox import SandboxViolation
from prism_challenge.evaluator.scoring import build_compute_block
from prism_challenge.evaluator.static_instantiation import (
    PARAM_CAP_RULE,
    check_build_model_static,
)


def test_dual_ladder_constants_locked() -> None:
    """VAL-RESLAB-003: explore 124M / promote 350M; 150M is not the dual default."""
    assert EXPLORE_MAX_PARAMETERS == 124_000_000
    assert PROMOTE_MAX_PARAMETERS == 350_000_000
    assert EXPLORE_MAX_PARAMETERS < LEGACY_SINGLE_PARAM_CAP < PROMOTE_MAX_PARAMETERS
    summary = dual_ladder_summary()
    assert summary["explore_max_parameters"] == 124_000_000
    assert summary["promote_max_parameters"] == 350_000_000
    assert summary["official_default_param_cap"] == 350_000_000
    assert summary["official_default_param_stage"] == STAGE_PROMOTE
    assert summary["provisional_crown_stage"] == STAGE_EXPLORE


def test_settings_default_explore_124m_not_150m() -> None:
    settings = PrismSettings(
        shared_token="secret",
        allow_insecure_signatures=True,
        database_url="sqlite+aiosqlite:////tmp/prism-ladder-settings.sqlite3",
    )
    assert settings.max_parameters == 124_000_000
    assert settings.explore_max_parameters == 124_000_000
    assert settings.promote_max_parameters == 350_000_000
    assert settings.param_ladder_stage == "explore"
    assert settings.max_parameters != LEGACY_SINGLE_PARAM_CAP
    assert settings.max_parameters_for_ladder_stage("explore") == 124_000_000
    assert settings.max_parameters_for_ladder_stage("promote") == 350_000_000
    stage, cap = settings.resolve_admission_max_parameters()
    assert stage == "explore" and cap == 124_000_000
    stage_p, cap_p = settings.resolve_admission_max_parameters("promote")
    assert stage_p == "promote" and cap_p == 350_000_000


def test_prism_context_default_explore_stage() -> None:
    ctx = PrismContext()
    assert ctx.max_parameters == 124_000_000
    assert ctx.param_ladder_stage == STAGE_EXPLORE
    assert ctx.ladder_stage == STAGE_EXPLORE
    assert ctx.ladder_stage_cap == 124_000_000
    assert ctx.max_params == 124_000_000


def test_prism_context_promote_stage() -> None:
    ctx = PrismContext(
        max_parameters=PROMOTE_MAX_PARAMETERS,
        param_ladder_stage=STAGE_PROMOTE,
    )
    assert ctx.ladder_stage == STAGE_PROMOTE
    assert ctx.ladder_stage_cap == 350_000_000
    assert ctx.max_params == 350_000_000


def test_resolve_and_normalize_stage_helpers() -> None:
    assert normalize_param_ladder_stage(None) == STAGE_EXPLORE
    assert normalize_param_ladder_stage("PROMOTE") == STAGE_PROMOTE
    with pytest.raises(ValueError):
        normalize_param_ladder_stage("labs")
    assert max_parameters_for_stage("explore") == 124_000_000
    assert max_parameters_for_stage("promote") == 350_000_000
    stage, cap = resolve_max_parameters(stage="promote")
    assert (stage, cap) == (STAGE_PROMOTE, 350_000_000)
    stage2, cap2 = resolve_max_parameters(max_parameters=50_000)
    assert stage2 == STAGE_EXPLORE and cap2 == 50_000


def test_stage_for_param_count_smallest_admitting() -> None:
    assert stage_for_param_count(1_000_000) == STAGE_EXPLORE
    assert stage_for_param_count(124_000_000) == STAGE_EXPLORE
    assert stage_for_param_count(124_000_001) == STAGE_PROMOTE
    assert stage_for_param_count(350_000_000) == STAGE_PROMOTE
    with pytest.raises(ValueError):
        stage_for_param_count(350_000_001)
    # Prefer keeps promote label even when under explore cap.
    assert stage_for_param_count(1_000, prefer="promote") == STAGE_PROMOTE


def test_static_explore_cap_rejects_between_124m_and_150m() -> None:
    """Models that fit legacy 150M but exceed explore 124M fail under dual default."""
    # 50304 * 2500 = 125_760_000 — over 124M, under legacy 150M
    arch = (
        "import torch\n"
        "from torch import nn\n\n"
        "def build_model(ctx):\n"
        "    return nn.Embedding(50304, 2500)\n"
    )
    ctx = PrismContext()  # explore default
    with pytest.raises(SandboxViolation) as raised:
        check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=ctx)
    assert raised.value.evidence[0].rule_id == PARAM_CAP_RULE


def test_static_promote_cap_admits_between_124m_and_350m() -> None:
    """Same model admitted under promote stage 350M."""
    arch = (
        "import torch\n"
        "from torch import nn\n\n"
        "def build_model(ctx):\n"
        "    return nn.Embedding(50304, 2500)\n"
    )
    ctx = PrismContext(
        max_parameters=PROMOTE_MAX_PARAMETERS,
        param_ladder_stage=STAGE_PROMOTE,
    )
    count = check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=ctx)
    assert count == 50304 * 2500
    assert is_within_stage_cap(count, STAGE_PROMOTE)
    assert not is_within_stage_cap(count, STAGE_EXPLORE)


def test_static_promote_cap_rejects_over_350m() -> None:
    # 50304 * 7000 = 352_128_000 > 350M
    arch = (
        "import torch\n"
        "from torch import nn\n\n"
        "def build_model(ctx):\n"
        "    return nn.Embedding(50304, 7000)\n"
    )
    ctx = PrismContext(
        max_parameters=PROMOTE_MAX_PARAMETERS,
        param_ladder_stage=STAGE_PROMOTE,
    )
    with pytest.raises(SandboxViolation) as raised:
        check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=ctx)
    assert raised.value.evidence[0].rule_id == PARAM_CAP_RULE


def test_ladder_labels_stage_on_scores_payload() -> None:
    explore = ladder_labels(STAGE_EXPLORE, param_count=1_000_000, score_valid=True)
    assert explore[PARAM_LADDER_STAGE_FIELD] == "explore"
    assert explore[PARAM_LADDER_CAP_FIELD] == 124_000_000
    assert explore[PARAM_LADDER_PROVISIONAL_CROWN_ELIGIBLE_FIELD] is True

    promote = ladder_labels(STAGE_PROMOTE, param_count=200_000_000, score_valid=True)
    assert promote[PARAM_LADDER_STAGE_FIELD] == "promote"
    assert promote[PARAM_LADDER_CAP_FIELD] == 350_000_000
    assert promote[PARAM_LADDER_PROVISIONAL_CROWN_ELIGIBLE_FIELD] is False


def test_provisional_crown_eligible_explore_only() -> None:
    """VAL-RESLAB-004 hook: only qualifying explore-stage scores provisional-crown."""
    assert provisional_crown_eligible(stage="explore", param_count=1000) is True
    assert provisional_crown_eligible(stage="promote", param_count=1000) is False
    assert provisional_crown_eligible(stage="explore", score_valid=False) is False
    assert provisional_crown_eligible(stage="explore", param_count=200_000_000) is False


def test_promote_path_confirm_or_revoke_hooks() -> None:
    """VAL-RESLAB-005 surface hooks (full weights machine may expand later)."""
    assert (
        promote_path_decision(
            provisional_stage="explore",
            promote_stage="promote",
            promote_valid=True,
            promote_beats_provisional=True,
        )
        == "confirm"
    )
    assert (
        promote_path_decision(
            provisional_stage="explore",
            promote_stage="promote",
            promote_valid=False,
        )
        == "revoke"
    )
    assert (
        promote_path_decision(
            provisional_stage="explore",
            promote_stage="promote",
            promote_valid=True,
            promote_beats_provisional=False,
        )
        == "revoke"
    )
    assert (
        promote_path_decision(
            provisional_stage="promote",
            promote_stage="explore",
            promote_valid=True,
        )
        == "ineligible"
    )


def test_official_pin_defaults_to_promote_with_dual_honesty() -> None:
    """Official pin follows promote cap; dual ladder numbers still visible."""
    assert OFFICIAL_PARAM_CAP == 350_000_000
    assert OFFICIAL_EXPLORE_PARAM_CAP == 124_000_000
    assert OFFICIAL_PROMOTE_PARAM_CAP == 350_000_000
    pin = ProtocolPin()
    assert pin.param_cap == 350_000_000
    assert pin.param_ladder_stage == "promote"
    dumped = pin.as_dict()
    assert dumped["param_cap"] == 350_000_000
    assert dumped["param_ladder_stage"] == "promote"
    assert dumped["param_ladder"]["explore_max_parameters"] == 124_000_000
    assert dumped["param_ladder"]["promote_max_parameters"] == 350_000_000

    explore_pin = ProtocolPin(
        param_cap=124_000_000,
        param_ladder_stage="explore",
    )
    assert explore_pin.param_cap == 124_000_000
    assert explore_pin.as_dict()["param_ladder_stage"] == "explore"

    constants = protocol_budget_constants()
    assert constants["param_cap"] == 350_000_000 != LEGACY_SINGLE_PARAM_CAP
    assert constants["explore_max_parameters"] == 124_000_000
    assert constants["promote_max_parameters"] == 350_000_000


def test_compute_block_carries_ladder_stage_labels() -> None:
    block = build_compute_block(
        gpu_count=1,
        world_size=1,
        nproc_per_node=1,
        device="cpu",
        model_params=1_050_000,
        param_ladder_stage="explore",
        param_ladder_cap=124_000_000,
    )
    assert block["model_params"] == 1_050_000
    assert block["param_ladder_stage"] == "explore"
    assert block["param_ladder_cap"] == 124_000_000


def test_app_context_wires_settings_ladder(tmp_path: Path) -> None:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'prism.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        distributed_contract_policy="off",
        param_ladder_stage="promote",
        max_parameters=350_000_000,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        # App state holds the worker with dual-ladder ctx.
        ctx = client.app.state.worker.ctx
        assert ctx.max_parameters == 350_000_000
        assert ctx.param_ladder_stage == "promote"


def test_pipeline_explore_default_rejects_125m_model(tmp_path: Path) -> None:
    """Black-box: default app explore 124M rejects a ~126M embedding before GPU."""
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'prism.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        distributed_contract_policy="off",
    )
    # 50304 * 2500 = 125_760_000 > explore 124M
    arch = (
        "import torch\n"
        "from torch import nn\n\n"
        "def build_model(ctx):\n"
        "    return nn.Embedding(50304, 2500)\n"
    )
    train = (
        "from architecture import build_model\n\n"
        "def train(ctx):\n"
        "    build_model(ctx)\n"
        "    return None\n"
    )
    with TestClient(create_app(settings)) as client:
        code = two_script_bundle(arch_code=arch, train_code=train)
        payload = {"code": code, "filename": "bundle.zip"}
        body = json.dumps(payload, separators=(",", ":")).encode()
        response = client.post(
            "/v1/submissions",
            content=body,
            headers={
                **signed_headers("secret", body, hotkey="hk-ladder", nonce="ladder-over-1"),
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200, response.text
        submission_id = response.json()["id"]
        proc = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert proc.status_code == 200, proc.text
        row = client.get(
            f"/v1/submissions/{submission_id}",
            headers={"Authorization": "Bearer secret"},
        )
        # Some routes require signed headers; fall back to internal row if needed.
        if row.status_code != 200:
            import anyio

            from prism_challenge.db import Database

            async def _fetch() -> dict:
                db = Database(settings.resolved_database_path)
                await db.init()
                async with db.connect() as conn:
                    cursor = await conn.execute(
                        "SELECT status, error FROM submissions WHERE id=?",
                        (submission_id,),
                    )
                    got = await cursor.fetchone()
                return dict(got) if got is not None else {}

            data = anyio.run(_fetch)
        else:
            data = row.json()
        assert data.get("status") == "rejected", data
        assert "parameter cap" in str(data.get("error") or "").lower(), data
