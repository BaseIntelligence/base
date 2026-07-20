"""P3 full-scale readiness + public K≥3 lock + research annex (VAL-SCALE-015/016/017).

Product surface only — dry-run readiness without 100BT spend. Missing mounts must
yield honest BLOCKED_with_reason (never invented READY). Research protocol annex is
explicitly non-emission. No Lium / no tee / no emission rewrite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.complete_view import (
    COMPLETE_VIEW_SCHEMA,
    COMPLETE_VIEW_SCORECARD_ID,
)
from prism_challenge.evaluator.dataset import (
    FINEWEB_EDU_SUBSETS,
    LOCKED_MANIFEST_FILENAME,
)
from prism_challenge.evaluator.modes import (
    FULL_SCALE_PHASE_1_TOKEN_TARGET,
    FULL_SCALE_PHASE_2_TOKEN_TARGET,
    GPU_PROXY_TOKEN_TARGET,
    execution_mode_from_value,
)
from prism_challenge.evaluator.official_comparison import (
    OFFICIAL_MIN_PUBLIC_SEEDS,
    SCORECARD_ID,
)
from prism_challenge.evaluator.scale_eval import (
    RESEARCH_PROTOCOL_ANNEX_ID,
    RESEARCH_PROTOCOL_ANNEX_SCHEMA,
    SCALE_LADDER_STAGES,
    SCALE_P3_DATASET_SUBSET_PHASE_1,
    SCALE_P3_DATASET_SUBSET_PHASE_2,
    SCALE_P3_EXECUTION_MODE,
    SCALE_P3_SEEDS,
    assert_public_multi_seed_pin,
    assert_research_protocol_annex,
    densify_entrypoints,
    probe_full_scale_readiness,
    research_protocol_annex,
    scale_ladder_document,
    scale_p3_protocol_pin,
    scale_pin_public_ok,
    scale_product_snapshot,
    tee_package_absent,
)
from prism_challenge.evaluator.schemas import ExecutionMode
from prism_challenge.runtime_config import runtime_policy_defaults

# ---------------------------------------------------------------------------
# VAL-SCALE-015: full_scale / sample-100BT readiness path
# ---------------------------------------------------------------------------


def test_execution_mode_full_scale_and_subset_contracts() -> None:
    assert ExecutionMode.FULL_SCALE_EVAL.value == "full_scale_eval"
    assert execution_mode_from_value("full_scale_eval") is ExecutionMode.FULL_SCALE_EVAL
    assert SCALE_P3_EXECUTION_MODE == ExecutionMode.FULL_SCALE_EVAL.value
    assert SCALE_P3_DATASET_SUBSET_PHASE_1 == "sample-10BT"
    assert SCALE_P3_DATASET_SUBSET_PHASE_2 == "sample-100BT"
    assert FINEWEB_EDU_SUBSETS["sample-100BT"]["official_mode"] == "full_scale_eval"
    phase2_tokens = int(FINEWEB_EDU_SUBSETS["sample-100BT"]["token_count"])
    assert phase2_tokens == FULL_SCALE_PHASE_2_TOKEN_TARGET
    assert int(FINEWEB_EDU_SUBSETS["sample-10BT"]["token_count"]) == GPU_PROXY_TOKEN_TARGET
    assert FULL_SCALE_PHASE_1_TOKEN_TARGET == GPU_PROXY_TOKEN_TARGET


def test_runtime_wires_full_scale_phase2_sample_100bt() -> None:
    targets = runtime_policy_defaults(PrismSettings())["execution_mode_targets"]
    full = targets["full_scale_eval"]
    assert full["phase_2_dataset_subset"] == "sample-100BT"
    assert full["phase_1_dataset_subset"] == "sample-10BT"
    assert int(full["phase_2_dataset_tokens"]) == 100_000_000_000
    assert full.get("official_score") is True


def test_probe_missing_mounts_is_blocked_with_reason(tmp_path: Path) -> None:
    """Dry-run readiness must not require 100BT spend; missing mount → BLOCKED."""
    missing = tmp_path / "no-such-mount"
    result = probe_full_scale_readiness(
        train_data_dir=missing,
        val_data_dir=missing / "val",
        phase2_data_dir=missing / "sample-100BT",
        dry_run=True,
    )
    payload = result.as_dict()
    assert payload["status"] == "BLOCKED"
    assert payload["dry_run"] is True
    assert payload["emission_changed"] is False
    assert payload["requires_100bt_spend"] is False
    assert payload["execution_mode"] == "full_scale_eval"
    assert payload["phase_2_dataset_subset"] == "sample-100BT"
    reasons = payload["reasons"]
    assert reasons
    assert any("BLOCKED" in r or "missing" in r.lower() for r in reasons)
    # Machine-readable block reasons
    assert any("train" in r.lower() or "mount" in r.lower() for r in reasons)
    assert result.ok is False


def test_probe_default_settings_paths_honest_when_absent() -> None:
    """Default config paths (/data/fineweb-edu/...) are usually unmounted here → BLOCKED."""
    result = probe_full_scale_readiness(dry_run=True)
    d = result.as_dict()
    assert d["status"] in {"BLOCKED", "READY"}
    assert d["dry_run"] is True
    assert d["requires_100bt_spend"] is False
    assert d["emission_changed"] is False
    if d["status"] == "BLOCKED":
        assert d["reasons"]
        assert "BLOCKED_with_reason" in d["status_label"] or d["status_label"].startswith(
            "BLOCKED"
        )
    # Never invent scores / never claim full 100BT train ran.
    assert d.get("full_scale_train_executed") is False
    assert "invented_metrics" not in d or d.get("invented_metrics") is False


def test_probe_ready_when_minimal_mounts_present(tmp_path: Path) -> None:
    """Synthetic locked mounts (manifest marker only) → READY dry-run without spend."""
    train = tmp_path / "fineweb-edu" / "train"
    val = tmp_path / "fineweb-edu" / "val"
    phase2 = tmp_path / "fineweb-edu" / "sample-100BT"
    for root in (train, val, phase2):
        root.mkdir(parents=True)
        (root / LOCKED_MANIFEST_FILENAME).write_text(
            json.dumps({"schema_version": "prism_locked_data.v1", "subset": root.name}),
            encoding="utf-8",
        )
        # Optional shard marker so emptiness checks pass.
        shard = root / "shard-000.jsonl"
        shard.write_text('{"id":"d0","text":"hello full scale"}\n', encoding="utf-8")

    result = probe_full_scale_readiness(
        train_data_dir=train,
        val_data_dir=val,
        phase2_data_dir=phase2,
        dry_run=True,
        require_manifest=True,
    )
    d = result.as_dict()
    assert d["status"] == "READY"
    assert d["ok"] is True
    assert d["dry_run"] is True
    assert d["requires_100bt_spend"] is False
    assert d["full_scale_train_executed"] is False
    assert d["emission_changed"] is False
    assert d["phase_2_dataset_subset"] == "sample-100BT"
    assert not d["reasons"]


def test_probe_phase2_missing_only_is_blocked(tmp_path: Path) -> None:
    train = tmp_path / "train"
    val = tmp_path / "val"
    for root in (train, val):
        root.mkdir()
        (root / LOCKED_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
        (root / "s.jsonl").write_text("x\n", encoding="utf-8")
    result = probe_full_scale_readiness(
        train_data_dir=train,
        val_data_dir=val,
        phase2_data_dir=tmp_path / "missing-100bt",
        dry_run=True,
    )
    assert result.status == "BLOCKED"
    joined = " ".join(result.reasons).lower()
    assert "100bt" in joined or "phase_2" in joined or "phase2" in joined or "sample-100" in joined


def test_scale_p3_protocol_pin_public_k_and_ladder() -> None:
    pin = scale_p3_protocol_pin()
    assert len(pin.seeds) >= OFFICIAL_MIN_PUBLIC_SEEDS
    assert pin.seeds == SCALE_P3_SEEDS or len(pin.seeds) >= 3
    assert pin.primary_form == "heldout_delta"
    assert_public_multi_seed_pin(pin)
    guard = scale_pin_public_ok(pin)
    assert guard.ok is True
    # P3 documents full_scale stage without silently dropping promote cap.
    fields = scale_product_snapshot()
    assert "p3_pin" in fields or "full_scale" in json.dumps(fields).lower()
    ladder = scale_ladder_document()
    assert ladder["stages"] == list(SCALE_LADDER_STAGES) or set(SCALE_LADDER_STAGES).issubset(
        set(ladder["stages"])
    )
    assert "explore" in ladder["stages"]
    assert "promote" in ladder["stages"]
    assert "full_scale" in ladder["stages"]
    assert ladder["min_public_seeds"] == OFFICIAL_MIN_PUBLIC_SEEDS
    assert ladder["k1_is_provisional"] is True


# ---------------------------------------------------------------------------
# VAL-SCALE-016: public multi-seed protocol lock
# ---------------------------------------------------------------------------


def test_public_k_ge_3_lock_constants() -> None:
    assert OFFICIAL_MIN_PUBLIC_SEEDS >= 3
    pin_ok = scale_p3_protocol_pin()
    assert len(pin_ok.seeds) >= 3
    with pytest.raises(ValueError, match="K≥|public|seed"):
        scale_p3_protocol_pin(seeds=(1337,), require_public_k=True)
    # Provisional path allows K=1 when explicitly opted out of public K.
    pin_prov = scale_p3_protocol_pin(seeds=(1337,), require_public_k=False)
    assert len(pin_prov.seeds) == 1
    guard = scale_pin_public_ok(pin_prov)
    assert guard.ok is False
    assert any("seed_count_below_public_min" in r for r in guard.reasons)


def test_scale_ladder_document_marks_k1_provisional() -> None:
    doc = scale_ladder_document()
    assert doc["schema"] == "prism_scale_ladder.v1"
    assert doc["min_public_seeds"] >= 3
    assert doc["k1_is_provisional"] is True
    assert "explore" in doc["stages"]
    assert "promote" in doc["stages"]
    assert "full_scale" in doc["stages"]
    # Prior K=1 language present for honesty.
    assert "provisional" in json.dumps(doc).lower()


# ---------------------------------------------------------------------------
# VAL-SCALE-017: research protocol annex explicitly non-emission
# ---------------------------------------------------------------------------


def test_research_protocol_annex_non_emission() -> None:
    annex = research_protocol_annex()
    assert annex["schema"] == RESEARCH_PROTOCOL_ANNEX_SCHEMA
    assert annex["annex_id"] == RESEARCH_PROTOCOL_ANNEX_ID
    assert annex["emission_weight_crown"] is False
    assert annex["non_emission"] is True
    assert annex["silent_emission_rewrite"] is False
    assert annex["complete_view_scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    assert annex["complete_view_schema"] == COMPLETE_VIEW_SCHEMA
    assert annex["multimetric_scorecard_id"] == SCORECARD_ID
    assert "research" in annex["role"].lower() or annex["role"] == "scientific_research_grade"
    assert_research_protocol_annex(annex)


def test_research_protocol_annex_rejects_emission_crown() -> None:
    bad = research_protocol_annex()
    bad = dict(bad)
    bad["emission_weight_crown"] = True
    with pytest.raises(ValueError, match="emission|non.emission|crown"):
        assert_research_protocol_annex(bad)
    bad2 = research_protocol_annex()
    bad2 = dict(bad2)
    bad2["non_emission"] = False
    with pytest.raises(ValueError, match="non.emission|emission"):
        assert_research_protocol_annex(bad2)


def test_densify_and_snapshot_expose_p3_surfaces() -> None:
    ep = densify_entrypoints()
    helpers = ep.get("scale_helpers") or {}
    assert helpers.get("p3_pin") == "scale_p3_protocol_pin" or "p3" in json.dumps(ep).lower()
    assert "full_scale_readiness" in ep or "probe_full_scale_readiness" in json.dumps(ep)
    snap = scale_product_snapshot()
    assert snap["min_public_seeds"] >= 3
    assert snap["tee_package_absent"] is True
    assert tee_package_absent() is True
    # Annex pointer present and non-emission.
    annex = snap.get("research_protocol_annex") or research_protocol_annex()
    assert annex["non_emission"] is True
    assert annex["emission_weight_crown"] is False
