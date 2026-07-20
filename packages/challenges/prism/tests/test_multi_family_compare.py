"""Multi-family Official Compare under one ProtocolPin (VAL-ARXEVAL-004/005/006)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from prism_challenge.evaluator.multi_family_compare import (
    ARXIV_FAIR_EVAL_FAMILY_IDS,
    DEEPLOOP_CONTROL_FAMILY_ID,
    FRONTIER_FAIR_EVAL_FAMILY_IDS,
    IMP_BASELINE_FAMILY_IDS,
    MULTI_FAMILY_REPORT_SCHEMA,
    NOVEL_ARXIV_FAMILY_IDS,
    NOVEL_FRONTIER_FAMILY_IDS,
    MultiFamilyArtifactsMissingError,
    agnostic_equal_metrics_equal_rank,
    explore_protocol_pin,
    package_unknown_style_families,
    rank_multi_family_records,
    records_from_lab_gpu_artifacts_multi,
    run_multi_family_lab_gpu_host_compare,
    run_multi_family_official_compare,
)
from prism_challenge.evaluator.official_compare_harness import (
    IMAGE_PIN_LABEL,
    LAB_GPU_DEFAULT_SEED,
    PROVIDER_TRUST_LABEL,
    SCORE_CLASS_FIXTURE,
    SCORE_CLASS_LAB_GPU,
    protocol_pin_hash,
    synth_challenge_manifest,
)
from prism_challenge.evaluator.official_comparison import (
    OFFICIAL_DEFAULT_SEEDS,
    OFFICIAL_EXPLORE_PARAM_CAP,
    OFFICIAL_EXPLORE_STAGE,
    OfficialScoreRecord,
    official_record_from_manifest,
)
from prism_challenge.seed_packaging import REQUIRED_ENTRY_SCRIPTS, SEED_FAMILIES


def test_explore_protocol_pin_matched_fairness_fields() -> None:
    """VAL-ARXEVAL-004: single pin holds token_budget, seeds, tokenizer, seq, batch, explore."""
    pin = explore_protocol_pin()
    assert pin.tokenizer == "gpt2"
    assert pin.param_ladder_stage == OFFICIAL_EXPLORE_STAGE
    assert pin.param_cap == OFFICIAL_EXPLORE_PARAM_CAP
    assert pin.primary_form == "heldout_delta"
    assert pin.force_iter_train_batches is True
    assert pin.seq_len > 0 and pin.batch_size > 0
    assert pin.token_budget > 0
    assert len(pin.seeds) >= 3  # prefer K≥3 default
    assert pin.seeds == OFFICIAL_DEFAULT_SEEDS
    d = pin.as_dict()
    assert d["param_ladder"]["stage"] == "explore"
    assert d["wall_clock_never_ranks"] is True
    # K=1 labelled path remains possible.
    pin_k1 = explore_protocol_pin(seeds=(1337,))
    assert pin_k1.seeds == (1337,)
    assert pin_k1.tokenizer == pin.tokenizer
    assert pin_k1.token_budget == pin.token_budget
    assert pin_k1.param_cap == pin.param_cap


def test_arxiv_family_registry_complete() -> None:
    assert set(IMP_BASELINE_FAMILY_IDS).issubset(SEED_FAMILIES)
    assert set(NOVEL_ARXIV_FAMILY_IDS).issubset(SEED_FAMILIES)
    assert len(ARXIV_FAIR_EVAL_FAMILY_IDS) == 5
    for fid in ARXIV_FAIR_EVAL_FAMILY_IDS:
        assert fid in SEED_FAMILIES


def test_frontier_family_registry_complete() -> None:
    """VAL-FRNTEVAL cup: Imp + deeploop winner + three frontier-inspired packs."""
    assert set(IMP_BASELINE_FAMILY_IDS).issubset(SEED_FAMILIES)
    assert DEEPLOOP_CONTROL_FAMILY_ID in SEED_FAMILIES
    assert set(NOVEL_FRONTIER_FAMILY_IDS).issubset(SEED_FAMILIES)
    assert len(FRONTIER_FAIR_EVAL_FAMILY_IDS) == 6
    assert FRONTIER_FAIR_EVAL_FAMILY_IDS[:2] == IMP_BASELINE_FAMILY_IDS
    assert FRONTIER_FAIR_EVAL_FAMILY_IDS[2] == DEEPLOOP_CONTROL_FAMILY_ID
    for fid in FRONTIER_FAIR_EVAL_FAMILY_IDS:
        assert fid in SEED_FAMILIES
    # Shared explore pin fields match prior arxiv fair spirit (500k applied at ops layer).
    pin = explore_protocol_pin(seeds=(1337,), token_budget=500_000)
    assert pin.tokenizer == "gpt2"
    assert pin.param_ladder_stage == OFFICIAL_EXPLORE_STAGE
    assert pin.param_cap == OFFICIAL_EXPLORE_PARAM_CAP
    assert pin.token_budget == 500_000
    assert pin.seeds == (1337,)
    assert pin.seq_len == 128
    assert pin.batch_size == 4


def test_package_n_families_unknown_style_contract(tmp_path: Path) -> None:
    packed = package_unknown_style_families(tmp_path, ARXIV_FAIR_EVAL_FAMILY_IDS)
    assert set(packed) == set(ARXIV_FAIR_EVAL_FAMILY_IDS)
    hashes = {item.content_sha256 for item in packed.values()}
    assert len(hashes) == len(ARXIV_FAIR_EVAL_FAMILY_IDS)  # distinct architectures
    for item in packed.values():
        names = set(item.entry_names)
        for required in REQUIRED_ENTRY_SCRIPTS:
            assert required in names
        with zipfile.ZipFile(item.zip_path) as zf:
            zip_names = set(zf.namelist())
        assert not any(n.endswith((".pem", ".key", ".env", ".pt", ".bin")) for n in zip_names)


def test_architecture_agnostic_equal_metrics_equal_rank_keys() -> None:
    """VAL-ARXEVAL-005: equal challenge metrics equal-rank across fictitious family tags."""
    keys = agnostic_equal_metrics_equal_rank(
        heldout_delta=0.9,
        bpb=1.2,
        family_tags=("transformer", "mamba", "deeploop", "gated_delta", "alien_xyz"),
    )
    assert len(keys) == 5
    # Rank keys ignore family tag text except total-order label — metrics portion equal.
    # Compare metric prefix of the key (everything before the label).
    metric_prefixes = [k[:-1] for k in keys]
    assert all(p == metric_prefixes[0] for p in metric_prefixes)


def test_frontier_fixture_score_table_under_shared_pin(tmp_path: Path) -> None:
    """VAL-FRNTEVAL-004 fixture path: one pin for Imp+deeploop+frontier sides."""
    pin = explore_protocol_pin(seeds=(1337,), token_budget=500_000)
    report = run_multi_family_official_compare(
        tmp_path / "frontier",
        family_ids=FRONTIER_FAIR_EVAL_FAMILY_IDS,
        pin=pin,
        device_class="fixture",
        score_class=SCORE_CLASS_FIXTURE,
    )
    assert report["schema"] == MULTI_FAMILY_REPORT_SCHEMA
    assert report["score_class"] == SCORE_CLASS_FIXTURE
    assert report["labels"]["provider_trust"] == PROVIDER_TRUST_LABEL
    assert report["labels"]["image_pin"] == IMAGE_PIN_LABEL
    assert report["labels"]["prism_tee_product"] is False
    assert "real_provider_tee" not in report
    assert report["protocol_hash"] == protocol_pin_hash(pin)
    assert report["pin"]["token_budget"] == 500_000
    assert report["pin"]["seeds"] == [1337]
    assert set(report["family_ids"]) == set(FRONTIER_FAIR_EVAL_FAMILY_IDS)
    assert len(report["score_table"]) == len(FRONTIER_FAIR_EVAL_FAMILY_IDS)
    assert report["validity"]["architecture_agnostic_path"] is True
    assert report["validity"]["all_sides_same_pin_hash"] is True
    pin_disk = json.loads(Path(report["pin_path"]).read_text(encoding="utf-8"))
    assert set(pin_disk["families"]) == set(FRONTIER_FAIR_EVAL_FAMILY_IDS)


def test_multi_family_fixture_score_table_under_shared_pin(tmp_path: Path) -> None:
    """Fixture multi-family run: shared pin + table + agnostic path (labelled fixture)."""
    pin = explore_protocol_pin()
    report = run_multi_family_official_compare(
        tmp_path,
        family_ids=ARXIV_FAIR_EVAL_FAMILY_IDS,
        pin=pin,
        device_class="fixture",
        score_class=SCORE_CLASS_FIXTURE,
    )
    assert report["schema"] == MULTI_FAMILY_REPORT_SCHEMA
    assert report["score_class"] == SCORE_CLASS_FIXTURE
    assert report["labels"]["fixture_labelled"] is True
    assert report["labels"]["provider_trust"] == PROVIDER_TRUST_LABEL
    assert report["labels"]["image_pin"] == IMAGE_PIN_LABEL
    assert report["labels"]["prism_tee_product"] is False
    assert "real_provider_tee" not in report
    assert report["protocol_hash"] == protocol_pin_hash(pin)
    assert report["pin"]["param_ladder_stage"] == "explore"
    assert report["pin"]["tokenizer"] == "gpt2"
    assert report["validity"]["matched_budget"] is True
    assert report["validity"]["architecture_agnostic_path"] is True
    assert report["validity"]["wall_clock_never_ranks"] is True
    assert report["ranking"]["architecture_agnostic"] is True
    assert report["ranking"]["family_branch_in_rank_key"] is False
    assert report["ranking"]["wall_clock_ignored_for_rank"] is True

    table = report["score_table"]
    assert len(table) == len(ARXIV_FAIR_EVAL_FAMILY_IDS)
    assert {row["family_id"] for row in table} == set(ARXIV_FAIR_EVAL_FAMILY_IDS)
    ranks = [row["rank"] for row in table]
    assert ranks == list(range(1, len(table) + 1))
    # Held-out descending within valid rows (primary).
    for i in range(len(table) - 1):
        a, b = table[i], table[i + 1]
        ha, hb = float(a["heldout_delta"]), float(b["heldout_delta"])
        if ha != hb:
            assert ha > hb or (a["status"] != "OK")
        else:
            # secondary bpb ascending when heldout ties
            assert float(a["bpb"]) <= float(b["bpb"])

    # Pin + score table artifacts written.
    assert Path(report["pin_path"]).is_file()
    assert Path(report["score_table_path"]).is_file()
    pin_disk = json.loads(Path(report["pin_path"]).read_text(encoding="utf-8"))
    assert pin_disk["protocol_hash"] == report["protocol_hash"]
    assert set(pin_disk["families"]) == set(ARXIV_FAIR_EVAL_FAMILY_IDS)


def test_rank_multi_family_records_sorts_heldout_then_bpb() -> None:
    strong = OfficialScoreRecord(
        label="strong",
        bpb=2.0,
        heldout_delta=1.5,
        valid=True,
        step0_anomaly=False,
        memorization_flag=False,
        overfit_rate=0.0,
        primary_form="heldout_delta",
        seed_count=1,
    )
    mid = OfficialScoreRecord(
        label="mid",
        bpb=1.0,  # better secondary but worse primary
        heldout_delta=0.5,
        valid=True,
        step0_anomaly=False,
        memorization_flag=False,
        overfit_rate=0.0,
        primary_form="heldout_delta",
        seed_count=1,
    )
    # Register temporary labels that exist in SEED_FAMILIES by aliasing Imp ids.
    aggregates = {
        "transformer-tiny-1m": strong,
        "mamba-tiny-1m": mid,
    }
    table = rank_multi_family_records(aggregates)
    assert table[0]["family_id"] == "transformer-tiny-1m"
    assert table[1]["family_id"] == "mamba-tiny-1m"
    assert table[0]["rank"] == 1


def test_lab_gpu_multi_family_missing_is_blocked_not_invented(tmp_path: Path) -> None:
    """VAL-ARXEVAL-006: missing side → BLOCKED_with_reason, no invented metrics."""
    empty_root = tmp_path / "arts"
    empty_root.mkdir()
    out = tmp_path / "out"
    pin = explore_protocol_pin(seeds=(LAB_GPU_DEFAULT_SEED,))
    report = run_multi_family_lab_gpu_host_compare(
        empty_root,
        out,
        family_ids=IMP_BASELINE_FAMILY_IDS,
        seeds=(LAB_GPU_DEFAULT_SEED,),
        pin=pin,
        allow_partial=True,
    )
    assert report["score_class"] == SCORE_CLASS_LAB_GPU
    assert report["score_table"] == []
    blocked = report["ranking"]["blocked_families"]
    assert set(blocked) == set(IMP_BASELINE_FAMILY_IDS)
    for reason in blocked.values():
        assert reason.startswith("BLOCKED_with_reason")
        assert "missing_lab_gpu_manifest" in reason
    # harden: allow_partial=False raises
    with pytest.raises(MultiFamilyArtifactsMissingError):
        run_multi_family_lab_gpu_host_compare(
            empty_root,
            out / "strict",
            family_ids=IMP_BASELINE_FAMILY_IDS,
            seeds=(LAB_GPU_DEFAULT_SEED,),
            pin=pin,
            allow_partial=False,
        )


def _lab_gpu_dual_artifact_root() -> Path | None:
    """Optional host LAB-GPU dual-train root (mission path never required on CI)."""
    import os

    env = os.environ.get("PRISM_LAB_GPU_ARTIFACTS_ROOT", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    # Local mission residual only — may be absent or unreadable on GitHub runners.
    candidates.append(
        Path(
            "/root/.factory/missions/a43a16a7-2230-4853-ba8a-a6bfe993a90f/"
            "evidence/lab-gpu-official/lium-train/artifacts/out"
        )
    )
    for root in candidates:
        manifest = root / "transformer-tiny-1m" / "seed-1337" / "prism_run_manifest.v2.json"
        try:
            if manifest.is_file() and os.access(manifest, os.R_OK):
                return root
        except OSError:
            continue
    return None


def test_lab_gpu_multi_family_partial_imp_scored_novels_blocked(tmp_path: Path) -> None:
    """VAL-ARXEVAL-006: partial LAB-GPU scores Imp; novels BLOCKED_with_reason.

    Uses synthetic manifests under tmp so CI never depends on mission absolute
    paths. Optional real dual-artifact root via PRISM_LAB_GPU_ARTIFACTS_ROOT.
    """
    from prism_challenge.evaluator.official_compare_harness import SynthSeedMetrics

    pin = explore_protocol_pin(seeds=(LAB_GPU_DEFAULT_SEED,))
    lab_root = tmp_path / "lab_out"
    # Imp pair only (novels intentionally missing → blocked).
    for fam, hd, bpb in (
        ("transformer-tiny-1m", 3.45, 0.121),
        ("mamba-tiny-1m", 4.62, 0.128),
    ):
        seed_dir = lab_root / fam / f"seed-{LAB_GPU_DEFAULT_SEED}"
        seed_dir.mkdir(parents=True)
        metrics = SynthSeedMetrics(seed=LAB_GPU_DEFAULT_SEED, bpb=bpb, heldout_delta=hd)
        man = synth_challenge_manifest(metrics, pin=pin, family_id=fam, device="cuda")
        (seed_dir / "prism_run_manifest.v2.json").write_text(
            json.dumps(man, indent=2), encoding="utf-8"
        )

    out = tmp_path / "out"
    report = run_multi_family_lab_gpu_host_compare(
        lab_root,
        out,
        family_ids=ARXIV_FAIR_EVAL_FAMILY_IDS,
        seeds=(LAB_GPU_DEFAULT_SEED,),
        pin=pin,
        allow_partial=True,
    )
    assert report["score_class"] == SCORE_CLASS_LAB_GPU
    assert report["labels"]["provider_trust"] == PROVIDER_TRUST_LABEL
    assert report["labels"]["image_pin"] == IMAGE_PIN_LABEL
    assert report["labels"]["prism_tee_product"] is False
    assert "real_provider_tee" not in report
    scored_ids = {row["family_id"] for row in report["score_table"]}
    assert "transformer-tiny-1m" in scored_ids
    assert "mamba-tiny-1m" in scored_ids
    blocked = report["ranking"]["blocked_families"]
    for novel in NOVEL_ARXIV_FAMILY_IDS:
        assert novel in blocked
        assert novel not in scored_ids
        assert "BLOCKED_with_reason" in blocked[novel]
    table_by_id = {r["family_id"]: r for r in report["score_table"]}
    assert float(table_by_id["mamba-tiny-1m"]["heldout_delta"]) > float(
        table_by_id["transformer-tiny-1m"]["heldout_delta"]
    )
    assert report["ranking"]["winner"] == "mamba-tiny-1m"
    assert Path(report["pin_path"]).is_file()
    assert Path(report["score_table_path"]).is_file()

    # Optional online residual IP: if a readable dual-train root exists, also assert there.
    real_root = _lab_gpu_dual_artifact_root()
    if real_root is not None:
        report_real = run_multi_family_lab_gpu_host_compare(
            real_root,
            tmp_path / "out_real",
            family_ids=IMP_BASELINE_FAMILY_IDS,
            seeds=(LAB_GPU_DEFAULT_SEED,),
            pin=pin,
            allow_partial=True,
        )
        assert report_real["score_class"] == SCORE_CLASS_LAB_GPU
        real_ids = {row["family_id"] for row in report_real["score_table"]}
        assert "transformer-tiny-1m" in real_ids
        assert "mamba-tiny-1m" in real_ids


def test_records_from_lab_gpu_multi_recomputes_via_official_path(tmp_path: Path) -> None:
    """Architecture-agnostic: written manifests recompute to official records for any tag."""
    pin = explore_protocol_pin(seeds=(1337,))
    root = tmp_path / "arts"
    for fam, hd, bpb in (
        ("deeploop-tiny-1m", 1.1, 1.4),
        ("alien-not-registered", 0.9, 1.2),  # still scorables as unknown if drafted
    ):
        seed_dir = root / fam / "seed-1337"
        seed_dir.mkdir(parents=True)
        # Use harness synth → challenge owned shell; secondary bpb via official path.
        from prism_challenge.evaluator.official_compare_harness import SynthSeedMetrics

        metrics = SynthSeedMetrics(seed=1337, bpb=bpb, heldout_delta=hd)
        man = synth_challenge_manifest(metrics, pin=pin, family_id=fam, device="fixture")
        # LAB-GPU layout uses protocol_pin key sometimes — official_record accepts either.
        (seed_dir / "prism_run_manifest.v2.json").write_text(
            json.dumps(man, indent=2), encoding="utf-8"
        )

    # Only request registered deeploop to keep packing path clean for the runner;
    # unit path here uses records_from_lab_gpu_artifacts_multi directly.
    records, missing = records_from_lab_gpu_artifacts_multi(
        root,
        ("deeploop-tiny-1m",),
        seeds=(1337,),
        pin=pin,
    )
    assert missing == {}
    assert len(records["deeploop-tiny-1m"]) == 1
    rec = records["deeploop-tiny-1m"][0]
    assert rec.valid is True
    assert rec.heldout_delta == pytest.approx(1.1)
    # Secondary bpb recomputed (positive finite).
    assert rec.bpb > 0 and rec.bpb < 10

    # Direct equal-path proof: same manifest / different label family text.
    m = json.loads(
        (root / "deeploop-tiny-1m" / "seed-1337" / "prism_run_manifest.v2.json").read_text()
    )
    r1 = official_record_from_manifest(m, label="tag-a")
    r2 = official_record_from_manifest(m, label="tag-b")
    assert r1.bpb == r2.bpb and r1.heldout_delta == r2.heldout_delta
