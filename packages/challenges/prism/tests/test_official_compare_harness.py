"""Dual-family Official Comparison CPU/fixture harness (VAL-COMP-009).

Proves two unknown-style seed packages (Transformer tiny-1m vs pure-torch Mamba)
under protocol fairness produce comparable official records and a clear A-vs-B
``compare_official`` outcome without NVIDIA. Synthetic challenge-owned metrics only;
no multi-hour train; PROVIDER_TRUST / LAB-GPU / IMAGE_PIN labels only.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from prism_challenge.evaluator.official_compare_harness import (
    DEFAULT_MAMBA_PROFILE,
    DEFAULT_TRANSFORMER_PROFILE,
    IMAGE_PIN_LABEL,
    LAB_GPU_DEFAULT_SEED,
    PROVIDER_TRUST_LABEL,
    REPORT_SCHEMA,
    SCORE_CLASS_LAB_GPU,
    SIDE_A_FAMILY_ID,
    SIDE_B_FAMILY_ID,
    FamilySynthProfile,
    LabGpuArtifactsMissingError,
    SynthSeedMetrics,
    aggregate_side,
    build_compare_report,
    default_protocol_pin,
    gpu_verification_status,
    lab_gpu_verification_status,
    package_unknown_style_pair,
    protocol_pin_hash,
    records_for_profile,
    records_from_lab_gpu_artifacts,
    run_dual_family_official_compare,
    run_lab_gpu_host_official_compare,
    synth_challenge_manifest,
)
from prism_challenge.evaluator.official_compare_harness import (
    main as harness_main,
)
from prism_challenge.evaluator.official_comparison import (
    OFFICIAL_DEFAULT_SEEDS,
    PROTOCOL_ID,
    OfficialScoreRecord,
    ProtocolPin,
    compare_official,
    protocol_budget_constants,
)
from prism_challenge.evaluator.train_series import (
    make_fixture_series,
    train_series_sha256,
    write_train_series_artifact,
)
from prism_challenge.seed_packaging import REQUIRED_ENTRY_SCRIPTS, SEED_FAMILIES


def test_gpu_verification_status_never_claims_gpu_pass() -> None:
    """VAL-COMP-009 / VAL-COMP-010: harness must not invent GPU verification PASS."""
    gpu = gpu_verification_status()
    assert gpu["claim_gpu_pass"] is False
    assert gpu["status"] in {"DEFERRED", "AVAILABLE"}
    # This mission host has no NVIDIA; status must be DEFERRED when absent.
    if not gpu["nvidia_smi"] or gpu["nvidia_device_count"] == 0:
        assert gpu["status"] == "DEFERRED"
        assert "not found" in gpu["reason"] or "absent" in gpu["reason"]


def test_package_unknown_style_pair_fair_outer_contract(tmp_path: Path) -> None:
    """Seed packaging already exists — dual packages share unknown-style outer shape."""
    assert SIDE_A_FAMILY_ID in SEED_FAMILIES
    assert SIDE_B_FAMILY_ID in SEED_FAMILIES
    packed = package_unknown_style_pair(tmp_path)
    assert set(packed) == {"a", "b"}
    a, b = packed["a"], packed["b"]
    assert a.family_id == SIDE_A_FAMILY_ID
    assert b.family_id == SIDE_B_FAMILY_ID
    assert a.zip_path.is_file() and b.zip_path.is_file()
    # Protocol fairness: same required entries, no secrets / weight blobs.
    for item in (a, b):
        names = set(item.entry_names)
        for required in REQUIRED_ENTRY_SCRIPTS:
            assert required in names
        with zipfile.ZipFile(item.zip_path) as zf:
            zip_names = set(zf.namelist())
        assert not any(n.endswith((".pem", ".key", ".env", ".pt", ".bin")) for n in zip_names)
    # Distinct payloads (different architectures) but identical entry requirements.
    assert a.content_sha256 != b.content_sha256
    assert set(REQUIRED_ENTRY_SCRIPTS).issubset(set(a.entry_names))
    assert set(REQUIRED_ENTRY_SCRIPTS).issubset(set(b.entry_names))


def test_synth_manifest_is_challenge_owned_not_miner_trust_root() -> None:
    pin = default_protocol_pin()
    metrics = SynthSeedMetrics(seed=1337, bpb=1.5, heldout_delta=0.7)
    manifest = synth_challenge_manifest(metrics, pin=pin, family_id=SIDE_A_FAMILY_ID)
    assert manifest["schema_version"] == "prism_run_manifest.v2"
    assert manifest["miner_reported_ignored"] is True
    assert manifest["protocol"]["unknown_style"] is True
    assert manifest["protocol"]["protocol_id"] == PROTOCOL_ID
    assert manifest["protocol"]["protocol_hash"] == protocol_pin_hash(pin)
    assert manifest["data"]["stopped_reason"] == "token_budget"
    # Fairness pin knobs are carried for both families identically.
    constants = protocol_budget_constants()
    assert manifest["protocol"]["token_budget"] == constants["token_budget"]
    assert manifest["protocol"]["tokenizer"] == constants["tokenizer"]
    assert manifest["protocol"]["param_cap"] == constants["param_cap"]


def test_dual_family_official_compare_clear_outcome_without_gpu(tmp_path: Path) -> None:
    """VAL-COMP-009: pytest dual compare without GPU; clear outcome A vs B."""
    report = run_dual_family_official_compare(
        tmp_path,
        side_a_profile=DEFAULT_TRANSFORMER_PROFILE,
        side_b_profile=DEFAULT_MAMBA_PROFILE,
        device_class="fixture",
    )
    assert report["schema"] == REPORT_SCHEMA
    assert report["protocol_id"] == PROTOCOL_ID
    assert report["mode"] == "ArchCompare"
    assert report["device_class"] == "fixture"
    assert report["gpu_verification"]["claim_gpu_pass"] is False
    assert "PROVIDER_TRUST" in report["provider_honesty"]

    # Both sides comparable official records under matched pin.
    a = report["side_a"]
    b = report["side_b"]
    assert a["family_id"] == SIDE_A_FAMILY_ID
    assert b["family_id"] == SIDE_B_FAMILY_ID
    assert a["unknown_style"] is True and b["unknown_style"] is True
    assert a["mean_heldout_delta"] is not None
    assert b["mean_heldout_delta"] is not None
    assert a["mean_bpb"] is not None and b["mean_bpb"] is not None
    assert a["seed_count"] == len(OFFICIAL_DEFAULT_SEEDS)
    assert b["seed_count"] == len(OFFICIAL_DEFAULT_SEEDS)

    # Primary held-out: transformer fixture generalizes better → winner A.
    # Secondary bpb alone would prefer Mamba (lower train bpb) — official invert is visible.
    assert float(a["mean_heldout_delta"]) > float(b["mean_heldout_delta"])
    assert float(a["mean_bpb"]) > float(b["mean_bpb"])
    ranking = report["ranking"]
    assert ranking["winner"] == "a"
    assert ranking["reason"] == "primary_heldout"
    assert ranking["outcome_label"]["winner_side"] == "a"
    assert ranking["outcome_label"]["winner_label"] == a["label"]
    assert ranking["rule"] == "heldout_primary_then_bpb_secondary"

    # Report written for operators.
    report_path = Path(report["report_path"])
    assert report_path.is_file()
    disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert disk["ranking"]["winner"] == "a"
    assert max(report["seeds"]) == max(OFFICIAL_DEFAULT_SEEDS)


def test_matched_pin_and_same_seed_list_for_both_families(tmp_path: Path) -> None:
    pin = default_protocol_pin()
    assert pin.seeds == OFFICIAL_DEFAULT_SEEDS
    assert len(pin.seeds) >= 3
    report = run_dual_family_official_compare(tmp_path, pin=pin)
    assert report["seeds"] == list(OFFICIAL_DEFAULT_SEEDS)
    assert report["pin"]["token_budget"] == pin.token_budget
    assert report["pin"]["tokenizer"] == pin.tokenizer
    assert report["pin"]["param_cap"] == pin.param_cap
    assert report["validity"]["matched_budget"] is True
    assert report["validity"]["wall_clock_never_ranks"] is True
    assert report["validity"]["miner_self_report_never_authoritative"] is True


def test_primary_heldout_invert_visible_against_secondary_bpb(tmp_path: Path) -> None:
    """Clear A-vs-B: held-out primary beats secondary train bpb (protocol invert)."""
    report = run_dual_family_official_compare(tmp_path)
    # Reconstruct pure compare_official to show clear outcome path.
    pin = default_protocol_pin()
    a_records = records_for_profile(DEFAULT_TRANSFORMER_PROFILE, pin=pin)
    b_records = records_for_profile(DEFAULT_MAMBA_PROFILE, pin=pin)
    assert len(a_records) == 3 and len(b_records) == 3
    assert all(isinstance(r, OfficialScoreRecord) for r in a_records + b_records)

    # Aggregate surfaces from the filed report.
    mean_a_held = report["aggregate"]["a"]["mean_heldout_delta"]
    mean_b_held = report["aggregate"]["b"]["mean_heldout_delta"]
    mean_a_bpb = report["aggregate"]["a"]["mean_bpb"]
    mean_b_bpb = report["aggregate"]["b"]["mean_bpb"]
    assert mean_a_held > mean_b_held
    assert mean_a_bpb > mean_b_bpb  # Mamba stronger on secondary
    assert report["ranking"]["winner"] == "a"
    assert report["ranking"]["reason"] == "primary_heldout"

    # Determinism.
    again = run_dual_family_official_compare(tmp_path / "again")
    assert again["ranking"]["winner"] == report["ranking"]["winner"]
    assert again["ranking"]["reason"] == report["ranking"]["reason"]
    assert again["protocol_hash"] == report["protocol_hash"]


def test_flipped_primary_makes_b_win(tmp_path: Path) -> None:
    """If Mamba synth primary held-out is better, outcome flips to B (clear reverse)."""
    strong = FamilySynthProfile(
        family_id=SIDE_B_FAMILY_ID,
        label="mamba-tiny-1m",
        architecture_family="mamba",
        seeds=(
            SynthSeedMetrics(seed=1337, bpb=1.5, heldout_delta=1.20),
            SynthSeedMetrics(seed=2027, bpb=1.5, heldout_delta=1.10),
            SynthSeedMetrics(seed=4242, bpb=1.5, heldout_delta=1.15),
        ),
    )
    weak = FamilySynthProfile(
        family_id=SIDE_A_FAMILY_ID,
        label="transformer-tiny-1m",
        architecture_family="transformer",
        seeds=(
            SynthSeedMetrics(seed=1337, bpb=1.0, heldout_delta=0.10),
            SynthSeedMetrics(seed=2027, bpb=1.0, heldout_delta=0.12),
            SynthSeedMetrics(seed=4242, bpb=1.0, heldout_delta=0.08),
        ),
    )
    report = run_dual_family_official_compare(
        tmp_path,
        side_a_profile=weak,
        side_b_profile=strong,
    )
    assert report["ranking"]["winner"] == "b"
    assert report["ranking"]["reason"] == "primary_heldout"
    assert report["ranking"]["outcome_label"]["winner_label"] == "mamba-tiny-1m"


def test_wall_clock_imbalance_does_not_reorder_outcome(tmp_path: Path) -> None:
    slow_transformer = FamilySynthProfile(
        family_id=SIDE_A_FAMILY_ID,
        label="transformer-tiny-1m",
        architecture_family="transformer",
        seeds=tuple(
            SynthSeedMetrics(
                seed=s.seed,
                bpb=s.bpb,
                heldout_delta=s.heldout_delta,
                train_heldout_gap=s.train_heldout_gap,
                wall_clock_seconds=10_000.0,
            )
            for s in DEFAULT_TRANSFORMER_PROFILE.seeds
        ),
    )
    fast_mamba = FamilySynthProfile(
        family_id=SIDE_B_FAMILY_ID,
        label="mamba-tiny-1m",
        architecture_family="mamba",
        seeds=tuple(
            SynthSeedMetrics(
                seed=s.seed,
                bpb=s.bpb,
                heldout_delta=s.heldout_delta,
                train_heldout_gap=s.train_heldout_gap,
                wall_clock_seconds=1.0,
            )
            for s in DEFAULT_MAMBA_PROFILE.seeds
        ),
    )
    report = run_dual_family_official_compare(
        tmp_path,
        side_a_profile=slow_transformer,
        side_b_profile=fast_mamba,
    )
    # Still A by held-out; wall-clock imbalance cannot crown the faster GPU.
    assert report["ranking"]["winner"] == "a"
    assert report["ranking"]["reason"] == "primary_heldout"


def test_build_compare_report_schema_matches_docs_sketch(tmp_path: Path) -> None:
    packed = package_unknown_style_pair(tmp_path)
    pin = default_protocol_pin()
    a = OfficialScoreRecord(
        label="transformer-tiny-1m",
        bpb=1.8,
        heldout_delta=0.95,
        seed_count=3,
        bpb_std=0.05,
    )
    b = OfficialScoreRecord(
        label="mamba-tiny-1m",
        bpb=1.1,
        heldout_delta=0.25,
        seed_count=3,
        bpb_std=0.05,
    )
    result = compare_official(a, b)
    report = build_compare_report(
        pin=pin,
        side_a=a,
        side_b=b,
        packed=packed,
        result=result,
        mode="ArchCompare",
        device_class="cpu",
        gpu={"status": "DEFERRED", "reason": "unit", "claim_gpu_pass": False},
    )
    for key in (
        "schema",
        "protocol_id",
        "protocol_hash",
        "mode",
        "primary_form",
        "side_a",
        "side_b",
        "seeds",
        "aggregate",
        "ranking",
        "validity",
        "provider_honesty",
        "gpu_verification",
        "labels",
    ):
        assert key in report
    assert report["ranking"]["winner"] in {"a", "b", "tie"}


def test_harness_cli_offline_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = harness_main(["--output-dir", str(tmp_path), "--device-class", "fixture"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "compare outcome" in captured
    assert "winner=" in captured
    assert (tmp_path / "prism_compare_report.v1.json").is_file()
    assert "gpu_verification" in captured


def _write_lab_gpu_family_manifest(
    root: Path,
    *,
    family_id: str,
    seed: int,
    bpb: float,
    heldout_delta: float,
    wall_clock_seconds: float,
) -> Path:
    """Write a challenge-owned LAB-GPU-shaped v2 manifest under the expected layout."""
    pin = ProtocolPin(seeds=(seed,))
    metrics = SynthSeedMetrics(
        seed=seed,
        bpb=bpb,
        heldout_delta=heldout_delta,
        wall_clock_seconds=wall_clock_seconds,
        train_heldout_gap=0.12,
    )
    manifest = synth_challenge_manifest(
        metrics,
        pin=pin,
        family_id=family_id,
        device="cuda",
    )
    manifest["score_class"] = SCORE_CLASS_LAB_GPU
    dest = root / family_id / f"seed-{seed}" / "prism_run_manifest.v2.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def test_lab_gpu_verification_status_not_deferred_for_lab_class() -> None:
    """VAL-GPULAB-006: lab scores class is LAB-GPU; local no-nvidia does not DEFER lab class."""
    gpu = lab_gpu_verification_status(train_host_note="unit-lium")
    assert gpu["status"] == SCORE_CLASS_LAB_GPU
    assert gpu["claim_gpu_pass"] is True  # lab scores only
    assert gpu["provider_trust"] == PROVIDER_TRUST_LABEL
    assert gpu["image_pin"] == IMAGE_PIN_LABEL
    assert gpu["not_deferred_for_missing_local_nvidia"] is True
    assert gpu["score_class"] == SCORE_CLASS_LAB_GPU
    # Fixture path stays DEFERRED / no claim on this host shape.
    fixture_gpu = gpu_verification_status()
    assert fixture_gpu["claim_gpu_pass"] is False


def test_records_from_lab_gpu_artifacts_recompute_and_missing(tmp_path: Path) -> None:
    """Host recompute yields non-null primary/secondary metrics; missing side BLOCKS list."""
    seed = LAB_GPU_DEFAULT_SEED
    _write_lab_gpu_family_manifest(
        tmp_path,
        family_id=SIDE_A_FAMILY_ID,
        seed=seed,
        bpb=1.2,
        heldout_delta=0.4,
        wall_clock_seconds=10.0,
    )
    a_rec, b_rec, missing = records_from_lab_gpu_artifacts(
        tmp_path,
        seeds=(seed,),
        pin=ProtocolPin(seeds=(seed,)),
    )
    assert len(a_rec) == 1
    assert a_rec[0].valid is True
    assert a_rec[0].heldout_delta is not None
    assert a_rec[0].bpb > 0.0
    assert a_rec[0].wall_clock_seconds == 10.0
    assert b_rec == []
    assert missing and SIDE_B_FAMILY_ID in missing[0]


def test_run_lab_gpu_host_official_compare_clear_winner(tmp_path: Path) -> None:
    """VAL-GPULAB-004/006: LAB-GPU report, primary held-out winner, wall_clock ignored."""
    seed = LAB_GPU_DEFAULT_SEED
    root = tmp_path / "artifacts"
    # Mamba better primary held-out but slower wall-clock; transformer better secondary bpb.
    _write_lab_gpu_family_manifest(
        root,
        family_id=SIDE_A_FAMILY_ID,
        seed=seed,
        bpb=0.90,
        heldout_delta=0.25,
        wall_clock_seconds=5.0,
    )
    _write_lab_gpu_family_manifest(
        root,
        family_id=SIDE_B_FAMILY_ID,
        seed=seed,
        bpb=1.10,
        heldout_delta=1.50,
        wall_clock_seconds=500.0,
    )
    out = tmp_path / "report"
    report = run_lab_gpu_host_official_compare(root, out, seeds=(seed,))
    assert report["schema"] == REPORT_SCHEMA
    assert report["score_class"] == SCORE_CLASS_LAB_GPU
    assert report["device_class"] == "lab-gpu"
    assert report["gpu_verification"]["status"] == SCORE_CLASS_LAB_GPU
    assert report["gpu_verification"]["claim_gpu_pass"] is True
    assert report["labels"]["provider_trust"] == PROVIDER_TRUST_LABEL
    assert report["labels"]["image_pin"] == IMAGE_PIN_LABEL
    assert report["labels"]["prism_tee_product"] is False
    assert "real_provider_tee" not in report
    assert "PROVIDER_TRUST" in report["provider_honesty"]
    assert report["validity"]["wall_clock_never_ranks"] is True
    assert report["ranking"]["wall_clock_ignored_for_rank"] is True
    assert report["ranking"]["reason"] == "primary_heldout"
    assert report["ranking"]["winner"] == "b"
    assert SIDE_B_FAMILY_ID in report["ranking"]["outcome_label"]["winner_label"]
    # Non-null official metrics on both sides.
    assert report["side_a"]["mean_heldout_delta"] is not None
    assert report["side_b"]["mean_heldout_delta"] is not None
    assert report["side_a"]["mean_bpb"] is not None
    assert report["side_b"]["mean_bpb"] is not None
    # Wall-clock imbalance must NOT flip the held-out winner (mamba is 100x slower).
    assert float(report["side_b"]["wall_clock_seconds"] or 0) > float(
        report["side_a"]["wall_clock_seconds"] or 0
    )
    report_path = Path(report["report_path"])
    assert report_path.is_file()
    disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert disk["score_class"] == SCORE_CLASS_LAB_GPU
    assert disk["ranking"]["winner"] == "b"
    assert disk["labels"]["not_deferred_for_no_nvidia"] is True
    assert disk["labels"]["not_fixture_only_synthetic"] is True


def test_run_lab_gpu_host_official_compare_blocked_when_missing(tmp_path: Path) -> None:
    """Missing LAB-GPU artifacts must fail closed (BLOCKED), not invent scores."""
    with pytest.raises(LabGpuArtifactsMissingError, match="BLOCKED"):
        run_lab_gpu_host_official_compare(tmp_path / "empty", tmp_path / "out")


def test_harness_cli_lab_gpu_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seed = LAB_GPU_DEFAULT_SEED
    root = tmp_path / "artifacts"
    _write_lab_gpu_family_manifest(
        root,
        family_id=SIDE_A_FAMILY_ID,
        seed=seed,
        bpb=1.0,
        heldout_delta=2.0,
        wall_clock_seconds=12.0,
    )
    _write_lab_gpu_family_manifest(
        root,
        family_id=SIDE_B_FAMILY_ID,
        seed=seed,
        bpb=1.2,
        heldout_delta=0.5,
        wall_clock_seconds=80.0,
    )
    out = tmp_path / "out"
    code = harness_main(
        [
            "--output-dir",
            str(out),
            "--lab-gpu-artifacts",
            str(root),
            "--seed",
            str(seed),
        ]
    )
    assert code == 0
    captured = capsys.readouterr().out
    assert "score_class=LAB-GPU" in captured
    assert "winner=" in captured
    assert "provider_trust=PROVIDER_TRUST" in captured
    assert "image_pin=IMAGE_PIN" in captured
    assert (out / "prism_compare_report.v1.json").is_file()


def test_harness_cli_lab_gpu_missing_exits_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = harness_main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--lab-gpu-artifacts",
            str(tmp_path / "missing"),
        ]
    )
    assert code == 2
    assert "BLOCKED" in capsys.readouterr().out


def test_require_train_series_pin_fail_closed_without_series(tmp_path: Path) -> None:
    """When pin.require_train_series, harness grade fails closed if series not supplied."""
    pin = ProtocolPin(require_train_series=True, seeds=OFFICIAL_DEFAULT_SEEDS)
    report = run_dual_family_official_compare(tmp_path, pin=pin)
    assert report["pin"]["require_train_series"] is True
    assert report["train_series_grade"]["require_train_series"] is True
    assert report["train_series_grade"]["grade_valid"] is False
    assert report["train_series_grade"]["silent_pass"] is False
    assert report["validity"]["ok"] is False
    assert report["validity"]["train_series_grade_ok"] is False
    assert any("train_series_missing" in r for r in report["validity"]["reasons"])


def test_require_train_series_pin_allows_good_series_with_disk_digest(tmp_path: Path) -> None:
    """Harness path: good dual series + compact on-disk digests → grade_valid=true."""
    pin = ProtocolPin(require_train_series=True, seeds=OFFICIAL_DEFAULT_SEEDS)
    series_a = make_fixture_series(
        submission_id="harness-a",
        run_id="prism-reexec-harness-a",
        family="transformer",
        n_points=8,
    )
    series_b = make_fixture_series(
        submission_id="harness-b",
        run_id="prism-reexec-harness-b",
        family="mamba",
        n_points=8,
        seed_offset=1.0,
    )
    art_a = tmp_path / "series_a"
    art_b = tmp_path / "series_b"
    art_a.mkdir()
    art_b.mkdir()
    _, dig_a = write_train_series_artifact(art_a, series_a)
    _, dig_b = write_train_series_artifact(art_b, series_b)
    assert train_series_sha256(series_a) == dig_a
    assert train_series_sha256(series_b) == dig_b

    packed = package_unknown_style_pair(tmp_path / "packages")
    agg_a = aggregate_side(DEFAULT_TRANSFORMER_PROFILE, pin=pin)
    agg_b = aggregate_side(DEFAULT_MAMBA_PROFILE, pin=pin)
    result = compare_official(agg_a, agg_b)
    report = build_compare_report(
        pin=pin,
        side_a=agg_a,
        side_b=agg_b,
        packed=packed,
        result=result,
        train_series_a=series_a,
        train_series_b=series_b,
        train_series_sha256_a=dig_a,
        train_series_sha256_b=dig_b,
    )
    assert report["train_series_grade"]["grade_valid"] is True
    assert report["train_series_grade"]["side_a"]["grade_valid"] is True
    assert report["train_series_grade"]["side_b"]["grade_valid"] is True
    assert report["validity"]["train_series_grade_ok"] is True
    assert report["validity"]["ok"] is True
