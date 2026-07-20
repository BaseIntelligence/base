"""Dual-family Official Comparison CPU/fixture harness (no NVIDIA required).

Packages the registered Transformer + Mamba seed families as unknown-style two-script
zips under one ProtocolPin, builds challenge-owned synthetic score manifests (Prism
recomputes secondary bpb), and emits a ``prism_compare_report.v1`` via
:func:`compare_official`.

This is the lab/offline ArchCompare surface for VAL-COMP-009. Long multi-step GPU
pair trains remain **DEFERRED** when the host lacks NVIDIA (see
:func:`gpu_verification_status`). Production leaderboard scoring is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from prism_challenge.seed_packaging import (
    REQUIRED_ENTRY_SCRIPTS,
    SEED_FAMILIES,
    PackedSeed,
    get_family,
    package_seed_zip,
)

from .official_comparison import (
    OFFICIAL_DEFAULT_SEEDS,
    PROTOCOL_ID,
    PROTOCOL_SCHEMA,
    SCORECARD_ID,
    CompareResult,
    OfficialScoreRecord,
    ProtocolPin,
    aggregate_official_records,
    apply_train_series_requirement_to_grade,
    attach_scorecard_to_report,
    compare_official_scorecard,
    official_record_from_manifest,
    protocol_budget_constants,
)

REPORT_SCHEMA = "prism_compare_report.v1"
CompareMode = Literal["ArchCompare", "TrainCompare", "SystemCompare"]

# Registered dual-family pair for the fixture harness (unknown architecture style).
SIDE_A_FAMILY_ID = "transformer-tiny-1m"
SIDE_B_FAMILY_ID = "mamba-tiny-1m"

DeviceClass = Literal["cpu", "cuda", "fixture", "lab-gpu"]
ScoreClass = Literal["fixture", "LAB-GPU", "CPU"]

# Lab-GPU Official Comparison (host recomputes Lium CUDA long-train artifacts).
# Local host may still lack NVIDIA; score class remains LAB-GPU (not DEFERRED-for-no-nvidia).
SCORE_CLASS_LAB_GPU: ScoreClass = "LAB-GPU"
SCORE_CLASS_FIXTURE: ScoreClass = "fixture"
# Provider-trust honesty labels (Prism NO TEE residual — no crypto TEE product).
PROVIDER_TRUST_LABEL = "PROVIDER_TRUST"
IMAGE_PIN_LABEL = "IMAGE_PIN"
LAB_GPU_MANIFEST_NAME = "prism_run_manifest.v2.json"
LAB_GPU_DEFAULT_SEED = 1337


@dataclass(frozen=True)
class SynthSeedMetrics:
    """One seed's challenge-owned capture shell used to recompute official scores.

    Values are synthetic (unit fixtures). They must look like a Prism host capture,
    never like a miner self-report trust root.
    """

    seed: int
    bpb: float
    heldout_delta: float
    covered_bytes: int = 65_536
    train_heldout_gap: float = 0.15
    memorization_flag: bool | None = None
    step0_anomaly: bool = False
    wall_clock_seconds: float = 120.0
    val_bpb_trained: float | None = None
    val_bpb_random_init: float | None = None


@dataclass(frozen=True)
class FamilySynthProfile:
    """Per-family multi-seed synthetic profile under the matched protocol pin."""

    family_id: str
    label: str
    architecture_family: str
    seeds: tuple[SynthSeedMetrics, ...]


# Deterministic dual-family fixtures: Transformer generalizes better on primary
# held-out axis; secondary bpb is intentionally weaker than Mamba so production
# leaderboard (bpb-primary) would disagree — Official Comparison invert is visible.
DEFAULT_TRANSFORMER_PROFILE = FamilySynthProfile(
    family_id=SIDE_A_FAMILY_ID,
    label="transformer-tiny-1m",
    architecture_family="transformer",
    seeds=(
        SynthSeedMetrics(seed=1337, bpb=1.80, heldout_delta=0.95, train_heldout_gap=0.12),
        SynthSeedMetrics(seed=2027, bpb=1.85, heldout_delta=0.90, train_heldout_gap=0.14),
        SynthSeedMetrics(seed=4242, bpb=1.75, heldout_delta=1.00, train_heldout_gap=0.10),
    ),
)
DEFAULT_MAMBA_PROFILE = FamilySynthProfile(
    family_id=SIDE_B_FAMILY_ID,
    label="mamba-tiny-1m",
    architecture_family="mamba",
    seeds=(
        SynthSeedMetrics(seed=1337, bpb=1.10, heldout_delta=0.25, train_heldout_gap=0.18),
        SynthSeedMetrics(seed=2027, bpb=1.15, heldout_delta=0.20, train_heldout_gap=0.20),
        SynthSeedMetrics(seed=4242, bpb=1.05, heldout_delta=0.30, train_heldout_gap=0.16),
    ),
)


def default_protocol_pin(*, device_class: DeviceClass = "fixture") -> ProtocolPin:
    """Matched ProtocolPin shared by both harness sides (fairness pins)."""
    del device_class  # pin is device-agnostic; device class is report metadata only
    return ProtocolPin(
        protocol_id=PROTOCOL_ID,
        token_budget=protocol_budget_constants()["token_budget"],
        seeds=OFFICIAL_DEFAULT_SEEDS,
        param_cap=int(protocol_budget_constants()["param_cap"]),
        seq_len=int(protocol_budget_constants()["seq_len"]),
        batch_size=int(protocol_budget_constants()["batch_size"]),
        tokenizer=str(protocol_budget_constants()["tokenizer"]),
        vocab_size=int(protocol_budget_constants()["vocab_size"]),
        scored_nproc=int(protocol_budget_constants()["scored_nproc"]),
        val_byte_budget=int(protocol_budget_constants()["val_byte_budget"]),
        force_iter_train_batches=True,
        require_trained_state=True,
        primary_form="heldout_delta",
    )


def protocol_pin_hash(pin: ProtocolPin) -> str:
    """Stable pin hash for report / operator evidence."""
    payload = json.dumps(pin.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _probe_host_nvidia() -> dict[str, Any]:
    """Host-local NVIDIA probe shared by fixture DEFERRED and lab-gpu host notes."""
    nvidia_smi = shutil.which("nvidia-smi")
    nvidia_devices = sorted(Path("/dev").glob("nvidia*"))
    has_nvidia_runtime = False
    docker = shutil.which("docker")
    if docker is not None:
        # Best-effort: presence of nvidia runtime listed by docker info is informative only.
        # We do not execute GPU containers here.
        try:
            proc = subprocess.run(
                [docker, "info", "--format", "{{json .Runtimes}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0 and "nvidia" in (proc.stdout or "").lower():
                has_nvidia_runtime = True
        except (OSError, subprocess.TimeoutExpired):
            has_nvidia_runtime = False
    return {
        "nvidia_smi": nvidia_smi is not None,
        "nvidia_device_count": len(nvidia_devices),
        "docker_nvidia_runtime": has_nvidia_runtime,
        "host_has_nvidia": bool(nvidia_smi) and bool(nvidia_devices),
    }


def gpu_verification_status() -> dict[str, Any]:
    """Probe host GPU readiness; never invent a GPU PASS.

    Returns classification DEFERRED when NVIDIA is absent (this mission host profile).
    Fixture/CPU harnesses use this path. Lab-GPU host ranking of remote CUDA artifacts
    uses :func:`lab_gpu_verification_status` instead (status LAB-GPU, not DEFERRED).
    """
    host = _probe_host_nvidia()
    available = bool(host["host_has_nvidia"]) and bool(host["docker_nvidia_runtime"])
    if available:
        status = "AVAILABLE"
        reason = "nvidia-smi, /dev/nvidia*, and docker nvidia runtime all present"
    else:
        status = "DEFERRED"
        reasons: list[str] = []
        if not host["nvidia_smi"]:
            reasons.append("nvidia-smi not found")
        if int(host["nvidia_device_count"]) == 0:
            reasons.append("/dev/nvidia* absent")
        if not host["docker_nvidia_runtime"]:
            reasons.append("docker nvidia runtime not advertised")
        reason = "; ".join(reasons) if reasons else "GPU path not ready"
    return {
        "status": status,
        "reason": reason,
        "nvidia_smi": host["nvidia_smi"],
        "nvidia_device_count": host["nvidia_device_count"],
        "docker_nvidia_runtime": host["docker_nvidia_runtime"],
        "claim_gpu_pass": False,  # fixture harness never elevates this to PASS
    }


def lab_gpu_verification_status(
    *,
    train_device: str = "cuda",
    train_host_note: str = "lium-paid-cuda",
) -> dict[str, Any]:
    """GPU verification block for LAB-GPU Official Comparison reports.

    Real CUDA trains already occurred on a remote paid GPU (e.g. Lium). Mission host may
    still lack NVIDIA; the score class is **LAB-GPU**, not fixture DEFERRED-for-no-nvidia.
    ``claim_gpu_pass`` is true only for **lab scores** (VAL-GPULAB-006). Honesty labels
    are PROVIDER_TRUST / LAB-GPU / IMAGE_PIN (no Prism TEE product).
    """
    host = _probe_host_nvidia()
    return {
        "status": SCORE_CLASS_LAB_GPU,
        "reason": (
            f"real CUDA train artifacts ({train_device}) from {train_host_note}; "
            "host ranks with compare_official; local NVIDIA not required for lab class"
        ),
        "nvidia_smi": host["nvidia_smi"],
        "nvidia_device_count": host["nvidia_device_count"],
        "docker_nvidia_runtime": host["docker_nvidia_runtime"],
        "host_has_nvidia": host["host_has_nvidia"],
        "train_device": train_device,
        "train_host_note": train_host_note,
        # Lab scores only under PROVIDER_TRUST (no Prism TEE verifier).
        "claim_gpu_pass": True,
        "provider_trust": PROVIDER_TRUST_LABEL,
        "image_pin": IMAGE_PIN_LABEL,
        "not_deferred_for_missing_local_nvidia": True,
        "score_class": SCORE_CLASS_LAB_GPU,
    }


def resolve_lab_gpu_manifest_path(
    artifacts_root: Path | str,
    family_id: str,
    *,
    seed: int = LAB_GPU_DEFAULT_SEED,
) -> Path:
    """Resolve ``{root}/{family}/seed-{seed}/prism_run_manifest.v2.json``."""
    root = Path(artifacts_root)
    return root / family_id / f"seed-{seed}" / LAB_GPU_MANIFEST_NAME


def load_lab_gpu_manifest(
    artifacts_root: Path | str,
    family_id: str,
    *,
    seed: int = LAB_GPU_DEFAULT_SEED,
) -> dict[str, Any]:
    """Load one real LAB-GPU challenge-owned v2 manifest or raise FileNotFoundError."""
    path = resolve_lab_gpu_manifest_path(artifacts_root, family_id, seed=seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"LAB-GPU train artifact missing for family={family_id} seed={seed}: {path}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"LAB-GPU manifest is not a JSON object: {path}")
    return data


def records_from_lab_gpu_artifacts(
    artifacts_root: Path | str,
    *,
    side_a_family_id: str = SIDE_A_FAMILY_ID,
    side_b_family_id: str = SIDE_B_FAMILY_ID,
    seeds: Sequence[int] = (LAB_GPU_DEFAULT_SEED,),
    pin: ProtocolPin | None = None,
    miner_reported: Mapping[str, Any] | None = None,
) -> tuple[list[OfficialScoreRecord], list[OfficialScoreRecord], list[str]]:
    """Load/recompute official records for both families from real LAB-GPU manifests.

    Secondary bpb and anti-overfit always recompute through
    :func:`official_record_from_manifest` (Prism host path). Miner self-report may be
    attached only as diagnostics. Returns ``(side_a_records, side_b_records, missing)``
    where ``missing`` is empty on full success.
    """
    active_pin = pin if pin is not None else ProtocolPin(seeds=tuple(seeds))
    missing: list[str] = []
    a_records: list[OfficialScoreRecord] = []
    b_records: list[OfficialScoreRecord] = []
    for family_id, bucket in (
        (side_a_family_id, a_records),
        (side_b_family_id, b_records),
    ):
        for seed in seeds:
            if seed not in active_pin.seeds:
                continue
            try:
                manifest = load_lab_gpu_manifest(artifacts_root, family_id, seed=seed)
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            rec = official_record_from_manifest(
                manifest,
                label=f"{family_id}:seed={seed}",
                primary_form=active_pin.primary_form,
                miner_reported=miner_reported,
            )
            bucket.append(rec)
    return a_records, b_records, missing


def synth_challenge_manifest(
    metrics: SynthSeedMetrics,
    *,
    pin: ProtocolPin,
    family_id: str,
    device: DeviceClass = "fixture",
) -> dict[str, Any]:
    """Build a challenge-owned v2 manifest shell from synthetic seed metrics.

    Secondary bpb is always re-derived by :func:`score_prequential_bpb` via
    ``official_record_from_manifest`` (miner self-report fields are absent).
    """
    covered = int(metrics.covered_bytes)
    sum_nll_nats = float(metrics.bpb) * covered * math.log(2.0)
    body: dict[str, Any] = {
        "schema_version": "prism_run_manifest.v2",
        "protocol": {
            "protocol_id": pin.protocol_id,
            "protocol_hash": protocol_pin_hash(pin),
            "family_id": family_id,
            "seed": metrics.seed,
            "token_budget": pin.token_budget,
            "tokenizer": pin.tokenizer,
            "param_cap": pin.param_cap,
            "unknown_style": True,
        },
        "data": {
            "covered_bytes": covered,
            "single_pass": True,
            "stopped_reason": "token_budget",
        },
        "metrics": {
            "online_loss": [2.0, 1.5, 1.0],
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "covered_bytes": covered,
            "predicted_tokens": max(1, covered // 2),
            "step0_loss": 2.0,
            "consumed_batches": 3,
            "random_init_baseline_nats": math.log(256),
            "nan_inf_batches": 0,
            "heldout_delta": metrics.heldout_delta,
            "held_out_delta": metrics.heldout_delta,
            "train_heldout_gap": metrics.train_heldout_gap,
        },
        "anti_cheat": {
            "step0_anomaly": metrics.step0_anomaly,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
        "compute": {
            "schema": "prism_compute.v1",
            "gpu_count": 0 if device != "cuda" else 1,
            "world_size": pin.scored_nproc,
            "nproc_per_node": pin.scored_nproc,
            "device": "cpu" if device != "cuda" else "cuda",
            "wall_clock_seconds": metrics.wall_clock_seconds,
        },
    }
    metrics_map = body["metrics"]
    assert isinstance(metrics_map, dict)
    if metrics.val_bpb_trained is not None:
        metrics_map["val_bpb_trained"] = metrics.val_bpb_trained
    if metrics.val_bpb_random_init is not None:
        metrics_map["val_bpb_random_init"] = metrics.val_bpb_random_init
    if metrics.memorization_flag is not None:
        metrics_map["memorization_flag"] = metrics.memorization_flag
    return body


def package_unknown_style_pair(
    output_dir: Path | str,
    *,
    side_a_family_id: str = SIDE_A_FAMILY_ID,
    side_b_family_id: str = SIDE_B_FAMILY_ID,
) -> dict[str, PackedSeed]:
    """Package both families as two-script submission zips under the shared outer contract.

    Both packages are treated as *unknown style*: same required entry scripts, same
    fingerprint surface, no family-specific score path. Seed packaging already exists;
    this couples packaging to the official compare pin.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = {
        "a": package_seed_zip(side_a_family_id, out),
        "b": package_seed_zip(side_b_family_id, out),
    }
    # Fairness: required entry names identical across families.
    a_entries = set(pairs["a"].entry_names)
    b_entries = set(pairs["b"].entry_names)
    missing_a = [n for n in REQUIRED_ENTRY_SCRIPTS if n not in a_entries]
    missing_b = [n for n in REQUIRED_ENTRY_SCRIPTS if n not in b_entries]
    if missing_a or missing_b:
        raise RuntimeError(f"unknown-style package missing entries a={missing_a} b={missing_b}")
    return pairs


def records_for_profile(
    profile: FamilySynthProfile,
    *,
    pin: ProtocolPin,
    device: DeviceClass = "fixture",
    miner_reported: Mapping[str, Any] | None = None,
) -> list[OfficialScoreRecord]:
    """Project one family synth profile into official score records (per seed)."""
    records: list[OfficialScoreRecord] = []
    for seed_metrics in profile.seeds:
        if seed_metrics.seed not in pin.seeds:
            # Protocol fairness: only pin seeds may contribute.
            continue
        manifest = synth_challenge_manifest(
            seed_metrics,
            pin=pin,
            family_id=profile.family_id,
            device=device,
        )
        rec = official_record_from_manifest(
            manifest,
            label=f"{profile.label}:seed={seed_metrics.seed}",
            primary_form=pin.primary_form,
            miner_reported=miner_reported,
        )
        records.append(rec)
    return records


def aggregate_side(
    profile: FamilySynthProfile,
    *,
    pin: ProtocolPin,
    device: DeviceClass = "fixture",
    miner_reported: Mapping[str, Any] | None = None,
) -> OfficialScoreRecord:
    """Multi-seed official aggregate for one package side."""
    per_seed = records_for_profile(profile, pin=pin, device=device, miner_reported=miner_reported)
    return aggregate_official_records(
        per_seed,
        label=profile.label,
        primary_form=pin.primary_form,
    )


def _side_bundle_block(packed: PackedSeed, aggregate: OfficialScoreRecord) -> dict[str, Any]:
    family = get_family(packed.family_id)
    return {
        "label": aggregate.label,
        "family_id": packed.family_id,
        "architecture_family": family.architecture_family,
        "unknown_style": True,
        "bundle_hash": packed.content_sha256,
        "zip_path": str(packed.zip_path),
        "entry_names": list(packed.entry_names),
        "size_bytes": packed.size_bytes,
        "mean_heldout_delta": aggregate.heldout_delta,
        "mean_bpb": aggregate.bpb,
        "std_bpb": aggregate.bpb_std,
        "overfit_rate": aggregate.overfit_rate,
        "memorization_flag": aggregate.memorization_flag,
        "step0_anomaly": aggregate.step0_anomaly,
        "valid": aggregate.valid,
        "seed_count": aggregate.seed_count,
        "wall_clock_seconds": aggregate.wall_clock_seconds,
        "miner_reported_bpb": aggregate.miner_reported_bpb,
    }


def build_compare_report(
    *,
    pin: ProtocolPin,
    side_a: OfficialScoreRecord,
    side_b: OfficialScoreRecord,
    packed: Mapping[str, PackedSeed],
    result: CompareResult,
    mode: CompareMode = "ArchCompare",
    device_class: DeviceClass = "fixture",
    gpu: Mapping[str, Any] | None = None,
    validity_reasons: Sequence[str] = (),
    score_class: ScoreClass = SCORE_CLASS_FIXTURE,
    artifact_source: str | None = None,
    train_series_a: Mapping[str, Any] | None = None,
    train_series_b: Mapping[str, Any] | None = None,
    train_series_sha256_a: str | None = None,
    train_series_sha256_b: str | None = None,
) -> dict[str, Any]:
    """Emit a ``prism_compare_report.v1`` document (docs §10 sketch).

    ``score_class=LAB-GPU`` labels host ranking of real remote CUDA train artifacts.
    Fixture/CPU dual-family remains the default and keeps host GPU DEFERRED honesty.

    When ``pin.require_train_series`` is True, apply the Official train-series grade gate
    (fail-closed on missing/empty/corrupt/digest mismatch) and fold the outcome into
    ``validity`` + a top-level ``train_series_grade`` block. Callers that already own
    challenge series may pass ``train_series_a`` / ``train_series_b`` (and optional
    on-disk digests); otherwise the pin-on path fail-closes for missing series.
    """
    if gpu is not None:
        gpu_info = dict(gpu)
    elif score_class == SCORE_CLASS_LAB_GPU:
        gpu_info = lab_gpu_verification_status()
    else:
        gpu_info = gpu_verification_status()
    validity_ok = side_a.valid and side_b.valid and not validity_reasons
    # Wall-clock recorded on sides for observability only — never a ranking input.
    wall_clock_recorded = {
        "a": side_a.wall_clock_seconds,
        "b": side_b.wall_clock_seconds,
        "used_for_rank": False,
    }
    # Official grade series pin (docs §17.4 / VAL-TELE-009): when require is set, missing or
    # bad challenge series fail-closes scientific grade (not silent PASS). Series never sole-rank.
    grade_a = apply_train_series_requirement_to_grade(
        record=side_a,
        series=train_series_a,
        pin=pin,
        expected_sha256=train_series_sha256_a,
    )
    grade_b = apply_train_series_requirement_to_grade(
        record=side_b,
        series=train_series_b,
        pin=pin,
        expected_sha256=train_series_sha256_b,
    )
    series_grade_ok = bool(grade_a["grade_valid"] and grade_b["grade_valid"])
    if pin.require_train_series and not series_grade_ok:
        validity_ok = False
    series_reasons: list[str] = []
    if pin.require_train_series:
        for prefix, grade in (("a", grade_a), ("b", grade_b)):
            for reason in grade.get("reasons") or []:
                series_reasons.append(f"{prefix}:{reason}")
    combined_validity_reasons = list(validity_reasons) + series_reasons
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "protocol_id": pin.protocol_id,
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_hash": protocol_pin_hash(pin),
        "mode": mode,
        "primary_form": pin.primary_form,
        "device_class": device_class,
        "score_class": score_class,
        "scorecard_id": SCORECARD_ID,
        "pin": pin.as_dict(),
        "side_a": _side_bundle_block(packed["a"], side_a),
        "side_b": _side_bundle_block(packed["b"], side_b),
        "seeds": list(pin.seeds),
        "aggregate": {
            "a": {
                "mean_heldout_delta": side_a.heldout_delta,
                "mean_bpb": side_a.bpb,
                "std_bpb": side_a.bpb_std,
                "overfit_rate": side_a.overfit_rate,
                "valid": side_a.valid,
                "seed_count": side_a.seed_count,
                "wall_clock_seconds": side_a.wall_clock_seconds,
            },
            "b": {
                "mean_heldout_delta": side_b.heldout_delta,
                "mean_bpb": side_b.bpb,
                "std_bpb": side_b.bpb_std,
                "overfit_rate": side_b.overfit_rate,
                "valid": side_b.valid,
                "seed_count": side_b.seed_count,
                "wall_clock_seconds": side_b.wall_clock_seconds,
            },
        },
        "ranking": {
            "winner": result.winner,
            "reason": result.reason,
            "rule": result.rule,
            "eps_heldout": result.eps_heldout,
            "eps_bpb": result.eps_bpb,
            "detail": result.detail,
            "tie_polar": result.tie_polar,
            "crown_allowed": result.crown_allowed,
            "default_v1_preserved_when_no_polar_conflict": not result.tie_polar,
            # Label-facing outcome: clear A vs B for operators/tests.
            "outcome_label": {
                "a": side_a.label,
                "b": side_b.label,
                "winner_side": result.winner,
                "winner_label": (
                    side_a.label
                    if result.winner == "a"
                    else side_b.label
                    if result.winner == "b"
                    else ("TIE_POLAR" if result.tie_polar else "tie")
                ),
                "crown_allowed": result.crown_allowed,
            },
            "wall_clock_ignored_for_rank": True,
        },
        "validity": {
            "ok": validity_ok,
            "reasons": list(combined_validity_reasons),
            "wall_clock_never_ranks": True,
            "miner_self_report_never_authoritative": True,
            "matched_budget": True,
            "required_entry_scripts": list(REQUIRED_ENTRY_SCRIPTS),
            "score_class": score_class,
            "require_train_series": bool(pin.require_train_series),
            "train_series_grade_ok": series_grade_ok,
        },
        "train_series_grade": {
            "require_train_series": bool(pin.require_train_series),
            "grade_valid": series_grade_ok,
            "silent_pass": False if (pin.require_train_series and not series_grade_ok) else None,
            "series_residual_only": True,
            "series_may_sole_rank": False,
            "side_a": grade_a,
            "side_b": grade_b,
        },
        "gpu_verification": gpu_info,
        "wall_clock_recorded": wall_clock_recorded,
        "labels": {
            "score_class": score_class,
            "provider_trust": PROVIDER_TRUST_LABEL,
            "image_pin": IMAGE_PIN_LABEL,
            "prism_tee_product": False,
            "wall_clock_never_ranks": True,
        },
        # Distinct from scorecard annex ``honesty_note`` (provisional K=1 language).
        "provider_honesty": (
            "PROVIDER_TRUST + LAB-GPU / IMAGE_PIN framing; Prism has no TEE verifier "
            "product; LAB-GPU rank is lab architecture comparison only"
        ),
    }
    if artifact_source is not None:
        report["artifact_source"] = artifact_source
    # Additive multimetric.v1.1 annex (does not rewrite emission leaderboard).
    return attach_scorecard_to_report(
        report,
        side_a,
        side_b,
        compare=result,
        matched_pin=True,
    )


def run_dual_family_official_compare(
    output_dir: Path | str,
    *,
    side_a_profile: FamilySynthProfile = DEFAULT_TRANSFORMER_PROFILE,
    side_b_profile: FamilySynthProfile = DEFAULT_MAMBA_PROFILE,
    pin: ProtocolPin | None = None,
    mode: CompareMode = "ArchCompare",
    device_class: DeviceClass = "fixture",
    package: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    """End-to-end dual-family official compare without NVIDIA.

    1. Package both unknown-style seed zips (optional if only synth metrics desired).
    2. Build multi-seed official records from synthetic challenge-owned metrics.
    3. Aggregate and :func:`compare_official`.
    4. Emit report dict (+ optional JSON under ``output_dir``).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    active_pin = pin if pin is not None else default_protocol_pin(device_class=device_class)
    gpu = gpu_verification_status()

    if package:
        packed = package_unknown_style_pair(
            out / "packages",
            side_a_family_id=side_a_profile.family_id,
            side_b_family_id=side_b_profile.family_id,
        )
    else:
        # Minimal stub packages for pure-metric unit paths that skip zip I/O:
        # still require families to be registered so fairness probes work.
        for fam in (side_a_profile.family_id, side_b_profile.family_id):
            if fam not in SEED_FAMILIES:
                raise KeyError(fam)
        packed = package_unknown_style_pair(
            out / "packages",
            side_a_family_id=side_a_profile.family_id,
            side_b_family_id=side_b_profile.family_id,
        )

    side_a = aggregate_side(side_a_profile, pin=active_pin, device=device_class)
    side_b = aggregate_side(side_b_profile, pin=active_pin, device=device_class)
    # Scorecard-aware compare: preserves v1 when no polar conflict; TIE_POLAR otherwise.
    result = compare_official_scorecard(side_a, side_b)
    report = build_compare_report(
        pin=active_pin,
        side_a=side_a,
        side_b=side_b,
        packed=packed,
        result=result,
        mode=mode,
        device_class=device_class,
        gpu=gpu,
        score_class=SCORE_CLASS_FIXTURE,
    )
    if write_report:
        report_path = out / "prism_compare_report.v1.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = {**report, "report_path": str(report_path)}
    return report


class LabGpuArtifactsMissingError(FileNotFoundError):
    """Raised when dual-family LAB-GPU train manifests are missing for host rank."""


def run_lab_gpu_host_official_compare(
    artifacts_root: Path | str,
    output_dir: Path | str,
    *,
    side_a_family_id: str = SIDE_A_FAMILY_ID,
    side_b_family_id: str = SIDE_B_FAMILY_ID,
    seeds: Sequence[int] = (LAB_GPU_DEFAULT_SEED,),
    pin: ProtocolPin | None = None,
    mode: CompareMode = "ArchCompare",
    write_report: bool = True,
    package: bool = True,
) -> dict[str, Any]:
    """Host-side Official Comparison from real LAB-GPU Lium train artifacts.

    Recomputes official scores with :func:`official_record_from_manifest`, then
    :func:`compare_official` under held-out primary / bpb secondary. Emits
    ``prism_compare_report.v1`` with ``score_class=LAB-GPU`` (not fixture synthetic;
    not DEFERRED-for-no-nvidia). Wall-clock is recorded but ignored for rank.
    Honesty labels: PROVIDER_TRUST / LAB-GPU / IMAGE_PIN (no Prism TEE product).

    Raises :class:`LabGpuArtifactsMissingError` when either family lacks manifests
    (callers should treat that as a clear BLOCKED handoff — no invented scores).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(artifacts_root)
    seed_tuple = tuple(int(s) for s in seeds)
    active_pin = (
        pin
        if pin is not None
        else ProtocolPin(
            protocol_id=PROTOCOL_ID,
            token_budget=int(protocol_budget_constants()["token_budget"]),
            seeds=seed_tuple,
            param_cap=int(protocol_budget_constants()["param_cap"]),
            seq_len=int(protocol_budget_constants()["seq_len"]),
            batch_size=int(protocol_budget_constants()["batch_size"]),
            tokenizer=str(protocol_budget_constants()["tokenizer"]),
            vocab_size=int(protocol_budget_constants()["vocab_size"]),
            scored_nproc=int(protocol_budget_constants()["scored_nproc"]),
            val_byte_budget=int(protocol_budget_constants()["val_byte_budget"]),
            force_iter_train_batches=True,
            require_trained_state=True,
            primary_form="heldout_delta",
        )
    )

    a_per_seed, b_per_seed, missing = records_from_lab_gpu_artifacts(
        root,
        side_a_family_id=side_a_family_id,
        side_b_family_id=side_b_family_id,
        seeds=seed_tuple,
        pin=active_pin,
    )
    if missing or not a_per_seed or not b_per_seed:
        raise LabGpuArtifactsMissingError(
            "LAB-GPU host compare BLOCKED: missing dual-family train artifacts: "
            + ("; ".join(missing) if missing else "empty official records")
        )

    side_a = aggregate_official_records(
        a_per_seed,
        label=side_a_family_id,
        primary_form=active_pin.primary_form,
    )
    side_b = aggregate_official_records(
        b_per_seed,
        label=side_b_family_id,
        primary_form=active_pin.primary_form,
    )
    # Preserve diagnostic wall_clock mean from per-seed recomputes on aggregates.
    # aggregate_official_records drops wall_clock; re-attach for observability only
    # while keeping multimetric.v1.1 scorecard fields via dataclasses.replace.
    a_clocks = [r.wall_clock_seconds for r in a_per_seed if r.wall_clock_seconds is not None]
    b_clocks = [r.wall_clock_seconds for r in b_per_seed if r.wall_clock_seconds is not None]
    if a_clocks:
        side_a = replace(side_a, wall_clock_seconds=sum(a_clocks) / len(a_clocks))
    if b_clocks:
        side_b = replace(side_b, wall_clock_seconds=sum(b_clocks) / len(b_clocks))

    if package:
        packed = package_unknown_style_pair(
            out / "packages",
            side_a_family_id=side_a_family_id,
            side_b_family_id=side_b_family_id,
        )
    else:
        packed = package_unknown_style_pair(
            out / "packages",
            side_a_family_id=side_a_family_id,
            side_b_family_id=side_b_family_id,
        )

    result = compare_official_scorecard(side_a, side_b)
    gpu = lab_gpu_verification_status()
    report = build_compare_report(
        pin=active_pin,
        side_a=side_a,
        side_b=side_b,
        packed=packed,
        result=result,
        mode=mode,
        device_class="lab-gpu",
        gpu=gpu,
        score_class=SCORE_CLASS_LAB_GPU,
        artifact_source=str(root),
    )
    # Per-seed recomputed surfaces for evidence without trusting miner summaries alone.
    report["per_seed"] = {
        "a": [
            {
                "label": r.label,
                "bpb": r.bpb,
                "heldout_delta": r.heldout_delta,
                "val_bpb_trained": r.val_bpb_trained,
                "wall_clock_seconds": r.wall_clock_seconds,
                "valid": r.valid,
                "step0_anomaly": r.step0_anomaly,
                "memorization_flag": r.memorization_flag,
            }
            for r in a_per_seed
        ],
        "b": [
            {
                "label": r.label,
                "bpb": r.bpb,
                "heldout_delta": r.heldout_delta,
                "val_bpb_trained": r.val_bpb_trained,
                "wall_clock_seconds": r.wall_clock_seconds,
                "valid": r.valid,
                "step0_anomaly": r.step0_anomaly,
                "memorization_flag": r.memorization_flag,
            }
            for r in b_per_seed
        ],
    }
    report["labels"] = {
        "score_class": SCORE_CLASS_LAB_GPU,
        "provider_trust": PROVIDER_TRUST_LABEL,
        "image_pin": IMAGE_PIN_LABEL,
        "prism_tee_product": False,
        "wall_clock_never_ranks": True,
        "miner_self_report_never_authoritative": True,
        "not_fixture_only_synthetic": True,
        "not_deferred_for_no_nvidia": True,
    }

    if write_report:
        report_path = out / "prism_compare_report.v1.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = {**report, "report_path": str(report_path)}
    return report


def main(argv: list[str] | None = None) -> int:
    """Operator CLI: offline dual-family official compare (CPU/fixture or LAB-GPU)."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Prism Official Comparison Protocol v1 dual-family harness "
            f"({SIDE_A_FAMILY_ID} vs {SIDE_B_FAMILY_ID}). Default path is CPU/fixture "
            "synthetic metrics. Use --lab-gpu-artifacts for host rank of real Lium "
            "CUDA train manifests (score_class=LAB-GPU). Labels: PROVIDER_TRUST / "
            "LAB-GPU / IMAGE_PIN (no Prism TEE product)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/official-compare"),
        help="Directory for seed zips + prism_compare_report.v1.json",
    )
    parser.add_argument(
        "--mode",
        choices=("ArchCompare", "TrainCompare", "SystemCompare"),
        default="ArchCompare",
        help="Official compare mode (default ArchCompare).",
    )
    parser.add_argument(
        "--device-class",
        choices=("cpu", "cuda", "fixture", "lab-gpu"),
        default="fixture",
        help=(
            "Report device class for the fixture path. Ignored when "
            "--lab-gpu-artifacts is set (then device_class=lab-gpu)."
        ),
    )
    parser.add_argument(
        "--lab-gpu-artifacts",
        type=Path,
        default=None,
        help=(
            "Root containing {family}/seed-{N}/prism_run_manifest.v2.json from real "
            "LAB-GPU trains. Host recomputes + compare_official; score_class=LAB-GPU."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Seed(s) to load under --lab-gpu-artifacts (default: 1337). Repeatable.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full compare report JSON to stdout.",
    )
    args = parser.parse_args(argv)

    if args.lab_gpu_artifacts is not None:
        seeds = tuple(args.seed) if args.seed else (LAB_GPU_DEFAULT_SEED,)
        try:
            report = run_lab_gpu_host_official_compare(
                args.lab_gpu_artifacts,
                args.output_dir,
                seeds=seeds,
                mode=args.mode,  # type: ignore[arg-type]
            )
        except LabGpuArtifactsMissingError as exc:
            print(f"BLOCKED: {exc}")
            return 2
    else:
        report = run_dual_family_official_compare(
            args.output_dir,
            mode=args.mode,  # type: ignore[arg-type]
            device_class=args.device_class,  # type: ignore[arg-type]
        )
    ranking = report["ranking"]
    gpu = report["gpu_verification"]
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"protocol: {report['protocol_id']} hash={report['protocol_hash'][:16]}…")
        print(
            f"mode: {report['mode']} device_class={report['device_class']} "
            f"score_class={report.get('score_class')}"
        )
        print(
            f"side A: {report['side_a']['label']} "
            f"heldout_delta={report['side_a']['mean_heldout_delta']} "
            f"bpb={report['side_a']['mean_bpb']}"
        )
        print(
            f"side B: {report['side_b']['label']} "
            f"heldout_delta={report['side_b']['mean_heldout_delta']} "
            f"bpb={report['side_b']['mean_bpb']}"
        )
        print(
            f"compare outcome: winner={ranking['winner']} "
            f"({ranking['outcome_label']['winner_label']}) "
            f"reason={ranking['reason']}"
        )
        print(
            f"gpu_verification: {gpu['status']} ({gpu['reason']}); "
            f"claim_gpu_pass={gpu['claim_gpu_pass']}"
        )
        labels = report.get("labels") or {}
        print(
            f"provider_trust={labels.get('provider_trust')} "
            f"image_pin={labels.get('image_pin')} "
            f"prism_tee_product={labels.get('prism_tee_product')}"
        )
        if report.get("provider_honesty"):
            print(f"provider_honesty: {report['provider_honesty']}")
        print(f"wall_clock_ignored_for_rank={ranking.get('wall_clock_ignored_for_rank')}")
        if report.get("report_path"):
            print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
