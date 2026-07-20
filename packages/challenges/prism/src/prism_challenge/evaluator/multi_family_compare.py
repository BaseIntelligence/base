"""N-family Official Comparison under one matched ProtocolPin (arXiv fair eval).

Extends dual-family Official Comparison for **arbitrary N packages** (Imp baselines +
novel seeds) under a single architecture-agnostic score path:

* One :class:`ProtocolPin` for all sides (token_budget, seeds, tokenizer gpt2,
  seq/batch matched, stage explore by default for this residual).
* Challenge-owned secondary bpb always recomputed by Prism.
* Held-out primary rank key shared with dual-family / emission Official axes.
* Score class ``LAB-GPU`` for real remote CUDA train manifests; ``fixture`` only when
  synthetic (must label). Failures are ``BLOCKED_with_reason`` — never invent metrics.
* Wall-clock may be recorded; never ranks. Honesty: PROVIDER_TRUST / LAB-GPU / IMAGE_PIN
  (no Prism TEE product).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from prism_challenge.seed_packaging import (
    REQUIRED_ENTRY_SCRIPTS,
    SEED_FAMILIES,
    PackedSeed,
    get_family,
    package_seed_zip,
)

from .official_compare_harness import (
    DEFAULT_MAMBA_PROFILE,
    DEFAULT_TRANSFORMER_PROFILE,
    IMAGE_PIN_LABEL,
    LAB_GPU_DEFAULT_SEED,
    LAB_GPU_MANIFEST_NAME,
    PROVIDER_TRUST_LABEL,
    REPORT_SCHEMA,
    SCORE_CLASS_FIXTURE,
    SCORE_CLASS_LAB_GPU,
    SIDE_A_FAMILY_ID,
    SIDE_B_FAMILY_ID,
    DeviceClass,
    FamilySynthProfile,
    ScoreClass,
    SynthSeedMetrics,
    aggregate_side,
    default_protocol_pin,
    gpu_verification_status,
    lab_gpu_verification_status,
    load_lab_gpu_manifest,
    protocol_pin_hash,
    records_for_profile,
)
from .official_comparison import (
    OFFICIAL_DEFAULT_SEEDS,
    OFFICIAL_EXPLORE_PARAM_CAP,
    OFFICIAL_EXPLORE_STAGE,
    PROTOCOL_ID,
    PROTOCOL_SCHEMA,
    OfficialScoreRecord,
    ProtocolPin,
    aggregate_official_records,
    official_rank_key,
    official_record_from_manifest,
)

# Imp baselines + three arXiv-class novels (VAL-ARXEVAL-002/003 sealed packages).
IMP_BASELINE_FAMILY_IDS: tuple[str, ...] = (
    SIDE_A_FAMILY_ID,  # transformer-tiny-1m
    SIDE_B_FAMILY_ID,  # mamba-tiny-1m
)
NOVEL_ARXIV_FAMILY_IDS: tuple[str, ...] = (
    "deeploop-tiny-1m",
    "gated-delta-tiny-1m",
    "hybrid-attn-ssm-tiny-1m",
)
ARXIV_FAIR_EVAL_FAMILY_IDS: tuple[str, ...] = IMP_BASELINE_FAMILY_IDS + NOVEL_ARXIV_FAMILY_IDS

# Frontier fair cup (VAL-FRNTEVAL-004/005): Imp + deeploop prior winner + three
# frontier-inspired mechanism distillations (MLA / DeepSeekMoE / KDA). Not full V4/K3.
DEEPLOOP_CONTROL_FAMILY_ID = "deeploop-tiny-1m"
NOVEL_FRONTIER_FAMILY_IDS: tuple[str, ...] = (
    "mla-tiny-1m",
    "ds-moe-tiny-1m",
    "kda-tiny-1m",
)
FRONTIER_FAIR_EVAL_FAMILY_IDS: tuple[str, ...] = (
    IMP_BASELINE_FAMILY_IDS + (DEEPLOOP_CONTROL_FAMILY_ID,) + NOVEL_FRONTIER_FAMILY_IDS
)

MULTI_FAMILY_REPORT_SCHEMA = "prism_multi_family_compare_report.v1"


class MultiFamilyArtifactsMissingError(FileNotFoundError):
    """Raised when LAB-GPU multi-family host rank cannot load required manifests."""


def explore_protocol_pin(
    *,
    seeds: Sequence[int] | None = None,
    token_budget: int | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
    k_label: str | None = None,
) -> ProtocolPin:
    """Matched ProtocolPin for arXiv fair multi-arch residual (stage **explore**).

    All families share: token_budget, seeds K, tokenizer gpt2, seq/batch, explore
    param_cap (124M). Prefer K≥3 when cheap; callers may set K=1 and label it.

    ``seq_len`` / ``token_budget`` pass through without a hardcoded 128-only trap
    (VAL-SCALE-006). Defaults remain Official short-ctx (128 / 500k) for residual
    continuity; P1+ scale cups pass raised values explicitly.
    """
    del k_label  # report metadata only; not a rank / pin key
    base = default_protocol_pin(device_class="fixture")
    seed_tuple = tuple(int(s) for s in seeds) if seeds is not None else base.seeds
    budget = int(token_budget) if token_budget is not None else int(base.token_budget)
    resolved_seq = int(seq_len) if seq_len is not None else int(base.seq_len)
    resolved_batch = int(batch_size) if batch_size is not None else int(base.batch_size)
    if resolved_seq <= 0:
        raise ValueError(f"seq_len must be positive; got {resolved_seq}")
    if budget <= 0:
        raise ValueError(f"token_budget must be positive; got {budget}")
    # Explore stage + explore ceiling (research-lab small-first residual).
    return ProtocolPin(
        protocol_id=PROTOCOL_ID,
        token_budget=budget,
        seeds=seed_tuple,
        param_cap=int(OFFICIAL_EXPLORE_PARAM_CAP),
        param_ladder_stage=str(OFFICIAL_EXPLORE_STAGE),
        seq_len=resolved_seq,
        batch_size=resolved_batch,
        tokenizer=str(base.tokenizer),
        vocab_size=int(base.vocab_size),
        scored_nproc=int(base.scored_nproc),
        val_byte_budget=int(base.val_byte_budget),
        force_iter_train_batches=True,
        require_trained_state=True,
        primary_form="heldout_delta",
        require_train_series=False,
        # carry k honesty via wall_clock diagnostic slot unused; k noted in reports
        # (k_label is report metadata, not a rank key).
        wall_clock_seconds=base.wall_clock_seconds,
        step_budget=base.step_budget,
        gap_threshold_bpb=base.gap_threshold_bpb,
    )


def package_unknown_style_families(
    output_dir: Path | str,
    family_ids: Sequence[str],
) -> dict[str, PackedSeed]:
    """Package N registered seed families as unknown-style two-script zips.

    Fairness: every zip must expose the same required entry scripts. Distinct
    architecture bytes yield distinct content hashes; no family-specific score path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    packed: dict[str, PackedSeed] = {}
    required = set(REQUIRED_ENTRY_SCRIPTS)
    for family_id in family_ids:
        if family_id not in SEED_FAMILIES:
            raise KeyError(f"unknown seed family {family_id!r}; known={list(SEED_FAMILIES)}")
        item = package_seed_zip(family_id, out)
        missing = [n for n in required if n not in set(item.entry_names)]
        if missing:
            raise RuntimeError(
                f"unknown-style package missing entries family={family_id} missing={missing}"
            )
        packed[family_id] = item
    return packed


def records_from_lab_gpu_artifacts_multi(
    artifacts_root: Path | str,
    family_ids: Sequence[str],
    *,
    seeds: Sequence[int] = (LAB_GPU_DEFAULT_SEED,),
    pin: ProtocolPin | None = None,
    miner_reported: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[OfficialScoreRecord]], dict[str, list[str]]]:
    """Load/recompute official records for N families from LAB-GPU manifests.

    Returns ``(records_by_family, missing_by_family)``. Missing is empty when full.
    Never invents metrics for absent families — callers must mark BLOCKED_with_reason.
    """
    active_pin = pin if pin is not None else explore_protocol_pin(seeds=seeds)
    records_by_family: dict[str, list[OfficialScoreRecord]] = {}
    missing_by_family: dict[str, list[str]] = {}
    for family_id in family_ids:
        bucket: list[OfficialScoreRecord] = []
        missing: list[str] = []
        for seed in seeds:
            if seed not in active_pin.seeds:
                continue
            try:
                manifest = load_lab_gpu_manifest(artifacts_root, family_id, seed=seed)
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            # Architecture-agnostic: same official_record_from_manifest for every family.
            rec = official_record_from_manifest(
                manifest,
                label=f"{family_id}:seed={seed}",
                primary_form=active_pin.primary_form,
                miner_reported=miner_reported,
            )
            bucket.append(rec)
        records_by_family[family_id] = bucket
        if missing:
            missing_by_family[family_id] = missing
    return records_by_family, missing_by_family


def _default_synth_profile(family_id: str) -> FamilySynthProfile:
    """Deterministic fixture profile for multi-family unit paths (labeled fixture).

    Held-out and bpb magnitudes differ by family so ranking can exercise
    architecture-agnostic sort without inventing LAB-GPU claims.
    """
    if family_id == SIDE_A_FAMILY_ID:
        return DEFAULT_TRANSFORMER_PROFILE
    if family_id == SIDE_B_FAMILY_ID:
        return DEFAULT_MAMBA_PROFILE
    # Novel defaults: distinct but fixed offsets from Imp (fixture only).
    offsets = {
        "deeploop-tiny-1m": (1.40, 0.70),
        "gated-delta-tiny-1m": (1.25, 0.55),
        "hybrid-attn-ssm-tiny-1m": (1.55, 0.80),
        # Frontier-inspired mechanism distillations (fixture paths only).
        "mla-tiny-1m": (1.48, 0.72),
        "ds-moe-tiny-1m": (1.35, 0.62),
        "kda-tiny-1m": (1.30, 0.58),
    }
    bpb_base, hd_base = offsets.get(family_id, (1.60, 0.50))
    fam = get_family(family_id)
    seeds = tuple(
        SynthSeedMetrics(
            seed=seed,
            bpb=bpb_base + 0.02 * i,
            heldout_delta=hd_base + 0.03 * ((-1) ** i),
            train_heldout_gap=0.15,
        )
        for i, seed in enumerate(OFFICIAL_DEFAULT_SEEDS)
    )
    return FamilySynthProfile(
        family_id=family_id,
        label=family_id,
        architecture_family=fam.architecture_family,
        seeds=seeds,
    )


def _aggregate_family(
    family_id: str,
    per_seed: list[OfficialScoreRecord],
    *,
    pin: ProtocolPin,
) -> OfficialScoreRecord | None:
    if not per_seed:
        return None
    agg = aggregate_official_records(
        per_seed,
        label=family_id,
        primary_form=pin.primary_form,
    )
    clocks = [r.wall_clock_seconds for r in per_seed if r.wall_clock_seconds is not None]
    if clocks:
        agg = replace(agg, wall_clock_seconds=sum(clocks) / len(clocks))
    return agg


def rank_multi_family_records(
    aggregates: Mapping[str, OfficialScoreRecord],
) -> list[dict[str, Any]]:
    """Sort family aggregates by Official rank key (held-out primary, bpb secondary).

    Wall-clock and family_id as architecture name never enter the bit rank; label is
    only a total-order tie break inside
    :func:`~prism_challenge.evaluator.official_comparison.official_rank_key`.
    """
    rows: list[tuple[tuple[Any, ...], str, OfficialScoreRecord]] = []
    for family_id, rec in aggregates.items():
        key = official_rank_key(rec)
        rows.append((key, family_id, rec))
    rows.sort(key=lambda item: item[0])
    table: list[dict[str, Any]] = []
    for rank, (_key, family_id, rec) in enumerate(rows, start=1):
        fam = get_family(family_id) if family_id in SEED_FAMILIES else None
        table.append(
            {
                "rank": rank,
                "family_id": family_id,
                "label": rec.label,
                "architecture_family": (fam.architecture_family if fam is not None else "unknown"),
                "heldout_delta": rec.heldout_delta,
                "bpb": rec.bpb,
                "bpb_std": rec.bpb_std,
                "overfit_rate": rec.overfit_rate,
                "memorization_flag": rec.memorization_flag,
                "step0_anomaly": rec.step0_anomaly,
                "valid": rec.valid,
                "seed_count": rec.seed_count,
                "wall_clock_seconds": rec.wall_clock_seconds,
                "status": "OK" if rec.valid and not rec.step0_anomaly else "INVALID",
            }
        )
    return table


def _side_block(
    family_id: str,
    packed: PackedSeed | None,
    aggregate: OfficialScoreRecord | None,
    *,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    fam = get_family(family_id) if family_id in SEED_FAMILIES else None
    block: dict[str, Any] = {
        "family_id": family_id,
        "architecture_family": fam.architecture_family if fam is not None else None,
        "unknown_style": True,
        "status": status,
        "reason": reason,
        "rank_key_path": "official_rank_key(heldout_primary_then_bpb_secondary)",
    }
    if packed is not None:
        block.update(
            {
                "bundle_hash": packed.content_sha256,
                "zip_path": str(packed.zip_path),
                "entry_names": list(packed.entry_names),
                "size_bytes": packed.size_bytes,
            }
        )
    if aggregate is not None:
        block.update(
            {
                "label": aggregate.label,
                "mean_heldout_delta": aggregate.heldout_delta,
                "mean_bpb": aggregate.bpb,
                "std_bpb": aggregate.bpb_std,
                "overfit_rate": aggregate.overfit_rate,
                "memorization_flag": aggregate.memorization_flag,
                "step0_anomaly": aggregate.step0_anomaly,
                "valid": aggregate.valid,
                "seed_count": aggregate.seed_count,
                "wall_clock_seconds": aggregate.wall_clock_seconds,
            }
        )
    return block


def build_multi_family_report(
    *,
    pin: ProtocolPin,
    family_ids: Sequence[str],
    aggregates: Mapping[str, OfficialScoreRecord],
    packed: Mapping[str, PackedSeed],
    blocked: Mapping[str, str],
    score_class: ScoreClass,
    device_class: DeviceClass,
    gpu: Mapping[str, Any] | None = None,
    artifact_source: str | None = None,
    k_label: str | None = None,
    seed_profiles_note: str | None = None,
) -> dict[str, Any]:
    """Emit multi-family compare document + architecture-agnostic score table."""
    if gpu is not None:
        gpu_info = dict(gpu)
    elif score_class == SCORE_CLASS_LAB_GPU:
        gpu_info = lab_gpu_verification_status()
    else:
        gpu_info = gpu_verification_status()

    table = rank_multi_family_records(aggregates)
    sides: dict[str, Any] = {}
    for family_id in family_ids:
        if family_id in blocked:
            sides[family_id] = _side_block(
                family_id,
                packed.get(family_id),
                None,
                status="BLOCKED_with_reason",
                reason=blocked[family_id],
            )
        elif family_id in aggregates:
            sides[family_id] = _side_block(
                family_id,
                packed.get(family_id),
                aggregates[family_id],
                status="OK",
            )
        else:
            sides[family_id] = _side_block(
                family_id,
                packed.get(family_id),
                None,
                status="BLOCKED_with_reason",
                reason="no_aggregate_record",
            )

    winner_row = table[0] if table else None
    runner_up = table[1] if len(table) > 1 else None
    seeds_list = list(pin.seeds)
    k = len(seeds_list)
    if k_label is None:
        if k >= 3:
            k_label = f"K={k} public multi-seed"
        else:
            k_label = f"K={k} lab single-seed (not public-claim K≥3)"

    validity_ok = bool(table) and all(r["valid"] and not r["step0_anomaly"] for r in table)
    report: dict[str, Any] = {
        "schema": MULTI_FAMILY_REPORT_SCHEMA,
        "dual_compare_schema_compat": REPORT_SCHEMA,
        "protocol_id": pin.protocol_id,
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_hash": protocol_pin_hash(pin),
        "mode": "ArchCompare",
        "primary_form": pin.primary_form,
        "device_class": device_class,
        "score_class": score_class,
        "pin": pin.as_dict(),
        "k_label": k_label,
        "seeds": seeds_list,
        "family_ids": list(family_ids),
        "sides": sides,
        "score_table": table,
        "ranking": {
            "rule": "heldout_primary_then_bpb_secondary",
            "wall_clock_ignored_for_rank": True,
            "architecture_agnostic": True,
            "family_branch_in_rank_key": False,
            "winner": winner_row["family_id"] if winner_row else None,
            "winner_heldout_delta": winner_row["heldout_delta"] if winner_row else None,
            "winner_bpb": winner_row["bpb"] if winner_row else None,
            "runner_up": runner_up["family_id"] if runner_up else None,
            "runner_up_heldout_delta": runner_up["heldout_delta"] if runner_up else None,
            "runner_up_bpb": runner_up["bpb"] if runner_up else None,
            "blocked_families": dict(blocked),
            "scored_count": len(table),
            "requested_count": len(family_ids),
        },
        "validity": {
            "ok": validity_ok and not blocked,
            "matched_budget": True,
            "wall_clock_never_ranks": True,
            "miner_self_report_never_authoritative": True,
            "architecture_agnostic_path": True,
            "required_entry_scripts": list(REQUIRED_ENTRY_SCRIPTS),
            "score_class": score_class,
            "blocked": dict(blocked),
            "all_sides_same_pin_hash": True,
        },
        "gpu_verification": gpu_info,
        "provider_honesty": (
            "PROVIDER_TRUST + LAB-GPU / IMAGE_PIN framing; multi-family LAB-GPU rank "
            "is lab architecture comparison only (no Prism TEE product)"
        ),
        "labels": {
            "score_class": score_class,
            "provider_trust": PROVIDER_TRUST_LABEL,
            "image_pin": IMAGE_PIN_LABEL,
            "prism_tee_product": False,
            "wall_clock_never_ranks": True,
            "architecture_agnostic": True,
            "not_fixture_only_synthetic": score_class == SCORE_CLASS_LAB_GPU,
            "fixture_labelled": score_class == SCORE_CLASS_FIXTURE,
        },
    }
    if artifact_source is not None:
        report["artifact_source"] = artifact_source
    if seed_profiles_note is not None:
        report["seed_profiles_note"] = seed_profiles_note
    return report


def run_multi_family_official_compare(
    output_dir: Path | str,
    *,
    family_ids: Sequence[str] = ARXIV_FAIR_EVAL_FAMILY_IDS,
    pin: ProtocolPin | None = None,
    profiles: Mapping[str, FamilySynthProfile] | None = None,
    device_class: DeviceClass = "fixture",
    package: bool = True,
    write_report: bool = True,
    score_class: ScoreClass = SCORE_CLASS_FIXTURE,
) -> dict[str, Any]:
    """End-to-end multi-family Official path without NVIDIA (fixture/CPU metrics).

    Synthetic metrics are **always** labeled ``score_class=fixture``. Prefer
    :func:`run_multi_family_lab_gpu_host_compare` for real LAB-GPU ranks.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    active_pin = pin if pin is not None else explore_protocol_pin()
    ids = tuple(family_ids)

    if package:
        packed = package_unknown_style_families(out / "packages", ids)
    else:
        packed = package_unknown_style_families(out / "packages", ids)

    aggregates: dict[str, OfficialScoreRecord] = {}
    blocked: dict[str, str] = {}
    for family_id in ids:
        profile = (
            profiles[family_id]
            if profiles is not None and family_id in profiles
            else _default_synth_profile(family_id)
        )
        # Agnostic path: same records_for_profile / official_record_from_manifest stack.
        per_seed = records_for_profile(profile, pin=active_pin, device=device_class)
        if not per_seed:
            blocked[family_id] = "BLOCKED_with_reason:no_seed_records_under_pin"
            continue
        agg = aggregate_side(profile, pin=active_pin, device=device_class)
        aggregates[family_id] = agg

    report = build_multi_family_report(
        pin=active_pin,
        family_ids=ids,
        aggregates=aggregates,
        packed=packed,
        blocked=blocked,
        score_class=score_class,
        device_class=device_class,
        seed_profiles_note=(
            "synthetic challenge-owned metrics; score_class=fixture "
            "(not LAB-GPU; PROVIDER_TRUST / IMAGE_PIN framing only)"
            if score_class == SCORE_CLASS_FIXTURE
            else None
        ),
    )
    if write_report:
        report_path = out / "multi_family_compare_report.json"
        table_path = out / "score_table.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        table_path.write_text(
            json.dumps(report["score_table"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pin_path = out / "pin.json"
        pin_path.write_text(
            json.dumps(
                {
                    **active_pin.as_dict(),
                    "protocol_hash": protocol_pin_hash(active_pin),
                    "families": list(ids),
                    "score_class_for_this_run": score_class,
                    "k_label": report.get("k_label"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report = {
            **report,
            "report_path": str(report_path),
            "score_table_path": str(table_path),
            "pin_path": str(pin_path),
        }
    return report


def run_multi_family_lab_gpu_host_compare(
    artifacts_root: Path | str,
    output_dir: Path | str,
    *,
    family_ids: Sequence[str] = ARXIV_FAIR_EVAL_FAMILY_IDS,
    seeds: Sequence[int] = (LAB_GPU_DEFAULT_SEED,),
    pin: ProtocolPin | None = None,
    package: bool = True,
    write_report: bool = True,
    allow_partial: bool = True,
) -> dict[str, Any]:
    """Host-side multi-family Official Comparison from real LAB-GPU manifests.

    Incomplete sides are marked ``BLOCKED_with_reason`` (never invented). When
    ``allow_partial`` is False and any family is missing, raises
    :class:`MultiFamilyArtifactsMissingError`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(artifacts_root)
    seed_tuple = tuple(int(s) for s in seeds)
    active_pin = pin if pin is not None else explore_protocol_pin(seeds=seed_tuple)
    # Ensure pin seeds match requested seed tuple when caller passed a custom pin.
    if pin is not None and set(seed_tuple) - set(pin.seeds):
        raise ValueError("requested seeds must be a subset of ProtocolPin.seeds")

    ids = tuple(family_ids)
    if package:
        packed = package_unknown_style_families(out / "packages", ids)
    else:
        packed = package_unknown_style_families(out / "packages", ids)

    records_by_family, missing_by_family = records_from_lab_gpu_artifacts_multi(
        root,
        ids,
        seeds=seed_tuple,
        pin=active_pin,
    )

    aggregates: dict[str, OfficialScoreRecord] = {}
    blocked: dict[str, str] = {}
    for family_id in ids:
        missing = missing_by_family.get(family_id) or []
        per_seed = records_by_family.get(family_id) or []
        if missing and not per_seed:
            blocked[family_id] = "BLOCKED_with_reason:missing_lab_gpu_manifest:" + ";".join(missing)
            continue
        if missing and per_seed:
            # Partial seeds under K: keep scored seeds, note residual missing.
            blocked[family_id] = "PARTIAL_with_reason:missing_some_seed_manifests:" + ";".join(
                missing
            )
        agg = _aggregate_family(family_id, per_seed, pin=active_pin)
        if agg is None:
            blocked[family_id] = "BLOCKED_with_reason:empty_official_records"
            continue
        aggregates[family_id] = agg

    if not allow_partial:
        hard_blocked = {k: v for k, v in blocked.items() if v.startswith("BLOCKED_with_reason")}
        if hard_blocked or not aggregates:
            raise MultiFamilyArtifactsMissingError(
                "LAB-GPU multi-family host compare BLOCKED: "
                + json.dumps(hard_blocked or {"_": "no aggregates"})
            )

    k = len(seed_tuple)
    k_label = (
        f"K={k} public multi-seed" if k >= 3 else f"K={k} lab single-seed (not public-claim K≥3)"
    )
    report = build_multi_family_report(
        pin=active_pin,
        family_ids=ids,
        aggregates=aggregates,
        packed=packed,
        blocked={k: v for k, v in blocked.items() if v.startswith("BLOCKED")},
        score_class=SCORE_CLASS_LAB_GPU,
        device_class="lab-gpu",
        artifact_source=str(root),
        k_label=k_label,
        seed_profiles_note=(
            "real LAB-GPU CUDA train manifests; host Prism recompute secondary bpb; "
            "PROVIDER_TRUST / IMAGE_PIN (no Prism TEE product)"
        ),
    )
    # Surface soft PARTIAL notes without inventing scores.
    soft = {k: v for k, v in blocked.items() if v.startswith("PARTIAL")}
    if soft:
        report["partial_notes"] = soft

    report["per_seed"] = {
        family_id: [
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
            for r in (records_by_family.get(family_id) or [])
        ]
        for family_id in ids
    }

    if write_report:
        report_path = out / "multi_family_compare_report.json"
        table_path = out / "score_table.json"
        pin_path = out / "pin.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        table_path.write_text(
            json.dumps(report["score_table"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pin_path.write_text(
            json.dumps(
                {
                    **active_pin.as_dict(),
                    "protocol_hash": protocol_pin_hash(active_pin),
                    "families": list(ids),
                    "score_class": SCORE_CLASS_LAB_GPU,
                    "artifact_source": str(root),
                    "k_label": k_label,
                    "lab_gpu_manifest_name": LAB_GPU_MANIFEST_NAME,
                    "provider_trust": PROVIDER_TRUST_LABEL,
                    "image_pin": IMAGE_PIN_LABEL,
                    "prism_tee_product": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report = {
            **report,
            "report_path": str(report_path),
            "score_table_path": str(table_path),
            "pin_path": str(pin_path),
        }
    return report


def agnostic_equal_metrics_equal_rank(
    *,
    heldout_delta: float = 1.0,
    bpb: float = 1.5,
    family_tags: Sequence[str] = ("transformer", "mamba", "deeploop", "alien"),
) -> list[tuple[Any, ...]]:
    """Prove fictitious family tags do not alter Official rank keys (VAL-ARXEVAL-005).

    Returns the rank keys — they must all be equal for equal metrics.
    """
    keys: list[tuple[Any, ...]] = []
    for tag in family_tags:
        rec = OfficialScoreRecord(
            label=f"tag:{tag}",
            bpb=float(bpb),
            heldout_delta=float(heldout_delta),
            valid=True,
            step0_anomaly=False,
            memorization_flag=False,
            overfit_rate=0.0,
            primary_form="heldout_delta",
            seed_count=1,
        )
        keys.append(official_rank_key(rec))
    return keys


def main(argv: list[str] | None = None) -> int:
    """Operator CLI: multi-family Official compare (fixture or LAB-GPU host)."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Prism multi-family Official Comparison under one ProtocolPin "
            f"(default families: {', '.join(ARXIV_FAIR_EVAL_FAMILY_IDS)}). "
            "Default path is fixture synthetic metrics. Use --lab-gpu-artifacts for "
            "host rank of real Lium CUDA train manifests (score_class=LAB-GPU). "
            "Labels: PROVIDER_TRUST / LAB-GPU / IMAGE_PIN (no Prism TEE product)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/multi-family-compare"),
        help="Directory for packages + multi_family_compare_report.json",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        help=(
            "Family id (repeatable). Default: Imp baselines + three arXiv novels. "
            "Pass frontier ids (mla/ds-moe/kda + controls) or use --frontier-cup."
        ),
    )
    parser.add_argument(
        "--frontier-cup",
        action="store_true",
        help=(
            "Use FRONTIER_FAIR_EVAL_FAMILY_IDS (Imp + deeploop + mla/ds-moe/kda) "
            "when --family is omitted."
        ),
    )
    parser.add_argument(
        "--lab-gpu-artifacts",
        type=Path,
        default=None,
        help="Root with {family}/seed-{N}/prism_run_manifest.v2.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Seed(s) under --lab-gpu-artifacts (default 1337). Repeatable.",
    )
    parser.add_argument(
        "--stage-explore",
        action="store_true",
        default=True,
        help="Use explore pin (124M). Default on for arXiv fair residual.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full report JSON to stdout.",
    )
    args = parser.parse_args(argv)
    if args.family:
        families = tuple(args.family)
    elif args.frontier_cup:
        families = FRONTIER_FAIR_EVAL_FAMILY_IDS
    else:
        families = ARXIV_FAIR_EVAL_FAMILY_IDS
    seeds = tuple(args.seed) if args.seed else (LAB_GPU_DEFAULT_SEED,)
    pin = explore_protocol_pin(seeds=seeds if args.lab_gpu_artifacts else OFFICIAL_DEFAULT_SEEDS)

    if args.lab_gpu_artifacts is not None:
        try:
            report = run_multi_family_lab_gpu_host_compare(
                args.lab_gpu_artifacts,
                args.output_dir,
                family_ids=families,
                seeds=seeds,
                pin=pin,
                allow_partial=True,
            )
        except MultiFamilyArtifactsMissingError as exc:
            print(f"BLOCKED: {exc}")
            return 2
    else:
        # Fixture path keeps default K=3 public seeds under explore pin.
        report = run_multi_family_official_compare(
            args.output_dir,
            family_ids=families,
            pin=explore_protocol_pin(),
            device_class="fixture",
            score_class=SCORE_CLASS_FIXTURE,
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"schema: {report['schema']} protocol_hash={report['protocol_hash'][:16]}…")
        print(
            f"score_class={report['score_class']} device_class={report['device_class']} "
            f"k_label={report.get('k_label')}"
        )
        print(
            f"pin stage={report['pin'].get('param_ladder_stage')} "
            f"param_cap={report['pin'].get('param_cap')} "
            f"tokenizer={report['pin'].get('tokenizer')} "
            f"token_budget={report['pin'].get('token_budget')}"
        )
        print("score_table:")
        for row in report.get("score_table") or []:
            print(
                f"  #{row['rank']} {row['family_id']}: "
                f"heldout={row['heldout_delta']} bpb={row['bpb']} "
                f"status={row['status']}"
            )
        blocked = report["ranking"].get("blocked_families") or {}
        if blocked:
            print("blocked:")
            for fam, reason in blocked.items():
                print(f"  {fam}: {reason}")
        print(
            f"winner={report['ranking'].get('winner')} "
            f"runner_up={report['ranking'].get('runner_up')}"
        )
        labels = report.get("labels") or {}
        print(
            f"provider_trust={labels.get('provider_trust')} "
            f"image_pin={labels.get('image_pin')} "
            f"architecture_agnostic={report['validity'].get('architecture_agnostic_path')}"
        )
        if report.get("report_path"):
            print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
