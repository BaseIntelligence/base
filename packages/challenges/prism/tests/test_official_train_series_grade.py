"""Official grade fail-closed when require_train_series and series missing (VAL-TELE-009).

Also asserts residual-only densify (VAL-TELE-010 companion): series never sole-rank
over heldout/bpb axes.
"""

from __future__ import annotations

import math

from prism_challenge.evaluator.official_comparison import (
    OfficialScoreRecord,
    ProtocolPin,
    apply_train_series_requirement_to_grade,
    compare_official,
    densify_stability_from_train_series,
    evaluate_train_series_for_official_grade,
    evaluate_validity_gates,
    official_rank_key,
)
from prism_challenge.evaluator.schemas import TRAIN_SERIES_V1_SCHEMA
from prism_challenge.evaluator.train_series import (
    build_train_series_v1,
    densify_sample_eff_from_train_series,
    make_fixture_series,
    serialize_train_series_v1,
    series_point,
    train_series_sha256,
    write_train_series_artifact,
)


def _good_points(n: int = 4) -> list[dict]:
    out = []
    for i in range(n):
        out.append(
            series_point(
                i=i,
                tokens_seen=512 * (i + 1),
                covered_bytes=float(512 * (i + 1) * 4),
                train_ce_nats=3.0 - 0.2 * i,
                running_bpb=2.5 - 0.1 * i,
                wall_s=0.1 * (i + 1),
                grad_norm=1.0 / (i + 1),
                clip_event=(i % 2 == 0),
                nan_inf=False,
            )
        )
    return out


def _good_series(**kwargs) -> dict:
    defaults = {
        "submission_id": "sub-a",
        "run_id": "prism-reexec-sub-a",
        "points": _good_points(),
        "token_budget": 4096,
        "nan_inf_batches": 0,
    }
    defaults.update(kwargs)
    return build_train_series_v1(**defaults)


def _clean_record(
    label: str = "side-a",
    *,
    heldout: float = 0.9,
    bpb: float = 1.5,
) -> OfficialScoreRecord:
    return OfficialScoreRecord(
        label=label,
        bpb=bpb,
        heldout_delta=heldout,
        valid=True,
        step0_anomaly=False,
        challenge_authored=True,
    )


def test_protocol_pin_exposes_require_train_series_default_false() -> None:
    pin = ProtocolPin()
    assert pin.require_train_series is False
    payload = pin.as_dict()
    assert payload["require_train_series"] is False


def test_require_off_series_absent_does_not_fail_grade() -> None:
    """When pin does not require series, missing series is not an Official invalidation."""
    gate = evaluate_train_series_for_official_grade(None, require_train_series=False)
    assert gate["required"] is False
    assert gate["ok"] is True
    assert gate["grade_valid"] is True

    grade = apply_train_series_requirement_to_grade(
        record=_clean_record(),
        series=None,
        require_train_series=False,
    )
    assert grade["grade_valid"] is True
    assert grade["require_train_series"] is False


def test_require_on_missing_series_fail_closed() -> None:
    """VAL-TELE-009: absence → grade invalid, not silent PASS."""
    pin = ProtocolPin(require_train_series=True)
    gate = evaluate_train_series_for_official_grade(None, require_train_series=True)
    assert gate["required"] is True
    assert gate["ok"] is False
    assert gate["grade_valid"] is False
    assert "train_series_missing" in gate["reasons"]
    assert gate["series_may_sole_rank"] is False

    grade = apply_train_series_requirement_to_grade(
        record=_clean_record(),
        series=None,
        pin=pin,
    )
    assert grade["grade_valid"] is False
    assert grade["official_rank_eligible"] is False
    assert grade["silent_pass"] is False
    assert "train_series_missing" in grade["reasons"]


def test_require_on_empty_series_fail_closed() -> None:
    empty = {
        "schema": TRAIN_SERIES_V1_SCHEMA,
        "submission_id": "s",
        "run_id": "r",
        "authority": "challenge",
        "points": [],
        "aggregates": {"n_points": 0},
        "miner_reported_ignored": True,
    }
    gate = evaluate_train_series_for_official_grade(empty, require_train_series=True)
    assert gate["ok"] is False
    assert gate["grade_valid"] is False
    assert "train_series_empty" in gate["reasons"]


def test_require_on_corrupt_series_fail_closed() -> None:
    corrupt = {
        "schema": "not_a_real_schema",
        "authority": "challenge",
        "points": _good_points(),
        "miner_reported_ignored": True,
    }
    gate = evaluate_train_series_for_official_grade(corrupt, require_train_series=True)
    assert gate["ok"] is False
    assert gate["grade_valid"] is False
    assert any(
        r in gate["reasons"] for r in ("train_series_corrupt", "train_series_not_challenge_owned")
    )


def test_require_on_nonfinite_series_fail_closed() -> None:
    pts = _good_points()
    pts[1]["train_ce_nats"] = float("nan")
    series = build_train_series_v1(
        submission_id="s", run_id="r", points=pts, token_budget=100, nan_inf_batches=1
    )
    gate = evaluate_train_series_for_official_grade(series, require_train_series=True)
    assert gate["ok"] is False
    assert "train_series_nonfinite" in gate["reasons"]
    assert gate["grade_valid"] is False


def test_require_on_miner_only_never_unblocks() -> None:
    """Miner dashboard while challenge series missing still fail-closes."""
    miner = {
        "schema": TRAIN_SERIES_V1_SCHEMA,
        "authority": "miner",
        "points": _good_points(),
        "miner_reported_ignored": False,
    }
    gate = evaluate_train_series_for_official_grade(
        None, require_train_series=True, miner_series=miner
    )
    assert gate["ok"] is False
    assert "train_series_missing" in gate["reasons"]
    assert gate["miner_series_ignored"] is True

    # Passing miner-only as the series still fails (not challenge-owned).
    gate2 = evaluate_train_series_for_official_grade(miner, require_train_series=True)
    assert gate2["ok"] is False
    assert "train_series_miner_only" in gate2["reasons"] or (
        "train_series_not_challenge_owned" in gate2["reasons"]
    )


def test_require_on_good_series_allows_grade() -> None:
    series = _good_series()
    digest = train_series_sha256(series)
    gate = evaluate_train_series_for_official_grade(
        series, require_train_series=True, expected_sha256=digest
    )
    assert gate["ok"] is True
    assert gate["grade_valid"] is True
    assert gate["scoreable"] is True
    assert gate["reasons"] == []

    grade = apply_train_series_requirement_to_grade(
        record=_clean_record(),
        series=series,
        pin=ProtocolPin(require_train_series=True),
        expected_sha256=digest,
    )
    assert grade["grade_valid"] is True
    assert grade["official_rank_eligible"] is True


def test_write_artifact_digest_matches_mapping_hash_under_require(tmp_path) -> None:
    """BLOCKING fix: artifact write digest and Mapping re-hash share one serialize form.

    Prior bug: write used pretty indent=2 while train_series_sha256(Mapping) used
    compact separators → evaluate_train_series_for_official_grade falsedigested
    train_series_digest_mismatch on good series.
    """
    series = _good_series()
    path, disk_digest = write_train_series_artifact(tmp_path, series)
    raw = path.read_bytes()
    assert train_series_sha256(raw) == disk_digest
    assert train_series_sha256(series) == disk_digest
    assert serialize_train_series_v1(series) == raw
    # Pretty re-encode would yield a different digest — guard the identity LOCk.
    pretty = __import__("json").dumps(dict(series), sort_keys=True, indent=2).encode("utf-8")
    assert train_series_sha256(pretty) != disk_digest

    gate = evaluate_train_series_for_official_grade(
        series, require_train_series=True, expected_sha256=disk_digest
    )
    assert gate["ok"] is True
    assert gate["grade_valid"] is True
    assert gate["reasons"] == []

    grade = apply_train_series_requirement_to_grade(
        record=_clean_record(),
        series=series,
        pin=ProtocolPin(require_train_series=True),
        expected_sha256=disk_digest,
    )
    assert grade["grade_valid"] is True
    assert grade["official_rank_eligible"] is True


def test_digest_mismatch_fail_closed() -> None:
    series = _good_series()
    gate = evaluate_train_series_for_official_grade(
        series, require_train_series=True, expected_sha256="0" * 64
    )
    assert gate["ok"] is False
    assert "train_series_digest_mismatch" in gate["reasons"]


def test_missing_series_does_not_silent_pass_compare_eligibility() -> None:
    """Even with strong heldout/bpb, required missing series blocks Official grade."""
    a = _clean_record("transformer", heldout=1.0, bpb=1.2)
    b = _clean_record("mamba", heldout=0.2, bpb=1.0)
    # compare_official still ranks challenge metrics, but grade block is invalid for pin
    result = compare_official(a, b)
    assert result.winner == "a"

    grade_a = apply_train_series_requirement_to_grade(
        record=a,
        series=None,
        require_train_series=True,
    )
    grade_b = apply_train_series_requirement_to_grade(
        record=b,
        series=_good_series(submission_id="b", run_id="prism-reexec-b"),
        require_train_series=True,
    )
    assert grade_a["grade_valid"] is False
    assert grade_b["grade_valid"] is True
    # Pair Official scientific report would combine:
    pair_ok = grade_a["grade_valid"] and grade_b["grade_valid"]
    assert pair_ok is False
    assert grade_a["silent_pass"] is False


def test_series_residual_never_sole_ranks_official_key() -> None:
    """VAL-TELE-010 companion: rank key still heldout/bpb; densify is residual only."""
    strong_holdout_weak_series_hints = OfficialScoreRecord(
        label="holdout-winner",
        bpb=2.0,
        heldout_delta=1.5,
        valid=True,
        grad_spike_rate=0.9,  # "bad" stability residual
        sample_eff_auc=0.1,
    )
    weak_holdout_strong_series_hints = OfficialScoreRecord(
        label="series-pretty",
        bpb=1.0,
        heldout_delta=0.1,
        valid=True,
        grad_spike_rate=0.0,
        sample_eff_auc=0.99,
    )
    # Official primary heldout still prefer strong_holdout regardless of residual densify.
    key_strong = official_rank_key(strong_holdout_weak_series_hints)
    key_weak = official_rank_key(weak_holdout_strong_series_hints)
    assert key_strong < key_weak  # ascending: smaller key ranks better

    series = make_fixture_series(
        submission_id="s",
        run_id="r",
        family="transformer",
        n_points=12,
        start_ce=5.0,
        end_ce=1.0,
    )
    residual = densify_stability_from_train_series(series)
    assert residual["ok"] is True
    assert residual["series_may_sole_rank"] is False
    assert residual["series_residual_only"] is True

    sample = densify_sample_eff_from_train_series(series, mark_tokens=(512, 1024, 2048))
    assert sample["ok"] is True
    assert sample["series_may_sole_rank"] is False
    # Official compare never consults residual densify marks as sole primary
    cmp = compare_official(strong_holdout_weak_series_hints, weak_holdout_strong_series_hints)
    assert cmp.winner == "a"
    assert cmp.reason == "primary_heldout"


def test_validity_gates_independent_of_series_still_fail_grade_on_series() -> None:
    record = _clean_record()
    v = evaluate_validity_gates(record)
    assert v.ok is True
    grade = apply_train_series_requirement_to_grade(
        record=record,
        series=None,
        require_train_series=True,
        validity=v,
    )
    assert grade["validity_ok"] is True
    assert grade["grade_valid"] is False  # series alone fails closed


def test_fixture_dual_family_series_shape() -> None:
    """Helper used by VAL-TELE-011 evidence has loss+grad+clip on both families."""
    t = make_fixture_series(
        submission_id="tele-xfmr",
        run_id="prism-reexec-tele-xfmr",
        family="transformer",
        n_points=16,
        start_ce=4.2,
        end_ce=1.8,
        grad_start=3.0,
        grad_end=0.5,
        clip_every=4,
    )
    m = make_fixture_series(
        submission_id="tele-mamba",
        run_id="prism-reexec-tele-mamba",
        family="mamba",
        n_points=16,
        start_ce=3.8,
        end_ce=1.2,
        grad_start=2.0,
        grad_end=0.3,
        clip_every=3,
        seed_offset=1.0,
    )
    for series in (t, m):
        assert series["schema"] == TRAIN_SERIES_V1_SCHEMA
        assert series["authority"] == "challenge"
        assert series["miner_reported_ignored"] is True
        points = series["points"]
        assert len(points) == 16
        assert all("train_ce_nats" in p and "grad_norm" in p and "clip_event" in p for p in points)
        assert all(math.isfinite(float(p["grad_norm"])) for p in points)
        assert series["aggregates"]["clip_events"] >= 1
