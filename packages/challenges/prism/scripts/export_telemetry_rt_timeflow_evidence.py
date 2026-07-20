#!/usr/bin/env python3
"""Export dual-family / dual-run train time-flow fixture evidence (VAL-TELE-011).

Writes challenge-owned ``prism_train_series.v1`` documents with loss + grad_norm +
clip to a destination directory (default: mission ``evidence/telemetry-rt/``).

Does **not**:
* claim a Prism TEE product path
* call live Swarm or set_weights
* require Lium / NVIDIA (CPU fixture synthetic series)

Usage::

    uv run python scripts/export_telemetry_rt_timeflow_evidence.py \\
        --out /root/.factory/missions/<id>/evidence/telemetry-rt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow `uv run python scripts/...` from prism root without install path gymnastics.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from prism_challenge.evaluator.official_comparison import (  # noqa: E402
    ProtocolPin,
    apply_train_series_requirement_to_grade,
    densify_stability_from_train_series,
    evaluate_train_series_for_official_grade,
)
from prism_challenge.evaluator.schemas import TRAIN_SERIES_V1_SCHEMA  # noqa: E402
from prism_challenge.evaluator.train_series import (  # noqa: E402
    densify_sample_eff_from_train_series,
    make_fixture_series,
    write_train_series_artifact,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_dual_timeflow_bundle() -> dict[str, Any]:
    """Two architecture series (transformer + mamba) with required axes."""
    transformer = make_fixture_series(
        submission_id="tele-rt-transformer-fixture",
        run_id="prism-reexec-tele-rt-transformer",
        family="transformer",
        n_points=32,
        start_ce=4.5,
        end_ce=1.7,
        tokens_per_step=512,
        wall_per_step=0.04,
        grad_start=3.2,
        grad_end=0.45,
        clip_every=4,
        seed_offset=0.0,
    )
    mamba = make_fixture_series(
        submission_id="tele-rt-mamba-fixture",
        run_id="prism-reexec-tele-rt-mamba",
        family="mamba",
        n_points=32,
        start_ce=4.0,
        end_ce=1.3,
        tokens_per_step=512,
        wall_per_step=0.035,
        grad_start=2.4,
        grad_end=0.35,
        clip_every=3,
        seed_offset=1.5,
    )
    # Second run of transformer (dual-run alternate) for two-run residual view.
    transformer_run2 = make_fixture_series(
        submission_id="tele-rt-transformer-fixture-run2",
        run_id="prism-reexec-tele-rt-transformer-run2",
        family="transformer",
        n_points=32,
        start_ce=4.4,
        end_ce=1.65,
        tokens_per_step=512,
        wall_per_step=0.042,
        grad_start=3.0,
        grad_end=0.5,
        clip_every=5,
        seed_offset=0.3,
    )
    return {
        "transformer": transformer,
        "mamba": mamba,
        "transformer_run2": transformer_run2,
    }


def export_evidence(out_dir: Path) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_dual_timeflow_bundle()
    series_dir = out_dir / "series"
    series_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Any] = {}
    for name, series in bundle.items():
        side_dir = series_dir / name
        side_dir.mkdir(parents=True, exist_ok=True)
        path, digest = write_train_series_artifact(side_dir, series)
        # Also copy a flat-named specimen for operators reading under telemetry-rt/.
        flat = series_dir / f"{name}.{TRAIN_SERIES_V1_SCHEMA}.json"
        flat.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        residual = densify_stability_from_train_series(series)
        sample = densify_sample_eff_from_train_series(series, mark_tokens=(512, 2048, 8192, 16384))
        gate = evaluate_train_series_for_official_grade(
            series, require_train_series=True, expected_sha256=digest
        )
        # Red case specimen metadata: missing series under require flag.
        require_pin = ProtocolPin(require_train_series=True)
        missing_grade = apply_train_series_requirement_to_grade(
            series=None,
            pin=require_pin,
        )
        written[name] = {
            "path": str(path.relative_to(out_dir)),
            "flat_path": str(flat.relative_to(out_dir)),
            "sha256": digest,
            "n_points": series["aggregates"]["n_points"],
            "clip_events": series["aggregates"]["clip_events"],
            "grad_spike_rate": series["aggregates"]["grad_spike_rate"],
            "authority": series["authority"],
            "schema": series["schema"],
            "stability_residual": residual,
            "sample_eff_residual": sample,
            "series_gate_require_on": gate,
            "missing_series_grade_under_require": {
                "grade_valid": missing_grade["grade_valid"],
                "reasons": missing_grade["reasons"],
                "silent_pass": missing_grade["silent_pass"],
            },
        }

    # Compact dual export for eyedrop-style charts (loss + grad + clip vs tokens).
    chart_rows: list[dict[str, Any]] = []
    for name, series in bundle.items():
        for point in series["points"]:
            chart_rows.append(
                {
                    "series": name,
                    "family": series["aggregates"].get("family"),
                    "i": point["i"],
                    "tokens_seen": point["tokens_seen"],
                    "train_ce_nats": point["train_ce_nats"],
                    "running_bpb": point["running_bpb"],
                    "wall_s": point["wall_s"],
                    "grad_norm": point["grad_norm"],
                    "clip_event": point["clip_event"],
                }
            )
    chart_path = out_dir / "timeflow_chart_rows.json"
    _write_json(chart_path, {"rows": chart_rows, "schema": TRAIN_SERIES_V1_SCHEMA})

    pin_on = ProtocolPin(require_train_series=True).as_dict()
    pin_off = ProtocolPin(require_train_series=False).as_dict()

    index = {
        "schema": "prism_telemetry_rt_timeflow_evidence.v1",
        "exported_at": datetime.now(UTC).isoformat(),
        "source": "fixture",
        "device_class": "fixture",
        "score_class": "fixture",
        "labels": {
            "provider_trust": "PROVIDER_TRUST",
            "image_pin": "IMAGE_PIN",
            "score_class": "fixture",
            "prism_tee_product": False,
        },
        "lium_used": False,
        "swarm_mutated": False,
        "require_train_series_pin_on": pin_on,
        "require_train_series_pin_off": pin_off,
        "series": written,
        "chart_rows_path": str(chart_path.relative_to(out_dir)),
        "assertions": {
            "VAL-TELE-011": ("dual-family + dual-run series with loss+grad+clip exported"),
            "VAL-TELE-009": ("missing series under require_train_series fails grade_valid"),
        },
        "non_claims": [
            "prism_tee_product",
            "live Swarm mutation",
            "set_weights",
            "series sole-ranking over heldout/bpb",
        ],
    }
    # Explicit red-case evidence for VAL-TELE-009 fail-closed semantics.
    red = apply_train_series_requirement_to_grade(
        series=None, pin=ProtocolPin(require_train_series=True)
    )
    empty = evaluate_train_series_for_official_grade(
        {
            "schema": TRAIN_SERIES_V1_SCHEMA,
            "authority": "challenge",
            "points": [],
            "miner_reported_ignored": True,
        },
        require_train_series=True,
    )
    corrupt_doc = {
        "schema": "bad",
        "authority": "challenge",
        "points": [{"i": 0}],
        "miner_reported_ignored": True,
    }
    corrupt = evaluate_train_series_for_official_grade(
        corrupt_doc,
        require_train_series=True,
    )
    red_path = out_dir / "VAL-TELE-009-failclosed-red-cases.json"
    _write_json(
        red_path,
        {
            "missing": red,
            "empty": empty,
            "corrupt": corrupt,
            "note": (
                "Official grade must not silent PASS when require_train_series and series invalid"
            ),
        },
    )
    index["val_tele_009_red_cases"] = str(red_path.relative_to(out_dir))
    index_path = out_dir / "timeflow_evidence_index.json"
    _write_json(index_path, index)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/root/.factory/missions/a43a16a7-2230-4853-ba8a-a6bfe993a90f/evidence/telemetry-rt"
        ),
        help="Destination directory for dual time-flow evidence",
    )
    args = parser.parse_args(argv)
    index = export_evidence(args.out)
    print(json.dumps({"ok": True, "out": str(args.out), "series": list(index["series"].keys())}))
    # Hard invariant: good dual series present; RED cases fail grade; good path is grade_valid.
    assert len(index["series"]) >= 2
    for name, meta in index["series"].items():
        gate = meta["series_gate_require_on"]
        assert gate["grade_valid"] is True, (
            f"good series {name} must grade_valid under require_train_series "
            f"(got reasons={gate.get('reasons')})"
        )
        assert gate["ok"] is True
        assert "train_series_digest_mismatch" not in (gate.get("reasons") or [])
    red_path = Path(args.out) / index["val_tele_009_red_cases"]
    red = json.loads(red_path.read_text(encoding="utf-8"))
    assert red["missing"]["grade_valid"] is False
    assert red["empty"]["grade_valid"] is False
    assert red["corrupt"]["grade_valid"] is False
    assert index["labels"]["prism_tee_product"] is False
    assert index["labels"]["provider_trust"] == "PROVIDER_TRUST"
    assert index["lium_used"] is False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
