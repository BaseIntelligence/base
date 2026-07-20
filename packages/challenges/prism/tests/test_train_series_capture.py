"""Challenge-owned train series capture + authority (VAL-TELE-002..006).

Extends the online-loss harness path for MAX train telemetry:
- CE / running bpb series (VAL-TELE-002)
- tokens_seen + wall_s per point (VAL-TELE-003)
- grad_norm series via challenge hooks (VAL-TELE-004)
- clip_event counts (VAL-TELE-005)
- miner self-report ignored / series hash into challenge path (VAL-TELE-006)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from test_prism_harness_online_loss import ARCH_LM, TRAIN_LEARN, _read_manifest, _run_runner

from prism_challenge.evaluator.schemas import (
    TRAIN_SERIES_V1_FILENAME,
    TRAIN_SERIES_V1_SCHEMA,
)
from prism_challenge.evaluator.train_series import (
    build_train_series_v1,
    load_challenge_series,
    serialize_train_series_v1,
    series_has_required_axes,
    series_is_challenge_owned,
    train_series_sha256,
    write_train_series_artifact,
)

# Honest train that also plants a forged miner_series / forged dashboards for authority checks.
TRAIN_WITH_MINER_FAKE_SERIES = """
import json
import pathlib

import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    fake = {
        "schema": "prism_train_series.v1",
        "submission_id": "miner-forged",
        "run_id": "miner-forged",
        "authority": "miner",
        "x_axis": "batch_index",
        "points": [
            {
                "i": 0,
                "tokens_seen": 1,
                "covered_bytes": 1.0,
                "train_ce_nats": 0.0001,
                "running_bpb": 0.0001,
                "wall_s": 0.0,
                "grad_norm": 0.0,
                "clip_event": False,
                "nan_inf": False,
            }
        ],
        "aggregates": {"n_points": 1, "clip_events": 0},
        "miner_reported_ignored": False,
    }
    pathlib.Path(ctx.artifacts_dir, "prism_train_series.v1.json").write_text(
        json.dumps(fake), encoding="utf-8"
    )
    pathlib.Path(ctx.artifacts_dir, "miner_dashboard.json").write_text(
        json.dumps({"grad_norm": [0.0, 0.0], "train_loss": [0.0001]}), encoding="utf-8"
    )
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad(set_to_none=True)
        logits = model(batch.tokens)
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
"""


def _load_series(artifacts: Path) -> dict:
    path = artifacts / TRAIN_SERIES_V1_FILENAME
    assert path.is_file(), f"expected challenge series at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_train_series_ce_bpb_challenge_owned(tmp_path) -> None:
    """VAL-TELE-002: challenge captures non-empty CE / running bpb series on scored train path."""
    proc, artifacts = _run_runner(
        tmp_path, run_name="tele002", arch_code=ARCH_LM, train_code=TRAIN_LEARN, step_budget=8
    )
    assert proc.returncode == 0, proc.stderr
    series = _load_series(artifacts)
    assert series_is_challenge_owned(series)
    assert series["schema"] == TRAIN_SERIES_V1_SCHEMA
    assert series["authority"] == "challenge"
    points = series["points"]
    assert len(points) >= 2
    for point in points:
        assert isinstance(point["train_ce_nats"], (int, float))
        assert math.isfinite(float(point["train_ce_nats"]))
        # running_bpb derived from cumulative instrument totals
        assert point["running_bpb"] is None or math.isfinite(float(point["running_bpb"]))
    # online_loss co-presence with series (same instrument cadence)
    manifest = _read_manifest(artifacts)
    assert len(manifest["metrics"]["online_loss"]) == len(points)


def test_train_series_tokens_and_wall(tmp_path) -> None:
    """VAL-TELE-003: each series point includes tokens_seen and wall_s (challenge-owned)."""
    proc, artifacts = _run_runner(
        tmp_path, run_name="tele003", arch_code=ARCH_LM, train_code=TRAIN_LEARN, step_budget=6
    )
    assert proc.returncode == 0, proc.stderr
    series = _load_series(artifacts)
    points = series["points"]
    assert points
    prev_tokens = 0
    prev_wall = -1.0
    for point in points:
        assert "tokens_seen" in point
        assert "wall_s" in point
        tokens = int(point["tokens_seen"])
        wall = float(point["wall_s"])
        assert tokens >= prev_tokens
        assert wall >= 0.0
        assert wall >= prev_wall - 1e-9
        prev_tokens = tokens
        prev_wall = wall
    # final cumulative aligns with verified tokens_seen on the manifest
    manifest = _read_manifest(artifacts)
    assert points[-1]["tokens_seen"] == manifest["metrics"]["tokens_seen"]


def test_train_series_grad_norm_populated(tmp_path) -> None:
    """VAL-TELE-004: challenge instrumentation records grad_norm on scored train fixtures."""
    proc, artifacts = _run_runner(
        tmp_path, run_name="tele004", arch_code=ARCH_LM, train_code=TRAIN_LEARN, step_budget=8
    )
    assert proc.returncode == 0, proc.stderr
    series = _load_series(artifacts)
    points = series["points"]
    # At least one point must have a finite challenge-owned grad_norm (hooks see miner backward).
    finite_grads = [
        float(p["grad_norm"])
        for p in points
        if isinstance(p.get("grad_norm"), (int, float)) and math.isfinite(float(p["grad_norm"]))
    ]
    assert finite_grads, f"expected grad_norm values, got points={points}"
    assert all(g >= 0.0 for g in finite_grads)
    # Schema theatre: grad_norm key is present on every point (mandatory column).
    assert all("grad_norm" in p for p in points)
    assert series_has_required_axes(series)


def test_train_series_clip_events_recorded(tmp_path) -> None:
    """VAL-TELE-005: clip_event series / aggregate is challenge-owned when clip fires or not."""
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="tele005",
        arch_code=ARCH_LM,
        train_code=TRAIN_WITH_MINER_FAKE_SERIES,
        step_budget=8,
    )
    assert proc.returncode == 0, proc.stderr
    series = _load_series(artifacts)
    points = series["points"]
    assert all("clip_event" in p for p in points)
    # clip_event is bool|null (challenge instrumentation), never dry-run absence of the field
    for p in points:
        assert p["clip_event"] is None or isinstance(p["clip_event"], bool)
    # Aggregate total matches True count
    clip_count = sum(1 for p in points if p.get("clip_event") is True)
    assert series["aggregates"]["clip_events"] == clip_count
    manifest = _read_manifest(artifacts)
    assert manifest["metrics"].get("clip_events") == clip_count


def test_miner_self_report_series_ignored(tmp_path) -> None:
    """VAL-TELE-006: miner-forged series never certifies grade; challenge re-authors + hashes."""
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="tele006",
        arch_code=ARCH_LM,
        train_code=TRAIN_WITH_MINER_FAKE_SERIES,
        step_budget=6,
        submission_id="sub-authority",
    )
    assert proc.returncode == 0, proc.stderr
    series = _load_series(artifacts)
    # Challenge re-author overwrote the miner plant
    assert series["authority"] == "challenge"
    assert series["submission_id"] == "sub-authority"
    assert series["miner_reported_ignored"] is True
    assert series["run_id"].startswith("prism-reexec-")
    # Digests on disk match the pointer in the challenge manifest (proof path participation)
    manifest = _read_manifest(artifacts)
    metrics = manifest["metrics"]
    assert metrics["train_series_schema"] == TRAIN_SERIES_V1_SCHEMA
    assert metrics["train_series_path"] == TRAIN_SERIES_V1_FILENAME
    on_disk_digest = train_series_sha256((artifacts / TRAIN_SERIES_V1_FILENAME).read_bytes())
    assert metrics["train_series_sha256"] == on_disk_digest
    assert manifest["artifacts"]["train_series_sha256"] == on_disk_digest
    # Worker-plane path embeds the full challenge series so finalize need not open artifacts.
    embedded = metrics.get("train_series")
    assert isinstance(embedded, dict)
    assert series_is_challenge_owned(embedded)
    assert train_series_sha256(embedded) == on_disk_digest
    assert embedded["points"] == series["points"]
    assert manifest["score"]["miner_reported_ignored"] is True
    assert manifest["miner_reported_ignored"] is True
    # Miner dashboard remains on disk but can never authorize
    dashboard = json.loads((artifacts / "miner_dashboard.json").read_text(encoding="utf-8"))
    assert dashboard["train_loss"][0] == 0.0001
    # Challenge CE series is nothing like the forgeries
    assert all(float(p["train_ce_nats"]) > 0.01 for p in series["points"])
    # Helper rejects the miner-authority document
    miner_doc = {
        "schema": TRAIN_SERIES_V1_SCHEMA,
        "authority": "miner",
        "points": series["points"],
        "miner_reported_ignored": False,
    }
    assert series_is_challenge_owned(miner_doc) is False


def test_build_train_series_helper_and_axes() -> None:
    """Unit helper builds schema-valid series with required residual aggregates."""
    points = [
        {
            "i": 0,
            "tokens_seen": 10,
            "covered_bytes": 40.0,
            "train_ce_nats": 3.0,
            "running_bpb": 2.0,
            "wall_s": 0.1,
            "grad_norm": 0.5,
            "clip_event": False,
            "nan_inf": False,
        },
        {
            "i": 1,
            "tokens_seen": 20,
            "covered_bytes": 80.0,
            "train_ce_nats": 2.5,
            "running_bpb": 1.8,
            "wall_s": 0.25,
            "grad_norm": 1.5,
            "clip_event": True,
            "nan_inf": False,
        },
    ]
    series = build_train_series_v1(
        submission_id="sub-u",
        run_id="prism-reexec-sub-u",
        points=points,
        token_budget=1000,
        nan_inf_batches=0,
    )
    assert series_is_challenge_owned(series)
    assert series_has_required_axes(series)
    assert series["aggregates"]["clip_events"] == 1
    assert series["aggregates"]["n_points"] == 2
    assert series["schema"] == TRAIN_SERIES_V1_SCHEMA


def test_load_challenge_series_rejects_wrong_digest(tmp_path: Path) -> None:
    points = [
        {
            "i": 0,
            "tokens_seen": 4,
            "covered_bytes": 8.0,
            "train_ce_nats": 2.0,
            "running_bpb": 1.0,
            "wall_s": 0.01,
            "grad_norm": 0.1,
            "clip_event": False,
            "nan_inf": False,
        }
    ]
    series = build_train_series_v1(
        submission_id="s", run_id="r", points=points, token_budget=None, nan_inf_batches=0
    )
    path, digest = write_train_series_artifact(tmp_path, series)
    assert path.is_file()
    # Mapping digest and on-disk digest share the compact canonicalize form.
    assert train_series_sha256(series) == digest
    assert train_series_sha256(path.read_bytes()) == digest
    assert serialize_train_series_v1(series) == path.read_bytes()
    assert load_challenge_series(tmp_path, expected_sha256=digest) is not None
    assert load_challenge_series(tmp_path, expected_sha256="0" * 64) is None
