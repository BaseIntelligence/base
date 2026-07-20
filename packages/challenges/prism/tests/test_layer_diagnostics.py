"""Layer / activation diagnostics densify helpers (VAL-EVALC-003)."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from prism_challenge.evaluator.layer_diagnostics import (
    LAYER_DIAGNOSTICS_SCHEMA,
    build_layer_diagnostics_v1,
    grad_norm_aggregates_from_series,
    load_weight_l2_from_trained_state,
    optional_activation_stats_one_batch,
    weight_l2_norms_from_state,
)
from prism_challenge.evaluator.train_series import build_train_series_v1, series_point


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(32, 8)
        self.block = nn.Linear(8, 8)
        self.head = nn.Linear(8, 32)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        x = self.block(x)
        return self.head(x)


def test_weight_l2_norms_from_plain_state_dict() -> None:
    model = _TinyLM()
    pack = weight_l2_norms_from_state(model.state_dict())
    assert pack["ok"] is True
    assert pack["n_tensors"] >= 3
    assert pack["global_l2"] is not None and pack["global_l2"] > 0
    assert "emb.weight" in pack["param_l2"]
    assert "emb.weight" in pack["layer_group_l2"] or "emb" in str(pack["layer_group_l2"])


def test_load_weight_l2_from_trained_state_file(tmp_path: Path) -> None:
    model = _TinyLM()
    path = tmp_path / "trained_state.pt"
    torch.save(model.state_dict(), path)
    pack = load_weight_l2_from_trained_state(path)
    assert pack["ok"] is True
    assert pack["n_parameters"] > 0
    missing = load_weight_l2_from_trained_state(tmp_path / "absent.pt")
    assert missing["ok"] is False
    assert missing["reason"] == "trained_state_missing"


def test_optional_activation_stats_one_batch_ok() -> None:
    model = _TinyLM()
    tokens = torch.randint(0, 32, (2, 6))
    out = optional_activation_stats_one_batch(model, tokens, max_modules=16)
    assert out["ok"] is True
    assert out["status"] == "ok"
    assert out["n_modules_ok"] and out["n_modules_ok"] >= 1
    # At least one ok module with finite mean/std
    ok_found = False
    for payload in out["modules"].values():
        if payload.get("status") == "ok":
            assert math.isfinite(float(payload["mean"]))
            assert math.isfinite(float(payload["std"]))
            ok_found = True
    assert ok_found


def test_optional_activation_blocked_on_bad_batch() -> None:
    model = _TinyLM()
    out = optional_activation_stats_one_batch(model, batch_tokens="not-a-tensor")
    assert out["ok"] is False
    assert out["status"] == "BLOCKED_with_reason"


def test_grad_norm_aggregates_from_series() -> None:
    points = [
        series_point(
            i=i,
            tokens_seen=(i + 1) * 10,
            covered_bytes=float((i + 1) * 40),
            train_ce_nats=2.0 - 0.1 * i,
            running_bpb=1.0 - 0.05 * i,
            wall_s=0.1 * (i + 1),
            grad_norm=1.0 + float(i),
            clip_event=(i % 2 == 0),
        )
        for i in range(6)
    ]
    # Force one spike
    points[-1]["grad_norm"] = 1000.0
    series = build_train_series_v1(
        submission_id="s",
        run_id="r",
        points=points,
        token_budget=60,
    )
    agg = grad_norm_aggregates_from_series(series)
    assert agg["ok"] is True
    assert agg["n_grad_points"] == 6
    assert agg["clip_events"] == 3
    assert agg["grad_norm_max"] == 1000.0
    assert agg["grad_spike_rate"] is not None and agg["grad_spike_rate"] > 0

    empty = grad_norm_aggregates_from_series(None)
    assert empty["ok"] is False


def test_build_layer_diagnostics_pack_schema() -> None:
    model = _TinyLM()
    weights = weight_l2_norms_from_state(model.state_dict())
    acts = optional_activation_stats_one_batch(model, torch.randint(0, 32, (1, 4)))
    packs = build_layer_diagnostics_v1(
        family_id="deeploop-tiny-1m",
        submission_id="lab",
        run_id="run",
        weight_norms=weights,
        activation=acts,
        grad_aggregates={"ok": False, "reason": "series_pending"},
        notes=["unit"],
    )
    assert packs["schema"] == LAYER_DIAGNOSTICS_SCHEMA
    assert packs["authority"] == "challenge"
    assert packs["weight_l2"]["ok"] is True
    assert packs["miner_reported_ignored"] is True
