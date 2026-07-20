"""API / operator time-flow for prism_train_series.v1 (VAL-TELE-007, VAL-TELE-008).

Route-level cases live with the curve suite; these unit tests pin the downsample
helper and schema contract used by GET /v1/submissions/{id}/curve.
"""

from __future__ import annotations

from prism_challenge.evaluator.schemas import TRAIN_SERIES_V1_SCHEMA
from prism_challenge.evaluator.train_series import (
    build_train_series_v1,
    downsample_train_series_for_api,
    sanitize_point_for_api,
    series_point,
)
from prism_challenge.models import TrainSeriesV1Response


def _points(n: int = 5) -> list[dict]:
    return [
        series_point(
            i=i,
            tokens_seen=(i + 1) * 8,
            covered_bytes=float((i + 1) * 16),
            train_ce_nats=3.0 - 0.05 * i,
            running_bpb=4.3 - 0.05 * i,
            wall_s=0.25 * i,
            grad_norm=0.5 + 0.01 * i,
            clip_event=(i % 3 == 0),
        )
        for i in range(n)
    ]


def test_downsample_preserves_schema_and_authority() -> None:
    series = build_train_series_v1(
        submission_id="s1",
        run_id="r1",
        points=_points(10),
        token_budget=80,
    )
    out = downsample_train_series_for_api(series, max_points=500)
    assert out is not None
    assert out["schema"] == TRAIN_SERIES_V1_SCHEMA
    assert out["authority"] == "challenge"
    assert out["miner_reported_ignored"] is True
    assert out["downsampled"] is False
    assert out["points_total"] == 10
    assert len(out["points"]) == 10
    assert out["points"][0]["grad_norm"] is not None
    assert any(p.get("clip_event") is True for p in out["points"])


def test_downsample_caps_and_keeps_endpoints() -> None:
    series = build_train_series_v1(
        submission_id="s1",
        run_id="r1",
        points=_points(900),
    )
    out = downsample_train_series_for_api(series, max_points=100)
    assert out is not None
    assert out["downsampled"] is True
    assert out["points_total"] == 900
    assert len(out["points"]) == 100
    assert out["points"][0]["i"] == 0
    assert out["points"][-1]["i"] == 899


def test_downsample_rejects_miner_and_empty() -> None:
    challenge = build_train_series_v1(submission_id="s1", run_id="r1", points=_points(3))
    miner = dict(challenge)
    miner["authority"] = "miner"
    miner["miner_reported_ignored"] = False
    assert downsample_train_series_for_api(miner) is None
    empty = dict(challenge)
    empty["points"] = []
    assert downsample_train_series_for_api(empty) is None
    assert downsample_train_series_for_api(None) is None


def test_sanitize_strips_unknown_keys() -> None:
    dirty = {
        "i": 0,
        "tokens_seen": 10,
        "train_ce_nats": 1.5,
        "wall_s": 0.1,
        "grad_norm": 1.0,
        "clip_event": False,
        "api_token": "secret",
        "wallet_mnemonic": "abandon",
    }
    clean = sanitize_point_for_api(dirty)
    assert clean is not None
    assert "api_token" not in clean
    assert "wallet_mnemonic" not in clean
    assert clean["grad_norm"] == 1.0


def test_response_model_wire_schema_alias() -> None:
    series = build_train_series_v1(submission_id="s1", run_id="r1", points=_points(2))
    payload = downsample_train_series_for_api(series)
    assert payload is not None
    model = TrainSeriesV1Response.model_validate(payload)
    dumped = model.model_dump(by_alias=True)
    assert dumped["schema"] == TRAIN_SERIES_V1_SCHEMA
    assert "schema_name" not in dumped
    assert dumped["points"][0]["grad_norm"] is not None
