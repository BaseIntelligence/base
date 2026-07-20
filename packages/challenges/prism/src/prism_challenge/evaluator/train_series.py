"""Challenge-owned train series helpers (``prism_train_series.v1`` telemetry-rt).

Authority model (docs/official-comparison.md §17, VAL-TELE-002..006):

* Series are authored by the Prism re-exec instrument (``_OnlineLossCapture``), not the miner.
* Required axes under full telemetry: train CE / running bpb, tokens_seen, wall time,
  ``grad_norm``, and ``clip_event`` / ``clip_events``.
* Miner-written dashboards / fake series files **never** authorize grade or unblock missing
  challenge series (``miner_reported_ignored: true``).
* Series participate in the proof path via a content hash pointer on the v2 manifest.
* Wall-clock series are observability-only and never feed ``final_score``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schemas import (
    TRAIN_SERIES_V1_FILENAME,
    TRAIN_SERIES_V1_SCHEMA,
)

TRAIN_SERIES_AUTHORITY = "challenge"

# One canonical on-disk / Mapping encode for digest identity (VAL-TELE + Official grade).
# Pretty (indent=2) vs compact must never diverge: artifact pointers use these exact bytes,
# and ``evaluate_train_series_for_official_grade(..., expected_sha256=...)`` re-hashes Mapping
# with the same form so good series no longer false-fail ``train_series_digest_mismatch``.
_TRAIN_SERIES_JSON_SEPARATORS = (",", ":")


def serialize_train_series_v1(payload: Mapping[str, Any]) -> bytes:
    """Canonical UTF-8 JSON bytes for ``prism_train_series.v1`` digests and side-cars.

    Always ``sort_keys=True`` with compact separators (no pretty indent). Callers that
    need identities commensurate with ``metrics.train_series_sha256`` / Official
    ``expected_sha256`` must use this path (or hash the resulting on-disk bytes unchanged).
    """
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=_TRAIN_SERIES_JSON_SEPARATORS,
        ensure_ascii=True,
    ).encode("utf-8")


def train_series_sha256(payload: Mapping[str, Any] | str | bytes) -> str:
    """Canonical sha256 of a train-series payload (stable JSON or raw on-disk bytes).

    * ``bytes`` / ``str``: hash the given material as-is (use for authenticating on-disk
      side-cars without re-encoding).
    * ``Mapping``: hash :func:`serialize_train_series_v1` so Mapping digests match
      compact artifact writes.
    """
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = serialize_train_series_v1(payload)
    return hashlib.sha256(raw).hexdigest()


def compute_running_bpb(*, sum_nll_nats: float, covered_bytes: float) -> float | None:
    """Running prequential bits-per-byte from cumulative instrument totals."""
    if covered_bytes <= 0 or not math.isfinite(sum_nll_nats):
        return None
    bits = sum_nll_nats / math.log(2.0)
    if not math.isfinite(bits):
        return None
    return bits / covered_bytes


def series_point(
    *,
    i: int,
    tokens_seen: int,
    covered_bytes: float,
    train_ce_nats: float,
    running_bpb: float | None,
    wall_s: float,
    grad_norm: float | None,
    clip_event: bool | None,
    nan_inf: bool = False,
    param_norm: float | None = None,
    lr: float | None = None,
) -> dict[str, Any]:
    """Build one challenge-owned series point (schema point field names)."""
    point: dict[str, Any] = {
        "i": int(i),
        "tokens_seen": int(tokens_seen),
        "covered_bytes": float(covered_bytes),
        "train_ce_nats": float(train_ce_nats),
        "running_bpb": running_bpb if running_bpb is None else float(running_bpb),
        "wall_s": float(wall_s),
        "grad_norm": None if grad_norm is None else float(grad_norm),
        "clip_event": None if clip_event is None else bool(clip_event),
        "param_norm": None if param_norm is None else float(param_norm),
        "lr": None if lr is None else float(lr),
        "nan_inf": bool(nan_inf),
    }
    return point


def build_train_series_v1(
    *,
    submission_id: str,
    run_id: str,
    points: Sequence[Mapping[str, Any]],
    token_budget: int | None = None,
    nan_inf_batches: int | None = None,
    extra_aggregates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a complete ``prism_train_series.v1`` document from challenge points."""
    cleaned: list[dict[str, Any]] = [dict(p) for p in points]
    walls = [float(p["wall_s"]) for p in cleaned if isinstance(p.get("wall_s"), int | float)]
    step_ms: list[float] = []
    prev = 0.0
    for wall in walls:
        delta = max(0.0, wall - prev)
        step_ms.append(delta * 1000.0)
        prev = wall
    clip_events = sum(1 for p in cleaned if p.get("clip_event") is True)
    grad_vals = [
        float(p["grad_norm"])
        for p in cleaned
        if isinstance(p.get("grad_norm"), int | float) and math.isfinite(float(p["grad_norm"]))
    ]
    # Spike rate: fraction of steps whose grad_norm exceeds 10x the median (stability residual).
    grad_spike_rate = 0.0
    if grad_vals:
        ordered = sorted(grad_vals)
        med = ordered[len(ordered) // 2]
        thr = max(med * 10.0, 1e-8)
        grad_spike_rate = sum(1 for g in grad_vals if g > thr) / float(len(grad_vals))
    aggregates: dict[str, Any] = {
        "n_points": len(cleaned),
        "mean_step_ms": (sum(step_ms) / len(step_ms)) if step_ms else 0.0,
        "p99_step_ms": (
            sorted(step_ms)[min(len(step_ms) - 1, int(math.ceil(0.99 * len(step_ms)) - 1))]
            if step_ms
            else 0.0
        ),
        "grad_spike_rate": grad_spike_rate,
        "nan_inf_batches": int(nan_inf_batches or 0),
        "clip_events": int(clip_events),
    }
    if extra_aggregates:
        aggregates.update(dict(extra_aggregates))
    return {
        "schema": TRAIN_SERIES_V1_SCHEMA,
        "submission_id": str(submission_id),
        "run_id": str(run_id),
        "authority": TRAIN_SERIES_AUTHORITY,
        "x_axis": "batch_index",
        "token_budget": token_budget,
        "points": cleaned,
        "aggregates": aggregates,
        "miner_reported_ignored": True,
    }


def series_is_challenge_owned(series: Mapping[str, Any] | None) -> bool:
    if not isinstance(series, Mapping):
        return False
    if series.get("schema") != TRAIN_SERIES_V1_SCHEMA:
        return False
    if series.get("authority") != TRAIN_SERIES_AUTHORITY:
        return False
    if series.get("miner_reported_ignored") is not True:
        return False
    points = series.get("points")
    return isinstance(points, list) and len(points) > 0


def series_has_required_axes(series: Mapping[str, Any] | None) -> bool:
    """True when every point has CE, tokens, wall and (when present) grad/clip columns exist."""
    if not series_is_challenge_owned(series):
        return False
    assert series is not None
    points = series["points"]
    for point in points:
        if not isinstance(point, Mapping):
            return False
        for key in ("i", "tokens_seen", "train_ce_nats", "wall_s"):
            if key not in point:
                return False
            if key != "i" and not isinstance(point[key], int | float):
                return False
        if "grad_norm" not in point or "clip_event" not in point:
            return False
    return True


def write_train_series_artifact(
    artifacts_dir: Path | str,
    series: Mapping[str, Any],
    *,
    filename: str = TRAIN_SERIES_V1_FILENAME,
) -> tuple[Path, str]:
    """Persist series JSON under artifacts; return path + sha256 of on-disk bytes.

    On-disk encoding is the canonical compact form (:func:`serialize_train_series_v1`),
    so ``train_series_sha256(series_mapping)`` equals the returned digest (and so Official
    grade ``expected_sha256`` checks against Mapping re-hash do not false-fail).
    """
    path = Path(artifacts_dir) / filename
    # Drop any miner-planted file first so a hostile forged series never survives.
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
    raw = serialize_train_series_v1(series)
    path.write_bytes(raw)
    digest = train_series_sha256(raw)
    return path, digest


def reject_miner_authored_series(
    artifacts_dir: Path | str,
    *,
    challenge_digest: str | None,
) -> None:
    """Remove leftover miner-only series files that do not match the challenge digest.

    The challenge re-authors its series after train(ctx). Any residual miner-planted
    train series that doesn't hash to the challenge digest is deleted.
    """
    root = Path(artifacts_dir)
    for name in (TRAIN_SERIES_V1_FILENAME, "prism_train_series.v1.jsonl"):
        candidate = root / name
        if not candidate.is_file():
            continue
        try:
            digest = train_series_sha256(candidate.read_bytes())
        except OSError:
            continue
        if challenge_digest is None or digest != challenge_digest:
            try:
                candidate.unlink()
            except OSError:
                pass


def manifest_series_pointers(
    *,
    schema: str = TRAIN_SERIES_V1_SCHEMA,
    path: str = TRAIN_SERIES_V1_FILENAME,
    sha256: str,
) -> dict[str, Any]:
    """Pointer fields folded into ``metrics`` of ``prism_run_manifest.v2``."""
    return {
        "train_series_schema": schema,
        "train_series_path": path,
        "train_series_sha256": sha256,
    }


def load_challenge_series(
    artifacts_dir: Path | str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Load and optionally verify the challenge-owned series side-car."""
    path = Path(artifacts_dir) / TRAIN_SERIES_V1_FILENAME
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if expected_sha256 is not None:
        if train_series_sha256(raw) != expected_sha256:
            return None
    if not series_is_challenge_owned(payload):
        return None
    return payload


# Allowed point keys for the public API time-flow (chart-safe; no secret material).
_API_POINT_KEYS = (
    "i",
    "tokens_seen",
    "covered_bytes",
    "train_ce_nats",
    "running_bpb",
    "wall_s",
    "grad_norm",
    "clip_event",
    "param_norm",
    "lr",
    "nan_inf",
)

# Allowed aggregate keys exposed on /curve (numerical residuals only).
_API_AGGREGATE_KEYS = (
    "n_points",
    "mean_step_ms",
    "p99_step_ms",
    "grad_spike_rate",
    "nan_inf_batches",
    "clip_events",
)


def _downsample_indices(n: int, cap: int) -> list[int]:
    """Even-stride indices that keep the first and last sample; identity when ``n <= cap``."""
    if n <= 0:
        return []
    if n <= cap:
        return list(range(n))
    return [round(i * (n - 1) / (cap - 1)) for i in range(cap)]


def sanitize_point_for_api(point: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project a series point onto chart-safe keys only (drops unknown/secret fields)."""
    if not isinstance(point, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in _API_POINT_KEYS:
        if key not in point:
            continue
        value = point[key]
        # Drop non-finite floats so chart libraries never receive NaN/Inf JSON issues.
        if isinstance(value, float) and not math.isfinite(value):
            out[key] = None
        else:
            out[key] = value
    # Required chart axes: index/tokens + train CE + wall.
    if "i" not in out and "tokens_seen" not in out:
        return None
    if "train_ce_nats" not in out and "running_bpb" not in out:
        return None
    return out


def densify_sample_eff_from_train_series(
    series: Mapping[str, Any] | None,
    *,
    mark_tokens: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Derive sample-efficiency residual marks from challenge series (never sole rank).

    Offline densify of running bpb / CE vs tokens_seen for Complete View / scorecard
    residual panels. Returns honest null+reason when series is missing or non-challenge.
    ``series_may_sole_rank`` is always False (VAL-TELE-010).
    """
    if not series_is_challenge_owned(series):
        return {
            "ok": False,
            "marks": {},
            "series_residual_only": True,
            "series_may_sole_rank": False,
            "reason": "series_not_challenge_owned_or_missing",
        }
    assert series is not None
    raw_points = series.get("points")
    points: list[Any] = list(raw_points) if isinstance(raw_points, list) else []
    default_marks = (10_000, 50_000, 100_000, 250_000, 500_000)
    marks = tuple(mark_tokens) if mark_tokens is not None else default_marks
    out_marks: dict[str, float | None] = {}
    for mark in marks:
        chosen: float | None = None
        for point in points:
            if not isinstance(point, Mapping):
                continue
            tokens = point.get("tokens_seen")
            if not isinstance(tokens, int | float):
                continue
            if int(tokens) < int(mark):
                continue
            bpb = point.get("running_bpb")
            if isinstance(bpb, int | float) and math.isfinite(float(bpb)):
                chosen = float(bpb)
                break
        out_marks[str(int(mark))] = chosen
    return {
        "ok": True,
        "marks": out_marks,
        "n_points": len(points),
        "series_residual_only": True,
        "series_may_sole_rank": False,
        "reason": None,
    }


def make_fixture_series(
    *,
    submission_id: str,
    run_id: str,
    family: str,
    n_points: int = 24,
    start_ce: float = 4.0,
    end_ce: float = 1.5,
    tokens_per_step: int = 512,
    wall_per_step: float = 0.05,
    grad_start: float = 2.5,
    grad_end: float = 0.4,
    clip_every: int = 5,
    seed_offset: float = 0.0,
) -> dict[str, Any]:
    """Build a synthetic challenge-owned series for dual-family time-flow evidence.

    Used by VAL-TELE-011 fixture export and unit tests that need two architecture
    (or two-run) series with loss + grad_norm + clip without paid Lium.
    """
    points: list[dict[str, Any]] = []
    for i in range(max(1, n_points)):
        frac = i / max(1, n_points - 1)
        ce = start_ce + (end_ce - start_ce) * frac + seed_offset * 0.01
        tokens = tokens_per_step * (i + 1)
        covered = float(tokens * 4)
        running = compute_running_bpb(
            sum_nll_nats=ce * tokens,  # rough proxy for fixture visualizations
            covered_bytes=covered,
        )
        grad = grad_start + (grad_end - grad_start) * frac
        clip = (i % max(1, clip_every) == 0) and i > 0
        points.append(
            series_point(
                i=i,
                tokens_seen=tokens,
                covered_bytes=covered,
                train_ce_nats=ce,
                running_bpb=running,
                wall_s=wall_per_step * (i + 1),
                grad_norm=grad,
                clip_event=clip,
                nan_inf=False,
                param_norm=None,
                lr=None,
            )
        )
    series = build_train_series_v1(
        submission_id=submission_id,
        run_id=run_id,
        points=points,
        token_budget=tokens_per_step * n_points,
        nan_inf_batches=0,
        extra_aggregates={"family": family, "fixture": True},
    )
    return series


def downsample_train_series_for_api(
    series: Mapping[str, Any] | None,
    *,
    max_points: int = 500,
) -> dict[str, Any] | None:
    """Return a downsample-safe, challenge-owned ``prism_train_series.v1`` for /curve charts.

    * Non-challenge / empty / corrupt documents become ``None`` (callers omit or null the field).
    * Points are stride-downsampled with first and last preserved (same policy as loss_curve).
    * Only chart keys (loss/bpb, tokens, wall, grad_norm, clip_event, …) are projected — never
      arbitrary miner/blob fields that might carry secrets.
    * Adds ``downsampled`` and ``points_total`` for UI clarity; does **not** recompute aggregates
      from the reduced set (clip_events / grad_spike_rate stay full-series truthful).
    """
    if not series_is_challenge_owned(series):
        return None
    assert series is not None
    raw_points = series.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        return None
    cleaned: list[dict[str, Any]] = []
    for raw in raw_points:
        projected = sanitize_point_for_api(raw) if isinstance(raw, Mapping) else None
        if projected is not None:
            cleaned.append(projected)
    if not cleaned:
        return None
    total = len(cleaned)
    indices = _downsample_indices(total, max_points)
    sampled = [cleaned[i] for i in indices]
    aggregates_in = series.get("aggregates")
    aggregates: dict[str, Any] = {}
    if isinstance(aggregates_in, Mapping):
        for key in _API_AGGREGATE_KEYS:
            if key in aggregates_in:
                aggregates[key] = aggregates_in[key]
    # Keep n_points equal to pre-downsample truth when present; else use total.
    aggregates.setdefault("n_points", total)
    payload: dict[str, Any] = {
        "schema": TRAIN_SERIES_V1_SCHEMA,
        "submission_id": str(series.get("submission_id", "")),
        "run_id": str(series.get("run_id", "")),
        "authority": TRAIN_SERIES_AUTHORITY,
        "x_axis": series.get("x_axis") if isinstance(series.get("x_axis"), str) else "batch_index",
        "token_budget": series.get("token_budget")
        if isinstance(series.get("token_budget"), int | float)
        else None,
        "points": sampled,
        "aggregates": aggregates,
        "miner_reported_ignored": True,
        "points_total": total,
        "downsampled": total > max_points,
    }
    return payload
