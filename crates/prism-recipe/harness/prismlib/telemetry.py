"""Miner-facing `prism_telemetry` shim + FinishEvaluation (recipe >= 1.1.0).

Recipe 1.3.0 addition: `report()` fires the harness probe hook (G6
intermediate probes) every `PRISM_PROBE_EVERY`-th report when a hook is
installed in `state["probe_hook"]`, appending
`{step, tokens_seen, wall_s, probe_loss}` to `state["probe_curve"]`. Probe
exceptions are caught and logged — a probe never kills training.
"""

import json
import math
import os
import time
import types

from . import MAX_LAYER_STAT_KEYS, MAX_TELEMETRY_POINTS


class FinishEvaluation(BaseException):
    """Stop signal raised through train() by finish_evaluation()."""


def _sanitize_layer_stats(stats):
    if not isinstance(stats, dict):
        return None
    out = {}
    for k, v in stats.items():
        if len(out) >= MAX_LAYER_STAT_KEYS:
            break
        key = str(k)[:64]
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, str):
            out[key] = v[:128]
    return out


def build_telemetry_module(log=print):
    """Create the miner-facing `prism_telemetry` shim module + its state.

    State keys: `series` (bounded loss curve), `reports` (total calls), `t0`
    (train start wall clock), `probe_curve` (G6 points), `probe_hook`
    (installed by the harness after build_model; signature
    `hook(state, step)`).
    """
    state = {
        "series": [],
        "reports": 0,
        "t0": time.time(),
        "probe_curve": [],
        "probe_hook": None,
    }
    mod = types.ModuleType("prism_telemetry")
    mod.__doc__ = "PRISM miner telemetry hook (harness-provided)."
    mod.FinishEvaluation = FinishEvaluation

    def report(loss=None, step=None, grad_norm=None, layer_stats=None):
        state["reports"] += 1
        if loss is None or step is None:
            raise ValueError("prism_telemetry.report requires loss= and step=")
        step = int(step)
        pt = {
            "step": step,
            "loss": float(loss),
            "at_secs": round(time.time() - state["t0"], 3),
        }
        if grad_norm is not None:
            pt["grad_norm"] = float(grad_norm)
        clean = _sanitize_layer_stats(layer_stats)
        if clean:
            pt["layer_stats"] = clean
        series = state["series"]
        series.append(pt)
        if len(series) > MAX_TELEMETRY_POINTS:
            # Decimate instead of truncating: keep the first point and halve
            # the rest so the stored curve still spans the whole run.
            del series[1::2]
        hook = state.get("probe_hook")
        if hook is not None:
            try:
                hook(state, step)
            except Exception as exc:  # noqa: BLE001
                log(f"probe hook error (ignored): {exc}")

    def finish_evaluation():
        raise FinishEvaluation("prism_telemetry.finish_evaluation() called")

    mod.report = report
    mod.finish_evaluation = finish_evaluation
    return mod, state


# Sidecar written by DDP workers (spawned processes do not share the parent
# in-memory prism_telemetry shim / probe_hook). Parent train_v3 ingests this
# after train() returns so G6/G8 stay complete.
DDP_SIDECAR_RELPATHS = (
    "prism_ddp/telemetry.json",
    "loopmoe_ddp/telemetry.json",
)


def sidecar_path(workdir, rel="prism_ddp/telemetry.json"):
    return os.path.join(str(workdir), rel)


def write_ddp_sidecar(workdir, reports, probe_curve=None, rel="loopmoe_ddp/telemetry.json"):
    """Miner/worker helper: persist rank-0 telemetry for the parent ingest."""
    os.makedirs(os.path.dirname(sidecar_path(workdir, rel)), exist_ok=True)
    payload = {
        "report_count": len(list(reports or [])),
        "reports": list(reports or []),
        "probe_curve": list(probe_curve or []),
    }
    path = sidecar_path(workdir, rel)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def ingest_ddp_sidecar(state, workdir):
    """Merge worker sidecar(s) into the parent telemetry state.

    Does **not** fire `probe_hook` (that would re-probe the final checkpoint
    at every historical step). Worker `probe_curve` points are appended;
    if workers only shipped a loss series, a probe curve is synthesized
    from those losses so G6 has ≥2 points.
    """
    if not workdir:
        return 0
    merged = 0
    for rel in DDP_SIDECAR_RELPATHS:
        path = sidecar_path(workdir, rel)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError):
            continue
        reports = blob.get("reports") or []
        for pt in reports:
            if not isinstance(pt, dict):
                continue
            if "loss" not in pt or "step" not in pt:
                continue
            rec = {
                "step": int(pt["step"]),
                "loss": float(pt["loss"]),
                "at_secs": float(pt.get("at_secs") or 0.0),
            }
            if pt.get("grad_norm") is not None:
                rec["grad_norm"] = float(pt["grad_norm"])
            state["series"].append(rec)
            state["reports"] += 1
            merged += 1
        curve = list(blob.get("probe_curve") or [])
        if not curve and reports:
            curve = _curve_from_reports(reports)
        for pt in curve:
            if not isinstance(pt, dict):
                continue
            if not _finite_probe_point(pt):
                continue
            state["probe_curve"].append(pt)
    return merged


def _finite_probe_point(pt):
    try:
        return math.isfinite(float(pt.get("probe_loss", float("nan")))) and int(
            pt.get("tokens_seen", -1)
        ) >= 0
    except (TypeError, ValueError):
        return False


def _curve_from_reports(reports):
    """Fallback G6 curve: train loss vs tokens (or step×512)."""
    out = []
    for pt in reports:
        if not isinstance(pt, dict) or pt.get("loss") is None or pt.get("step") is None:
            continue
        try:
            loss = float(pt["loss"])
            step = int(pt["step"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(loss):
            continue
        tokens = pt.get("tokens_seen")
        try:
            tokens = int(tokens) if tokens is not None else max(1, step * 512)
        except (TypeError, ValueError):
            tokens = max(1, step * 512)
        out.append(
            {
                "step": step,
                "tokens_seen": tokens,
                "bytes_seen": max(1, tokens * 4),
                "flops_spent": float(pt.get("flops_spent") or 0.0),
                "wall_s": float(pt.get("at_secs") or 0.0),
                "probe_loss": loss,
                "synthesized_from_report": True,
            }
        )
    return out
