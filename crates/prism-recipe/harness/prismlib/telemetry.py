"""Miner-facing `prism_telemetry` shim + FinishEvaluation (recipe >= 1.1.0).

Recipe 1.3.0 addition: `report()` fires the harness probe hook (G6
intermediate probes) every `PRISM_PROBE_EVERY`-th report when a hook is
installed in `state["probe_hook"]`, appending
`{step, tokens_seen, wall_s, probe_loss}` to `state["probe_curve"]`. Probe
exceptions are caught and logged — a probe never kills training.
"""

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
