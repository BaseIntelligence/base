"""Best-effort layer / activation diagnostics for eval-complete multi-family cups.

Challenge-owned densify path (VAL-EVALC-003):

* Weight L2 norms per named parameter and coarse layer group from ``trained_state.pt``.
* Optional one-batch activation mean/std via forward hooks when a pure-torch module can
  be reconstructed and a single forward succeeds.
* Grad-norm stream aggregates from challenge-owned ``prism_train_series.v1`` when present.

Full per-layer activation *maps* under AST/opaque packs may return
``BLOCKED_with_reason`` for that slice; weight norms + grad series still ship.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .train_series import series_is_challenge_owned

LAYER_DIAGNOSTICS_SCHEMA = "prism_layer_diagnostics.v1"
LAYER_DIAGNOSTICS_FILENAME = "prism_layer_diagnostics.v1.json"


def _as_tensor_dict(payload: Any) -> dict[str, Any] | None:
    """Normalize torch.load payloads / plain mappings into a name→tensor mapping."""
    if not isinstance(payload, Mapping):
        return None
    # Nested checkpoint form.
    for key in ("state_dict", "model", "weights", "params"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and nested:
            sample = next(iter(nested.values()), None)
            if hasattr(sample, "detach") or hasattr(sample, "shape"):
                return dict(nested)
    # Plain state_dict-like mapping.
    values = list(payload.values())
    if not values:
        return None
    if any(hasattr(v, "detach") or hasattr(v, "shape") for v in values[:8]):
        return {
            str(k): v for k, v in payload.items() if hasattr(v, "detach") or hasattr(v, "shape")
        }
    return None


def _param_layer_group(name: str) -> str:
    parts = [p for p in str(name).split(".") if p]
    if not parts:
        return "unknown"
    if len(parts) == 1:
        return parts[0]
    # blocks.N.* → blocks.N; otherwise token_emb.weight → token_emb.weight first two
    if parts[0] in {"blocks", "layers", "h", "encoder", "decoder"} and len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return f"{parts[0]}.{parts[1]}"


def weight_l2_norms_from_state(
    state: Mapping[str, Any] | Any,
    *,
    max_params: int = 20_000,
) -> dict[str, Any]:
    """Compute L2 norms per named parameter and coarse layer groups.

    Accepts a plain state_dict mapping or a torch.load checkpoint. Returns an
    honest empty pack with reason when no tensors are found.
    """
    tensors = _as_tensor_dict(state)
    if tensors is None:
        return {
            "ok": False,
            "reason": "trained_state_not_tensor_mapping",
            "param_l2": {},
            "layer_group_l2": {},
            "global_l2": None,
            "n_tensors": 0,
            "n_parameters": 0,
        }

    param_l2: dict[str, float] = {}
    layer_sq: dict[str, float] = {}
    global_sq = 0.0
    n_parameters = 0
    n_tensors = 0

    for name, value in list(tensors.items())[: max(1, int(max_params))]:
        try:
            if hasattr(value, "detach"):
                tensor = value.detach().float().cpu()
            elif hasattr(value, "float"):
                tensor = value.float()
            else:
                continue
            norm = float(tensor.norm().item())
            if not math.isfinite(norm):
                continue
            param_l2[str(name)] = norm
            n_parameters += int(tensor.numel())
            n_tensors += 1
            global_sq += norm * norm
            group = _param_layer_group(str(name))
            layer_sq[group] = layer_sq.get(group, 0.0) + norm * norm
        except Exception:
            continue

    if n_tensors == 0:
        return {
            "ok": False,
            "reason": "no_finite_tensor_norms",
            "param_l2": {},
            "layer_group_l2": {},
            "global_l2": None,
            "n_tensors": 0,
            "n_parameters": 0,
        }

    layer_group_l2 = {
        key: float(math.sqrt(sq)) for key, sq in sorted(layer_sq.items(), key=lambda kv: kv[0])
    }
    return {
        "ok": True,
        "reason": None,
        "param_l2": param_l2,
        "layer_group_l2": layer_group_l2,
        "global_l2": float(math.sqrt(global_sq)),
        "n_tensors": int(n_tensors),
        "n_parameters": int(n_parameters),
    }


def load_weight_l2_from_trained_state(
    path: Path | str,
    *,
    max_params: int = 20_000,
) -> dict[str, Any]:
    """Load ``trained_state.pt`` and return weight L2 diagnostics (host/CPU safe)."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment without torch
        return {
            "ok": False,
            "reason": f"torch_unavailable:{type(exc).__name__}",
            "param_l2": {},
            "layer_group_l2": {},
            "global_l2": None,
            "n_tensors": 0,
            "n_parameters": 0,
        }
    p = Path(path)
    if not p.is_file():
        return {
            "ok": False,
            "reason": "trained_state_missing",
            "param_l2": {},
            "layer_group_l2": {},
            "global_l2": None,
            "n_tensors": 0,
            "n_parameters": 0,
        }
    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch without weights_only kwarg.
        payload = torch.load(p, map_location="cpu")
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"trained_state_load_failed:{type(exc).__name__}",
            "param_l2": {},
            "layer_group_l2": {},
            "global_l2": None,
            "n_tensors": 0,
            "n_parameters": 0,
        }
    return weight_l2_norms_from_state(payload, max_params=max_params)


def grad_norm_aggregates_from_series(series: Mapping[str, Any] | None) -> dict[str, Any]:
    """Aggregate grad_norm / clip_events from challenge-owned train series."""
    if not series_is_challenge_owned(series):
        return {
            "ok": False,
            "reason": "series_not_challenge_owned_or_missing",
            "n_points": 0,
            "n_grad_points": 0,
            "grad_norm_mean": None,
            "grad_norm_median": None,
            "grad_norm_max": None,
            "grad_norm_p95": None,
            "grad_spike_rate": None,
            "clip_events": None,
            "clip_event_rate": None,
        }
    assert series is not None
    raw_points = series.get("points")
    points: list[Any] = raw_points if isinstance(raw_points, list) else []
    grads: list[float] = []
    clips = 0
    for point in points:
        if not isinstance(point, Mapping):
            continue
        g = point.get("grad_norm")
        if isinstance(g, int | float) and math.isfinite(float(g)):
            grads.append(float(g))
        if point.get("clip_event") is True:
            clips += 1
    raw_aggregates = series.get("aggregates")
    aggregates: Mapping[str, Any] = raw_aggregates if isinstance(raw_aggregates, Mapping) else {}
    if not grads:
        return {
            "ok": False,
            "reason": "no_finite_grad_norms_in_series",
            "n_points": len(points),
            "n_grad_points": 0,
            "grad_norm_mean": None,
            "grad_norm_median": None,
            "grad_norm_max": None,
            "grad_norm_p95": None,
            "grad_spike_rate": aggregates.get("grad_spike_rate"),
            "clip_events": int(aggregates.get("clip_events", clips)),
            "clip_event_rate": None,
        }
    ordered = sorted(grads)
    n = len(ordered)
    med = ordered[n // 2]
    thr = max(med * 10.0, 1e-8)
    spike = sum(1 for g in grads if g > thr) / float(n)
    p95 = ordered[min(n - 1, int(math.ceil(0.95 * n) - 1))]
    clip_total = int(aggregates.get("clip_events", clips))
    return {
        "ok": True,
        "reason": None,
        "n_points": len(points),
        "n_grad_points": n,
        "grad_norm_mean": float(sum(grads) / n),
        "grad_norm_median": float(med),
        "grad_norm_max": float(ordered[-1]),
        "grad_norm_p95": float(p95),
        "grad_spike_rate": float(spike),
        "clip_events": clip_total,
        "clip_event_rate": float(clip_total / max(1, len(points))),
    }


def optional_activation_stats_one_batch(
    model: Any,
    batch_tokens: Any,
    *,
    max_modules: int = 64,
) -> dict[str, Any]:
    """Sample forward-hook mean/std on immediate/named children for one batch.

    Best-effort: when hooks fail, opaque modules error, or tensors are non-finite,
    returns ``BLOCKED_with_reason`` rather than fabricating maps.
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "status": "BLOCKED_with_reason",
            "reason": f"torch_unavailable:{type(exc).__name__}",
            "modules": {},
        }
    if not isinstance(model, nn.Module):
        return {
            "ok": False,
            "status": "BLOCKED_with_reason",
            "reason": "model_not_nn_module",
            "modules": {},
        }
    if not isinstance(batch_tokens, torch.Tensor):
        return {
            "ok": False,
            "status": "BLOCKED_with_reason",
            "reason": "batch_not_tensor",
            "modules": {},
        }

    stats: dict[str, dict[str, float | int | str]] = {}
    handles: list[Any] = []

    def _make_hook(name: str):
        def _hook(_module, _inp, out):
            try:
                tensor = out
                if isinstance(out, (tuple, list)):
                    tensor = next((x for x in out if isinstance(x, torch.Tensor)), None)
                elif hasattr(out, "last_hidden_state"):
                    tensor = out.last_hidden_state
                if not isinstance(tensor, torch.Tensor):
                    stats[name] = {"status": "skipped_non_tensor"}
                    return
                flat = tensor.detach().float().reshape(-1)
                if flat.numel() == 0:
                    stats[name] = {"status": "empty"}
                    return
                mean = float(flat.mean().item())
                std = float(flat.std(unbiased=False).item())
                if not (math.isfinite(mean) and math.isfinite(std)):
                    stats[name] = {"status": "non_finite"}
                    return
                stats[name] = {
                    "status": "ok",
                    "mean": mean,
                    "std": std,
                    "numel": int(flat.numel()),
                    "shape": list(tensor.shape),
                }
            except Exception as hook_exc:  # noqa: BLE001
                stats[name] = {
                    "status": "hook_error",
                    "error": type(hook_exc).__name__,
                }

        return _hook

    registered = 0
    try:
        for name, child in model.named_modules():
            if name == "":
                continue
            if registered >= max_modules:
                break
            # Prefer leaf-ish children; still allow one level deeper.
            try:
                handle = child.register_forward_hook(_make_hook(name or child.__class__.__name__))
                handles.append(handle)
                registered += 1
            except Exception:
                continue
        if registered == 0:
            return {
                "ok": False,
                "status": "BLOCKED_with_reason",
                "reason": "no_forward_hooks_registered",
                "modules": {},
            }
        model.eval()
        with torch.no_grad():
            _ = model(batch_tokens)
    except Exception as exc:  # noqa: BLE001
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        return {
            "ok": False,
            "status": "BLOCKED_with_reason",
            "reason": f"forward_failed:{type(exc).__name__}:{exc}",
            "modules": {},
        }
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    ok_modules = {
        k: v for k, v in stats.items() if isinstance(v, Mapping) and v.get("status") == "ok"
    }
    if not ok_modules:
        return {
            "ok": False,
            "status": "BLOCKED_with_reason",
            "reason": "activation_hooks_yielded_no_ok_stats",
            "modules": stats,
        }
    return {
        "ok": True,
        "status": "ok",
        "reason": None,
        "modules": stats,
        "n_modules_ok": len(ok_modules),
        "n_modules_registered": registered,
    }


def build_layer_diagnostics_v1(
    *,
    family_id: str,
    submission_id: str,
    run_id: str,
    weight_norms: Mapping[str, Any] | None,
    activation: Mapping[str, Any] | None,
    grad_aggregates: Mapping[str, Any] | None,
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assemble a complete ``prism_layer_diagnostics.v1`` pack for one family."""
    w = dict(weight_norms or {})
    a = dict(activation or {})
    g = dict(grad_aggregates or {})
    activation_status = "ok" if a.get("ok") is True else "BLOCKED_with_reason"
    if a.get("status"):
        activation_status = str(a.get("status"))
    return {
        "schema": LAYER_DIAGNOSTICS_SCHEMA,
        "family_id": str(family_id),
        "submission_id": str(submission_id),
        "run_id": str(run_id),
        "authority": "challenge",
        "weight_l2": {
            "ok": bool(w.get("ok")),
            "reason": w.get("reason"),
            "global_l2": w.get("global_l2"),
            "n_tensors": w.get("n_tensors", 0),
            "n_parameters": w.get("n_parameters", 0),
            "param_l2": w.get("param_l2") if isinstance(w.get("param_l2"), Mapping) else {},
            "layer_group_l2": w.get("layer_group_l2")
            if isinstance(w.get("layer_group_l2"), Mapping)
            else {},
        },
        "activation_one_batch": {
            "ok": bool(a.get("ok")),
            "status": activation_status,
            "reason": a.get("reason"),
            "n_modules_ok": a.get("n_modules_ok"),
            "n_modules_registered": a.get("n_modules_registered"),
            # Keep module map when present (even when BLOCKED for partial evidence).
            "modules": a.get("modules") if isinstance(a.get("modules"), Mapping) else {},
            "note": (
                "Full spatial activation maps are out of scope when packs are AST-opaque; "
                "this field is mean/std sample only when a forward succeeds."
            ),
        },
        "grad_norm_aggregates": {
            "ok": bool(g.get("ok")),
            "reason": g.get("reason"),
            "n_points": g.get("n_points"),
            "n_grad_points": g.get("n_grad_points"),
            "grad_norm_mean": g.get("grad_norm_mean"),
            "grad_norm_median": g.get("grad_norm_median"),
            "grad_norm_max": g.get("grad_norm_max"),
            "grad_norm_p95": g.get("grad_norm_p95"),
            "grad_spike_rate": g.get("grad_spike_rate"),
            "clip_events": g.get("clip_events"),
            "clip_event_rate": g.get("clip_event_rate"),
        },
        "notes": list(notes or []),
        "miner_reported_ignored": True,
    }
