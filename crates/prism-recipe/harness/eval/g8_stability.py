"""G8 — training stability + µP LR-transfer (research/04 §6.9, research/11 §3).

Loss-spike count and divergence fraction come from harness telemetry
(`ctx["telemetry_series"]`, `ctx["probe_curve"]`) — no model needed.

The µP LR-stability micro-sweep (Tensor Programs V fairness substrate):
fresh 1× and 4×-width builds of the miner's architecture (via the
`prism_width_multiplier` ctx knob, documented for miners) trained for a
handful of steps at 3 LRs each on the harness micro stream; the metric
is |log2(best_lr_wide / best_lr_base)| — 0 means perfect LR transfer.

**Probe base (not full submission size).** The sweep does **not** start from
production `build_ctx` width/depth. Near the 1B cap, 4× width is
unbuildable on the eval GPU (~multi-billion params / ~100GB AdamW). Instead
the harness overlays a fixed small width/depth probe (`_MUP_PROBE_ARCH`) so
1× and 4× stay on-device for any submission size. Miners must honor
top-level / `arch` width-depth overrides **and** `prism_width_multiplier`
(reference baselines do).

Semantics for `org.g8.mup_lr_stability` (via rollup):
- sweep succeeds → `1/(1+|log2 ratio|)` in [0, 1]
- sweep **ran** but diverged / build failed / width unsupported / budget →
  **0.0** (fail-closed floor; composite always receives the org key)
- tiny_caps skip (tests) → stub only; org key omitted

v2.1 scaling-slope probe (`org.g8.mup_scaling_slope`, anchors ≥ v1): the
same sweep already trains the 1× and 4× width builds — the probe reuses
their best micro-losses to estimate the local scaling exponent
`(ln L_base − ln L_wide) / (ln N_wide − ln N_base)`, clamped at 0 when the
wide build is no better. Same fail-closed contract as `mup_lr_stability`:
0.0 after a failed real sweep, omitted on tiny-caps skips. Under anchor
set v0 the extra key is ignored by the composite (unknown keys are
inert), so emitting it is always safe.

Never silent-omit after a real sweep attempt (that made G8 incomplete).
"""

import math

from . import common

_SPIKE_MAD_K = 6.0

# Fixed µP probe geometry — independent of the scored submission's size.
# 4× width (~2× linear dims on d_model/mlp) must remain buildable on the
# eval GPU for every submission under the 1B cap. Keep vocab/tokenizer/
# device/seed from production build_ctx; only width/depth are replaced.
_MUP_PROBE_ARCH = {
    "d_model": 128,
    "n_layer": 4,
    "n_head": 4,
    "mlp_hidden": 320,  # 2.5 × 128 (Transformer++ SwiGLU ratio)
    # Hybrid delta-net extras (ignored by pure Transformer builds).
    "attn_heads": 4,
    "delta_key_dim": 64,
    "delta_value_dim": 128,
}


def mup_probe_base_ctx(build_ctx):
    """Overlay fixed small width/depth on production build_ctx for the µP sweep.

    Preserves vocab_size / tokenizer / device / seed and any non-geometry keys.
    Writes both top-level keys and an `arch` dict so baselines and miners that
    read either path see the probe geometry.
    """
    base = dict(build_ctx or {})
    arch = dict(base.get("arch") or {}) if isinstance(base.get("arch"), dict) else {}
    arch.update(_MUP_PROBE_ARCH)
    base["arch"] = arch
    for key, value in _MUP_PROBE_ARCH.items():
        base[key] = value
    # Never inherit a stale multiplier from a prior probe attempt.
    base.pop("prism_width_multiplier", None)
    return base


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _spike_stats(series):
    losses = [float(p["loss"]) for p in series if isinstance(p.get("loss"), (int, float))]
    losses = [x for x in losses if math.isfinite(x)]
    if len(losses) < 10:
        return None, None
    med = _median(losses)
    mad = _median([abs(x - med) for x in losses]) or 1e-9
    spikes = 0
    for i, x in enumerate(losses):
        if x <= med + _SPIKE_MAD_K * mad:
            continue
        lo = max(0, i - 20)
        if x > 1.25 * _median(losses[lo:i] or losses[:1]):
            spikes += 1
    return spikes, spikes / (len(losses) / 1000.0)


def _nan_frac(points, key):
    vals = [p.get(key) for p in points]
    if not vals:
        return None
    bad = sum(1 for v in vals if not isinstance(v, (int, float)) or not math.isfinite(v))
    return bad / len(vals)


def _micro_train_steps(model, stream, lr, steps, device):
    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    best = float("inf")
    for _ in range(steps):
        input_ids, labels = stream.next_batch()
        with torch.enable_grad():
            logits = common._logits_of(model(input_ids))
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1)
            )
        if math.isfinite(float(loss.item())):
            best = min(best, float(loss.item()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return best


def _scaling_slope(best_loss, n_params):
    """Local scaling exponent from the two width points (v2.1 probe).

    `(ln L_base − ln L_wide) / (ln N_wide − ln N_base)`, clamped ≥ 0.
    None when either side is missing/non-finite (caller fail-closes).
    """
    l_base, l_wide = best_loss.get(1.0), best_loss.get(4.0)
    n_base, n_wide = n_params.get(1.0), n_params.get(4.0)
    if not all(
        isinstance(v, (int, float)) and math.isfinite(v) and v > 0
        for v in (l_base, l_wide, n_base, n_wide)
    ):
        return None
    denom = math.log(n_wide) - math.log(n_base)
    if denom <= 0:
        return None
    return max(0.0, (math.log(l_base) - math.log(l_wide)) / denom)


def _mup_sweep(ctx, budget):
    """Returns (log2_ratio | None, slope | None, reason)."""
    import torch

    build = ctx.get("build_model")
    stream = ctx.get("micro_stream")
    if build is None or stream is None or not callable(build):
        return None, None, "no_build_model"
    device = ctx["device"]
    # Reduced fixed probe base — not full production build_ctx geometry.
    base_ctx = mup_probe_base_ctx(ctx.get("build_ctx"))
    # v2.1 field fix (2026-08-14 A/B runs): the fixed grid diverged at 4x
    # width for EVERY architecture tested (dense, hybrid delta, looped MoE),
    # zeroing mup_lr_stability across the board. Two sub-peak points keep at
    # least one finite loss per width so the transfer ratio (and the v2.1
    # scaling-slope probe) stay measurable.
    lrs = [1e-4, 3e-4, 1e-3, 3e-3]
    steps = 4 if common.tiny_caps() else 10
    best_by_width = {}
    best_loss_by_width = {}
    params_by_width = {}
    secret = common.resolve_secret_seed(ctx)
    for mult in (1.0, 4.0):
        bctx = dict(base_ctx)
        bctx["prism_width_multiplier"] = mult
        # Seeding is harness-owned: a failure here is our bug — surface it
        # as seed_error, never hide it behind the miner-facing build_failed.
        try:
            torch.manual_seed(common.torch_seed(secret, "g8/mup"))
        except Exception as exc:  # noqa: BLE001
            common.log(f"g8 mup seed failure: {type(exc).__name__}: {str(exc)[:200]}")
            return None, None, "seed_error"
        try:
            m = build(bctx)
            n_params = sum(p.numel() for p in m.parameters())
            m = m.to(device)
        except Exception as exc:  # noqa: BLE001 — genuinely miner-attributable
            common.log(
                f"g8 mup build failed (width x{mult}): {type(exc).__name__}: {str(exc)[:200]}"
            )
            return None, None, "build_failed"
        if mult == 1.0:
            base_params = n_params
        else:
            if base_params <= 0 or n_params <= int(1.5 * base_params):
                return None, None, "width_knob_unsupported"
        params_by_width[mult] = n_params
        per_lr = []
        for lr in lrs:
            if not budget.ok():
                return None, None, "budget"
            try:
                # Fresh init per LR point (same seed → comparable draws).
                torch.manual_seed(common.torch_seed(secret, f"g8/mup/{mult}/{lr}"))
            except Exception as exc:  # noqa: BLE001 — harness-owned; see above
                common.log(f"g8 mup seed failure: {type(exc).__name__}: {str(exc)[:200]}")
                return None, None, "seed_error"
            try:
                m2 = build(dict(bctx))
                m2 = m2.to(device)
                per_lr.append((_micro_train_steps(m2, stream, lr, steps, device), lr))
                del m2
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                per_lr.append((float("inf"), lr))
        del m
        finite = [(l, lr) for l, lr in per_lr if math.isfinite(l)]
        if not finite:
            return None, None, "sweep_diverged"
        best_loss, best_lr = min(finite)
        best_by_width[mult] = best_lr
        best_loss_by_width[mult] = best_loss
    ratio = best_by_width[4.0] / best_by_width[1.0]
    slope = _scaling_slope(best_loss_by_width, params_by_width)
    return abs(math.log2(ratio)), slope, None


def run(model, ctx):
    out = {}
    series = list(ctx.get("telemetry_series") or [])
    probes = list(ctx.get("probe_curve") or [])

    spikes, rate = _spike_stats(series)
    common.emit(out, "g8.spikes.count", 0.0 if spikes is None else spikes)
    common.emit(out, "g8.spikes.per_1k_steps", 0.0 if rate is None else rate)
    # Empty parent series (DDP workers never reported) is a measured "no
    # NaNs observed" — emit 0.0 so rollup always produces org.g8.loss_spike_score
    # instead of omitting the key. A documented stub, not a silent hole.
    series_nan = _nan_frac(series, "loss")
    probe_nan = _nan_frac(probes, "probe_loss")
    common.emit(out, "g8.divergence.series_nan_frac", 0.0 if series_nan is None else series_nan)
    common.emit(out, "g8.divergence.probe_nan_frac", 0.0 if probe_nan is None else probe_nan)
    if not series:
        out["g8.loss_spike.stub"] = 1.0
        out["g8.loss_spike.stub_reason_empty_series"] = 1.0

    # Share of the global battery budget (`PRISM_EVAL_G8_SWEEP_S` still
    # overrides for operator debugging).
    budget = common.Budget(
        common.float_env("PRISM_EVAL_G8_SWEEP_S", common.group_budget_s("g8"))
    )
    # The sweep needs real GPU-minutes: stubbed under tiny test caps
    # unless explicitly forced with PRISM_EVAL_G8_SWEEP=1.
    sweep_forced = common.float_env("PRISM_EVAL_G8_SWEEP", 0.0) == 1.0
    if common.tiny_caps() and not sweep_forced:
        out["g8.mup.stub"] = 1.0
        out["g8.mup.stub_reason_tiny_caps"] = 1.0
        return out
    ratio, slope, reason = _mup_sweep(ctx, budget)
    if ratio is None:
        out["g8.mup.stub"] = 1.0
        out[f"g8.mup.stub_reason_{reason}"] = 1.0
        # Fail-closed floor signal for rollup → org.g8.mup_lr_stability = 0.0
        # (and org.g8.mup_scaling_slope = 0.0, anchors ≥ v1) when the sweep
        # path was entered (not a tiny_caps skip).
        out["g8.mup.stability"] = 0.0
        out["g8.mup.scaling_slope"] = 0.0
    else:
        out["g8.mup.stub"] = 0.0
        common.emit(out, "g8.mup.lr_ratio_log2_abs", ratio)
        out["g8.mup.stability"] = 1.0 / (1.0 + max(0.0, ratio))
        # v2.1 scaling-slope probe: a slope the width points cannot support
        # (missing/non-finite losses) fail-closes to 0.0 like stability.
        out["g8.mup.scaling_slope"] = slope if slope is not None else 0.0
    return out
