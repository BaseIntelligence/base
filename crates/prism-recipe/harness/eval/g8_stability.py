"""G8 — training stability + µP LR-transfer (research/04 §6.9, research/11 §3).

Loss-spike count and divergence fraction come from harness telemetry
(`ctx["telemetry_series"]`, `ctx["probe_curve"]`) — no model needed.

The µP LR-stability micro-sweep (Tensor Programs V fairness substrate):
fresh 1× and 4×-width builds of the miner's architecture (via the
`prism_width_multiplier` ctx knob, documented for miners) trained for a
handful of steps at 3 LRs each on the harness micro stream; the metric
is |log2(best_lr_wide / best_lr_base)| — 0 means perfect LR transfer.
Skipped under tiny test caps or when the architecture ignores the width
knob → explicit stub keys with a machine-readable reason (never an
error, never silent).
"""

import math

from . import common

_SPIKE_MAD_K = 6.0


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


def _mup_sweep(ctx, budget):
    """Returns (log2_ratio | None, reason)."""
    import torch

    build = ctx.get("build_model")
    stream = ctx.get("micro_stream")
    if build is None or stream is None or not callable(build):
        return None, "no_build_model"
    device = ctx["device"]
    base_ctx = dict(ctx.get("build_ctx") or {})
    lrs = [3e-4, 1e-3, 3e-3]
    steps = 4 if common.tiny_caps() else 10
    best_by_width = {}
    for mult in (1.0, 4.0):
        try:
            bctx = dict(base_ctx)
            bctx["prism_width_multiplier"] = mult
            torch.manual_seed(common.task_seed(common.resolve_secret_seed(ctx), "g8/mup"))
            m = build(bctx)
            n_params = sum(p.numel() for p in m.parameters())
            m = m.to(device)
        except Exception:  # noqa: BLE001
            return None, "build_failed"
        if mult == 1.0:
            base_params = n_params
        else:
            if base_params <= 0 or n_params <= int(1.5 * base_params):
                return None, "width_knob_unsupported"
        per_lr = []
        for lr in lrs:
            if not budget.ok():
                return None, "budget"
            try:
                # Fresh init per LR point (same seed → comparable draws).
                torch.manual_seed(common.task_seed(
                    common.resolve_secret_seed(ctx), f"g8/mup/{mult}/{lr}"
                ))
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
            return None, "sweep_diverged"
        best_by_width[mult] = min(finite)[1]
    ratio = best_by_width[4.0] / best_by_width[1.0]
    return abs(math.log2(ratio)), None


def run(model, ctx):
    out = {}
    series = list(ctx.get("telemetry_series") or [])
    probes = list(ctx.get("probe_curve") or [])

    spikes, rate = _spike_stats(series)
    common.emit(out, "g8.spikes.count", spikes)
    common.emit(out, "g8.spikes.per_1k_steps", rate)
    common.emit(out, "g8.divergence.series_nan_frac", _nan_frac(series, "loss"))
    common.emit(out, "g8.divergence.probe_nan_frac", _nan_frac(probes, "probe_loss"))

    budget = common.Budget(common.float_env("PRISM_EVAL_G8_SWEEP_S", 300.0))
    # The sweep needs real GPU-minutes: stubbed under tiny test caps
    # unless explicitly forced with PRISM_EVAL_G8_SWEEP=1.
    sweep_forced = common.float_env("PRISM_EVAL_G8_SWEEP", 0.0) == 1.0
    if common.tiny_caps() and not sweep_forced:
        out["g8.mup.stub"] = 1.0
        out["g8.mup.stub_reason_tiny_caps"] = 1.0
        return out
    ratio, reason = _mup_sweep(ctx, budget)
    if ratio is None:
        out["g8.mup.stub"] = 1.0
        out[f"g8.mup.stub_reason_{reason}"] = 1.0
    else:
        out["g8.mup.stub"] = 0.0
        common.emit(out, "g8.mup.lr_ratio_log2_abs", ratio)
    return out
