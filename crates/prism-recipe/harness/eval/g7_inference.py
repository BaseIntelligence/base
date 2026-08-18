"""G7 — inference efficiency (research/07 §1, §2, §4, §8).

Measured on the harness machine, same checkpoint, same decoding loop for
every submission: TTFT (prefill) vs context at batch 1, TPOT vs context
at batch 1, decode throughput at effective concurrency {1, 32}
(sequential saturated-batch simulation on one GPU), peak VRAM, a state
card (measured bytes/token slope from peak-memory-vs-context vs the
miner's analytic `state_bytes_per_token(ctx_len)` declaration when
present), J/token via nvidia-smi when available (skipped on CPU), and
W8A8 / W4A16 fake-quant quality deltas on a val mini-suite (dynamic
per-channel torch quant — no new pod image).

CUDA-only sub-metrics emit explicit `*.skipped=1` flags on CPU so the
composite can distinguish "measured zero" from "not measured".
"""

import copy
import os
import shutil
import subprocess
import time

from . import common

_GRID_FULL = (1024, 4096, 16384, 32768)
_GRID_TINY = (512, 2048)
_CENSORED_THROUGHPUT = 1.0e-12
_CENSORED_LATENCY_MS = 1.0e12
_CENSORED_STATE_BYTES_PER_TOKEN = 1.0e15
_CENSORED_JOULES_PER_TOKEN = 1.0e12


def _rand_ids(torch, n, vocab, device, rng):
    return torch.randint(0, vocab, (1, n), generator=rng).to(device)


def _vocab_size(model, tok):
    """Model declaration first, else the submitted tokenizer's vocab (never
    a hardcoded GPT-2 50257: the tokenizer is part of the submission)."""
    return common.vocab_size(tok, model=model)


def _nvidia_smi_power_w():
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        r = subprocess.run(
            [exe, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None
    return None


def _start_power_sampler():
    """Sample board power during a workload (best effort, no shell)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        return subprocess.Popen(
            [
                exe,
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                "--loop-ms=100",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return None


def _stop_power_sampler(proc):
    if proc is None:
        return []
    try:
        proc.terminate()
        stdout, _ = proc.communicate(timeout=3)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
            stdout, _ = proc.communicate(timeout=3)
        except Exception:  # noqa: BLE001
            return []
    vals = []
    for line in (stdout or "").splitlines():
        try:
            vals.append(float(line.strip().split()[0]))
        except (ValueError, IndexError):
            continue
    return vals


def _fake_quant_linear(lin, wbits, act8=False, group=128):
    """In-place per-output-channel symmetric fake-quant of a Linear."""
    import torch

    w = lin.weight.data.float()
    qmax = 2 ** (wbits - 1) - 1
    if wbits <= 4:
        # Group-wise (g128) scales, W4A16-style.
        out_f, in_f = w.shape
        w_g = w.reshape(out_f, -1, group) if in_f % group == 0 else None
        if w_g is None:
            scale = w.abs().amax(dim=1, keepdim=True) / qmax
            lin.weight.data = (torch.round(w / scale).clamp(-qmax, qmax) * scale).to(lin.weight.dtype)
            return
        scale = w_g.abs().amax(dim=2, keepdim=True) / qmax
        w_q = (torch.round(w_g / scale).clamp(-qmax, qmax) * scale).reshape(out_f, in_f)
        lin.weight.data = w_q.to(lin.weight.dtype)
    else:
        scale = w.abs().amax(dim=1, keepdim=True) / qmax
        lin.weight.data = (torch.round(w / scale).clamp(-qmax, qmax) * scale).to(lin.weight.dtype)


def _quantized_copy(model, wbits, act8=False):
    import torch

    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            _fake_quant_linear(mod, wbits, act8=act8)
    if act8:
        # W8A8: dynamic per-token int8 fake-quant on Linear inputs.
        def hook(_mod, inputs):
            x = inputs[0]
            if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                return None
            xf = x.float()
            scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
            return ((torch.round(xf / scale).clamp(-127, 127) * scale).to(x.dtype),)

        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.register_forward_pre_hook(hook)
    return m


def _mini_suite_bpb(model, tok, texts, device, cap):
    ces = []
    for txt in texts[:cap]:
        try:
            stats = common.doc_loss_stats(model, tok, txt, device, ())
        except Exception:  # noqa: BLE001
            continue
        if stats is not None:
            ces.append(stats["ce"])
    return common.bpb(common.mean(ces))


def run(model, ctx):
    import torch

    tok, device = ctx["tokenizer"], ctx["device"]
    cuda = device == "cuda" and torch.cuda.is_available()
    budget = common.Budget(common.group_budget_s("g7"))
    out = {}
    tiny = common.tiny_caps()
    grid = _GRID_TINY if tiny else _GRID_FULL
    decode_k = 8 if tiny else 32
    rng = torch.Generator(device="cpu").manual_seed(common.torch_seed(
        common.resolve_secret_seed(ctx), "g7/shapes"
    ))
    vocab = _vocab_size(model, tok)
    model.eval()

    peak_by_len = {}
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # -- TTFT (prefill) vs context, batch 1 -------------------------------
    for L in grid:
        if not budget.ok():
            out["g7.partial"] = 1.0
            break
        ids = _rand_ids(torch, L, vocab, device, rng)
        try:
            with torch.no_grad():
                common._logits_of(model(ids))  # warmup this length
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                common._logits_of(model(ids))
                if cuda:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
        except Exception:  # noqa: BLE001 — e.g. OOM past a length
            break
        common.emit(out, f"g7.ttft.L{L}.ms", dt * 1000.0)
        common.record(ctx, "g7.ttft", f"L{L}", dt * 1000.0)
        if cuda:
            peak_by_len[L] = torch.cuda.max_memory_allocated()

    # -- TPOT vs context, batch 1 -----------------------------------------
    for L in grid:
        if not budget.ok():
            out["g7.partial"] = 1.0
            break
        ids = _rand_ids(torch, L, vocab, device, rng)
        times = []
        try:
            with torch.no_grad():
                common._logits_of(model(ids))  # warm the prefix
                cur = ids
                for _ in range(decode_k):
                    # CPU generator + CUDA device is illegal; mirror `_rand_ids`.
                    nxt = torch.randint(0, vocab, (1, 1), generator=rng).to(device)
                    cur = torch.cat([cur, nxt], dim=1)
                    if cuda:
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    common._logits_of(model(cur))
                    if cuda:
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
        except Exception:  # noqa: BLE001
            break
        if times:
            common.emit(out, f"g7.tpot.L{L}.ms", common.mean(times) * 1000.0)

    # -- Throughput at concurrency {1, 32} (saturated batch) ---------------
    for b, tag in ((1, "b1"), (4 if tiny else 32, "b32")):
        if not budget.ok():
            out["g7.partial"] = 1.0
            break
        pre = min(grid[0], 1024)
        ids = torch.randint(0, vocab, (b, pre), device=device)
        try:
            with torch.no_grad():
                common._logits_of(model(ids))
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                cur = ids
                for _ in range(decode_k):
                    nxt = torch.randint(0, vocab, (b, 1), device=device)
                    cur = torch.cat([cur, nxt], dim=1)
                    common._logits_of(model(cur))
                if cuda:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            continue
        toks = b * decode_k
        common.emit(out, f"g7.throughput.{tag}.toks", toks / dt)
        common.record(ctx, "g7.throughput", tag, toks / dt)

    if cuda:
        common.emit(out, "g7.vram.peak_gb", torch.cuda.max_memory_allocated() / 1e9)
    else:
        out["g7.vram.skipped"] = 1.0

    # -- State card: measured bytes/token slope vs analytic declaration ----
    if cuda and len(peak_by_len) >= 2:
        ls = sorted(peak_by_len)
        slope = (peak_by_len[ls[-1]] - peak_by_len[ls[0]]) / float(ls[-1] - ls[0])
        common.emit(out, "g7.state.bytes_per_token.measured", slope)
        analytic_fn = getattr(model, "state_bytes_per_token", None)
        if callable(analytic_fn):
            try:
                analytic = float(analytic_fn(ls[-1]))
                common.emit(out, "g7.state.bytes_per_token.analytic", analytic)
                if analytic > 0:
                    common.emit(
                        out, "g7.state.gap_pct", abs(slope - analytic) / analytic * 100.0
                    )
            except Exception:  # noqa: BLE001
                out["g7.state.analytic_unavailable"] = 1.0
        else:
            out["g7.state.analytic_unavailable"] = 1.0
    else:
        out["g7.state.skipped"] = 1.0

    # -- Energy: total board J/token via nvidia-smi ------------------------
    p_idle = _nvidia_smi_power_w()
    if cuda and p_idle is not None and budget.ok():
        b, pre = (4 if tiny else 32), min(grid[0], 1024)
        ids = torch.randint(0, vocab, (b, pre), device=device)
        sampler = None
        try:
            with torch.no_grad():
                if cuda:
                    torch.cuda.synchronize()
                sampler = _start_power_sampler()
                t0 = time.perf_counter()
                cur = ids
                for _ in range(decode_k):
                    nxt = torch.randint(0, vocab, (b, 1), device=device)
                    cur = torch.cat([cur, nxt], dim=1)
                    common._logits_of(model(cur))
                if cuda:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            samples = _stop_power_sampler(sampler)
            sampler = None
            p_after = _nvidia_smi_power_w()
            watts = common.mean(samples)
            if watts is None and p_after is not None:
                watts = 0.5 * (p_idle + p_after)
            if watts is not None and watts > 0 and dt > 0:
                # Total board energy is stable even when a short workload
                # finishes before a post-hoc power query observes load.
                j = watts * dt
                common.emit(out, "g7.energy.j_per_token", j / (b * decode_k))
                common.emit(out, "g7.energy.samples", len(samples))
            else:
                out["g7.energy.skipped"] = 1.0
        except Exception:  # noqa: BLE001
            out["g7.energy.skipped"] = 1.0
        finally:
            _stop_power_sampler(sampler)
    else:
        out["g7.energy.skipped"] = 1.0

    # -- Fake-quant deltas on a val mini-suite ------------------------------
    texts = list(ctx.get("val_texts") or [])
    if texts and budget.ok():
        cap = 2 if tiny else 8
        base = _mini_suite_bpb(model, tok, texts, device, cap)
        if base is not None:
            for tag, wbits, act8 in (("w8a8", 8, True), ("w4a16", 4, False)):
                if not budget.ok():
                    out["g7.partial"] = 1.0
                    break
                try:
                    qmodel = _quantized_copy(model, wbits, act8=act8)
                    q = _mini_suite_bpb(qmodel, tok, texts, device, cap)
                    del qmodel
                    if cuda:
                        torch.cuda.empty_cache()
                    if q is not None:
                        common.emit(out, f"g7.quant.{tag}.dbpb", q - base)
                except Exception:  # noqa: BLE001
                    out[f"g7.quant.{tag}.skipped"] = 1.0
    # Anchored G7 keys are total: unsupported context, OOM, exhausted budget,
    # CPU smoke, and unavailable power telemetry are honest right-censoring
    # events, not reasons to omit a key and invalidate the whole composite.
    censored = 0
    for key, value in (
        ("g7.throughput.b32.toks", _CENSORED_THROUGHPUT),
        ("g7.ttft.L32768.ms", _CENSORED_LATENCY_MS),
        ("g7.tpot.L32768.ms", _CENSORED_LATENCY_MS),
        ("g7.state.bytes_per_token.measured", _CENSORED_STATE_BYTES_PER_TOKEN),
        ("g7.energy.j_per_token", _CENSORED_JOULES_PER_TOKEN),
    ):
        if key not in out:
            out[key] = value
            censored += 1
    out["g7.anchor_censored"] = float(censored)
    return out
