"""PRISM v3 reference baseline training: hybrid delta-net (recipe >= 1.3.0).

Identical recipe structure to the Transformer++ baseline (see its
training.py for the full contract notes); the only tuning difference is the
peak LR: 3e-4 instead of 6e-4. Gated delta-rule stacks (per-channel decay +
beta learning rates + output gates) are more prone to loss spikes than
plain attention at this scale, so the hybrid runs a more conservative peak
with the same warmup/cosine shape (see NOTES.md).

- AdamW, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 — no decay on norms,
  embeddings (tied head), or any <2D parameter
- LR: linear warmup over 2% of steps, cosine decay to 10% of peak
- grad clip 1.0; bf16 autocast on CUDA, plain fp32 on CPU
- data: `ctx["train_stream"]` primarily; legacy `dataset_path` fallback
- `guard()` every step; wall-clock self-monitored (cap minus margin);
  `prism_telemetry.report` every 10 steps; `finish_evaluation()` at the end
"""

import math
import random
import time

import torch
import torch.nn.functional as F

try:
    import prism_telemetry
except ImportError:
    # Local dev outside the operator harness: no-op stand-in, same API.
    class _TelemetryFallback:
        @staticmethod
        def report(**_kwargs):
            return None

        @staticmethod
        def finish_evaluation():
            return None

    prism_telemetry = _TelemetryFallback()


PEAK_LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
EPS = 1e-8
WARMUP_FRAC = 0.02
MIN_LR_FRAC = 0.10
GRAD_CLIP = 1.0
REPORT_EVERY = 10
# Stop training this many seconds before the harness wall cap (the cap
# clock starts at build_model, and a tripped guard fails the whole run).
WALL_MARGIN_S = 180.0


def _param_groups(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No weight decay on norms, embeddings (incl. the tied head) and
        # any <2D parameter (biases, gate/decay vectors).
        if p.ndim < 2 or "emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _lr_at(step, total_steps):
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    if step < warmup:
        return PEAK_LR * float(step + 1) / float(warmup)
    t = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return PEAK_LR * (MIN_LR_FRAC + (1.0 - MIN_LR_FRAC) * cos)


def _stream_batches(stream):
    while True:
        yield next(stream)


def _tokenizer(ctx):
    """The submitted tokenizer, via the contract — never a bare `from_pretrained`.

    `ctx["tokenizer"]` is what the harness resolved for this submission
    (recipe >= 1.3.0). This baseline declares no tokenizer of its own, so it
    gets the pinned default; under an older harness (no `ctx["tokenizer"]`)
    we ask the contract for that same default. The pin is *our* choice, not a
    challenge rule — a submission may ship `tokenizer/` files or a
    `build_tokenizer(ctx)` hook instead.
    """
    tok = ctx.get("tokenizer")
    if tok is not None:
        return tok
    from prismlib.tokenizer import load_default

    return load_default()


def _legacy_batches(ctx):
    """Pre-1.3.0 fallback: tokenize the pinned shard directly.

    Mirrors the harness stream semantics (seeded per-epoch doc shuffle,
    EOS-separated concat, (input_ids, labels) windows) so behavior stays
    comparable when `ctx["train_stream"]` is absent.
    """
    import pyarrow.parquet as pq

    device = ctx["device"]
    seed = int(ctx.get("seed", 0))
    seq_len = int(ctx.get("seq_len", 512))
    batch_size = int(ctx.get("batch_size", 8))
    tok = _tokenizer(ctx)
    table = pq.read_table(ctx["dataset_path"], columns=["text"])
    texts = [t for t in table.column("text").to_pylist() if isinstance(t, str) and len(t) >= 200]
    texts = texts[: int(ctx.get("train_rows", 2048))]
    eos = getattr(tok, "eos_token_id", None)

    epoch = 0
    while True:
        order = list(range(len(texts)))
        random.Random(seed + epoch).shuffle(order)
        epoch += 1
        buf = []
        need = batch_size * (seq_len + 1)
        for idx in order:
            ids = tok(texts[idx], add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            buf.extend(ids)
            if eos is not None:
                buf.append(eos)
            while len(buf) >= need:
                window = buf[:need]
                del buf[:need]
                t = torch.tensor(window, dtype=torch.long).view(batch_size, seq_len + 1)
                yield t[:, :-1].contiguous().to(device), t[:, 1:].contiguous().to(device)


def train(model, ctx):
    """Recipe contract entrypoint: returns a metrics dict (val is harness-side)."""
    device = ctx["device"]
    seed = int(ctx.get("seed", 0))
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    guard = ctx.get("guard") or (lambda: None)

    max_steps = int(ctx.get("max_train_steps", 20000))
    cap_s = float(ctx.get("train_hours_cap", 6.0)) * 3600.0
    stop_s = max(60.0, cap_s - WALL_MARGIN_S)

    stream = ctx.get("train_stream")
    batches = _stream_batches(stream) if stream is not None else _legacy_batches(ctx)

    opt = torch.optim.AdamW(_param_groups(model), lr=PEAK_LR, betas=BETAS, eps=EPS)
    use_amp = device == "cuda" and torch.cuda.is_available()

    model.train()
    t0 = time.time()
    step = 0
    last_loss = 0.0
    grad_norm = 0.0
    batches = iter(batches)
    # Stop conditions are checked BEFORE pulling the next batch: the harness
    # counts tokens on yielded batches (stream.tokens_seen is authoritative),
    # so an unconsumed batch must never be pulled.
    while step < max_steps and (time.time() - t0) <= stop_s:
        try:
            guard()
        except Exception:  # noqa: BLE001 — harness cap tripped; finish gracefully
            break
        input_ids, labels = next(batches)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(input_ids)
            logits = out.logits if hasattr(out, "logits") else out
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        lr = _lr_at(step, max_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        last_loss = float(loss.item())
        step += 1
        if step == 1 or step % REPORT_EVERY == 0:
            prism_telemetry.report(loss=last_loss, step=step, grad_norm=grad_norm)

    # Signal the harness to stop the eval now (raises through train() under
    # the harness; the return below only runs under the local fallback shim).
    prism_telemetry.finish_evaluation()
    return {
        "train_loss": last_loss,
        "train_steps": step,
        "train_seconds": time.time() - t0,
        "peak_lr": PEAK_LR,
        "final_lr": _lr_at(max(step - 1, 0), max_steps),
        "tokens_seen": int(getattr(stream, "tokens_seen", 0)) if stream is not None else 0,
    }
