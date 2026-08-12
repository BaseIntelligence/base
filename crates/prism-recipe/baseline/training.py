"""PRISM baseline training (recipe v1).

Reads `ctx["train_rows"]` texts from the pinned shard, runs AdamW at a modest
LR for a small fixed step budget (baseline is deliberately mediocre: it is
the calibration anchor, not a competitive recipe).

Telemetry hook contract (recipe >= 1.1.0): the harness provides a
`prism_telemetry` module (also at `ctx["telemetry"]`). Call
`report(loss=..., step=..., grad_norm=..., layer_stats=...)`
periodically and `finish_evaluation()` to end the eval early — the harness
scores the in-memory model either way.

Tokenizer contract (recipe >= 1.3.0): use `ctx["tokenizer"]`. It is the
tokenizer *this submission* declared (`tokenizer/` files or a
`build_tokenizer(ctx)` hook) or the pinned fallback when it declares none,
as here. Never `from_pretrained` a name yourself: the training subprocess has
no network, and the eval phase re-resolves the same tokenizer from the
submission.
"""

import time

import pyarrow.parquet as pq
import torch

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


def _texts(path, n):
    table = pq.read_table(path, columns=["text"])
    xs = [t for t in table.column("text").to_pylist() if isinstance(t, str) and len(t) >= 200]
    return xs[:n]


def _tokenizer(ctx):
    """`ctx["tokenizer"]`, or the contract's pinned fallback (old harness)."""
    tok = ctx.get("tokenizer")
    if tok is None:
        from prismlib.tokenizer import load_default

        tok = load_default()
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    return tok


def train(model, ctx):
    """Recipe contract entrypoint: returns a metrics dict (val is harness-side)."""
    device = ctx["device"]
    torch.manual_seed(int(ctx["seed"]))
    guard = ctx.get("guard", lambda: None)

    tok = _tokenizer(ctx)
    texts = _texts(ctx["dataset_path"], int(ctx.get("train_rows", 2048)))
    block = model.block if hasattr(model, "block") else 512

    g = torch.Generator().manual_seed(int(ctx["seed"]))
    perm = torch.randperm(len(texts), generator=g).tolist()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()
    steps = 0
    bs = 4
    last = 0.0
    t0 = time.time()
    for i in range(0, min(2000, len(perm) - bs), bs):
        guard()
        batch_txt = [texts[j] for j in perm[i : i + bs]]
        enc = tok(
            batch_txt, return_tensors="pt", truncation=True, max_length=block, padding=True
        ).to(device)
        ids = enc.input_ids
        out = model(ids[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=tok.pad_token_id
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        last = float(loss.item())
        steps += 1
        if steps == 1 or steps % 10 == 0:
            prism_telemetry.report(loss=last, step=steps, grad_norm=grad_norm)
        if steps >= int(ctx.get("max_train_steps", 20000)):
            break
    # Signal the harness to stop the eval now (raises through train() under
    # the harness; the return below only runs under the local fallback shim).
    prism_telemetry.finish_evaluation()
    return {
        "train_loss": last,
        "train_steps": steps,
        "train_seconds": time.time() - t0,
    }
