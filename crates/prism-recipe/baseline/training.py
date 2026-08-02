"""PRISM baseline training (recipe v1).

Reads `ctx["train_rows"]` texts from the pinned shard, runs AdamW at a modest
LR for a small fixed step budget (baseline is deliberately mediocre: it is
the calibration anchor, not a competitive recipe).
"""

import time

import pyarrow.parquet as pq
import torch
from transformers import GPT2TokenizerFast


def _texts(path, n):
    table = pq.read_table(path, columns=["text"])
    xs = [t for t in table.column("text").to_pylist() if isinstance(t, str) and len(t) >= 200]
    return xs[:n]


def train(model, ctx):
    """Recipe contract entrypoint: returns a metrics dict (val is harness-side)."""
    device = ctx["device"]
    torch.manual_seed(int(ctx["seed"]))
    guard = ctx.get("guard", lambda: None)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
        steps += 1
        if steps >= int(ctx.get("max_train_steps", 20000)):
            break
    return {
        "train_loss": last,
        "train_steps": steps,
        "train_seconds": time.time() - t0,
    }
