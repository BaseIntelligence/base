"""Harness-owned frozen-val scoring (v1 semantics, unchanged).

Runs inside the miner subprocess **after** `train()` returns — the harness
drives the val loop, miner code never touches it. Mean teacher-forced CE
over the frozen val cut -> bpb = CE / ln 2. Miner architectures may
self-truncate to a shorter context: the target window is aligned to the
positions the model actually scored (last t logits).
"""

from . import LN2


def val_ce_bpb(model, tok, val_texts, device):
    import torch

    model.eval()
    losses = []
    n_tokens = 0
    with torch.no_grad():
        for txt in val_texts:
            ids = tok(txt, return_tensors="pt").input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            out = model(ids[:, :-1])
            logits = out.logits if hasattr(out, "logits") else out
            if logits.shape[1] < 1:
                continue
            tgt = ids[:, 1:][:, -logits.shape[1] :]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt.reshape(-1),
                reduction="mean",
            )
            losses.append(loss.item())
            n_tokens += tgt.numel()
    if not losses:
        raise RuntimeError("no scored tokens")
    ce = sum(losses) / len(losses)
    return ce, ce / LN2, n_tokens
