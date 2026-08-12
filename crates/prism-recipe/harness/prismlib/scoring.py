"""Harness-owned frozen-val scoring (v1 semantics, unchanged).

Runs inside the miner subprocess **after** `train()` returns — the harness
drives the val loop, miner code never touches it. Mean teacher-forced CE
over the frozen val cut -> the historical `bpb` = CE / ln 2. Miner
architectures may self-truncate to a shorter context: the target window is
aligned to the positions the model actually scored (last t logits).

Unit note (modular tokenizer): `bpb` is bits per **token** of the submitted
tokenizer and stays the v1/v2 scoring key, bit-identical for a submission
that keeps the pinned default. The additional `bits_per_byte` is total bits
over the UTF-8 bytes actually scored — the tokenizer-neutral number to
compare submissions that chose different vocabularies.
"""

from . import LN2
from .tokenizer import encode_tensor


def val_ce_bpb(model, tok, val_texts, device):
    """`(ce, bpb_per_token, n_tokens, bits_per_byte)` over the frozen val cut."""
    import torch

    model.eval()
    losses = []
    n_tokens = 0
    bits, scored_bytes = 0.0, 0.0
    with torch.no_grad():
        for txt in val_texts:
            ids = encode_tensor(tok, txt, device)
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
            n_scored = tgt.numel()
            n_tokens += n_scored
            bits += loss.item() * n_scored / LN2
            # Self-truncating models score only the tail: prorate the doc's
            # UTF-8 length by the fraction of targets actually scored.
            scored_bytes += len(txt.encode("utf-8", errors="replace")) * (
                n_scored / float(max(1, ids.shape[1] - 1))
            )
    if not losses:
        raise RuntimeError("no scored tokens")
    ce = sum(losses) / len(losses)
    return ce, ce / LN2, n_tokens, bits / max(1.0, scored_bytes)
