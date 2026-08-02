"""PRISM baseline architecture (recipe v1).

Tiny GPT-style causal transformer (~12M params) — the reference submission
against which LLM similarity review calibrates "copied".
"""

import torch
import torch.nn as nn


class TinyGPT(nn.Module):
    def __init__(self, vocab=50257, d=256, n_layer=4, n_head=4, block=512):
        super().__init__()
        self.block = block
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(block, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_head, dim_feedforward=4 * d, batch_first=True,
            norm_first=True, dropout=0.0, activation="gelu",
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # tied
        self.logits = None

    def forward(self, ids):
        b, t = ids.shape
        t = min(t, self.block)
        ids = ids[:, -t:]
        pos = torch.arange(t, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None, :, :]
        causal = torch.triu(
            torch.full((t, t), float("-inf"), device=ids.device), diagonal=1
        )
        x = self.tr(x, mask=causal)
        x = self.ln(x)
        logits = self.head(x)      # (b, t, vocab)
        self.logits = logits
        return logits


def build_model(ctx):
    """Recipe contract entrypoint. ctx carries device/seed/caps (unused here)."""
    torch.manual_seed(int(ctx.get("seed", 0)))
    return TinyGPT()
