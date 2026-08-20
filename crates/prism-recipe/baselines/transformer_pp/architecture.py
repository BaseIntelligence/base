"""PRISM v3 reference baseline: Transformer++ (recipe >= 1.3.0).

Modern GPT, ~341M params at the default config (hard cap 1B, enforced by
the harness after `build_model`; the reference geometry is unchanged by the
cap raise — the headroom is for miners):

- pre-norm RMSNorm, no biases anywhere, tied input/output embeddings
- RoPE (GPT-NeoX half-split convention, theta=50000) — no learned position
  table, so eval works at any sequence length (the G5 battery probes up to
  32k; training is at ctx["seq_len"]=512)
- SwiGLU MLP, hidden = 2.5 * d_model (2560 at d=1024)
- full causal attention via `F.scaled_dot_product_attention(is_causal=True)`
  (memory-efficient at long context, no O(n^2) mask materialization)

Contract notes:
- `build_model(ctx)` returns an `nn.Module` on CPU; the harness moves it to
  `ctx["device"]` itself. Default config is the anchor; optional width/depth
  overrides are read from ctx (top-level keys or an `arch` dict) for
  test/tiny profiles.
- `forward(ids)` returns an object with `.logits` AND sets `self.logits`
  (both harness read patterns: `out.logits if hasattr(out, "logits") else
  out`). Logits cover the whole vocab at every input position; no
  self-truncation.
- the tokenizer is part of a submission, so the embedding is sized from
  `ctx["vocab_size"]` (the harness reports the resolved tokenizer's vocab).
  The 50257 default below is the pinned fallback this baseline chooses, and
  the figure `count_params.py` / NOTES.md do the cap math against.

Pure stdlib + torch. Deterministic init under `ctx["seed"]`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Anchor configuration: 340,920,320 params (see NOTES.md for the math).
DEFAULTS = {
    "vocab_size": 50257,  # overridden by ctx["vocab_size"] (submitted tokenizer)
    "d_model": 1024,
    "n_layer": 24,
    "n_head": 16,
    "mlp_hidden": 2560,
    "rope_theta": 50000.0,
    "init_std": 0.02,
}

_OVERRIDE_KEYS = tuple(DEFAULTS.keys())


class ModelOutput:
    """Minimal output carrier (harness reads `out.logits`)."""

    __slots__ = ("logits",)

    def __init__(self, logits):
        self.logits = logits


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), weight=self.weight, eps=self.eps)


def _rope_tables(t, head_dim, theta, device, dtype):
    """cos/sin of shape (t, head_dim // 2), NeoX (half-split) convention."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(t, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x, cos, sin):
    """x: (b, h, t, hd); cos/sin: (t, hd//2) -> rotated, same shape."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, rope_theta):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must divide n_head")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.rope_theta = float(rope_theta)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self._cos = None
        self._sin = None

    def _rope(self, q, k):
        t = q.shape[-2]
        if (
            self._cos is None
            or self._cos.shape[0] < t
            or self._cos.device != q.device
        ):
            # Cache double the requested length (amortized growth on
            # growing eval contexts); computed in fp32, stored in q.dtype.
            cos, sin = _rope_tables(2 * t, self.head_dim, self.rope_theta, q.device, q.dtype)
            self._cos, self._sin = cos, sin
        return _apply_rope(q, self._cos[:t], self._sin[:t]), _apply_rope(
            k, self._cos[:t], self._sin[:t]
        )

    def forward(self, x):
        b, t, d = x.shape
        q = self.wq(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self._rope(q, k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(b, t, d)
        return self.wo(o)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d_model, n_head, mlp_hidden, rope_theta):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, rope_theta)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerPP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d = int(cfg["d_model"])
        self.tok_emb = nn.Embedding(int(cfg["vocab_size"]), d)
        self.blocks = nn.ModuleList(
            Block(d, int(cfg["n_head"]), int(cfg["mlp_hidden"]), float(cfg["rope_theta"]))
            for _ in range(int(cfg["n_layer"]))
        )
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, int(cfg["vocab_size"]), bias=False)
        self.head.weight = self.tok_emb.weight  # tied embeddings
        self.logits = None
        self._init_weights(float(cfg["init_std"]))

    def _init_weights(self, std):
        n_layer = len(self.blocks)
        for name, p in self.named_parameters():
            if p.ndim >= 2:
                # Residual-branch outputs get the GPT-2 depth scaling.
                if name.endswith(("wo.weight", "w2.weight")):
                    nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * n_layer))
                else:
                    nn.init.normal_(p, mean=0.0, std=std)
            # norms stay at 1.0

    def forward(self, ids):
        x = self.tok_emb(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        self.logits = logits
        return ModelOutput(logits)


def _config_from_ctx(ctx):
    cfg = dict(DEFAULTS)
    if isinstance(ctx, dict):
        overrides = ctx.get("arch")
        if isinstance(overrides, dict):
            cfg.update({k: v for k, v in overrides.items() if k in cfg})
        for k in _OVERRIDE_KEYS:
            if k in ctx:
                cfg[k] = ctx[k]
        # G8 µP LR-transfer sweep: scale width dims so 4× exceeds 1.5× params.
        # Default 1.0 leaves the anchor config unchanged (~341M, ≤1B cap).
        mult = float(ctx.get("prism_width_multiplier", 1.0) or 1.0)
        if abs(mult - 1.0) > 1e-12:
            if mult <= 0:
                raise ValueError("prism_width_multiplier must be > 0")
            cfg["d_model"] = max(1, int(round(int(cfg["d_model"]) * mult)))
            cfg["mlp_hidden"] = max(1, int(round(int(cfg["mlp_hidden"]) * mult)))
            # Keep head_dim constant when possible so n_head scales with width.
            n_head = int(cfg["n_head"])
            head_dim = int(DEFAULTS["d_model"]) // int(DEFAULTS["n_head"])
            if head_dim > 0 and cfg["d_model"] % head_dim == 0:
                cfg["n_head"] = cfg["d_model"] // head_dim
            elif cfg["d_model"] % n_head != 0:
                # Fall back: clamp n_head to a divisor of d_model.
                h = min(n_head, cfg["d_model"])
                while h > 1 and cfg["d_model"] % h != 0:
                    h -= 1
                cfg["n_head"] = h
    return cfg


def build_model(ctx):
    """Recipe contract entrypoint. Defaults are the ~341M anchor config.

    Optional overrides (test/tiny profiles): top-level ctx keys or an
    `arch` dict with any of DEFAULTS' keys (d_model, n_layer, n_head,
    mlp_hidden, vocab_size, rope_theta, init_std).

    Honors `ctx["prism_width_multiplier"]` (G8 µP sweep): scales d_model /
    mlp_hidden (and n_head to keep head_dim) so a 4× build exceeds 1.5×
    base params. Multiplier 1.0 (default) leaves the anchor unchanged.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    torch.manual_seed(int(ctx.get("seed", 0)))
    return TransformerPP(_config_from_ctx(ctx))
