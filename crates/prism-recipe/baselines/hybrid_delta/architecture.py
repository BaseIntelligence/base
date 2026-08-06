"""PRISM v3 reference baseline: 3:1 gated delta-net / attention hybrid.

Kimi Linear / Qwen3-Next style hybrid (see docs/spikes/prism-v3/research/03):
every 4th block is full softmax attention (sliding-window 512 + RoPE), the
other three are gated delta-rule linear-attention blocks — the delta rule
("erase then write" online regression) with per-channel forget gates and a
sigmoid output gate:

    S_t = S_{t-1} Diag(alpha_t) (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_t = S_t q_t

with S in R^{dv x dk} per head, alpha_t in (0,1)^dk (per-channel decay),
beta_t in (0,1) per head, and l2-normalized keys.

The delta recurrence is evaluated with the chunked WY-representation
algorithm (chunk size 64) in PURE torch — no FLA / mamba / Triton deps.
Within a chunk all rank-1 updates are composed in parallel via one unit
lower-triangular solve; the sequential state is carried across chunks only
(n/chunk sequential steps). The recurrence core runs in fp32 for state
stability under bf16 autocast. Peak extra memory is O(batch * heads *
chunk^2 * dk) — see NOTES.md for the slowdown analysis vs fused kernels.

Length generalization: no learned positions anywhere (RoPE in attention
blocks, recurrence elsewhere); the chunk loop and the windowed attention
handle arbitrary eval sequence lengths (G5 probes up to 32k).

Contract notes: identical to the Transformer++ baseline — `build_model(ctx)`
returns a CPU module (the harness moves it), forward returns an object with
`.logits` AND sets `self.logits`, logits cover the full GPT-2 vocab.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Anchor configuration: 341,309,696 params (see NOTES.md for the math).
DEFAULTS = {
    "vocab_size": 50257,
    "d_model": 1024,
    "n_layer": 24,
    "attn_heads": 16,
    "delta_heads": 8,
    "delta_key_dim": 128,
    "delta_value_dim": 128,
    "mlp_hidden": 2048,
    "window": 512,
    "chunk": 64,
    "conv_kernel": 4,
    "rope_theta": 50000.0,
    # Target per-token forget rate at init: alpha = exp(-0.02) ~ 0.98.
    "decay_init": 0.02,
    "init_std": 0.02,
}

_OVERRIDE_KEYS = tuple(DEFAULTS.keys())
# Per-step log-decay clamp: alpha >= e^-2 ~ 0.135 per token. Beyond this the
# channel forgets within one step anyway; the clamp bounds exp(-L/2) <= e^64
# inside a 64-token chunk so the fp32 sqrt-split factors cannot overflow.
_MAX_LOG_DECAY = 2.0


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
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(t, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class SlidingWindowAttention(nn.Module):
    """Exact sliding-window causal attention (window W) with RoPE.

    t <= W: one `is_causal` SDPA call (flash/mem-efficient backend).
    t >  W: query-blocked exact SWA — each query block of <= W positions
    attends to the key span [qs - W + 1, qe) under a small banded bool mask,
    so every query sees exactly its last W keys. Peak mask memory per block
    is W x (2W - 1) bools (~0.5 MB at W=512) — no O(t^2) materialization at
    32k eval contexts.
    """

    def __init__(self, d_model, n_head, window, rope_theta):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must divide n_head")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.window = int(window)
        self.rope_theta = float(rope_theta)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self._cos = None
        self._sin = None

    def _rope(self, q, k):
        t = q.shape[-2]
        if self._cos is None or self._cos.shape[0] < t or self._cos.device != q.device:
            cos, sin = _rope_tables(2 * t, self.head_dim, self.rope_theta, q.device, q.dtype)
            self._cos, self._sin = cos, sin
        return _apply_rope(q, self._cos[:t], self._sin[:t]), _apply_rope(
            k, self._cos[:t], self._sin[:t]
        )

    def _windowed(self, q, k, v):
        b, h, t, hd = q.shape
        w = self.window
        outs = []
        for qs in range(0, t, w):
            qe = min(qs + w, t)
            k0 = max(0, qs - w + 1)
            qb = q[:, :, qs:qe]
            kb = k[:, :, k0:qe]
            vb = v[:, :, k0:qe]
            qi = qs + torch.arange(qe - qs, device=q.device)[:, None]
            kj = k0 + torch.arange(qe - k0, device=q.device)[None, :]
            mask = (kj <= qi) & (kj > qi - w)  # True = attend
            outs.append(F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask))
        return torch.cat(outs, dim=2)

    def forward(self, x):
        b, t, d = x.shape
        q = self.wq(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self._rope(q, k)
        if t <= self.window:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = self._windowed(q, k, v)
        o = o.transpose(1, 2).reshape(b, t, d)
        return self.wo(o)


def _causal_depthwise_conv(x, weight):
    """x (b, t, c) -> causal depthwise conv1d -> (b, t, c)."""
    k = weight.shape[-1]
    y = F.conv1d(x.transpose(1, 2), weight, padding=k - 1, groups=weight.shape[0])
    return y[..., : x.shape[1]].transpose(1, 2)


class GatedDeltaMixer(nn.Module):
    """Gated delta-rule linear attention (per-channel decay + output gate)."""

    def __init__(self, d_model, n_head, key_dim, value_dim, chunk, conv_kernel, decay_init):
        super().__init__()
        self.n_head = n_head
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.chunk = int(chunk)
        self.wq = nn.Linear(d_model, n_head * key_dim, bias=False)
        self.wk = nn.Linear(d_model, n_head * key_dim, bias=False)
        self.wv = nn.Linear(d_model, n_head * value_dim, bias=False)
        self.wa = nn.Linear(d_model, n_head * key_dim, bias=False)
        self.wbeta = nn.Linear(d_model, n_head, bias=False)
        self.wgate = nn.Linear(d_model, n_head * value_dim, bias=False)
        self.wo = nn.Linear(n_head * value_dim, d_model, bias=False)
        self.conv_q = nn.Parameter(torch.zeros(n_head * key_dim, 1, conv_kernel))
        self.conv_k = nn.Parameter(torch.zeros(n_head * key_dim, 1, conv_kernel))
        self.conv_v = nn.Parameter(torch.zeros(n_head * value_dim, 1, conv_kernel))
        # softplus(a_bias) = decay_init  ->  alpha ~ exp(-decay_init) at init.
        self.a_bias = nn.Parameter(
            torch.full((n_head * key_dim,), math.log(math.expm1(float(decay_init))))
        )
        self.head_norm = nn.Parameter(torch.ones(value_dim))

    def _chunked_delta(self, q, k, v, beta, la):
        """Chunked gated delta rule (WY representation), fp32.

        q/k: (b, h, t, dk); v: (b, h, t, dv); beta: (b, h, t);
        la: (b, h, t, dk) log-decay <= 0. Returns o: (b, h, t, dv).

        Per chunk (length c, cumulative log-decay L = cumsum(la), and the
        masked pairwise decay tensor Dec[t, i] = e^{L_t - L_i} for i <= t):
          A    = tril_{-1}( beta * sum_c k_t k_i Dec[t, i] )
          U    = (I + A)^{-1} diag(beta) (V - (k e^L) S^T)
          O    = (q e^L) S^T + (sum_c q_t k_i Dec[t, i]) U
          S   <- S Diag(e^{L_c}) + U^T (k e^{L_c - L})
        The pairwise tensor is materialized per chunk (never factored as
        e^{L_t} * e^{-L_i}, which can overflow fp32): every entry is <= 1,
        masked to 0 above the diagonal via -inf before exp. Peak extra
        memory is O(b * h * chunk^2 * dk) — see NOTES.md.
        """
        in_dtype = q.dtype
        q, k, v, beta, la = (t_.float() for t_ in (q, k, v, beta, la))
        b, h, t, dk = q.shape
        dv = v.shape[-1]
        c_chunk = self.chunk
        state = q.new_zeros(b, h, dv, dk)
        outs = []
        for s in range(0, t, c_chunk):
            e = min(s + c_chunk, t)
            qc = q[:, :, s:e]
            kc = k[:, :, s:e]
            vc = v[:, :, s:e]
            bc = beta[:, :, s:e]
            c = e - s
            L = la[:, :, s:e].cumsum(dim=2)
            # Pairwise decayed inner products, masked i <= t (no overflow:
            # masked entries are -inf -> exp -> 0, live entries are <= 0).
            Ldiff = L[:, :, :, None, :] - L[:, :, None, :, :]  # (b,h,c,c,dk)
            tril_mask = torch.ones(c, c, dtype=torch.bool, device=q.device).tril()
            dec = Ldiff.masked_fill(~tril_mask[..., None], float("-inf")).exp()
            A = bc.unsqueeze(-1) * torch.einsum("bhtc,bhic,bhtic->bhti", kc, kc, dec)
            A = A.tril(-1)
            Bm = torch.einsum("bhtc,bhic,bhtic->bhti", qc, kc, dec)
            eye = torch.eye(c, device=q.device, dtype=q.dtype)
            rhs = bc.unsqueeze(-1) * (vc - (kc * L.exp()) @ state.transpose(-1, -2))
            U = torch.linalg.solve_triangular(
                A + eye, rhs, upper=False, unitriangular=True
            )
            o = (qc * L.exp()) @ state.transpose(-1, -2) + Bm @ U
            e_lc = L[:, :, -1:, :].exp()
            k_tail = kc * (L[:, :, -1:, :] - L).exp()
            state = state * e_lc + U.transpose(-1, -2) @ k_tail
            outs.append(o)
        return torch.cat(outs, dim=2).to(in_dtype)

    def forward(self, x):
        b, t, _ = x.shape
        h, dk, dv = self.n_head, self.key_dim, self.value_dim
        q = F.silu(_causal_depthwise_conv(self.wq(x), self.conv_q))
        k = F.silu(_causal_depthwise_conv(self.wk(x), self.conv_k))
        v = F.silu(_causal_depthwise_conv(self.wv(x), self.conv_v))
        q = q.view(b, t, h, dk).transpose(1, 2)
        k = k.view(b, t, h, dk).transpose(1, 2)
        v = v.view(b, t, h, dv).transpose(1, 2)
        k = F.normalize(k, p=2, dim=-1)  # unit keys: delta-rule convention
        beta = torch.sigmoid(self.wbeta(x)).transpose(1, 2)  # (b, h, t)
        la = -F.softplus(self.wa(x) + self.a_bias).view(b, t, h, dk).transpose(1, 2)
        la = la.clamp(min=-_MAX_LOG_DECAY)
        o = self._chunked_delta(q, k, v, beta, la)
        o = o.transpose(1, 2)  # (b, t, h, dv)
        o = F.rms_norm(o, (dv,), weight=self.head_norm, eps=1e-6)
        o = o.reshape(b, t, h * dv)
        o = o * torch.sigmoid(self.wgate(x))
        return self.wo(o)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeltaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = int(cfg["d_model"])
        self.norm1 = RMSNorm(d)
        self.mixer = GatedDeltaMixer(
            d,
            int(cfg["delta_heads"]),
            int(cfg["delta_key_dim"]),
            int(cfg["delta_value_dim"]),
            int(cfg["chunk"]),
            int(cfg["conv_kernel"]),
            float(cfg["decay_init"]),
        )
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, int(cfg["mlp_hidden"]))

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttnBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = int(cfg["d_model"])
        self.norm1 = RMSNorm(d)
        self.attn = SlidingWindowAttention(
            d, int(cfg["attn_heads"]), int(cfg["window"]), float(cfg["rope_theta"])
        )
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, int(cfg["mlp_hidden"]))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HybridDelta(nn.Module):
    """3:1 delta/attention stack: blocks 3, 7, 11, ... (0-indexed) attend."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d = int(cfg["d_model"])
        n_layer = int(cfg["n_layer"])
        self.tok_emb = nn.Embedding(int(cfg["vocab_size"]), d)
        self.blocks = nn.ModuleList(
            AttnBlock(cfg) if (i + 1) % 4 == 0 else DeltaBlock(cfg) for i in range(n_layer)
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
                if name.endswith(("wo.weight", "w2.weight")):
                    nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * n_layer))
                else:
                    nn.init.normal_(p, mean=0.0, std=std)
            # norms stay at 1.0; a_bias keeps its decay-target init

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
    return cfg


def build_model(ctx):
    """Recipe contract entrypoint. Defaults are the ~341M anchor config.

    Optional overrides (test/tiny profiles): top-level ctx keys or an
    `arch` dict with any of DEFAULTS' keys.
    """
    ctx = ctx if isinstance(ctx, dict) else {}
    torch.manual_seed(int(ctx.get("seed", 0)))
    return HybridDelta(_config_from_ctx(ctx))
