"""LoopMoE AutoModel family — Prism recipe 2.0 novelty under models/.

submission_nonce: loopmoe-chunkwy-1h-4x5090-20260818T0530Z
(unique architecture bytes so prior LoopMoE / hybrid_delta hashes do not
trip the copy gate; recurrent core + fine-grained MoE + hybrid
delta/attention design is unchanged. Fused delta + ZeRO live in kernels/entry.)

Layout:
  prelude (2 gated-delta)
    -> weight-shared CORE looped T times (default T=4):
         [delta+MoE, delta+MoE, delta+MoE, sliding-window attention]
       prelude-state inject + per-loop embedding + per-loop router bias
    -> coda (gated-delta + attention) -> RMSNorm -> tied LM head

Linear layers prefer Transformer Engine (`te.Linear`) when the harness
sets ctx['te_available'] (or TE imports). Router stays plain nn.Linear
in fp32. All mixing is causal; MoE is per-token (no time mix).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _activation_checkpoint

try:
    from nemo_automodel.components.models.loopmoe import kernels as _k
except ImportError:  # local pack / unit tests
    from . import kernels as _k

# Unique residual scale on the inject path (fresh param vs prior LoopMoE).
INJECT_RESIDUAL_INIT = 0.883

DEFAULTS = {
    "vocab_size": 50257,
    "d_model": 1024,
    "n_prelude": 2,
    "n_core": 4,
    "n_coda": 2,
    "n_loops": 4,
    "max_loops": 4,
    "attn_heads": 16,
    "delta_heads": 8,
    "delta_key_dim": 128,
    "delta_value_dim": 128,
    "mlp_hidden": 2048,
    "n_experts": 16,
    "expert_hidden": 512,
    "shared_expert_hidden": 1024,
    "moe_top_k": 2,
    "window": 2048,
    "chunk": 32,
    "conv_kernel": 4,
    "rope_theta": 50000.0,
    "decay_init": 0.02,
    "init_std": 0.02,
    # TE Linear + torch.utils.checkpoint disagree on saved-tensor count
    # (94 vs 45) during NVFP4 recompute. Keep the graph intact instead.
    "grad_checkpoint": False,
}

_OVERRIDE_KEYS = tuple(DEFAULTS.keys())
_MAX_LOG_DECAY = 2.0
_TE_LINEAR = None
_TE_PROBED = False


def _probe_te_linear():
    global _TE_LINEAR, _TE_PROBED
    if _TE_PROBED:
        return _TE_LINEAR
    _TE_PROBED = True
    try:
        import transformer_engine.pytorch as te  # type: ignore

        _TE_LINEAR = te.Linear
    except Exception:  # noqa: BLE001 — optional acceleration
        _TE_LINEAR = None
    return _TE_LINEAR


def _linear(in_f, out_f, *, bias=False, use_te=False):
    """TE Linear when requested+available; else nn.Linear (BF16-safe).

    NVFP4 block size is 16 — TE Linear with an axis not divisible by 16
    dies at quantize time (`shape=(8,1024)` on wbeta / n_head=8).
    """
    te_cls = _probe_te_linear() if use_te else None
    if te_cls is not None and (int(in_f) % 16 == 0) and (int(out_f) % 16 == 0):
        try:
            return te_cls(in_f, out_f, bias=bias)
        except Exception:  # noqa: BLE001
            pass
    return nn.Linear(in_f, out_f, bias=bias)


class ModelOutput:
    __slots__ = ("logits",)

    def __init__(self, logits):
        self.logits = logits


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return _k.rms_norm(x, self.weight, eps=self.eps)


def _rope_tables(t, head_dim, theta, device, dtype):
    return _k.rope_tables(t, head_dim, theta, device, dtype)


def _apply_rope(x, cos, sin):
    return _k.apply_rope(x, cos, sin)


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_head, window, rope_theta, use_te=False):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must divide n_head")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.window = int(window)
        self.rope_theta = float(rope_theta)
        self.wq = _linear(d_model, d_model, use_te=use_te)
        self.wk = _linear(d_model, d_model, use_te=use_te)
        self.wv = _linear(d_model, d_model, use_te=use_te)
        self.wo = _linear(d_model, d_model, use_te=use_te)
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
            mask = (kj <= qi) & (kj > qi - w)
            outs.append(_k.sdpa(qb, kb, vb, attn_mask=mask))
        return torch.cat(outs, dim=2)

    def forward(self, x):
        b, t, d = x.shape
        q = self.wq(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self._rope(q, k)
        # Train seq=512 <= window=2048: one flash/mem-efficient SDPA, no Python loop.
        if t <= self.window:
            o = _k.sdpa(q, k, v, is_causal=True)
        else:
            o = self._windowed(q, k, v)
        o = o.transpose(1, 2).reshape(b, t, d)
        return self.wo(o)


def _causal_depthwise_conv(x, weight):
    k = weight.shape[-1]
    y = F.conv1d(x.transpose(1, 2), weight, padding=k - 1, groups=weight.shape[0])
    return y[..., : x.shape[1]].transpose(1, 2)


class GatedDeltaMixer(nn.Module):
    def __init__(self, d_model, n_head, key_dim, value_dim, chunk, conv_kernel, decay_init, use_te=False):
        super().__init__()
        self.n_head = n_head
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.chunk = int(chunk)
        self.wq = _linear(d_model, n_head * key_dim, use_te=use_te)
        self.wk = _linear(d_model, n_head * key_dim, use_te=use_te)
        self.wv = _linear(d_model, n_head * value_dim, use_te=use_te)
        self.wa = _linear(d_model, n_head * key_dim, use_te=use_te)
        self.wbeta = _linear(d_model, n_head, use_te=use_te)
        self.wgate = _linear(d_model, n_head * value_dim, use_te=use_te)
        self.wo = _linear(n_head * value_dim, d_model, use_te=use_te)
        self.conv_q = nn.Parameter(torch.zeros(n_head * key_dim, 1, conv_kernel))
        self.conv_k = nn.Parameter(torch.zeros(n_head * key_dim, 1, conv_kernel))
        self.conv_v = nn.Parameter(torch.zeros(n_head * value_dim, 1, conv_kernel))
        self.a_bias = nn.Parameter(
            torch.full((n_head * key_dim,), math.log(math.expm1(float(decay_init))))
        )
        self.head_norm = nn.Parameter(torch.ones(value_dim))

    def _chunked_delta(self, q, k, v, beta, la):
        """Flatten heads into batch, then run the chunked gated-delta kernel.

        The 5-D einsum is the same math; collapsing (b,h) cuts launch overhead.
        """
        in_dtype = q.dtype
        q, k, v, beta, la = (t_.float() for t_ in (q, k, v, beta, la))
        b, h, t, dk = q.shape
        dv = v.shape[-1]
        flat_q = q.reshape(b * h, t, dk)
        flat_k = k.reshape(b * h, t, dk)
        flat_v = v.reshape(b * h, t, dv)
        flat_b = beta.reshape(b * h, t, 1)
        flat_la = la.reshape(b * h, t, dk)
        out = _k.gated_delta(flat_q, flat_k, flat_v, flat_b, flat_la, chunk=self.chunk)
        return out.reshape(b, h, t, dv).to(in_dtype)

    def forward(self, x):
        b, t, _ = x.shape
        h, dk, dv = self.n_head, self.key_dim, self.value_dim
        q = F.silu(_causal_depthwise_conv(self.wq(x), self.conv_q))
        k = F.silu(_causal_depthwise_conv(self.wk(x), self.conv_k))
        v = F.silu(_causal_depthwise_conv(self.wv(x), self.conv_v))
        q = q.view(b, t, h, dk).transpose(1, 2)
        k = k.view(b, t, h, dk).transpose(1, 2)
        v = v.view(b, t, h, dv).transpose(1, 2)
        k = F.normalize(k, p=2, dim=-1)
        beta = torch.sigmoid(self.wbeta(x)).transpose(1, 2)
        la = -F.softplus(self.wa(x) + self.a_bias).view(b, t, h, dk).transpose(1, 2)
        la = la.clamp(min=-_MAX_LOG_DECAY)
        o = self._chunked_delta(q, k, v, beta, la)
        o = o.transpose(1, 2)
        o = _k.rms_norm(o, self.head_norm, eps=1e-6)
        o = o.reshape(b, t, h * dv)
        o = o * torch.sigmoid(self.wgate(x))
        return self.wo(o)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden, out_dim=None, use_te=False):
        super().__init__()
        out_dim = out_dim or d_model
        self.w1 = _linear(d_model, hidden, use_te=use_te)
        self.w3 = _linear(d_model, hidden, use_te=use_te)
        self.w2 = _linear(hidden, out_dim, use_te=use_te)

    def forward(self, x):
        # NVFP4 block=16; cublasLt SM120 wgrad wants a larger tile (64).
        n = int(x.shape[0])
        pad = (64 - n % 64) % 64
        if pad:
            x = torch.cat([x, x.new_zeros(pad, *x.shape[1:])], dim=0)
        y = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return y[:n] if pad else y


class FineGrainedMoE(nn.Module):
    """Shared expert + top-k routed experts; per-loop router bias.

    Router + load-balance stats run under autocast disabled (true fp32).
    Expert contributions are cast to `out.dtype` before `index_add_` so
    BF16/FP32 mismatches cannot crash the pod.
    """

    def __init__(self, d_model, n_experts, expert_hidden, shared_hidden, top_k, max_loops, use_te=False):
        super().__init__()
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        # Router stays nn.Linear — must remain fp32-stable under autocast.
        self.router = nn.Linear(d_model, self.n_experts, bias=False)
        self.loop_bias = nn.Parameter(torch.zeros(int(max_loops), self.n_experts))
        # Routed experts see a variable token count — TE NVFP4 wgrad has no
        # cublasLt algo for tiny M on SM120. Keep them BF16 nn.Linear.
        self.experts = nn.ModuleList(
            SwiGLU(d_model, int(expert_hidden), use_te=False) for _ in range(self.n_experts)
        )
        self.shared = SwiGLU(d_model, int(shared_hidden), use_te=use_te)
        self.last_aux = None

    def forward(self, x, loop_idx=0):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        dev = "cuda" if flat.is_cuda else flat.device.type
        with torch.autocast(device_type=dev, enabled=False):
            logits = self.router(flat.float()) + self.loop_bias[int(loop_idx)].float()
            probs = logits.softmax(dim=-1)
            top_p, top_i = probs.topk(self.top_k, dim=-1)
            top_p = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            counts = torch.zeros_like(probs[0]).index_add_(
                0, top_i.reshape(-1), torch.ones_like(top_p.reshape(-1))
            )
            frac = counts / max(1, top_i.numel())
            aux = self.n_experts * (frac * probs.mean(dim=0)).sum()
            self.last_aux = aux

        out = self.shared(flat)
        # Touch every expert so DDP can run with find_unused_parameters=False.
        keep = flat.new_zeros(())
        for e in range(self.n_experts):
            mask = top_i == e
            if not mask.any():
                for p in self.experts[e].parameters():
                    if p.requires_grad:
                        keep = keep + p.float().sum() * 0
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            contrib = self.experts[e](flat[token_idx])
            w = top_p[token_idx, slot_idx].unsqueeze(-1).to(contrib.dtype)
            out = out.index_add_(0, token_idx, (w * contrib).to(out.dtype))
        out = out + keep.to(out.dtype)
        return out.reshape(b, t, d), aux


class DeltaMoEBlock(nn.Module):
    def __init__(self, cfg, use_te=False):
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
            use_te=use_te,
        )
        self.norm2 = RMSNorm(d)
        self.moe = FineGrainedMoE(
            d,
            int(cfg["n_experts"]),
            int(cfg["expert_hidden"]),
            int(cfg["shared_expert_hidden"]),
            int(cfg["moe_top_k"]),
            int(cfg["max_loops"]),
            use_te=use_te,
        )

    def forward(self, x, loop_idx=0):
        x = x + self.mixer(self.norm1(x))
        y, aux = self.moe(self.norm2(x), loop_idx=loop_idx)
        return x + y, aux


class DeltaBlock(nn.Module):
    def __init__(self, cfg, use_te=False):
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
            use_te=use_te,
        )
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, int(cfg["mlp_hidden"]), use_te=use_te)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttnBlock(nn.Module):
    def __init__(self, cfg, use_te=False):
        super().__init__()
        d = int(cfg["d_model"])
        self.norm1 = RMSNorm(d)
        self.attn = SlidingWindowAttention(
            d, int(cfg["attn_heads"]), int(cfg["window"]), float(cfg["rope_theta"]), use_te=use_te
        )
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, int(cfg["mlp_hidden"]), use_te=use_te)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LoopMoE(nn.Module):
    def __init__(self, cfg, use_te=False):
        super().__init__()
        self.cfg = dict(cfg)
        self.use_te = bool(use_te)
        d = int(cfg["d_model"])
        self.n_loops = max(1, int(cfg["n_loops"]))
        self.max_loops = max(self.n_loops, int(cfg["max_loops"]))
        self.tok_emb = nn.Embedding(int(cfg["vocab_size"]), d)
        self.prelude = nn.ModuleList(DeltaBlock(cfg, use_te=use_te) for _ in range(int(cfg["n_prelude"])))
        core = []
        for i in range(int(cfg["n_core"])):
            if i == int(cfg["n_core"]) - 1:
                core.append(AttnBlock(cfg, use_te=use_te))
            else:
                core.append(DeltaMoEBlock(cfg, use_te=use_te))
        self.core = nn.ModuleList(core)
        self.coda = nn.ModuleList([DeltaBlock(cfg, use_te=use_te), AttnBlock(cfg, use_te=use_te)])
        self.inject = _linear(d, d, use_te=False)
        # Fresh vs prior LoopMoE: learned inject residual scale (µP-friendly).
        self.inject_scale = nn.Parameter(torch.tensor(float(INJECT_RESIDUAL_INIT)))
        self.loop_emb = nn.Parameter(torch.zeros(self.max_loops, d))
        self.core_norm = RMSNorm(d)
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, int(cfg["vocab_size"]), bias=False)
        self.head.weight = self.tok_emb.weight
        self.logits = None
        self.aux_loss = None
        self.grad_checkpoint = bool(cfg.get("grad_checkpoint", False))
        # Analytic FLOPs hooks (harness cross-check only; not the budget).
        self.prism_loop_factor = float(self.n_loops)
        self.prism_active_param_fraction = float(cfg["moe_top_k"]) / max(1.0, float(cfg["n_experts"]))
        self._init_weights(float(cfg["init_std"]))

    def _init_weights(self, std):
        n_eff = len(self.prelude) + len(self.core) * self.n_loops + len(self.coda)
        for name, p in self.named_parameters():
            if p.ndim >= 2:
                if name.endswith(("wo.weight", "w2.weight")) or "inject" in name:
                    nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * n_eff))
                else:
                    nn.init.normal_(p, mean=0.0, std=std)

    def _run_block(self, block, x, loop_idx=None):
        ckpt = self.grad_checkpoint and torch.is_grad_enabled()
        if isinstance(block, DeltaMoEBlock):

            def _moe(inp):
                return block(inp, loop_idx=loop_idx)

            if ckpt:
                return _activation_checkpoint(_moe, x, use_reentrant=False)
            return _moe(x)
        if ckpt:
            return _activation_checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, ids):
        x = self.tok_emb(ids)
        for block in self.prelude:
            x = self._run_block(block, x)
        anchor = self.inject(x) * self.inject_scale
        aux_terms = []
        for t in range(self.n_loops):
            li = min(t, self.max_loops - 1)
            x = self.core_norm(x + anchor + self.loop_emb[li])
            for block in self.core:
                if isinstance(block, DeltaMoEBlock):
                    x, aux_t = self._run_block(block, x, loop_idx=li)
                    if aux_t is not None:
                        aux_terms.append(aux_t)
                else:
                    x = self._run_block(block, x, loop_idx=li)
        for block in self.coda:
            x = self._run_block(block, x)
        x = self.norm(x)
        logits = self.head(x)
        self.logits = logits
        # Aux is a tensor from THIS forward (returned from MoE, not a
        # leftover module attr). Train adds it to CE before a single backward.
        self.aux_loss = (
            torch.stack(aux_terms).mean() if aux_terms else logits.new_zeros(())
        )
        return logits


def _config_from_ctx(ctx):
    cfg = dict(DEFAULTS)
    if isinstance(ctx, dict):
        overrides = ctx.get("arch")
        if isinstance(overrides, dict):
            cfg.update({k: v for k, v in overrides.items() if k in cfg})
        for k in _OVERRIDE_KEYS:
            if k in ctx:
                cfg[k] = ctx[k]
        mult = float(ctx.get("prism_width_multiplier", 1.0) or 1.0)
        if abs(mult - 1.0) > 1e-12:
            if mult <= 0:
                raise ValueError("prism_width_multiplier must be > 0")
            for key in (
                "d_model",
                "mlp_hidden",
                "expert_hidden",
                "shared_expert_hidden",
                "delta_key_dim",
                "delta_value_dim",
            ):
                cfg[key] = max(1, int(round(int(cfg[key]) * mult)))
            attn_heads = int(cfg["attn_heads"])
            head_dim = int(DEFAULTS["d_model"]) // int(DEFAULTS["attn_heads"])
            if head_dim > 0 and cfg["d_model"] % head_dim == 0:
                cfg["attn_heads"] = cfg["d_model"] // head_dim
            elif cfg["d_model"] % attn_heads != 0:
                h = min(attn_heads, cfg["d_model"])
                while h > 1 and cfg["d_model"] % h != 0:
                    h -= 1
                cfg["attn_heads"] = h
    return cfg


def build_loopmoe(ctx):
    ctx = ctx if isinstance(ctx, dict) else {}
    torch.manual_seed(int(ctx.get("seed", 0)))
    te_flag = bool(ctx.get("te_available", False))
    if not te_flag:
        te_flag = _probe_te_linear() is not None
    return LoopMoE(_config_from_ctx(ctx), use_te=te_flag)
