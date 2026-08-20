"""Attention / RMS / RoPE / CE helpers for the dense 1B reference.

submission_nonce: dense-1b-b200-20260819T1952Z

Hot-path train (seq=512) is one Flash/SDPA call — no Python token loop.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

ATTN_KERNEL = "sdpa"
RMS_KERNEL = "torch"
CE_KERNEL = "torch"
SWIGLU_KERNEL = "eager"
ROPE_KERNEL = "torch"


def kernel_map():
    return {
        "attn_kernel": ATTN_KERNEL,
        "rmsnorm_kernel": RMS_KERNEL,
        "ce_kernel": CE_KERNEL,
        "swiglu_kernel": SWIGLU_KERNEL,
        "rope_kernel": ROPE_KERNEL,
    }


def enable_attn_backends():
    global ATTN_KERNEL
    if not torch.cuda.is_available():
        ATTN_KERNEL = "math"
        return
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:  # noqa: BLE001
        pass
    attn = os.environ.get("DENSE1B_ATTN_KERNEL", "").strip().lower()
    if attn:
        ATTN_KERNEL = attn
        return
    try:
        import flash_attn  # noqa: F401

        ATTN_KERNEL = "fa2"
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        import transformer_engine.pytorch as te  # noqa: F401

        if hasattr(te, "DotProductAttention"):
            ATTN_KERNEL = "te_avail"
    except Exception:  # noqa: BLE001
        pass
    ATTN_KERNEL = "sdpa"


def sdpa(q, k, v, *, is_causal=False, attn_mask=None):
    """q/k/v: (b, h, t, d). Uses the fastest enabled SDPA backend."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


def rms_norm(x, weight, eps=1e-6):
    global RMS_KERNEL
    RMS_KERNEL = "torch"
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def apply_rope(x, cos, sin):
    global ROPE_KERNEL
    ROPE_KERNEL = "torch"
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def rope_tables(t, head_dim, theta, device, dtype):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    pos = torch.arange(t, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


_CE_FN = None
_CE_PROBED = False


def _probe_ce():
    global _CE_FN, _CE_PROBED, CE_KERNEL
    if _CE_PROBED:
        return _CE_FN
    _CE_PROBED = True
    try:
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss  # type: ignore

        _CE_FN = LigerCrossEntropyLoss(reduction="mean")
        CE_KERNEL = "liger"
    except Exception:  # noqa: BLE001
        _CE_FN = None
        CE_KERNEL = "torch"
    return _CE_FN


def cross_entropy(logits, labels):
    """logits (N, V) float, labels (N,)."""
    fn = _probe_ce()
    if fn is not None:
        try:
            return fn(logits, labels)
        except Exception as exc:  # noqa: BLE001
            print(f"[dense1b] liger CE failed ({exc}); torch", flush=True)
    global CE_KERNEL
    CE_KERNEL = "torch"
    return F.cross_entropy(logits, labels)


def log_kernel_banner():
    enable_attn_backends()
    _probe_ce()
    print(
        f"[dense1b] kernel_map attn={ATTN_KERNEL} rmsnorm={RMS_KERNEL} "
        f"ce={CE_KERNEL} swiglu={SWIGLU_KERNEL} rope={ROPE_KERNEL}",
        flush=True,
    )
