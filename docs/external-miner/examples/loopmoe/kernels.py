"""Fused kernels for LoopMoE — gated-delta scan, attention, RMS, CE.

submission_nonce: loopmoe-1b-zero1-20260819T1200Z

Default delta path is **in-pack factored chunked WY** (batched GEMMs).
Never the sequential-T Triton scan. Never auto-select FLA `chunk_kda` —
its Triton autotune cache races under DDP spawn (seen on 01f64b4f).

Optional:
  LOOPMOE_DELTA_KERNEL=kda     — fla-core chunk_kda (per-rank Triton cache)
  LOOPMOE_DELTA_KERNEL=triton  — recurrent scan (slower; do not ship)
  LOOPMOE_DELTA_KERNEL=eager   — legacy per-chunk WY loop

Hot-path train (seq=512 <= window) never walks `for t in range(seq)`.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

DELTA_KERNEL = "uninitialized"
ATTN_KERNEL = "sdpa"
RMS_KERNEL = "torch"
CE_KERNEL = "torch"
SWIGLU_KERNEL = "eager"
ROPE_KERNEL = "torch"

_DELTA_LOGGED = False
_TRITON_OK = None
_KDA_FN = None
_KDA_PROBED = False
_GDR_FN = None
_GDR_PROBED = False
_KDA_CHECKED = False
_KDA_OK = False


def kernel_map():
    return {
        "delta_kernel": DELTA_KERNEL,
        "attn_kernel": ATTN_KERNEL,
        "rmsnorm_kernel": RMS_KERNEL,
        "ce_kernel": CE_KERNEL,
        "swiglu_kernel": SWIGLU_KERNEL,
        "rope_kernel": ROPE_KERNEL,
    }


def _env_force():
    return os.environ.get("LOOPMOE_DELTA_KERNEL", "").strip().lower()


def _probe_kda():
    """FLA `chunk_kda` — per-channel decay, same recurrence as eager WY."""
    global _KDA_FN, _KDA_PROBED
    if _KDA_PROBED:
        return _KDA_FN
    _KDA_PROBED = True
    if _env_force() != "kda":
        return None
    try:
        from fla.ops.kda import chunk_kda  # type: ignore

        _KDA_FN = chunk_kda
    except Exception:  # noqa: BLE001
        _KDA_FN = None
    return _KDA_FN


def _probe_gdr():
    """FLA `chunk_gated_delta_rule` — per-head scalar g only. Never default."""
    global _GDR_FN, _GDR_PROBED
    if _GDR_PROBED:
        return _GDR_FN
    _GDR_PROBED = True
    if _env_force() not in {"gdr", "fla"}:
        return None
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore

        _GDR_FN = chunk_gated_delta_rule
    except Exception:  # noqa: BLE001
        _GDR_FN = None
    return _GDR_FN


def _probe_triton():
    global _TRITON_OK
    if _TRITON_OK is not None:
        return _TRITON_OK
    # Recurrent scan is opt-in only — do not even compile it on the default path.
    if _env_force() != "triton":
        _TRITON_OK = False
        return False
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        _TRITON_OK = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        _TRITON_OK = False
    return _TRITON_OK


def _log_delta(name):
    global DELTA_KERNEL, _DELTA_LOGGED
    DELTA_KERNEL = name
    if not _DELTA_LOGGED:
        print(f"[loopmoe] delta_kernel={name}", flush=True)
        _DELTA_LOGGED = True


# ---------------------------------------------------------------------------
# Eager sequential (reference) + batched WY (no token loop)
# ---------------------------------------------------------------------------


def _delta_sequential(q, k, v, beta, la):
    """Exact recurrent gated-delta. q/k/v/beta/la float, heads flattened."""
    bh, t, dk = q.shape
    dv = v.shape[-1]
    state = q.new_zeros(bh, dv, dk)
    outs = []
    for i in range(t):
        alpha = la[:, i, :].exp()
        state = state * alpha.unsqueeze(1)
        kt = k[:, i, :]
        vt = v[:, i, :]
        bt = beta[:, i, :]
        qt = q[:, i, :]
        sk = torch.einsum("bvd,bd->bv", state, kt)
        u = bt * (vt - sk)
        state = state + u.unsqueeze(-1) * kt.unsqueeze(1)
        outs.append(torch.einsum("bvd,bd->bv", state, qt))
    return torch.stack(outs, dim=1)


def _delta_chunk_loop(q, k, v, beta, la, chunk):
    """Original WY chunk loop — last-resort fallback."""
    _, t, dk = q.shape
    dv = v.shape[-1]
    state = q.new_zeros(q.shape[0], dv, dk)
    outs = []
    eye_full = torch.eye(chunk, device=q.device, dtype=q.dtype)
    tril_full = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    for s in range(0, t, chunk):
        e = min(s + chunk, t)
        c = e - s
        qc, kc, vc = q[:, s:e], k[:, s:e], v[:, s:e]
        bc = beta[:, s:e]
        L = la[:, s:e].cumsum(dim=1)
        ldiff = L[:, :, None, :] - L[:, None, :, :]
        dec = ldiff.masked_fill(~tril_full[:c, :c][None, :, :, None], float("-inf")).exp()
        a = (bc * torch.einsum("btc,bic,btic->bti", kc, kc, dec)).tril(-1)
        bm = torch.einsum("btc,bic,btic->bti", qc, kc, dec)
        lexp = L.exp()
        rhs = bc * (vc - (kc * lexp) @ state.transpose(-1, -2))
        u = torch.linalg.solve_triangular(
            a + eye_full[:c, :c], rhs, upper=False, unitriangular=True
        )
        outs.append((qc * lexp) @ state.transpose(-1, -2) + bm @ u)
        e_lc = L[:, -1:, :].exp()
        k_tail = kc * (L[:, -1:, :] - L).exp()
        state = state * e_lc + u.transpose(-1, -2) @ k_tail
    return torch.cat(outs, dim=1)


def _delta_vectorized(q, k, v, beta, la, chunk):
    """Chunked WY via factored batched GEMMs (FLA algorithm, per-channel decay).

    Dec[t,i,d] = exp(L[t,d] − L[i,d]) is never materialized as a 5-D tensor.
    Intra-chunk A/B are ``(x ⊙ e^L) @ (k ⊙ e^{−L})^T`` (cuBLAS), then a short
    state-carry over n_chunks (16 at seq 512). Safe in fp32 for chunk<=40
    with |la|<=2 (exp(80) still finite). Longer chunks fall back to the
    masked 5-D path.
    """
    bh, t, dk = q.shape
    dv = v.shape[-1]
    n_chunks = (t + chunk - 1) // chunk
    pad = n_chunks * chunk - t
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        la = F.pad(la, (0, 0, 0, pad))
    qc = q.view(bh, n_chunks, chunk, dk)
    kc = k.view(bh, n_chunks, chunk, dk)
    vc = v.view(bh, n_chunks, chunk, dv)
    bc = beta.view(bh, n_chunks, chunk, 1)
    lac = la.view(bh, n_chunks, chunk, dk)
    l = lac.cumsum(dim=2)
    lexp = l.exp()
    tril = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    if int(chunk) <= 40:
        inv = (-l).exp()
        k_neg_t = (kc * inv).transpose(-1, -2)
        gram = torch.matmul(kc * lexp, k_neg_t)
        a = (bc * gram).tril(-1)
        bm = torch.matmul(qc * lexp, k_neg_t).masked_fill(~tril.view(1, 1, chunk, chunk), 0)
    else:
        ldiff = l.unsqueeze(3) - l.unsqueeze(2)
        dec = ldiff.masked_fill(~tril.view(1, 1, chunk, chunk, 1), float("-inf")).exp()
        a = (bc * torch.einsum("bntd,bnid,bntid->bnti", kc, kc, dec)).tril(-1)
        bm = torch.einsum("bntd,bnid,bntid->bnti", qc, kc, dec)
    eye = torch.eye(chunk, device=q.device, dtype=q.dtype)
    state = q.new_zeros(bh, dv, dk)
    outs = q.new_zeros(bh, n_chunks, chunk, dv)
    # State carry across chunks — not a token loop.
    for i in range(n_chunks):
        rhs = bc[:, i] * (vc[:, i] - (kc[:, i] * lexp[:, i]) @ state.transpose(-1, -2))
        u = torch.linalg.solve_triangular(a[:, i] + eye, rhs, upper=False, unitriangular=True)
        outs[:, i] = (qc[:, i] * lexp[:, i]) @ state.transpose(-1, -2) + bm[:, i] @ u
        e_lc = lexp[:, i, -1:, :]
        k_tail = kc[:, i] * (l[:, i, -1:, :] - l[:, i]).exp()
        state = state * e_lc + u.transpose(-1, -2) @ k_tail
    return outs.reshape(bh, n_chunks * chunk, dv)[:, :t]


# ---------------------------------------------------------------------------
# Triton fused recurrent (per-channel decay) + custom bwd
# ---------------------------------------------------------------------------


def _triton_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        b_ptr,
        g_ptr,
        o_ptr,
        chk_ptr,
        T,
        DK,
        DV,
        stride_q_bh,
        stride_q_t,
        stride_q_d,
        stride_k_bh,
        stride_k_t,
        stride_k_d,
        stride_v_bh,
        stride_v_t,
        stride_v_d,
        stride_b_bh,
        stride_b_t,
        stride_g_bh,
        stride_g_t,
        stride_g_d,
        stride_o_bh,
        stride_o_t,
        stride_o_d,
        stride_c_bh,
        stride_c_n,
        stride_c_v,
        stride_c_k,
        CHUNK: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        bh = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        mask_k = offs_k < DK
        mask_v = offs_v < DV
        mask_h = mask_v[:, None] & mask_k[None, :]
        s = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)
        q_bh = q_ptr + bh * stride_q_bh
        k_bh = k_ptr + bh * stride_k_bh
        v_bh = v_ptr + bh * stride_v_bh
        b_bh = b_ptr + bh * stride_b_bh
        g_bh = g_ptr + bh * stride_g_bh
        o_bh = o_ptr + bh * stride_o_bh
        c_bh = chk_ptr + bh * stride_c_bh
        tl.store(
            c_bh + offs_v[:, None] * stride_c_v + offs_k[None, :] * stride_c_k,
            s,
            mask=mask_h,
        )
        nchk = 1
        for t in range(0, T):
            q = tl.load(q_bh + t * stride_q_t + offs_k * stride_q_d, mask=mask_k, other=0.0).to(
                tl.float32
            )
            k = tl.load(k_bh + t * stride_k_t + offs_k * stride_k_d, mask=mask_k, other=0.0).to(
                tl.float32
            )
            v = tl.load(v_bh + t * stride_v_t + offs_v * stride_v_d, mask=mask_v, other=0.0).to(
                tl.float32
            )
            beta = tl.load(b_bh + t * stride_b_t).to(tl.float32)
            gk = tl.load(g_bh + t * stride_g_t + offs_k * stride_g_d, mask=mask_k, other=0.0).to(
                tl.float32
            )
            alpha = tl.exp(gk)
            s = s * alpha[None, :]
            sk = tl.sum(s * k[None, :], axis=1)
            u = beta * (v - sk)
            s = s + u[:, None] * k[None, :]
            o = tl.sum(s * q[None, :], axis=1)
            tl.store(
                o_bh + t * stride_o_t + offs_v * stride_o_d, o.to(o_ptr.dtype.element_ty), mask=mask_v
            )
            if (t + 1) % CHUNK == 0:
                tl.store(
                    c_bh
                    + nchk * stride_c_n
                    + offs_v[:, None] * stride_c_v
                    + offs_k[None, :] * stride_c_k,
                    s,
                    mask=mask_h,
                )
                nchk = nchk + 1

    @triton.jit
    def bwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        b_ptr,
        g_ptr,
        do_ptr,
        chk_ptr,
        scratch_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        db_ptr,
        dg_ptr,
        T,
        NCHK,
        DK,
        DV,
        stride_q_bh,
        stride_q_t,
        stride_q_d,
        stride_k_bh,
        stride_k_t,
        stride_k_d,
        stride_v_bh,
        stride_v_t,
        stride_v_d,
        stride_b_bh,
        stride_b_t,
        stride_g_bh,
        stride_g_t,
        stride_g_d,
        stride_o_bh,
        stride_o_t,
        stride_o_d,
        stride_c_bh,
        stride_c_n,
        stride_c_v,
        stride_c_k,
        stride_s_bh,
        stride_s_j,
        stride_s_v,
        stride_s_k,
        CHUNK: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        bh = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        mask_k = offs_k < DK
        mask_v = offs_v < DV
        mask_h = mask_v[:, None] & mask_k[None, :]
        q_bh = q_ptr + bh * stride_q_bh
        k_bh = k_ptr + bh * stride_k_bh
        v_bh = v_ptr + bh * stride_v_bh
        b_bh = b_ptr + bh * stride_b_bh
        g_bh = g_ptr + bh * stride_g_bh
        do_bh = do_ptr + bh * stride_o_bh
        c_bh = chk_ptr + bh * stride_c_bh
        sc_bh = scratch_ptr + bh * stride_s_bh
        dq_bh = dq_ptr + bh * stride_q_bh
        dk_bh = dk_ptr + bh * stride_k_bh
        dv_bh = dv_ptr + bh * stride_v_bh
        db_bh = db_ptr + bh * stride_b_bh
        dg_bh = dg_ptr + bh * stride_g_bh
        ds = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)
        for ic in range(0, NCHK):
            ci = NCHK - 1 - ic
            s = tl.load(
                c_bh
                + ci * stride_c_n
                + offs_v[:, None] * stride_c_v
                + offs_k[None, :] * stride_c_k,
                mask=mask_h,
                other=0.0,
            ).to(tl.float32)
            for j in range(0, CHUNK):
                t = ci * CHUNK + j
                k = tl.load(k_bh + t * stride_k_t + offs_k * stride_k_d, mask=mask_k, other=0.0).to(
                    tl.float32
                )
                v = tl.load(v_bh + t * stride_v_t + offs_v * stride_v_d, mask=mask_v, other=0.0).to(
                    tl.float32
                )
                beta = tl.load(b_bh + t * stride_b_t).to(tl.float32)
                gk = tl.load(g_bh + t * stride_g_t + offs_k * stride_g_d, mask=mask_k, other=0.0).to(
                    tl.float32
                )
                alpha = tl.exp(gk)
                s = s * alpha[None, :]
                sk = tl.sum(s * k[None, :], axis=1)
                u = beta * (v - sk)
                s = s + u[:, None] * k[None, :]
                tl.store(
                    sc_bh
                    + j * stride_s_j
                    + offs_v[:, None] * stride_s_v
                    + offs_k[None, :] * stride_s_k,
                    s,
                    mask=mask_h,
                )
            for jj in range(0, CHUNK):
                j = CHUNK - 1 - jj
                t = ci * CHUNK + j
                q = tl.load(q_bh + t * stride_q_t + offs_k * stride_q_d, mask=mask_k, other=0.0).to(
                    tl.float32
                )
                k = tl.load(k_bh + t * stride_k_t + offs_k * stride_k_d, mask=mask_k, other=0.0).to(
                    tl.float32
                )
                v = tl.load(v_bh + t * stride_v_t + offs_v * stride_v_d, mask=mask_v, other=0.0).to(
                    tl.float32
                )
                beta = tl.load(b_bh + t * stride_b_t).to(tl.float32)
                gk = tl.load(g_bh + t * stride_g_t + offs_k * stride_g_d, mask=mask_k, other=0.0).to(
                    tl.float32
                )
                dout = tl.load(
                    do_bh + t * stride_o_t + offs_v * stride_o_d, mask=mask_v, other=0.0
                ).to(tl.float32)
                alpha = tl.exp(gk)
                st = tl.load(
                    sc_bh
                    + j * stride_s_j
                    + offs_v[:, None] * stride_s_v
                    + offs_k[None, :] * stride_s_k,
                    mask=mask_h,
                    other=0.0,
                ).to(tl.float32)
                if j == 0:
                    s_prev = tl.load(
                        c_bh
                        + ci * stride_c_n
                        + offs_v[:, None] * stride_c_v
                        + offs_k[None, :] * stride_c_k,
                        mask=mask_h,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    s_prev = tl.load(
                        sc_bh
                        + (j - 1) * stride_s_j
                        + offs_v[:, None] * stride_s_v
                        + offs_k[None, :] * stride_s_k,
                        mask=mask_h,
                        other=0.0,
                    ).to(tl.float32)
                s_mid = s_prev * alpha[None, :]
                smk = tl.sum(s_mid * k[None, :], axis=1)
                r = v - smk
                u = beta * r
                dq = tl.sum(st * dout[:, None], axis=0)
                ds = ds + dout[:, None] * q[None, :]
                du = tl.sum(ds * k[None, :], axis=1)
                dk = tl.sum(ds * u[:, None], axis=0)
                dbeta = tl.sum(du * r)
                dr = du * beta
                dk = dk + (-1.0) * tl.sum(s_mid * dr[:, None], axis=0)
                ds_mid = ds + (-1.0) * dr[:, None] * k[None, :]
                dalpha = tl.sum(ds_mid * s_prev, axis=0)
                dgk = dalpha * alpha
                ds = ds_mid * alpha[None, :]
                tl.store(
                    dq_bh + t * stride_q_t + offs_k * stride_q_d, dq.to(dq_ptr.dtype.element_ty), mask=mask_k
                )
                tl.store(
                    dk_bh + t * stride_k_t + offs_k * stride_k_d, dk.to(dk_ptr.dtype.element_ty), mask=mask_k
                )
                tl.store(
                    dv_bh + t * stride_v_t + offs_v * stride_v_d, dr.to(dv_ptr.dtype.element_ty), mask=mask_v
                )
                tl.store(db_bh + t * stride_b_t, dbeta.to(db_ptr.dtype.element_ty))
                tl.store(
                    dg_bh + t * stride_g_t + offs_k * stride_g_d, dgk.to(dg_ptr.dtype.element_ty), mask=mask_k
                )

    return triton, fwd_kernel, bwd_kernel


def _pad_time(q, k, v, beta, la, chunk):
    t = q.shape[1]
    pad = (chunk - t % chunk) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        la = F.pad(la, (0, 0, 0, pad))
    return q, k, v, beta, la, t, pad


class _TritonGatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, la, chunk):
        triton, fwd_kernel, _ = _triton_kernels()
        q, k, v, beta, la, t_orig, pad = _pad_time(q, k, v, beta, la, chunk)
        bh, t, dk = q.shape
        dv = v.shape[-1]
        nchk = t // chunk
        o = torch.empty(bh, t, dv, device=q.device, dtype=q.dtype)
        chk = torch.zeros(bh, nchk + 1, dv, dk, device=q.device, dtype=torch.float32)
        block_k = triton.next_power_of_2(dk)
        block_v = triton.next_power_of_2(dv)
        fwd_kernel[(bh,)](
            q,
            k,
            v,
            beta,
            la,
            o,
            chk,
            t,
            dk,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            beta.stride(0),
            beta.stride(1),
            la.stride(0),
            la.stride(1),
            la.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            chk.stride(0),
            chk.stride(1),
            chk.stride(2),
            chk.stride(3),
            CHUNK=chunk,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(q, k, v, beta, la, chk)
        ctx.chunk = chunk
        ctx.t_orig = t_orig
        return o[:, :t_orig]

    @staticmethod
    def backward(ctx, do):
        triton, _, bwd_kernel = _triton_kernels()
        q, k, v, beta, la, chk = ctx.saved_tensors
        chunk = ctx.chunk
        t_orig = ctx.t_orig
        bh, t, dk = q.shape
        dv = v.shape[-1]
        nchk = t // chunk
        if do.shape[1] < t:
            do = F.pad(do.contiguous(), (0, 0, 0, t - do.shape[1]))
        else:
            do = do.contiguous()
        dq = torch.empty_like(q)
        dkt = torch.empty_like(k)
        dvt = torch.empty_like(v)
        dbeta = torch.empty_like(beta)
        dla = torch.empty_like(la)
        scratch = torch.empty(bh, chunk, dv, dk, device=q.device, dtype=torch.float32)
        block_k = triton.next_power_of_2(dk)
        block_v = triton.next_power_of_2(dv)
        bwd_kernel[(bh,)](
            q,
            k,
            v,
            beta,
            la,
            do,
            chk,
            scratch,
            dq,
            dkt,
            dvt,
            dbeta,
            dla,
            t,
            nchk,
            dk,
            dv,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            beta.stride(0),
            beta.stride(1),
            la.stride(0),
            la.stride(1),
            la.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            chk.stride(0),
            chk.stride(1),
            chk.stride(2),
            chk.stride(3),
            scratch.stride(0),
            scratch.stride(1),
            scratch.stride(2),
            scratch.stride(3),
            CHUNK=chunk,
            BLOCK_K=block_k,
            BLOCK_V=block_v,
            num_warps=4,
            num_stages=2,
        )
        return (
            dq[:, :t_orig],
            dkt[:, :t_orig],
            dvt[:, :t_orig],
            dbeta[:, :t_orig],
            dla[:, :t_orig],
            None,
        )


def _run_triton(q, k, v, beta, la, chunk):
    return _TritonGatedDelta.apply(q, k, v, beta, la, int(chunk))


def _run_kda(q, k, v, beta, la, chunk):
    """FLA `chunk_kda`: per-channel g, scale=1 (matches eager WY, not 1/sqrt(K))."""
    fn = _probe_kda()
    q4 = q.unsqueeze(2)
    k4 = k.unsqueeze(2)
    v4 = v.unsqueeze(2)
    g = la.unsqueeze(2)
    b = beta.squeeze(-1).unsqueeze(-1)
    cs = int(chunk) if int(chunk) in (16, 32, 64) else 32
    o, _ = fn(
        q4,
        k4,
        v4,
        g,
        b,
        scale=1.0,
        use_qk_l2norm_in_kernel=False,
        chunk_size=cs,
    )
    return o.squeeze(2)


def _run_gdr(q, k, v, beta, la, chunk):
    """FLA GDN kernel — scalar gate = mean of per-channel decay. Opt-in only."""
    fn = _probe_gdr()
    q4 = q.unsqueeze(2)
    k4 = k.unsqueeze(2)
    v4 = v.unsqueeze(2)
    g = la.mean(dim=-1).unsqueeze(-1)
    b = beta.squeeze(-1).unsqueeze(-1)
    cs = int(chunk) if int(chunk) in (16, 32, 64) else 32
    o, _ = fn(
        q4,
        k4,
        v4,
        g,
        b,
        scale=1.0,
        use_qk_l2norm_in_kernel=False,
        chunk_size=cs,
    )
    return o.squeeze(2)


def _maybe_check_kda(q, k, v, beta, la, chunk, kda_out):
    """Once on CUDA: refuse KDA as default if it diverges from eager WY."""
    global _KDA_CHECKED, _KDA_OK
    if _KDA_CHECKED:
        return _KDA_OK
    _KDA_CHECKED = True
    if not q.is_cuda:
        _KDA_OK = False
        return False
    with torch.no_grad():
        sl = slice(0, min(2, q.shape[0]))
        st = slice(0, min(q.shape[1], int(chunk) * 2))
        ref = _delta_vectorized(q[sl, st], k[sl, st], v[sl, st], beta[sl, st], la[sl, st], chunk)
        got = kda_out[sl, st]
        err = (ref - got).abs().max().item()
        scale = max(ref.abs().max().item(), 1e-6)
        rel = err / scale
        print(f"[loopmoe] kda-vs-wy maxabs={err:.3e} rel={rel:.3e}", flush=True)
        if rel > 5e-2 and err > 5e-3:
            print("[loopmoe] kda diverges from WY; defaulting to chunk_wy", flush=True)
            _KDA_OK = False
            return False
    _KDA_OK = True
    return True


def gated_delta(q, k, v, beta, la, chunk=32):
    """q/k/v/beta/la already float, heads flattened. Returns float (BH, T, DV)."""
    force = _env_force()
    if force == "triton" and q.is_cuda and _probe_triton():
        try:
            out = _run_triton(q, k, v, beta, la, chunk)
            _log_delta("triton")
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"[loopmoe] triton delta failed ({exc}); chunk_wy", flush=True)
    if force in {"gdr", "fla"} and q.is_cuda and _probe_gdr() is not None:
        try:
            out = _run_gdr(q, k, v, beta, la, chunk)
            _log_delta("gdr")
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"[loopmoe] gdr delta failed ({exc}); chunk_wy", flush=True)
    if force == "kda" and q.is_cuda:
        if (not _KDA_CHECKED or _KDA_OK) and _probe_kda() is not None:
            try:
                out = _run_kda(q, k, v, beta, la, chunk)
                if _maybe_check_kda(q, k, v, beta, la, chunk, out):
                    _log_delta("chunk_kda")
                    return out
            except Exception as exc:  # noqa: BLE001
                print(f"[loopmoe] chunk_kda failed ({exc}); chunk_wy", flush=True)
    if force == "eager":
        _log_delta("eager")
        return _delta_chunk_loop(q, k, v, beta, la, chunk)
    _log_delta("chunk_wy")
    return _delta_vectorized(q, k, v, beta, la, chunk)


# ---------------------------------------------------------------------------
# Attention / RMS / RoPE / CE helpers
# ---------------------------------------------------------------------------


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
    # Prefer FA-3 / FA-2 python bindings if present (SM120 often lacks them).
    attn = os.environ.get("LOOPMOE_ATTN_KERNEL", "").strip().lower()
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
            print(f"[loopmoe] liger CE failed ({exc}); torch", flush=True)
    global CE_KERNEL
    CE_KERNEL = "torch"
    return F.cross_entropy(logits, labels)


def log_kernel_banner():
    enable_attn_backends()
    _probe_ce()
    print(
        f"[loopmoe] kernel_map delta={DELTA_KERNEL} attn={ATTN_KERNEL} "
        f"rmsnorm={RMS_KERNEL} ce={CE_KERNEL} swiglu={SWIGLU_KERNEL} rope={ROPE_KERNEL}",
        flush=True,
    )
