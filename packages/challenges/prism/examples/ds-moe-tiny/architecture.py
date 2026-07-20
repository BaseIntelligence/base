# PRISM ds-moe-tiny ~4-12M-parameter DeepSeekMoE-style fine-grained MoE architecture.
#
# Architecture half of a v2 two-script bundle. Implements pure-torch fine-grained
# MoE FFN with shared expert isolation + top-k router (DeepSeekMoE arXiv 2401.06066;
# V2 MoE 2405.04434; K2 sparsity notes 2507.20534 motif only). Dense causal MHA
# residual outer. NOT full DeepSeek-V3/V4 multi-hundred-B MoE weights. No expert
# parallelism / DualPipe / device-limited routing required for correctness.
#
# Sandbox contract notes (see evaluator/sandbox.py):
#   * No module-level docstring (a top-level ast.Expr is rejected) -> # comments.
#   * build_model is pure: it never reads data, opens files, or touches the network.
import torch
import torch.nn.functional as F
from torch import nn

from prism_challenge.evaluator.interface import PrismContext

MODEL_DIM = 128
MODEL_HEADS = 4
MODEL_LAYERS = 2
MODEL_N_ROUTED = 8
MODEL_TOP_K = 2
MODEL_N_SHARED = 1
MODEL_EXPERT_MULT = 2
MODEL_MLP_RATIO_ATTN_SIDE = 0
DEFAULT_EPS = 0.000001
EMB_INIT_STD = 0.13
ROUTER_NOISE_STD = 0.01


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = DEFAULT_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.mul(x).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        if n_heads <= 0:
            raise Exception("n_heads must be positive")
        if dim % n_heads != 0:
            raise Exception("dim must be divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool)
        causal = torch.triu(causal, diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v
        context = context.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(context)


class SwiGLUExpert(nn.Module):
    # Tiny SwiGLU expert (fine-grained: expert_mult often < dense ×4).
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class MoEFeedForward(nn.Module):
    # DeepSeekMoE-class: n_shared always-on experts + n_routed top-k routed experts.
    # top_k routing is pure torch; no EP / token drop required for correctness.
    def __init__(
        self,
        dim: int,
        n_routed: int,
        top_k: int,
        n_shared: int,
        expert_mult: int,
    ) -> None:
        super().__init__()
        if dim <= 0 or n_routed <= 0 or top_k <= 0 or n_shared < 0 or expert_mult <= 0:
            raise Exception("MoEFeedForward dimensions must be positive")
        if top_k > n_routed:
            raise Exception("top_k cannot exceed n_routed")
        self.dim = dim
        self.n_routed = n_routed
        self.top_k = top_k
        self.n_shared = n_shared
        hidden = max(dim, dim * expert_mult)
        self.router = nn.Linear(dim, n_routed, bias=False)
        routed: list[SwiGLUExpert] = []
        for _ in range(n_routed):
            routed.append(SwiGLUExpert(dim, hidden))
        self.routed_experts = nn.ModuleList(routed)
        shared: list[SwiGLUExpert] = []
        for _ in range(n_shared):
            shared.append(SwiGLUExpert(dim, hidden))
        self.shared_experts = nn.ModuleList(shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        batch, seq_len, dim = x.shape
        flat = x.reshape(-1, dim)
        logits = self.router(flat)
        if self.training:
            logits = logits + torch.randn_like(logits) * ROUTER_NOISE_STD
        scores = torch.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(scores, k=self.top_k, dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(DEFAULT_EPS)

        routed_out = flat.new_zeros(flat.shape)
        # Dense gather over selected experts (N small for thrash: 8 experts).
        for expert_i, expert in enumerate(self.routed_experts):
            mask = top_idx.eq(expert_i)
            if not mask.any():
                continue
            weight = (top_vals * mask.to(top_vals.dtype)).sum(dim=-1, keepdim=True)
            # Only pay expert when any token routes here (still pure torch).
            if weight.sum() <= 0:
                continue
            out_e = expert(flat)
            routed_out = routed_out + out_e * weight

        shared_out = flat.new_zeros(flat.shape)
        if self.n_shared > 0:
            for expert in self.shared_experts:
                shared_out = shared_out + expert(flat)
            shared_out = shared_out / float(self.n_shared)

        return (routed_out + shared_out).view(batch, seq_len, dim)


class DSMoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_routed: int,
        top_k: int,
        n_shared: int,
        expert_mult: int,
    ) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = MoEFeedForward(dim, n_routed, top_k, n_shared, expert_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class DSMoELM(nn.Module):
    # Weight-tied DeepSeekMoE-style LM. Total params include all experts (activated
    # << total). forward(tokens) -> logits [B, T, V].
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_heads: int,
        n_layers: int,
        n_routed: int,
        top_k: int,
        n_shared: int,
        expert_mult: int,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise Exception("n_layers must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        blocks: list[DSMoEBlock] = []
        for _ in range(n_layers):
            blocks.append(DSMoEBlock(dim, n_heads, n_routed, top_k, n_shared, expert_mult))
        self.blocks = nn.ModuleList(blocks)
        self.norm_final = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        with torch.no_grad():
            self.token_emb.weight.normal_(0.0, EMB_INIT_STD)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        return self.lm_head(x)

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(tokens)
        vocab = logits.shape[-1]
        return F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1) % vocab)


def build_model(ctx: PrismContext) -> DSMoELM:
    # Pure factory under explore 124M; thrash geometry targets ~4-12M total params.
    # top_k kept explicit for mechanism signature / tests.
    _ = MODEL_TOP_K
    _ = MODEL_MLP_RATIO_ATTN_SIDE
    return DSMoELM(
        vocab_size=ctx.vocab_size,
        dim=MODEL_DIM,
        n_heads=MODEL_HEADS,
        n_layers=MODEL_LAYERS,
        n_routed=MODEL_N_ROUTED,
        top_k=MODEL_TOP_K,
        n_shared=MODEL_N_SHARED,
        expert_mult=MODEL_EXPERT_MULT,
    )
