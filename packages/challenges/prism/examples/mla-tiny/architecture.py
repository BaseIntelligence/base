# PRISM mla-tiny ~3-8M-parameter Multi-Head Latent Attention architecture.
#
# Architecture half of a v2 two-script bundle. Implements a pure-torch Distillation
# of DeepSeek-V2 Multi-Head Latent Attention (MLA) (arXiv 2405.04434; inherited by
# DeepSeek-V3 2412.19437). Low-rank joint KV compress into latent c_kv, up-proj to
# multi-head K/V, decoupled RoPE on a short PE path. NOT full DeepSeek-V3/V4 weights.
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
MODEL_LAYERS = 3
MODEL_KV_LORA_RANK = 32
MODEL_Q_LORA_RANK = 0
MODEL_ROPE_DIM = 16
MODEL_MLP_RATIO = 4
DEFAULT_EPS = 0.000001
EMB_INIT_STD = 0.13
ROPE_BASE = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = DEFAULT_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.mul(x).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, R]; cos/sin: [T, R]
    return (x * cos.unsqueeze(0).unsqueeze(0)) + (_rotate_half(x) * sin.unsqueeze(0).unsqueeze(0))


class MultiHeadLatentAttention(nn.Module):
    # Pure-torch MLA: joint low-rank KV latent + optional Q latent + decoupled RoPE.
    # Training path materializes K/V heads (no absorption CUDA kernel required).
    def __init__(
        self,
        dim: int,
        n_heads: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        rope_dim: int,
    ) -> None:
        super().__init__()
        if dim <= 0 or n_heads <= 0 or kv_lora_rank <= 0 or rope_dim <= 0:
            raise Exception("MultiHeadLatentAttention dimensions must be positive")
        if dim % n_heads != 0:
            raise Exception("dim must be divisible by n_heads")
        if rope_dim % 2 != 0:
            raise Exception("rope_dim must be even")
        if q_lora_rank < 0:
            raise Exception("q_lora_rank must be non-negative")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_dim = rope_dim
        self.scale = self.head_dim**-0.5

        # Joint KV compression: x -> c_kv (latent cache surface).
        self.kv_a = nn.Linear(dim, kv_lora_rank, bias=False)
        self.kv_a_norm = RMSNorm(kv_lora_rank)
        # Up-project latent into multi-head K and V content (nope).
        self.k_b = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)
        self.v_b = nn.Linear(kv_lora_rank, n_heads * self.head_dim, bias=False)

        # Decoupled RoPE key path (short PE dimension shared across heads).
        self.k_pe = nn.Linear(dim, n_heads * rope_dim, bias=False)

        if q_lora_rank > 0:
            self.q_a = nn.Linear(dim, q_lora_rank, bias=False)
            self.q_a_norm = RMSNorm(q_lora_rank)
            self.q_b = nn.Linear(q_lora_rank, n_heads * self.head_dim, bias=False)
            self.q_pe = nn.Linear(q_lora_rank, n_heads * rope_dim, bias=False)
        else:
            self.q_a = None
            self.q_a_norm = None
            self.q_b = nn.Linear(dim, n_heads * self.head_dim, bias=False)
            self.q_pe = nn.Linear(dim, n_heads * rope_dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        inv_freq = 1.0 / (
            ROPE_BASE ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _dim = x.shape
        # Latent KV path (materialized for training correctness without FlashMLA).
        c_kv = self.kv_a_norm(self.kv_a(x))
        k = self.k_b(c_kv).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_b(c_kv).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.q_a is not None and self.q_a_norm is not None:
            c_q = self.q_a_norm(self.q_a(x))
            q = self.q_b(c_q).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            q_pe = self.q_pe(c_q).view(batch, seq_len, self.n_heads, self.rope_dim).transpose(1, 2)
        else:
            q = self.q_b(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            q_pe = self.q_pe(x).view(batch, seq_len, self.n_heads, self.rope_dim).transpose(1, 2)

        k_pe = self.k_pe(x).view(batch, seq_len, self.n_heads, self.rope_dim).transpose(1, 2)
        cos, sin = self._rope_cos_sin(seq_len, x.device, x.dtype)
        q_pe = _apply_rope(q_pe, cos, sin)
        k_pe = _apply_rope(k_pe, cos, sin)

        # Concat content head_dim with decoupled PE dim for the softmax product.
        q_cat = torch.cat((q, q_pe), dim=-1)
        k_cat = torch.cat((k, k_pe), dim=-1)
        scale = (self.head_dim + self.rope_dim) ** -0.5
        scores = torch.matmul(q_cat, k_cat.transpose(-2, -1)) * scale
        causal = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        causal = torch.triu(causal, diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.out_proj(context)


class MLABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        rope_dim: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(dim)
        self.attn = MultiHeadLatentAttention(dim, n_heads, kv_lora_rank, q_lora_rank, rope_dim)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = GatedMLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class MLALM(nn.Module):
    # Weight-tied MLA decoder LM. forward(tokens) -> logits [B, T, V].
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_heads: int,
        n_layers: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        rope_dim: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise Exception("n_layers must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        blocks: list[MLABlock] = []
        for _ in range(n_layers):
            blocks.append(MLABlock(dim, n_heads, kv_lora_rank, q_lora_rank, rope_dim, mlp_ratio))
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


def build_model(ctx: PrismContext) -> MLALM:
    # Pure factory under explore 124M; thrash geometry targets ~3-8M params.
    # kv_lora_rank kept explicit for mechanism signature / tests.
    _ = MODEL_KV_LORA_RANK
    return MLALM(
        vocab_size=ctx.vocab_size,
        dim=MODEL_DIM,
        n_heads=MODEL_HEADS,
        n_layers=MODEL_LAYERS,
        kv_lora_rank=MODEL_KV_LORA_RANK,
        q_lora_rank=MODEL_Q_LORA_RANK,
        rope_dim=MODEL_ROPE_DIM,
        mlp_ratio=MODEL_MLP_RATIO,
    )
