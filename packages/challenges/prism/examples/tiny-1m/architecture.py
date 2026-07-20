# PRISM tiny ~1M-parameter v2 architecture script.
#
# This is the architecture half of a v2 two-script bundle: it exposes ONLY the
# build_model(ctx) factory and defines the model. The training loop lives in the
# sibling training.py. A weight-tied decoder transformer (dim=128, heads=4,
# 2 layers, SwiGLU MLP, vocab=4096 -> ~1.05M params) used to prove the Prism
# GPU-eval pipeline end-to-end.
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
MODEL_MLP_RATIO = 4
DEFAULT_EPS = 0.000001
EMB_INIT_STD = 0.13


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


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.norm_attn = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = GatedMLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class TinyDecoderLM(nn.Module):
    # Weight-tied decoder LM: pre-norm residual blocks (RMSNorm + multi-head
    # causal self-attention + SwiGLU GatedMLP) with a tied token embedding /
    # output head. forward(tokens) returns next-token logits [B, T, V].
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise Exception("n_layers must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        blocks: list[TransformerBlock] = []
        for _ in range(n_layers):
            blocks.append(TransformerBlock(dim, n_heads, mlp_ratio))
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


def build_model(ctx: PrismContext) -> TinyDecoderLM:
    # Pure factory: size the vocabulary from ctx; dim/layers stay small so the
    # parameter count is ~1.05M (far under ctx.max_params and the 124M explore cap).
    return TinyDecoderLM(
        vocab_size=ctx.vocab_size,
        dim=MODEL_DIM,
        n_heads=MODEL_HEADS,
        n_layers=MODEL_LAYERS,
        mlp_ratio=MODEL_MLP_RATIO,
    )
