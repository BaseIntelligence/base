# PRISM deeploop-tiny ~1M-parameter DeepLoop-class architecture script.
#
# This is the architecture half of a v2 two-script bundle: it exposes ONLY the
# build_model(ctx) factory and defines the model. The training loop lives in the
# sibling training.py. A looped residual decoder LM in the DeepLoop spirit
# (shared-weight physical blocks unrolled L times with residual loop scale):
# arXiv 2607.13491 class; also LT2 / LoopFormer lineage. Pure torch only.
#
# Sandbox contract notes (see evaluator/sandbox.py):
#   * No module-level docstring (a top-level ast.Expr is rejected) -> # comments.
#   * build_model is pure: it never reads data, opens files, or touches the network.
import torch
import torch.nn.functional as F
from torch import nn

from prism_challenge.evaluator.interface import PrismContext

# Geometry stays near Imp transformer-tiny dims for fair thrash (~1-1.5M params).
MODEL_DIM = 128
MODEL_HEADS = 4
MODEL_PHYSICAL_BLOCKS = 1
MODEL_LOOPS = 4
MODEL_MLP_RATIO = 4
DEFAULT_EPS = 0.000001
EMB_INIT_STD = 0.13
LOOP_RESIDUAL_SCALE_INIT = 0.1


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


class DeepLoopBlock(nn.Module):
    # One physical residual block (RMSNorm + causal MHA + gated MLP). DeepLoop
    # unrolls the same parameters loops times with a learnable residual scale.
    def __init__(self, dim: int, n_heads: int, mlp_ratio: int, loops: int) -> None:
        super().__init__()
        if loops <= 0:
            raise Exception("loops must be positive")
        self.loops = loops
        hidden_dim = dim * mlp_ratio
        self.norm_attn = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = GatedMLP(dim, hidden_dim)
        # Per-loop residual scales (DeepLoop-class deterministic residual parameterization).
        self.loop_scale = nn.Parameter(torch.full((loops,), LOOP_RESIDUAL_SCALE_INIT))

    def _physical_step(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for loop_idx in range(self.loops):
            step = self._physical_step(x)
            # Shared-weight depth: residual mix uses loop_idx scale, not unique layers.
            x = x + self.loop_scale[loop_idx] * (step - x)
        return x


class DeepLoopLM(nn.Module):
    # Weight-tied DeepLoop-class LM: few physical blocks unrolled with shared
    # parameters. forward(tokens) returns next-token logits [B, T, V].
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_heads: int,
        n_physical_blocks: int,
        loops: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        if n_physical_blocks <= 0:
            raise Exception("n_physical_blocks must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        blocks: list[DeepLoopBlock] = []
        for _ in range(n_physical_blocks):
            blocks.append(DeepLoopBlock(dim, n_heads, mlp_ratio, loops))
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


def build_model(ctx: PrismContext) -> DeepLoopLM:
    # Pure factory: DeepLoop geometry keeps realized params near ~1M under Imp
    # thrash band; hard cap remains explore 124M.
    return DeepLoopLM(
        vocab_size=ctx.vocab_size,
        dim=MODEL_DIM,
        n_heads=MODEL_HEADS,
        n_physical_blocks=MODEL_PHYSICAL_BLOCKS,
        loops=MODEL_LOOPS,
        mlp_ratio=MODEL_MLP_RATIO,
    )
