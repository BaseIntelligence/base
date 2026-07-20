# PRISM hybrid-attn-ssm-tiny ~2-4M-parameter hybrid architecture script.
#
# Architecture half of a v2 two-script bundle. Hymba/Jamba/Zamba-mini spirit:
# interleave pure-torch selective SSM blocks with sparse causal multi-head
# attention (arXiv 2411.13676 / 2403.19887 / 2405.16712 class). No mamba_ssm /
# flash_attn required — reuses the mamba-tiny pure scan pattern + stock MHA.
#
# Sandbox contract notes (see evaluator/sandbox.py):
#   * No module-level docstring (a top-level ast.Expr is rejected) -> # comments.
#   * build_model is pure: it never reads data, opens files, or touches the network.
import math

import torch
import torch.nn.functional as F
from torch import nn

from prism_challenge.evaluator.interface import PrismContext

MODEL_DIM = 128
MODEL_LAYERS = 3
MODEL_HEADS = 4
MODEL_D_STATE = 16
MODEL_D_CONV = 4
MODEL_EXPAND = 2
MODEL_DT_RANK = 8
MODEL_MLP_RATIO = 4
# Insert a full causal attention hop every ATTN_EVERY layers (1-indexed layers).
MODEL_ATTN_EVERY = 2
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


class SelectiveSSM(nn.Module):
    # Pure-PyTorch selective scan (S6-style) — same family as Imp mamba-tiny.
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int,
    ) -> None:
        super().__init__()
        if dim <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0 or dt_rank <= 0:
            raise Exception("SelectiveSSM dimensions must be positive")
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = dim * expand
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.act = nn.SiLU()

        with torch.no_grad():
            dt_init = math.log(math.exp(0.001) - 1.0)
            self.dt_proj.bias.fill_(dt_init)

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, d_inner = u.shape
        n_state = a.shape[-1]
        delta_a = torch.exp(delta.unsqueeze(-1) * a.view(1, 1, d_inner, n_state))
        delta_b_u = delta.unsqueeze(-1) * b.unsqueeze(2) * u.unsqueeze(-1)
        state = u.new_zeros(batch, d_inner, n_state)
        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            state = state * delta_a[:, t] + delta_b_u[:, t]
            y_t = torch.einsum("bdn,bn->bd", state, c[:, t])
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _dim = x.shape
        projected = self.in_proj(x)
        x_branch, z_branch = projected.chunk(2, dim=-1)
        conv_in = x_branch.transpose(1, 2)
        conv_out = self.conv1d(conv_in)[..., :seq_len]
        x_conv = self.act(conv_out.transpose(1, 2))
        projected_params = self.x_proj(x_conv)
        dt_raw, b, c = torch.split(
            projected_params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_raw))
        a = -torch.exp(self.A_log.float())
        y = self._selective_scan(x_conv, delta, a, b, c)
        y = y + x_conv * self.D
        y = y * self.act(z_branch)
        return self.out_proj(y)


class HybridBlock(nn.Module):
    # SSM residual always; optional causal MHA hop (Hymba/Jamba-mini interleave).
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int,
        mlp_ratio: int,
        use_attn: bool,
    ) -> None:
        super().__init__()
        self.use_attn = use_attn
        self.norm_ssm = RMSNorm(dim)
        self.ssm = SelectiveSSM(dim, d_state, d_conv, expand, dt_rank)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = GatedMLP(dim, dim * mlp_ratio)
        if use_attn:
            self.norm_attn = RMSNorm(dim)
            self.attn = CausalSelfAttention(dim, n_heads)
        else:
            self.norm_attn = None
            self.attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm_ssm(x))
        if self.use_attn and self.attn is not None and self.norm_attn is not None:
            x = x + self.attn(self.norm_attn(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class HybridAttnSsmLM(nn.Module):
    # Weight-tied hybrid LM. forward(tokens) -> logits [B, T, V].
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int,
        mlp_ratio: int,
        attn_every: int,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise Exception("n_layers must be positive")
        if attn_every <= 0:
            raise Exception("attn_every must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        blocks: list[HybridBlock] = []
        for layer_idx in range(n_layers):
            # 1-indexed layer numbers: attention hop every attn_every layers.
            use_attn = ((layer_idx + 1) % attn_every) == 0
            blocks.append(
                HybridBlock(
                    dim,
                    n_heads,
                    d_state,
                    d_conv,
                    expand,
                    dt_rank,
                    mlp_ratio,
                    use_attn=use_attn,
                )
            )
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


def build_model(ctx: PrismContext) -> HybridAttnSsmLM:
    # Pure factory under explore 124M; thrash geometry targets ~2-4M params.
    return HybridAttnSsmLM(
        vocab_size=ctx.vocab_size,
        dim=MODEL_DIM,
        n_layers=MODEL_LAYERS,
        n_heads=MODEL_HEADS,
        d_state=MODEL_D_STATE,
        d_conv=MODEL_D_CONV,
        expand=MODEL_EXPAND,
        dt_rank=MODEL_DT_RANK,
        mlp_ratio=MODEL_MLP_RATIO,
        attn_every=MODEL_ATTN_EVERY,
    )
