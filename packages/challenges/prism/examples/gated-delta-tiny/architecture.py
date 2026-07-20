# PRISM gated-delta-tiny ~1-3M-parameter gated delta linear recurrence architecture.
#
# Architecture half of a v2 two-script bundle. Implements a pure-torch sequential
# gated delta-rule linear recurrence in the DeltaNet / Gated DeltaNet spirit
# (arXiv 2406.06484; gated comparative study 2607.07953; KDA 2510.26692 class).
# No flash_attn / flash_linear_attn / Tritonc chunk kernels required.
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
MODEL_LAYERS = 2
MODEL_N_HEADS = 4
MODEL_D_STATE = 32
MODEL_D_CONV = 4
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


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class GatedDeltaRecurrence(nn.Module):
    # Pure-PyTorch sequential gated delta update (no fused linear-attn kernel).
    # Per head memory S in R^{d_state x head_dim}; delta write with input gate beta.
    def __init__(self, dim: int, n_heads: int, d_state: int, d_conv: int) -> None:
        super().__init__()
        if dim <= 0 or n_heads <= 0 or d_state <= 0 or d_conv <= 0:
            raise Exception("GatedDeltaRecurrence dimensions must be positive")
        if dim % n_heads != 0:
            raise Exception("dim must be divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.d_state = d_state
        self.d_conv = d_conv
        self.scale = self.head_dim**-0.5

        self.in_proj = nn.Linear(dim, dim * 3, bias=False)
        self.beta_proj = nn.Linear(dim, n_heads, bias=True)
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.conv1d = nn.Conv1d(
            dim,
            dim,
            kernel_size=d_conv,
            groups=dim,
            padding=d_conv - 1,
            bias=True,
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.act = nn.SiLU()
        with torch.no_grad():
            # Softplus/sigmoid friendly small write rate at init.
            self.beta_proj.bias.fill_(-math.log(3.0))

    def _delta_scan(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        # k,v: [B, H, T, Dh]; beta: [B, H, T, 1] -> y: [B, H, T, Dh]
        # State S is [B, H, d_state, Dh]; keys act as state-address via a learned
        # linear map into d_state (absorbed into k as L2-normalized address).
        batch, n_heads, seq_len, head_dim = k.shape
        # Address keys into d_state dims by a fixed hash-free pad/rep projection:
        # reuse the first min(d_state, head_dim) of k and expand/tile if needed.
        if self.d_state == head_dim:
            key_addr = k
        elif self.d_state < head_dim:
            key_addr = k[..., : self.d_state]
        else:
            reps = (self.d_state + head_dim - 1) // head_dim
            key_addr = k.repeat(1, 1, 1, reps)[..., : self.d_state]
        key_addr = F.normalize(key_addr, dim=-1, eps=DEFAULT_EPS)
        state = k.new_zeros(batch, n_heads, self.d_state, head_dim)
        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            kt = key_addr[:, :, t]  # [B, H, Ns]
            vt = v[:, :, t]  # [B, H, Dh]
            bt = beta[:, :, t]  # [B, H, 1]
            # Delta rule: S <- S - b * (S k) k^T + b * v k^T  (outer form)
            # sk: [B, H, Dh] = einsum S,k over state axis
            sk = torch.einsum("bhsd,bhs->bhd", state, kt)
            # Remove old association along k, write new v (gated by beta).
            state = state - bt.unsqueeze(-1) * torch.einsum("bhd,bhs->bhsd", sk, kt)
            state = state + bt.unsqueeze(-1) * torch.einsum("bhd,bhs->bhsd", vt, kt)
            yt = torch.einsum("bhsd,bhs->bhd", state, kt)
            outputs.append(yt)
        return torch.stack(outputs, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        batch, seq_len, _dim = x.shape
        conv_in = x.transpose(1, 2)
        conv_out = self.conv1d(conv_in)[..., :seq_len]
        x_conv = self.act(conv_out.transpose(1, 2))

        qkv = self.in_proj(x_conv)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x_conv)).transpose(1, 2).unsqueeze(-1)
        y = self._delta_scan(k, v, beta)
        # Optional query blend (read scale) without O(T^2) attention.
        y = y * (q * self.scale).sigmoid()
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        y = y * self.act(self.gate_proj(x_conv))
        return self.out_proj(y)


class GatedDeltaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_state: int,
        d_conv: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.norm_rec = RMSNorm(dim)
        self.rec = GatedDeltaRecurrence(dim, n_heads, d_state, d_conv)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = GatedMLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.rec(self.norm_rec(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class GatedDeltaLM(nn.Module):
    # Weight-tied gated-delta LM. forward(tokens) -> logits [B, T, V].
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        d_state: int,
        d_conv: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise Exception("n_layers must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        blocks: list[GatedDeltaBlock] = []
        for _ in range(n_layers):
            blocks.append(GatedDeltaBlock(dim, n_heads, d_state, d_conv, mlp_ratio))
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


def build_model(ctx: PrismContext) -> GatedDeltaLM:
    # Pure factory under explore 124M; thrash geometry targets ~1.5-3M params.
    return GatedDeltaLM(
        vocab_size=ctx.vocab_size,
        dim=MODEL_DIM,
        n_layers=MODEL_LAYERS,
        n_heads=MODEL_N_HEADS,
        d_state=MODEL_D_STATE,
        d_conv=MODEL_D_CONV,
        mlp_ratio=MODEL_MLP_RATIO,
    )
