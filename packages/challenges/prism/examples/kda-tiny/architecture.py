# PRISM kda-tiny ~3-10M-parameter Kimi Delta Attention (KDA) architecture.
#
# Architecture half of a v2 two-script bundle. Implements a pure-torch sequential
# Kimi Delta Attention class recurrence (Kimi Linear arXiv 2510.26692; refines
# Gated DeltaNet 2412.06464) with **channel-wise** forget gates on the finite
# state memory. Distinct from the softer gated-delta-tiny pack. NOT full Kimi
# Linear-48B / K2 / "K3" production weights. No flash-linear-attention / FLA
# kernel required for correctness.
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
MODEL_N_HEADS = 4
MODEL_D_STATE = 48
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


class KimiDeltaAttention(nn.Module):
    # Pure-PyTorch sequential KDA-class update (Kimi Linear finish).
    # Differences vs soft gated-delta:
    #   * channel_gate: per-channel (head_dim) forget on state ranging + write
    #   * separate beta write gate and channel_gate forget (finer than scalar beta)
    #   * state address uses learned key→d_state map (not pad/tile)
    def __init__(self, dim: int, n_heads: int, d_state: int, d_conv: int) -> None:
        super().__init__()
        if dim <= 0 or n_heads <= 0 or d_state <= 0 or d_conv <= 0:
            raise Exception("KimiDeltaAttention dimensions must be positive")
        if dim % n_heads != 0:
            raise Exception("dim must be divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.d_state = d_state
        self.d_conv = d_conv
        self.scale = self.head_dim**-0.5

        self.in_proj = nn.Linear(dim, dim * 3, bias=False)
        # Write strength per head (scalar beta, KDA still has a write gate).
        self.beta_proj = nn.Linear(dim, n_heads, bias=True)
        # channel_gate: per-head, per-channel forget (Kimi fine-grained finish).
        self.channel_gate = nn.Linear(dim, n_heads * self.head_dim, bias=True)
        # Learned address map head_dim -> d_state (not pad/tile).
        self.key_addr = nn.Linear(self.head_dim, d_state, bias=False)
        self.out_gate = nn.Linear(dim, dim, bias=False)
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
            # Soft write + mild forget at init.
            self.beta_proj.bias.fill_(-math.log(3.0))
            self.channel_gate.bias.fill_(math.log(3.0))

    def _kda_scan(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        channel_gate: torch.Tensor,
    ) -> torch.Tensor:
        # k,v: [B, H, T, Dh]; beta: [B, H, T, 1]; channel_gate: [B, H, T, Dh]
        # State S is [B, H, d_state, Dh].
        batch, n_heads, seq_len, head_dim = k.shape
        # Learned address into d_state.
        key_addr = self.key_addr(k)
        key_addr = F.normalize(key_addr, dim=-1, eps=DEFAULT_EPS)
        state = k.new_zeros(batch, n_heads, self.d_state, head_dim)
        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            kt = key_addr[:, :, t]  # [B, H, Ns]
            vt = v[:, :, t]  # [B, H, Dh]
            bt = beta[:, :, t]  # [B, H, 1]
            cg = channel_gate[:, :, t]  # [B, H, Dh]
            # Channel-wise forget melts prior state association (KDA finish).
            forget = cg.unsqueeze(2)  # [B, H, 1, Dh]
            state = state * forget
            # Delta write: remove old assoc along k, write gated v.
            sk = torch.einsum("bhsd,bhs->bhd", state, kt)
            state = state - bt.unsqueeze(-1) * torch.einsum("bhd,bhs->bhsd", sk, kt)
            state = state + bt.unsqueeze(-1) * torch.einsum("bhd,bhs->bhsd", vt * cg, kt)
            yt = torch.einsum("bhsd,bhs->bhd", state, kt)
            outputs.append(yt)
        return torch.stack(outputs, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # channel_gate in (0,1) per channel — forget retaininess.
        cg = torch.sigmoid(self.channel_gate(x_conv))
        cg = cg.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        y = self._kda_scan(k, v, beta, cg)
        y = y * (q * self.scale).sigmoid()
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        y = y * self.act(self.out_gate(x_conv))
        return self.out_proj(y)


class KDABlock(nn.Module):
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
        self.rec = KimiDeltaAttention(dim, n_heads, d_state, d_conv)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = GatedMLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.rec(self.norm_rec(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class KDALM(nn.Module):
    # Weight-tied Kimi Delta Attention LM. forward(tokens) -> logits [B, T, V].
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
        blocks: list[KDABlock] = []
        for _ in range(n_layers):
            blocks.append(KDABlock(dim, n_heads, d_state, d_conv, mlp_ratio))
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


def build_model(ctx: PrismContext) -> KDALM:
    # Pure factory under explore 124M; thrash geometry targets ~3-10M params.
    # channel_gate module attribute is the Kimi-class mechanism signature.
    return KDALM(
        vocab_size=ctx.vocab_size,
        dim=MODEL_DIM,
        n_layers=MODEL_LAYERS,
        n_heads=MODEL_N_HEADS,
        d_state=MODEL_D_STATE,
        d_conv=MODEL_D_CONV,
        mlp_ratio=MODEL_MLP_RATIO,
    )
