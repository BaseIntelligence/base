# PRISM mamba-tiny ~1M-parameter pure-PyTorch Mamba/SSM architecture script.
#
# This is the architecture half of a v2 two-script bundle: it exposes ONLY the
# build_model(ctx) factory and defines the model. The training loop lives in the
# sibling training.py. A selective state-space (Mamba-style) language model built
# entirely from torch.nn primitives -- no mamba_ssm C++/CUDA extension and no
# torch.utils.cpp_extension -- sized near the Transformer tiny-1m lab seed.
#
# Sandbox contract notes (see evaluator/sandbox.py):
#   * No module-level docstring (a top-level ast.Expr is rejected) -> # comments.
#   * build_model is pure: it never reads data, opens files, or touches the network.
#   * Only allowlisted import roots (torch / allowed stdlib / prism_challenge).
import math

import torch
import torch.nn.functional as F
from torch import nn

from prism_challenge.evaluator.interface import PrismContext

# Geometry chosen to land near ~1M realized params with a 4096-class lab vocab
# (still far under the family-agnostic 124M explore ladder with gpt2's 50257 vocab).
MODEL_DIM = 128
MODEL_LAYERS = 2
MODEL_D_STATE = 16
MODEL_D_CONV = 4
MODEL_EXPAND = 2
MODEL_DT_RANK = 8
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


class SelectiveSSM(nn.Module):
    # Pure-PyTorch selective scan (S6-style). Preferable lab seed path: no blocked
    # native mamba_ssm / custom CUDA extension is required for static or runtime
    # acceptance of this package. Complexity is sequential in T on purpose so the
    # dependency surface stays torch-only.
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
        # A is parameterized in log space (stable negative eigenvalues after softplus/exp).
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.act = nn.SiLU()

        with torch.no_grad():
            # Softplus inverse of a small positive dt bias (stable discretization start).
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
        # u, delta: [B, T, D]; a: [D, N]; b, c: [B, T, N] -> y: [B, T, D]
        batch, seq_len, d_inner = u.shape
        n_state = a.shape[-1]
        # Discretize: dA = exp(delta * A), dB = delta * B (ZOH-style)
        delta_a = torch.exp(delta.unsqueeze(-1) * a.view(1, 1, d_inner, n_state))
        delta_b_u = delta.unsqueeze(-1) * b.unsqueeze(2) * u.unsqueeze(-1)
        # Explicit sequential scan keeps the package free of the mamba_ssm binary.
        state = u.new_zeros(batch, d_inner, n_state)
        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            state = state * delta_a[:, t] + delta_b_u[:, t]
            y_t = torch.einsum("bdn,bn->bd", state, c[:, t])
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        batch, seq_len, _dim = x.shape
        projected = self.in_proj(x)
        x_branch, z_branch = projected.chunk(2, dim=-1)
        # Depthwise causal conv over the token axis.
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


class MambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ssm = SelectiveSSM(dim, d_state, d_conv, expand, dt_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class TinyMambaLM(nn.Module):
    # Weight-tied pure-torch Mamba LM: stacked selective SSM residual blocks with a tied
    # token embedding / output head. forward(tokens) returns next-token logits [B, T, V].
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_rank: int,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise Exception("n_layers must be positive")
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        blocks: list[MambaBlock] = []
        for _ in range(n_layers):
            blocks.append(MambaBlock(dim, d_state, d_conv, expand, dt_rank))
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


def build_model(ctx: PrismContext) -> TinyMambaLM:
    # Pure factory: size the vocabulary from ctx; SSM dims stay small so the
    # parameter count is ~1M (far under ctx.max_params and the 124M explore cap). No
    # native mamba_ssm import is used or required.
    return TinyMambaLM(
        vocab_size=ctx.vocab_size,
        dim=MODEL_DIM,
        n_layers=MODEL_LAYERS,
        d_state=MODEL_D_STATE,
        d_conv=MODEL_D_CONV,
        expand=MODEL_EXPAND,
        dt_rank=MODEL_DT_RANK,
    )
