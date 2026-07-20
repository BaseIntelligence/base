# PRISM hybrid-attn-ssm-tiny training script (v2 two-script bundle).
#
# Outer contract matches Imp transformer-tiny / mamba-tiny: forced-init friendly,
# rank-0 save, single-node multi-GPU static primitives. Challenge owns scores.
import torch
import torch.distributed as dist
import torch.nn.functional as F
from architecture import build_model
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from prism_challenge.evaluator.interface import PrismContext

# Between Imp transformer (0.005) and pure-torch SSMscan (0.003) defaults.
EFFECTIVE_LEARNING_RATE = 0.0035
GRAD_CLIP_NORM = 1.0
LOCAL_BATCH = 4


def _maybe_init_distributed(ctx: PrismContext) -> bool:
    if ctx.world_size > 1 and not dist.is_initialized():
        backend = "nccl" if ctx.device.startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend)
        if ctx.device.startswith("cuda"):
            torch.cuda.set_device(ctx.local_rank)
        return True
    return False


def _next_token_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    vocab = logits.shape[-1]
    predictions = logits[:, :-1, :].reshape(-1, vocab)
    targets = tokens[:, 1:].reshape(-1) % vocab
    return F.cross_entropy(predictions, targets)


def train(ctx: PrismContext) -> None:
    torch.manual_seed(ctx.seed)
    initialized = _maybe_init_distributed(ctx)
    if ctx.device.startswith("cuda"):
        torch.cuda.set_device(ctx.local_rank)
    model = build_model(ctx).to(ctx.device)
    wrapped: nn.Module = model
    if ctx.world_size > 1:
        device_ids = [ctx.local_rank] if ctx.device.startswith("cuda") else None
        wrapped = DistributedDataParallel(model, device_ids=device_ids)
    _ = DistributedSampler(
        range(ctx.world_size * LOCAL_BATCH),
        num_replicas=ctx.world_size,
        rank=ctx.rank,
    )
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=EFFECTIVE_LEARNING_RATE)
    tokenizer = ctx.reference_tokenizer("gpt2")

    for batch in ctx.iter_train_batches(
        model, batch_size=LOCAL_BATCH, seq_len=ctx.max_seq_len, tokenizer=tokenizer
    ):
        tokens = batch.tokens
        optimizer.zero_grad(set_to_none=True)
        loss = _next_token_loss(wrapped(tokens), tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

    if ctx.rank == 0 and ctx.artifacts_dir:
        torch.save(model.state_dict(), ctx.artifacts_dir + "/trained_state.pt")
    if initialized:
        dist.barrier()
        dist.destroy_process_group()
