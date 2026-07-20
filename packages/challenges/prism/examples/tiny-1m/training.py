# PRISM tiny ~1M-parameter v2 training script.
#
# This is the training half of a v2 two-script bundle: it exposes train(ctx),
# the miner-owned loop. The challenge re-executes this loop under a forced random
# init on the locked FineWeb-Edu train split and computes the prequential
# bits-per-byte score ITSELF, so any value this script returns is ignored.
#
# The data plane is the challenge's: train(ctx) consumes ctx.iter_train_batches,
# a single-pass, challenge-controlled predict-then-train instrument. The challenge
# records the model's loss on each NEW batch BEFORE the optimizer updates on it
# (that recorded online loss IS the prequential code-length), then the miner
# trains on the same batch. The loop never opens files or touches the network;
# reading the locked data and owning its order belong to the harness.
#
# The loop is single-node multi-GPU safe: it initializes the process group when
# launched under torchrun (world_size > 1), binds local_rank, wraps the model in
# DDP, references DistributedSampler for the static data-sharding primitive, and
# tears the group down on exit. It also works at world_size=1 (scored nproc=1).
import torch
import torch.distributed as dist
import torch.nn.functional as F
from architecture import build_model
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from prism_challenge.evaluator.interface import PrismContext

EFFECTIVE_LEARNING_RATE = 0.005
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
    # Match the challenge instrument: predict tokens[:, 1:] from the leading positions.
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
    # Static multi-GPU data-sharding marker (DistributedSampler). The challenge owns the real
    # FineWeb train order via iter_train_batches; this sampler constructs a zero-cost rank view
    # so the AST contract and a pure-PyTorch DDP script stay aligned up to 8 GPUs / one node.
    _ = DistributedSampler(
        range(ctx.world_size * LOCAL_BATCH),
        num_replicas=ctx.world_size,
        rank=ctx.rank,
    )
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=EFFECTIVE_LEARNING_RATE)
    tokenizer = ctx.reference_tokenizer("gpt2")

    # The challenge owns the data order and the loss capture; it instruments the SCORED model, so
    # pass the unwrapped module and run the optimizer through the DDP wrapper (shared parameters).
    for batch in ctx.iter_train_batches(
        model, batch_size=LOCAL_BATCH, seq_len=ctx.max_seq_len, tokenizer=tokenizer
    ):
        tokens = batch.tokens
        optimizer.zero_grad(set_to_none=True)
        loss = _next_token_loss(wrapped(tokens), tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

    # Rank-0 may persist an artifact under the only writable path. The save path is anchored
    # directly at ctx.artifacts_dir so the sandbox recognizes it as a trusted write. (The challenge
    # also holds the trained model itself for the held-out delta, so this is optional.)
    if ctx.rank == 0 and ctx.artifacts_dir:
        torch.save(model.state_dict(), ctx.artifacts_dir + "/trained_state.pt")
    if initialized:
        dist.barrier()
        dist.destroy_process_group()
