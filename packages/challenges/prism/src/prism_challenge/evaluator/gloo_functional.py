"""Gloo multi-rank functional test for the multi-GPU contract (architecture.md section 8, Gate B).

With a single physical GPU the distributed code path cannot be validated for real NCCL scaling, so
correctness is validated two ways: a static AST contract check (:mod:`distributed_contract`, Gate A)
and -- here -- a functional multi-rank run on CPU with the **gloo** backend (Gate B). This module
drives a representative miner-style DDP training loop under a forced seed at ``WORLD_SIZE`` 2 and 4
and records the diagnostics a validator asserts on:

* the online (world-averaged) loss decreases AND parameters stay byte-identical across ranks, which
  proves DDP gradient synchronization is correct (VAL-GPU-009 / VAL-GPU-010);
* per-rank data sharding is disjoint -- no rank sees another rank's batches (VAL-GPU-011);
* clean collective teardown -- every rank reaches ``barrier()`` then ``destroy_process_group()`` and
  exits 0 with no hang/orphan (VAL-GPU-012);
* all-reduced metrics (loss, tokens_seen) are world-consistent (VAL-GPU-013);
* only rank 0 writes the checkpoint/manifest into ``artifacts_dir`` (VAL-GPU-015).

An advisory NCCL ``nproc=2`` single-GPU launch (``NCCL_P2P_DISABLE=1``) is provided separately; it
is ADVISORY (single-device NCCL is a non-standard config that can hang) and never gates a
submission (VAL-GPU-014).

The ranks run in spawned subprocesses, so the worker entrypoints and the tiny model live at module
scope (the ``spawn`` start method re-imports this module in each child). Each rank writes a small
JSON diagnostics file; the parent collects them into a :class:`GlooFunctionalResult`.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

DEFAULT_SEED = 1337
DEFAULT_NUM_SAMPLES = 64
DEFAULT_SEQ_LEN = 8
DEFAULT_VOCAB_SIZE = 32
DEFAULT_DIM = 16
DEFAULT_BATCH_SIZE = 4
DEFAULT_STEPS = 30
DEFAULT_LR = 0.05
DEFAULT_TIMEOUT_S = 180.0
MANIFEST_SCHEMA_VERSION = "prism_run_manifest.v2"


@dataclass(frozen=True)
class GlooRankDiagnostics:
    """Per-rank diagnostics captured during the functional run."""

    rank: int
    world_size: int
    reduced_loss_first: float
    reduced_loss_last: float
    param_hash: str
    consumed_indices: tuple[int, ...]
    local_tokens_seen: int
    reduced_tokens_seen: int
    wrote_artifacts: bool
    clean_exit: bool


@dataclass(frozen=True)
class GlooFunctionalResult:
    """Aggregated outcome of a gloo multi-rank functional run."""

    world_size: int
    num_samples: int
    ranks: tuple[GlooRankDiagnostics, ...]
    exit_codes: tuple[int, ...]
    artifacts_files: tuple[str, ...]
    manifest_valid: bool
    duration_s: float

    @property
    def params_synced(self) -> bool:
        """True iff every rank holds byte-identical parameters after gradient sync."""
        return bool(self.ranks) and len({rank.param_hash for rank in self.ranks}) == 1

    @property
    def world_loss_decreased(self) -> bool:
        """True iff the world-averaged loss decreased on every rank."""
        return bool(self.ranks) and all(
            rank.reduced_loss_last < rank.reduced_loss_first for rank in self.ranks
        )

    @property
    def sharding_disjoint(self) -> bool:
        """True iff per-rank consumed sample indices are disjoint and cover the dataset."""
        seen: set[int] = set()
        union: set[int] = set()
        for rank in self.ranks:
            idx = set(rank.consumed_indices)
            if not idx or (seen & idx):
                return False
            seen |= idx
            union |= idx
        return union == set(range(self.num_samples))

    @property
    def metrics_world_consistent(self) -> bool:
        """True iff all-reduced metrics match across ranks and tokens == per-rank sum."""
        if not self.ranks:
            return False
        reduced_tokens = {rank.reduced_tokens_seen for rank in self.ranks}
        reduced_losses = {round(rank.reduced_loss_last, 6) for rank in self.ranks}
        if len(reduced_tokens) != 1 or len(reduced_losses) != 1:
            return False
        return next(iter(reduced_tokens)) == sum(rank.local_tokens_seen for rank in self.ranks)

    @property
    def clean_teardown(self) -> bool:
        """True iff every rank exited 0 and reached barrier + destroy_process_group."""
        return (
            bool(self.exit_codes)
            and all(code == 0 for code in self.exit_codes)
            and bool(self.ranks)
            and all(rank.clean_exit for rank in self.ranks)
        )

    @property
    def rank0_is_sole_writer(self) -> bool:
        """True iff rank 0 is the only rank that wrote artifacts and exactly one set exists."""
        if not self.ranks or not self.ranks[0].wrote_artifacts:
            return False
        if any(rank.wrote_artifacts for rank in self.ranks[1:]):
            return False
        return sorted(self.artifacts_files) == ["checkpoint.pt", "prism_run_manifest.v2.json"]


@dataclass(frozen=True)
class NcclAdvisoryResult:
    """Outcome of the advisory single-GPU NCCL ``nproc`` launch (never a gate)."""

    attempted: bool
    cuda_available: bool
    succeeded: bool
    ranks_launched: int
    p2p_disabled: bool
    detail: str
    diagnostics: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class _TinyLM(nn.Module):
    """A tiny token-embedding + linear-head next-token model used for the functional run."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(tokens))


def _build_dataset(num_samples: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    """Build a deterministic, easily-learnable next-token dataset (token[t+1] == token[t] + 1)."""
    rows = [[(i + j) % vocab_size for j in range(seq_len)] for i in range(num_samples)]
    return torch.tensor(rows, dtype=torch.long)


def _param_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _all_reduce_mean(value: torch.Tensor) -> float:
    reduced = value.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return float(reduced.item())


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _gloo_rank_worker(rank: int, config: dict[str, Any]) -> None:
    """Run one rank of the gloo functional DDP loop and write its diagnostics file.

    Mirrors the multi-GPU miner contract: init_process_group(gloo) -> DistributedSampler shard ->
    DDP-wrapped model -> all-reduced metrics -> rank-0-only artifact write -> barrier +
    destroy_process_group. Uses ``MASTER_ADDR=127.0.0.1`` to avoid the c10d hostname rendezvous
    hang seen in containers.
    """
    world_size = int(config["world_size"])
    vocab_size = int(config["vocab_size"])
    seq_len = int(config["seq_len"])

    os.environ["MASTER_ADDR"] = str(config["master_addr"])
    os.environ["MASTER_PORT"] = str(config["master_port"])
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.manual_seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    clean_exit = False
    try:
        dataset = TensorDataset(_build_dataset(int(config["num_samples"]), seq_len, vocab_size))
        sampler: DistributedSampler[int] = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        consumed_indices = sorted({int(i) for i in sampler})
        loader = DataLoader(dataset, batch_size=int(config["batch_size"]), sampler=sampler)

        torch.manual_seed(int(config["seed"]))
        model = _TinyLM(vocab_size, int(config["dim"]))
        ddp_model = DDP(model)
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=float(config["lr"]))

        reduced_loss_first: float | None = None
        reduced_loss_last = 0.0
        local_tokens_seen = 0
        for _ in range(int(config["steps"])):
            for (batch,) in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = ddp_model(batch)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, vocab_size),
                    batch[:, 1:].reshape(-1),
                )
                world_loss = _all_reduce_mean(loss.detach())
                if reduced_loss_first is None:
                    reduced_loss_first = world_loss
                reduced_loss_last = world_loss
                loss.backward()
                optimizer.step()
                local_tokens_seen += int(batch.numel())

        tokens_tensor = torch.tensor([local_tokens_seen], dtype=torch.long)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        reduced_tokens_seen = int(tokens_tensor.item())

        param_hash = _param_hash(model)

        wrote_artifacts = False
        artifacts_dir = config.get("artifacts_dir")
        if rank == 0 and artifacts_dir:
            torch.save(model.state_dict(), os.path.join(artifacts_dir, "checkpoint.pt"))
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "backend": "gloo",
                "world_size": world_size,
                "writer_rank": rank,
                "reduced_loss_first": reduced_loss_first,
                "reduced_loss_last": reduced_loss_last,
                "reduced_tokens_seen": reduced_tokens_seen,
                "param_hash": param_hash,
            }
            with open(
                os.path.join(artifacts_dir, "prism_run_manifest.v2.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(manifest, handle, sort_keys=True)
            wrote_artifacts = True

        dist.barrier()
        clean_exit = True
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    diagnostics = {
        "rank": rank,
        "world_size": world_size,
        "reduced_loss_first": reduced_loss_first,
        "reduced_loss_last": reduced_loss_last,
        "param_hash": param_hash,
        "consumed_indices": consumed_indices,
        "local_tokens_seen": local_tokens_seen,
        "reduced_tokens_seen": reduced_tokens_seen,
        "wrote_artifacts": wrote_artifacts,
        "clean_exit": clean_exit,
    }
    with open(
        os.path.join(config["diag_dir"], f"rank_{rank}.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(diagnostics, handle)


def run_gloo_functional(
    world_size: int,
    *,
    seed: int = DEFAULT_SEED,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seq_len: int = DEFAULT_SEQ_LEN,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    dim: int = DEFAULT_DIM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> GlooFunctionalResult:
    """Drive a ``world_size``-rank gloo DDP functional run on CPU and collect diagnostics.

    ``num_samples`` must be divisible by ``world_size`` so the per-rank shards are disjoint with no
    padding duplication. The run is bounded by ``timeout_s``: a deadlocked/hung rank is terminated
    and a :class:`TimeoutError` is raised rather than hanging the suite.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if num_samples % world_size != 0:
        raise ValueError("num_samples must be divisible by world_size for disjoint sharding")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="prism-gloo-") as workdir:
        diag_dir = os.path.join(workdir, "diag")
        artifacts_dir = os.path.join(workdir, "artifacts")
        os.makedirs(diag_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

        config: dict[str, Any] = {
            "world_size": world_size,
            "seed": seed,
            "num_samples": num_samples,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "dim": dim,
            "batch_size": batch_size,
            "steps": steps,
            "lr": lr,
            "master_addr": "127.0.0.1",
            "master_port": _free_port(),
            "diag_dir": diag_dir,
            "artifacts_dir": artifacts_dir,
        }

        start = time.monotonic()
        spawn_context = mp.start_processes(
            _gloo_rank_worker,
            args=(config,),
            nprocs=world_size,
            join=False,
            start_method="spawn",
        )
        while not spawn_context.join(timeout=2.0):
            if time.monotonic() - start > timeout_s:
                for process in spawn_context.processes:
                    if process.is_alive():
                        process.terminate()
                raise TimeoutError(
                    f"gloo functional run (world_size={world_size}) exceeded {timeout_s}s "
                    "(possible collective hang)"
                )
        duration_s = time.monotonic() - start
        exit_codes = tuple(int(process.exitcode) for process in spawn_context.processes)

        ranks = tuple(_read_rank_diagnostics(diag_dir, rank) for rank in range(world_size))
        artifacts_files = tuple(sorted(os.listdir(artifacts_dir)))
        manifest_valid = _manifest_valid(artifacts_dir)

    return GlooFunctionalResult(
        world_size=world_size,
        num_samples=num_samples,
        ranks=ranks,
        exit_codes=exit_codes,
        artifacts_files=artifacts_files,
        manifest_valid=manifest_valid,
        duration_s=duration_s,
    )


def _read_rank_diagnostics(diag_dir: str, rank: int) -> GlooRankDiagnostics:
    with open(os.path.join(diag_dir, f"rank_{rank}.json"), encoding="utf-8") as handle:
        raw = json.load(handle)
    return GlooRankDiagnostics(
        rank=int(raw["rank"]),
        world_size=int(raw["world_size"]),
        reduced_loss_first=float(raw["reduced_loss_first"]),
        reduced_loss_last=float(raw["reduced_loss_last"]),
        param_hash=str(raw["param_hash"]),
        consumed_indices=tuple(int(i) for i in raw["consumed_indices"]),
        local_tokens_seen=int(raw["local_tokens_seen"]),
        reduced_tokens_seen=int(raw["reduced_tokens_seen"]),
        wrote_artifacts=bool(raw["wrote_artifacts"]),
        clean_exit=bool(raw["clean_exit"]),
    )


def _manifest_valid(artifacts_dir: str) -> bool:
    path = os.path.join(artifacts_dir, "prism_run_manifest.v2.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION


def _nccl_rank_worker(rank: int, config: dict[str, Any]) -> None:
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["MASTER_ADDR"] = str(config["master_addr"])
    os.environ["MASTER_PORT"] = str(config["master_port"])
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(config["world_size"])
    os.environ["LOCAL_RANK"] = "0"

    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=int(config["world_size"]))
    try:
        tensor = torch.ones(1, device="cuda:0")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        with open(
            os.path.join(config["diag_dir"], f"nccl_rank_{rank}.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump({"rank": rank, "allreduce": float(tensor.item())}, handle)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_nccl_advisory(nproc: int = 2, *, timeout_s: float = 60.0) -> NcclAdvisoryResult:
    """Attempt an advisory NCCL ``nproc``-rank launch time-sharing one GPU (``NCCL_P2P_DISABLE=1``).

    This is ADVISORY ONLY (VAL-GPU-014): single-device NCCL is a non-standard config that can hang,
    so any failure is captured and returned -- this function never raises and never gates a
    submission. On a host without a usable CUDA device the launch is skipped.
    """
    if not torch.cuda.is_available():
        return NcclAdvisoryResult(
            attempted=False,
            cuda_available=False,
            succeeded=False,
            ranks_launched=0,
            p2p_disabled=True,
            detail="no CUDA device available; advisory NCCL nproc launch skipped (non-gating)",
        )

    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="prism-nccl-") as workdir:
            diag_dir = os.path.join(workdir, "diag")
            os.makedirs(diag_dir, exist_ok=True)
            config: dict[str, Any] = {
                "world_size": nproc,
                "master_addr": "127.0.0.1",
                "master_port": _free_port(),
                "diag_dir": diag_dir,
            }
            spawn_context = mp.start_processes(
                _nccl_rank_worker,
                args=(config,),
                nprocs=nproc,
                join=False,
                start_method="spawn",
            )
            start = time.monotonic()
            while not spawn_context.join(timeout=2.0):
                if time.monotonic() - start > timeout_s:
                    for process in spawn_context.processes:
                        if process.is_alive():
                            process.terminate()
                    return NcclAdvisoryResult(
                        attempted=True,
                        cuda_available=True,
                        succeeded=False,
                        ranks_launched=nproc,
                        p2p_disabled=True,
                        detail=f"advisory NCCL launch timed out after {timeout_s}s (non-gating)",
                    )
            diagnostics = tuple(
                _load_nccl_diag(diag_dir, rank)
                for rank in range(nproc)
                if os.path.isfile(os.path.join(diag_dir, f"nccl_rank_{rank}.json"))
            )
            return NcclAdvisoryResult(
                attempted=True,
                cuda_available=True,
                succeeded=len(diagnostics) == nproc,
                ranks_launched=len(diagnostics),
                p2p_disabled=True,
                detail=f"advisory NCCL launch completed with {len(diagnostics)}/{nproc} ranks",
                diagnostics=diagnostics,
            )
    except Exception as exc:  # noqa: BLE001 - advisory path must never raise/gate
        return NcclAdvisoryResult(
            attempted=True,
            cuda_available=True,
            succeeded=False,
            ranks_launched=0,
            p2p_disabled=True,
            detail=f"advisory NCCL launch failed (non-gating): {exc}",
        )


def _load_nccl_diag(diag_dir: str, rank: int) -> dict[str, Any]:
    with open(os.path.join(diag_dir, f"nccl_rank_{rank}.json"), encoding="utf-8") as handle:
        return dict(json.load(handle))
