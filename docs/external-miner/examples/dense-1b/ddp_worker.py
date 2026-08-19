"""Dense 1B DDP worker — one process per GPU via torch.multiprocessing.spawn.

Rendezvous is tcp://127.0.0.1 (never hostname localhost — AF_INET6 errno 97).
No socket/subprocess imports (intake static_source NetworkExfil).

submission_nonce: dense-1b-b200-20260819T1952Z
"""

from __future__ import annotations

import os

import torch.multiprocessing as mp


def _entry(rank, world, port, payload_path):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["DENSE1B_PAYLOAD"] = payload_path
    os.environ["TRITON_CACHE_DIR"] = f"/tmp/dense1b_triton_r{rank}"
    os.environ.setdefault("DENSE1B_PARALLEL", os.environ.get("DENSE1B_PARALLEL", "zero1"))
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    from nemo_automodel.components.models.dense1b.entry import ddp_worker_main

    ddp_worker_main(payload_path=payload_path, rank=rank, world=world, port=port)


def spawn_workers(world, port, payload_path):
    """Parent-side spawn. Children re-import this module (real package path)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    mp.spawn(_entry, nprocs=int(world), args=(int(world), int(port), str(payload_path)), join=True)
