#!/usr/bin/env python
"""In-pod worker agent runner for the live Lium end-to-end (VAL-CROSS-005).

Runs INSIDE a rented Lium pod (the worker image) and enrolls the miner-funded
:class:`base.worker.runtime.WorkerAgent` with a LOCAL mission master reached over
a reverse SSH tunnel (``master_url`` = ``http://127.0.0.1:<rport>``). It reuses the
real worker plane end-to-end -- the worker keypair signs coordination requests +
the ExecutionProof, the miner-signed binding (pre-signed on the host so the pod
never holds the miner key) authenticates enrollment, and the CPU
:class:`StubManifestExecutor` executes each pulled gpu unit into a deterministic
manifest hash. Every emitted ExecutionProof carries the LIUM provider provenance
(provider name + the REAL pod id + the pinned image digest), so the master records
a proof that names the pod it ran in.

CONFIG-DRIVEN (JSON path in ``argv[1]``). The signing keypair is provided by the
Rust-backed ``bittensor_wallet`` (no torch), falling back to ``bittensor`` when
present; both yield the same ss58 address + sr25519 signatures. NOT for
production.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

try:  # light, torch-free keypair (Rust wheel) preferred inside the pod
    from bittensor_wallet import Keypair
except ImportError:  # pragma: no cover - host dry-run may only have full bittensor
    from bittensor import Keypair  # type: ignore[no-redef]

from base.validator.agent import BrokerConfig
from base.validator.agent.signing import KeypairRequestSigner
from base.worker.coordination_client import WorkerCoordinationClient
from base.worker.executor import (
    StubManifestExecutor,
    WorkerProofExecutor,
    WorkerProvenance,
)
from base.worker.runtime import WorkerAgent, WorkerBinding


def _load_config() -> dict[str, Any]:
    if len(sys.argv) < 2:
        raise SystemExit("usage: pod_worker_agent.py <config.json>")
    return json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))


def _build_agent(config: dict[str, Any]) -> WorkerAgent:
    worker_keypair = Keypair.create_from_uri(config["worker_uri"])
    signer = KeypairRequestSigner(worker_keypair)
    binding = WorkerBinding(
        miner_hotkey=str(config["miner_hotkey"]),
        signature=str(config["binding_signature"]),
        nonce=str(config["binding_nonce"]),
    )
    provenance = WorkerProvenance(
        provider_name=str(config.get("provider", "lium")),
        miner_hotkey=str(config["miner_hotkey"]),
        executor_id=config.get("executor_id"),
        pod_id=str(config["pod_id"]),
        image_digest=config.get("image_digest"),
    )
    executor = WorkerProofExecutor(
        StubManifestExecutor(), signer=signer, provenance=provenance
    )
    master_url = str(config["master_url"])
    client = WorkerCoordinationClient(
        master_url,
        signer,
        timeout_seconds=float(config.get("request_timeout_seconds", 15.0)),
    )
    return WorkerAgent(
        client=client,
        executor=executor,
        broker=BrokerConfig(
            broker_url=str(config.get("broker_url", "http://127.0.0.1:0"))
        ),
        binding=binding,
        provider=str(config.get("provider", "lium")),
        provider_instance_ref=str(config["pod_id"]),
        capabilities=list(config.get("capabilities", ["gpu"])),
        gateway_url=master_url,
        heartbeat_interval_seconds=int(config.get("heartbeat_interval_seconds", 5)),
        poll_interval_seconds=float(config.get("poll_interval_seconds", 2.0)),
    )


async def _run(config: dict[str, Any]) -> None:
    agent = _build_agent(config)
    print(f"[pod-agent] worker_pubkey={agent.worker_pubkey}", flush=True)
    shutdown = asyncio.Event()

    run_seconds = float(config.get("run_seconds", 120.0))

    async def _deadline() -> None:
        await asyncio.sleep(run_seconds)
        shutdown.set()

    worker_id = await agent.register(shutdown)
    print(f"[pod-agent] registered worker_id={worker_id}", flush=True)

    async def _heartbeat() -> None:
        while not shutdown.is_set():
            try:
                await agent.heartbeat_once()
            except Exception as exc:  # noqa: BLE001 - keep heartbeating
                print(f"[pod-agent] heartbeat error: {exc}", flush=True)
            await asyncio.sleep(agent.heartbeat_interval)

    async def _work() -> None:
        while not shutdown.is_set():
            try:
                summary = await agent.process_pending_assignments()
                if summary.pulled:
                    print(
                        f"[pod-agent] cycle pulled={summary.pulled} "
                        f"completed={summary.completed} failed={summary.failed}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001 - keep polling
                print(f"[pod-agent] assignment error: {exc}", flush=True)
            await asyncio.sleep(float(config.get("poll_interval_seconds", 2.0)))

    await asyncio.gather(_deadline(), _heartbeat(), _work())
    print("[pod-agent] shutdown", flush=True)


def main() -> None:
    config = _load_config()
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
