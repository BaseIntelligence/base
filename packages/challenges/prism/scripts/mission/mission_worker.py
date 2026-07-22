#!/usr/bin/env python
"""Local mission worker agent: a base worker plane agent whose executor is the prism CPU re-exec.

Part of the cross-repo local end-to-end harness (base advanced/local harness (not shipping)). Enrolls
as a miner-funded GPU worker under a distinct owner (miner) hotkey, then for each assigned prism
work unit runs the repo's OWN deterministic CPU re-exec (``evaluate_cpu_reexec``), normalizes the
volatile timing fields so honest replicas of the same submission agree on one ``manifest_sha256``,
signs a tier-0 ExecutionProof over ``sha256(manifest_sha256:unit_id)``, and posts it.

A worker may be configured with ``divergence_hotkey``: for a unit submitted by that hotkey it
CORRUPTS its manifest (a distinct byte) so its hash diverges from the honest replica, which is how
the divergence drill provokes a dispute + validator audit. CONFIG-DRIVEN (JSON path in ``argv[1]``
/ ``$MISSION_WORKER_CONFIG``). NOT for production.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from base.validator.agent import AssignmentContext, BrokerConfig, ExecutionResult
from base.validator.agent.signing import KeypairRequestSigner
from base.worker.coordination_client import WorkerCoordinationClient
from base.worker.deploy import build_signed_binding
from base.worker.proof import build_execution_proof
from base.worker.runtime import WorkerAgent

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.cpu_test_mode import (
    configure_cpu_reexec_test_mode,
    evaluate_cpu_reexec,
)
from prism_challenge.keypair import keypair_from_uri
from prism_challenge.proof import compute_manifest_sha256


class CpuReexecWorkerExecutor:
    """Run the prism CPU re-exec for an assigned unit and post a signed ExecutionProof.

    Satisfies the base :class:`AssignmentExecutor` protocol. The unit's ``work_unit_id`` is the
    prism submission id, so both honest replicas (and the auditor) re-execute the SAME submission
    and converge on one normalized ``manifest_sha256``.
    """

    def __init__(
        self,
        *,
        settings: PrismSettings,
        worker_signer: KeypairRequestSigner,
        divergence_hotkey: str | None = None,
    ) -> None:
        self._settings = settings
        self._signer = worker_signer
        self._divergence_hotkey = divergence_hotkey

    async def execute(self, context: AssignmentContext, *, progress: Any) -> ExecutionResult:
        assignment = context.assignment
        unit_id = assignment.work_unit_id
        outcome = await asyncio.to_thread(
            evaluate_cpu_reexec, self._settings, submission_id=unit_id
        )
        manifest = outcome.manifest
        manifest_sha256 = outcome.manifest_sha256
        if self._divergence_hotkey and assignment.submission_ref == self._divergence_hotkey:
            manifest = dict(manifest)
            manifest["mission_divergence_marker"] = self._signer.hotkey
            manifest_sha256 = compute_manifest_sha256(manifest)
        proof = build_execution_proof(
            signer=self._signer, manifest_sha256=manifest_sha256, unit_id=unit_id, tier=0
        )
        payload = {
            "execution_proof": proof.model_dump(mode="json"),
            "manifest_sha256": manifest_sha256,
            "run_manifest": manifest,
        }
        return ExecutionResult(success=True, payload=payload)


def _load_config() -> dict[str, Any]:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import os

        path = os.environ.get("MISSION_WORKER_CONFIG")
    if not path:
        raise SystemExit("usage: mission_worker.py <config.json>")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _worker_settings(prism: dict[str, Any]) -> PrismSettings:
    token = prism["token"]
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{prism['artifact_root']}/worker.sqlite3",
        shared_token=token,
        shared_token_file=None,
        allow_insecure_signatures=True,
        llm_review_enabled=False,
        llm_review_required=False,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://mission-broker:8082",
        docker_broker_token=token,
        base_eval_artifact_root=Path(prism["artifact_root"]),
        worker_plane={
            "enabled": True,
            "cpu_reexec_test_mode": True,
            "cpu_reexec_train_data_dir": prism["train_data_dir"],
            "cpu_reexec_sequence_length": int(prism.get("sequence_length", 16)),
        },
    )


async def _run(config: dict[str, Any]) -> None:
    prism = config["prism"]
    Path(prism["artifact_root"]).mkdir(parents=True, exist_ok=True)
    settings = _worker_settings(prism)
    configure_cpu_reexec_test_mode(settings)

    miner_kp = keypair_from_uri(config["miner_uri"])
    worker_kp = keypair_from_uri(config["worker_uri"])
    worker_signer = KeypairRequestSigner(worker_kp)
    binding = build_signed_binding(
        worker_pubkey=worker_kp.ss58_address,
        miner_signer=KeypairRequestSigner(miner_kp),
    )
    client = WorkerCoordinationClient(config["master_url"], worker_signer)
    executor = CpuReexecWorkerExecutor(
        settings=settings,
        worker_signer=worker_signer,
        divergence_hotkey=config.get("divergence_hotkey"),
    )
    agent = WorkerAgent(
        client=client,
        executor=executor,
        broker=BrokerConfig(broker_url="http://mission-broker:8082"),
        binding=binding,
        provider=config.get("provider", "local"),
        provider_instance_ref=config.get("name"),
        capabilities=config.get("capabilities", ["gpu"]),
        gateway_url=config["master_url"],
        heartbeat_interval_seconds=int(config.get("heartbeat_interval_seconds", 3)),
        poll_interval_seconds=float(config.get("poll_interval_seconds", 1.0)),
    )

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with_suppress = getattr(loop, "add_signal_handler", None)
        if with_suppress is not None:
            try:
                loop.add_signal_handler(sig, shutdown.set)
            except (NotImplementedError, RuntimeError):
                pass
    await agent.run_forever(shutdown)


def _flush_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (ValueError, OSError):
            pass


def _configure_harness_logging(level: int = logging.INFO) -> None:
    """Line-buffer stdout/stderr and route logs there so drill logs are
    inspectable after teardown.

    The harness redirects each spawned process' stdout/stderr to a log file and
    tears it down with SIGTERM; block-buffered output would be lost on kill,
    leaving a 0-byte log. Line-buffering flushes every completed log line
    immediately, and an ``atexit`` flush covers the graceful-shutdown path.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(line_buffering=True)
            except (ValueError, OSError):
                pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    atexit.register(_flush_streams)


def main() -> None:
    _configure_harness_logging()
    asyncio.run(_run(_load_config()))


if __name__ == "__main__":
    main()
