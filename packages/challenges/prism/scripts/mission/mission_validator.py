#!/usr/bin/env python
"""Local mission stub validator: audits disputed prism units by replaying the CPU re-exec.

Part of the cross-repo local end-to-end harness (base docs/operations/mission-harness.md). Enrolls
as a gpu-capable validator (its hotkey holds a validator permit in the mock metagraph). When the
base master disputes a divergent unit it creates a validator-kind audit work unit
(``<uid>:audit``); this agent pulls it, reads the audited submission from
``assignment.payload["audit_of_work_unit_id"]``, replays the SAME deterministic CPU re-exec to
obtain the AUTHORITATIVE ``manifest_sha256``, and posts it. The master's reconciliation then
attributes a ``WorkerFault`` to every replica whose hash diverged from this authoritative one.

CONFIG-DRIVEN (JSON path in ``argv[1]`` / ``$MISSION_VALIDATOR_CONFIG``). NOT for production.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any

from base.validator.agent import (
    AssignmentContext,
    BrokerConfig,
    CoordinationClient,
    ExecutionResult,
    ValidatorAgent,
)
from base.validator.agent.signing import KeypairRequestSigner

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.cpu_test_mode import (
    configure_cpu_reexec_test_mode,
    evaluate_cpu_reexec,
)
from prism_challenge.keypair import keypair_from_uri

AUDIT_OF_PAYLOAD_KEY = "audit_of_work_unit_id"


class AuditReplayExecutor:
    """Replay the deterministic CPU re-exec for an audited submission to get the authoritative hash.

    Satisfies the base :class:`AssignmentExecutor` protocol. Only audit units (validator-kind,
    carrying ``audit_of_work_unit_id``) are meaningful here; the replay re-executes the ORIGINAL
    submission id so the hash matches an honest worker replica exactly.
    """

    def __init__(self, *, settings: PrismSettings) -> None:
        self._settings = settings

    async def execute(self, context: AssignmentContext, *, progress: Any) -> ExecutionResult:
        payload = context.assignment.payload or {}
        audited = payload.get(AUDIT_OF_PAYLOAD_KEY) or context.assignment.work_unit_id
        outcome = await asyncio.to_thread(
            evaluate_cpu_reexec, self._settings, submission_id=str(audited)
        )
        return ExecutionResult(
            success=True,
            payload={
                "execution_proof": {"manifest_sha256": outcome.manifest_sha256},
                "manifest_sha256": outcome.manifest_sha256,
                "audited_submission_id": str(audited),
            },
        )


def _load_config() -> dict[str, Any]:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import os

        path = os.environ.get("MISSION_VALIDATOR_CONFIG")
    if not path:
        raise SystemExit("usage: mission_validator.py <config.json>")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validator_settings(prism: dict[str, Any]) -> PrismSettings:
    token = prism["token"]
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{prism['artifact_root']}/validator.sqlite3",
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
    settings = _validator_settings(prism)
    configure_cpu_reexec_test_mode(settings)

    validator_kp = keypair_from_uri(config["validator_uri"])
    client = CoordinationClient(config["master_url"], KeypairRequestSigner(validator_kp))
    agent = ValidatorAgent(
        client=client,
        executor=AuditReplayExecutor(settings=settings),
        broker=BrokerConfig(broker_url="http://mission-broker:8082"),
        capabilities=config.get("capabilities", ["gpu"]),
        version=config.get("version", "0.1.0"),
        gateway_url=config["master_url"],
        heartbeat_interval_seconds=int(config.get("heartbeat_interval_seconds", 3)),
        poll_interval_seconds=float(config.get("poll_interval_seconds", 1.0)),
    )

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except (NotImplementedError, RuntimeError):
            pass
    await agent.run_forever(shutdown)


def main() -> None:
    asyncio.run(_run(_load_config()))


if __name__ == "__main__":
    main()
