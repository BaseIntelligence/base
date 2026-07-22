#!/usr/bin/env python
"""Local mission LEGACY validator: run an assigned prism PRIMARY unit via ``validator_dispatch``.

Part of the cross-repo local end-to-end harness (base ``advanced/local harness (not shipping)``); used
by the flags-OFF legacy regression smoke (VAL-CROSS-006). Enrolls as a gpu-capable validator (its
hotkey holds a validator permit in the mock metagraph). With the worker plane OFF the base master
routes the single prism gpu work unit to this VALIDATOR (never a worker); the agent pulls it and
runs the REAL prism ``validator_dispatch`` path (``dispatch_assignment`` ->
``run_primary_execution_cycle``) on the deterministic CPU re-exec seam, finalizing the score in the
SHARED prism database exactly as a pre-mission decentralized validator would.

The base master (mock metagraph) mints no scoped gateway token in this harness, so a no-op gateway
token is injected here BEFORE dispatch (prism LLM review is disabled, so it is never used) purely to
satisfy the primary-path gateway config; nothing is sent to any real gateway.

CONFIG-DRIVEN (JSON path in ``argv[1]`` / ``$MISSION_LEGACY_VALIDATOR_CONFIG``). NOT for production.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from base.validator.agent import BrokerConfig, CoordinationClient, ValidatorAgent
from base.validator.agent.adapters.prism import PrismCycleExecutor
from base.validator.agent.challenge_dispatch import ChallengeDispatchExecutor
from base.validator.agent.signing import KeypairRequestSigner

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.cpu_test_mode import configure_cpu_reexec_test_mode
from prism_challenge.keypair import keypair_from_uri
from prism_challenge.validator_dispatch import CHALLENGE_SLUG, dispatch_assignment

# A no-op gateway token/URL: the primary dispatch path requires a scoped gateway config, but the
# prism LLM review is disabled in this harness so the token is never used and no gateway is called.
_NOOP_GATEWAY_TOKEN = "mission-legacy-noop-token"
_NOOP_GATEWAY_URL = "http://127.0.0.1:3199/llm/v1"


def _load_config() -> dict[str, Any]:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import os

        path = os.environ.get("MISSION_LEGACY_VALIDATOR_CONFIG")
    if not path:
        raise SystemExit("usage: mission_legacy_validator.py <config.json>")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _legacy_settings(prism: dict[str, Any]) -> PrismSettings:
    """Prism settings for the legacy validator: SHARED db, worker plane OFF, CPU re-exec seam."""

    token = prism["token"]
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{prism['db_path']}",
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
        sequence_length=int(prism.get("sequence_length", 16)),
        base_eval_artifact_root=Path(prism["artifact_root"]),
        worker_plane={
            "enabled": False,
            "cpu_reexec_test_mode": True,
            "cpu_reexec_train_data_dir": prism["train_data_dir"],
            "cpu_reexec_sequence_length": int(prism.get("sequence_length", 16)),
        },
    )


def _build_dispatch(settings: PrismSettings):
    """Bind the real prism dispatch to the legacy settings + a no-op gateway token."""

    async def _dispatch(
        *,
        work_unit_id: str,
        payload: Mapping[str, Any],
        broker_url: str,
        broker_token: str | None = None,
        broker_token_file: str | None = None,
    ) -> Mapping[str, Any]:
        print(
            f"[legacy-validator] pulled + executing prism unit {work_unit_id} "
            "via validator_dispatch",
            flush=True,
        )
        enriched = dict(payload)
        enriched.setdefault("gateway_token", _NOOP_GATEWAY_TOKEN)
        enriched.setdefault("gateway_url", _NOOP_GATEWAY_URL)
        result = await dispatch_assignment(
            work_unit_id=work_unit_id,
            payload=enriched,
            broker_url=broker_url,
            broker_token=broker_token,
            broker_token_file=broker_token_file,
            settings=settings,
        )
        print(f"[legacy-validator] dispatch result for {work_unit_id}: {dict(result)}", flush=True)
        return result

    return _dispatch


async def _run(config: dict[str, Any]) -> None:
    prism = config["prism"]
    Path(prism["artifact_root"]).mkdir(parents=True, exist_ok=True)
    settings = _legacy_settings(prism)
    configure_cpu_reexec_test_mode(settings)

    validator_kp = keypair_from_uri(config["validator_uri"])
    client = CoordinationClient(config["master_url"], KeypairRequestSigner(validator_kp))
    executor = ChallengeDispatchExecutor(
        executors={CHALLENGE_SLUG: PrismCycleExecutor(dispatch=_build_dispatch(settings))}
    )
    agent = ValidatorAgent(
        client=client,
        executor=executor,
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
