#!/usr/bin/env python
"""Local mission master service: mock-metagraph base master with the worker plane ON.

Part of the cross-repo local end-to-end harness (see
``docs/operations/mission-harness.md``). Stands up a real base master proxy API on a
loopback port with:

* a STATIC (no-chain) mock metagraph seeded from config -- owner miner hotkeys (no
  permit) plus a stub validator hotkey (with a permit) so signed requests authenticate;
* the miner-funded GPU worker plane wired ON (register/heartbeat/pull/result + fleet
  read);
* the validator coordination + assignment-coordination planes wired ON (so a stub
  validator can be assigned + post audit results);
* the live orchestration driver bridging a single challenge's HTTP-exposed pending work
  units into ``work_assignments``, running balanced worker assignment + reconciliation,
  and forwarding accepted results back to the challenge.

It is CONFIG-DRIVEN (a JSON file path in ``argv[1]`` or ``$MISSION_MASTER_CONFIG``) and
holds no real chain/provider secrets. NOT for production: production uses
``base master proxy`` with a real subtensor-backed metagraph.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn

from base.bittensor.metagraph_cache import MetagraphCache
from base.db.base import Base
from base.db.session import create_engine, create_session_factory
from base.master.app_proxy import create_proxy_app
from base.master.assignment import CAPABILITY_GPU, AssignmentService
from base.master.assignment_coordination import AssignmentCoordinationService
from base.master.challenge_work_source import (
    HttpChallengeFoldTrigger,
    HttpChallengeResultForwarder,
    HttpChallengeWorkSource,
)
from base.master.orchestration import MasterOrchestrationDriver
from base.master.validator_coordination import ValidatorCoordinationService
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_assignment_engine import WorkerAssignmentEngine
from base.master.worker_coordination import WorkerCoordinationService
from base.master.worker_reconciliation import WorkerReconciliationService
from base.master.worker_unit_status import WorkerUnitStatusService
from base.security.miner_auth import SqlAlchemyMinerNonceStore
from base.security.validator_auth import (
    MetagraphValidatorEligibility,
    SqlAlchemyValidatorNonceStore,
    ValidatorSignedRequestVerifier,
)
from base.security.worker_auth import (
    CoordinationReadEligibility,
    MetagraphMinerMembership,
    RegisteredWorkerEligibility,
    SqlAlchemyWorkerNonceStore,
    WorkerSignedRequestVerifier,
)


@dataclass(frozen=True)
class _ChallengeRecord:
    """The minimal challenge descriptor the HTTP work-source seams read."""

    slug: str
    internal_base_url: str
    status: Any = None


class _MockRegistry:
    """A one-challenge registry mapping a slug to a challenge's internal URL + token."""

    def __init__(self, *, slug: str, internal_base_url: str, token: str) -> None:
        self._record = _ChallengeRecord(slug=slug, internal_base_url=internal_base_url)
        self._token = token

    def list(self, *, active_only: bool = False) -> list[_ChallengeRecord]:
        return [self._record]

    def get(self, slug: str) -> _ChallengeRecord:
        if slug != self._record.slug:
            raise KeyError(slug)
        return self._record

    def get_token(self, slug: str) -> str:
        return self._token


def _load_config() -> dict[str, Any]:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import os

        path = os.environ.get("MISSION_MASTER_CONFIG")
    if not path:
        raise SystemExit("usage: mission_master.py <config.json>")
    return json.loads(Path(path).read_text(encoding="utf-8"))


async def _init_db(db_url: str) -> None:
    engine = create_engine(db_url)
    import base.db.models  # noqa: F401  (register ORM tables on Base.metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


def _seed_metagraph(config: dict[str, Any]) -> MetagraphCache:
    entries = config["metagraph"]
    cache = MetagraphCache(netuid=int(config.get("netuid", 100)), static=True)
    cache.update_from_metagraph(
        [str(entry["hotkey"]) for entry in entries],
        uids=[int(entry.get("uid", index)) for index, entry in enumerate(entries)],
        validator_permits=[
            bool(entry.get("validator_permit", False)) for entry in entries
        ],
        stakes=[float(entry.get("stake", 0.0)) for entry in entries],
    )
    return cache


def build_app(config: dict[str, Any]) -> Any:
    db_url = config["db_url"]
    engine = create_engine(db_url)
    session_factory = create_session_factory(engine)
    cache = _seed_metagraph(config)

    prism = config["prism"]
    registry = _MockRegistry(
        slug=prism["slug"],
        internal_base_url=prism["internal_base_url"],
        token=prism["token"],
    )

    worker_ttl = int(config.get("worker_heartbeat_ttl_seconds", 12))
    nonce_ttl = int(config.get("nonce_ttl_seconds", 300))
    sig_ttl = int(config.get("signature_ttl_seconds", 300))
    lease_seconds = int(config.get("assignment_lease_seconds", 900))

    worker_service = WorkerCoordinationService(
        session_factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(
            session_factory, ttl_seconds=nonce_ttl
        ),
        heartbeat_ttl_seconds=worker_ttl,
    )
    worker_verifier = WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(session_factory, ttl_seconds=nonce_ttl),
        eligibility=CoordinationReadEligibility(session_factory, cache),
        ttl_seconds=sig_ttl,
    )
    worker_assignment_service = WorkerAssignmentService(
        session_factory, worker_service=worker_service, lease_seconds=lease_seconds
    )
    worker_assignment_verifier = WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(session_factory, ttl_seconds=nonce_ttl),
        eligibility=RegisteredWorkerEligibility(session_factory),
        ttl_seconds=sig_ttl,
    )
    worker_unit_status_service = WorkerUnitStatusService(session_factory)

    validator_service = ValidatorCoordinationService(
        session_factory,
        heartbeat_interval_seconds=int(
            config.get("validator_heartbeat_interval_seconds", 10)
        ),
        heartbeat_timeout_seconds=int(
            config.get("validator_heartbeat_timeout_seconds", 30)
        ),
    )
    validator_verifier = ValidatorSignedRequestVerifier(
        nonce_store=SqlAlchemyValidatorNonceStore(
            session_factory, ttl_seconds=nonce_ttl
        ),
        eligibility=MetagraphValidatorEligibility(cache),
        ttl_seconds=sig_ttl,
    )
    assignment_service = AssignmentCoordinationService(
        session_factory, lease_seconds=lease_seconds
    )

    driver = _build_orchestration_driver(
        session_factory,
        registry,
        validator_service=validator_service,
        worker_service=worker_service,
        worker_assignment_service=worker_assignment_service,
        replication_factor=int(config.get("replication_factor", 2)),
        seed=config.get("orchestration_seed"),
        worker_plane_enabled=bool(config.get("worker_plane_enabled", True)),
    )

    return create_proxy_app(
        registry=registry,
        metagraph_cache=cache,
        nonce_store=SqlAlchemyMinerNonceStore(session_factory, ttl_seconds=nonce_ttl),
        netuid=int(config.get("netuid", 100)),
        validator_service=validator_service,
        validator_verifier=validator_verifier,
        validator_health_interval_seconds=float(
            config.get("health_interval_seconds", 2.0)
        ),
        worker_service=worker_service,
        worker_verifier=worker_verifier,
        worker_health_interval_seconds=float(
            config.get("health_interval_seconds", 2.0)
        ),
        worker_assignment_service=worker_assignment_service,
        worker_assignment_verifier=worker_assignment_verifier,
        worker_unit_status_service=worker_unit_status_service,
        assignment_coordination_service=assignment_service,
        orchestration_driver=driver,
        orchestration_interval_seconds=float(
            config.get("orchestration_interval_seconds", 1.0)
        ),
    )


def _build_orchestration_driver(
    session_factory: Any,
    registry: Any,
    *,
    validator_service: ValidatorCoordinationService,
    worker_service: WorkerCoordinationService,
    worker_assignment_service: WorkerAssignmentService,
    replication_factor: int,
    seed: Any,
    worker_plane_enabled: bool = True,
) -> MasterOrchestrationDriver:
    # With the worker plane OFF (legacy) no capability is owned by the worker plane, so
    # gpu units route to online gpu validators byte-for-byte as pre-mission
    # (VAL-MASTER-013 / VAL-CROSS-006).
    worker_plane_capabilities = (
        frozenset({CAPABILITY_GPU}) if worker_plane_enabled else frozenset()
    )
    assignment_service = AssignmentService(
        session_factory, worker_plane_capabilities=worker_plane_capabilities
    )
    worker_engine = WorkerAssignmentEngine(
        session_factory,
        assignment_service=worker_assignment_service,
        worker_service=worker_service,
        replication_factor=replication_factor,
    )
    worker_reconciler = WorkerReconciliationService(
        session_factory,
        result_forwarder=HttpChallengeResultForwarder(registry),
    )
    return MasterOrchestrationDriver(
        assignment_service=assignment_service,
        validator_service=validator_service,
        work_source=HttpChallengeWorkSource(registry),
        fold_trigger=HttpChallengeFoldTrigger(registry),
        worker_assignment_engine=worker_engine,
        worker_reconciler=worker_reconciler,
        seed=seed,
    )


def main() -> None:
    config = _load_config()
    asyncio.run(_init_db(config["db_url"]))
    app = build_app(config)
    uvicorn.run(
        app,
        host=str(config.get("host", "127.0.0.1")),
        port=int(config["port"]),
        log_level=str(config.get("log_level", "warning")),
    )


if __name__ == "__main__":
    main()
