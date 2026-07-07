"""Long-running miner-funded worker agent loop (architecture.md sec 3.2).

Mirrors :class:`base.validator.agent.runtime.ValidatorAgent` (they share the pure
loop primitives in :mod:`base.coordination.agent_loop`) but enrolls as a WORKER:
it registers with the master under a miner-signed binding, heartbeats to stay
``active``, pulls gpu work units assigned to it, executes each via the
:class:`AssignmentExecutor` seam on its OWN local broker, and posts results that
always carry an ``ExecutionProof`` envelope. The agent authenticates every
pull/post as its worker keypair, never as a metagraph validator permit.

Resilience: an execution failure (e.g. an unreachable local broker) is posted as
a ``success=false`` result and never crashes the loop; the agent keeps
heartbeating and pulling. Registration is an idempotent server-side upsert keyed
on the worker pubkey, so a restart re-registers into the same fleet entry.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from base.coordination.agent_loop import (
    AgentCycleSummary,
    BackoffPolicy,
    backoff_sleep,
    is_transient_error,
    sleep_until,
)
from base.validator.agent.executor import (
    AssignmentContext,
    AssignmentExecutor,
    BrokerConfig,
    gateway_env_for_assignment,
)
from base.worker.coordination_client import WorkerCoordinationClient

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = frozenset({"assigned", "running"})
_DEFAULT_HEARTBEAT_INTERVAL = 30


@dataclass(frozen=True)
class WorkerBinding:
    """The miner-signed binding the agent presents at registration.

    The MINER signs ``worker-binding:{worker_pubkey}:{miner_hotkey}:{nonce}``
    (sr25519) out-of-band (by the deploy CLI); the agent only carries the signed
    material. A fresh ``nonce`` is required per registration (a restart supplies
    a new one), so a replayed binding is rejected while a same-owner re-enroll
    with a fresh nonce is idempotent.
    """

    miner_hotkey: str
    signature: str
    nonce: str


class WorkerAgent:
    """Coordinated GPU worker: register, heartbeat, pull, execute, post proofs."""

    def __init__(
        self,
        *,
        client: WorkerCoordinationClient,
        executor: AssignmentExecutor,
        broker: BrokerConfig,
        binding: WorkerBinding,
        provider: str,
        provider_instance_ref: str | None = None,
        capabilities: list[str] | None = None,
        gateway_url: str = "",
        heartbeat_interval_seconds: int | None = None,
        poll_interval_seconds: float = 5.0,
        last_seen_meta_factory: Callable[[], Mapping[str, Any]] | None = None,
        backoff: BackoffPolicy | None = None,
    ) -> None:
        self._client = client
        self._executor = executor
        self._broker = broker
        self._binding = binding
        self._provider = provider
        self._provider_instance_ref = provider_instance_ref
        self._capabilities = list(capabilities or ["gpu"])
        self._gateway_url = gateway_url
        self._configured_interval = heartbeat_interval_seconds
        self._poll_interval = poll_interval_seconds
        self._last_seen_meta_factory = last_seen_meta_factory
        self._backoff = backoff or BackoffPolicy()
        self._worker_id: str | None = None
        self._heartbeat_ttl_seconds: int | None = None

    @property
    def worker_pubkey(self) -> str:
        return self._client.worker_pubkey

    @property
    def worker_id(self) -> str | None:
        return self._worker_id

    @property
    def heartbeat_interval(self) -> int:
        if self._configured_interval is not None:
            return self._configured_interval
        if self._heartbeat_ttl_seconds:
            return max(1, self._heartbeat_ttl_seconds // 2)
        return _DEFAULT_HEARTBEAT_INTERVAL

    async def register(self, shutdown_event: asyncio.Event | None = None) -> str:
        """Register (idempotent upsert) and resolve the worker id + heartbeat TTL.

        Transient master failures (transport errors / ``429``/``5xx``) are retried
        with bounded exponential backoff; a permanent error (``4xx``, e.g. a
        forged binding or replayed nonce) fails fast so the caller surfaces it. A
        set ``shutdown_event`` aborts the retry loop (re-raising the last error).
        """

        failures = 0
        while True:
            try:
                response = await self._client.register(
                    miner_hotkey=self._binding.miner_hotkey,
                    binding_signature=self._binding.signature,
                    nonce=self._binding.nonce,
                    provider=self._provider,
                    provider_instance_ref=self._provider_instance_ref,
                    capabilities=self._capabilities,
                    last_seen_meta=self._meta(),
                )
            except Exception as exc:
                if not is_transient_error(exc):
                    raise
                failures += 1
                delay = self._backoff.delay(failures)
                logger.warning(
                    "worker agent register attempt %d failed (%s); retrying in %.1fs",
                    failures,
                    exc,
                    delay,
                )
                if not await backoff_sleep(shutdown_event, delay):
                    raise
                continue
            self._worker_id = response.worker.worker_id
            self._heartbeat_ttl_seconds = response.heartbeat_ttl_seconds
            return self._worker_id

    async def heartbeat_once(self) -> None:
        if self._worker_id is None:
            raise RuntimeError("worker agent must register before heartbeating")
        await self._client.heartbeat(
            worker_id=self._worker_id, last_seen_meta=self._meta()
        )

    async def process_pending_assignments(self) -> AgentCycleSummary:
        """Pull, execute, and post results for all currently-assigned units."""

        assignments = await self._client.pull()
        completed = 0
        failed = 0
        for assignment in assignments:
            if assignment.status not in _ACTIVE_STATUSES:
                continue
            if await self._execute_one(assignment):
                completed += 1
            else:
                failed += 1
        return AgentCycleSummary(
            pulled=len(assignments), completed=completed, failed=failed
        )

    async def run_heartbeat_loop(self, shutdown_event: asyncio.Event) -> None:
        failures = 0
        while not shutdown_event.is_set():
            try:
                await self.heartbeat_once()
                failures = 0
            except Exception:
                failures += 1
                logger.exception("worker agent heartbeat failed")
            delay = (
                self._backoff.delay(failures) if failures else self.heartbeat_interval
            )
            await sleep_until(shutdown_event, delay)

    async def run_assignment_loop(self, shutdown_event: asyncio.Event) -> None:
        failures = 0
        while not shutdown_event.is_set():
            try:
                await self.process_pending_assignments()
                failures = 0
            except Exception:
                failures += 1
                logger.exception("worker agent assignment pass failed")
            delay = self._backoff.delay(failures) if failures else self._poll_interval
            await sleep_until(shutdown_event, delay)

    async def run_forever(self, shutdown_event: asyncio.Event | None = None) -> None:
        shutdown_event = shutdown_event or asyncio.Event()
        await self.register(shutdown_event)
        await asyncio.gather(
            self.run_heartbeat_loop(shutdown_event),
            self.run_assignment_loop(shutdown_event),
        )

    async def _execute_one(self, assignment: Any) -> bool:
        gateway_env = gateway_env_for_assignment(
            assignment, gateway_url=self._gateway_url
        )
        context = AssignmentContext(
            assignment=assignment, gateway_env=gateway_env, broker=self._broker
        )

        try:
            result = await self._executor.execute(context, progress=_noop_progress)
        except Exception as exc:
            logger.exception("worker agent execution failed for %s", assignment.id)
            await self._client.post_result(
                assignment.id, success=False, payload={"error": str(exc)}
            )
            return False

        await self._client.post_result(
            assignment.id,
            success=result.success,
            payload=dict(result.payload),
            checkpoint_ref=result.checkpoint_ref,
        )
        return result.success

    def _meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "capabilities": list(self._capabilities),
            "broker_url": self._broker.broker_url,
        }
        if self._last_seen_meta_factory is not None:
            meta.update(dict(self._last_seen_meta_factory()))
        return meta


async def _noop_progress(
    *,
    checkpoint_ref: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Progress heartbeats are not part of the worker plane yet (no-op)."""

    return None


__all__ = [
    "AgentCycleSummary",
    "BackoffPolicy",
    "WorkerAgent",
    "WorkerBinding",
]
