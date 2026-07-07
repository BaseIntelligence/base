"""Per-service update locks shared across supervisor rollout loops.

The ``config-sync`` and ``image-updater`` scheduled tasks run on their OWN
daemon threads and both force-roll the SAME first-party Swarm services
(``base-master-proxy`` / ``base-docker-broker``) — one on a config-digest
change, the other on an image-digest change. Without coordination the two loops
can issue overlapping ``docker service update`` calls against the same service
(rollout churn / "update out of sequence").

:class:`ServiceUpdateLocks` is a tiny registry of one :class:`threading.Lock`
per service name. Both loops take the SAME registry (created once in
``build_scheduled_tasks`` and threaded into both builders) and acquire a
service's lock around its ``docker service update``. Each loop only ever holds
ONE service lock at a time (they process services sequentially), so the scheme
is deadlock-free.
"""

from __future__ import annotations

import threading

__all__ = ["ServiceUpdateLocks"]


class ServiceUpdateLocks:
    """Lazily-created, thread-safe registry of per-service update locks."""

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def get(self, service: str) -> threading.Lock:
        """Return the lock for ``service``, creating it once on first request."""
        with self._registry_lock:
            lock = self._locks.get(service)
            if lock is None:
                lock = threading.Lock()
                self._locks[service] = lock
            return lock
