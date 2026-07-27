"""Best-effort mid-run progress posts from the Eval CVM to the master.

Posts closed progress bodies to
``POST /evaluation/v1/runs/{eval_run_id}/progress`` with Bearer
``EVAL_RUN_TOKEN``. Failures are swallowed so observability can never change a
score or abort a trial.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PROGRESS_BASE_URL_ENV = "EVAL_PROGRESS_BASE_URL"
EVAL_RUN_ID_ENV = "EVAL_RUN_ID"
EVAL_SUBMISSION_ID_ENV = "EVAL_SUBMISSION_ID"
EVAL_RUN_TOKEN_ENV = "EVAL_RUN_TOKEN"
PROGRESS_TIMEOUT_ENV = "EVAL_PROGRESS_TIMEOUT_SECONDS"

DEFAULT_TIMEOUT_SECONDS = 5.0
SAFE_PHASES = frozenset({"assigned", "starting", "waiting", "running", "completed", "failed"})


@dataclass
class ProgressReporter:
    """Posts per-task phase transitions (best-effort, score-free)."""

    base_url: str
    eval_run_id: str
    submission_id: str
    token: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    _sequence: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ProgressReporter | None:
        """Build a reporter from injected env, or ``None`` if not configured."""

        source = os.environ if env is None else env
        base_url = (source.get(PROGRESS_BASE_URL_ENV) or "").strip()
        eval_run_id = (source.get(EVAL_RUN_ID_ENV) or "").strip()
        submission_id = (source.get(EVAL_SUBMISSION_ID_ENV) or "").strip()
        token = (source.get(EVAL_RUN_TOKEN_ENV) or "").strip()
        if not (base_url and eval_run_id and submission_id and token):
            return None
        return cls(
            base_url=base_url.rstrip("/"),
            eval_run_id=eval_run_id,
            submission_id=submission_id,
            token=token,
            timeout_seconds=_parse_timeout(source.get(PROGRESS_TIMEOUT_ENV)),
        )

    @property
    def url(self) -> str:
        return f"{self.base_url}/evaluation/v1/runs/{self.eval_run_id}/progress"

    def emit(
        self,
        *,
        task_id: str,
        status: str,
        progress: float | None = None,
        message: str | None = None,
        event_type: str = "task.status",
    ) -> None:
        """POST one progress event; swallow any transport error."""

        if status not in SAFE_PHASES:
            logger.warning("progress reporter refused unknown phase %s", status)
            return
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        body: dict[str, object] = {
            "schema_version": 1,
            "eval_run_id": self.eval_run_id,
            "submission_id": self.submission_id,
            "task_id": task_id,
            "sequence": sequence,
            "status": status,
            "event_type": event_type,
            "progress": progress,
            "message": message,
        }
        payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds):
                return
        except (urllib.error.URLError, OSError, ValueError):
            logger.warning("progress POST to %s failed", self.url, exc_info=True)


def _parse_timeout(raw: str | None) -> float:
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "EVAL_RUN_ID_ENV",
    "EVAL_RUN_TOKEN_ENV",
    "EVAL_SUBMISSION_ID_ENV",
    "PROGRESS_BASE_URL_ENV",
    "PROGRESS_TIMEOUT_ENV",
    "ProgressReporter",
    "SAFE_PHASES",
]
