"""Best-effort real-time log streaming from an own_runner job to the validator.

In broker (Swarm) mode the own_runner job runs on a worker, so its on-disk
per-trial logs never reach the validator's ``job_dir``. This streamer POSTs each
finished trial's log channels to the challenge internal ingest route so the live
SSE feed surfaces them in real time.

Design invariants:

* BEST-EFFORT: every transport failure is swallowed. The authoritative job
  result is always the ``BASE_BENCHMARK_RESULT=`` stdout line parsed by the
  host; the stream only ever appends observability log lines and can never
  change a score.
* NO-OP unless the dispatcher injected the ``BASE_LOG_STREAM_*`` env (so
  CLI / local runs and the whole test suite stream nothing by default).
* stdlib only (``urllib``) -- the job container must not need extra deps.
* The bearer token is a per-attempt SCOPED HMAC (see
  :func:`agent_challenge.sdk.auth.mint_attempt_stream_token`), never the raw
  internal shared token, because the miner agent shares this process and env.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Job-env var names injected by the dispatcher (runner._terminal_bench_stream_env).
STREAM_URL_ENV = "BASE_LOG_STREAM_URL"
STREAM_ATTEMPT_ID_ENV = "BASE_LOG_STREAM_ATTEMPT_ID"
STREAM_TOKEN_ENV = "BASE_LOG_STREAM_TOKEN"
STREAM_SLUG_ENV = "BASE_LOG_STREAM_SLUG"
STREAM_TIMEOUT_ENV = "BASE_LOG_STREAM_TIMEOUT_SECONDS"

DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class LogStreamer:
    """Posts NDJSON log events for one evaluation attempt (best-effort)."""

    base_url: str
    attempt_id: int
    token: str
    slug: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LogStreamer | None:
        """Build a streamer from injected env, or ``None`` if not configured."""

        source = os.environ if env is None else env
        base_url = (source.get(STREAM_URL_ENV) or "").strip()
        attempt_raw = (source.get(STREAM_ATTEMPT_ID_ENV) or "").strip()
        token = (source.get(STREAM_TOKEN_ENV) or "").strip()
        slug = (source.get(STREAM_SLUG_ENV) or "").strip()
        if not (base_url and attempt_raw and token and slug):
            return None
        try:
            attempt_id = int(attempt_raw)
        except ValueError:
            return None
        return cls(
            base_url=base_url.rstrip("/"),
            attempt_id=attempt_id,
            token=token,
            slug=slug,
            timeout_seconds=_parse_timeout(source.get(STREAM_TIMEOUT_ENV)),
        )

    @property
    def url(self) -> str:
        return f"{self.base_url}/internal/v1/evaluations/{self.attempt_id}/events"

    def emit(self, events: list[dict[str, object]]) -> None:
        """POST a batch of events as NDJSON; swallow any transport error."""

        payload = "\n".join(json.dumps(event, separators=(",", ":")) for event in events)
        if not payload.strip():
            return
        request = urllib.request.Request(
            self.url,
            data=payload.encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/x-ndjson",
                "Authorization": f"Bearer {self.token}",
                "X-Base-Challenge-Slug": self.slug,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds):
                return
        except (urllib.error.URLError, OSError, ValueError):
            logger.warning("log-stream POST to %s failed", self.url, exc_info=True)


def build_log_events(
    *,
    trial_name: str,
    task_id: str,
    status: str,
    channels: Mapping[str, str],
) -> list[dict[str, object]]:
    """Turn a finished trial's channel map into one ``log`` event per channel."""

    events: list[dict[str, object]] = []
    for stream, content in channels.items():
        if content and content.strip():
            events.append(
                {
                    "kind": "log",
                    "trial_name": trial_name,
                    "task_id": task_id,
                    "status": status,
                    "stream": stream,
                    "message": content,
                }
            )
    return events


def build_incremental_log_event(
    *,
    trial_name: str,
    task_id: str,
    stream: str,
    message: str,
    status: str | None = None,
) -> dict[str, object]:
    """Build one incremental ``log`` event for a still-running trial.

    Mirrors :func:`build_log_events`' per-channel event shape but carries a single
    mid-trial delta (e.g. a fresh chunk of the agent's tmux pane) rather than a
    finished trial's full channel. ``status`` is omitted when ``None`` because a
    live delta has no terminal status; the ingest route treats it as optional.
    """

    event: dict[str, object] = {
        "kind": "log",
        "trial_name": trial_name,
        "task_id": task_id,
        "stream": stream,
        "message": message,
    }
    if status is not None:
        event["status"] = status
    return event


def _parse_timeout(raw: str | None) -> float:
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS
