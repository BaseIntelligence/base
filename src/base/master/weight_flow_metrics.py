"""In-process weight-flow metrics and correlation (VAL-WEIGHT-068).

Counters/histograms stay process-local so they never require Prometheus or
third-party exporters. Labels are limited to low-cardinality fields: outcome,
challenge_slug, epoch, revision, and digests — never secrets/tokens/headers.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WeightFlowMetrics:
    """Thread-safe counters for raw-weight, aggregation, and submit lifecycle."""

    pushes: Counter[str] = field(default_factory=Counter)
    aggregation_outcomes: Counter[str] = field(default_factory=Counter)
    submit_outcomes: Counter[str] = field(default_factory=Counter)
    fetch_failures: int = 0
    retry_exhaustion: int = 0
    aggregation_duration_ms: list[float] = field(default_factory=list)
    vector_age_seconds: list[float] = field(default_factory=list)
    correlations: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_push(
        self,
        *,
        outcome: str,
        challenge_slug: str | None = None,
        epoch: int | None = None,
        revision: int | None = None,
        snapshot_id: str | None = None,
        payload_digest: str | None = None,
        vector_id: str | None = None,
    ) -> None:
        with self._lock:
            self.pushes[outcome] += 1
            if challenge_slug is not None:
                self.correlations.append(
                    {
                        "challenge_slug": challenge_slug,
                        "epoch": epoch,
                        "revision": revision,
                        "snapshot_id": snapshot_id,
                        "payload_digest": payload_digest,
                        "vector_id": vector_id,
                        "outcome": outcome,
                    }
                )
                # Bound memory for long-lived masters; keep recent window only.
                if len(self.correlations) > 512:
                    del self.correlations[:-256]

    def record_aggregation(
        self,
        *,
        outcome: str,
        duration_ms: float,
        vector_id: str | None = None,
        vector_age: float | None = None,
        challenge_slug: str | None = None,
        epoch: int | None = None,
        snapshot_digest: str | None = None,
    ) -> None:
        with self._lock:
            self.aggregation_outcomes[outcome] += 1
            self.aggregation_duration_ms.append(float(duration_ms))
            if len(self.aggregation_duration_ms) > 256:
                del self.aggregation_duration_ms[:-128]
            if vector_age is not None:
                self.vector_age_seconds.append(float(vector_age))
                if len(self.vector_age_seconds) > 256:
                    del self.vector_age_seconds[:-128]
            if challenge_slug is not None or vector_id is not None:
                self.correlations.append(
                    {
                        "challenge_slug": challenge_slug,
                        "epoch": epoch,
                        "payload_digest": snapshot_digest,
                        "vector_id": vector_id,
                        "outcome": f"aggregation:{outcome}",
                    }
                )
                if len(self.correlations) > 512:
                    del self.correlations[:-256]

    def record_submit(self, *, outcome: str, attempt: int = 1) -> None:
        with self._lock:
            self.submit_outcomes[outcome] += 1
            if outcome in {"retry_exhausted", "exhausted"}:
                self.retry_exhaustion += 1
            _ = attempt

    def record_fetch_failure(self) -> None:
        with self._lock:
            self.fetch_failures += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pushes": dict(self.pushes),
                "aggregation_outcomes": dict(self.aggregation_outcomes),
                "submit_outcomes": dict(self.submit_outcomes),
                "fetch_failures": self.fetch_failures,
                "retry_exhaustion": self.retry_exhaustion,
                "aggregation_duration_ms": list(self.aggregation_duration_ms),
                "vector_age_seconds": list(self.vector_age_seconds),
                "correlations": list(self.correlations),
            }

    def reset(self) -> None:
        with self._lock:
            self.pushes.clear()
            self.aggregation_outcomes.clear()
            self.submit_outcomes.clear()
            self.fetch_failures = 0
            self.retry_exhaustion = 0
            self.aggregation_duration_ms.clear()
            self.vector_age_seconds.clear()
            self.correlations.clear()


_GLOBAL = WeightFlowMetrics()


def get_weight_flow_metrics() -> WeightFlowMetrics:
    return _GLOBAL


def timed_ms(start: float | None = None) -> float:
    """Return elapsed milliseconds from ``start`` (or 0 if start is None)."""

    if start is None:
        return 0.0
    return (time.perf_counter() - start) * 1000.0


def prometheus_text(metrics: WeightFlowMetrics | None = None) -> str:
    """Render a small text exposition without high-cardinality secrets."""

    data = (metrics or _GLOBAL).snapshot()
    lines: list[str] = [
        "# HELP base_raw_weight_pushes_total Raw weight push outcomes",
        "# TYPE base_raw_weight_pushes_total counter",
    ]
    for outcome, count in sorted(data["pushes"].items()):
        safe = str(outcome).replace('"', "")
        lines.append(f'base_raw_weight_pushes_total{{outcome="{safe}"}} {int(count)}')
    lines.extend(
        [
            "# HELP base_aggregation_outcomes_total Aggregation seal outcomes",
            "# TYPE base_aggregation_outcomes_total counter",
        ]
    )
    for outcome, count in sorted(data["aggregation_outcomes"].items()):
        safe = str(outcome).replace('"', "")
        lines.append(
            f'base_aggregation_outcomes_total{{outcome="{safe}"}} {int(count)}'
        )
    lines.extend(
        [
            "# HELP base_weight_submit_outcomes_total Validator submit outcomes",
            "# TYPE base_weight_submit_outcomes_total counter",
        ]
    )
    for outcome, count in sorted(data["submit_outcomes"].items()):
        safe = str(outcome).replace('"', "")
        lines.append(
            f'base_weight_submit_outcomes_total{{outcome="{safe}"}} {int(count)}'
        )
    lines.append("# HELP base_weight_fetch_failures_total Vector fetch failures")
    lines.append("# TYPE base_weight_fetch_failures_total counter")
    lines.append(f"base_weight_fetch_failures_total {int(data['fetch_failures'])}")
    lines.append("# HELP base_weight_retry_exhaustion_total Retry exhaustion events")
    lines.append("# TYPE base_weight_retry_exhaustion_total counter")
    lines.append(f"base_weight_retry_exhaustion_total {int(data['retry_exhaustion'])}")
    durations = data["aggregation_duration_ms"]
    if durations:
        lines.append("# HELP base_aggregation_duration_ms_sum Aggregation durations")
        lines.append("# TYPE base_aggregation_duration_ms_sum counter")
        lines.append(
            f"base_aggregation_duration_ms_sum {sum(float(v) for v in durations):.3f}"
        )
        lines.append(f"base_aggregation_duration_ms_count {len(durations)}")
    ages = data["vector_age_seconds"]
    if ages:
        lines.append("# HELP base_vector_age_seconds_sum Served vector ages")
        lines.append("# TYPE base_vector_age_seconds_sum counter")
        lines.append(f"base_vector_age_seconds_sum {sum(float(v) for v in ages):.3f}")
        lines.append(f"base_vector_age_seconds_count {len(ages)}")
    lines.append("")
    return "\n".join(lines)


# Silence optional unused import if Histogram-like dict form is preferred later.
def low_cardinality_labels(**labels: Any) -> dict[str, str]:
    allowed = {
        "outcome",
        "challenge_slug",
        "epoch",
        "revision",
        "snapshot_id",
        "payload_digest",
        "vector_id",
    }
    return {
        key: str(value)
        for key, value in labels.items()
        if key in allowed and value is not None
    }


# Keep a module-level factory used by tests.
def _fresh_metrics() -> WeightFlowMetrics:
    return WeightFlowMetrics()


__all__ = [
    "WeightFlowMetrics",
    "get_weight_flow_metrics",
    "low_cardinality_labels",
    "prometheus_text",
    "timed_ms",
]
