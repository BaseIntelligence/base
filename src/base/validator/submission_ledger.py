"""Durable per-validator on-chain submission ledger.

Persists vector-keyed attempt/outcome/reconciliation state on the validator's
own volume so restarts, ambiguous chain outcomes, and new-vector supersession
remain safe. Keys include validator hotkey, vector_id, digests, netuid, and
chain identity. Never stores wallet secrets or private keys.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

DEFAULT_LEDGER_FILENAME = "submission_ledger.json"


class SubmissionStatus(StrEnum):
    """Durable outcome of a vector-keyed submission attempt."""

    PENDING = "pending"
    PRE_SEND_FAILED = "pre_send_failed"
    SUBMITTING = "submitting"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    RETRY_EXHAUSTED = "retry_exhausted"
    SUPERSEDED = "superseded"


@dataclass
class SubmissionRecord:
    """One durable ledger row for a (hotkey, vector) attempt trail."""

    validator_hotkey: str
    vector_id: str
    vector_digest: str
    netuid: int
    chain_endpoint: str
    status: str = SubmissionStatus.PENDING.value
    attempt_count: int = 0
    last_error: str | None = None
    last_attempt_at: str | None = None
    accepted_at: str | None = None
    reconciled_at: str | None = None
    uids: list[int] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    superseded_by: str | None = None
    observed_to_master: bool = False

    @property
    def key(self) -> str:
        return submission_key(
            validator_hotkey=self.validator_hotkey,
            vector_id=self.vector_id,
            vector_digest=self.vector_digest,
            netuid=self.netuid,
            chain_endpoint=self.chain_endpoint,
        )


def submission_key(
    *,
    validator_hotkey: str,
    vector_id: str,
    vector_digest: str,
    netuid: int,
    chain_endpoint: str,
) -> str:
    endpoint = (chain_endpoint or "").strip()
    return f"{validator_hotkey}|{vector_id}|{vector_digest}|{int(netuid)}|{endpoint}"


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist JSON with temp file + fsync + os.replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class ValidatorSubmissionLedger:
    """Filesystem-backed ledger of this validator's chain submissions."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._records: dict[str, SubmissionRecord] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            self._records = {}
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._records = {}
            return
        items = raw.get("records", {}) if isinstance(raw, dict) else {}
        loaded: dict[str, SubmissionRecord] = {}
        if isinstance(items, dict):
            for key, value in items.items():
                if not isinstance(value, dict):
                    continue
                try:
                    record = SubmissionRecord(
                        validator_hotkey=str(value["validator_hotkey"]),
                        vector_id=str(value["vector_id"]),
                        vector_digest=str(value["vector_digest"]),
                        netuid=int(value["netuid"]),
                        chain_endpoint=str(value.get("chain_endpoint") or ""),
                        status=str(value.get("status") or SubmissionStatus.PENDING),
                        attempt_count=int(value.get("attempt_count") or 0),
                        last_error=value.get("last_error"),
                        last_attempt_at=value.get("last_attempt_at"),
                        accepted_at=value.get("accepted_at"),
                        reconciled_at=value.get("reconciled_at"),
                        uids=[int(u) for u in (value.get("uids") or [])],
                        weights=[float(w) for w in (value.get("weights") or [])],
                        superseded_by=value.get("superseded_by"),
                        observed_to_master=bool(value.get("observed_to_master")),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                loaded[str(key)] = record
        self._records = loaded

    def _persist(self) -> None:
        payload = {
            "schema_version": 1,
            "updated_at": _utcnow_iso(),
            "records": {key: asdict(record) for key, record in self._records.items()},
        }
        atomic_write_json(self._path, payload)

    def get(
        self,
        *,
        validator_hotkey: str,
        vector_id: str,
        vector_digest: str,
        netuid: int,
        chain_endpoint: str,
    ) -> SubmissionRecord | None:
        key = submission_key(
            validator_hotkey=validator_hotkey,
            vector_id=vector_id,
            vector_digest=vector_digest,
            netuid=netuid,
            chain_endpoint=chain_endpoint,
        )
        with self._lock:
            return self._records.get(key)

    def upsert(self, record: SubmissionRecord) -> SubmissionRecord:
        with self._lock:
            self._records[record.key] = record
            self._persist()
            return record

    def mark_status(
        self,
        record: SubmissionRecord,
        *,
        status: SubmissionStatus | str,
        error: str | None = None,
        increment_attempt: bool = False,
        accepted: bool = False,
        reconciled: bool = False,
    ) -> SubmissionRecord:
        with self._lock:
            current = self._records.get(record.key) or record
            if increment_attempt:
                current.attempt_count = int(current.attempt_count) + 1
                current.last_attempt_at = _utcnow_iso()
            current.status = str(status)
            if error is not None:
                current.last_error = error
            if accepted:
                current.accepted_at = _utcnow_iso()
                current.last_error = None
            if reconciled:
                current.reconciled_at = _utcnow_iso()
            self._records[current.key] = current
            self._persist()
            return current

    def supersede_active(
        self,
        *,
        validator_hotkey: str,
        keep_vector_id: str,
        superseded_by: str,
    ) -> list[SubmissionRecord]:
        """Mark non-terminal rows for other vectors as superseded."""

        terminal = {
            SubmissionStatus.ACCEPTED.value,
            SubmissionStatus.SUPERSEDED.value,
            SubmissionStatus.RETRY_EXHAUSTED.value,
        }
        changed: list[SubmissionRecord] = []
        with self._lock:
            for record in self._records.values():
                if record.validator_hotkey != validator_hotkey:
                    continue
                if record.vector_id == keep_vector_id:
                    continue
                if record.status in terminal:
                    continue
                record.status = SubmissionStatus.SUPERSEDED.value
                record.superseded_by = superseded_by
                record.last_error = (
                    f"superseded by vector {superseded_by} before"
                    " terminal outcome of previous vector"
                )
                changed.append(record)
            if changed:
                self._persist()
        return changed

    def all_records(self) -> list[SubmissionRecord]:
        with self._lock:
            return list(self._records.values())


__all__ = [
    "DEFAULT_LEDGER_FILENAME",
    "SubmissionRecord",
    "SubmissionStatus",
    "ValidatorSubmissionLedger",
    "atomic_write_json",
    "submission_key",
]
