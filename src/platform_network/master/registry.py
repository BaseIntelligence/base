"""Challenge registry storage and serialization helpers."""

from __future__ import annotations

import hashlib
import json
import secrets
import stat
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Any

from platform_network.schemas.challenge import (
    ChallengeAdminView,
    ChallengeCreate,
    ChallengeRecord,
    ChallengeStatus,
    ChallengeUpdate,
    RegistryChallenge,
    RegistryResponse,
)


class ChallengeAlreadyExistsError(ValueError):
    """Raised when a challenge slug already exists."""


class ChallengeNotFoundError(KeyError):
    """Raised when a challenge slug is unknown."""


def _hash_token(token: str) -> str:
    """Return a deterministic non-reversible hash for a challenge token."""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_hint(token: str) -> str:
    """Return a non-secret token hint suitable for admin display."""

    return f"{token[:4]}…{token[-4:]}"


def default_internal_base_url(slug: str) -> str:
    """Build the Docker-network URL for a challenge container."""

    return f"http://challenge-{slug}:8000"


def default_public_proxy_base_path(slug: str) -> str:
    """Build the public proxy base path for a challenge."""

    return f"/challenges/{slug}"


def default_sqlite_volume_name(slug: str) -> str:
    """Build the default Docker volume name for challenge SQLite storage."""

    return f"platform_{slug.replace('-', '_')}_sqlite"


class ChallengeRegistry:
    """Thread-safe in-memory challenge registry.

    The class is deliberately small and storage-agnostic so it can be replaced by
    database-backed repositories without changing the FastAPI layers.
    """

    def __init__(
        self,
        *,
        network: str = "platform",
        api_version: str = "1.0",
        master_uid: int = 0,
    ) -> None:
        self.network = network
        self.api_version = api_version
        self.master_uid = master_uid
        self._records: dict[str, ChallengeRecord] = {}
        self._lock = RLock()

    def create(self, payload: ChallengeCreate) -> tuple[ChallengeRecord, str]:
        """Create a challenge record and return it with the one-time clear token."""

        with self._lock:
            if payload.slug in self._records:
                raise ChallengeAlreadyExistsError(payload.slug)

            token = secrets.token_urlsafe(32)
            volumes = dict(payload.volumes)
            volumes.setdefault("sqlite", default_sqlite_volume_name(payload.slug))

            now = datetime.now(UTC)
            record = ChallengeRecord(
                slug=payload.slug,
                name=payload.name,
                image=payload.image,
                version=payload.version,
                emission_percent=payload.emission_percent,
                status=payload.status,
                token_hash=_hash_token(token),
                token_hint=_token_hint(token),
                description=payload.description,
                api_version=payload.api_version,
                internal_base_url=payload.internal_base_url
                or default_internal_base_url(payload.slug),
                public_proxy_base_path=default_public_proxy_base_path(payload.slug),
                required_capabilities=list(payload.required_capabilities),
                resources=dict(payload.resources),
                volumes=volumes,
                env=dict(payload.env),
                secrets=list(payload.secrets),
                metadata=dict(payload.metadata),
                created_at=now,
                updated_at=now,
            )
            self._records[payload.slug] = record
            return record, token

    def update(self, slug: str, payload: ChallengeUpdate) -> ChallengeRecord:
        """Patch mutable metadata for an existing challenge."""

        with self._lock:
            record = self._get_locked(slug)
            updates = payload.model_dump(exclude_unset=True)
            if not updates:
                return record

            data = record.model_dump()
            data.update(updates)
            data["updated_at"] = datetime.now(UTC)
            updated = ChallengeRecord(**data)
            self._records[slug] = updated
            return updated

    def set_status(self, slug: str, status: ChallengeStatus) -> ChallengeRecord:
        """Set the lifecycle status for a challenge."""

        return self.update(slug, ChallengeUpdate(status=status))

    def get(self, slug: str) -> ChallengeRecord:
        """Return a challenge by slug."""

        with self._lock:
            return self._get_locked(slug)

    def list(self, *, active_only: bool = False) -> list[ChallengeRecord]:
        """List challenges, optionally filtering to active records only."""

        with self._lock:
            records = list(self._records.values())
        if active_only:
            return [
                record for record in records if record.status == ChallengeStatus.ACTIVE
            ]
        return records

    def registry_response(self) -> RegistryResponse:
        """Serialize active challenges for normal validators."""

        return RegistryResponse(
            network=self.network,
            api_version=self.api_version,
            master_uid=self.master_uid,
            challenges=[
                record_to_registry_view(record)
                for record in self.list(active_only=True)
            ],
        )

    def _get_locked(self, slug: str) -> ChallengeRecord:
        record = self._records.get(slug)
        if record is None:
            raise ChallengeNotFoundError(slug)
        return record


class FileChallengeRegistry(ChallengeRegistry):
    """Small persistent registry shared by admin/proxy processes.

    PostgreSQL remains the production source of truth for the master, but this
    file-backed adapter gives local compose and split admin/proxy apps a shared
    registry without exposing challenge tokens or requiring both processes to
    share memory.
    """

    def __init__(
        self,
        state_file: str | Path,
        secret_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.state_file = Path(state_file)
        self.secret_dir = Path(secret_dir) if secret_dir else self.state_file.parent
        super().__init__(**kwargs)
        self._load()

    def create(self, payload: ChallengeCreate) -> tuple[ChallengeRecord, str]:
        record, token = super().create(payload)
        self._write_token(record.slug, token)
        self._save()
        return record, token

    def update(self, slug: str, payload: ChallengeUpdate) -> ChallengeRecord:
        record = super().update(slug, payload)
        self._save()
        return record

    def set_status(self, slug: str, status: ChallengeStatus) -> ChallengeRecord:
        record = super().set_status(slug, status)
        self._save()
        return record

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        data = json.loads(self.state_file.read_text(encoding="utf-8"))
        records = data.get("records", {})
        if not isinstance(records, dict):
            return
        with self._lock:
            self._records = {
                slug: ChallengeRecord.model_validate(record)
                for slug, record in records.items()
            }

    def get(self, slug: str) -> ChallengeRecord:
        self._load()
        return super().get(slug)

    def list(self, *, active_only: bool = False) -> list[ChallengeRecord]:
        self._load()
        return super().list(active_only=active_only)

    def get_token(self, slug: str) -> str:
        path = self._token_path(slug)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _token_path(self, slug: str) -> Path:
        return self.secret_dir / f"{slug}_challenge_token"

    def _write_token(self, slug: str, token: str) -> None:
        self.secret_dir.mkdir(parents=True, exist_ok=True)
        path = self._token_path(slug)
        path.write_text(token, encoding="utf-8")
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": {
                slug: record.model_dump(mode="json")
                for slug, record in self._records.items()
            }
        }
        self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_to_admin_view(record: ChallengeRecord) -> ChallengeAdminView:
    """Convert internal metadata to an admin-safe response model."""

    data = record.model_dump(exclude={"token_hash"})
    return ChallengeAdminView(**data)


def record_to_registry_view(record: ChallengeRecord) -> RegistryChallenge:
    """Convert internal metadata to the validator-facing registry model."""

    return RegistryChallenge(
        slug=record.slug,
        name=record.name,
        image=record.image,
        version=record.version,
        emission_percent=Decimal(record.emission_percent),
        status=record.status,
        internal_base_url=record.internal_base_url,
        public_proxy_base_path=record.public_proxy_base_path,
        required_capabilities=list(record.required_capabilities),
        resources=dict(record.resources),
        volumes=dict(record.volumes),
        env=dict(record.env),
        secrets=list(record.secrets),
    )
