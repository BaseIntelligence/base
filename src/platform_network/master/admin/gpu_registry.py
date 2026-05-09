from __future__ import annotations

from typing import Protocol

from platform_network.gpu.registry import (
    GpuServerAlreadyExistsError,
    GpuServerNotFoundError,
)
from platform_network.schemas.gpu_server import (
    GpuServerCreate,
    GpuServerRecord,
    GpuServerUpdate,
)


class GpuServerRegistry(Protocol):
    def list(self) -> list[GpuServerRecord]: ...
    def get(self, server_id: str) -> GpuServerRecord: ...
    def create(self, payload: GpuServerCreate) -> GpuServerRecord: ...
    def update(self, server_id: str, payload: GpuServerUpdate) -> GpuServerRecord: ...
    def delete(self, server_id: str) -> None: ...
    def set_enabled(self, server_id: str, enabled: bool) -> GpuServerRecord: ...
    def get_token(self, server_id: str) -> str: ...


class InMemoryGpuServerRegistry:
    def __init__(self) -> None:
        self.records: dict[str, GpuServerRecord] = {}
        self.tokens: dict[str, str] = {}

    def list(self) -> list[GpuServerRecord]:
        return list(self.records.values())

    def get(self, server_id: str) -> GpuServerRecord:
        record = self.records.get(server_id)
        if record is None:
            raise GpuServerNotFoundError(server_id)
        return record

    def create(self, payload: GpuServerCreate) -> GpuServerRecord:
        if payload.id in self.records:
            raise GpuServerAlreadyExistsError(payload.id)
        record = GpuServerRecord(
            id=payload.id,
            base_url=payload.base_url,
            enabled=payload.enabled,
            verify_tls=payload.verify_tls,
            timeout_seconds=payload.timeout_seconds,
            description=payload.description,
            labels=payload.labels,
            min_gpu_count=payload.min_gpu_count,
            token_hint="****" if payload.token or payload.token_file else None,
        )
        self.records[payload.id] = record
        if payload.token:
            self.tokens[payload.id] = payload.token
        return record

    def update(self, server_id: str, payload: GpuServerUpdate) -> GpuServerRecord:
        record = self.get(server_id)
        data = record.model_dump()
        updates = payload.model_dump(exclude_unset=True)
        token = updates.pop("token", None)
        updates.pop("token_file", None)
        data.update(updates)
        if token:
            self.tokens[server_id] = token
            data["token_hint"] = "****"
        updated = GpuServerRecord(**data)
        self.records[server_id] = updated
        return updated

    def delete(self, server_id: str) -> None:
        self.get(server_id)
        self.records.pop(server_id, None)
        self.tokens.pop(server_id, None)

    def set_enabled(self, server_id: str, enabled: bool) -> GpuServerRecord:
        return self.update(server_id, GpuServerUpdate(enabled=enabled))

    def get_token(self, server_id: str) -> str:
        return self.tokens.get(server_id, "")
