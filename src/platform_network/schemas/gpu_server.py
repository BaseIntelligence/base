from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class GpuServerCreate(BaseModel):
    id: str = Field(..., min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$")
    base_url: str = Field(..., min_length=1)
    token: str | None = None
    token_file: str | None = None
    enabled: bool = True
    verify_tls: bool = True
    timeout_seconds: float = 30.0
    description: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    min_gpu_count: int = Field(default=1, ge=0)


class GpuServerUpdate(BaseModel):
    base_url: str | None = Field(default=None, min_length=1)
    token: str | None = None
    token_file: str | None = None
    enabled: bool | None = None
    verify_tls: bool | None = None
    timeout_seconds: float | None = None
    description: str | None = None
    labels: dict[str, str] | None = None
    min_gpu_count: int | None = Field(default=None, ge=0)


class GpuServerRecord(BaseModel):
    id: str
    base_url: str
    enabled: bool = True
    verify_tls: bool = True
    timeout_seconds: float = 30.0
    description: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    min_gpu_count: int = 1
    token_hint: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GpuServerView(BaseModel):
    id: str
    base_url: str
    enabled: bool
    verify_tls: bool
    timeout_seconds: float
    description: str | None = None
    labels: dict[str, str]
    min_gpu_count: int
    token_hint: str | None = None
    created_at: datetime
    updated_at: datetime


class GpuServerHealth(BaseModel):
    id: str
    status: str
    detail: str | None = None
