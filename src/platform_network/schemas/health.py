from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    slug: str
    version: str


class VersionResponse(BaseModel):
    api_version: str = "1.0"
    challenge_version: str
    sdk_version: str = "1.0.0"
    capabilities: list[str] = Field(
        default_factory=lambda: ["get_weights", "proxy_routes"]
    )
