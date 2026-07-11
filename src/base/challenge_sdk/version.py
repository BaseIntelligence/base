"""Immutable release identity for the canonical Base challenge SDK."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import metadata, resources
from types import MappingProxyType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class ReleaseManifest(BaseModel):
    """Identity and public imports embedded in every Base distribution artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution_name: str
    artifact_version: str
    sdk_contract_version: str
    api_version: str
    release_id: str
    public_imports: Mapping[
        str,
        Annotated[tuple[str, ...], Field(min_length=1)],
    ]

    @field_validator("public_imports")
    @classmethod
    def _freeze_symbol_lists(
        cls,
        value: dict[str, tuple[str, ...]],
    ) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {module: tuple(symbols) for module, symbols in value.items()}
        )

    @field_serializer("public_imports")
    def _serialize_public_imports(
        self,
        value: Mapping[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        return dict(value)


def _load_release_manifest() -> ReleaseManifest:
    manifest_file = resources.files(__package__).joinpath("release_manifest.json")
    return ReleaseManifest.model_validate(
        json.loads(manifest_file.read_text(encoding="utf-8"))
    )


RELEASE_MANIFEST = _load_release_manifest()
DISTRIBUTION_NAME = RELEASE_MANIFEST.distribution_name
ARTIFACT_VERSION = RELEASE_MANIFEST.artifact_version
SDK_CONTRACT_VERSION = RELEASE_MANIFEST.sdk_contract_version
API_VERSION = RELEASE_MANIFEST.api_version
RELEASE_ID = RELEASE_MANIFEST.release_id
PUBLIC_IMPORTS = RELEASE_MANIFEST.public_imports


def validate_installed_distribution() -> None:
    """Reject an artifact whose wheel metadata contradicts its SDK manifest."""

    installed_version = metadata.version(DISTRIBUTION_NAME)
    if installed_version != ARTIFACT_VERSION:
        raise RuntimeError(
            "Base release identity mismatch: "
            f"manifest={ARTIFACT_VERSION!r}, distribution={installed_version!r}"
        )


__all__ = [
    "API_VERSION",
    "ARTIFACT_VERSION",
    "DISTRIBUTION_NAME",
    "PUBLIC_IMPORTS",
    "RELEASE_ID",
    "RELEASE_MANIFEST",
    "ReleaseManifest",
    "SDK_CONTRACT_VERSION",
    "validate_installed_distribution",
]
