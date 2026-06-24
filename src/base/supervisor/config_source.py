"""Backend-neutral config-sync core (fetch / validate / render / digest).

Pure helpers shared by the Swarm supervisor config-sync task. They fetch
plain YAML config from a Git source, reject Secret manifests, render the
runtime-config payload, and digest it. There is no orchestrator coupling.

The source is a plain runtime-config YAML (optionally wrapped in a
``ConfigMap``-kind document, whose ``master.yaml`` data key is unwrapped).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx
import yaml

CONFIG_MAP_KEY = "master.yaml"


@dataclass(frozen=True)
class ConfigSyncSource:
    repository: str
    branch: str
    paths: tuple[str, ...]
    sync_secrets: bool = False
    allowed_kinds: tuple[str, ...] = ("ConfigMap",)
    fetcher: Callable[[ConfigSyncSource], str] | None = field(
        default=None, compare=False, repr=False
    )

    @classmethod
    def default(
        cls, fetcher: Callable[[ConfigSyncSource], str] | None = None
    ) -> ConfigSyncSource:
        return cls(
            repository="BaseIntelligence/base",
            branch="main",
            paths=("deploy/swarm/master.yaml",),
            sync_secrets=False,
            allowed_kinds=("ConfigMap",),
            fetcher=fetcher,
        )


@dataclass(frozen=True)
class ConfigSyncResult:
    changed: bool
    reason: str
    current_digest: str | None = None
    new_digest: str | None = None


class SecretSyncRejected(ValueError):
    pass


def validate_config_text(config_text: str, *, allowed_kinds: Sequence[str]) -> None:
    """Validate fetched config text.

    Raises :class:`SecretSyncRejected` for any Secret manifest and
    :class:`ValueError` for kinds outside ``allowed_kinds`` or unparseable
    YAML. Plain runtime-config YAML (no ``kind``) passes unchanged.
    """
    documents = list(yaml.safe_load_all(config_text))
    for document in documents:
        if not isinstance(document, dict):
            continue
        kind = document.get("kind")
        if not kind:
            continue
        if str(kind).lower() == "secret":
            raise SecretSyncRejected("refusing to sync plaintext Secret manifest")
        if kind not in allowed_kinds:
            raise ValueError(f"unsupported config kind: {kind}")


def fetch_github_config(source: ConfigSyncSource) -> str:
    texts: list[str] = []
    with httpx.Client(timeout=30.0) as client:
        for path in source.paths:
            url = _github_raw_url(source.repository, source.branch, path)
            response = client.get(url)
            response.raise_for_status()
            texts.append(response.text)
    return "\n---\n".join(texts)


def _github_raw_url(repository: str, branch: str, path: str) -> str:
    repo = repository.strip("/")
    encoded_branch = quote(branch, safe="")
    encoded_path = "/".join(quote(part, safe="") for part in path.strip("/").split("/"))
    return f"https://raw.githubusercontent.com/{repo}/{encoded_branch}/{encoded_path}"


def _digest(config_text: str) -> str:
    return f"sha256:{hashlib.sha256(config_text.encode('utf-8')).hexdigest()}"


def _runtime_config_payload(
    config_text: str, *, config_map: str, namespace: str
) -> str:
    """Return the runtime-config YAML payload from fetched text.

    If the source is a ConfigMap document its ``master.yaml`` data key is
    unwrapped; otherwise the plain YAML is returned verbatim. ``config_map``
    and ``namespace`` are retained for call-site compatibility.
    """
    del config_map, namespace
    documents = list(yaml.safe_load_all(config_text))
    for document in documents:
        if not isinstance(document, dict):
            continue
        if document.get("kind") != "ConfigMap":
            continue
        data = document.get("data")
        if isinstance(data, dict) and isinstance(data.get(CONFIG_MAP_KEY), str):
            return data[CONFIG_MAP_KEY]
    return config_text
