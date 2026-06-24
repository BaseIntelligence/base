"""Backend-neutral container image reference + registry digest helpers.

Pure functions with no orchestrator coupling (only ``httpx`` for registry
calls). Used by the Swarm supervisor image updaters and the CLI to parse
image references and resolve remote tag digests against an OCI registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.request import parse_http_list, parse_keqv_list

import httpx

DIGEST_ANNOTATION = "joinbase.ai/image-digest"
MEDIA_TYPES = ", ".join(
    [
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
    ]
)
_DIGEST_RE = re.compile(r"sha256:[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class ImageReference:
    registry: str
    repository: str
    tag: str
    digest: str | None = None

    @property
    def immutable(self) -> bool:
        return self.digest is not None

    @property
    def tagged(self) -> str:
        return f"{self.registry}/{self.repository}:{self.tag}"

    def pinned(self, digest: str) -> str:
        return f"{self.tagged}@{digest}"


def parse_image_reference(image: str) -> ImageReference:
    name, _, digest = image.partition("@")
    digest_value = digest or None
    if "/" not in name:
        registry = "docker.io"
        remainder = f"library/{name}"
    else:
        first, rest = name.split("/", 1)
        if "." in first or ":" in first or first == "localhost":
            registry = first
            remainder = rest
        else:
            registry = "docker.io"
            remainder = name
    repository, separator, tag = remainder.rpartition(":")
    if not separator or "/" in tag:
        repository = remainder
        tag = "latest"
    return ImageReference(
        registry=registry,
        repository=repository,
        tag=tag,
        digest=digest_value,
    )


def extract_digest(value: str | None) -> str | None:
    if not value:
        return None
    match = _DIGEST_RE.search(value)
    return match.group(0).lower() if match else None


def _parse_www_authenticate(header: str) -> dict[str, str]:
    scheme, _, params = header.partition(" ")
    if scheme.lower() != "bearer":
        return {}
    parsed = parse_keqv_list(parse_http_list(params))
    return {str(key): str(value) for key, value in parsed.items()}


def resolve_remote_digest(
    image: ImageReference,
    *,
    registry_endpoint: str | None = None,
    timeout_seconds: float = 30.0,
) -> str:
    if image.digest:
        return image.digest.lower()
    base_url = (
        registry_endpoint.rstrip("/")
        if registry_endpoint
        else f"https://{image.registry}"
    )
    manifest_path = f"/v2/{image.repository}/manifests/{image.tag}"
    headers = {"Accept": MEDIA_TYPES}
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.head(f"{base_url}{manifest_path}", headers=headers)
        if response.status_code == 401:
            challenge = _parse_www_authenticate(
                response.headers.get("www-authenticate", "")
            )
            if challenge.get("realm"):
                token_response = client.get(
                    challenge["realm"],
                    params={
                        key: value
                        for key, value in {
                            "service": challenge.get("service"),
                            "scope": challenge.get("scope"),
                        }.items()
                        if value
                    },
                )
                token_response.raise_for_status()
                token = token_response.json().get("token") or token_response.json().get(
                    "access_token"
                )
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                    response = client.head(
                        f"{base_url}{manifest_path}", headers=headers
                    )
        response.raise_for_status()
    digest = response.headers.get("Docker-Content-Digest")
    parsed = extract_digest(digest)
    if not parsed:
        raise RuntimeError(
            f"registry did not return a sha256 digest for {image.tagged}"
        )
    return parsed
