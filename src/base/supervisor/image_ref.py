"""Backend-neutral container image reference + registry digest helpers.

Pure functions with no orchestrator coupling (only ``httpx`` for registry
calls). Used by the Swarm supervisor image updaters and the CLI to parse
image references and resolve remote tag digests against an OCI registry.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import parse_http_list, parse_keqv_list

import httpx

logger = logging.getLogger(__name__)

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
class RegistryCredentials:
    """Username + secret used to mint an authenticated registry bearer token.

    For GHCR this is a GitHub username (or any non-empty placeholder) plus a PAT
    / ``GITHUB_TOKEN`` with ``read:packages`` scope. The credentials are sent as
    HTTP Basic auth to the realm's token endpoint so the returned bearer token
    carries pull scope for PRIVATE ``ghcr.io/baseintelligence/*`` packages
    (anonymous tokens only resolve PUBLIC packages).
    """

    username: str
    password: str


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
    credentials: RegistryCredentials | None = None,
    timeout_seconds: float = 30.0,
    transport: httpx.BaseTransport | None = None,
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
    client_kwargs: dict[str, object] = {"timeout": timeout_seconds}
    if transport is not None:
        client_kwargs["transport"] = transport
    with httpx.Client(**client_kwargs) as client:  # type: ignore[arg-type]
        response = client.head(f"{base_url}{manifest_path}", headers=headers)
        if response.status_code == 401:
            challenge = _parse_www_authenticate(
                response.headers.get("www-authenticate", "")
            )
            if challenge.get("realm"):
                # Authenticate the token request with Basic auth when credentials
                # are supplied; the realm then returns a token scoped to PULL the
                # PRIVATE repository (an anonymous token would only resolve PUBLIC
                # packages and the manifest HEAD below would 401 again).
                auth = (
                    httpx.BasicAuth(credentials.username, credentials.password)
                    if credentials is not None
                    else None
                )
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
                    auth=auth,
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


def _read_secret_file(path: str | Path) -> str | None:
    """Return the stripped contents of a secret file, or None if unreadable."""
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _credentials_from_docker_config(
    registry: str, config_path: str | Path
) -> RegistryCredentials | None:
    """Decode ``auths[<registry>].auth`` (base64 ``user:password``) if present.

    Reads the docker ``config.json`` written by ``docker login <registry>``
    (e.g. the manager's ``/root/.docker/config.json`` created by the installer's
    ``ghcr_login`` step). A ``credsStore``/``credHelpers`` entry stores the
    secret OUTSIDE this file, so when the ``auth`` field is absent this returns
    None and the caller should fall back to explicit credentials. Never raises.
    """
    try:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    auths = raw.get("auths")
    if not isinstance(auths, dict):
        return None
    entry = None
    for key in (registry, f"https://{registry}", f"https://{registry}/v2/"):
        candidate = auths.get(key)
        if isinstance(candidate, dict):
            entry = candidate
            break
    if entry is None:
        return None
    encoded = entry.get("auth")
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, ValueError):
        return None
    username, separator, password = decoded.partition(":")
    if not separator or not username or not password:
        return None
    return RegistryCredentials(username=username, password=password)


def load_registry_credentials(
    registry: str,
    *,
    username: str | None = None,
    password: str | None = None,
    password_file: str | Path | None = None,
    docker_config_path: str | Path | None = None,
) -> RegistryCredentials | None:
    """Resolve registry credentials from explicit config or a docker config.json.

    Precedence: an explicit ``username`` + (``password`` or ``password_file``)
    wins; otherwise the base64 ``auth`` for ``registry`` in ``docker_config_path``
    is used. Returns None when nothing is resolvable, so the digest resolver
    stays anonymous (the PUBLIC-package path, behaviour-preserving when unset).
    """
    resolved_password = password
    if resolved_password is None and password_file is not None:
        resolved_password = _read_secret_file(password_file)
    if username and resolved_password:
        return RegistryCredentials(username=username, password=resolved_password)
    if docker_config_path is not None:
        return _credentials_from_docker_config(registry, docker_config_path)
    return None


def build_registry_digest_resolver(
    credentials: RegistryCredentials | None,
    *,
    registry: str = "ghcr.io",
    timeout_seconds: float = 30.0,
) -> Callable[[ImageReference], str]:
    """Build a digest resolver that authenticates ONLY for ``registry``.

    With no credentials this returns the anonymous :func:`resolve_remote_digest`
    unchanged. With credentials it attaches them when resolving a reference whose
    registry matches ``registry`` (so a third-party public image is still
    resolved anonymously), enabling PRIVATE ``ghcr.io/baseintelligence/*`` pulls.
    """
    if credentials is None:
        return resolve_remote_digest

    def resolve(reference: ImageReference) -> str:
        creds = credentials if reference.registry == registry else None
        return resolve_remote_digest(
            reference, credentials=creds, timeout_seconds=timeout_seconds
        )

    return resolve
