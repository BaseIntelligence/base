"""Authenticated GHCR digest resolution (G4 / VAL-CODE-UPD-001).

The image-updater digest resolver must be able to resolve PRIVATE
``ghcr.io/baseintelligence/*`` digests via a credentialed token, not only the
anonymous (public-package) path. Registry traffic is faked with
``httpx.MockTransport`` (no network): the token endpoint only mints a
PULL-scoped token when Basic-auth credentials are presented, mirroring GHCR's
behaviour for a private package.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from base.supervisor import image_ref
from base.supervisor.image_ref import (
    ImageReference,
    RegistryCredentials,
    build_registry_digest_resolver,
    load_registry_credentials,
    parse_image_reference,
    resolve_remote_digest,
)

DIGEST = "sha256:" + "a" * 64
REPO = "baseintelligence/base-master"
TAG = "latest"
MANIFEST_PATH = f"/v2/{REPO}/manifests/{TAG}"
REALM = "https://ghcr.io/token"
PRIVATE_IMAGE = parse_image_reference(f"ghcr.io/{REPO}:{TAG}")


def _private_registry_transport() -> tuple[httpx.MockTransport, dict[str, object]]:
    """A PRIVATE GHCR repo: a pull-scoped token requires Basic auth.

    Returns the transport plus a state dict recording the Authorization header
    seen at the token endpoint (so a test can assert credentials were sent).
    """
    state: dict[str, object] = {"token_auth": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == MANIFEST_PATH:
            if request.headers.get("Authorization") == "Bearer good-token":
                return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})
            return httpx.Response(
                401,
                headers={
                    "www-authenticate": (
                        f'Bearer realm="{REALM}",service="ghcr.io",'
                        f'scope="repository:{REPO}:pull"'
                    )
                },
            )
        if str(request.url).startswith(REALM):
            authz = request.headers.get("Authorization")
            state["token_auth"] = authz
            # Private package: only credentialed (Basic-auth) requests get a
            # token that can actually pull; anonymous gets an unscoped token.
            if authz and authz.startswith("Basic "):
                return httpx.Response(200, json={"token": "good-token"})
            return httpx.Response(200, json={"token": "anon-token"})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    return httpx.MockTransport(handler), state


def test_authenticated_resolution_resolves_private_digest() -> None:
    transport, state = _private_registry_transport()
    credentials = RegistryCredentials(username="ci-bot", password="ghp_secret")

    digest = resolve_remote_digest(
        PRIVATE_IMAGE, credentials=credentials, transport=transport
    )

    assert digest == DIGEST
    # The token endpoint received Basic auth carrying the credentials.
    token_auth = state["token_auth"]
    assert isinstance(token_auth, str) and token_auth.startswith("Basic ")
    decoded = base64.b64decode(token_auth.split(" ", 1)[1]).decode("utf-8")
    assert decoded == "ci-bot:ghp_secret"


def test_anonymous_resolution_fails_on_private_repo() -> None:
    transport, _state = _private_registry_transport()
    # No credentials => only an unscoped (anonymous) token => the manifest HEAD
    # stays 401 => raise_for_status raises (the pre-fix anonymous-only failure).
    with pytest.raises(httpx.HTTPStatusError):
        resolve_remote_digest(PRIVATE_IMAGE, transport=transport)


def test_load_credentials_from_docker_config(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    auth = base64.b64encode(b"ci-bot:ghp_secret").decode("ascii")
    config.write_text(json.dumps({"auths": {"ghcr.io": {"auth": auth}}}))

    creds = load_registry_credentials("ghcr.io", docker_config_path=config)

    assert creds == RegistryCredentials(username="ci-bot", password="ghp_secret")


def test_load_credentials_from_docker_config_missing_registry(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"auths": {"other.io": {"auth": "x"}}}))

    assert load_registry_credentials("ghcr.io", docker_config_path=config) is None


def test_load_credentials_credstore_only_returns_none(tmp_path: Path) -> None:
    # A credsStore entry has no inline ``auth`` field — nothing to decode.
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"auths": {"ghcr.io": {}}, "credsStore": "desktop"}))

    assert load_registry_credentials("ghcr.io", docker_config_path=config) is None


def test_load_credentials_explicit_username_and_password_file(tmp_path: Path) -> None:
    pw = tmp_path / "token"
    pw.write_text("ghp_secret\n")

    creds = load_registry_credentials("ghcr.io", username="ci-bot", password_file=pw)

    assert creds == RegistryCredentials(username="ci-bot", password="ghp_secret")


def test_load_credentials_none_when_nothing_resolvable(tmp_path: Path) -> None:
    assert (
        load_registry_credentials(
            "ghcr.io", docker_config_path=tmp_path / "absent.json"
        )
        is None
    )


def test_build_resolver_without_credentials_is_anonymous() -> None:
    assert build_registry_digest_resolver(None) is resolve_remote_digest


def test_build_resolver_attaches_credentials_only_for_matching_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[RegistryCredentials | None] = []

    def fake_resolve(
        reference: ImageReference,
        *,
        credentials: RegistryCredentials | None = None,
        timeout_seconds: float = 30.0,
    ) -> str:
        seen.append(credentials)
        return DIGEST

    monkeypatch.setattr(image_ref, "resolve_remote_digest", fake_resolve)
    credentials = RegistryCredentials(username="ci-bot", password="ghp_secret")
    resolve = build_registry_digest_resolver(credentials, registry="ghcr.io")

    resolve(parse_image_reference("ghcr.io/baseintelligence/base-master:latest"))
    resolve(parse_image_reference("docker.io/library/postgres:16"))

    # ghcr.io ref carries the credentials; the third-party ref stays anonymous.
    assert seen == [credentials, None]
