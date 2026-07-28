"""Strict offline-testable deployment adapter for the canonical review CVM."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol

from dstack_sdk import EnvVar, encrypt_env_vars_sync

from agent_challenge.review.compose import (
    DEFAULT_REVIEW_APP_IDENTITY,
    REVIEW_ALLOWED_ENVS,
    generate_review_app_compose,
    render_review_app_compose,
    review_app_compose_hash,
)
from agent_challenge.review.deployment import ReviewDeploymentError as ReviewAcknowledgementError
from agent_challenge.review.deployment import (
    build_review_deployed_acknowledgement,
    validate_review_deployed_acknowledgement,
)
from agent_challenge.review.schemas import validate_review_assignment
from agent_challenge.review.urls import (
    PINNED_REVIEW_API_BASE_URL,
    ReviewApiBaseUrlError,
    assert_pinned_review_api_base_url,
)
from agent_challenge.selfdeploy.phala import (
    extract_cvm_id_from_create_response,
    resolve_cvm_id_from_list,
)
from agent_challenge.selfdeploy.provision_identity import (
    DiscoveredPhalaAppIdentity,
    ProvisionIdentityError,
    assert_provision_trust_anchors,
    env_keys_from_allowed,
    optional_verify_env_encrypt_pubkey,
    parse_discovered_identity,
)
from agent_challenge.selfdeploy.shapes import (
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_OS_IMAGE,
    DEFAULT_REVIEW_DISK_SIZE_GB,
    validate_cpu_only,
    validate_disk_size,
)

#: Capacity-safe default (bare ``us-west`` → ERR-02-002 No teepod found).
DEFAULT_REGION = "us-west-1"

#: Phala CREATE-style app_id is a 40-hex handle minted by Phala at provision.
#: Assignment may carry a 40-hex string as an *advisory* pin only — never assert
#: it against the provision response and never send it (or a nonce) on the
#: discovery provision request. Non-hex moniker still feeds compose name and
#: may be sent as app_id alone (legal offline combo).
_APP_ID_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


class ReviewDeploymentError(ReviewAcknowledgementError):
    """A locally prepared review deployment violates its signed assignment."""


@dataclass(frozen=True)
class ReviewDeploymentPlan:
    """Exact review deployment material, with plaintext capability hidden from repr."""

    assignment: dict[str, Any]
    compose: dict[str, Any]
    compose_text: str
    compose_hash: str
    app_identity: str
    image_ref: str
    kms_public_key_hex: str
    kms_public_key_sha256: str
    measurement: dict[str, str]
    measurement_allowlist_sha256: str
    review_session_token: str = field(repr=False)
    instance_type: str = DEFAULT_INSTANCE_TYPE
    region: str = DEFAULT_REGION
    os_image: str = DEFAULT_OS_IMAGE
    #: Stable moniker measured into app-compose ``name`` (compose_hash binding).
    #: Distinct from :attr:`app_identity` when the latter is a Phala 40-hex app_id.
    compose_name: str = DEFAULT_REVIEW_APP_IDENTITY
    #: Deprecated unused field retained so hand-built plans cannot smuggle a
    #: provision ``nonce``. Deploy never emits nonce; discovery omits app_id.
    phala_app_nonce: int | None = None
    disk_size_gb: int = DEFAULT_REVIEW_DISK_SIZE_GB


@dataclass(frozen=True)
class EncryptedReviewSecrets:
    """Validated review secrets; ciphertext may be plan-key or discovered-key.

    ``secret_values`` holds validated plaintext so deploy can re-encrypt to the
    provision-discovered env-encrypt pubkey (assignment KMS pin is not the
    trust anchor for encryption). Never appears in repr.
    """

    ciphertext: str
    env_keys: tuple[str, ...]
    assignment_id: str
    app_identity: str
    kms_public_key_sha256: str
    measurement_allowlist_sha256: str
    secret_values: Mapping[str, str] = field(default_factory=dict, repr=False)


class PhalaPost(Protocol):
    """Minimal production boundary used for provision/create request transmission.

    Implementations may also expose ``get(path)`` for create-ack list fallback.
    """

    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST one request and return a decoded response mapping."""


def build_review_deployment_plan(prepare_response: Mapping[str, Any]) -> ReviewDeploymentPlan:
    """Verify signed Review assignment v1 and derive its exact deployed compose."""

    # The production route wraps the exact assignment in stable identity fields
    # (session_id, assignment_id, attempt).  The older low-level adapter accepts
    # the two-field form used by offline callers.  Neither form allows caller
    # supplied image/KMS/measurement overrides.
    if set(prepare_response) == {"assignment", "review_session_token"}:
        assignment = prepare_response["assignment"]
        token = prepare_response["review_session_token"]
    elif {
        "session_id",
        "assignment_id",
        "attempt",
        "assignment",
        "review_session_token",
    } == set(prepare_response):
        assignment = prepare_response["assignment"]
        token = prepare_response["review_session_token"]
        if (
            not isinstance(assignment, Mapping)
            or assignment.get("assignment_core", {}).get("assignment_id")
            != prepare_response["assignment_id"]
        ):
            raise ReviewDeploymentError("review prepare identity does not match assignment")
    else:
        raise ReviewDeploymentError(
            "prepare response must contain only assignment and one-time token"
        )
    if not isinstance(assignment, Mapping) or not isinstance(token, str) or not token:
        raise ReviewDeploymentError("prepare response lacks immutable assignment or session token")
    try:
        validate_review_assignment(assignment)
    except Exception as exc:
        raise ReviewDeploymentError("review assignment is invalid") from exc
    core = assignment["assignment_core"]
    review_app = core["review_app"]
    if sha256(token.encode("utf-8")).hexdigest() != core["session_token_sha256"]:
        raise ReviewDeploymentError("review session token is not bound to assignment")
    try:
        instance_type = validate_cpu_only(
            instance_type=str(review_app["measurement"]["vm_shape"]).replace("-", "."),
            os_image=DEFAULT_OS_IMAGE,
        ).name
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewDeploymentError(
            "review assignment does not identify a CPU Intel TDX shape"
        ) from exc

    # Compose ``name`` does NOT flip to Phala's 40-hex app_id: changing it would
    # rehash compose_hash. 40-hex app_identity is an advisory Phala pin only;
    # moniker app_identity still feeds compose name (compose_hash binding).
    app_identity = str(review_app["app_identity"])
    compose_name = DEFAULT_REVIEW_APP_IDENTITY
    if _APP_ID_HEX40_RE.fullmatch(app_identity.lower()):
        app_identity = app_identity.lower()
        # Discovery path: Phala mints app_id; do not pre-bind a nonce.
        phala_app_nonce: int | None = None
    else:
        # Legacy moniker-only identity (unit/offline tests): compose name binds
        # the signed moniker. Provision may advertise moniker as app_id alone.
        compose_name = app_identity
        phala_app_nonce = None

    compose = generate_review_app_compose(
        review_image=review_app["image_ref"],
        app_identity=compose_name,
    )
    compose_hash = review_app_compose_hash(compose)
    if compose_hash != review_app["compose_hash"]:
        raise ReviewDeploymentError(
            "signed review compose hash does not match canonical deployment"
        )
    allowlist_sha256 = review_app.get("measurement_allowlist_sha256")
    if not isinstance(allowlist_sha256, str) or not allowlist_sha256:
        raise ReviewDeploymentError(
            "signed review assignment is missing bound measurement allowlist identity"
        )
    allowlist = review_app.get("measurement_allowlist")
    if not isinstance(allowlist, list) or not allowlist:
        raise ReviewDeploymentError(
            "signed review assignment is missing bound measurement allowlist entries"
        )
    return ReviewDeploymentPlan(
        assignment=dict(assignment),
        compose=compose,
        compose_text=render_review_app_compose(compose),
        compose_hash=compose_hash,
        app_identity=app_identity,
        image_ref=review_app["image_ref"],
        kms_public_key_hex=review_app["kms_public_key_hex"],
        kms_public_key_sha256=review_app["kms_public_key_sha256"],
        measurement=dict(review_app["measurement"]),
        measurement_allowlist_sha256=allowlist_sha256,
        review_session_token=token,
        instance_type=instance_type,
        os_image=DEFAULT_OS_IMAGE,
        compose_name=compose_name,
        phala_app_nonce=phala_app_nonce,
        disk_size_gb=DEFAULT_REVIEW_DISK_SIZE_GB,
    )


def encrypt_review_secrets(
    plan: ReviewDeploymentPlan,
    secrets: Mapping[str, str],
    *,
    env_encrypt_pubkey: str | None = None,
    app_identity: str | None = None,
) -> EncryptedReviewSecrets:
    """Validate allowed secrets and encrypt to the given (or plan) X25519 key.

    Deploy re-invokes this after provision discovery with the discovered pubkey
    and app_id. Offline unit tests may call it with the plan key alone.
    """

    if set(secrets) != set(REVIEW_ALLOWED_ENVS):
        raise ReviewDeploymentError("review encrypted_env names must be exactly the allowed names")
    values = {name: secrets[name] for name in REVIEW_ALLOWED_ENVS}
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ReviewDeploymentError("review encrypted_env values must be non-empty strings")
    if values["REVIEW_SESSION_TOKEN"] != plan.review_session_token:
        raise ReviewDeploymentError("review session token does not match signed prepare response")
    # Hard-pin measured callback authority. Miner-supplied non-joinbase values
    # fail closed; honest joinbase (with optional trailing slash) is accepted.
    # Override authority only with CHALLENGE_ALLOW_DEV_URLS=1 (non-prod).
    try:
        values["REVIEW_API_BASE_URL"] = assert_pinned_review_api_base_url(
            values["REVIEW_API_BASE_URL"]
        )
    except ReviewApiBaseUrlError as exc:
        raise ReviewDeploymentError(str(exc)) from exc
    pubkey = env_encrypt_pubkey if env_encrypt_pubkey is not None else plan.kms_public_key_hex
    bound_app_id = app_identity if app_identity is not None else plan.app_identity
    try:
        ciphertext = encrypt_env_vars_sync(
            [EnvVar(key=name, value=values[name]) for name in REVIEW_ALLOWED_ENVS],
            pubkey,
        )
    except Exception as exc:
        raise ReviewDeploymentError("review encrypted_env encryption failed") from exc
    if not ciphertext:
        raise ReviewDeploymentError("review encrypted_env ciphertext is empty")
    return EncryptedReviewSecrets(
        ciphertext=ciphertext,
        env_keys=REVIEW_ALLOWED_ENVS,
        assignment_id=plan.assignment["assignment_core"]["assignment_id"],
        app_identity=bound_app_id,
        kms_public_key_sha256=plan.kms_public_key_sha256,
        measurement_allowlist_sha256=plan.measurement_allowlist_sha256,
        secret_values=dict(values),
    )


class HttpReviewPhalaDeployment:
    """Transmit canonical review provision/create requests through an injected API."""

    def __init__(self, api: PhalaPost) -> None:
        self._api = api

    def deploy(
        self,
        plan: ReviewDeploymentPlan,
        encrypted: EncryptedReviewSecrets,
    ) -> dict[str, str]:
        """Provision (names only) → discover app_id → encrypt → create."""

        if (
            encrypted.assignment_id != plan.assignment["assignment_core"]["assignment_id"]
            or encrypted.kms_public_key_sha256 != plan.kms_public_key_sha256
            or encrypted.measurement_allowlist_sha256 != plan.measurement_allowlist_sha256
            or encrypted.env_keys != REVIEW_ALLOWED_ENVS
            or not encrypted.secret_values
        ):
            raise ReviewDeploymentError("review encrypted_env is not bound to this assignment")

        env_keys = env_keys_from_allowed(REVIEW_ALLOWED_ENVS)
        provision_request: dict[str, Any] = {
            # Compose name stays the product moniker so offline compose_hash
            # matches live. app_id is discovered from the response — do not
            # require the assignment pin here.
            "name": plan.compose_name,
            "instance_type": plan.instance_type,
            "region": plan.region,
            "compose_file": plan.compose,
            "env_keys": env_keys,
            "image": plan.os_image,
            "disk_size": validate_disk_size(plan.disk_size_gb),
        }
        # Phala contract: either (no nonce, no app_id) for discovery, or
        # (app_id alone) for legacy moniker. Never send nonce without app_id
        # (live HTTP 422). Never send nonce at all on this path — Phala mints.
        if plan.app_identity and not _APP_ID_HEX40_RE.fullmatch(plan.app_identity.lower()):
            provision_request["app_id"] = plan.app_identity
        # plan.phala_app_nonce is intentionally ignored (cannot force illegal shape).

        provision = self._api.post("/cvms/provision", provision_request)
        identity = self._verify_provision_response(plan, provision)

        sealed = encrypt_review_secrets(
            plan,
            encrypted.secret_values,
            env_encrypt_pubkey=identity.app_env_encrypt_pubkey,
            app_identity=identity.app_id,
        )
        # Pin site 2: ciphertext binding uses the *discovered* app_id.
        if (
            sealed.app_identity != identity.app_id
            or sealed.assignment_id != plan.assignment["assignment_core"]["assignment_id"]
            or sealed.env_keys != REVIEW_ALLOWED_ENVS
            or not sealed.ciphertext
        ):
            raise ReviewDeploymentError("review encrypted_env is not bound to this assignment")

        create_request = {
            "app_id": identity.app_id,
            "compose_hash": plan.compose_hash,
            "encrypted_env": sealed.ciphertext,
            "env_keys": list(sealed.env_keys),
        }
        created = self._api.post("/cvms", create_request)
        cvm_id = self._resolve_created_cvm_id(identity.app_id, created)
        request_id = created.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            # Numeric create id also serves as request identity when API
            # omits a separate request_id field (live cloud schema).
            request_id = cvm_id
        created_at_ms = created.get("created_at_ms")
        if not isinstance(created_at_ms, int) or isinstance(created_at_ms, bool):
            created_at_ms = 0
        acknowledgement = build_review_deployed_acknowledgement(
            assignment=plan.assignment,
            cvm_id=cvm_id,
            request_id=request_id,
            receipt_sha256=sha256(repr(sorted(created.items())).encode("utf-8")).hexdigest(),
            created_at_ms=created_at_ms,
            app_id=identity.app_id,
        )
        try:
            validate_review_deployed_acknowledgement(plan.assignment, acknowledgement)
        except ReviewAcknowledgementError as exc:
            raise ReviewDeploymentError("generated review acknowledgement is invalid") from exc
        return acknowledgement

    def _resolve_created_cvm_id(
        self,
        discovered_app_id: str,
        created: Mapping[str, Any],
    ) -> str:
        """Map create response (or safe app_id list fallback) to a CVM id string.

        Live Phala create schema returns numeric ``id`` plus ``app_id`` (handle).
        Fail closed only when neither create fields nor GET ``/cvms`` listing by
        discovered ``app_id`` identify a CVM. Never invent measurements or ids.
        """

        try:
            return extract_cvm_id_from_create_response(created)
        except ValueError:
            pass
        list_cvms = getattr(self._api, "list_cvms", None)
        getter = getattr(self._api, "get", None)
        try:
            if callable(list_cvms):
                snap = list_cvms()
                listing = {
                    "items": [dict(x) for x in snap.items],
                    "total": int(snap.total),
                }
            elif callable(getter):
                listing = getter("/cvms")
            else:
                raise ReviewDeploymentError(
                    "Phala create response does not identify the review CVM"
                )
        except ReviewDeploymentError:
            raise
        except Exception as exc:
            raise ReviewDeploymentError(
                "Phala create response does not identify the review CVM"
            ) from exc
        if not isinstance(listing, (Mapping, list)):
            raise ReviewDeploymentError("Phala create response does not identify the review CVM")
        resolved = resolve_cvm_id_from_list(listing, app_id=discovered_app_id)
        if resolved is None:
            raise ReviewDeploymentError("Phala create response does not identify the review CVM")
        return resolved

    @staticmethod
    def _verify_provision_response(
        plan: ReviewDeploymentPlan,
        provision: Mapping[str, Any],
    ) -> DiscoveredPhalaAppIdentity:
        """Hard-check compose_hash + OS; discover app_id (never pin-assert)."""

        try:
            assert_provision_trust_anchors(
                plan_compose_hash=plan.compose_hash,
                plan_measurement=plan.measurement,
                provision=provision,
            )
        except ProvisionIdentityError as exc:
            msg = str(exc)
            if "compose_hash" in msg and "os_image" not in msg:
                raise ReviewDeploymentError(
                    "Phala provision compose hash mismatches signed assignment"
                ) from exc
            # Preserve catalog / dstack_mr_image detail; only rewrite plain OS mismatch.
            if "dstack_mr_image" in msg or "catalog" in msg:
                raise ReviewDeploymentError(msg) from exc
            if "os_image_hash" in msg or "measurement" in msg:
                raise ReviewDeploymentError(
                    "Phala provision os_image_hash mismatches signed assignment measurement"
                ) from exc
            raise ReviewDeploymentError(msg) from exc
        try:
            identity = parse_discovered_identity(provision)
            optional_verify_env_encrypt_pubkey(identity)
        except ProvisionIdentityError as exc:
            raise ReviewDeploymentError(str(exc)) from exc
        return identity


class ReviewPhalaDeployment(HttpReviewPhalaDeployment):
    """In-memory Phala adapter used to capture the exact offline request contract."""

    def __init__(
        self,
        *,
        provision_response: Mapping[str, Any],
        create_response: Mapping[str, Any],
        list_response: Mapping[str, Any] | None = None,
    ) -> None:
        self.provision_response = dict(provision_response)
        self.create_response = dict(create_response)
        self.list_response = dict(list_response) if list_response is not None else None
        self.provision_requests: list[dict[str, Any]] = []
        self.create_requests: list[dict[str, Any]] = []
        self.get_paths: list[str] = []
        super().__init__(self)

    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = dict(payload)
        if path == "/cvms/provision":
            self.provision_requests.append(request)
            return self.provision_response
        if path == "/cvms":
            self.create_requests.append(request)
            return self.create_response
        raise AssertionError(f"unexpected Phala API path {path}")

    def get(self, path: str) -> Mapping[str, Any]:
        self.get_paths.append(path)
        if path != "/cvms" or self.list_response is None:
            raise AssertionError(f"unexpected Phala GET path {path}")
        return self.list_response


__all__ = [
    "DEFAULT_OS_IMAGE",
    "DEFAULT_REGION",
    "PINNED_REVIEW_API_BASE_URL",
    "REVIEW_ALLOWED_ENVS",
    "EncryptedReviewSecrets",
    "HttpReviewPhalaDeployment",
    "ReviewDeploymentError",
    "ReviewDeploymentPlan",
    "ReviewPhalaDeployment",
    "build_review_deployment_plan",
    "encrypt_review_secrets",
]
