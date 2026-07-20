"""Ordered, encrypted deployment of the canonical Eval application.

This module is deliberately independent from the legacy ``deploy`` helper.
Eval deployment accepts only the validator-issued Eval plan produced after a
verified review allow, derives the canonical compose from that plan, and sends
the resulting ciphertext to Phala.  It never creates database state or invents
an authorization locally.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Protocol

from dstack_sdk import EnvVar, encrypt_env_vars_sync

from agent_challenge.canonical import eval_wire
from agent_challenge.canonical.compose import (
    DEFAULT_ALLOWED_ENVS,
    generate_app_compose,
    render_app_compose,
)
from agent_challenge.selfdeploy.measurements import (
    ProvisionOsIdentityError,
    verify_provision_os_identity,
)
from agent_challenge.selfdeploy.phala import (
    extract_cvm_id_from_create_response,
    resolve_cvm_id_from_list,
)
from agent_challenge.selfdeploy.shapes import (
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_OS_IMAGE,
    validate_cpu_only,
)

#: Capacity-safe default (bare ``us-west`` → ERR-02-002 No teepod found).
DEFAULT_REGION = "us-west-1"
EVAL_ALLOWED_ENVS: tuple[str, ...] = DEFAULT_ALLOWED_ENVS
# VAL-ACAT-013: production eval encrypted_env must NOT require Base LLM gateway
# secrets. Gateway routing is removed; only eval-run capability + attestation
# plan bindings (and optional cost limit) are required.
EVAL_REQUIRED_SECRET_ENVS: frozenset[str] = frozenset(
    {
        "CHALLENGE_PHALA_ATTESTATION_ENABLED",
        "CHALLENGE_PHALA_EVAL_PLAN",
        "EVAL_RUN_TOKEN",
        "LLM_COST_LIMIT",
    }
)

#: Product moniker seeds measured compose ``name`` (compose_hash). Phala 40-hex
#: app_id is pinned separately as plan app_identity when using deterministic
#: provision (nonce). Never invent moniker→hex melt.
DEFAULT_EVAL_COMPOSE_NAME = "agent-challenge-eval-v1"
_APP_ID_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
#: Default nonce for the eval domain (disjoint from review's 0).
DEFAULT_EVAL_PHALA_APP_NONCE = 1
#: Measure-time offline pin placeholder for ``key_release_url`` when the operator
#: pin pack was built without baking a live RA-TLS authority into the measured
#: app-compose. The live residual pin ``04011776…`` used this HTTPS value so the
#: compose_hash is stable across operator endpoint changes. Guest still resolves
#: the real endpoint from the signed plan /
#: ``CHALLENGE_PHALA_EVAL_PLAN.key_release_endpoint`` (never invent KR materials).
MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER = "https://validator-kr.example.invalid:8701"


class EvalDeploymentError(ValueError):
    """The validator-issued Eval plan or deployment request is unsafe."""

    attributable_cvm_id: str | None = None


@dataclass(frozen=True)
class EvalDeploymentPlan:
    """Canonical Eval deployment material.

    The run token is intentionally excluded from the normal representation.
    Callers should encrypt it immediately and should never serialize this
    object as evidence or status.
    """

    plan: dict[str, Any]
    plan_sha256: str
    compose: dict[str, Any]
    compose_text: str
    compose_hash: str
    app_identity: str
    image_ref: str
    kms_public_key_hex: str
    kms_public_key_sha256: str
    measurement: dict[str, str]
    eval_run_id: str
    eval_run_token: str = field(repr=False)
    instance_type: str = DEFAULT_INSTANCE_TYPE
    region: str = DEFAULT_REGION
    os_image: str = DEFAULT_OS_IMAGE
    compose_name: str = DEFAULT_EVAL_COMPOSE_NAME
    phala_app_nonce: int | None = None


@dataclass(frozen=True)
class EncryptedEvalSecrets:
    """Ciphertext-only Eval secret delivery."""

    ciphertext: str
    env_keys: tuple[str, ...]
    eval_run_id: str
    app_identity: str
    kms_public_key_sha256: str


class PhalaPost(Protocol):
    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """POST one Phala Cloud API request."""


def _plan_digest(plan: Mapping[str, Any]) -> str:
    try:
        canonical = eval_wire.canonical_json_v1(eval_wire.validate_eval_plan(plan))
    except eval_wire.EvalWireError as exc:
        raise EvalDeploymentError("Eval plan is not canonical") from exc
    return sha256(canonical).hexdigest()


def build_eval_deployment_plan(
    prepare_response: Mapping[str, Any],
) -> EvalDeploymentPlan:
    """Validate the exact signed Eval prepare wrapper and derive its compose.

    The response must be the first wrapper returned by the production signed
    ``POST /submissions/{id}/eval/prepare`` route.  That route is the sole
    authorization gate and only returns after a persisted verified review allow.
    This helper intentionally has no caller-controlled authorization boolean.
    """

    if not isinstance(prepare_response, Mapping):
        raise EvalDeploymentError("Eval prepare response must be an object")
    if set(prepare_response) != {"schema_version", "plan", "plan_sha256", "secret_delivery"}:
        raise EvalDeploymentError("Eval prepare response has unexpected fields")
    if prepare_response["schema_version"] != 1:
        raise EvalDeploymentError("unsupported Eval prepare schema version")
    plan_raw = prepare_response["plan"]
    if not isinstance(plan_raw, Mapping):
        raise EvalDeploymentError("Eval prepare response has no immutable plan")
    try:
        plan = eval_wire.validate_eval_plan(plan_raw)
    except eval_wire.EvalWireError as exc:
        raise EvalDeploymentError("Eval plan is invalid") from exc
    expected_digest = _plan_digest(plan)
    if prepare_response["plan_sha256"] != expected_digest:
        raise EvalDeploymentError("Eval plan digest does not match canonical plan bytes")
    if (
        not isinstance(plan["authorizing_review_digest"], str)
        or not plan["authorizing_review_digest"]
    ):
        raise EvalDeploymentError("Eval plan is missing validator review authorization")
    delivery = prepare_response["secret_delivery"]
    if not isinstance(delivery, Mapping) or set(delivery) != {"env_key", "token"}:
        raise EvalDeploymentError(
            "first Eval prepare must deliver exactly one EVAL_RUN_TOKEN capability"
        )
    if delivery["env_key"] != "EVAL_RUN_TOKEN" or not isinstance(delivery["token"], str):
        raise EvalDeploymentError("Eval prepare delivered an invalid run capability")
    token = delivery["token"]
    if not token or sha256(token.encode("utf-8")).hexdigest() != plan["run_token_sha256"]:
        raise EvalDeploymentError("Eval run token is not bound to the immutable plan")

    app = plan["eval_app"]
    try:
        shape_name = str(app["measurement"]["vm_shape"]).replace("-", ".")
        shape = validate_cpu_only(instance_type=shape_name)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalDeploymentError("Eval plan does not identify a CPU Intel TDX shape") from exc
    # The app identity, KMS key, measurement, and image all come from the
    # validator-signed plan.  Never accept a CLI override for any of them.
    allowed = set(EVAL_ALLOWED_ENVS)
    # The signed plan pins the exact compose_hash. Offline/default depends omit
    # the live-registry side-manifest; live smoke pins it. Operator pin packs may
    # also have been measured with a non-routable HTTPS key-release placeholder
    # (compositionally stable; guest uses signed plan endpoint at runtime).
    # Choose the generator mode whose rendered hash matches the signed plan
    # fail-closed — never invent compose bytes / MRTD / KR roots.
    live_registry_candidates = (
        None,
        "/opt/agent-challenge/golden/live-registry-refs.json",
    )
    # Prefer plan endpoint, then measure-time placeholder used for the live
    # joinbase pin ``04011776…`` (tee-pin-pack / eval residual after KR).
    plan_endpoint = str(plan.get("key_release_endpoint") or "").strip() or None
    key_release_candidates: list[str | None] = []
    for candidate_url in (
        plan_endpoint,
        MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER,
    ):
        if candidate_url and candidate_url not in key_release_candidates:
            key_release_candidates.append(candidate_url)
    if not key_release_candidates:
        key_release_candidates.append(None)
    compose = None
    compose_text = ""
    compose_hash = ""
    app_identity = str(app["app_identity"])
    if _APP_ID_HEX40_RE.fullmatch(app_identity.lower()):
        app_identity = app_identity.lower()
        compose_name = DEFAULT_EVAL_COMPOSE_NAME
        phala_app_nonce: int | None = DEFAULT_EVAL_PHALA_APP_NONCE
    else:
        compose_name = app_identity
        phala_app_nonce = None
    name_candidates = (compose_name,)
    # Also try signed identity as compost name for moniker-only legacy pins.
    if compose_name != app_identity:
        name_candidates = (compose_name, app_identity)
    for live_path in live_registry_candidates:
        for name in name_candidates:
            for key_release_url in key_release_candidates:
                candidate = generate_app_compose(
                    orchestrator_image=app["image_ref"],
                    name=name,
                    key_release_url=key_release_url,
                    allowed_envs=tuple(sorted(allowed)),
                    live_registry_manifest_path=live_path,
                )
                candidate_text = render_app_compose(candidate)
                candidate_hash = sha256(candidate_text.encode("utf-8")).hexdigest()
                if candidate_hash == app["compose_hash"]:
                    compose = candidate
                    compose_text = candidate_text
                    compose_hash = candidate_hash
                    compose_name = name
                    break
            if compose is not None:
                break
        if compose is not None:
            break
    if compose is None or compose_hash != app["compose_hash"]:
        raise EvalDeploymentError("canonical Eval compose hash mismatches signed plan")
    if app["kms_key_algorithm"] != "x25519":
        raise EvalDeploymentError("Eval plan uses an unsupported KMS algorithm")
    if sha256(bytes.fromhex(app["kms_public_key_hex"])).hexdigest() != app["kms_public_key_sha256"]:
        raise EvalDeploymentError("Eval KMS public key digest mismatch")
    return EvalDeploymentPlan(
        plan=dict(plan),
        plan_sha256=expected_digest,
        compose=compose,
        compose_text=compose_text,
        compose_hash=compose_hash,
        app_identity=app_identity,
        image_ref=app["image_ref"],
        kms_public_key_hex=app["kms_public_key_hex"],
        kms_public_key_sha256=app["kms_public_key_sha256"],
        measurement=dict(app["measurement"]),
        eval_run_id=plan["eval_run_id"],
        eval_run_token=token,
        instance_type=shape.name,
        os_image=DEFAULT_OS_IMAGE,
        compose_name=compose_name,
        phala_app_nonce=phala_app_nonce,
    )


def encrypt_eval_secrets(
    plan: EvalDeploymentPlan,
    secrets: Mapping[str, str],
) -> EncryptedEvalSecrets:
    """Encrypt the Eval run token and attestation plan bindings (no Base gateway)."""

    if not set(secrets) <= set(EVAL_ALLOWED_ENVS) or not EVAL_REQUIRED_SECRET_ENVS <= set(secrets):
        raise EvalDeploymentError(
            "Eval encrypted_env names must be scoped allowed names with the required run "
            "and attestation plan capabilities (Base LLM gateway secrets are not allowed)"
        )
    forbidden_gateway = {
        "BASE_GATEWAY_TOKEN",
        "BASE_LLM_GATEWAY_URL",
        "GATEWAY_TOKEN",
        "CENTRAL_GATEWAY_TOKEN",
    }
    if forbidden_gateway & set(secrets):
        raise EvalDeploymentError(
            "Eval encrypted_env must not include Base LLM gateway secrets "
            "(BASE_GATEWAY_TOKEN / BASE_LLM_GATEWAY_URL / …)"
        )
    env_keys = tuple(name for name in EVAL_ALLOWED_ENVS if name in secrets)
    values = {name: secrets[name] for name in env_keys}
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise EvalDeploymentError("Eval encrypted_env values must be non-empty strings")
    if values["EVAL_RUN_TOKEN"] != plan.eval_run_token:
        raise EvalDeploymentError("Eval run token does not match signed prepare response")
    try:
        ciphertext = encrypt_env_vars_sync(
            [EnvVar(key=name, value=values[name]) for name in env_keys],
            plan.kms_public_key_hex,
        )
    except Exception as exc:
        raise EvalDeploymentError("Eval encrypted_env encryption failed") from exc
    if not ciphertext:
        raise EvalDeploymentError("Eval encrypted_env ciphertext is empty")
    return EncryptedEvalSecrets(
        ciphertext=ciphertext,
        env_keys=env_keys,
        eval_run_id=plan.eval_run_id,
        app_identity=plan.app_identity,
        kms_public_key_sha256=plan.kms_public_key_sha256,
    )


class HttpEvalPhalaDeployment:
    """Transmit exact provision/create bytes to Phala Cloud."""

    def __init__(self, api: PhalaPost) -> None:
        self._api = api

    def deploy(
        self,
        plan: EvalDeploymentPlan,
        encrypted: EncryptedEvalSecrets,
    ) -> dict[str, str]:
        if (
            encrypted.eval_run_id != plan.eval_run_id
            or encrypted.app_identity != plan.app_identity
            or encrypted.kms_public_key_sha256 != plan.kms_public_key_sha256
            or not set(encrypted.env_keys) <= set(EVAL_ALLOWED_ENVS)
            or not encrypted.ciphertext
        ):
            raise EvalDeploymentError("Eval encrypted_env is not bound to this run")
        provision_request: dict[str, Any] = {
            "app_id": plan.app_identity,
            "name": plan.compose_name,
            "instance_type": plan.instance_type,
            "region": plan.region,
            "compose_file": plan.compose,
            "env_keys": list(encrypted.env_keys),
            "image": plan.os_image,
        }
        if plan.phala_app_nonce is not None:
            provision_request["nonce"] = plan.phala_app_nonce
        provision = self._api.post("/cvms/provision", provision_request)
        if provision.get("compose_hash") != plan.compose_hash:
            raise EvalDeploymentError("Phala provision compose hash mismatches Eval plan")
        # Honest: equality to pin only. Production pins are Phala 40-hex app_id
        # (plus nonce) so Phala returns the same app_id and stable encrypt pubkey.
        if provision.get("app_id") != plan.app_identity:
            raise EvalDeploymentError("Phala provision app identity mismatches Eval plan")
        if provision.get("app_env_encrypt_pubkey") != plan.kms_public_key_hex:
            raise EvalDeploymentError("Phala provision KMS key mismatches Eval plan")
        self._verify_provision_os_identity(plan, provision)
        created = self._api.post(
            "/cvms",
            {
                "app_id": plan.app_identity,
                "compose_hash": plan.compose_hash,
                "encrypted_env": encrypted.ciphertext,
                "env_keys": list(encrypted.env_keys),
            },
        )
        # Match review path: live Phala create uses numeric id; coerce + fallback.
        try:
            cvm_id = extract_cvm_id_from_create_response(created)
        except ValueError:
            cvm_id = None
            getter = getattr(self._api, "get", None)
            if callable(getter):
                try:
                    listing = getter("/cvms")
                except Exception:
                    listing = None
                if isinstance(listing, Mapping):
                    cvm_id = resolve_cvm_id_from_list(listing, app_id=plan.app_identity)
        if not isinstance(cvm_id, str) or not cvm_id:
            raise EvalDeploymentError("Phala create response does not identify the Eval CVM")
        try:
            return {
                "eval_run_id": plan.eval_run_id,
                "cvm_id": cvm_id,
                "app_identity": plan.app_identity,
                "image_ref": plan.image_ref,
                "compose_hash": plan.compose_hash,
                "kms_public_key_sha256": plan.kms_public_key_sha256,
                "phala_create_receipt_sha256": sha256(
                    repr(sorted(created.items())).encode("utf-8")
                ).hexdigest(),
            }
        except Exception as exc:  # pragma: no cover - defensive post-create binder
            if isinstance(exc, EvalDeploymentError):
                exc.attributable_cvm_id = cvm_id
            else:
                wrapped = EvalDeploymentError(str(exc))
                wrapped.attributable_cvm_id = cvm_id
                raise wrapped from exc
            raise

    @staticmethod
    def _verify_provision_os_identity(
        plan: EvalDeploymentPlan,
        provision: Mapping[str, Any],
    ) -> None:
        try:
            verify_provision_os_identity(
                measurement=plan.measurement,
                provision_os=provision.get("os_image_hash"),
                mismatch_message=("Phala provision os_image_hash mismatches Eval plan measurement"),
            )
        except ProvisionOsIdentityError as exc:
            raise EvalDeploymentError(str(exc)) from exc


class EvalPhalaDeployment(HttpEvalPhalaDeployment):
    """In-memory adapter used by contract tests."""

    def __init__(
        self,
        *,
        provision_response: Mapping[str, Any],
        create_response: Mapping[str, Any],
    ) -> None:
        self.provision_response = dict(provision_response)
        self.create_response = dict(create_response)
        self.provision_requests: list[dict[str, Any]] = []
        self.create_requests: list[dict[str, Any]] = []
        super().__init__(self)

    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if path == "/cvms/provision":
            self.provision_requests.append(dict(payload))
            return self.provision_response
        if path == "/cvms":
            self.create_requests.append(dict(payload))
            return self.create_response
        raise AssertionError(f"unexpected Phala API path {path}")


__all__ = [
    "DEFAULT_EVAL_COMPOSE_NAME",
    "DEFAULT_EVAL_PHALA_APP_NONCE",
    "DEFAULT_OS_IMAGE",
    "DEFAULT_REGION",
    "EVAL_ALLOWED_ENVS",
    "EVAL_REQUIRED_SECRET_ENVS",
    "MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER",
    "EncryptedEvalSecrets",
    "EvalDeploymentError",
    "EvalDeploymentPlan",
    "EvalPhalaDeployment",
    "HttpEvalPhalaDeployment",
    "build_eval_deployment_plan",
    "encrypt_eval_secrets",
]
