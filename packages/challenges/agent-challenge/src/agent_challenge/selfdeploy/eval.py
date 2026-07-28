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
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Protocol

from dstack_sdk import EnvVar, encrypt_env_vars_sync

from agent_challenge.canonical import eval_wire
from agent_challenge.canonical.compose import (
    DEFAULT_ALLOWED_ENVS,
    generate_app_compose,
    render_app_compose,
)
from agent_challenge.canonical.key_release_endpoint import parse_key_release_authority
from agent_challenge.keyrelease.client import KEY_RELEASE_URL_ENV
from agent_challenge.selfdeploy.measurements import (
    ProvisionOsIdentityError,
    verify_provision_os_identity,
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
    DEFAULT_EVAL_DISK_SIZE_GB,
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_OS_IMAGE,
    validate_cpu_only,
    validate_disk_size,
)

#: Capacity-safe default (bare ``us-west`` → ERR-02-002 No teepod found).
DEFAULT_REGION = "us-west-1"
EVAL_ALLOWED_ENVS: tuple[str, ...] = DEFAULT_ALLOWED_ENVS
#: Env names for guest-side miner ZIP fetch (evaluation/artifact_import.py).
EVAL_ARTIFACT_URL_ENV = "CHALLENGE_PHALA_EVAL_ARTIFACT_URL"
EVAL_ARTIFACT_TOKEN_ENV = "CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN"
#: Short-lived grant TTL for one eval run. Eval wall-clock is typically well
#: under an hour (task suite + DooD); 2h covers retries/queue jitter without
#: leaving a long-lived download capability on a leaked guest env dump.
EVAL_ARTIFACT_GRANT_TTL = timedelta(hours=2)
# VAL-ACAT-013: production eval encrypted_env must NOT require Base LLM gateway
# secrets. Gateway routing is removed; only eval-run capability + attestation
# plan bindings (and optional cost limit) are required. Artifact URL+token are
# required so the guest can prove it executed the uploaded miner ZIP.
EVAL_REQUIRED_SECRET_ENVS: frozenset[str] = frozenset(
    {
        "CHALLENGE_PHALA_ATTESTATION_ENABLED",
        "CHALLENGE_PHALA_EVAL_PLAN",
        EVAL_ARTIFACT_TOKEN_ENV,
        EVAL_ARTIFACT_URL_ENV,
        "EVAL_RUN_TOKEN",
        "LLM_COST_LIMIT",
    }
)

#: Product moniker seeds measured compose ``name`` (compose_hash). A 40-hex
#: ``app_identity`` is an *advisory* Phala handle only — never asserted against
#: the provision response and never sent on the discovery provision request.
#: Discover the real app_id from provision; compose ``name`` stays
#: :data:`DEFAULT_EVAL_COMPOSE_NAME` on the hex/absent path.
DEFAULT_EVAL_COMPOSE_NAME = "agent-challenge-eval-v1"
_APP_ID_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
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
    #: Deprecated unused field; deploy never emits provision nonce.
    phala_app_nonce: int | None = None
    disk_size_gb: int = DEFAULT_EVAL_DISK_SIZE_GB


@dataclass(frozen=True)
class EncryptedEvalSecrets:
    """Ciphertext-only Eval secret delivery (plus deferred plaintext for deploy)."""

    ciphertext: str
    env_keys: tuple[str, ...]
    eval_run_id: str
    app_identity: str
    kms_public_key_sha256: str
    #: Plaintext pairs for post-discovery re-encrypt inside :meth:`HttpEvalPhalaDeployment.deploy`.
    #: Never logged; excluded from repr/eq so evidence dumps stay ciphertext-only.
    _deploy_secret_pairs: tuple[tuple[str, str], ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


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
    # The KMS key, measurement, and image come from the validator-signed plan.
    # Phala app_id is discovered at provision time (not a plan trust pin).
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
    # joinbase pin ``04011776…`` (tee-pin-pack / eval residual after KR), then
    # ``None`` for pins measured with no key_release_url baked into compose
    # (staging / default generator → ``0647b4d9…``). Guest still resolves the
    # real endpoint from the signed plan at runtime.
    plan_endpoint = str(plan.get("key_release_endpoint") or "").strip() or None
    key_release_candidates: list[str | None] = []
    for candidate_url in (
        plan_endpoint,
        MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER,
        None,
    ):
        if candidate_url not in key_release_candidates:
            key_release_candidates.append(candidate_url)
    compose = None
    compose_text = ""
    compose_hash = ""
    # app_identity overload:
    # - absent / empty → default compose moniker; discovery (no nonce/app_id)
    # - 40-hex → advisory Phala pin (never asserted); discovery path
    # - non-hex moniker → compose name (feeds compose_hash); may send app_id alone
    app_identity_raw = app.get("app_identity")
    if not isinstance(app_identity_raw, str) or not app_identity_raw:
        app_identity = ""
        compose_name = DEFAULT_EVAL_COMPOSE_NAME
        phala_app_nonce: int | None = None
    elif _APP_ID_HEX40_RE.fullmatch(app_identity_raw.lower()):
        app_identity = app_identity_raw.lower()
        compose_name = DEFAULT_EVAL_COMPOSE_NAME
        phala_app_nonce = None
    else:
        app_identity = app_identity_raw
        compose_name = app_identity
        phala_app_nonce = None
    name_candidates = (compose_name,)
    # Also try signed identity as compose name for moniker-only legacy pins.
    if compose_name != app_identity and app_identity:
        name_candidates = (compose_name, app_identity)
    # allowed_envs candidates (order matters — prefer current full set first):
    # 1) current EVAL_ALLOWED_ENVS (includes artifact URL/token)
    # 2) pre-artifact set — live joinbase pin daf0f209… was measured before
    #    CHALLENGE_PHALA_EVAL_ARTIFACT_{URL,TOKEN} entered DEFAULT_ALLOWED_ENVS
    #    (T8 / 2026-07-26). Searching this set is hash-determine only; never
    #    invent compose bytes. When matched, encrypted_env must not inject
    #    names absent from the measured allowed_envs list.
    allowed_envs_candidates: list[tuple[str, ...]] = [
        tuple(sorted(allowed)),
        tuple(
            sorted(
                name
                for name in allowed
                if name not in {EVAL_ARTIFACT_URL_ENV, EVAL_ARTIFACT_TOKEN_ENV}
            )
        ),
    ]
    # De-dupe while preserving order (full set may equal pre-artifact if names drop).
    seen_allowed: set[tuple[str, ...]] = set()
    unique_allowed_candidates: list[tuple[str, ...]] = []
    for cand in allowed_envs_candidates:
        if cand in seen_allowed:
            continue
        seen_allowed.add(cand)
        unique_allowed_candidates.append(cand)
    for live_path in live_registry_candidates:
        for name in name_candidates:
            for key_release_url in key_release_candidates:
                for allowed_envs in unique_allowed_candidates:
                    candidate = generate_app_compose(
                        orchestrator_image=app["image_ref"],
                        name=name,
                        key_release_url=key_release_url,
                        allowed_envs=allowed_envs,
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
        disk_size_gb=DEFAULT_EVAL_DISK_SIZE_GB,
    )


def build_eval_artifact_env_values(
    plan: EvalDeploymentPlan,
    *,
    secret: str,
    api_base_url: str,
    now: datetime | None = None,
    ttl: timedelta | None = None,
) -> dict[str, str]:
    """Mint the short-lived artifact grant and build encrypted_env VALUES.

    Returns only the two artifact delivery names. Callers merge into the full
    secrets map before :func:`encrypt_eval_secrets`. Never logs the token.
    """

    from agent_challenge.api.eval_artifact_routes import mint_eval_artifact_grant

    base = _require_https_api_base(api_base_url)
    eval_run_id = plan.eval_run_id
    agent_hash = plan.plan.get("agent_hash")
    if not isinstance(agent_hash, str) or not agent_hash:
        raise EvalDeploymentError("Eval plan is missing agent_hash for artifact grant")
    if "/" in eval_run_id or "." in eval_run_id:
        raise EvalDeploymentError("eval_run_id is invalid for artifact grant")

    now_utc = now or datetime.now(UTC)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    else:
        now_utc = now_utc.astimezone(UTC)
    grant_ttl = EVAL_ARTIFACT_GRANT_TTL if ttl is None else ttl
    if not isinstance(grant_ttl, timedelta) or grant_ttl <= timedelta(0):
        raise EvalDeploymentError("Eval artifact grant TTL must be a positive duration")
    # Cap at 6h so callers cannot mint multi-day download capabilities by mistake.
    if grant_ttl > timedelta(hours=6):
        raise EvalDeploymentError("Eval artifact grant TTL exceeds the 6h maximum")

    expires_at = now_utc + grant_ttl
    try:
        token = mint_eval_artifact_grant(
            secret=secret,
            eval_run_id=eval_run_id,
            agent_hash=agent_hash,
            expires_at=expires_at,
        )
    except ValueError as exc:
        # Never surface secret/token material — only a stable reason.
        raise EvalDeploymentError("Eval artifact grant mint failed") from exc

    url = f"{base}/eval/v1/runs/{eval_run_id}/artifact"
    return {
        EVAL_ARTIFACT_URL_ENV: url,
        EVAL_ARTIFACT_TOKEN_ENV: token,
    }


def _require_https_api_base(api_base_url: str) -> str:
    if not isinstance(api_base_url, str) or not api_base_url.strip():
        raise EvalDeploymentError("Eval artifact API base URL is required (https only)")
    base = api_base_url.strip().rstrip("/")
    if not base.startswith("https://"):
        raise EvalDeploymentError(
            "Eval artifact API base URL must be https (plaintext http is refused)"
        )
    return base


def _require_https_artifact_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise EvalDeploymentError("Eval artifact URL must be a non-empty https URL")
    cleaned = url.strip()
    if not cleaned.startswith("https://"):
        raise EvalDeploymentError("Eval artifact URL must be https (plaintext http is refused)")
    return cleaned


def encrypt_eval_secrets(
    plan: EvalDeploymentPlan,
    secrets: Mapping[str, str],
    *,
    discovered: DiscoveredPhalaAppIdentity | None = None,
) -> EncryptedEvalSecrets:
    """Encrypt Eval secrets; optionally bind to a provision-discovered Phala identity.

    When ``discovered`` is set, ciphertext is sealed to that env-encrypt pubkey and
    ``app_identity`` / KMS digest track the discovered handle. Callers that encrypt
    before provision still get a plan-key ciphertext for offline checks, plus
    deferred plaintext pairs so :meth:`HttpEvalPhalaDeployment.deploy` can
    re-seal after discovery.
    """

    # Reject gateway secrets on the caller map before compose scoping (VAL-ACAT-013).
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
    # Measured compose allowed_envs is the Phala injection allowlist. Secrets
    # outside that list cannot be delivered (and would change compose_hash if
    # forced into the measured document). Scope required names to the intersection.
    compose_allowed = {
        str(name)
        for name in (plan.compose.get("allowed_envs") or ())
        if isinstance(name, str) and name and "=" not in name
    }
    if not compose_allowed:
        compose_allowed = set(EVAL_ALLOWED_ENVS)
    if not compose_allowed <= set(EVAL_ALLOWED_ENVS):
        raise EvalDeploymentError(
            "Eval compose allowed_envs contains names outside EVAL_ALLOWED_ENVS"
        )
    # Drop secrets the measured compose cannot accept (e.g. artifact grant on
    # pre-artifact pins such as daf0f209…). Never invent alternate delivery.
    scoped_secrets = {
        name: value
        for name, value in secrets.items()
        if name in compose_allowed
    }
    required = frozenset(
        name for name in EVAL_REQUIRED_SECRET_ENVS if name in compose_allowed
    )
    # Always require run token + attestation plan + cost limit when present in
    # the product required set and the compose allowlist.
    if not set(scoped_secrets) <= set(EVAL_ALLOWED_ENVS) or not required <= set(scoped_secrets):
        raise EvalDeploymentError(
            "Eval encrypted_env names must be scoped allowed names with the required run "
            "and attestation plan capabilities (Base LLM gateway secrets are not allowed)"
        )
    # VAL-ACLOCK-009: free CHALLENGE_PHALA_KEY_RELEASE_URL is not a miner trust
    # root. Prefer plan key_release_endpoint + KEY_RELEASE_RA_TLS_HOST/PORT.
    # Name may remain in allowed_envs for measure-time pin hash stability, but
    # any encrypted_env value must be the same RA-TLS authority as the signed
    # plan (free HTTP(S) URLs always refuse).
    if KEY_RELEASE_URL_ENV in scoped_secrets:
        free_url = scoped_secrets[KEY_RELEASE_URL_ENV]
        plan_endpoint = str(plan.plan.get("key_release_endpoint") or "").strip()
        plan_auth = parse_key_release_authority(plan_endpoint)
        free_auth = parse_key_release_authority(free_url if isinstance(free_url, str) else "")
        if plan_auth is None or free_auth is None or free_auth != plan_auth:
            raise EvalDeploymentError(
                "Eval encrypted_env CHALLENGE_PHALA_KEY_RELEASE_URL is not miner-"
                "authoritative; value must match plan key_release_endpoint RA-TLS "
                "authority (prefer KEY_RELEASE_RA_TLS_HOST/PORT). Free HTTP(S) KR "
                "URLs are refused."
            )
    # Artifact delivery: only when the measured compose lists the names.
    # Pre-artifact pins (daf0f209…) omit them — guest cannot receive the grant.
    has_artifact_url = EVAL_ARTIFACT_URL_ENV in scoped_secrets
    has_artifact_token = EVAL_ARTIFACT_TOKEN_ENV in scoped_secrets
    if has_artifact_url or has_artifact_token:
        if not has_artifact_url or not has_artifact_token:
            raise EvalDeploymentError(
                "Eval artifact grant requires both URL and token when either is present"
            )
        artifact_url = _require_https_artifact_url(scoped_secrets[EVAL_ARTIFACT_URL_ENV])
        artifact_token = scoped_secrets[EVAL_ARTIFACT_TOKEN_ENV]
        if not isinstance(artifact_token, str) or not artifact_token.strip():
            raise EvalDeploymentError("Eval artifact grant token must be a non-empty string")
        expected_suffix = f"/eval/v1/runs/{plan.eval_run_id}/artifact"
        if not artifact_url.endswith(expected_suffix):
            raise EvalDeploymentError("Eval artifact URL is not bound to this eval_run_id")
        scoped_secrets = dict(scoped_secrets)
        scoped_secrets[EVAL_ARTIFACT_URL_ENV] = artifact_url

    env_keys = tuple(name for name in EVAL_ALLOWED_ENVS if name in scoped_secrets)
    values = {name: scoped_secrets[name] for name in env_keys}
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise EvalDeploymentError("Eval encrypted_env values must be non-empty strings")
    if values["EVAL_RUN_TOKEN"] != plan.eval_run_token:
        raise EvalDeploymentError("Eval run token does not match signed prepare response")
    if discovered is not None:
        encrypt_pubkey = discovered.app_env_encrypt_pubkey
        bound_app_identity = discovered.app_id
        bound_kms_sha = discovered.kms_public_key_sha256
    else:
        encrypt_pubkey = plan.kms_public_key_hex
        bound_app_identity = plan.app_identity
        bound_kms_sha = plan.kms_public_key_sha256
    try:
        ciphertext = encrypt_env_vars_sync(
            [EnvVar(key=name, value=values[name]) for name in env_keys],
            encrypt_pubkey,
        )
    except Exception as exc:
        raise EvalDeploymentError("Eval encrypted_env encryption failed") from exc
    if not ciphertext:
        raise EvalDeploymentError("Eval encrypted_env ciphertext is empty")
    return EncryptedEvalSecrets(
        ciphertext=ciphertext,
        env_keys=env_keys,
        eval_run_id=plan.eval_run_id,
        app_identity=bound_app_identity,
        kms_public_key_sha256=bound_kms_sha,
        _deploy_secret_pairs=tuple((name, values[name]) for name in env_keys),
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
        """Provision (names only) → discover app_id → encrypt → create.

        Order is mandatory: env ciphertext must be sealed to the *discovered*
        env-encrypt pubkey. ``plan.app_identity`` is never asserted against Phala.
        """

        if encrypted.eval_run_id != plan.eval_run_id or not set(encrypted.env_keys) <= set(
            EVAL_ALLOWED_ENVS
        ):
            raise EvalDeploymentError("Eval encrypted_env is not bound to this run")
        if not encrypted.env_keys:
            raise EvalDeploymentError("Eval encrypted_env is not bound to this run")

        try:
            env_keys = env_keys_from_allowed(
                EVAL_ALLOWED_ENVS,
                selected=set(encrypted.env_keys),
            )
        except ProvisionIdentityError as exc:
            raise EvalDeploymentError(str(exc)) from exc

        # Provision with env *names* only — no ciphertext, no assignment app_id pin.
        # Phala contract: discovery sends neither nonce nor app_id (live 200);
        # moniker may send app_id alone. Never nonce-without-app_id (live 422).
        provision_request: dict[str, Any] = {
            "name": plan.compose_name,
            "instance_type": plan.instance_type,
            "region": plan.region,
            "compose_file": plan.compose,
            "env_keys": env_keys,
            "image": plan.os_image,
            # Sibling of compose_file — never mutate plan.compose.
            "disk_size": validate_disk_size(plan.disk_size_gb),
        }
        if plan.app_identity and not _APP_ID_HEX40_RE.fullmatch(plan.app_identity.lower()):
            provision_request["app_id"] = plan.app_identity
        # plan.phala_app_nonce is intentionally ignored (cannot force illegal shape).
        provision = self._api.post("/cvms/provision", provision_request)
        try:
            assert_provision_trust_anchors(
                plan_compose_hash=plan.compose_hash,
                plan_measurement=plan.measurement,
                provision=provision,
            )
            identity = parse_discovered_identity(provision)
            optional_verify_env_encrypt_pubkey(identity)
        except ProvisionIdentityError as exc:
            raise EvalDeploymentError(str(exc)) from exc

        if encrypted._deploy_secret_pairs is not None:
            # Re-seal to the discovered pubkey after trust anchors pass.
            encrypted = encrypt_eval_secrets(
                plan,
                dict(encrypted._deploy_secret_pairs),
                discovered=identity,
            )
        elif (
            encrypted.app_identity != identity.app_id
            or encrypted.kms_public_key_sha256 != identity.kms_public_key_sha256
            or not encrypted.ciphertext
        ):
            raise EvalDeploymentError(
                "Eval encrypted_env is not bound to discovered Phala app identity"
            )
        if (
            encrypted.app_identity != identity.app_id
            or encrypted.kms_public_key_sha256 != identity.kms_public_key_sha256
            or not encrypted.ciphertext
        ):
            raise EvalDeploymentError(
                "Eval encrypted_env is not bound to discovered Phala app identity"
            )

        created = self._api.post(
            "/cvms",
            {
                "app_id": identity.app_id,
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
                    cvm_id = resolve_cvm_id_from_list(listing, app_id=identity.app_id)
        if not isinstance(cvm_id, str) or not cvm_id:
            raise EvalDeploymentError("Phala create response does not identify the Eval CVM")
        try:
            return {
                "eval_run_id": plan.eval_run_id,
                "cvm_id": cvm_id,
                "app_identity": identity.app_id,
                "image_ref": plan.image_ref,
                "compose_hash": plan.compose_hash,
                "kms_public_key_sha256": identity.kms_public_key_sha256,
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
    "DEFAULT_OS_IMAGE",
    "DEFAULT_REGION",
    "EVAL_ALLOWED_ENVS",
    "EVAL_ARTIFACT_GRANT_TTL",
    "EVAL_ARTIFACT_TOKEN_ENV",
    "EVAL_ARTIFACT_URL_ENV",
    "EVAL_REQUIRED_SECRET_ENVS",
    "MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER",
    "EncryptedEvalSecrets",
    "EvalDeploymentError",
    "EvalDeploymentPlan",
    "EvalPhalaDeployment",
    "HttpEvalPhalaDeployment",
    "build_eval_artifact_env_values",
    "build_eval_deployment_plan",
    "encrypt_eval_secrets",
]
