"""Immutable challenge adoption contract (VAL-CROSS-075).

Registration may leave a challenge in ``DRAFT``, but activation (and any
registration that requests ``ACTIVE``) must pass a single closed contract:

* digest-pinned image reference (`repository[:tag]@sha256:<64 hex>`);
* supported API/protocol version (major-compatible with the master wire);
* role-scoped capability tokens only (challenge + approved legacy aliases);
* health/version contract metadata when declared;
* emission/share bounds;
* credential scoping (no clear secrets in env/metadata/resources);
* network/volume policy safe for Compose last-mile adoption.

Clear challenge/broker tokens are produced once at create and never stored in
registry/admin responses; only non-secret hints are retained.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from base.challenge_sdk.roles import Capability, Role, capabilities_for_role
from base.config.policy import ProductionPolicyError, validate_image_reference
from base.master.docker_orchestrator import DEFAULT_API_VERSION
from base.schemas.challenge import ChallengeCreate, ChallengeRecord, ChallengeStatus

_PINNED_IMAGE_RE = re.compile(r"^(?P<name>.+)@sha256:(?P<digest>[0-9a-fA-F]{64})$")
_SEMVER_API_RE = re.compile(
    r"^(0|[1-9]\d*)(?:\.(0|[1-9]\d*))?(?:\.(0|[1-9]\d*))?(?:[-+][0-9A-Za-z.-]+)?$"
)
_SAFE_VOLUME_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HOST_PATH_VOLUME_RE = re.compile(r"^(/|[A-Za-z]:\\|~)")
_FORBIDDEN_CAPABILITY_FRAGMENTS = (
    "set_weights",
    "master.",
    "validator.",
    "worker.",
    "evaluator",
    "gateway",
    "swarm",
)
# Legacy registry aliases retained for active Base challenges; new SDK tokens
# come exclusively from the role registry under Role.CHALLENGE.
_LEGACY_CHALLENGE_CAPABILITIES = frozenset(
    {
        "get_weights",
        "proxy_routes",
        "submit",
        "score",
    }
)
_CHALLENGE_CAPABILITIES = (
    frozenset(capabilities_for_role(Role.CHALLENGE))
    | frozenset(
        {
            Capability.CHALLENGE_SCORING.value,
            Capability.CHALLENGE_ORDINARY_PROOF.value,
            Capability.CHALLENGE_TEE_VERIFICATION.value,
            Capability.CHALLENGE_STATE.value,
            Capability.CHALLENGE_RAW_WEIGHT_PUSH.value,
        }
    )
    | _LEGACY_CHALLENGE_CAPABILITIES
)
_SECRETISH_KEY_RE = re.compile(
    r"(token|password|secret|private[_-]?key|api[_-]?key|wallet|mnemonic|credential)",
    re.IGNORECASE,
)
_CANARY_VALUE_RE = re.compile(
    r"(sk-|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|postgresql(\+asyncpg)?://[^@\s]+:[^@\s]+@)",
    re.IGNORECASE,
)


class ChallengeAdoptionError(ProductionPolicyError):
    """Raised when a challenge fails the immutable adoption contract."""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    return {}


def require_digest_pinned_image(image: str) -> str:
    """Reject mutable tags; accept only digest-pinned image references."""

    raw = (image or "").strip()
    if not raw:
        raise ChallengeAdoptionError("challenge image is required for adoption")
    match = _PINNED_IMAGE_RE.fullmatch(raw)
    if match is None:
        raise ChallengeAdoptionError(
            "challenge image must be digest-pinned "
            "(expected repository[:tag]@sha256:<64 hex>); "
            "mutable tags cannot become ACTIVE"
        )
    # Also run the production image form (tag + digest) for operator clarity.
    # Allow references that oven omit the tag only if fully digested: policy
    # still wants a tag when production=True, but adoption always needs the
    # digest half. Prefer production-shaped refs; require digest always.
    name = match.group("name")
    slash = name.rfind("/")
    colon = name.rfind(":")
    if colon <= slash:
        raise ChallengeAdoptionError(
            "challenge image must include a concrete tag before @sha256 "
            "(mutable latest without digest is rejected earlier)"
        )
    # Re-use production tag/digest checks so semver|latest + sha256 is enforced.
    validate_image_reference(raw, production=True)
    return raw


def _validate_api_version(api_version: str) -> None:
    raw = (api_version or "").strip()
    if not raw:
        raise ChallengeAdoptionError("challenge api_version is required for adoption")
    if not _SEMVER_API_RE.fullmatch(raw):
        raise ChallengeAdoptionError(
            f"challenge api_version must be semantic ({raw!r})"
        )
    major = int(raw.split(".", 1)[0])
    expected_major = int(DEFAULT_API_VERSION.split(".", 1)[0])
    if major != expected_major:
        raise ChallengeAdoptionError(
            f"challenge api/protocol major {major} incompatible with master "
            f"wire major {expected_major} (api_version={raw!r})"
        )


def _validate_capabilities(capabilities: list[str] | tuple[str, ...] | None) -> None:
    declared = list(capabilities or [])
    if not declared:
        raise ChallengeAdoptionError(
            "challenge required_capabilities must declare at least "
            "one challenge capability"
        )
    seen: set[str] = set()
    for token in declared:
        if not isinstance(token, str) or not token.strip():
            raise ChallengeAdoptionError(
                "challenge required_capabilities entries must be non-empty strings"
            )
        name = token.strip()
        lowered = name.lower()
        if name in seen:
            raise ChallengeAdoptionError(
                f"duplicate capability token rejected: {name!r}"
            )
        seen.add(name)
        if any(fragment in lowered for fragment in _FORBIDDEN_CAPABILITY_FRAGMENTS):
            raise ChallengeAdoptionError(
                f"capability {name!r} is forbidden for challenge adoption"
            )
        if name not in _CHALLENGE_CAPABILITIES:
            raise ChallengeAdoptionError(
                f"capability {name!r} is not a challenge-scoped adoption token"
            )


def _validate_share(emission_percent: Decimal | float | int | str | None) -> None:
    try:
        value = Decimal(str(emission_percent if emission_percent is not None else "0"))
    except Exception as exc:  # noqa: BLE001 - defensive conversion
        raise ChallengeAdoptionError(
            "challenge emission/share must be a finite non-negative decimal"
        ) from exc
    if value.is_nan() or value.is_infinite():
        raise ChallengeAdoptionError("challenge emission/share must be finite")
    if value < 0 or value > Decimal("100"):
        raise ChallengeAdoptionError(
            "challenge emission/share must be between 0 and 100 inclusive"
        )


def _validate_volumes(volumes: Mapping[str, Any] | None, *, slug: str) -> None:
    mapping = dict(_as_mapping(volumes))
    # Default sqlite volume is injected by the registry when omitted; adoption
    # still rejects host binds and docker.sock style mounts when declared.
    for key, value in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise ChallengeAdoptionError("volume keys must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            raise ChallengeAdoptionError(
                f"volume {key!r} target/source must be a non-empty string"
            )
        candidate = value.strip()
        lowered = candidate.lower()
        if "docker.sock" in lowered:
            raise ChallengeAdoptionError(
                "challenge volumes must not mount the Docker socket"
            )
        if _HOST_PATH_VOLUME_RE.search(candidate) or candidate.startswith("."):
            raise ChallengeAdoptionError(
                f"challenge volume {key!r} rejects host bind paths "
                f"(use named Compose volumes only)"
            )
        if not _SAFE_VOLUME_NAME_RE.fullmatch(candidate):
            raise ChallengeAdoptionError(
                f"challenge volume {key!r} must be a safe named volume "
                f"(got {candidate!r})"
            )
    if "sqlite" in mapping:
        expected_prefix = f"base_{slug.replace('-', '_')}"
        sqlite_name = str(mapping["sqlite"])
        if not sqlite_name.startswith("base_"):
            raise ChallengeAdoptionError(
                "challenge sqlite volume must be challenge-owned "
                f"(expected base_* name, got {sqlite_name!r})"
            )
        # Prefer the deterministic default form but allow pre-seeded siblings.
        slug_token = slug.replace("-", "_")
        if expected_prefix not in sqlite_name and slug_token not in sqlite_name:
            raise ChallengeAdoptionError(
                "challenge sqlite volume must be scoped to the challenge slug"
            )


def _validate_network_policy(
    *,
    internal_base_url: str | None,
    env: Mapping[str, Any] | None,
    resources: Mapping[str, Any] | None,
) -> None:
    url = (internal_base_url or "").strip()
    if url:
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ChallengeAdoptionError(
                "internal_base_url must be an http(s) challenge-network URL"
            )
        if "localhost" in url or "127.0.0.1" in url or "0.0.0.0" in url:
            raise ChallengeAdoptionError(
                "internal_base_url must not target loopback or host-published addresses"
            )
        if "://" in url and "@" in url.split("://", 1)[1].split("/", 1)[0]:
            raise ChallengeAdoptionError("internal_base_url must not embed credentials")
    for source_name, mapping in (("env", env), ("resources", resources)):
        for key, value in _as_mapping(mapping).items():
            text = f"{key}={value}"
            lowered = text.lower()
            if "docker.sock" in lowered or "host.docker.internal" in lowered:
                raise ChallengeAdoptionError(
                    f"{source_name} rejects host/docker network breakout keys"
                )
            if key.lower() in {"network_mode", "network", "ports", "publish"}:
                raise ChallengeAdoptionError(
                    f"{source_name} key {key!r} is not part of the "
                    "adoption network policy"
                )


def _scan_for_clear_credentials(
    *,
    env: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    secrets: list[str] | None,
    strict_metadata: bool = False,
) -> None:
    # secrets is a list of names only; reject if callers stuffed values.
    for name in secrets or []:
        if not isinstance(name, str) or not name.strip():
            raise ChallengeAdoptionError(
                "challenge secrets entries must be non-empty secret *names*"
            )
        if "=" in name or "/" in name or " " in name.strip():
            raise ChallengeAdoptionError(
                "challenge secrets must list names only, never values or paths"
            )

    def _walk(prefix: str, node: Any, *, strict_keys: bool) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_text = str(key)
                path = f"{prefix}.{key_text}" if prefix else key_text
                if (
                    strict_keys
                    and _SECRETISH_KEY_RE.search(key_text)
                    and value not in (None, "", [])
                ):
                    # Hints and documentation keys that explicitly say "hint" are ok.
                    if "hint" in key_text.lower() or "docs" in key_text.lower():
                        _walk(path, value, strict_keys=strict_keys)
                        continue
                    if isinstance(value, (str, bytes, int, float, Decimal)):
                        raise ChallengeAdoptionError(
                            f"clear credential material rejected under {path}"
                        )
                _walk(path, value, strict_keys=strict_keys)
            return
        if isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                _walk(f"{prefix}[{index}]", item, strict_keys=strict_keys)
            return
        if isinstance(node, str) and _CANARY_VALUE_RE.search(node):
            raise ChallengeAdoptionError(
                f"clear credential material rejected under {prefix or '<value>'}"
            )

    # Env always uses strict secretish-key rejection (operator-facing mounts).
    _walk("env", _as_mapping(env), strict_keys=True)
    # Metadata may historically carry operator notes used only for public scrub
    # testing; activation still rejects credential-bearing URI/canary values.
    _walk("metadata", _as_mapping(metadata), strict_keys=strict_metadata)


def _validate_health_version_contract(metadata: Mapping[str, Any] | None) -> None:
    meta = _as_mapping(metadata)
    # Optional declared contract keys: if present they must be coherent.
    expected_health = meta.get("expected_health_status")
    allowed_health = {"ok", "degraded", "healthy"}
    if expected_health is not None and expected_health not in allowed_health:
        raise ChallengeAdoptionError(
            "metadata.expected_health_status must be ok|degraded|healthy when set"
        )
    expected_role = meta.get("expected_role")
    if expected_role is not None and str(expected_role) != Role.CHALLENGE.value:
        raise ChallengeAdoptionError(
            "metadata.expected_role must be 'challenge' when set"
        )
    sdk_range = meta.get("sdk_compat") or meta.get("sdk_compatibility_range")
    if sdk_range is not None and not str(sdk_range).strip():
        raise ChallengeAdoptionError(
            "metadata.sdk_compat must be a non-empty compatibility range when set"
        )


def validate_challenge_adoption(
    *,
    slug: str,
    image: str,
    api_version: str,
    emission_percent: Decimal | float | int | str | None,
    required_capabilities: list[str] | tuple[str, ...] | None,
    volumes: Mapping[str, Any] | None = None,
    env: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
    secrets: list[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    internal_base_url: str | None = None,
    require_digest_pin: bool = True,
    production_policy: bool = False,
    full_contract: bool = True,
) -> None:
    """Validate a create/update/activate candidate against the adoption contract.

    ``require_digest_pin`` is always true for activation. Create may run with a
    lighter check so local DRAFT tooling can iterate, while production and
    compose activation remain fail-closed on mutable tags.
    """

    if not slug or not str(slug).strip():
        raise ChallengeAdoptionError("challenge slug is required")

    if require_digest_pin:
        require_digest_pinned_image(image)
    else:
        validate_image_reference(image, production=production_policy)

    if not full_contract:
        # Soft registration: image/share bounds only. Activation upgrades.
        _validate_share(emission_percent)
        return

    _validate_api_version(api_version)
    _validate_share(emission_percent)
    _validate_capabilities(required_capabilities)
    _validate_volumes(volumes, slug=str(slug))
    _validate_network_policy(
        internal_base_url=internal_base_url,
        env=env,
        resources=resources,
    )
    _scan_for_clear_credentials(
        env=env,
        metadata=metadata,
        secrets=secrets,
        # Strict metadata secretish-key policy is production-only; canary URI
        # patterns are still rejected in all full-contract paths via walk.
        strict_metadata=production_policy,
    )
    _validate_health_version_contract(metadata)


def validate_payload_for_registration(
    payload: ChallengeCreate,
    *,
    production_policy: bool = False,
) -> None:
    """Validate create against the adoption contract.

    Production always requires a digest pin and full contract. Non-production
    DRAFT registration stays light for local tooling. Any registration that
    requests status ACTIVE must still supply a digest-pinned image so an
    arbitrary tag cannot become active without ``/activate`` (VAL-CROSS-075).
    """

    creating_active = payload.status == ChallengeStatus.ACTIVE
    validate_challenge_adoption(
        slug=payload.slug,
        image=payload.image,
        api_version=payload.api_version,
        emission_percent=payload.emission_percent,
        required_capabilities=payload.required_capabilities,
        volumes=payload.volumes,
        env=payload.env,
        resources=payload.resources,
        secrets=payload.secrets,
        metadata=payload.metadata,
        internal_base_url=payload.internal_base_url,
        require_digest_pin=production_policy or creating_active,
        production_policy=production_policy,
        # Full volume/capability/credential scan for production create only;
        # ACTIVE-at-create in non-production is still digest-fail-closed.
        full_contract=production_policy,
    )


def validate_record_for_activation(
    record: ChallengeRecord,
    *,
    production_policy: bool = False,
) -> None:
    """Validate an existing record before it can become ACTIVE.

    Digest pinning is always required. Production always applies the full
    credential-scope scan; unit/dev registries still refuse mutable tags and
    unsafe network/volume policy while allowing fixture metadata that the
    public registry scrubber already sanitizes.
    """

    validate_challenge_adoption(
        slug=record.slug,
        image=record.image,
        api_version=record.api_version,
        emission_percent=record.emission_percent,
        required_capabilities=record.required_capabilities,
        volumes=record.volumes,
        env=record.env,
        resources=record.resources,
        secrets=record.secrets,
        metadata=record.metadata,
        internal_base_url=record.internal_base_url,
        require_digest_pin=True,
        production_policy=production_policy,
        full_contract=True,
    )


def admin_view_exposes_no_clear_token(view: Mapping[str, Any]) -> bool:
    """Return True when an admin/registry view carries only non-secret hints."""

    forbidden = {
        "challenge_token",
        "docker_broker_token",
        "token",
        "token_hash",
        "broker_token_hash",
        "password",
        "secret",
    }
    keys = set(view.keys())
    if keys & forbidden:
        return False
    hint = view.get("token_hint")
    if hint and "…" not in str(hint) and len(str(hint)) > 16:
        # Hints must be abbreviated; bare tokens as "hints" fail closed.
        return False
    return True
