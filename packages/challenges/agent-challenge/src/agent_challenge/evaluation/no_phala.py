"""Host-trust unattested execution mode (product path after T40 Phala delete).

When ``NO_PHALA`` / ``CHALLENGE_NO_PHALA`` is active the master validator runs
benchmark/eval jobs on the host via the existing own_runner local Docker path
instead of provisioning Phala CVMs. There is **no TEE, no TDX quote, and no
attestation**. Results are explicitly marked unattested and cannot be forged
into an attested envelope from this path.

Phala TEE is removed from the product path. This module is the honesty
branch for host-trust runs. Never claim TEE / tamper-proof / independent
verification from this path.

Precedence for the env switch (see :func:`resolve_no_phala_from_environ`):

1. ``CHALLENGE_NO_PHALA`` if set (challenge-prefix convention)
2. else plain ``NO_PHALA`` (operator convenience on the master host)
3. else off

Never infer the mode from a missing Phala API key or a failed Phala call.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Final

logger = logging.getLogger(__name__)

#: Challenge-prefixed env (pydantic-settings field ``no_phala``).
CHALLENGE_NO_PHALA_ENV: Final = "CHALLENGE_NO_PHALA"
#: Operator-facing plain env accepted on the master host.
NO_PHALA_ENV: Final = "NO_PHALA"
#: Canonical unattested-execution env (T40 product name).
CHALLENGE_UNATTESTED_EXECUTION_ENV: Final = "CHALLENGE_UNATTESTED_EXECUTION"

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_FALSY: Final = frozenset({"0", "false", "no", "off", ""})

#: Wire / stored execution mode label for host-local unattested runs.
EXECUTION_MODE_NO_PHALA_HOST: Final = "no_phala_host"
#: Explicit attestation status — never "attested" from this module.
ATTESTATION_STATUS_UNATTESTED: Final = "unattested"

#: Result envelope keys written by this mode.
RESULT_KEY_ATTESTED: Final = "attested"
RESULT_KEY_ATTESTATION_STATUS: Final = "attestation_status"
RESULT_KEY_EXECUTION_MODE: Final = "execution_mode"
RESULT_KEY_GUEST_ARTIFACT_PROOF: Final = "guest_artifact_proof"

#: Keys that would make a result look Phala-attested — stripped on mark.
_ATTESTED_LOOKING_KEYS: Final = frozenset(
    {
        "execution_proof",
        "attestation_binding",
        "tdx_quote",
        "phala_attestation",
    }
)

CONTRADICTION_MESSAGE: Final = (
    "NO_PHALA mode cannot be combined with attestation flags: "
    "CHALLENGE_PHALA_ATTESTATION_ENABLED and CHALLENGE_ATTESTED_REVIEW_ENABLED "
    "must both be off when NO_PHALA/CHALLENGE_NO_PHALA is on. "
    "Attested TEE path and host-local unattested path are mutually exclusive."
)

STARTUP_BANNER: Final = (
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    "!!  NO_PHALA MODE ACTIVE — host-local unattested execution              !!\n"
    "!!  No TEE / TDX quote / DCAP / compose_hash attestation is performed.  !!\n"
    "!!  Results are marked unattested and MUST NOT be treated as attested.  !!\n"
    "!!  Disable via NO_PHALA=false (or unset) and restart.                  !!\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
)


class NoPhalaModeError(RuntimeError):
    """Raised when NO_PHALA mode forbids an operation (e.g. Phala client use)."""


class ArtifactProvenanceError(ValueError):
    """Raised when expected/download/executed artifact hashes disagree."""


def _parse_bool_env(raw: str | None) -> bool | None:
    """Return True/False if ``raw`` is a recognized boolean token, else None."""

    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return None


def resolve_no_phala_from_environ(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve unattested/NO_PHALA from env with documented precedence.

    1. ``CHALLENGE_UNATTESTED_EXECUTION`` if present
    2. else ``CHALLENGE_NO_PHALA`` if present
    3. else ``NO_PHALA`` if present
    4. else ``False`` (default OFF — never inferred from missing keys)
    """

    env = os.environ if environ is None else environ
    if CHALLENGE_UNATTESTED_EXECUTION_ENV in env:
        parsed = _parse_bool_env(env.get(CHALLENGE_UNATTESTED_EXECUTION_ENV))
        return bool(parsed)
    if CHALLENGE_NO_PHALA_ENV in env:
        parsed = _parse_bool_env(env.get(CHALLENGE_NO_PHALA_ENV))
        return bool(parsed)
    if NO_PHALA_ENV in env:
        parsed = _parse_bool_env(env.get(NO_PHALA_ENV))
        return bool(parsed)
    return False


def resolve_unattested_execution_from_environ(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Alias of :func:`resolve_no_phala_from_environ` (T40 product name)."""

    return resolve_no_phala_from_environ(environ)


def is_unattested_execution_enabled(
    *,
    settings_flag: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Alias of :func:`is_no_phala_enabled` (T40 product name)."""

    return is_no_phala_enabled(settings_flag=settings_flag, environ=environ)


def is_no_phala_enabled(
    *,
    settings_flag: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether NO_PHALA host mode is active.

    Prefer the settings field when provided (already resolved at startup).
    Fall back to env resolution for in-process guest/CLI contexts that do not
    load :class:`~agent_challenge.sdk.config.ChallengeSettings`.
    """

    if settings_flag is not None:
        return bool(settings_flag)
    return resolve_no_phala_from_environ(environ)


def assert_no_phala_compatible(
    *,
    no_phala: bool,
    phala_attestation_enabled: bool,
    attested_review_enabled: bool,
) -> None:
    """Fail closed when NO_PHALA collides with either attestation flag."""

    if no_phala and (phala_attestation_enabled or attested_review_enabled):
        raise ValueError(CONTRADICTION_MESSAGE)


def log_no_phala_startup_banner() -> None:
    """Emit a loud, unmistakable startup warning (call once from app lifespan)."""

    for line in STARTUP_BANNER.splitlines():
        logger.critical(line)


def refuse_phala_client(operation: str = "Phala Cloud API") -> None:
    """Raise if NO_PHALA is active — provisioning seam must never be invoked."""

    if resolve_no_phala_from_environ():
        raise NoPhalaModeError(
            f"{operation} is forbidden while NO_PHALA mode is active "
            f"(set {NO_PHALA_ENV}=false / unset {CHALLENGE_NO_PHALA_ENV} to use Phala)"
        )


def build_guest_artifact_proof(
    *,
    expected_hash: str,
    download_hash: str,
    executed_hash: str,
) -> dict[str, Any]:
    """Build the provenance triple ``expected == download == executed``.

    Cheap and valuable even without a TEE. Raises
    :class:`ArtifactProvenanceError` on mismatch or empty hashes so a bad
    artifact cannot be reported as matching.
    """

    expected = (expected_hash or "").strip().lower()
    download = (download_hash or "").strip().lower()
    executed = (executed_hash or "").strip().lower()
    if not expected or len(expected) != 64:
        raise ArtifactProvenanceError("expected_hash must be a 64-char hex digest")
    if not all(c in "0123456789abcdef" for c in expected + download + executed):
        raise ArtifactProvenanceError("artifact hashes must be lowercase hex")
    if not (expected == download == executed):
        raise ArtifactProvenanceError(
            "artifact provenance mismatch: "
            f"expected={expected} download={download} executed={executed}"
        )
    return {
        "expected_hash": expected,
        "download_hash": download,
        "executed_hash": executed,
        "match": True,
    }


def mark_result_unattested(
    payload: Mapping[str, Any],
    *,
    artifact_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a result envelope explicitly marked unattested (unforgeable).

    Always sets ``attested=False`` and ``attestation_status=unattested``.
    Strips any Phala-looking proof blocks. Callers cannot pass
    ``attested=True`` through this function — the boolean is hard-coded.
    """

    out = dict(payload)
    for key in _ATTESTED_LOOKING_KEYS:
        out.pop(key, None)
    # Hard-coded False — never accept a caller-supplied True.
    out[RESULT_KEY_ATTESTED] = False
    out[RESULT_KEY_ATTESTATION_STATUS] = ATTESTATION_STATUS_UNATTESTED
    out[RESULT_KEY_EXECUTION_MODE] = EXECUTION_MODE_NO_PHALA_HOST
    if artifact_proof is not None:
        # Re-validate so a forged match:false / mismatched triple cannot slip in.
        proof = build_guest_artifact_proof(
            expected_hash=str(artifact_proof.get("expected_hash") or ""),
            download_hash=str(artifact_proof.get("download_hash") or ""),
            executed_hash=str(artifact_proof.get("executed_hash") or ""),
        )
        out[RESULT_KEY_GUEST_ARTIFACT_PROOF] = proof
    return out


def assert_envelope_not_attested(payload: Mapping[str, Any]) -> None:
    """Fail closed if an envelope claims attestation while in NO_PHALA mode."""

    if payload.get(RESULT_KEY_ATTESTED) is True:
        raise ValueError("NO_PHALA result envelope must not claim attested=true")
    if payload.get(RESULT_KEY_ATTESTATION_STATUS) == "attested":
        raise ValueError("NO_PHALA result envelope must not claim attestation_status=attested")
    if payload.get("execution_proof") is not None:
        raise ValueError("NO_PHALA result envelope must not carry execution_proof")
    if payload.get(RESULT_KEY_EXECUTION_MODE) != EXECUTION_MODE_NO_PHALA_HOST:
        raise ValueError(
            f"NO_PHALA result envelope must set execution_mode={EXECUTION_MODE_NO_PHALA_HOST!r}"
        )


def health_fields(*, no_phala: bool) -> dict[str, Any]:
    """Fields merged into ``/health`` so operators can see the live mode."""

    return {
        "no_phala": bool(no_phala),
        "attestation_mode": (EXECUTION_MODE_NO_PHALA_HOST if no_phala else "standard"),
    }


__all__ = [
    "ATTESTATION_STATUS_UNATTESTED",
    "CHALLENGE_NO_PHALA_ENV",
    "CONTRADICTION_MESSAGE",
    "EXECUTION_MODE_NO_PHALA_HOST",
    "NO_PHALA_ENV",
    "RESULT_KEY_ATTESTATION_STATUS",
    "RESULT_KEY_ATTESTED",
    "RESULT_KEY_EXECUTION_MODE",
    "RESULT_KEY_GUEST_ARTIFACT_PROOF",
    "STARTUP_BANNER",
    "ArtifactProvenanceError",
    "NoPhalaModeError",
    "assert_envelope_not_attested",
    "assert_no_phala_compatible",
    "build_guest_artifact_proof",
    "health_fields",
    "is_no_phala_enabled",
    "is_unattested_execution_enabled",
    "resolve_unattested_execution_from_environ",
    "log_no_phala_startup_banner",
    "mark_result_unattested",
    "refuse_phala_client",
    "resolve_no_phala_from_environ",
]
