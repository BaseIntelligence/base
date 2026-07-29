"""Unattested-execution flag + unforgeable mark for prism (T19/T20/T21).

Prism does **not** depend on ``agent_challenge`` (separate workspace package;
importing it would couple challenge packages and risk circular install graphs).
This module is a **thin duplicate** of the env precedence and unforgeable mark
in ``agent_challenge.evaluation.no_phala``:

1. ``CHALLENGE_UNATTESTED_EXECUTION`` if present (canonical)
2. else ``CHALLENGE_NO_PHALA`` if present (deprecated alias)
3. else ``NO_PHALA`` if present (operator convenience / deprecated alias)
4. else ``False`` — never inferred from missing keys

Keep this file in lockstep with T19/T21. Do not invent a second flag name.
Default remains **off** (fail-closed constation gate when bundle missing).

T21: :func:`mark_result_unattested` always forces ``attested=False`` /
``attestation_status=unattested`` / ``execution_mode=no_phala_host``. Miner-
supplied ``attested:true`` cannot survive. Unlike agent-challenge, prism keeps
worker ``execution_proof`` (sr25519 tier-0 proof) — only TEE-looking claim keys
are stripped.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Final

#: Canonical challenge-prefixed env for unattested host execution.
CHALLENGE_UNATTESTED_EXECUTION_ENV: Final = "CHALLENGE_UNATTESTED_EXECUTION"
#: Deprecated alias for :data:`CHALLENGE_UNATTESTED_EXECUTION_ENV` (same flag).
CHALLENGE_NO_PHALA_ENV: Final = "CHALLENGE_NO_PHALA"
#: Operator-facing plain env (also deprecated alias).
NO_PHALA_ENV: Final = "NO_PHALA"

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_FALSY: Final = frozenset({"0", "false", "no", "off", ""})

#: Explicit attestation status — never "attested" / "verified" from this module.
ATTESTATION_STATUS_UNATTESTED: Final = "unattested"
#: Wire / stored execution mode label for host-local unattested runs (T21 parity).
EXECUTION_MODE_NO_PHALA_HOST: Final = "no_phala_host"

RESULT_KEY_ATTESTED: Final = "attested"
RESULT_KEY_ATTESTATION_STATUS: Final = "attestation_status"
RESULT_KEY_EXECUTION_MODE: Final = "execution_mode"

#: TEE / Phala-looking claim keys stripped on mark. Worker ``execution_proof`` is
#: intentionally NOT listed — prism requires it for tier-0 verification.
_ATTESTED_LOOKING_KEYS: Final = frozenset(
    {
        "attestation_binding",
        "tdx_quote",
        "phala_attestation",
        "tdx_quote_b64",
        "gpu_eat_jwt",
    }
)


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


def resolve_unattested_execution_from_environ(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve unattested-execution mode from env (T19 precedence)."""

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


def is_unattested_execution_enabled(
    *,
    settings_flag: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether unattested (non-TEE) execution mode is active.

    Prefer ``settings_flag`` when the process already resolved the flag at
    startup. Otherwise read env via :func:`resolve_unattested_execution_from_environ`.
    """

    if settings_flag is not None:
        return bool(settings_flag)
    return resolve_unattested_execution_from_environ(environ)


def mark_result_unattested(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a result envelope explicitly marked unattested (unforgeable).

    Always sets ``attested=False`` and ``attestation_status=unattested`` and
    ``execution_mode=no_phala_host``. Callers cannot pass ``attested=True``
    through this function — the boolean is hard-coded (T21).

    Strips TEE/Phala-looking claim keys. Preserves prism worker
    ``execution_proof`` (not a TEE claim).
    """

    out = dict(payload)
    for key in _ATTESTED_LOOKING_KEYS:
        out.pop(key, None)
    # Hard-coded False — never accept a caller-supplied True.
    out[RESULT_KEY_ATTESTED] = False
    out[RESULT_KEY_ATTESTATION_STATUS] = ATTESTATION_STATUS_UNATTESTED
    out[RESULT_KEY_EXECUTION_MODE] = EXECUTION_MODE_NO_PHALA_HOST
    return out


def project_public_attestation(
    *,
    stored: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Unforgeable public projection of attestation fields (server-side only).

    When unattested execution is active, always returns the same mark as
    :func:`mark_result_unattested`. Never promotes miner-supplied claims.
    """

    del stored  # never trust client/stored claims under unattested mode path
    if is_unattested_execution_enabled():
        return {
            RESULT_KEY_ATTESTED: False,
            RESULT_KEY_ATTESTATION_STATUS: ATTESTATION_STATUS_UNATTESTED,
            RESULT_KEY_EXECUTION_MODE: EXECUTION_MODE_NO_PHALA_HOST,
        }
    # Flag off: honest default is unattested/standard — never invent TEE.
    return {
        RESULT_KEY_ATTESTED: False,
        RESULT_KEY_ATTESTATION_STATUS: ATTESTATION_STATUS_UNATTESTED,
        RESULT_KEY_EXECUTION_MODE: "standard",
    }


__all__ = [
    "ATTESTATION_STATUS_UNATTESTED",
    "CHALLENGE_NO_PHALA_ENV",
    "CHALLENGE_UNATTESTED_EXECUTION_ENV",
    "EXECUTION_MODE_NO_PHALA_HOST",
    "NO_PHALA_ENV",
    "RESULT_KEY_ATTESTATION_STATUS",
    "RESULT_KEY_ATTESTED",
    "RESULT_KEY_EXECUTION_MODE",
    "is_unattested_execution_enabled",
    "mark_result_unattested",
    "project_public_attestation",
    "resolve_unattested_execution_from_environ",
]
