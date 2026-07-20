"""Centralized reason-code taxonomy for the own-runner evaluation backend.

This module is the single source of truth for the ``reason_code`` strings the
new own-runner backend emits when an evaluation attempt does not complete
cleanly. It centralizes the codes that previously lived as ad-hoc literals and
frozensets in :mod:`agent_challenge.evaluation.terminal_bench` (the legacy
harbor backend).

Compatibility contract (BINDING)
--------------------------------
Miners and dashboards key on these exact strings. Therefore:

* The own-runner backend MUST emit codes that are **byte-identical** to the
  legacy harbor backend. No silent renames.
* :data:`REASON_CODES` is a **superset** of every legacy ``harbor_*`` reason
  code (and of every legacy ``terminal_bench_*`` reason code), so the new
  backend can express every outcome the old one could.
* If a code is ever renamed or removed, the change MUST be recorded in
  :data:`REMAP` (old -> new) so consumers can migrate explicitly. The snapshot
  test (``tests/test_own_runner_reason_codes.py``) fails on any undocumented
  drift.

Taxonomy provenance (legacy ``terminal_bench.py`` as of centralization)
-----------------------------------------------------------------------
* Retryable codes  -> ``TERMINAL_BENCH_RETRYABLE_REASON_CODES`` (lines 43-52)
* Final codes      -> ``TERMINAL_BENCH_FINAL_REASON_CODES`` (lines 53-70)
* Sentinel codes   -> emitted as fallbacks but not held in either frozenset:
  ``terminal_bench_attempt_not_running`` (terminal_bench.py:248) and
  ``terminal_bench_failed`` (terminal_bench.py:375, runner.py:1056,
  reconciler.py:286).

The reward/result codes confirmed by gate G2
(``harbor_reward_empty``/``_missing``/``_parse_error`` and
``harbor_result_missing``/``_malformed``) are all included below.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Retryable harbor codes (legacy TERMINAL_BENCH_RETRYABLE_REASON_CODES, harbor_*)
# ---------------------------------------------------------------------------
HARBOR_RETRYABLE_REASON_CODES: frozenset[str] = frozenset(
    {
        "harbor_broker_connection_failed",
        "harbor_cancelled_error",
        "harbor_environment_start_timeout_error",
    }
)

# ---------------------------------------------------------------------------
# Final harbor codes (legacy TERMINAL_BENCH_FINAL_REASON_CODES)
# Includes every G2-confirmed reward/result code.
# ---------------------------------------------------------------------------
HARBOR_FINAL_REASON_CODES: frozenset[str] = frozenset(
    {
        "harbor_agent_timeout_error",
        "harbor_nonzero_exit",
        "harbor_result_invalid",
        "harbor_result_malformed",
        "harbor_result_missing",
        "harbor_result_partial",
        "harbor_reward_empty",
        "harbor_reward_missing",
        "harbor_reward_parse_error",
        "harbor_submission_code_failed",
        "harbor_trial_failed",
        "harbor_trial_result_malformed",
        "harbor_trial_result_missing",
        "harbor_verifier_timeout_error",
    }
)

# ---------------------------------------------------------------------------
# Infrastructure (orchestration) retryable codes. Part of the legacy
# retryable frozenset but namespaced terminal_bench_* rather than harbor_*
# because they describe the runner/broker plumbing, not harbor itself.
# ---------------------------------------------------------------------------
INFRA_RETRYABLE_REASON_CODES: frozenset[str] = frozenset(
    {
        "terminal_bench_broker_ref_missing",
        "terminal_bench_job_dir_missing",
        "terminal_bench_lease_expired",
    }
)

# ---------------------------------------------------------------------------
# Sentinel / fallback codes. Emitted as outcome reason codes but not held in
# either legacy classification frozenset.
#   * terminal_bench_attempt_not_running -> stale attempt (terminal_bench.py:248)
#   * terminal_bench_failed              -> generic failure fallback when no
#                                           specific reason is known.
#   * phala_attestation_failed           -> Phala path fail-closed: a genuine
#                                           TDX quote could not be produced, so
#                                           no attested result is emitted.
#   * phala_key_release_failed            -> Phala path fail-closed: the validator
#                                           golden-key-release could not be
#                                           obtained (deny / unreachable / dropped
#                                           mid-exchange), so the eval never runs
#                                           the verifier against golden and emits
#                                           no passing score.
#   * phala_golden_decrypt_failed         -> Phala path fail-closed: the released
#                                           key did not unseal the encrypted-at-rest
#                                           golden in-enclave (wrong key / tampered
#                                           or missing ciphertext), so the eval never
#                                           runs and emits no passing score.
# ---------------------------------------------------------------------------
SENTINEL_REASON_CODES: frozenset[str] = frozenset(
    {
        "terminal_bench_attempt_not_running",
        "terminal_bench_failed",
        "phala_attestation_failed",
        "phala_key_release_failed",
        "phala_golden_decrypt_failed",
    }
)

# Every harbor_* code the backend may emit (superset target for the snapshot
# test's harbor-coverage assertion).
HARBOR_REASON_CODES: frozenset[str] = HARBOR_RETRYABLE_REASON_CODES | HARBOR_FINAL_REASON_CODES

# Legacy retryable / final mirrors. Kept identical to terminal_bench.py so the
# own-runner backend can classify retries with the same semantics.
RETRYABLE_REASON_CODES: frozenset[str] = (
    HARBOR_RETRYABLE_REASON_CODES | INFRA_RETRYABLE_REASON_CODES
)
FINAL_REASON_CODES: frozenset[str] = HARBOR_FINAL_REASON_CODES

# The complete taxonomy: every reason code the own-runner backend may emit.
REASON_CODES: frozenset[str] = (
    HARBOR_RETRYABLE_REASON_CODES
    | HARBOR_FINAL_REASON_CODES
    | INFRA_RETRYABLE_REASON_CODES
    | SENTINEL_REASON_CODES
)

# ---------------------------------------------------------------------------
# Explicit old -> new remap table.
#
# EMPTY BY DESIGN: the own-runner backend keeps every code byte-identical to the
# legacy harbor backend, so no remapping is required. If a code must ever be
# renamed, add an ``"old_code": "new_code"`` entry here (and keep "new_code" in
# REASON_CODES). The snapshot test enforces that every value is a known code and
# every key is NOT (i.e. the old code has truly been retired).
# ---------------------------------------------------------------------------
REMAP: dict[str, str] = {}


def is_known_reason_code(value: str | None) -> bool:
    """Return ``True`` if ``value`` is a current taxonomy reason code."""
    return value is not None and value in REASON_CODES


def remap_reason_code(value: str | None) -> str | None:
    """Apply :data:`REMAP` to ``value``.

    Returns the remapped code when ``value`` is a retired code listed in
    :data:`REMAP`, otherwise returns ``value`` unchanged (identity). ``None``
    passes through as ``None``.
    """
    if value is None:
        return None
    return REMAP.get(value, value)


__all__ = [
    "FINAL_REASON_CODES",
    "HARBOR_FINAL_REASON_CODES",
    "HARBOR_REASON_CODES",
    "HARBOR_RETRYABLE_REASON_CODES",
    "INFRA_RETRYABLE_REASON_CODES",
    "REASON_CODES",
    "REMAP",
    "RETRYABLE_REASON_CODES",
    "SENTINEL_REASON_CODES",
    "is_known_reason_code",
    "remap_reason_code",
]
