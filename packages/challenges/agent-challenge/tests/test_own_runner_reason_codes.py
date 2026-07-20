"""Snapshot + drift-detection tests for the own-runner reason-code taxonomy.

These tests pin the full reason-code set and guarantee the own-runner taxonomy
stays a superset of every legacy ``harbor_*`` / ``terminal_bench_*`` reason code
that miners and dashboards consume. Any future change that drops or renames a
code without updating the taxonomy (and ``REMAP``) makes one of these fail.
"""

from __future__ import annotations

from agent_challenge.evaluation import terminal_bench
from agent_challenge.evaluation.own_runner import reason_codes

# ---------------------------------------------------------------------------
# Frozen snapshot of the COMPLETE taxonomy. This is the pin: changing it is the
# explicit, reviewed act of changing the public reason-code surface.
# ---------------------------------------------------------------------------
EXPECTED_REASON_CODES = frozenset(
    {
        # harbor retryable
        "harbor_broker_connection_failed",
        "harbor_cancelled_error",
        "harbor_environment_start_timeout_error",
        # harbor final
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
        # infrastructure retryable
        "terminal_bench_broker_ref_missing",
        "terminal_bench_job_dir_missing",
        "terminal_bench_lease_expired",
        # sentinel / fallback
        "terminal_bench_attempt_not_running",
        "terminal_bench_failed",
        "phala_attestation_failed",
        "phala_key_release_failed",
        "phala_golden_decrypt_failed",
    }
)

EXPECTED_HARBOR_REASON_CODES = frozenset(
    code for code in EXPECTED_REASON_CODES if code.startswith("harbor_")
)


def test_reason_codes_match_frozen_snapshot() -> None:
    """The full taxonomy must match the pinned snapshot exactly (no drift)."""
    assert reason_codes.REASON_CODES == EXPECTED_REASON_CODES


def test_harbor_reason_codes_match_frozen_snapshot() -> None:
    assert reason_codes.HARBOR_REASON_CODES == EXPECTED_HARBOR_REASON_CODES


def test_taxonomy_is_superset_of_all_harbor_codes() -> None:
    """Taxonomy must contain EVERY existing harbor_* code (the core contract)."""
    assert reason_codes.HARBOR_REASON_CODES <= reason_codes.REASON_CODES
    assert EXPECTED_HARBOR_REASON_CODES <= reason_codes.REASON_CODES


def test_taxonomy_covers_legacy_harbor_codes_exactly() -> None:
    """Pin against the live legacy source so adding a legacy harbor_* code there
    without updating this taxonomy fails the build (no silent additions)."""
    legacy_union = (
        terminal_bench.TERMINAL_BENCH_RETRYABLE_REASON_CODES
        | terminal_bench.TERMINAL_BENCH_FINAL_REASON_CODES
    )
    legacy_harbor = frozenset(c for c in legacy_union if c.startswith("harbor_"))
    # Identical: taxonomy neither drops nor silently renames a legacy harbor code.
    assert reason_codes.HARBOR_REASON_CODES == legacy_harbor


def test_taxonomy_covers_every_legacy_classified_code() -> None:
    """Every code the legacy backend classifies (harbor_* AND terminal_bench_*)
    must be expressible by the own-runner taxonomy."""
    legacy_union = (
        terminal_bench.TERMINAL_BENCH_RETRYABLE_REASON_CODES
        | terminal_bench.TERMINAL_BENCH_FINAL_REASON_CODES
    )
    missing = legacy_union - reason_codes.REASON_CODES
    assert not missing, f"taxonomy missing legacy codes: {sorted(missing)}"


def test_legacy_aliases_resolve_into_taxonomy() -> None:
    """Every legacy alias target must be a known taxonomy code."""
    for target in terminal_bench._TERMINAL_BENCH_REASON_ALIASES.values():
        assert target in reason_codes.REASON_CODES


def test_retryable_and_final_partitions_mirror_legacy() -> None:
    assert reason_codes.RETRYABLE_REASON_CODES == (
        terminal_bench.TERMINAL_BENCH_RETRYABLE_REASON_CODES
    )
    assert reason_codes.FINAL_REASON_CODES == terminal_bench.TERMINAL_BENCH_FINAL_REASON_CODES


def test_category_frozensets_are_disjoint_and_complete() -> None:
    parts = [
        reason_codes.HARBOR_RETRYABLE_REASON_CODES,
        reason_codes.HARBOR_FINAL_REASON_CODES,
        reason_codes.INFRA_RETRYABLE_REASON_CODES,
        reason_codes.SENTINEL_REASON_CODES,
    ]
    # Disjoint: no code lives in two categories.
    seen: set[str] = set()
    for part in parts:
        assert seen.isdisjoint(part), f"overlap: {sorted(seen & part)}"
        seen |= part
    # Complete: the categories exactly reconstruct the full taxonomy.
    assert frozenset(seen) == reason_codes.REASON_CODES


def test_remap_is_consistent() -> None:
    """REMAP keys are retired codes (absent from taxonomy); values are current."""
    for old, new in reason_codes.REMAP.items():
        assert old not in reason_codes.REASON_CODES, f"remap key still live: {old}"
        assert new in reason_codes.REASON_CODES, f"remap target unknown: {new}"


def test_remap_reason_code_identity_when_not_remapped() -> None:
    assert reason_codes.remap_reason_code("harbor_nonzero_exit") == "harbor_nonzero_exit"
    assert reason_codes.remap_reason_code(None) is None


def test_is_known_reason_code() -> None:
    assert reason_codes.is_known_reason_code("harbor_reward_empty")
    assert not reason_codes.is_known_reason_code("not_a_real_code")
    assert not reason_codes.is_known_reason_code(None)
