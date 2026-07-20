"""Low-rate replay-audit sampler (architecture sec 4 C6 / sec 8, defense-in-depth).

Behavioral contract for the ``replay-audit-sampling`` feature: the sampler picks
a small fraction of the ATTESTED submission population for a defense-in-depth
replay, at a tier-driven rate (higher trust => strictly lower rate), is
deterministic/seedable, a rate of 0 disables it, and it is inert (zero sampling)
when the Phala attestation flag is off. Anchored to the mission assertions:

* VAL-SCORE-016 -- samples ~= rate * N of the attested population.
* VAL-SCORE-017 -- deterministic and seedable (same seed => identical subset; a
  different seed => a different subset at the same rate).
* VAL-SCORE-018 -- a rate of 0 disables the audit entirely (zero sampled).
* VAL-SCORE-025 -- tier-driven rate: higher-trust tier gets a strictly lower
  rate, and an unverifiable/failed attestation never buys the reduced rate.
* VAL-SCORE-026 -- the population is the attested submissions and the sampler is
  inert when the feature flag is off.
"""

from __future__ import annotations

import pytest

from agent_challenge.evaluation.replay_audit import (
    AUDIT_TIER_ATTESTED,
    AUDIT_TIER_UNVERIFIED,
    AuditCandidate,
    InvalidAuditRateError,
    ReplayAuditSampler,
    replay_audit_sampler_from_settings,
)
from agent_challenge.sdk.config import ChallengeSettings


def _attested_population(n: int, *, prefix: str = "sub") -> list[AuditCandidate]:
    """A population of ``n`` verified-attestation (high-trust) submissions."""

    return [AuditCandidate(submission_id=f"{prefix}-{index}") for index in range(n)]


# --------------------------------------------------------------------------- #
# VAL-SCORE-016: samples ~= rate * N of the attested population.
# --------------------------------------------------------------------------- #
def test_samples_configured_fraction_of_population() -> None:
    population = _attested_population(5000)
    sampler = ReplayAuditSampler(attested_rate=0.02, unverified_rate=0.10, seed=7)

    selected = sampler.sample(population)

    # ~= 0.02 * 5000 = 100, and never all/none for 0 < rate < 1.
    assert len(selected) == 100
    assert 0 < len(selected) < len(population)
    observed = len(selected) / len(population)
    assert observed == pytest.approx(0.02, abs=0.005)


def test_selected_are_a_subset_of_the_population() -> None:
    population = _attested_population(2000)
    sampler = ReplayAuditSampler(attested_rate=0.05, unverified_rate=0.20, seed=3)

    selected = sampler.sample(population)
    ids = {candidate.submission_id for candidate in population}

    assert set(selected) <= ids
    assert len(set(selected)) == len(selected)  # no duplicates


def test_sampled_fraction_tracks_the_configured_rate() -> None:
    population = _attested_population(4000)
    for rate, expected in ((0.01, 40), (0.05, 200), (0.25, 1000)):
        sampler = ReplayAuditSampler(attested_rate=rate, unverified_rate=0.5, seed=11)
        assert len(sampler.sample(population)) == expected


# --------------------------------------------------------------------------- #
# VAL-SCORE-017: deterministic and seedable.
# --------------------------------------------------------------------------- #
def test_same_seed_selects_the_identical_subset() -> None:
    population = _attested_population(5000)
    sampler = ReplayAuditSampler(attested_rate=0.02, unverified_rate=0.10, seed=42)

    first = sampler.sample(population)
    second = sampler.sample(population)

    assert first == second  # reproducible run-to-run (order included)
    # A second sampler with the same seed reproduces the identical id set.
    twin = ReplayAuditSampler(attested_rate=0.02, unverified_rate=0.10, seed=42)
    assert set(twin.sample(population)) == set(first)


def test_different_seed_selects_a_different_subset_at_the_same_rate() -> None:
    population = _attested_population(5000)
    a = ReplayAuditSampler(attested_rate=0.02, unverified_rate=0.10, seed=1)
    b = ReplayAuditSampler(attested_rate=0.02, unverified_rate=0.10, seed=2)

    selected_a = a.sample(population)
    selected_b = b.sample(population)

    assert set(selected_a) != set(selected_b)  # the seed actually matters
    assert len(selected_a) == len(selected_b) == 100  # same rate => same count


# --------------------------------------------------------------------------- #
# VAL-SCORE-018: a rate of 0 disables the audit entirely.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1000, 5000])
@pytest.mark.parametrize("seed", [0, 1, 99])
def test_rate_zero_samples_nothing(n: int, seed: int) -> None:
    population = _attested_population(n)
    sampler = ReplayAuditSampler(attested_rate=0.0, unverified_rate=0.0, seed=seed)

    assert sampler.sample(population) == []


def test_rate_zero_for_one_tier_disables_only_that_tier() -> None:
    verified = [AuditCandidate(f"v-{i}", verified=True) for i in range(2000)]
    unverified = [AuditCandidate(f"u-{i}", verified=False) for i in range(2000)]
    sampler = ReplayAuditSampler(attested_rate=0.0, unverified_rate=0.10, seed=5)

    selected = set(sampler.sample(verified + unverified))

    assert not any(cid.startswith("v-") for cid in selected)  # attested rate 0
    assert any(cid.startswith("u-") for cid in selected)  # unverified still audited


# --------------------------------------------------------------------------- #
# VAL-SCORE-025: tier-driven rate (higher trust => strictly lower rate).
# --------------------------------------------------------------------------- #
def test_higher_trust_tier_gets_a_strictly_lower_observed_rate() -> None:
    verified = [AuditCandidate(f"v-{i}", verified=True) for i in range(5000)]
    unverified = [AuditCandidate(f"u-{i}", verified=False) for i in range(5000)]
    sampler = ReplayAuditSampler(attested_rate=0.02, unverified_rate=0.10, seed=8)

    selected = set(sampler.sample(verified + unverified))
    high_fraction = sum(cid.startswith("v-") for cid in selected) / 5000
    low_fraction = sum(cid.startswith("u-") for cid in selected) / 5000

    assert high_fraction < low_fraction  # higher trust audited strictly less


def test_default_rates_order_high_trust_below_low_trust() -> None:
    sampler = replay_audit_sampler_from_settings(ChallengeSettings())
    assert sampler.rate_for_tier(AUDIT_TIER_ATTESTED) < sampler.rate_for_tier(AUDIT_TIER_UNVERIFIED)


def test_unverifiable_submission_is_classified_low_trust() -> None:
    # An attestation that failed to verify must NOT buy the reduced (high-trust)
    # rate: it is audited at the low-trust (higher) rate.
    verified = AuditCandidate("ok", verified=True)
    unverifiable = AuditCandidate("bad", verified=False)

    assert verified.tier == AUDIT_TIER_ATTESTED
    assert unverifiable.tier == AUDIT_TIER_UNVERIFIED

    sampler = ReplayAuditSampler(attested_rate=0.02, unverified_rate=0.10)
    assert sampler.rate_for_tier(unverifiable.tier) == 0.10
    assert sampler.rate_for_tier(verified.tier) == 0.02


def test_unverifiable_submission_audited_at_low_trust_rate_in_population() -> None:
    # With attested rate 0 and a non-zero unverified rate, only the unverifiable
    # submissions can be sampled -- an unverifiable claim never gets the reduced
    # (zero) high-trust rate.
    verified = [AuditCandidate(f"v-{i}", verified=True) for i in range(3000)]
    unverifiable = [AuditCandidate(f"u-{i}", verified=False) for i in range(3000)]
    sampler = ReplayAuditSampler(attested_rate=0.0, unverified_rate=0.10, seed=13)

    selected = set(sampler.sample(verified + unverifiable))

    assert selected  # some unverifiable ones sampled
    assert all(cid.startswith("u-") for cid in selected)


# --------------------------------------------------------------------------- #
# VAL-SCORE-026: attested-only population and inert when the flag is off.
# --------------------------------------------------------------------------- #
def test_legacy_non_attested_submissions_are_never_sampled() -> None:
    attested = [AuditCandidate(f"a-{i}", attested=True) for i in range(2000)]
    legacy = [AuditCandidate(f"legacy-{i}", attested=False) for i in range(2000)]
    sampler = ReplayAuditSampler(attested_rate=0.5, unverified_rate=0.5, seed=4)

    selected = set(sampler.sample(attested + legacy))

    assert selected  # attested ones are audited
    assert not any(cid.startswith("legacy-") for cid in selected)


def test_disabled_sampler_is_inert() -> None:
    population = _attested_population(5000)
    sampler = ReplayAuditSampler(attested_rate=0.5, unverified_rate=0.5, seed=1, enabled=False)

    assert sampler.sample(population) == []


def test_empty_population_samples_nothing() -> None:
    sampler = ReplayAuditSampler(attested_rate=0.5, unverified_rate=0.5, seed=1)

    assert sampler.sample([]) == []
    # A population of only legacy (non-attested) submissions is empty for audit.
    legacy = [AuditCandidate(f"legacy-{i}", attested=False) for i in range(10)]
    assert sampler.sample(legacy) == []


def test_rate_one_samples_the_entire_tier() -> None:
    population = _attested_population(500)
    sampler = ReplayAuditSampler(attested_rate=1.0, unverified_rate=1.0, seed=2)

    selected = sampler.sample(population)

    assert set(selected) == {c.submission_id for c in population}
    # Original population order is preserved.
    assert selected == [c.submission_id for c in population]


def test_from_settings_is_inert_when_flag_off() -> None:
    settings = ChallengeSettings(phala_attestation_enabled=False)
    sampler = replay_audit_sampler_from_settings(settings)

    assert sampler.enabled is False
    assert sampler.sample(_attested_population(5000)) == []


def test_from_settings_enables_sampler_when_flag_on() -> None:
    settings = ChallengeSettings(
        attested_review_enabled=True,
        phala_attestation_enabled=True,
    )
    sampler = replay_audit_sampler_from_settings(settings)

    assert sampler.enabled is True
    assert len(sampler.sample(_attested_population(5000))) > 0


# --------------------------------------------------------------------------- #
# Fail-closed configuration.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_rate", [-0.01, 1.01, 2.0, -1.0])
def test_out_of_range_rate_is_rejected(bad_rate: float) -> None:
    with pytest.raises(InvalidAuditRateError):
        ReplayAuditSampler(attested_rate=bad_rate, unverified_rate=0.1)
    with pytest.raises(InvalidAuditRateError):
        ReplayAuditSampler(attested_rate=0.1, unverified_rate=bad_rate)


def test_unknown_tier_is_rejected() -> None:
    sampler = ReplayAuditSampler(attested_rate=0.02, unverified_rate=0.10)
    with pytest.raises(InvalidAuditRateError):
        sampler.rate_for_tier("mystery-tier")


@pytest.mark.parametrize("bad_rate", [-0.01, 1.01])
def test_settings_reject_out_of_range_replay_audit_rate(bad_rate: float) -> None:
    with pytest.raises(ValueError):
        ChallengeSettings(replay_audit_attested_rate=bad_rate)
    with pytest.raises(ValueError):
        ChallengeSettings(replay_audit_unverified_rate=bad_rate)


def test_settings_defaults_are_low_rate_and_disabled_by_flag() -> None:
    settings = ChallengeSettings()
    assert 0.0 < settings.replay_audit_attested_rate < settings.replay_audit_unverified_rate
    assert settings.phala_attestation_enabled is False  # audit inert by default
