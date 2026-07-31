//! Verifier-fault → leaf mapping (`AGENT_CHALLENGE` §7.3 priority 7).
//!
//! Operator-side Harbor faults must never silence a hotkey in `E` and must never
//! be charged as miner solve failure. Park / `AttestationNotVerified` is D13
//! attestation-only — **not** reused for verifier outages.
//!
//! # Mapping table (`map_verify_error`)
//!
//! | [`VerifyError`] | [`ScoreOrAbsence`] | Attribution |
//! |-----------------|--------------------|-------------|
//! | `Timeout` (verifier wall) | `NoScore(ChallengeInternal)` | operator |
//! | `Docker` (crash, pull, API) | `NoScore(ChallengeInternal)` | operator |
//! | `MalformedOutput` (junit/reward parse) | `NoScore(ChallengeInternal)` | operator |
//! | `Staging` | `NoScore(ChallengeInternal)` | operator |
//! | `MissingHeldOut` | `NoScore(ChallengeInternal)` | operator |
//! | `ApplyFailed` | `Score { value: 0 }` | miner |
//! | `RewardZero` | `Score { value: 0 }` | miner |
//!
//! Successful `Reward(1)` → `Score { SCORE_MAX }`; `Reward(0)` → `Score { 0 }`.

use std::collections::{BTreeMap, BTreeSet};

use agent_pack::HarborPack;
use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use crypto::KEY_LEN;
use docker_engine::DockerError;

use crate::score::SCORE_MAX;
use crate::verify::{Reward, Verifier, VerifyError};

/// Miner hotkey bytes.
pub type Hotkey = [u8; KEY_LEN];

/// Max retries after the first grade attempt for **operator** faults only.
///
/// Total attempts = `1 + MAX_VERIFY_RETRIES` (never unbounded; seal-safe).
pub const MAX_VERIFY_RETRIES: u32 = 2;

/// Total grade attempts allowed for operator faults.
pub const MAX_VERIFY_ATTEMPTS: u32 = 1 + MAX_VERIFY_RETRIES;

/// True when the error is operator infrastructure (not miner patch quality).
#[must_use]
pub fn is_operator_fault(err: &VerifyError) -> bool {
    matches!(
        err,
        VerifyError::Timeout { .. }
            | VerifyError::Docker(_)
            | VerifyError::MalformedOutput { .. }
            | VerifyError::Staging { .. }
            | VerifyError::MissingHeldOut { .. }
    )
}

/// True when a retry may help (transient operator faults).
///
/// Malformed output and missing held-out are deterministic — do not retry.
#[must_use]
pub fn is_retryable_operator_fault(err: &VerifyError) -> bool {
    match err {
        VerifyError::Timeout { .. } | VerifyError::Staging { .. } => true,
        VerifyError::Docker(d) => !matches!(
            d,
            DockerError::NotAllowlisted { .. } | DockerError::BadName { .. }
        ),
        VerifyError::MalformedOutput { .. }
        | VerifyError::MissingHeldOut { .. }
        | VerifyError::ApplyFailed { .. }
        | VerifyError::RewardZero { .. } => false,
    }
}

/// Map a terminal [`VerifyError`] to a leaf payload.
///
/// Miner-attributable zeros use `Score { 0 }` (same lattice as wrong answer).
/// Operator faults use `ChallengeInternal` — never Park / `AttestationNotVerified`.
#[must_use]
pub fn map_verify_error(err: &VerifyError) -> ScoreOrAbsence {
    match err {
        VerifyError::ApplyFailed { .. } | VerifyError::RewardZero { .. } => {
            ScoreOrAbsence::Score { value: 0 }
        }
        VerifyError::Timeout { .. }
        | VerifyError::Docker(_)
        | VerifyError::MalformedOutput { .. }
        | VerifyError::Staging { .. }
        | VerifyError::MissingHeldOut { .. } => ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal,
        },
    }
}

/// Map a successful binary reward to the score lattice.
#[must_use]
pub fn map_reward(reward: Reward) -> ScoreOrAbsence {
    if reward.is_resolve() {
        ScoreOrAbsence::Score { value: SCORE_MAX }
    } else {
        ScoreOrAbsence::Score { value: 0 }
    }
}

/// Map `Result` from a single grade (no retries).
#[must_use]
pub fn score_from_verify_result(result: &Result<Reward, VerifyError>) -> ScoreOrAbsence {
    match result {
        Ok(r) => map_reward(*r),
        Err(e) => map_verify_error(e),
    }
}

/// Grade with bounded operator retries; always returns a leaf payload (never silence).
pub fn grade_to_score_or_absence(
    verifier: &dyn Verifier,
    pack: &HarborPack,
    model_patch: &[u8],
) -> ScoreOrAbsence {
    grade_loop(verifier, pack, model_patch, MAX_VERIFY_ATTEMPTS)
}

/// Cover every hotkey in `expected` with exactly one leaf from verify results.
///
/// Missing map entries → `ChallengeInternal` (silence is a bug). Extra keys ignored.
#[must_use]
pub fn cover_expected_verify_leaves(
    expected: &BTreeSet<Hotkey>,
    results: &BTreeMap<Hotkey, Result<Reward, VerifyError>>,
) -> BTreeMap<Hotkey, ScoreOrAbsence> {
    expected
        .iter()
        .map(|h| {
            let soa = results.get(h).map_or_else(
                || ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::ChallengeInternal,
                },
                score_from_verify_result,
            );
            (*h, soa)
        })
        .collect()
}

/// Cap attempts by remaining seal budget so retries cannot outrun the deadline.
///
/// Returns 0 when no full attempt fits; otherwise `min(floor(budget/per), MAX)`.
#[must_use]
pub fn attempts_within_seal_budget(remaining_ms: u64, per_attempt_ms: u64) -> u32 {
    if remaining_ms == 0 || per_attempt_ms == 0 {
        return 0;
    }
    let by_budget = remaining_ms / per_attempt_ms;
    if by_budget == 0 {
        return 0;
    }
    let by_budget_u32 = u32::try_from(by_budget).unwrap_or(u32::MAX);
    by_budget_u32.min(MAX_VERIFY_ATTEMPTS)
}

/// Like [`grade_to_score_or_absence`] but stops early when seal budget is exhausted.
pub fn grade_to_score_or_absence_budgeted(
    verifier: &dyn Verifier,
    pack: &HarborPack,
    model_patch: &[u8],
    remaining_ms: u64,
    per_attempt_ms: u64,
) -> ScoreOrAbsence {
    let allowed = attempts_within_seal_budget(remaining_ms, per_attempt_ms);
    if allowed == 0 {
        return ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal,
        };
    }
    grade_loop(verifier, pack, model_patch, allowed)
}

fn grade_loop(
    verifier: &dyn Verifier,
    pack: &HarborPack,
    model_patch: &[u8],
    allowed: u32,
) -> ScoreOrAbsence {
    let mut attempts = 0u32;
    let mut last_operator = None;
    while attempts < allowed {
        attempts = attempts.saturating_add(1);
        match verifier.grade(pack, model_patch) {
            Ok(r) => return map_reward(r),
            Err(e) if !is_operator_fault(&e) => return map_verify_error(&e),
            Err(e) => {
                let retry = is_retryable_operator_fault(&e) && attempts < allowed;
                last_operator = Some(e);
                if !retry {
                    break;
                }
            }
        }
    }
    last_operator.as_ref().map_or_else(
        || ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal,
        },
        map_verify_error,
    )
}
