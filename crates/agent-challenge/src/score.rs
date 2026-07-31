//! Pure integer scoring rule (`AGENT_CHALLENGE` §5.4, scoring_version = 2).
//!
//! Live path uses pack-bound v2 task identity and `answer_digest_v2(model.patch)`.
//! v2 scores **pure correctness** only — latency decay is removed. `duration_ms` is
//! ignored by the scorer; epoch deadline is a hard `CallOutcome::Timeout` boundary,
//! not a decay curve. The v1 echo answer is retired.

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use crypto::KEY_LEN;

use crate::task_gen::{answer_digest_v2, task_id_v2, CHALLENGE_ID, SCORING_VERSION};

/// Maximum score value.
pub const SCORE_MAX: u64 = 1_000_000;
/// Connect timeout (ms) — protocol constant (not used in pure scorer).
pub const CONNECT_MS: u64 = 3_000;
/// Max HTTP attempts per miner.
pub const MAX_ATTEMPTS: u32 = 2;

/// Attestation outcome for `(netuid, epoch, miner)` this epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttestationStatus {
    /// Same-epoch Verified — may emit `Score`.
    Verified,
    /// Rejected — must `NoScore(AttestationNotVerified)`.
    Rejected,
    /// Parked — must `NoScore(AttestationNotVerified)` (D13).
    Parked,
    /// Missing / undecided — must `NoScore(AttestationNotVerified)`.
    #[default]
    Missing,
}

/// Inputs for the pure scoring function after a miner attempt (or terminal failure).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreInputs {
    /// Subnet netuid (for expected digests).
    pub netuid: u16,
    /// Epoch.
    pub epoch: u64,
    /// Miner hotkey.
    pub miner_hotkey: [u8; KEY_LEN],
    /// Pack id bound into v2 task identity (UTF-8 bytes).
    pub pack_id: Vec<u8>,
    /// Oracle / fixture expected `model.patch` for [`answer_digest_v2`].
    pub expected_model_patch: Vec<u8>,
    /// Attestation status this epoch.
    pub attestation: AttestationStatus,
    /// Challenge-side wall time of the attempt (informational; **ignored** by v2 scorer).
    pub duration_ms: u64,
    /// Terminal miner/call outcome.
    pub outcome: CallOutcome,
}

/// Terminal outcome of the challenge↔miner hop (after retries exhausted).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallOutcome {
    /// HTTP 200 body accepted for scoring.
    Http200 {
        /// Response `challenge_id`.
        challenge_id: String,
        /// Response epoch.
        epoch: u64,
        /// Response `task_id` (32 bytes).
        task_id: [u8; 32],
        /// Response `answer_digest` (32 bytes) — must equal `answer_digest_v2(model.patch)`.
        answer_digest: [u8; 32],
        /// Response `agent_version`.
        agent_version: String,
    },
    /// Timeout / transport exhausted / epoch deadline exceeded.
    Timeout,
    /// HTTP 500 / `agent_internal`.
    MinerError,
    /// Schema / 400 / 403 / hotkey mismatch / field disagree.
    InvalidResponse,
    /// HTTP 429 exhausted.
    RateLimited,
    /// Challenge-side fault after retries.
    ChallengeInternal,
}

/// Score a miner from pure inputs (`AGENT_CHALLENGE` §5.4, v2 correctness).
///
/// Attestation is checked **before** honoring any reward. Correct answer →
/// [`SCORE_MAX`]; wrong answer → `Score(0)`. Bare floating point is forbidden —
/// integer lattice only. `duration_ms` never decays the score.
#[must_use]
pub fn score_from_outcome(input: &ScoreInputs) -> ScoreOrAbsence {
    use NoScoreReasonCode as R;
    if input.attestation != AttestationStatus::Verified {
        return ScoreOrAbsence::NoScore {
            reason: R::AttestationNotVerified,
        };
    }
    let _ = input.duration_ms; // v2: no latency decay
    let ns = |r| ScoreOrAbsence::NoScore { reason: r };
    match &input.outcome {
        CallOutcome::Timeout => ns(R::Timeout),
        CallOutcome::MinerError => ns(R::MinerError),
        CallOutcome::InvalidResponse => ns(R::InvalidResponse),
        CallOutcome::RateLimited => ns(R::RateLimited),
        CallOutcome::ChallengeInternal => ns(R::ChallengeInternal),
        CallOutcome::Http200 {
            challenge_id,
            epoch,
            task_id: tid,
            answer_digest: ans,
            agent_version,
        } => {
            let exp_tid = task_id_v2(
                input.netuid,
                input.epoch,
                &input.miner_hotkey,
                &input.pack_id,
                SCORING_VERSION,
            );
            let exp_ans = answer_digest_v2(&input.expected_model_patch);
            if challenge_id != CHALLENGE_ID
                || *epoch != input.epoch
                || tid.as_slice() != exp_tid.as_slice()
                || agent_version != "1"
            {
                return ns(R::InvalidResponse);
            }
            if ans.as_slice() == exp_ans.as_slice() {
                ScoreOrAbsence::Score { value: SCORE_MAX }
            } else {
                ScoreOrAbsence::Score { value: 0 }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// Scorer source must stay integer-only and free of latency constants.
    #[test]
    fn scorer_source_no_float_no_soft_hard_ms() {
        let src = include_str!("score.rs");
        let prod = src
            .split("#[cfg(test)]")
            .next()
            .expect("production half of score.rs");
        assert!(
            !prod.contains("f32") && !prod.contains("f64"),
            "scorer production code must not use f32/f64"
        );
        assert!(
            !prod.contains("SOFT_MS") && !prod.contains("HARD_MS"),
            "SOFT_MS/HARD_MS must be deleted from scorer"
        );
        assert!(
            !prod.contains("score_latency"),
            "score_latency must be deleted"
        );
    }
}
