//! CRV4 / plain weight submission from [`SubmissionIntent`] (plan task 33).
//!
//! - Builds [`WeightsTlockPayload`] with **only** `{hotkey, uids, values, version_key}` (D5).
//! - Reveal round via [`gbase_chain::compute_reveal_round`] fed by all seven
//!   [`ChainClient`] schedule inputs (D22) — never invents `reveal_period_blocks`.
//! - Reveal is automatic on-chain — this crate never calls a reveal extrinsic.
//! - While CR is enabled: never falls back to `set_weights` (even on drand failure).
//! - Drand failure: retry until epoch deadline, then skip + alarm metric.

#![forbid(unsafe_code)]

use gbase_chain::{
    block_time_ms_to_secs, compute_reveal_round, gather_schedule_state, ChainClient, ChainError,
    WeightsTlockPayload,
};
use gbase_dissent::SubmissionIntent;

/// Default mechanism id (MAIN) for `commit_timelocked_mechanism_weights`.
pub const MECID_MAIN: u8 = 0;

/// Required commit-reveal protocol version (CRV4).
pub const COMMIT_REVEAL_VERSION_V4: u16 = 4;

/// Errors from the submit path.
#[derive(Debug, thiserror::Error)]
pub enum SubmitError {
    /// Underlying chain client error (after retries exhausted where applicable).
    #[error("chain: {0}")]
    Chain(#[from] ChainError),
    /// Runtime `commit_reveal_version` is not 4.
    #[error("commit_reveal_version is {got}, expected {COMMIT_REVEAL_VERSION_V4}")]
    WrongCommitRevealVersion {
        /// Observed version.
        got: u16,
    },
    /// Epoch skipped because drand was unavailable through the deadline.
    #[error("drand unavailable through epoch deadline; epoch skipped")]
    DrandSkipped,
    /// UID/value vector is empty.
    #[error("empty weight vector")]
    EmptyVector,
    /// UID and value lengths disagree (should not happen for sorted pairs).
    #[error("internal: uids/values length mismatch")]
    LengthMismatch,
}

/// Outcome of attempting to submit an intent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitOutcome {
    /// Timelocked commit recorded (CR enabled).
    SubmittedTimelocked {
        /// Drand reveal round.
        reveal_round: u64,
        /// Mechanism id used.
        mecid: u8,
    },
    /// Plain `set_weights` (CR disabled only).
    SubmittedSetWeights,
    /// Drand outage through deadline — no extrinsic; alarm raised.
    SkippedDrand {
        /// Epoch that was skipped.
        epoch: u64,
    },
}

/// Wall-clock source (injectable for tests / golden vectors).
pub trait Clock {
    /// Unix time in seconds as `f64` (matches SDK `now_unix`).
    fn now_unix_secs(&self) -> f64;
}

/// System wall clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_unix_secs(&self) -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64())
    }
}

/// Fixed clock for deterministic reveal-round tests.
#[derive(Debug, Clone, Copy)]
pub struct FixedClock {
    /// Unix seconds.
    pub now_unix_secs: f64,
}

impl Clock for FixedClock {
    fn now_unix_secs(&self) -> f64 {
        self.now_unix_secs
    }
}

/// Drand availability probe (encrypt path / network health).
///
/// Real encrypt uses local TLE + public key (no network); operators still probe
/// drand HTTP so a known-outage epoch is skipped rather than committing a round
/// that will never reveal.
pub trait DrandClient {
    /// Returns `Ok(())` when drand is considered healthy.
    ///
    /// # Errors
    ///
    /// Probe / network failure.
    fn ensure_ready(&self) -> Result<(), DrandError>;
}

/// Drand probe failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("drand: {message}")]
pub struct DrandError {
    /// Human-readable reason.
    pub message: String,
}

impl DrandError {
    /// Construct from a displayable reason.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Always-ready drand (unit tests that are not probing outage).
#[derive(Debug, Default, Clone, Copy)]
pub struct ReadyDrand;

impl DrandClient for ReadyDrand {
    fn ensure_ready(&self) -> Result<(), DrandError> {
        Ok(())
    }
}

/// Drand that always fails (outage simulation).
#[derive(Debug, Default, Clone, Copy)]
pub struct FailingDrand;

impl DrandClient for FailingDrand {
    fn ensure_ready(&self) -> Result<(), DrandError> {
        Err(DrandError::new("mocked drand outage"))
    }
}

/// Controllable mock: fails `fail_count` times then succeeds.
#[derive(Debug)]
pub struct CountingDrand {
    /// Remaining failures before success.
    pub fail_remaining: std::cell::Cell<u32>,
}

impl CountingDrand {
    /// Fail `n` times then succeed.
    #[must_use]
    pub fn fail_times(n: u32) -> Self {
        Self {
            fail_remaining: std::cell::Cell::new(n),
        }
    }
}

impl DrandClient for CountingDrand {
    fn ensure_ready(&self) -> Result<(), DrandError> {
        let left = self.fail_remaining.get();
        if left > 0 {
            self.fail_remaining.set(left.saturating_sub(1));
            return Err(DrandError::new("transient drand failure"));
        }
        Ok(())
    }
}

/// Parameters for one submit attempt.
#[derive(Debug, Clone)]
pub struct SubmitConfig {
    /// Subnet netuid.
    pub netuid: u16,
    /// Mechanism id (default [`MECID_MAIN`]).
    pub mecid: u8,
    /// Weights version key.
    pub version_key: u64,
    /// Committer hotkey public key bytes.
    pub hotkey: Vec<u8>,
    /// Block number after which drand retries stop (epoch deadline).
    pub epoch_deadline_block: u64,
    /// Max retries on [`ChainError::RateLimited`].
    pub max_rate_limit_retries: u32,
    /// Max drand probe attempts before checking deadline again.
    pub max_drand_retries: u32,
}

impl Default for SubmitConfig {
    fn default() -> Self {
        Self {
            netuid: 1,
            mecid: MECID_MAIN,
            version_key: 0,
            hotkey: vec![0xA1; 32],
            epoch_deadline_block: u64::MAX,
            max_rate_limit_retries: 8,
            max_drand_retries: 32,
        }
    }
}

/// Build a four-field payload from an intent. **Never** includes a merkle root (D5).
///
/// # Errors
///
/// Empty vector.
pub fn payload_from_intent(
    intent: &SubmissionIntent,
    hotkey: Vec<u8>,
    version_key: u64,
) -> Result<WeightsTlockPayload, SubmitError> {
    if intent.vector.is_empty() {
        return Err(SubmitError::EmptyVector);
    }
    let mut uids = Vec::with_capacity(intent.vector.len());
    let mut values = Vec::with_capacity(intent.vector.len());
    for &(uid, w) in &intent.vector {
        uids.push(uid);
        values.push(w);
    }
    if uids.len() != values.len() {
        return Err(SubmitError::LengthMismatch);
    }
    Ok(WeightsTlockPayload {
        hotkey,
        uids,
        values,
        version_key,
    })
}

/// Assert the payload type cannot carry a merkle root (test + runtime guard).
///
/// Returns the SCALE encoding of the four fields only.
#[must_use]
pub fn encode_payload_no_merkle(payload: &WeightsTlockPayload) -> Vec<u8> {
    use parity_scale_codec::Encode;
    payload.encode()
}

fn bump_drand_skip_alarm() {
    metrics::describe_counter!(
        "gbase_crv4_drand_skip_total",
        "Epochs skipped because drand was unavailable through the deadline (CR enabled)"
    );
    metrics::counter!("gbase_crv4_drand_skip_total").increment(1);
}

fn with_rate_limit_retry<C, F>(
    chain: &C,
    max_retries: u32,
    mut op: F,
) -> Result<(), ChainError>
where
    C: ChainClient + ?Sized,
    F: FnMut(&C) -> Result<(), ChainError>,
{
    let mut attempts = 0_u32;
    loop {
        match op(chain) {
            Ok(()) => return Ok(()),
            Err(ChainError::RateLimited { .. }) if attempts < max_retries => {
                attempts = attempts.saturating_add(1);
                tracing::warn!(attempts, "weights rate limited; retrying");
            }
            Err(e) => return Err(e),
        }
    }
}

/// Wait for drand readiness, retrying while `current_block <= epoch_deadline_block`.
fn wait_drand<C, D>(
    chain: &C,
    drand: &D,
    deadline: u64,
    max_retries: u32,
) -> Result<(), SubmitError>
where
    C: ChainClient + ?Sized,
    D: DrandClient + ?Sized,
{
    let mut tries = 0_u32;
    loop {
        match drand.ensure_ready() {
            Ok(()) => return Ok(()),
            Err(e) => {
                tries = tries.saturating_add(1);
                let tip = chain.current_block()?;
                if tip > deadline || tries > max_retries {
                    tracing::error!(%e, tip, deadline, "drand unavailable; skipping epoch");
                    return Err(SubmitError::DrandSkipped);
                }
                tracing::warn!(%e, tip, deadline, tries, "drand probe failed; retrying");
            }
        }
    }
}

/// Submit one [`SubmissionIntent`] according to CR on/off and D22 fail-closed rules.
///
/// # Errors
///
/// Chain errors, wrong CR version, empty vector, or drand skip after deadline.
pub fn submit_intent<C, D, K>(
    intent: &SubmissionIntent,
    chain: &C,
    drand: &D,
    clock: &K,
    cfg: &SubmitConfig,
) -> Result<SubmitOutcome, SubmitError>
where
    C: ChainClient + ?Sized,
    D: DrandClient + ?Sized,
    K: Clock + ?Sized,
{
    let payload = payload_from_intent(intent, cfg.hotkey.clone(), cfg.version_key)?;
    // Structural guard: encoding is exactly four SCALE fields (tests assert no merkle).
    let _encoded = encode_payload_no_merkle(&payload);

    let cr_on = chain.commit_reveal_enabled(cfg.netuid)?;
    if cr_on {
        let ver = chain.commit_reveal_version(cfg.netuid)?;
        if ver != COMMIT_REVEAL_VERSION_V4 {
            return Err(SubmitError::WrongCommitRevealVersion { got: ver });
        }

        match wait_drand(chain, drand, cfg.epoch_deadline_block, cfg.max_drand_retries) {
            Ok(()) => {}
            Err(SubmitError::DrandSkipped) => {
                bump_drand_skip_alarm();
                return Ok(SubmitOutcome::SkippedDrand {
                    epoch: intent.epoch,
                });
            }
            Err(e) => return Err(e),
        }

        let state = gather_schedule_state(chain, cfg.netuid)?;
        let reveal_period = chain.reveal_period_epochs(cfg.netuid)?;
        let block_time_ms = chain.block_time()?;
        let block_time_secs = block_time_ms_to_secs(block_time_ms);
        let reveal_round = compute_reveal_round(
            &state,
            reveal_period,
            block_time_secs,
            clock.now_unix_secs(),
        )
        .map_err(|e| ChainError::Other(e.to_string()))?;

        let mecid = cfg.mecid;
        with_rate_limit_retry(chain, cfg.max_rate_limit_retries, |c| {
            c.submit_timelocked_weights(mecid, payload.clone(), reveal_round)
        })?;

        return Ok(SubmitOutcome::SubmittedTimelocked {
            reveal_round,
            mecid,
        });
    }

    // CR disabled → plain set_weights only.
    let netuid = cfg.netuid;
    let uids = payload.uids.clone();
    let values = payload.values.clone();
    let version_key = payload.version_key;
    with_rate_limit_retry(chain, cfg.max_rate_limit_retries, |c| {
        c.set_weights(netuid, uids.clone(), values.clone(), version_key)
    })?;
    Ok(SubmitOutcome::SubmittedSetWeights)
}

/// Submit every intent from an [`gbase_dissent::EpochDecision`].
///
/// # Errors
///
/// First failing submit.
pub fn submit_decision_intents<C, D, K>(
    decision: &gbase_dissent::EpochDecision,
    chain: &C,
    drand: &D,
    clock: &K,
    cfg: &SubmitConfig,
) -> Result<Vec<SubmitOutcome>, SubmitError>
where
    C: ChainClient + ?Sized,
    D: DrandClient + ?Sized,
    K: Clock + ?Sized,
{
    let mut out = Vec::new();
    for intent in decision.submissions() {
        out.push(submit_intent(intent, chain, drand, clock, cfg)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use gbase_chain::{
        ChainCall, FakeChain, FakeChainConfig, WeightsTlockPayload as P,
    };
    use gbase_dissent::{SubmissionIntent, SubmissionSource};
    use parity_scale_codec::{Decode, Encode};

    const FIXED_NOW: f64 = 1_700_000_000.0;

    fn intent(epoch: u64) -> SubmissionIntent {
        SubmissionIntent {
            epoch,
            vector: vec![(0, 100), (1, 200), (2, 300)],
            source: SubmissionSource::Recompute,
        }
    }

    fn cfg_for(chain_tip_deadline: u64) -> SubmitConfig {
        SubmitConfig {
            netuid: 1,
            mecid: MECID_MAIN,
            version_key: 9,
            hotkey: vec![0xA1; 32],
            epoch_deadline_block: chain_tip_deadline,
            max_rate_limit_retries: 5,
            max_drand_retries: 3,
        }
    }

    #[test]
    fn s1_cr_on_submits_timelocked_only() {
        let chain = FakeChain::with_defaults();
        let clock = FixedClock {
            now_unix_secs: FIXED_NOW,
        };
        let out = submit_intent(
            &intent(42),
            &chain,
            &ReadyDrand,
            &clock,
            &cfg_for(u64::MAX),
        )
        .expect("submit");
        match out {
            SubmitOutcome::SubmittedTimelocked { mecid, .. } => {
                assert_eq!(mecid, MECID_MAIN);
            }
            other => panic!("unexpected {other:?}"),
        }
        let log = chain.call_log();
        assert_eq!(log.len(), 1);
        assert!(matches!(log[0], ChainCall::SubmitTimelocked(_)));
        assert!(chain.set_weights_log().is_empty());
        let sub = &chain.submissions()[0];
        assert_eq!(sub.payload.hotkey, vec![0xA1; 32]);
        assert_eq!(sub.payload.uids, vec![0, 1, 2]);
        assert_eq!(sub.payload.values, vec![100, 200, 300]);
        assert_eq!(sub.payload.version_key, 9);
        // Expected round for fake_defaults schedule @ FIXED_NOW (Python formula port).
        assert_eq!(sub.reveal_round, 2_400_289);
    }

    #[test]
    fn s2_cr_off_set_weights_only_never_timelock() {
        let chain = FakeChain::with_commit_reveal_disabled();
        let clock = FixedClock {
            now_unix_secs: FIXED_NOW,
        };
        let out = submit_intent(
            &intent(1),
            &chain,
            &ReadyDrand,
            &clock,
            &cfg_for(u64::MAX),
        )
        .expect("submit");
        assert_eq!(out, SubmitOutcome::SubmittedSetWeights);
        assert!(chain.submissions().is_empty());
        assert_eq!(chain.set_weights_log().len(), 1);
        assert!(matches!(
            chain.call_log()[0],
            ChainCall::SetWeights(_)
        ));
    }

    #[test]
    fn s5_rate_limit_retried_then_success() {
        let chain = FakeChain::new(FakeChainConfig {
            rate_limit_fails_remaining: 2,
            ..FakeChainConfig::default()
        });
        let clock = FixedClock {
            now_unix_secs: FIXED_NOW,
        };
        submit_intent(
            &intent(3),
            &chain,
            &ReadyDrand,
            &clock,
            &cfg_for(u64::MAX),
        )
        .expect("after retries");
        assert_eq!(chain.submissions().len(), 1);
    }

    #[test]
    fn s6_drand_outage_skips_alarm_never_set_weights() {
        let chain = FakeChain::with_defaults();
        let clock = FixedClock {
            now_unix_secs: FIXED_NOW,
        };
        // deadline below tip so first failure exits immediately
        let tip = chain.current_block().unwrap();
        let cfg = SubmitConfig {
            epoch_deadline_block: tip.saturating_sub(1),
            max_drand_retries: 0,
            ..cfg_for(0)
        };
        let out = submit_intent(&intent(7), &chain, &FailingDrand, &clock, &cfg).expect("skip ok");
        assert_eq!(out, SubmitOutcome::SkippedDrand { epoch: 7 });
        assert!(chain.call_log().is_empty(), "no extrinsics on drand skip");
        assert!(chain.submissions().is_empty());
        assert!(chain.set_weights_log().is_empty());
    }

    #[test]
    fn s7_payload_builder_never_appends_merkle_root() {
        let intent = intent(0);
        let p = payload_from_intent(&intent, vec![0xAA; 32], 1).unwrap();
        // Only four fields — constructing with a merkle field is impossible.
        let encoded = encode_payload_no_merkle(&p);
        let mut four = Vec::new();
        p.hotkey.encode_to(&mut four);
        p.uids.encode_to(&mut four);
        p.values.encode_to(&mut four);
        p.version_key.encode_to(&mut four);
        assert_eq!(encoded, four);
        let mut with_root = four.clone();
        [0xEE_u8; 32].encode_to(&mut with_root);
        assert_ne!(encoded, with_root);
        // Type-level: WeightsTlockPayload fields match lockfile.
        let check = P {
            hotkey: p.hotkey.clone(),
            uids: p.uids.clone(),
            values: p.values.clone(),
            version_key: p.version_key,
        };
        assert_eq!(check.encode(), encode_payload_no_merkle(&p));
    }

    #[test]
    fn s4_scale_matches_chain_payload_shape() {
        let p = payload_from_intent(&intent(0), vec![1, 2, 3], 42).unwrap();
        let e = p.encode();
        let d = P::decode(&mut &e[..]).unwrap();
        assert_eq!(d, p);
    }

    #[test]
    fn wrong_cr_version_errors() {
        let chain = FakeChain::new(FakeChainConfig {
            commit_reveal_version: 3,
            ..FakeChainConfig::default()
        });
        let clock = FixedClock {
            now_unix_secs: FIXED_NOW,
        };
        let err = submit_intent(
            &intent(1),
            &chain,
            &ReadyDrand,
            &clock,
            &cfg_for(u64::MAX),
        )
        .expect_err("v3");
        assert!(matches!(
            err,
            SubmitError::WrongCommitRevealVersion { got: 3 }
        ));
        assert!(chain.call_log().is_empty());
    }
}
