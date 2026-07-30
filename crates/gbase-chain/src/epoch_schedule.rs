//! Stateful epoch schedule for CRV4 reveal-round prediction (D22).
//!
//! Exact port of `bittensor_core::timelock::epoch_schedule` + the reveal-round
//! arithmetic from `generate_commit_v2` / Python `get_encrypted_commit_v2`,
//! with an **injectable** `now_unix_secs` so fixtures are deterministic.
//!
//! Source pin: RaoFoundation/subtensor @ e4ffa2e1325c6c7db618dbceaf396310a170990c
//! (`sdk/bittensor-core/src/timelock/{epoch_schedule,mod,constants}.rs`).

#![allow(clippy::module_name_repetitions)]

/// Upper bound for owner-set tempo (≈ 7 days at 12 s/block) — SDK `MAX_TEMPO`.
pub const MAX_TEMPO: u16 = 50_400;

/// `MAX_TEMPO` as `u64`.
pub const MAX_TEMPO_U64: u64 = MAX_TEMPO as u64;

/// Drand Quicknet genesis timestamp (Unix seconds).
pub const GENESIS_TIME: u64 = 1_692_803_367;

/// Drand Quicknet round period in seconds.
pub const DRAND_PERIOD: u64 = 3;

/// Extra blocks so the targeted pulse is already ingested on-chain.
pub const SECURITY_BLOCK_OFFSET: u64 = 3;

/// Extrinsic lands in the next block relative to the queried head.
pub const COMMIT_INCLUSION_BLOCK_OFFSET: u64 = 1;

/// Snapshot of on-chain epoch schedule state (SDK `EpochScheduleState`).
///
/// The seven lockfile schedule inputs plus `current_block`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochScheduleState {
    /// Block of the last completed epoch.
    pub last_epoch_block: u64,
    /// Owner-triggered pending epoch block (`0` if none).
    pub pending_epoch_at: u64,
    /// Monotonic subnet epoch counter.
    pub subnet_epoch_index: u64,
    /// Epoch duration in blocks.
    pub tempo: u16,
    /// Blocks since last step.
    pub blocks_since_last_step: u64,
    /// Chain head block number.
    pub current_block: u64,
}

/// Errors from reveal-block simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpochScheduleError {
    /// Simulation exceeded budget.
    BoundExceeded,
    /// Tempo is zero — subnet does not run epochs.
    TempoIsZero,
}

impl std::fmt::Display for EpochScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundExceeded => write!(f, "reveal block simulation exceeded budget"),
            Self::TempoIsZero => write!(f, "tempo is zero; subnet does not run epochs"),
        }
    }
}

impl std::error::Error for EpochScheduleError {}

/// Upper bound on blocks simulated when searching for the reveal block.
#[must_use]
pub fn max_simulation_blocks(reveal_period_epochs: u64) -> u64 {
    reveal_period_epochs
        .saturating_mul(MAX_TEMPO_U64)
        .saturating_add(MAX_TEMPO_U64)
}

/// Port of `run_coinbase.rs` should-run-epoch predicate.
#[must_use]
pub fn should_run_epoch(state: &EpochScheduleState, block: u64) -> bool {
    let tempo = state.tempo;
    if tempo == 0 {
        return false;
    }
    let pending = state.pending_epoch_at;
    if pending > 0 && block >= pending {
        return true;
    }
    if state.blocks_since_last_step > MAX_TEMPO_U64 {
        return true;
    }
    let blocks_since = block.saturating_sub(state.last_epoch_block);
    blocks_since >= u64::from(tempo)
}

/// Epoch index used by reveal (runs **before** `run_coinbase` in `block_step`).
#[must_use]
pub fn current_epoch_pre_run_coinbase(state: &EpochScheduleState, block: u64) -> u64 {
    let base = state.subnet_epoch_index;
    if should_run_epoch(state, block) {
        base.saturating_add(1)
    } else {
        base
    }
}

/// Simulate `run_coinbase` for one block.
#[must_use]
pub fn simulate_run_coinbase(state: &EpochScheduleState, block: u64) -> EpochScheduleState {
    let mut next = state.clone();
    next.blocks_since_last_step = next.blocks_since_last_step.saturating_add(1);
    next.current_block = block;

    if should_run_epoch(&next, block) {
        next.last_epoch_block = block;
        next.pending_epoch_at = 0;
        next.subnet_epoch_index = next.subnet_epoch_index.saturating_add(1);
        next.blocks_since_last_step = 0;
    }
    next
}

/// Apply `simulate_run_coinbase` for each block in `start..=end`.
#[must_use]
pub fn advance_blocks(from: &EpochScheduleState, start: u64, end: u64) -> EpochScheduleState {
    let mut state = from.clone();
    if start > end {
        return state;
    }
    for b in start..=end {
        state = simulate_run_coinbase(&state, b);
    }
    state
}

/// Predict the first block at which the chain will reveal a commit submitted
/// against `head_state` (SDK `predict_first_reveal_block`).
///
/// # Errors
///
/// Tempo zero or simulation budget exceeded.
pub fn predict_first_reveal_block(
    head_state: &EpochScheduleState,
    reveal_period_epochs: u64,
) -> Result<u64, EpochScheduleError> {
    if head_state.tempo == 0 {
        return Err(EpochScheduleError::TempoIsZero);
    }

    let head_block = head_state.current_block;
    let extrinsic_block = head_block.saturating_add(COMMIT_INCLUSION_BLOCK_OFFSET);

    let post_before_extrinsic = if extrinsic_block == head_block.saturating_add(1) {
        head_state.clone()
    } else {
        advance_blocks(
            head_state,
            head_block.saturating_add(1),
            extrinsic_block.saturating_sub(1),
        )
    };

    let commit_epoch = current_epoch_pre_run_coinbase(&post_before_extrinsic, extrinsic_block);
    let target_epoch = commit_epoch.saturating_add(reveal_period_epochs);
    let max_sim = max_simulation_blocks(reveal_period_epochs);

    let mut post_prev = post_before_extrinsic;
    let end = extrinsic_block.saturating_add(max_sim);
    for r in extrinsic_block..=end {
        if current_epoch_pre_run_coinbase(&post_prev, r) == target_epoch {
            return Ok(r);
        }
        post_prev = simulate_run_coinbase(&post_prev, r);
    }

    Err(EpochScheduleError::BoundExceeded)
}

/// Compute the drand reveal round for a CRV4 commit (port of `generate_commit_v2`
/// round arithmetic with injectable wall clock).
///
/// `block_time_secs` is **seconds** (SDK / Python), not milliseconds.
///
/// # Errors
///
/// Propagates [`EpochScheduleError`] from reveal-block prediction.
pub fn compute_reveal_round(
    state: &EpochScheduleState,
    subnet_reveal_period_epochs: u64,
    block_time_secs: f64,
    now_unix_secs: f64,
) -> Result<u64, EpochScheduleError> {
    let first_reveal_blk = predict_first_reveal_block(state, subnet_reveal_period_epochs)?;
    let target_ingest_blk = first_reveal_blk.saturating_add(SECURITY_BLOCK_OFFSET);
    let blocks_until_ingest = target_ingest_blk.saturating_sub(state.current_block);
    #[allow(clippy::cast_precision_loss)]
    let secs_until_ingest = (blocks_until_ingest as f64) * block_time_secs;
    let target_secs = now_unix_secs + secs_until_ingest;
    // Matches generate_commit_v2: floor((target_secs - GENESIS_TIME) / DRAND_PERIOD).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let mut reveal_round =
        ((target_secs - (GENESIS_TIME as f64)) / (DRAND_PERIOD as f64)).floor() as u64;
    if reveal_round < 1 {
        reveal_round = 1;
    }
    Ok(reveal_round)
}

/// Gather schedule state from a [`crate::ChainClient`] (all seven lockfile inputs
/// + `current_block`). Tempo is truncated to `u16` (chain stores u16).
///
/// # Errors
///
/// Chain read failures, or tempo that does not fit in `u16`.
pub fn gather_schedule_state<C: crate::ChainClient + ?Sized>(
    chain: &C,
    netuid: u16,
) -> Result<EpochScheduleState, crate::ChainError> {
    let tempo_u64 = chain.tempo(netuid)?;
    let tempo = u16::try_from(tempo_u64).map_err(|_| {
        crate::ChainError::Other(format!("tempo {tempo_u64} exceeds u16"))
    })?;
    Ok(EpochScheduleState {
        last_epoch_block: chain.last_epoch_block(netuid)?,
        pending_epoch_at: chain.pending_epoch_at(netuid)?,
        subnet_epoch_index: chain.subnet_epoch_index(netuid)?,
        tempo,
        blocks_since_last_step: chain.blocks_since_last_step(netuid)?,
        current_block: chain.current_block()?,
    })
}

/// Convert chain `block_time()` milliseconds to seconds (`f64`) for the SDK formula.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn block_time_ms_to_secs(block_time_ms: u64) -> f64 {
    (block_time_ms as f64) / 1000.0
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Fixed wall clock for golden reveal-round vectors (not wall time).
    const FIXED_NOW: f64 = 1_700_000_000.0;

    #[test]
    fn s3_predict_cycle_reset_matches_sdk_vector() {
        let state = EpochScheduleState {
            last_epoch_block: 10,
            pending_epoch_at: 0,
            subnet_epoch_index: 0,
            tempo: 50,
            blocks_since_last_step: 0,
            current_block: 10,
        };
        assert_eq!(predict_first_reveal_block(&state, 1).unwrap(), 60);
    }

    #[test]
    fn s3_predict_pending_fires_matches_sdk_vector() {
        let state = EpochScheduleState {
            last_epoch_block: 80,
            pending_epoch_at: 95,
            subnet_epoch_index: 0,
            tempo: 20,
            blocks_since_last_step: 0,
            current_block: 91,
        };
        assert_eq!(predict_first_reveal_block(&state, 1).unwrap(), 95);
    }

    #[test]
    fn s3_predict_tempo_zero_errors() {
        let state = EpochScheduleState {
            last_epoch_block: 10,
            pending_epoch_at: 0,
            subnet_epoch_index: 0,
            tempo: 0,
            blocks_since_last_step: 0,
            current_block: 10,
        };
        assert_eq!(
            predict_first_reveal_block(&state, 1),
            Err(EpochScheduleError::TempoIsZero)
        );
    }

    /// Golden vectors: port of `generate_commit_v2` round math with `FIXED_NOW`.
    /// Produced by independent Python port of the same formula (task-33 evidence).
    #[test]
    fn s3_reveal_round_matches_python_get_encrypted_commit_v2_formula_three_fixtures() {
        let cases = [
            // name, state, rpe, block_time_secs, expected_round
            (
                "cycle_reset",
                EpochScheduleState {
                    last_epoch_block: 10,
                    pending_epoch_at: 0,
                    subnet_epoch_index: 0,
                    tempo: 50,
                    blocks_since_last_step: 0,
                    current_block: 10,
                },
                1u64,
                12.0_f64,
                2_399_089_u64,
            ),
            (
                "pending_fires",
                EpochScheduleState {
                    last_epoch_block: 80,
                    pending_epoch_at: 95,
                    subnet_epoch_index: 0,
                    tempo: 20,
                    blocks_since_last_step: 0,
                    current_block: 91,
                },
                1,
                12.0,
                2_398_905,
            ),
            (
                "mid_tempo_fake_defaults",
                EpochScheduleState {
                    last_epoch_block: 1000,
                    pending_epoch_at: 0,
                    subnet_epoch_index: 42,
                    tempo: 360,
                    blocks_since_last_step: 10,
                    current_block: 1010,
                },
                1,
                12.0,
                2_400_289,
            ),
        ];
        for (name, state, rpe, bt, expected) in cases {
            let got = compute_reveal_round(&state, rpe, bt, FIXED_NOW)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(got, expected, "fixture {name}");
        }
    }

    #[test]
    fn s3_block_time_ms_to_secs() {
        assert!((block_time_ms_to_secs(12_000) - 12.0).abs() < f64::EPSILON);
    }
}
