//! Deterministic simulated cluster backend (no IB / no GPU timing claims).

use std::collections::HashSet;

use sha2::{Digest, Sha256};

use crate::backend::ClusterBackend;
use crate::error::ClusterError;
use crate::types::{
    CheckpointHash, ExclusiveSlot, PKeyId, SegmentConfig, SegmentResult, SegmentTelemetry, Topology,
};

/// Base wallclock floor in ms before fingerprint-derived spread (sim only).
const SIM_BASE_WALLCLOCK_MS: u64 = 1_000;
/// Fingerprint-derived spread range in ms (exclusive upper bound of added term).
const SIM_SPREAD_MS: u64 = 50_000;
/// Tokens-per-step used only for fake telemetry step counts.
const SIM_TOKENS_PER_STEP: u64 = 1_000_000;

/// Software cluster: deterministic wallclock from code fingerprint + optional noise.
///
/// Exposes `PKey` partition ids on the API surface without real InfiniBand ops.
#[derive(Debug, Default)]
pub struct SimBackend {
    allocated: HashSet<PKeyId>,
    next_handle: u64,
}

impl SimBackend {
    /// Create an empty sim backend (no slots allocated).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Release a previously allocated exclusive slot (test / teardown helper).
    pub fn release_slot(&mut self, pkey_id: PKeyId) {
        self.allocated.remove(&pkey_id);
    }

    fn next_handle_id(&mut self) -> u64 {
        let id = self.next_handle;
        self.next_handle = self.next_handle.saturating_add(1);
        id
    }
}

impl ClusterBackend for SimBackend {
    fn check_topology_mirror(
        &self,
        master: Topology,
        slot: Topology,
    ) -> Result<(), ClusterError> {
        if master == slot {
            Ok(())
        } else {
            Err(ClusterError::TopologyMismatch { master, slot })
        }
    }

    fn allocate_exclusive_slot(&mut self, pkey_id: PKeyId) -> Result<ExclusiveSlot, ClusterError> {
        if !self.allocated.insert(pkey_id) {
            return Err(ClusterError::SlotBusy { pkey_id });
        }
        Ok(ExclusiveSlot {
            pkey_id,
            handle: self.next_handle_id(),
        })
    }

    fn run_segment(&mut self, cfg: &SegmentConfig) -> Result<SegmentResult, ClusterError> {
        if cfg.budget_tokens == 0 {
            return Err(ClusterError::InvalidConfig(
                "budget_tokens must be > 0".into(),
            ));
        }
        self.check_topology_mirror(cfg.master_topology, cfg.slot_topology)?;

        let slot = if self.allocated.contains(&cfg.pkey_id) {
            // Re-use occupancy; handle is synthetic for telemetry only.
            ExclusiveSlot {
                pkey_id: cfg.pkey_id,
                handle: self.next_handle.saturating_sub(1),
            }
        } else {
            self.allocate_exclusive_slot(cfg.pkey_id)?
        };

        let wallclock_ms = sim_wallclock_ms(cfg);
        let checkpoint_hash = fake_checkpoint_hash(cfg);
        let steps = cfg
            .budget_tokens
            .div_ceil(SIM_TOKENS_PER_STEP.max(1));

        Ok(SegmentResult {
            wallclock_ms,
            checkpoint_hash,
            telemetry: SegmentTelemetry {
                tokens_processed: cfg.budget_tokens,
                steps,
                backend: "sim",
                pkey_id: slot.pkey_id,
                slot_handle: slot.handle,
            },
            seeds: cfg.seeds.clone(),
        })
    }
}

/// Deterministic wallclock: base + fingerprint mix + optional noise term.
///
/// Noise is derived from `(fingerprint ‖ noise_ms ‖ seeds)` so it is stable
/// for a given config (not wall-clock RNG). `noise_ms == 0` disables the term.
fn sim_wallclock_ms(cfg: &SegmentConfig) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(b"hypertraining-sim-wallclock-v1");
    hasher.update(cfg.code_fingerprint);
    hasher.update(cfg.budget_tokens.to_le_bytes());
    hasher.update(cfg.seeds.run_seed.to_le_bytes());
    hasher.update(cfg.seeds.aux_seed.to_le_bytes());
    let digest = hasher.finalize();
    let mix = u64::from_le_bytes(digest[0..8].try_into().unwrap_or([0; 8]));
    let spread = mix % SIM_SPREAD_MS;
    let noise = if cfg.noise_ms == 0 {
        0
    } else {
        let mut nh = Sha256::new();
        nh.update(b"hypertraining-sim-noise-v1");
        nh.update(cfg.code_fingerprint);
        nh.update(cfg.noise_ms.to_le_bytes());
        nh.update(cfg.seeds.run_seed.to_le_bytes());
        let nd = nh.finalize();
        let n = u64::from_le_bytes(nd[0..8].try_into().unwrap_or([0; 8]));
        n % u64::from(cfg.noise_ms).saturating_add(1)
    };
    SIM_BASE_WALLCLOCK_MS
        .saturating_add(spread)
        .saturating_add(noise)
}

/// Fake checkpoint hash from fingerprint + seeds + budget (not a real model ckpt).
fn fake_checkpoint_hash(cfg: &SegmentConfig) -> CheckpointHash {
    let mut hasher = Sha256::new();
    hasher.update(b"hypertraining-sim-checkpoint-v1");
    hasher.update(cfg.code_fingerprint);
    hasher.update(cfg.budget_tokens.to_le_bytes());
    hasher.update(cfg.seeds.run_seed.to_le_bytes());
    hasher.update(cfg.seeds.aux_seed.to_le_bytes());
    hasher.update(cfg.master_topology.tp.to_le_bytes());
    hasher.update(cfg.master_topology.pp.to_le_bytes());
    hasher.update(cfg.master_topology.ep.to_le_bytes());
    hasher.update(cfg.master_topology.cp.to_le_bytes());
    hasher.finalize().into()
}
