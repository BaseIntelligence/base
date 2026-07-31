//! Shared types for cluster backends (topology, segments, telemetry).

/// Parallelism topology that must mirror between master and tournament slot.
///
/// Normative fields from brief §4.2 / sealed surface: TP, PP, EP, CP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Topology {
    /// Tensor-parallel degree.
    pub tp: u32,
    /// Pipeline-parallel degree.
    pub pp: u32,
    /// Expert-parallel degree.
    pub ep: u32,
    /// Context-parallel degree.
    pub cp: u32,
}

impl Topology {
    /// Build a topology from the four parallelism axes.
    #[must_use]
    pub const fn new(tp: u32, pp: u32, ep: u32, cp: u32) -> Self {
        Self { tp, pp, ep, cp }
    }

    /// Total GPU count implied by the product of degrees (when axes multiply).
    #[must_use]
    pub const fn gpu_product(self) -> u64 {
        (self.tp as u64)
            .saturating_mul(self.pp as u64)
            .saturating_mul(self.ep as u64)
            .saturating_mul(self.cp as u64)
    }
}

/// Host-assigned InfiniBand partition key id (API surface; no real IB ops in sim).
pub type PKeyId = u16;

/// Opaque exclusive-slot handle returned after allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExclusiveSlot {
    /// Partition id bound to this exclusive placement.
    pub pkey_id: PKeyId,
    /// Stable handle id for the allocated créneau (sim-local counter).
    pub handle: u64,
}

/// RNG / data-order seeds sealed for a segment (identical across competitors).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentSeeds {
    /// Primary run seed (data order / dropout / etc.).
    pub run_seed: u64,
    /// Optional secondary seed (e.g. eval subsample).
    pub aux_seed: u64,
}

/// Validator-owned segment configuration (brief §5.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentConfig {
    /// Code fingerprint of the candidate (normalized binary / tree hash).
    pub code_fingerprint: [u8; 32],
    /// Fixed token budget `T_seg` (sealed).
    pub budget_tokens: u64,
    /// Seeds for the segment (shared across competitors).
    pub seeds: SegmentSeeds,
    /// Master topology that the slot must mirror.
    pub master_topology: Topology,
    /// Slot topology offered for this evaluation créneau.
    pub slot_topology: Topology,
    /// Host-assigned `PKey` partition for exclusive placement.
    pub pkey_id: PKeyId,
    /// Optional wallclock noise amplitude in milliseconds (sim only; 0 = pure).
    pub noise_ms: u32,
}

/// Fake / real checkpoint content digest (32 bytes).
pub type CheckpointHash = [u8; 32];

/// Tensor-core MMA instruction family observed in a segment (Guard 3 / Nsight later).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MmaFamily {
    /// IEEE-ish BF16 tensor cores (default sim / expected harness).
    #[default]
    Bf16,
    /// FP16 tensor cores.
    Fp16,
    /// TF32 path (silent precision downgrade when harness forbids TF32).
    Tf32,
    /// FP8 tensor cores.
    Fp8,
    /// No tensor-core MMA observed (unexpected for `MoE` training path).
    None,
}

/// Telemetry counters produced by a segment run (sim fixtures or real Nsight later).
///
/// Physics counters (`dram_bytes`, `tensor_ops`, `mma_family`, peak BW) feed Guard 3
/// (physical plausibility) in `hypertraining-eval`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SegmentTelemetry {
    /// Tokens processed (should equal budget on success).
    pub tokens_processed: u64,
    /// Simulated or measured step count.
    pub steps: u64,
    /// Backend label (`"sim"` or `"real"`).
    pub backend: &'static str,
    /// `PKey` partition id used for the run (API surface; not real IB in sim).
    pub pkey_id: PKeyId,
    /// Exclusive slot handle used for the run.
    pub slot_handle: u64,
    /// DRAM bytes moved (sim fixture or Nsight).
    pub dram_bytes: u64,
    /// Tensor-core operation count (sim fixture or Nsight).
    pub tensor_ops: u64,
    /// Dominant MMA instruction family observed.
    pub mma_family: MmaFamily,
    /// Peak DRAM bandwidth assumed for roofline (bytes/s); 0 = unset.
    pub peak_dram_bandwidth_bytes_per_s: u64,
}

/// Result of [`crate::ClusterBackend::run_segment`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentResult {
    /// Wall-clock duration in milliseconds (primary score input).
    pub wallclock_ms: u64,
    /// Checkpoint content hash (fake in sim).
    pub checkpoint_hash: CheckpointHash,
    /// Telemetry counters for guards / plausibility.
    pub telemetry: SegmentTelemetry,
    /// Echo of seeds used (audit).
    pub seeds: SegmentSeeds,
}
