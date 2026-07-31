//! [`ClusterBackend`] trait — exclusive slot, topology mirror, segment run.

use crate::error::ClusterError;
use crate::types::{ExclusiveSlot, PKeyId, SegmentConfig, SegmentResult, Topology};

/// Validator-owned cluster measurement surface (brief §4–§5).
///
/// Implementations:
/// - [`crate::SimBackend`] — deterministic software path (CI / no B300)
/// - [`crate::RealBackend`] — stub until owner enables real hardware
pub trait ClusterBackend {
    /// Reject when `slot` does not exactly reproduce `master` (TP/PP/EP/CP).
    ///
    /// # Errors
    /// [`ClusterError::TopologyMismatch`] when any axis differs.
    /// [`ClusterError::NotConfigured`] on Real until B300 is enabled.
    fn check_topology_mirror(&self, master: Topology, slot: Topology) -> Result<(), ClusterError>;

    /// Allocate an exclusive tournament créneau bound to `pkey_id`.
    ///
    /// Sim tracks occupancy in-process. Real returns [`ClusterError::NotConfigured`].
    ///
    /// # Errors
    /// [`ClusterError::SlotBusy`] if the partition is already held.
    /// [`ClusterError::NotConfigured`] on Real until B300 is enabled.
    fn allocate_exclusive_slot(&mut self, pkey_id: PKeyId) -> Result<ExclusiveSlot, ClusterError>;

    /// Run one sealed segment and return wallclock + checkpoint + telemetry.
    ///
    /// Must call topology mirror check against `cfg.master_topology` / `cfg.slot_topology`
    /// before measuring. Must NOT claim real GPU timing from Sim.
    ///
    /// # Errors
    /// Topology mismatch, not configured, slot busy, or invalid config.
    fn run_segment(&mut self, cfg: &SegmentConfig) -> Result<SegmentResult, ClusterError>;
}
