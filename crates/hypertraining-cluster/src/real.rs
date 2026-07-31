//! Real B300 cluster backend stub — always [`ClusterError::NotConfigured`].

use crate::backend::ClusterBackend;
use crate::error::ClusterError;
use crate::types::{ExclusiveSlot, PKeyId, SegmentConfig, SegmentResult, Topology};

/// Hardware path placeholder until owner runs the B300 enablement runbook.
///
/// Every method returns [`ClusterError::NotConfigured`]. Must NOT produce
/// fabricated GPU wallclock numbers.
#[derive(Debug, Default, Clone, Copy)]
pub struct RealBackend;

impl RealBackend {
    /// Construct the deferred real backend stub.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl ClusterBackend for RealBackend {
    fn check_topology_mirror(
        &self,
        _master: Topology,
        _slot: Topology,
    ) -> Result<(), ClusterError> {
        Err(ClusterError::NotConfigured)
    }

    fn allocate_exclusive_slot(&mut self, _pkey_id: PKeyId) -> Result<ExclusiveSlot, ClusterError> {
        Err(ClusterError::NotConfigured)
    }

    fn run_segment(&mut self, _cfg: &SegmentConfig) -> Result<SegmentResult, ClusterError> {
        Err(ClusterError::NotConfigured)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SegmentSeeds, Topology};

    fn sample_cfg() -> SegmentConfig {
        let topo = Topology::new(4, 2, 4, 1);
        SegmentConfig {
            code_fingerprint: [1u8; 32],
            budget_tokens: 1_000,
            seeds: SegmentSeeds {
                run_seed: 1,
                aux_seed: 2,
            },
            master_topology: topo,
            slot_topology: topo,
            pkey_id: 1,
            noise_ms: 0,
        }
    }

    #[test]
    fn real_run_segment_returns_not_configured() {
        let mut backend = RealBackend::new();
        let err = backend
            .run_segment(&sample_cfg())
            .expect_err("real deferred");
        assert_eq!(err, ClusterError::NotConfigured);
        let msg = err.to_string();
        assert!(
            msg.contains("B300") && msg.contains("deferred"),
            "message must mention B300 deferred, got: {msg}"
        );
        assert!(
            msg.contains("SimBackend") || msg.contains("not configured"),
            "message must be actionable, got: {msg}"
        );
    }

    #[test]
    fn real_allocate_and_topology_not_configured() {
        let mut backend = RealBackend::new();
        assert_eq!(
            backend.allocate_exclusive_slot(1),
            Err(ClusterError::NotConfigured)
        );
        assert_eq!(
            backend.check_topology_mirror(Topology::new(1, 1, 1, 1), Topology::new(1, 1, 1, 1)),
            Err(ClusterError::NotConfigured)
        );
    }

    #[test]
    fn real_does_not_fabricate_wallclock() {
        let mut backend = RealBackend::new();
        // Only Ok path would yield wallclock; Real must never Ok.
        assert!(backend.run_segment(&sample_cfg()).is_err());
    }
}
