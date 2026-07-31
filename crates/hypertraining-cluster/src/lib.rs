//! Hypertraining cluster measurement backends.
//!
//! - [`ClusterBackend`] — exclusive slot, topology mirror, segment run
//! - [`SimBackend`] — deterministic wallclock from code fingerprint (+ optional noise);
//!   fake checkpoint hash; `PKey` ids on the API without real InfiniBand
//! - [`RealBackend`] — stub returning [`ClusterError::NotConfigured`] (B300 deferred)
//!
//! Must NOT claim live GPU / B300 timing from the sim path.

#![forbid(unsafe_code)]

mod backend;
mod error;
mod real;
mod sim;
mod types;

pub use backend::ClusterBackend;
pub use error::ClusterError;
pub use real::RealBackend;
pub use sim::SimBackend;
pub use types::{
    CheckpointHash, ExclusiveSlot, MmaFamily, PKeyId, SegmentConfig, SegmentResult, SegmentSeeds,
    SegmentTelemetry, Topology,
};

/// Crate identity for smoke / wiring checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "hypertraining-cluster"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_is_hypertraining_cluster() {
        assert_eq!(crate_name(), "hypertraining-cluster");
    }

    #[test]
    fn public_api_exports_backends() {
        let _sim = SimBackend::new();
        let _real = RealBackend::new();
        let _topo = Topology::new(4, 2, 4, 1);
    }
}
