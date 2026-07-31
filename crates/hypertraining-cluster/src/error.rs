//! Cluster backend errors.

use thiserror::Error;

use crate::types::{PKeyId, Topology};

/// Failures from [`crate::ClusterBackend`] operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ClusterError {
    /// Master topology does not match the evaluation slot (brief §4.2).
    #[error(
        "topology mirror mismatch: master {master:?} != slot {slot:?} \
         (evaluation slot must reproduce master TP/PP/EP/CP exactly)"
    )]
    TopologyMismatch {
        /// Topology of the master cluster A.
        master: Topology,
        /// Topology of the tournament slot on cluster B.
        slot: Topology,
    },
    /// Real / hardware path is not enabled (owner B300 deferred).
    #[error(
        "cluster backend not configured: RealBackend B300 path is deferred \
         (owner enablement runbook required; use SimBackend until then)"
    )]
    NotConfigured,
    /// Requested `PKey` partition is already held exclusively.
    #[error("exclusive slot already allocated for pkey_id={pkey_id}")]
    SlotBusy {
        /// Partition id that could not be allocated.
        pkey_id: PKeyId,
    },
    /// Invalid segment configuration.
    #[error("invalid segment config: {0}")]
    InvalidConfig(String),
}
