//! Orchestrator ↔ runner JSON envelopes (HTTPS hop).
//!
//! Protocol label: [`DISPATCH_PROTOCOL`]. Receipt cryptography lives in
//! [`crate::receipt`]; this module only carries the JSON surface.

use serde::{Deserialize, Serialize};

/// JSON protocol id for dispatch descriptor / result envelopes.
pub const DISPATCH_PROTOCOL: &str = "gbase-agent-dispatch-v1";

/// Orchestrator → runner task descriptor (stripped pack; no solution/tests).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskDescriptorV1 {
    /// Must be [`DISPATCH_PROTOCOL`].
    pub protocol: String,
    /// Challenge id string (`agent-v1`).
    pub challenge_id: String,
    /// Scoring version bound into the receipt.
    pub scoring_version: u16,
    /// Epoch index.
    pub epoch: u64,
    /// Miner hotkey as 64 lowercase hex chars.
    pub miner_hotkey_hex: String,
    /// Pack id the runner must load.
    pub pack_id: String,
    /// Absolute deadline (unix ms); runner SHOULD stop after.
    pub deadline_unix_ms: u64,
}

/// Runner outcome status for a dispatched pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatusV1 {
    /// Pack finished; `model_patch` may be present.
    Completed,
    /// Hard deadline exceeded.
    TimedOut,
    /// Runner / environment failure.
    Failed,
}

/// Runner → orchestrator result envelope (`model.patch` + signed receipt).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskResultV1 {
    /// Must be [`DISPATCH_PROTOCOL`].
    pub protocol: String,
    /// Challenge id echoed from the descriptor.
    pub challenge_id: String,
    /// Scoring version echoed.
    pub scoring_version: u16,
    /// Epoch echoed.
    pub epoch: u64,
    /// Miner hotkey hex echoed.
    pub miner_hotkey_hex: String,
    /// Pack id echoed.
    pub pack_id: String,
    /// Terminal status.
    pub status: TaskStatusV1,
    /// Unified diff text when produced; omitted/empty on timeout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_patch: Option<String>,
    /// Hex of `sha256(model.patch bytes)`; zero digest when no patch.
    pub patch_sha256_hex: String,
    /// Hex of the 64-byte work-receipt signature.
    pub receipt_sig_hex: String,
}

impl TaskDescriptorV1 {
    /// Build a descriptor with the canonical protocol label.
    #[must_use]
    pub fn new(
        challenge_id: impl Into<String>,
        scoring_version: u16,
        epoch: u64,
        miner_hotkey_hex: impl Into<String>,
        pack_id: impl Into<String>,
        deadline_unix_ms: u64,
    ) -> Self {
        Self {
            protocol: DISPATCH_PROTOCOL.into(),
            challenge_id: challenge_id.into(),
            scoring_version,
            epoch,
            miner_hotkey_hex: miner_hotkey_hex.into(),
            pack_id: pack_id.into(),
            deadline_unix_ms,
        }
    }
}
