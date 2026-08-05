//! `GET /v1/weights/latest` response shape served by the master gateway.
//!
//! This is a drop-in projection of the Python master's `MasterWeightsResponse`
//! (`src/base/schemas/weights.py`): identical field names, types and semantics,
//! plus the two Rust-only extras (`merkle_root`, `final_vector`) that the
//! validator already consumes, so the body stays a strict superset.
//!
//! Sealed fields are derived from the [`EpochBundleV1`] and the seal record
//! captured by the bundle store at seal time. When no sealed bundle is available
//! (or decode fails), the gateway serves [`build_burn_fallback`] — uid 0 at 100%
//! under `burn-uid0.v1` — rather than 404, so validators never see an empty
//! weights endpoint.
//!
//! HTTP serve applies [`refresh_serve_freshness`] so wall-clock fields stay
//! inside the Python client's pydantic / stale checks even when the underlying
//! seal is older than [`FRESHNESS_SECONDS`] (immutable seal content unchanged).

#![forbid(unsafe_code)]

mod vector;

pub use vector::{build_burn_fallback, build_latest, refresh_serve_freshness};

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};

/// Response `protocol_version` (master vector protocol, not the SCALE bundle version).
pub const PROTOCOL_VERSION: &str = "1.0";
/// Emission policy identifier emitted by the reference service.
pub const EMISSION_POLICY_VERSION: &str = "emission-shares.absolute.v1";
/// Burn policy identifier emitted by the reference service.
pub const BURN_POLICY_VERSION: &str = "burn-uid0.v1";
/// Hotkey→uid mapping policy identifier emitted by the reference service.
pub const MAPPING_POLICY_VERSION: &str = "hotkey-to-uid.v1";
/// UID that absorbs unallocated emission (`burn-uid0.v1`).
pub const BURN_UID: u16 = 0;
/// Vector time-to-live: `expires_at = computed_at + 720s`
/// (`MASTER_WEIGHTS_FRESHNESS_SECONDS`).
pub const FRESHNESS_SECONDS: u64 = 720;
/// Source outcome for a challenge whose leaves are in the sealed bundle.
pub const OUTCOME_ACCEPTED: &str = "accepted";
/// Source outcome for an expected challenge with no sealed leaves.
pub const OUTCOME_MISSING: &str = "missing";

/// Seal-time provenance recorded when a bundle is first persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealRecord {
    /// Wall-clock instant the bundle was persisted, in microseconds since the
    /// Unix epoch (`computed_at`).
    pub sealed_at_micros: u64,
    /// Number of accepted seals for this epoch; 1 for an immutable first seal.
    pub revision: u32,
}

impl SealRecord {
    /// Record with `sealed_at_micros` taken from the system clock now.
    #[must_use]
    pub fn now(revision: u32) -> Self {
        let micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX));
        Self {
            sealed_at_micros: micros,
            revision,
        }
    }
}

/// Insertion-ordered `str -> f64` object.
///
/// Float aggregation is order dependent, so the source ordering is preserved
/// instead of being re-sorted through a map type.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FloatMap(Vec<(String, f64)>);

impl FloatMap {
    /// Wrap ordered entries.
    #[must_use]
    pub fn new(entries: Vec<(String, f64)>) -> Self {
        Self(entries)
    }

    /// Entries in insertion order.
    #[must_use]
    pub fn entries(&self) -> &[(String, f64)] {
        &self.0
    }
}

impl Serialize for FloatMap {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in &self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

/// Per-challenge contribution summary (`ChallengeWeightsResult`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChallengeWeightsResult {
    /// Challenge slug.
    pub slug: String,
    /// Emission share as a percentage (0..100).
    pub emission_percent: f64,
    /// Always empty: the reference service does not republish miner weights here.
    pub weights: FloatMap,
    /// Whether the source contributed to the sealed vector.
    pub ok: bool,
    /// Reason code when `ok` is false.
    pub error: Option<String>,
}

/// Provenance reference to a durable raw-weight snapshot (`SourceRef`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceRef {
    /// Challenge slug.
    pub challenge_slug: String,
    /// Snapshot identity.
    pub snapshot_id: String,
    /// Hex digest of the snapshot payload.
    pub payload_digest: String,
    /// Snapshot outcome.
    pub outcome: String,
}

/// Versioned per-challenge source outcome recorded at seal (`SourceOutcome`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceOutcome {
    /// Challenge slug.
    pub challenge_slug: String,
    /// Outcome label.
    pub outcome: String,
    /// Machine-readable reason for `outcome`.
    pub reason_code: String,
    /// Snapshot identity when a snapshot exists.
    pub snapshot_id: Option<String>,
    /// Hex digest of the snapshot payload when a snapshot exists.
    pub payload_digest: Option<String>,
    /// Snapshot revision; absent when the source has no revisioned snapshot row.
    pub revision: Option<u32>,
}

/// Metagraph identity block mirrored into `metagraph_identity`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MetagraphIdentity {
    /// Metagraph root hex.
    pub hash: Option<String>,
    /// Block the metagraph was read at.
    pub block: Option<u64>,
    /// Number of hotkeys in the uid map.
    pub uid_count: usize,
    /// Burn uid under `burn-uid0.v1`.
    pub burn_uid: u16,
}

/// Latest sealed weights vector served to validators.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WeightsLatestResponse {
    /// Master vector protocol version.
    pub protocol_version: String,
    /// Stable vector identity derived from `vector_digest`.
    pub vector_id: Option<String>,
    /// Hex digest that uniquely identifies this vector.
    pub vector_digest: Option<String>,
    /// Sealed epoch.
    pub epoch: Option<u64>,
    /// Seal revision for this epoch.
    pub revision: u32,
    /// Subnet netuid.
    pub netuid: u16,
    /// Configured chain endpoint.
    pub chain_endpoint: String,
    /// UIDs in the final vector, ascending.
    pub uids: Vec<u16>,
    /// Normalised weights aligned with `uids`.
    pub weights: Vec<f64>,
    /// Miner hotkey (SS58) → normalised weight.
    pub hotkey_weights: FloatMap,
    /// Canonical JSON of the chain-domain payload the weights commit to.
    pub chain_domain_bytes: Option<String>,
    /// Instant the vector was sealed.
    pub computed_at: String,
    /// `computed_at + FRESHNESS_SECONDS`.
    pub expires_at: String,
    /// Per-challenge contribution summary.
    pub source_challenges: Vec<ChallengeWeightsResult>,
    /// Snapshots that fed the vector.
    pub source_snapshots: Vec<SourceRef>,
    /// Outcome for every expected source.
    pub source_outcomes: Vec<SourceOutcome>,
    /// Emission policy identifier.
    pub emission_policy_version: Option<String>,
    /// Challenge slug → absolute emission share (0..1).
    pub emission_shares: FloatMap,
    /// Burn policy identifier.
    pub burn_policy_version: Option<String>,
    /// Hotkey→uid mapping policy identifier.
    pub mapping_policy_version: Option<String>,
    /// Metagraph identity block.
    pub metagraph_identity: MetagraphIdentity,
    /// Metagraph root hex.
    pub metagraph_hash: Option<String>,
    /// Block the metagraph was read at.
    pub metagraph_block: Option<u64>,
    /// Whether the burn uid carries weight in this vector.
    pub burn_outcome: Option<bool>,
    /// Instant the metagraph snapshot was taken.
    pub metagraph_updated_at: String,
    /// Merkle root hex of the sealed bundle (Rust-only extra).
    pub merkle_root: String,
    /// Chain-scale `(uid, weight)` pairs (Rust-only extra).
    pub final_vector: Vec<(u16, u16)>,
    /// `true` when backed by a sealed [`EpochBundleV1`]; `false` for the
    /// fail-closed burn fallback (Rust-only extra).
    pub sealed: bool,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
mod tests;
