//! Shared types for offers, specs, instances, exec results.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Rentable GPU offer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Offer {
    /// Provider offer / executor id.
    pub id: String,
    /// GPU type string from provider (e.g. `NVIDIA A100-SXM4-80GB`).
    pub gpu_type: String,
    /// GPUs on the offer.
    pub gpu_count: u32,
    /// Price per GPU-hour (USD).
    pub price_per_hour: f64,
    /// Provider label.
    pub provider: String,
}

/// Provision request with mandatory cost guardrails.
#[derive(Debug, Clone)]
pub struct InstanceSpec {
    /// Pod name (unique per rent).
    pub name: String,
    /// Max lifetime hours (≥ 1).
    pub max_lifetime_hours: f64,
    /// Max price per GPU-hour.
    pub max_price_per_hour: f64,
    /// GPU count requested.
    pub gpu_count: u32,
    /// Optional image / template digest pin (integrity).
    pub image_digest: Option<String>,
    /// SSH public keys (required for real Lium rent).
    pub ssh_public_keys: Vec<String>,
    /// Optional Lium SSH key name for `ensure_ssh_key`.
    pub ssh_key_name: Option<String>,
    /// Optional preferred offer id (skip cheapest-of-filter).
    pub preferred_offer_id: Option<String>,
    /// Optional Lium template id (required by rent API if no dockerfile).
    pub template_id: Option<String>,
    /// Optional template name to ensure/resolve (e.g. prism-mission-e2e).
    pub template_name: Option<String>,
}

impl Default for InstanceSpec {
    fn default() -> Self {
        Self {
            name: "prism-eval".into(),
            max_lifetime_hours: 1.0,
            max_price_per_hour: 2.5,
            gpu_count: 1,
            image_digest: None,
            ssh_public_keys: vec![],
            ssh_key_name: Some("prism-mission-worker".into()),
            preferred_offer_id: None,
            template_id: None,
            template_name: Some("prism-mission-e2e".into()),
        }
    }
}

/// Provisioned instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    /// Pod id.
    pub id: String,
    /// Status string (`RUNNING`, `CREATION_FAILED`, …).
    pub status: String,
    /// Provider label.
    pub provider: String,
    /// GPU type if known from pod payload.
    #[serde(default)]
    pub gpu_type: Option<String>,
    /// Raw `ssh_connect_cmd` when present (no secrets).
    #[serde(default)]
    pub ssh_connect_cmd: Option<String>,
}

/// Ordered GPU preference list (capability filter, not hard SKU lock).
#[derive(Debug, Clone, Default)]
pub struct GpuPreference {
    /// Substrings matched against `Offer.gpu_type` (case-insensitive), first wins.
    pub prefer: Vec<String>,
}

impl GpuPreference {
    /// Default preference: Blackwell → H100 → A100 (any match).
    #[must_use]
    pub fn default_prism() -> Self {
        Self {
            prefer: vec![
                "BLACKWELL".into(),
                "B200".into(),
                "H100".into(),
                "A100".into(),
            ],
        }
    }

    /// Rank offers: lower is better. Unmatched get large rank.
    #[must_use]
    pub fn rank(&self, gpu_type: &str) -> usize {
        let upper = gpu_type.to_ascii_uppercase();
        for (i, needle) in self.prefer.iter().enumerate() {
            if upper.contains(&needle.to_ascii_uppercase()) {
                return i;
            }
        }
        usize::MAX / 2
    }
}

/// Result of a remote (or sim) PRISM eval execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteExecResult {
    /// Finite quality metric used for scoring (higher is better after inversion if needed).
    /// PRISM prequential BPB: lower is better; we store raw bpb and map in score layer.
    pub bpb: f64,
    /// Optional tokens seen.
    pub tokens_seen: u64,
    /// Optional wall-clock seconds.
    pub wall_clock_seconds: f64,
    /// GPU type used (if known).
    pub gpu_type: Option<String>,
    /// Free-form notes (no secrets).
    pub notes: String,
}

/// Optional Real-client SSH configuration (paths only; never logs key material).
#[derive(Debug, Clone, Default)]
pub struct LiumSshConfig {
    /// Path to ed25519 private key used for pod SSH.
    pub private_key_path: Option<PathBuf>,
    /// Max seconds waiting for RUNNING after rent.
    pub running_timeout_secs: u64,
    /// SSH attempts while pod networking settles.
    pub ssh_attempts: u32,
    /// Seconds between SSH retries.
    pub ssh_retry_secs: u64,
    /// Kitchen-timer training cap (hours) the harness enforces in-pod.
    pub train_hours_cap: f64,
}

impl LiumSshConfig {
    /// Defaults matching historical `live_lium_e2e` smoke.
    #[must_use]
    pub fn default_live() -> Self {
        Self {
            private_key_path: None, // resolve via LIUM_SSH_PRIVATE_KEY / default path
            running_timeout_secs: 300,
            ssh_attempts: 8,
            ssh_retry_secs: 5,
            train_hours_cap: prism_recipe::TRAIN_HOURS_CAP,
        }
    }
}
