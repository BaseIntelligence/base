//! Lium GPU rental client for PRISM master-centralized eval.
//!
//! **No Phala CVM.** The challenge operator holds `LIUM_API_KEY` and rents
//! ephemeral pods per eval. Cost guardrails refuse unbounded lifetime / price
//! **before** any rent call. Every provision path terminates + verifies on failure.
//!
//! # Backends
//!
//! * [`LiumClient`] — real HTTPS to `https://lium.io/api` (`X-API-Key`) + SSH exec.
//! * [`SimLiumBackend`] — offline deterministic metrics for CI (no network).
//!
//! # Secrets
//!
//! The API key is never logged, never placed in `Debug`/`Display`, never sent to
//! the pod environment. SSH private key path only; key material never logged.

#![forbid(unsafe_code)]
// Pedantic noise matching hypertraining-eval / thin HTTP wrappers.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_fields_in_debug)] // api_key intentionally redacted
#![allow(clippy::too_many_lines)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::duration_suboptimal_units)]
#![allow(clippy::manual_clamp)]

mod artifacts;
mod client;
mod sim;
mod ssh;

pub use artifacts::harvest_checkpoint_ssh;
pub use client::LiumClient;
pub use prism_artifacts::{
    artifact_dir_for, artifact_root, checkpoint_path_for, write_sim_checkpoint,
    MAX_CHECKPOINT_BYTES, POD_WORKDIR,
};
pub use sim::SimLiumBackend;
pub use ssh::{parse_ssh_target, resolve_private_key, truncate_tail, SshTarget};
// The data contract lives in `prism-lium-types` (per-crate LOC cap); it is
// re-exported wholesale so `prism_lium::…` stays the single import path.
pub use prism_lium_types::{
    CostGuardrailError, EvalReceipt, EvalTelemetry, GpuPreference, Instance, InstanceSpec,
    LiumError, LiumSshConfig, NoScoreGate, Offer, ProbePoint, RemoteExecResult, TelemetryPoint,
};

use async_trait::async_trait;

/// Master-side eval job backend (Sim or Real Lium).
#[async_trait]
pub trait EvalJobBackend: Send + Sync {
    /// List rentable offers (filtered by max price when set).
    async fn list_offers(&self, max_price_per_hour: Option<f64>) -> Result<Vec<Offer>, LiumError>;

    /// Provision under cost guardrails; fail-closed cleanup on error.
    async fn provision(&self, spec: &InstanceSpec) -> Result<Instance, LiumError>;

    /// Terminate (idempotent).
    async fn terminate(&self, instance_id: &str) -> Result<(), LiumError>;

    /// True when the instance is absent from the provider.
    async fn verify_terminated(&self, instance_id: &str) -> Result<bool, LiumError>;

    /// Run a sealed PRISM eval payload on the instance and return metrics.
    ///
    /// Sim backends ignore `instance_id` and compute from payload bytes.
    /// Real backends wait RUNNING, SSH GPU-attest, then emit metrics.
    /// `tree_blob` is a packed `prism_tree::StagedTree` for v3 source-tree
    /// submissions (`None` for legacy two-script rows).
    async fn exec_eval(
        &self,
        instance_id: &str,
        architecture_py: &str,
        training_py: &str,
        tree_blob: Option<&[u8]>,
    ) -> Result<RemoteExecResult, LiumError>;

    /// Best-effort tail of the on-pod harness log (before terminate/reclaim).
    ///
    /// Default is empty — Sim has nothing to fetch. Live backends SSH
    /// `tail` of `/tmp/prism_eval/harness.log` so stuck-sweep / timeout
    /// paths retain the fatal end of a multi-hour train instead of a blank
    /// `swept: stuck beyond grace`.
    async fn harvest_logs(&self, _instance_id: &str) -> Result<String, LiumError> {
        Ok(String::new())
    }

    /// Pull trained weights from the pod into `dest_dir` **before** terminate.
    ///
    /// `n_params` is the harness-measured count (drives BF16×1.5 size budget).
    /// Default errors — callers that need artifacts must use a backend that
    /// implements harvest (live Lium or Sim stub). Fail-closed: missing
    /// checkpoint → `Err` (orchestrator may still score but must not claim
    /// a top-model weight publish).
    async fn harvest_artifacts(
        &self,
        _instance_id: &str,
        _dest_dir: &std::path::Path,
        _seed: &[u8],
        _n_params: Option<u64>,
    ) -> Result<std::path::PathBuf, LiumError> {
        Err(LiumError::Exec(
            "artifact harvest not supported on this backend".into(),
        ))
    }
}

/// Bytes retained when surfacing harness stderr / harvested logs into
/// `error_detail` / stage events. Prefer the **tail** (fatals land at the
/// end); a prior 4 KiB head cap ate inductor autotune spam and dropped the
/// real traceback (~4054 chars stored).
pub const HARNESS_LOG_RETAIN_BYTES: usize = 32_768;

/// Default Lium API base URL.
pub const LIUM_API_BASE_URL: &str = "https://lium.io/api";

/// Floor for `max_lifetime_hours` (Lium `termination_hours` is 1-hour granularity).
pub const MIN_LIFETIME_HOURS: f64 = 1.0;

/// Serializes tests that mutate `PRISM_EVAL_ASSETS_DIR` (client + sim).
#[cfg(test)]
pub(crate) static ASSETS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Crate identity smoke.
#[must_use]
pub fn crate_name() -> &'static str {
    "prism-lium"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_ok() {
        assert_eq!(crate_name(), "prism-lium");
        // Floor is a named constant; compare via binding so clippy does not
        // treat it as a pure constant assertion.
        let floor = MIN_LIFETIME_HOURS;
        assert!((floor - 1.0).abs() < f64::EPSILON || floor > 1.0);
    }
}
