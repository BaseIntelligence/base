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

mod client;
mod error;
mod receipt;
mod sim;
mod ssh;
mod types;

pub use client::LiumClient;
pub use error::{CostGuardrailError, LiumError};
pub use receipt::{EvalReceipt, NoScoreGate};
pub use sim::SimLiumBackend;
pub use ssh::{parse_ssh_target, resolve_private_key, SshTarget};
pub use types::{
    EvalTelemetry, GpuPreference, Instance, InstanceSpec, LiumSshConfig, Offer, RemoteExecResult,
    TelemetryPoint,
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
    async fn exec_eval(
        &self,
        instance_id: &str,
        architecture_py: &str,
        training_py: &str,
    ) -> Result<RemoteExecResult, LiumError>;
}

/// Default Lium API base URL.
pub const LIUM_API_BASE_URL: &str = "https://lium.io/api";

/// Floor for `max_lifetime_hours` (Lium `termination_hours` is 1-hour granularity).
pub const MIN_LIFETIME_HOURS: f64 = 1.0;

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
