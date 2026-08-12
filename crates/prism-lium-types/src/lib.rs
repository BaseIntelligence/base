//! `prism-lium-types` — the Lium eval data contract, split out of
//! `prism-lium` for the per-crate non-test LOC cap (the repo's standard
//! crate-split pattern, same as `prism-zoneb` for Zone B).
//!
//! A pure leaf: error taxonomy ([`LiumError`], [`CostGuardrailError`]),
//! provider shapes (offers, specs, instances, GPU preference, SSH config),
//! the telemetry/probe series a pod reports back, and the signed
//! [`EvalReceipt`] with its [`NoScoreGate`]. No HTTP, no SSH, no I/O — the
//! client and backends live in `prism-lium`, which re-exports every name
//! here so `prism_lium::…` paths keep working.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::doc_markdown)]

mod error;
mod receipt;
mod types;

pub use error::{CostGuardrailError, LiumError};
pub use receipt::{EvalReceipt, NoScoreGate};
pub use types::{
    effective_gpu_count, gpu_count_from_label, EvalTelemetry, GpuPreference, Instance,
    InstanceSpec, LiumSshConfig, Offer, ProbePoint, RemoteExecResult, TelemetryPoint,
};
