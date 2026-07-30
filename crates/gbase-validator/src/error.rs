//! Validator error types.

use thiserror::Error;

/// Failures while building or running the validator skeleton.
#[derive(Debug, Error)]
pub enum ValidatorError {
    /// Configuration / validation failure.
    #[error("config: {0}")]
    Config(String),
    /// Telemetry install or router build failed.
    #[error("telemetry: {0}")]
    Telemetry(#[from] gbase_telemetry::TelemetryError),
    /// Database connect / migrate failure.
    #[error("database: {0}")]
    Database(String),
    /// HTTP bind / serve failure.
    #[error("serve: {0}")]
    Serve(String),
    /// Coordination (gateway) client error.
    #[error("coordination: {0}")]
    Coordination(String),
    /// Chain read failure.
    #[error("chain: {0}")]
    Chain(String),
}
