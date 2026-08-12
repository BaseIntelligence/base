//! Error types for Lium client / eval backend.

use thiserror::Error;

/// Cost guardrail refusal (never billed).
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CostGuardrailError {
    /// Lifetime missing or non-positive.
    #[error("InstanceSpec.max_lifetime_hours must be a positive bound (hours)")]
    LifetimeMissing,
    /// Lifetime below Lium 1-hour granularity.
    #[error(
        "InstanceSpec.max_lifetime_hours must be at least 1 hour (Lium termination_hours granularity)"
    )]
    LifetimeBelowFloor,
    /// Price bound missing or non-positive.
    #[error("InstanceSpec.max_price_per_hour must be a positive bound")]
    PriceMissing,
    /// Selected offer exceeds max price.
    #[error("offer price {offer_price} exceeds max_price_per_hour {max_price}")]
    PriceExceeded { offer_price: f64, max_price: f64 },
    /// No offer matches GPU preference / caps.
    #[error("no Lium offer matches GPU preference and price caps (no_capacity)")]
    NoCapacity,
}

/// Lium / eval backend errors.
#[derive(Debug, Error)]
pub enum LiumError {
    /// Cost guardrail (typed).
    #[error(transparent)]
    Cost(#[from] CostGuardrailError),
    /// HTTP / API failure (message must never contain the API key).
    #[error("lium api: {0}")]
    Api(String),
    /// Transport failure.
    #[error("lium transport: {0}")]
    Transport(String),
    /// Remote exec / eval failure.
    #[error("lium exec: {0}")]
    Exec(String),
    /// Feature not available on this backend.
    #[error("not implemented: {0}")]
    NotImplemented(String),
    /// Integrity gate failed (digest, terminate, partial artifacts).
    #[error("integrity: {0}")]
    Integrity(String),
}
