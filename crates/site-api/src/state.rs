//! Shared axum state for `/v1/site/*`.

use std::sync::Arc;

use gateway_registry::Registry;

/// Chain handle used for honest network figures (block/epoch/hotkeys).
pub type SharedChain = Arc<dyn chain::ChainClient + Send + Sync>;

/// Site aggregator state (registry + HTTP client + optional chain).
#[derive(Clone)]
pub struct SiteState {
    /// Backend registry (design / prism).
    pub registry: Arc<Registry>,
    /// Outbound client for challenge backends.
    pub client: reqwest::Client,
    /// Optional chain for network/validator surfaces.
    pub chain: Option<SharedChain>,
    /// Netuid for chain reads.
    pub netuid: u16,
}

impl SiteState {
    /// Build from gateway pieces.
    #[must_use]
    pub fn new(
        registry: Arc<Registry>,
        client: reqwest::Client,
        chain: Option<SharedChain>,
        netuid: u16,
    ) -> Self {
        Self {
            registry,
            client,
            chain,
            netuid,
        }
    }
}
