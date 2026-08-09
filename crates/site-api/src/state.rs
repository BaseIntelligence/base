//! Shared axum state for `/v1/site/*`.

use std::sync::Arc;

use gateway_registry::Registry;
use site_data::price::TaoPriceCache;
use trustroot::ChallengesBody;
use weights_api::SealRecord;

/// Chain handle used for honest network figures (block/epoch/hotkeys).
pub type SharedChain = Arc<dyn chain::ChainClient + Send + Sync>;

/// Latest sealed bundle bytes + seal record, when a seal exists. A closure so
/// the gateway can adapt its `BundleStore` without a crate cycle.
pub type LatestSeal = Arc<dyn Fn() -> Option<(Vec<u8>, SealRecord)> + Send + Sync>;

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
    /// Owner-signed trust root: configured per-challenge emission shares.
    pub challenges: Option<Arc<ChallengesBody>>,
    /// Latest sealed epoch bundle (bytes + seal record) when present.
    pub latest_seal: Option<LatestSeal>,
    /// TAO/USD cache (TTL in `site_data::price`); shared across handlers.
    pub tao_price: Arc<TaoPriceCache>,
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
            challenges: None,
            latest_seal: None,
            tao_price: Arc::new(TaoPriceCache::new(None)),
        }
    }

    /// Attach the trust root + sealed-bundle provider (gateway deployment).
    #[must_use]
    pub fn with_weights(
        mut self,
        challenges: Arc<ChallengesBody>,
        latest_seal: LatestSeal,
    ) -> Self {
        self.challenges = Some(challenges);
        self.latest_seal = Some(latest_seal);
        self
    }

    /// Trust root as a reference (site-data input).
    #[must_use]
    pub fn trust_root(&self) -> Option<&ChallengesBody> {
        self.challenges.as_deref()
    }

    /// Latest sealed bundle snapshot (site-data input).
    #[must_use]
    pub fn latest_sealed_bundle(&self) -> Option<(Vec<u8>, SealRecord)> {
        self.latest_seal.as_ref().and_then(|f| f())
    }
}
