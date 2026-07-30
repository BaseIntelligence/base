//! Peer discovery for bundle mirror fetch (task 30).
//!
//! Peers are configured as `(hotkey → base_url)` and filtered against the
//! current [`Metagraph`] from `FakeChain` / live chain so only on-metagraph
//! validators are contacted. Task 31 will tighten identity (signed roots /
//! `validator_permit`); this module only supplies HTTP endpoints for root fetch.

use gbase_chain::{ChainClient, Metagraph};

/// One validator peer that may re-serve verified bundles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerEndpoint {
    /// Neuron hotkey bytes (typically 32-byte sr25519).
    pub hotkey: Vec<u8>,
    /// HTTP base URL (no trailing slash), e.g. `http://127.0.0.1:9001`.
    pub base_url: String,
}

/// Configured peer book; discovery intersects with the metagraph hotkey set.
#[derive(Debug, Clone, Default)]
pub struct PeerBook {
    peers: Vec<PeerEndpoint>,
    /// Optional local hotkey — excluded from outbound peer fetch.
    own_hotkey: Option<Vec<u8>>,
}

impl PeerBook {
    /// Empty book.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from endpoints.
    #[must_use]
    pub fn from_peers(peers: Vec<PeerEndpoint>) -> Self {
        Self {
            peers,
            own_hotkey: None,
        }
    }

    /// Exclude this hotkey when listing peers (self).
    #[must_use]
    pub fn with_own_hotkey(mut self, hotkey: Vec<u8>) -> Self {
        self.own_hotkey = Some(hotkey);
        self
    }

    /// Register a peer.
    pub fn insert(&mut self, peer: PeerEndpoint) {
        self.peers.push(peer);
    }

    /// All configured peers (unfiltered).
    #[must_use]
    pub fn all(&self) -> &[PeerEndpoint] {
        &self.peers
    }

    /// Peers whose hotkey appears in `metagraph.hotkeys`, excluding self.
    #[must_use]
    pub fn discover(&self, metagraph: &Metagraph) -> Vec<PeerEndpoint> {
        self.peers
            .iter()
            .filter(|p| {
                if let Some(own) = &self.own_hotkey {
                    if &p.hotkey == own {
                        return false;
                    }
                }
                metagraph.hotkeys.iter().any(|h| h == &p.hotkey)
            })
            .cloned()
            .collect()
    }

    /// Discover peers from the chain tip metagraph.
    ///
    /// # Errors
    ///
    /// Propagates chain tip / metagraph read failures as strings.
    pub fn discover_from_chain<C: ChainClient>(
        &self,
        chain: &C,
    ) -> Result<Vec<PeerEndpoint>, String> {
        let tip = chain.current_block().map_err(|e| e.to_string())?;
        let hash = chain.block_hash(tip).map_err(|e| e.to_string())?;
        let mg = chain.metagraph_at(&hash).map_err(|e| e.to_string())?;
        Ok(self.discover(&mg))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use gbase_chain::{FakeChain, FakeChainConfig};

    #[test]
    fn discover_filters_to_metagraph_and_excludes_self() {
        let hk_a = vec![0xA1; 32];
        let hk_b = vec![0xB2; 32];
        let hk_c = vec![0xC3; 32];
        let hk_x = vec![0xFF; 32]; // not on metagraph

        let book = PeerBook::from_peers(vec![
            PeerEndpoint {
                hotkey: hk_a.clone(),
                base_url: "http://a".into(),
            },
            PeerEndpoint {
                hotkey: hk_b.clone(),
                base_url: "http://b".into(),
            },
            PeerEndpoint {
                hotkey: hk_x,
                base_url: "http://x".into(),
            },
        ])
        .with_own_hotkey(hk_a.clone());

        let chain = FakeChain::new(FakeChainConfig {
            hotkeys: vec![hk_a, hk_b.clone(), hk_c],
            ..FakeChainConfig::default()
        });
        let peers = book.discover_from_chain(&chain).unwrap();
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].hotkey, hk_b);
        assert_eq!(peers[0].base_url, "http://b");
    }
}
