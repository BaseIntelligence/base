//! Registration state for the validator process.
//!
//! [`resolve_on_chain`] reads the real UID assignment from the metagraph;
//! [`RegistrationStub`] caches it so readiness and logs can report it without
//! a chain round trip per request. Registering a *new* hotkey (the
//! `burned_register` extrinsic) is an operator action, not something the
//! validator does on its own.

use chain::{ChainClient, ChainError};
use serde::Serialize;

/// Local registration view for the validator process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RegistrationStatus {
    /// Whether a UID has been assigned in this process.
    pub registered: bool,
    /// Assigned UID when registered.
    pub uid: Option<u16>,
}

impl RegistrationStatus {
    /// Fresh process: not yet registered on-chain (or not loaded).
    #[must_use]
    pub const fn unregistered() -> Self {
        Self {
            registered: false,
            uid: None,
        }
    }

    /// Mark this process as holding `uid` (stub for later chain sync).
    #[must_use]
    pub const fn with_uid(uid: u16) -> Self {
        Self {
            registered: true,
            uid: Some(uid),
        }
    }
}

/// Read this hotkey's UID assignment from the metagraph at the current tip.
///
/// A hotkey absent from the metagraph is unregistered, which is a normal
/// answer rather than an error; only chain access failures return `Err`.
///
/// # Errors
///
/// Tip, block-hash, or metagraph read failures.
pub fn resolve_on_chain(
    chain: &dyn ChainClient,
    hotkey: &[u8; 32],
) -> Result<RegistrationStatus, ChainError> {
    let tip = chain.current_block()?;
    let block_hash = chain.block_hash(tip)?;
    let metagraph = chain.metagraph_at(&block_hash)?;
    Ok(metagraph
        .hotkeys
        .iter()
        .position(|h| h.as_slice() == hotkey.as_slice())
        .and_then(|uid| u16::try_from(uid).ok())
        .map_or_else(
            RegistrationStatus::unregistered,
            RegistrationStatus::with_uid,
        ))
}

/// Mutable registration view held by the running validator.
#[derive(Debug, Clone)]
pub struct RegistrationStub {
    status: RegistrationStatus,
}

impl RegistrationStub {
    /// Start unregistered.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            status: RegistrationStatus::unregistered(),
        }
    }

    /// Current status snapshot.
    #[must_use]
    pub fn status(&self) -> RegistrationStatus {
        self.status.clone()
    }

    /// Seed from an already-resolved status (see [`resolve_on_chain`]).
    #[must_use]
    pub const fn from_status(status: RegistrationStatus) -> Self {
        Self { status }
    }

    /// Record a UID observed on chain.
    pub fn set_uid(&mut self, uid: u16) {
        self.status = RegistrationStatus::with_uid(uid);
    }

    /// Clear registration (e.g. after a failed permit check later).
    pub fn clear(&mut self) {
        self.status = RegistrationStatus::unregistered();
    }
}

impl Default for RegistrationStub {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s6_registration_stub_default_and_set_uid() {
        let mut stub = RegistrationStub::new();
        assert!(!stub.status().registered);
        assert_eq!(stub.status().uid, None);
        stub.set_uid(7);
        assert!(stub.status().registered);
        assert_eq!(stub.status().uid, Some(7));
        stub.clear();
        assert!(!stub.status().registered);
    }

    #[test]
    fn registration_uid_comes_from_metagraph_order() {
        let hk = |b: u8| vec![b; 32];
        let chain = chain::FakeChain::new(chain::FakeChainConfig {
            hotkeys: vec![hk(1), hk(2), hk(3)],
            ..chain::FakeChainConfig::default()
        });

        let mine = [2u8; 32];
        assert_eq!(
            resolve_on_chain(&chain, &mine).expect("resolve"),
            RegistrationStatus::with_uid(1),
            "uid must be the metagraph index, not a local default"
        );

        let stranger = [9u8; 32];
        assert_eq!(
            resolve_on_chain(&chain, &stranger).expect("resolve"),
            RegistrationStatus::unregistered(),
            "an absent hotkey is unregistered, not an error"
        );
    }
}
