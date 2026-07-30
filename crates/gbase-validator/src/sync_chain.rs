//! Make any [`ChainClient`] `Send + Sync` via an interior mutex.
//!
//! [`gbase_chain::FakeChain`] uses `RefCell` and is therefore `!Sync`. The
//! validator health server needs `Arc<dyn ChainClient + Send + Sync>`.

use std::sync::Mutex;

use gbase_chain::{ChainClient, ChainError, Metagraph, WeightsTlockPayload};

/// Mutex-wrapped [`ChainClient`] that is `Send + Sync`.
#[derive(Debug)]
pub struct SyncChain<C> {
    inner: Mutex<C>,
}

impl<C> SyncChain<C> {
    /// Wrap `inner`.
    #[must_use]
    pub const fn new(inner: C) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, C>, ChainError> {
        self.inner
            .lock()
            .map_err(|_| ChainError::Other("chain mutex poisoned".into()))
    }
}

impl<C: ChainClient> ChainClient for SyncChain<C> {
    fn current_block(&self) -> Result<u64, ChainError> {
        self.lock()?.current_block()
    }

    fn block_hash(&self, n: u64) -> Result<[u8; 32], ChainError> {
        self.lock()?.block_hash(n)
    }

    fn metagraph_at(&self, block_hash: &[u8; 32]) -> Result<Metagraph, ChainError> {
        self.lock()?.metagraph_at(block_hash)
    }

    fn subnet_owner_hotkey(&self, netuid: u16) -> Result<Vec<u8>, ChainError> {
        self.lock()?.subnet_owner_hotkey(netuid)
    }

    fn commit_reveal_enabled(&self, netuid: u16) -> Result<bool, ChainError> {
        self.lock()?.commit_reveal_enabled(netuid)
    }

    fn commit_reveal_version(&self, netuid: u16) -> Result<u16, ChainError> {
        self.lock()?.commit_reveal_version(netuid)
    }

    fn tempo(&self, netuid: u16) -> Result<u64, ChainError> {
        self.lock()?.tempo(netuid)
    }

    fn reveal_period_epochs(&self, netuid: u16) -> Result<u64, ChainError> {
        self.lock()?.reveal_period_epochs(netuid)
    }

    fn block_time(&self) -> Result<u64, ChainError> {
        self.lock()?.block_time()
    }

    fn last_epoch_block(&self, netuid: u16) -> Result<u64, ChainError> {
        self.lock()?.last_epoch_block(netuid)
    }

    fn pending_epoch_at(&self, netuid: u16) -> Result<u64, ChainError> {
        self.lock()?.pending_epoch_at(netuid)
    }

    fn subnet_epoch_index(&self, netuid: u16) -> Result<u64, ChainError> {
        self.lock()?.subnet_epoch_index(netuid)
    }

    fn blocks_since_last_step(&self, netuid: u16) -> Result<u64, ChainError> {
        self.lock()?.blocks_since_last_step(netuid)
    }

    fn submit_timelocked_weights(
        &self,
        mecid: u8,
        payload: WeightsTlockPayload,
        reveal_round: u64,
    ) -> Result<(), ChainError> {
        self.lock()?
            .submit_timelocked_weights(mecid, payload, reveal_round)
    }

    fn set_weights(
        &self,
        netuid: u16,
        uids: Vec<u16>,
        values: Vec<u16>,
        version_key: u64,
    ) -> Result<(), ChainError> {
        self.lock()?
            .set_weights(netuid, uids, values, version_key)
    }
}
