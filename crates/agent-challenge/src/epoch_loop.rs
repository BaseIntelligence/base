//! Headless epoch dispatch + single active signer (Metis N2).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agent_dispatch::{TaskDescriptorV1, TaskResultV1, TaskStatusV1};
use agent_pack::{select_pack, PackId};
use crypto::KEY_LEN;
use thiserror::Error;
use tokio::task::JoinSet;
use tokio::time::timeout;

use crate::expected_set::{hex32, ExpectedSet};

/// R1 deadline numerator (≈60% of epoch).
pub const R1_DEADLINE_FRACTION_NUM: u64 = 60;
/// R1 deadline denominator.
pub const R1_DEADLINE_FRACTION_DEN: u64 = 100;
/// Testnet tempo blocks.
pub const TESTNET_TEMPO_BLOCKS: u64 = 360;

/// Runner capacity advertisement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunnerCapacity {
    pub max_concurrency: u32,
    pub current_load: u32,
}

/// Raw per-miner outcome (pre-grading).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MinerEpochOutcome {
    /// Done before deadline.
    Completed { pack_id: String, result: TaskResultV1 },
    /// Deadline hit.
    TimedOut { pack_id: String },
    /// Failed early.
    Failed { pack_id: String, reason: String },
    /// No free slots.
    CapacityExhausted { pack_id: String },
}

/// Epoch result with `|E|` outcomes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochDispatchResult {
    pub epoch: u64,
    pub block_hash: [u8; 32],
    pub outcomes: BTreeMap<[u8; KEY_LEN], MinerEpochOutcome>,
}

/// Loop errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EpochLoopError {
    /// Empty E.
    #[error("empty expected set")]
    EmptyExpectedSet,
    /// Empty catalog.
    #[error("empty pack catalog")]
    EmptyCatalog,
    /// Double signer.
    #[error("signer already active for challenge={challenge_id} epoch={epoch}")]
    SignerAlreadyActive {
        /// Id.
        challenge_id: String,
        /// Epoch.
        epoch: u64,
    },
    /// Select failed.
    #[error("pack select: {0}")]
    PackSelect(String),
    /// Panic.
    #[error("dispatch task panicked: {0}")]
    Join(String),
}

/// Headless epoch inputs.
#[derive(Debug, Clone)]
pub struct EpochDispatchConfig {
    pub challenge_id: String,
    pub scoring_version: u16,
    pub epoch: u64,
    pub expected: ExpectedSet,
    pub catalog: Vec<PackId>,
    pub deadline: Duration,
    pub deadline_unix_ms: u64,
}


/// Dispatch client (fake or HTTP).
pub trait EpochDispatchClient: Send + Sync + 'static {
    /// Capacity.
    fn capacity(
        &self,
        miner: [u8; KEY_LEN],
    ) -> impl std::future::Future<Output = RunnerCapacity> + Send;
    /// Run pack.
    fn run_pack(
        &self,
        miner: [u8; KEY_LEN],
        descriptor: TaskDescriptorV1,
    ) -> impl std::future::Future<Output = Result<TaskResultV1, String>> + Send;
}

/// Single-active-signer registry.
#[derive(Debug, Default)]
pub struct ActiveSignerRegistry {
    held: Mutex<BTreeMap<(String, u64), ()>>,
}

/// RAII lease.
#[derive(Debug)]
pub struct SignerGuard {
    reg: Arc<ActiveSignerRegistry>,
    key: (String, u64),
}

impl ActiveSignerRegistry {
    /// Shared empty registry.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Exclusive signer lease.
    ///
    /// # Errors
    /// Already held.
    pub fn try_acquire(
        self: &Arc<Self>,
        challenge_id: &str,
        epoch: u64,
    ) -> Result<SignerGuard, EpochLoopError> {
        let key = (challenge_id.to_owned(), epoch);
        let mut g = self.held.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if g.contains_key(&key) {
            return Err(EpochLoopError::SignerAlreadyActive {
                challenge_id: challenge_id.to_owned(),
                epoch,
            });
        }
        g.insert(key.clone(), ());
        Ok(SignerGuard {
            reg: Arc::clone(self),
            key,
        })
    }
}

impl Drop for SignerGuard {
    fn drop(&mut self) {
        self.reg
            .held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.key);
    }
}

/// One epoch: parallel dispatch, deadline → TimedOut, `|E|` outcomes.
///
/// # Errors
/// Empty E/catalog, signer conflict, pack select, join panic.
pub async fn run_epoch_dispatch<C: EpochDispatchClient>(
    cfg: &EpochDispatchConfig,
    client: Arc<C>,
    signers: &Arc<ActiveSignerRegistry>,
) -> Result<EpochDispatchResult, EpochLoopError> {
    if cfg.expected.participants.is_empty() {
        return Err(EpochLoopError::EmptyExpectedSet);
    }
    if cfg.catalog.is_empty() {
        return Err(EpochLoopError::EmptyCatalog);
    }
    let _lease = signers.try_acquire(&cfg.challenge_id, cfg.epoch)?;
    let end = Instant::now() + cfg.deadline;
    let mut outcomes = BTreeMap::new();
    let mut set = JoinSet::new();
    for p in &cfg.expected.participants {
        let miner = p.hotkey;
        let pack_id = select_pack(cfg.epoch, &miner, &cfg.catalog)
            .map_err(|e| EpochLoopError::PackSelect(e.to_string()))?
            .as_str()
            .to_owned();
        let cap = client.capacity(miner).await;
        if cap.current_load >= cap.max_concurrency {
            outcomes.insert(miner, MinerEpochOutcome::CapacityExhausted { pack_id });
            continue;
        }
        let desc = TaskDescriptorV1::new(
            cfg.challenge_id.clone(),
            cfg.scoring_version,
            cfg.epoch,
            hex32(&miner),
            pack_id.clone(),
            cfg.deadline_unix_ms,
        );
        let c = Arc::clone(&client);
        let rem = end.saturating_duration_since(Instant::now());
        set.spawn(async move {
            let o = match timeout(rem, c.run_pack(miner, desc)).await {
                Ok(Ok(r)) => match r.status {
                    TaskStatusV1::TimedOut => MinerEpochOutcome::TimedOut { pack_id },
                    TaskStatusV1::Failed => MinerEpochOutcome::Failed {
                        pack_id,
                        reason: "runner status=failed".into(),
                    },
                    TaskStatusV1::Completed => {
                        MinerEpochOutcome::Completed {
                            pack_id,
                            result: r,
                        }
                    }
                },
                Ok(Err(reason)) => MinerEpochOutcome::Failed { pack_id, reason },
                Err(_) => MinerEpochOutcome::TimedOut { pack_id },
            };
            (miner, o)
        });
    }
    while let Some(j) = set.join_next().await {
        let (m, o) = j.map_err(|e| EpochLoopError::Join(e.to_string()))?;
        outcomes.insert(m, o);
    }
    Ok(EpochDispatchResult {
        epoch: cfg.epoch,
        block_hash: cfg.expected.block_hash,
        outcomes,
    })
}

