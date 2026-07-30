//! Chain client abstraction for gbase validators/miners.
//!
//! # Commit-reveal (CRV4)
//!
//! Default path is **commit-reveal v4** via [`ChainClient::submit_timelocked_weights`].
//! There is **no reveal extrinsic** in this crate (D22: reveal is epoch-driven on-chain;
//! `reveal_period_epochs` is measured in epochs, not blocks).
//!
//! When [`ChainClient::commit_reveal_enabled`] is `false`, callers must use the alternate
//! path **`set_weights`** (pallet `SubtensorModule`, call index from `metadata/testnet.lock`).
//! That alternate is intentionally **not** implemented here as the default; `FakeChain`
//! rejects timelocked submit when CR is off so tests assert the branch.
//!
//! # Schedule inputs
//!
//! The seven getters named after `epoch_schedule_inputs` in `metadata/testnet.lock` are
//! the complete input set for SDK `get_encrypted_commit_v2` / `generate_commit_v2`:
//! `tempo`, `reveal_period_epochs`, `block_time`, `last_epoch_block`, `pending_epoch_at`,
//! `subnet_epoch_index`, `blocks_since_last_step`.
//!
//! # Implementations
//!
//! - [`FakeChain`] — deterministic in-memory (required for unit tests).
//! - [`NotImplementedChain`] — stub returning [`ChainError::NotImplemented`] (SDK/live later).
//! - [`LiveRpcChain`] (feature `live`) — minimal JSON-RPC for `current_block` / headers.

#![forbid(unsafe_code)]

mod epoch_schedule;

pub use epoch_schedule::{
    advance_blocks, block_time_ms_to_secs, compute_reveal_round, current_epoch_pre_run_coinbase,
    gather_schedule_state, max_simulation_blocks, predict_first_reveal_block, should_run_epoch,
    simulate_run_coinbase, EpochScheduleError, EpochScheduleState, COMMIT_INCLUSION_BLOCK_OFFSET,
    DRAND_PERIOD, GENESIS_TIME, MAX_TEMPO, MAX_TEMPO_U64, SECURITY_BLOCK_OFFSET,
};

use std::cell::RefCell;
use std::fmt;

use parity_scale_codec::{Decode, Encode};

/// SCALE-shaped weights payload for timelock commit (SDK `WeightsTlockPayload`).
///
/// Fields match `metadata/testnet.lock` → `weights_tlock_payload.fields`:
/// `hotkey`, `uids`, `values`, `version_key`. **No merkle field** (D5).
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct WeightsTlockPayload {
    /// Hotkey public key bytes (typically 32-byte sr25519).
    pub hotkey: Vec<u8>,
    /// Destination UIDs.
    pub uids: Vec<u16>,
    /// Weight values aligned with `uids`.
    pub values: Vec<u16>,
    /// Weights version key.
    pub version_key: u64,
}

/// Minimal metagraph view at a block hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metagraph {
    /// Subnet netuid.
    pub netuid: u16,
    /// Neuron hotkeys in UID order.
    pub hotkeys: Vec<Vec<u8>>,
    /// Owner hotkey for the subnet (may equal first neuron or a dedicated owner).
    pub owner_hotkey: Vec<u8>,
}

/// One recorded `submit_timelocked_weights` invocation (`FakeChain` assertions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelockedWeightsSubmission {
    /// Mechanism id (`mecid`).
    pub mecid: u8,
    /// Committed payload (no merkle).
    pub payload: WeightsTlockPayload,
    /// Reveal round supplied by the commit scheduler.
    pub reveal_round: u64,
}

/// One recorded `set_weights` invocation (`FakeChain` assertions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetWeightsSubmission {
    /// Subnet netuid.
    pub netuid: u16,
    /// Destination UIDs.
    pub uids: Vec<u16>,
    /// Weight values.
    pub values: Vec<u16>,
    /// Weights version key.
    pub version_key: u64,
}

/// Extrinsic call recorded by [`FakeChain`] (CR on/off branch assertions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainCall {
    /// `commit_timelocked_mechanism_weights` path.
    SubmitTimelocked(TimelockedWeightsSubmission),
    /// Plain `set_weights` path (CR disabled only).
    SetWeights(SetWeightsSubmission),
}

/// Errors from chain reads/writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainError {
    /// Block number or hash unknown to this client.
    UnknownBlock {
        /// Requested block number when known.
        number: Option<u64>,
    },
    /// Metagraph not available for the given hash.
    UnknownMetagraph,
    /// Commit-reveal disabled; use `set_weights` instead.
    CommitRevealDisabled {
        /// Human-readable alternate path.
        alternate: &'static str,
    },
    /// Operation not wired yet (live SDK / RPC).
    NotImplemented {
        /// What was requested.
        what: &'static str,
    },
    /// Weights rate limit hit; caller should retry.
    RateLimited {
        /// Optional retry-after hint in blocks.
        retry_after_blocks: Option<u64>,
    },
    /// Generic transport or decode failure.
    Other(String),
}

impl fmt::Display for ChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownBlock { number: Some(n) } => {
                write!(f, "unknown block number {n}")
            }
            Self::UnknownBlock { number: None } => write!(f, "unknown block hash"),
            Self::UnknownMetagraph => write!(f, "metagraph not found for block hash"),
            Self::CommitRevealDisabled { alternate } => {
                write!(
                    f,
                    "commit-reveal disabled; use alternate path `{alternate}` (see metadata/testnet.lock call_indices.set_weights)"
                )
            }
            Self::NotImplemented { what } => {
                write!(f, "chain client not implemented: {what}")
            }
            Self::RateLimited { retry_after_blocks } => match retry_after_blocks {
                Some(n) => write!(f, "weights rate limited; retry after {n} blocks"),
                None => write!(f, "weights rate limited"),
            },
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ChainError {}

/// Read/write surface over Subtensor for weight commit scheduling.
///
/// Schedule methods are named **exactly** after lockfile `epoch_schedule_inputs` keys.
pub trait ChainClient {
    /// Best-known chain tip block number.
    ///
    /// # Errors
    ///
    /// Transport or decode failures.
    fn current_block(&self) -> Result<u64, ChainError>;

    /// Block hash for block number `n`.
    ///
    /// # Errors
    ///
    /// Unknown block or transport failure.
    fn block_hash(&self, n: u64) -> Result<[u8; 32], ChainError>;

    /// Metagraph snapshot at `block_hash`.
    ///
    /// # Errors
    ///
    /// Unknown hash or transport failure.
    fn metagraph_at(&self, block_hash: &[u8; 32]) -> Result<Metagraph, ChainError>;

    /// Subnet owner hotkey for `netuid`.
    ///
    /// # Errors
    ///
    /// Missing subnet or transport failure.
    fn subnet_owner_hotkey(&self, netuid: u16) -> Result<Vec<u8>, ChainError>;

    /// Whether commit-reveal is enabled for `netuid`.
    ///
    /// # Errors
    ///
    /// Transport failure.
    fn commit_reveal_enabled(&self, netuid: u16) -> Result<bool, ChainError>;

    /// Commit-reveal protocol version (testnet lock: **4** / CRV4).
    ///
    /// # Errors
    ///
    /// Transport failure.
    fn commit_reveal_version(&self, netuid: u16) -> Result<u16, ChainError>;

    /// `epoch_schedule_inputs.tempo` — blocks per epoch step (per-netuid).
    ///
    /// # Errors
    ///
    /// Transport failure.
    fn tempo(&self, netuid: u16) -> Result<u64, ChainError>;

    /// `epoch_schedule_inputs.reveal_period_epochs` — epochs, not blocks (D22).
    ///
    /// # Errors
    ///
    /// Transport failure.
    fn reveal_period_epochs(&self, netuid: u16) -> Result<u64, ChainError>;

    /// `epoch_schedule_inputs.block_time` — Aura slot duration in **milliseconds**.
    ///
    /// # Errors
    ///
    /// Transport failure.
    fn block_time(&self) -> Result<u64, ChainError>;

    /// `epoch_schedule_inputs.last_epoch_block`.
    ///
    /// # Errors
    ///
    /// Transport failure.
    fn last_epoch_block(&self, netuid: u16) -> Result<u64, ChainError>;

    /// `epoch_schedule_inputs.pending_epoch_at` — `0` when no owner-triggered pending epoch.
    ///
    /// # Errors
    ///
    /// Transport failure.
    fn pending_epoch_at(&self, netuid: u16) -> Result<u64, ChainError>;

    /// `epoch_schedule_inputs.subnet_epoch_index`.
    ///
    /// # Errors
    ///
    /// Transport failure.
    fn subnet_epoch_index(&self, netuid: u16) -> Result<u64, ChainError>;

    /// `epoch_schedule_inputs.blocks_since_last_step`.
    ///
    /// # Errors
    ///
    /// Transport failure.
    fn blocks_since_last_step(&self, netuid: u16) -> Result<u64, ChainError>;

    /// Submit timelocked (commit-reveal) weights. **No reveal call** (D22).
    ///
    /// When CR is disabled, implementations return [`ChainError::CommitRevealDisabled`]
    /// pointing at `set_weights`.
    ///
    /// # Errors
    ///
    /// CR disabled, transport, or not implemented.
    fn submit_timelocked_weights(
        &self,
        mecid: u8,
        payload: WeightsTlockPayload,
        reveal_round: u64,
    ) -> Result<(), ChainError>;

    /// Submit plain weights when commit-reveal is **disabled**.
    ///
    /// Must not be used while CR is enabled (D22: never downgrade).
    ///
    /// # Errors
    ///
    /// Rate limit, transport, or not implemented.
    fn set_weights(
        &self,
        netuid: u16,
        uids: Vec<u16>,
        values: Vec<u16>,
        version_key: u64,
    ) -> Result<(), ChainError>;
}

// ---------------------------------------------------------------------------
// FakeChain — deterministic in-memory
// ---------------------------------------------------------------------------

/// Default `FakeChain` schedule values (stable unit-test snapshot; lockfile records
/// *sources*, not volatile values).
pub mod fake_defaults {
    /// Typical subnet tempo (blocks).
    pub const TEMPO: u64 = 360;
    /// Reveal window in epochs (D22).
    pub const REVEAL_PERIOD_EPOCHS: u64 = 1;
    /// Aura slot duration ms (12s blocks → 12000).
    pub const BLOCK_TIME_MS: u64 = 12_000;
    /// Synthetic last epoch boundary.
    pub const LAST_EPOCH_BLOCK: u64 = 1_000;
    /// No pending owner epoch.
    pub const PENDING_EPOCH_AT: u64 = 0;
    /// Synthetic epoch index.
    pub const SUBNET_EPOCH_INDEX: u64 = 42;
    /// Blocks into current step.
    pub const BLOCKS_SINCE_LAST_STEP: u64 = 10;
    /// CRV4.
    pub const COMMIT_REVEAL_VERSION: u16 = 4;
    /// Tip block number.
    pub const CURRENT_BLOCK: u64 = 1_010;
    /// Default netuid under test.
    pub const NETUID: u16 = 1;
}

/// Configurable state for [`FakeChain`].
#[derive(Debug, Clone)]
pub struct FakeChainConfig {
    /// Tip block.
    pub current_block: u64,
    /// Netuid this fake serves.
    pub netuid: u16,
    /// Commit-reveal enabled flag.
    pub commit_reveal_enabled: bool,
    /// CR version (4 = CRV4).
    pub commit_reveal_version: u16,
    /// Blocks per epoch step.
    pub tempo: u64,
    /// Reveal window in epochs.
    pub reveal_period_epochs: u64,
    /// Block time in milliseconds.
    pub block_time_ms: u64,
    /// Last epoch boundary block.
    pub last_epoch_block: u64,
    /// Pending epoch block (0 if none).
    pub pending_epoch_at: u64,
    /// Subnet epoch counter.
    pub subnet_epoch_index: u64,
    /// Blocks since last step.
    pub blocks_since_last_step: u64,
    /// Owner hotkey bytes.
    pub owner_hotkey: Vec<u8>,
    /// Neuron hotkeys (UID order).
    pub hotkeys: Vec<Vec<u8>>,
    /// Number of subsequent weight submits that should return [`ChainError::RateLimited`].
    pub rate_limit_fails_remaining: u32,
}

impl Default for FakeChainConfig {
    fn default() -> Self {
        Self {
            current_block: fake_defaults::CURRENT_BLOCK,
            netuid: fake_defaults::NETUID,
            commit_reveal_enabled: true,
            commit_reveal_version: fake_defaults::COMMIT_REVEAL_VERSION,
            tempo: fake_defaults::TEMPO,
            reveal_period_epochs: fake_defaults::REVEAL_PERIOD_EPOCHS,
            block_time_ms: fake_defaults::BLOCK_TIME_MS,
            last_epoch_block: fake_defaults::LAST_EPOCH_BLOCK,
            pending_epoch_at: fake_defaults::PENDING_EPOCH_AT,
            subnet_epoch_index: fake_defaults::SUBNET_EPOCH_INDEX,
            blocks_since_last_step: fake_defaults::BLOCKS_SINCE_LAST_STEP,
            owner_hotkey: vec![0xA1; 32],
            hotkeys: vec![vec![0xA1; 32], vec![0xB2; 32], vec![0xC3; 32]],
            rate_limit_fails_remaining: 0,
        }
    }
}

/// Deterministic in-memory [`ChainClient`] for tests.
#[derive(Debug)]
pub struct FakeChain {
    cfg: FakeChainConfig,
    submissions: RefCell<Vec<TimelockedWeightsSubmission>>,
    set_weights_log: RefCell<Vec<SetWeightsSubmission>>,
    call_log: RefCell<Vec<ChainCall>>,
    rate_limit_fails_remaining: RefCell<u32>,
}

impl FakeChain {
    /// Build from config (CR on + CRV4 by default).
    #[must_use]
    pub fn new(cfg: FakeChainConfig) -> Self {
        let rate = cfg.rate_limit_fails_remaining;
        Self {
            cfg,
            submissions: RefCell::new(Vec::new()),
            set_weights_log: RefCell::new(Vec::new()),
            call_log: RefCell::new(Vec::new()),
            rate_limit_fails_remaining: RefCell::new(rate),
        }
    }

    /// Default config: CR enabled, version 4.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(FakeChainConfig::default())
    }

    /// CR disabled (forces `set_weights` alternate path on submit).
    #[must_use]
    pub fn with_commit_reveal_disabled() -> Self {
        Self::new(FakeChainConfig {
            commit_reveal_enabled: false,
            ..FakeChainConfig::default()
        })
    }

    /// Recorded timelocked submissions (for assertions).
    #[must_use]
    pub fn submissions(&self) -> Vec<TimelockedWeightsSubmission> {
        self.submissions.borrow().clone()
    }

    /// Recorded plain `set_weights` calls.
    #[must_use]
    pub fn set_weights_log(&self) -> Vec<SetWeightsSubmission> {
        self.set_weights_log.borrow().clone()
    }

    /// Full extrinsic call log (timelocked + `set_weights`), in order.
    #[must_use]
    pub fn call_log(&self) -> Vec<ChainCall> {
        self.call_log.borrow().clone()
    }

    /// Configure remaining synthetic rate-limit failures (tests).
    pub fn set_rate_limit_fails_remaining(&self, n: u32) {
        *self.rate_limit_fails_remaining.borrow_mut() = n;
    }

    fn consume_rate_limit(&self) -> Result<(), ChainError> {
        let mut left = self.rate_limit_fails_remaining.borrow_mut();
        if *left > 0 {
            *left = left.saturating_sub(1);
            return Err(ChainError::RateLimited {
                retry_after_blocks: Some(1),
            });
        }
        Ok(())
    }

    fn hash_for_block(n: u64) -> [u8; 32] {
        let mut h = [0_u8; 32];
        h[..8].copy_from_slice(&n.to_le_bytes());
        // tag so hashes are not all-zero for n=0 confusion in tests
        h[8] = 0xFC;
        h
    }
}

impl ChainClient for FakeChain {
    fn current_block(&self) -> Result<u64, ChainError> {
        Ok(self.cfg.current_block)
    }

    fn block_hash(&self, n: u64) -> Result<[u8; 32], ChainError> {
        if n > self.cfg.current_block {
            return Err(ChainError::UnknownBlock { number: Some(n) });
        }
        Ok(Self::hash_for_block(n))
    }

    fn metagraph_at(&self, block_hash: &[u8; 32]) -> Result<Metagraph, ChainError> {
        // Accept any hash that matches a known block ≤ tip.
        let mut found = false;
        for n in 0..=self.cfg.current_block {
            if Self::hash_for_block(n) == *block_hash {
                found = true;
                break;
            }
        }
        if !found {
            return Err(ChainError::UnknownMetagraph);
        }
        Ok(Metagraph {
            netuid: self.cfg.netuid,
            hotkeys: self.cfg.hotkeys.clone(),
            owner_hotkey: self.cfg.owner_hotkey.clone(),
        })
    }

    fn subnet_owner_hotkey(&self, netuid: u16) -> Result<Vec<u8>, ChainError> {
        if netuid != self.cfg.netuid {
            return Err(ChainError::Other(format!("unknown netuid {netuid}")));
        }
        Ok(self.cfg.owner_hotkey.clone())
    }

    fn commit_reveal_enabled(&self, _netuid: u16) -> Result<bool, ChainError> {
        Ok(self.cfg.commit_reveal_enabled)
    }

    fn commit_reveal_version(&self, _netuid: u16) -> Result<u16, ChainError> {
        Ok(self.cfg.commit_reveal_version)
    }

    fn tempo(&self, _netuid: u16) -> Result<u64, ChainError> {
        Ok(self.cfg.tempo)
    }

    fn reveal_period_epochs(&self, _netuid: u16) -> Result<u64, ChainError> {
        Ok(self.cfg.reveal_period_epochs)
    }

    fn block_time(&self) -> Result<u64, ChainError> {
        Ok(self.cfg.block_time_ms)
    }

    fn last_epoch_block(&self, _netuid: u16) -> Result<u64, ChainError> {
        Ok(self.cfg.last_epoch_block)
    }

    fn pending_epoch_at(&self, _netuid: u16) -> Result<u64, ChainError> {
        Ok(self.cfg.pending_epoch_at)
    }

    fn subnet_epoch_index(&self, _netuid: u16) -> Result<u64, ChainError> {
        Ok(self.cfg.subnet_epoch_index)
    }

    fn blocks_since_last_step(&self, _netuid: u16) -> Result<u64, ChainError> {
        Ok(self.cfg.blocks_since_last_step)
    }

    fn submit_timelocked_weights(
        &self,
        mecid: u8,
        payload: WeightsTlockPayload,
        reveal_round: u64,
    ) -> Result<(), ChainError> {
        if !self.cfg.commit_reveal_enabled {
            return Err(ChainError::CommitRevealDisabled {
                alternate: "set_weights",
            });
        }
        // CRV4 branch: version must be 4 for the default commit path.
        if self.cfg.commit_reveal_version != fake_defaults::COMMIT_REVEAL_VERSION {
            return Err(ChainError::Other(format!(
                "unsupported commit_reveal_version {} (expected {})",
                self.cfg.commit_reveal_version,
                fake_defaults::COMMIT_REVEAL_VERSION
            )));
        }
        self.consume_rate_limit()?;
        let sub = TimelockedWeightsSubmission {
            mecid,
            payload,
            reveal_round,
        };
        self.submissions.borrow_mut().push(sub.clone());
        self.call_log
            .borrow_mut()
            .push(ChainCall::SubmitTimelocked(sub));
        Ok(())
    }

    fn set_weights(
        &self,
        netuid: u16,
        uids: Vec<u16>,
        values: Vec<u16>,
        version_key: u64,
    ) -> Result<(), ChainError> {
        // Callers must only use this when CR is disabled (enforced by submit path).
        self.consume_rate_limit()?;
        let sub = SetWeightsSubmission {
            netuid,
            uids,
            values,
            version_key,
        };
        self.set_weights_log.borrow_mut().push(sub.clone());
        self.call_log.borrow_mut().push(ChainCall::SetWeights(sub));
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Stub / live
// ---------------------------------------------------------------------------

/// Placeholder client until a full SDK-backed impl lands.
///
/// TODO(task-13-followup): wire bittensor-core or expand [`LiveRpcChain`] beyond tip reads.
#[derive(Debug, Default, Clone, Copy)]
pub struct NotImplementedChain;

impl ChainClient for NotImplementedChain {
    fn current_block(&self) -> Result<u64, ChainError> {
        Err(ChainError::NotImplemented {
            what: "current_block",
        })
    }

    fn block_hash(&self, _n: u64) -> Result<[u8; 32], ChainError> {
        Err(ChainError::NotImplemented { what: "block_hash" })
    }

    fn metagraph_at(&self, _block_hash: &[u8; 32]) -> Result<Metagraph, ChainError> {
        Err(ChainError::NotImplemented {
            what: "metagraph_at",
        })
    }

    fn subnet_owner_hotkey(&self, _netuid: u16) -> Result<Vec<u8>, ChainError> {
        Err(ChainError::NotImplemented {
            what: "subnet_owner_hotkey",
        })
    }

    fn commit_reveal_enabled(&self, _netuid: u16) -> Result<bool, ChainError> {
        Err(ChainError::NotImplemented {
            what: "commit_reveal_enabled",
        })
    }

    fn commit_reveal_version(&self, _netuid: u16) -> Result<u16, ChainError> {
        Err(ChainError::NotImplemented {
            what: "commit_reveal_version",
        })
    }

    fn tempo(&self, _netuid: u16) -> Result<u64, ChainError> {
        Err(ChainError::NotImplemented { what: "tempo" })
    }

    fn reveal_period_epochs(&self, _netuid: u16) -> Result<u64, ChainError> {
        Err(ChainError::NotImplemented {
            what: "reveal_period_epochs",
        })
    }

    fn block_time(&self) -> Result<u64, ChainError> {
        Err(ChainError::NotImplemented { what: "block_time" })
    }

    fn last_epoch_block(&self, _netuid: u16) -> Result<u64, ChainError> {
        Err(ChainError::NotImplemented {
            what: "last_epoch_block",
        })
    }

    fn pending_epoch_at(&self, _netuid: u16) -> Result<u64, ChainError> {
        Err(ChainError::NotImplemented {
            what: "pending_epoch_at",
        })
    }

    fn subnet_epoch_index(&self, _netuid: u16) -> Result<u64, ChainError> {
        Err(ChainError::NotImplemented {
            what: "subnet_epoch_index",
        })
    }

    fn blocks_since_last_step(&self, _netuid: u16) -> Result<u64, ChainError> {
        Err(ChainError::NotImplemented {
            what: "blocks_since_last_step",
        })
    }

    fn submit_timelocked_weights(
        &self,
        _mecid: u8,
        _payload: WeightsTlockPayload,
        _reveal_round: u64,
    ) -> Result<(), ChainError> {
        Err(ChainError::NotImplemented {
            what: "submit_timelocked_weights",
        })
    }

    fn set_weights(
        &self,
        _netuid: u16,
        _uids: Vec<u16>,
        _values: Vec<u16>,
        _version_key: u64,
    ) -> Result<(), ChainError> {
        Err(ChainError::NotImplemented {
            what: "set_weights",
        })
    }
}

#[cfg(feature = "live")]
mod live {
    use super::{ChainClient, ChainError, Metagraph, WeightsTlockPayload};
    use serde_json::{json, Value};

    /// Finney testnet default (same as `metadata/testnet.lock` / config).
    pub const DEFAULT_TESTNET_ENDPOINT: &str = "wss://test.finney.opentensor.ai:443";

    /// Minimal HTTPS JSON-RPC chain client (no bittensor-core).
    ///
    /// Fully implements only tip/`current_block` and `block_hash` for smoke tests;
    /// other methods return [`ChainError::NotImplemented`] until expanded.
    #[derive(Debug)]
    pub struct LiveRpcChain {
        http: reqwest::blocking::Client,
        endpoint: String,
    }

    impl LiveRpcChain {
        /// Connect using `wss://` or `https://` endpoint (WSS rewritten to HTTPS).
        ///
        /// # Errors
        ///
        /// HTTP client build failure.
        pub fn connect(endpoint: &str) -> Result<Self, ChainError> {
            let http = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .map_err(|e| ChainError::Other(format!("http client: {e}")))?;
            Ok(Self {
                http,
                endpoint: http_endpoint(endpoint),
            })
        }

        fn rpc(&self, method: &str, params: Value) -> Result<Value, ChainError> {
            let body = json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params,
            });
            let resp = self
                .http
                .post(&self.endpoint)
                .json(&body)
                .send()
                .map_err(|e| ChainError::Other(format!("rpc send: {e}")))?;
            let v: Value = resp
                .json()
                .map_err(|e| ChainError::Other(format!("rpc json: {e}")))?;
            if let Some(err) = v.get("error") {
                return Err(ChainError::Other(format!("rpc error: {err}")));
            }
            v.get("result")
                .cloned()
                .ok_or_else(|| ChainError::Other("rpc missing result".into()))
        }
    }

    fn http_endpoint(endpoint: &str) -> String {
        if let Some(rest) = endpoint.strip_prefix("wss://") {
            format!("https://{rest}")
        } else if let Some(rest) = endpoint.strip_prefix("ws://") {
            format!("http://{rest}")
        } else {
            endpoint.to_owned()
        }
    }

    fn parse_hex_u64(hex_num: &str) -> Result<u64, ChainError> {
        let s = hex_num.strip_prefix("0x").unwrap_or(hex_num);
        u64::from_str_radix(s, 16).map_err(|e| ChainError::Other(format!("bad hex u64: {e}")))
    }

    impl ChainClient for LiveRpcChain {
        fn current_block(&self) -> Result<u64, ChainError> {
            let header = self.rpc("chain_getHeader", json!([]))?;
            let num = header
                .get("number")
                .and_then(Value::as_str)
                .ok_or_else(|| ChainError::Other("header.number missing".into()))?;
            parse_hex_u64(num)
        }

        fn block_hash(&self, n: u64) -> Result<[u8; 32], ChainError> {
            let hex_n = format!("0x{n:x}");
            let result = self.rpc("chain_getBlockHash", json!([hex_n]))?;
            let s = result
                .as_str()
                .ok_or_else(|| ChainError::Other("block hash not string".into()))?;
            let s = s.strip_prefix("0x").unwrap_or(s);
            if s.len() != 64 {
                return Err(ChainError::Other(format!(
                    "expected 32-byte hash, got len {}",
                    s.len() / 2
                )));
            }
            let mut out = [0_u8; 32];
            for i in 0..32 {
                out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
                    .map_err(|e| ChainError::Other(format!("hex: {e}")))?;
            }
            Ok(out)
        }

        fn metagraph_at(&self, _block_hash: &[u8; 32]) -> Result<Metagraph, ChainError> {
            Err(ChainError::NotImplemented {
                what: "live metagraph_at",
            })
        }

        fn subnet_owner_hotkey(&self, _netuid: u16) -> Result<Vec<u8>, ChainError> {
            Err(ChainError::NotImplemented {
                what: "live subnet_owner_hotkey",
            })
        }

        fn commit_reveal_enabled(&self, _netuid: u16) -> Result<bool, ChainError> {
            Err(ChainError::NotImplemented {
                what: "live commit_reveal_enabled",
            })
        }

        fn commit_reveal_version(&self, _netuid: u16) -> Result<u16, ChainError> {
            Err(ChainError::NotImplemented {
                what: "live commit_reveal_version",
            })
        }

        fn tempo(&self, _netuid: u16) -> Result<u64, ChainError> {
            Err(ChainError::NotImplemented { what: "live tempo" })
        }

        fn reveal_period_epochs(&self, _netuid: u16) -> Result<u64, ChainError> {
            Err(ChainError::NotImplemented {
                what: "live reveal_period_epochs",
            })
        }

        fn block_time(&self) -> Result<u64, ChainError> {
            Err(ChainError::NotImplemented {
                what: "live block_time",
            })
        }

        fn last_epoch_block(&self, _netuid: u16) -> Result<u64, ChainError> {
            Err(ChainError::NotImplemented {
                what: "live last_epoch_block",
            })
        }

        fn pending_epoch_at(&self, _netuid: u16) -> Result<u64, ChainError> {
            Err(ChainError::NotImplemented {
                what: "live pending_epoch_at",
            })
        }

        fn subnet_epoch_index(&self, _netuid: u16) -> Result<u64, ChainError> {
            Err(ChainError::NotImplemented {
                what: "live subnet_epoch_index",
            })
        }

        fn blocks_since_last_step(&self, _netuid: u16) -> Result<u64, ChainError> {
            Err(ChainError::NotImplemented {
                what: "live blocks_since_last_step",
            })
        }

        fn submit_timelocked_weights(
            &self,
            _mecid: u8,
            _payload: WeightsTlockPayload,
            _reveal_round: u64,
        ) -> Result<(), ChainError> {
            Err(ChainError::NotImplemented {
                what: "live submit_timelocked_weights",
            })
        }

        fn set_weights(
            &self,
            _netuid: u16,
            _uids: Vec<u16>,
            _values: Vec<u16>,
            _version_key: u64,
        ) -> Result<(), ChainError> {
            Err(ChainError::NotImplemented {
                what: "live set_weights",
            })
        }
    }
}

#[cfg(feature = "live")]
pub use live::{LiveRpcChain, DEFAULT_TESTNET_ENDPOINT};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Lockfile field names that must appear as trait methods (compile-time + runtime list).
    const SCHEDULE_INPUT_NAMES: &[&str] = &[
        "tempo",
        "reveal_period_epochs",
        "block_time",
        "last_epoch_block",
        "pending_epoch_at",
        "subnet_epoch_index",
        "blocks_since_last_step",
    ];

    fn sample_payload() -> WeightsTlockPayload {
        WeightsTlockPayload {
            hotkey: vec![0xA1; 32],
            uids: vec![0, 1, 2],
            values: vec![u16::MAX / 3, u16::MAX / 3, u16::MAX / 3 + 1],
            version_key: 1,
        }
    }

    #[test]
    fn s1_crv4_enabled_submit_records_payload_without_merkle() {
        let chain = FakeChain::with_defaults();
        assert!(chain
            .commit_reveal_enabled(fake_defaults::NETUID)
            .expect("cr flag"));
        assert_eq!(
            chain
                .commit_reveal_version(fake_defaults::NETUID)
                .expect("ver"),
            4
        );

        let payload = sample_payload();
        chain
            .submit_timelocked_weights(0, payload.clone(), 99)
            .expect("submit");

        let subs = chain.submissions();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].mecid, 0);
        assert_eq!(subs[0].reveal_round, 99);
        assert_eq!(subs[0].payload, payload);
        assert_eq!(subs[0].payload.hotkey, vec![0xA1; 32]);
        assert_eq!(subs[0].payload.uids, vec![0, 1, 2]);
        assert_eq!(
            subs[0].payload.values,
            vec![u16::MAX / 3, u16::MAX / 3, u16::MAX / 3 + 1]
        );
        assert_eq!(subs[0].payload.version_key, 1);

        // Structural: WeightsTlockPayload has exactly the four lockfile fields (no merkle).
        let _ = WeightsTlockPayload {
            hotkey: subs[0].payload.hotkey.clone(),
            uids: subs[0].payload.uids.clone(),
            values: subs[0].payload.values.clone(),
            version_key: subs[0].payload.version_key,
        };
    }

    #[test]
    fn s2_cr_disabled_rejects_timelock_points_at_set_weights() {
        let chain = FakeChain::with_commit_reveal_disabled();
        assert!(!chain
            .commit_reveal_enabled(fake_defaults::NETUID)
            .expect("flag"));

        let err = chain
            .submit_timelocked_weights(0, sample_payload(), 1)
            .expect_err("must refuse");
        match err {
            ChainError::CommitRevealDisabled { alternate } => {
                assert_eq!(alternate, "set_weights");
            }
            other => panic!("unexpected error: {other}"),
        }
        assert!(chain.submissions().is_empty());
        assert!(err.to_string().contains("set_weights"));
    }

    #[test]
    fn s2_schedule_inputs_match_fake_defaults_and_seven_names() {
        assert_eq!(SCHEDULE_INPUT_NAMES.len(), 7);

        let chain = FakeChain::with_defaults();
        let n = fake_defaults::NETUID;

        // Named exactly after lockfile keys — call each by trait method name.
        assert_eq!(chain.tempo(n).expect("tempo"), fake_defaults::TEMPO);
        assert_eq!(
            chain.reveal_period_epochs(n).expect("reveal_period_epochs"),
            fake_defaults::REVEAL_PERIOD_EPOCHS
        );
        assert_eq!(
            chain.block_time().expect("block_time"),
            fake_defaults::BLOCK_TIME_MS
        );
        assert_eq!(
            chain.last_epoch_block(n).expect("last_epoch_block"),
            fake_defaults::LAST_EPOCH_BLOCK
        );
        assert_eq!(
            chain.pending_epoch_at(n).expect("pending_epoch_at"),
            fake_defaults::PENDING_EPOCH_AT
        );
        assert_eq!(
            chain.subnet_epoch_index(n).expect("subnet_epoch_index"),
            fake_defaults::SUBNET_EPOCH_INDEX
        );
        assert_eq!(
            chain
                .blocks_since_last_step(n)
                .expect("blocks_since_last_step"),
            fake_defaults::BLOCKS_SINCE_LAST_STEP
        );

        // Ensure the string list stays aligned (documentation / drift guard).
        for name in SCHEDULE_INPUT_NAMES {
            assert!(
                matches!(
                    *name,
                    "tempo"
                        | "reveal_period_epochs"
                        | "block_time"
                        | "last_epoch_block"
                        | "pending_epoch_at"
                        | "subnet_epoch_index"
                        | "blocks_since_last_step"
                ),
                "unexpected schedule name {name}"
            );
        }
    }

    #[test]
    fn s3_block_hash_metagraph_owner_and_tip() {
        let chain = FakeChain::with_defaults();
        let tip = chain.current_block().expect("tip");
        assert_eq!(tip, fake_defaults::CURRENT_BLOCK);
        let h = chain.block_hash(tip).expect("hash");
        let mg = chain.metagraph_at(&h).expect("meta");
        assert_eq!(mg.netuid, fake_defaults::NETUID);
        assert_eq!(mg.hotkeys.len(), 3);
        let owner = chain
            .subnet_owner_hotkey(fake_defaults::NETUID)
            .expect("owner");
        assert_eq!(owner, mg.owner_hotkey);
        assert!(matches!(
            chain.block_hash(tip + 1),
            Err(ChainError::UnknownBlock { .. })
        ));
    }

    #[test]
    fn s3_not_implemented_stub_errors_clearly() {
        let c = NotImplementedChain;
        let err = c.current_block().expect_err("stub");
        assert!(matches!(err, ChainError::NotImplemented { .. }));
        assert!(err.to_string().contains("not implemented"));
    }

    #[test]
    fn s3_unsupported_cr_version_rejected_on_submit() {
        let chain = FakeChain::new(FakeChainConfig {
            commit_reveal_version: 3,
            ..FakeChainConfig::default()
        });
        let err = chain
            .submit_timelocked_weights(0, sample_payload(), 0)
            .expect_err("v3");
        assert!(err
            .to_string()
            .contains("unsupported commit_reveal_version"));
    }

    #[test]
    fn s4_payload_scale_encodes_four_fields_only() {
        let p = sample_payload();
        let encoded = p.encode();
        let decoded = WeightsTlockPayload::decode(&mut &encoded[..]).expect("decode");
        assert_eq!(decoded, p);
        // Manual SCALE of the four fields must match (no trailing merkle bytes).
        let mut manual = Vec::new();
        p.hotkey.encode_to(&mut manual);
        p.uids.encode_to(&mut manual);
        p.values.encode_to(&mut manual);
        p.version_key.encode_to(&mut manual);
        assert_eq!(encoded, manual);
    }

    #[test]
    fn s4_payload_struct_has_no_merkle_field() {
        // Compile-time shape: only these four fields exist on the type.
        let p = WeightsTlockPayload {
            hotkey: vec![1],
            uids: vec![0],
            values: vec![1],
            version_key: 0,
        };
        let names = ["hotkey", "uids", "values", "version_key"];
        assert_eq!(names.len(), 4);
        // Runtime: encoding length equals four-field manual encode (no extra root).
        let mut four = Vec::new();
        p.hotkey.encode_to(&mut four);
        p.uids.encode_to(&mut four);
        p.values.encode_to(&mut four);
        p.version_key.encode_to(&mut four);
        assert_eq!(p.encode(), four);
        // A 32-byte merkle root appended would lengthen SCALE — prove we do not.
        let mut with_fake_root = four.clone();
        [0_u8; 32].encode_to(&mut with_fake_root);
        assert_ne!(p.encode(), with_fake_root);
    }

    #[test]
    fn s1_set_weights_records_call_log() {
        let chain = FakeChain::with_commit_reveal_disabled();
        chain
            .set_weights(1, vec![0, 1], vec![100, 200], 7)
            .expect("set");
        let log = chain.call_log();
        assert_eq!(log.len(), 1);
        match &log[0] {
            ChainCall::SetWeights(s) => {
                assert_eq!(s.netuid, 1);
                assert_eq!(s.uids, vec![0, 1]);
                assert_eq!(s.values, vec![100, 200]);
                assert_eq!(s.version_key, 7);
            }
            ChainCall::SubmitTimelocked(s) => panic!("expected SetWeights, got {s:?}"),
        }
        assert!(chain.submissions().is_empty());
    }

    #[test]
    fn s5_rate_limit_then_success() {
        let chain = FakeChain::new(FakeChainConfig {
            rate_limit_fails_remaining: 2,
            ..FakeChainConfig::default()
        });
        let p = sample_payload();
        assert!(matches!(
            chain.submit_timelocked_weights(0, p.clone(), 1),
            Err(ChainError::RateLimited { .. })
        ));
        assert!(matches!(
            chain.submit_timelocked_weights(0, p.clone(), 1),
            Err(ChainError::RateLimited { .. })
        ));
        chain
            .submit_timelocked_weights(0, p, 1)
            .expect("third succeeds");
        assert_eq!(chain.submissions().len(), 1);
    }

    /// Live testnet smoke: `current_block() > 0`.
    ///
    /// Run with:
    /// `cargo test -p chain --features live testnet_current_block -- --ignored --nocapture`
    #[test]
    #[ignore = "requires network access to finney testnet"]
    fn testnet_current_block_positive() {
        #[cfg(feature = "live")]
        {
            let client = LiveRpcChain::connect(DEFAULT_TESTNET_ENDPOINT).expect("connect testnet");
            let n = client.current_block().expect("current_block");
            assert!(n > 0, "expected tip > 0, got {n}");
        }
        #[cfg(not(feature = "live"))]
        {
            panic!(
                "enable --features live to run testnet_current_block_positive against {}",
                "wss://test.finney.opentensor.ai:443"
            );
        }
    }
}
