//! Live Bittensor JSON-RPC chain client with sr25519 signed extrinsics.
//!
//! Re-exports [`ChainClient`], [`ChainError`], [`Metagraph`], [`WeightsTlockPayload`]
//! from the `chain` crate and provides [`LiveChainClient`] — a full
//! implementation backed by blocking HTTPS JSON-RPC.

#![forbid(unsafe_code)]

mod extrinsic;
mod rpc;
mod storage;
mod tlock;

#[cfg(test)]
mod tests;

pub use chain::{AxonInfo, ChainClient, ChainError, Metagraph, WeightsTlockPayload};
pub use extrinsic::{
    build_and_sign_commit_timelocked, build_and_sign_serve_axon, build_and_sign_set_weights,
    commit_timelocked_call, derive_public_key, serve_axon_call, set_weights_call, Era,
    ServeAxonParams,
};
pub use rpc::{LiveChainRpc, RuntimeVersion, StorageEntry};
pub use storage::{
    decode_axon_info, decode_bool, decode_double_map_account_k2, decode_double_map_k2,
    decode_hotkey, decode_metagraph, decode_u16, decode_u64, decode_vec_u64, decode_vec_vec_u8,
    storage_double_map_key_u16_account, storage_double_map_key_u16_u16,
    storage_double_map_prefix_u16, storage_key, storage_map_key_account_blake2,
    storage_map_key_identity, storage_map_key_twox64, storage_map_key_u16, ACCOUNT_ID_LEN,
};
pub use tlock::encrypt_commit;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use parity_scale_codec::Encode;

/// Lockfile-pinned commit-reveal version (CRV4 = 4).
///
/// `spec_version` / `transaction_version` are **not** pinned here: signing
/// always uses the live values from `state_getRuntimeVersion` so a Finney
/// runtime bump cannot stall weight submits waiting for a lockfile bump.
const COMMIT_REVEAL_VERSION: u16 = 4;
/// Default netuid (lockfile `snapshot_netuid` = 1).
const DEFAULT_NETUID: u16 = 1;
/// `SubtensorModule` pallet name.
const PALLET_SUBTENSOR: &str = "SubtensorModule";
/// `RevealPeriodEpochs` `ValueQuery` default when the key is absent.
const DEFAULT_REVEAL_PERIOD_EPOCHS: u16 = 1;
/// Page size for `state_getKeysPaged` when enumerating a netuid's neurons.
const KEYS_PAGE_SIZE: u32 = 512;
/// Upper bound on enumerated neurons, guarding against a runaway pager.
const MAX_NEURONS: usize = 16_384;
/// `WeightsSetRateLimit` fallback when the sparse storage key is absent
/// (Finney netuid 100 observes 100 via hyperparams with an empty map).
const DEFAULT_WEIGHTS_RATE_LIMIT: u64 = 100;
/// Blocks to wait for a weight extrinsic to advance `LastUpdate` before
/// declaring the submit unconfirmed. A failed dispatch (e.g. rate limit) is
/// included in a block but never advances `LastUpdate`.
const WEIGHT_CONFIRM_BLOCKS: u64 = 4;
/// Poll interval while waiting for weight-submit dispatch confirmation.
const WEIGHT_CONFIRM_POLL: Duration = Duration::from_secs(4);
/// How long a successful [`LiveChainClient::metagraph_at`] snapshot is reused
/// for the same `(netuid, block_hash)` before the next bulk RPC refresh.
///
/// Emitters and sealers pin the epoch-start metagraph for ~72 minutes; a
/// 15-minute TTL collapses per-tick `state_getKeysPaged` /
/// `state_queryStorageAt` storms on public Finney RPCs without changing the
/// bulk enumeration path.
const METAGRAPH_CACHE_TTL: Duration = Duration::from_mins(15);

/// One cached metagraph snapshot keyed by `(netuid, block_hash)`.
#[derive(Debug, Clone)]
struct CachedMetagraph {
    fetched_at: Instant,
    metagraph: Metagraph,
}

/// In-process TTL cache for [`LiveChainClient::metagraph_at`].
///
/// The mutex is held across a cache-miss refresh so concurrent callers for the
/// same key share a single bulk RPC (no thundering herd).
#[derive(Debug, Default)]
struct MetagraphCache {
    entries: HashMap<(u16, [u8; 32]), CachedMetagraph>,
}

/// Live chain client with sr25519 signed extrinsic submission.
pub struct LiveChainClient {
    rpc: LiveChainRpc,
    netuid: u16,
    signing_key: Option<[u8; 32]>,
    metagraph_cache: Mutex<MetagraphCache>,
    /// Overrideable for tests; production always uses [`METAGRAPH_CACHE_TTL`].
    metagraph_cache_ttl: Duration,
}

impl std::fmt::Debug for LiveChainClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveChainClient")
            .field("netuid", &self.netuid)
            .field(
                "signing_key",
                &self.signing_key.as_ref().map(|_| "<redacted>"),
            )
            .finish_non_exhaustive()
    }
}

impl LiveChainClient {
    /// Connect to a JSON-RPC endpoint (no signing key loaded).
    ///
    /// `wss://` is rewritten to `https://`. Read-only methods work immediately;
    /// `set_weights` / `submit_timelocked_weights` require [`Self::with_signing_key`].
    ///
    /// `endpoint` may be a comma-separated ordered failover list (see
    /// [`LiveChainRpc::connect`]); each call tries endpoints in order and
    /// cools rate-limited / unreachable ones.
    ///
    /// # Errors
    /// HTTP client build failure, or an empty endpoint list.
    pub fn connect(endpoint: &str) -> Result<Self, ChainError> {
        let rpc = LiveChainRpc::connect(endpoint)?;
        Ok(Self {
            rpc,
            netuid: DEFAULT_NETUID,
            signing_key: None,
            metagraph_cache: Mutex::new(MetagraphCache::default()),
            metagraph_cache_ttl: METAGRAPH_CACHE_TTL,
        })
    }

    /// Override the metagraph cache TTL (tests only).
    #[cfg(test)]
    pub fn set_metagraph_cache_ttl(&mut self, ttl: Duration) {
        self.metagraph_cache_ttl = ttl;
    }

    /// Fetch the metagraph at `block_hash`, serving from the in-process TTL
    /// cache on hit. On miss/expiry runs the bulk `Keys` enumeration path
    /// once while holding the cache lock so concurrent waiters reuse it.
    fn metagraph_at_cached(&self, block_hash: &[u8; 32]) -> Result<Metagraph, ChainError> {
        let key = (self.netuid, *block_hash);
        let ttl = self.metagraph_cache_ttl;
        let mut cache = self
            .metagraph_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(hit) = cache.entries.get(&key) {
            if hit.fetched_at.elapsed() < ttl {
                return Ok(hit.metagraph.clone());
            }
        }
        let mg = self.fetch_metagraph_bulk(block_hash)?;
        cache.entries.insert(
            key,
            CachedMetagraph {
                fetched_at: Instant::now(),
                metagraph: mg.clone(),
            },
        );
        Ok(mg)
    }

    /// Bulk metagraph read (no cache). Kept separate so the cache path can
    /// hold the mutex across the refresh without recursion.
    fn fetch_metagraph_bulk(&self, block_hash: &[u8; 32]) -> Result<Metagraph, ChainError> {
        let netuid = self.netuid;
        // A zero hash means "tip"; callers that do not track a block pass it.
        let at = if block_hash == &[0_u8; 32] {
            None
        } else {
            Some(block_hash)
        };
        let keys = self.enumerate_hotkeys(netuid, at)?;
        let coldkeys = self.fetch_coldkeys_for_hotkeys(&keys, at)?;
        let owner = self.read_owner_hotkey(netuid, at)?;
        Ok(storage::decode_metagraph(keys, coldkeys, owner, netuid))
    }

    /// Bulk-read `SubtensorModule.Owner(hotkey) → coldkey` for every hotkey.
    ///
    /// Uses batched `state_queryStorageAt` (same path as `Keys`), never
    /// per-UID RPCs. Missing / default (all-zero) owners become zero vectors
    /// so the UID alignment with `hotkeys` is preserved.
    fn fetch_coldkeys_for_hotkeys(
        &self,
        hotkeys: &[Vec<u8>],
        at: Option<&[u8; 32]>,
    ) -> Result<Vec<Vec<u8>>, ChainError> {
        if hotkeys.is_empty() {
            return Ok(Vec::new());
        }
        let mut storage_keys = Vec::with_capacity(hotkeys.len());
        let mut index_of: HashMap<Vec<u8>, usize> = HashMap::with_capacity(hotkeys.len());
        for (i, hk) in hotkeys.iter().enumerate() {
            let account = account_id(hk)?;
            let sk = storage::storage_map_key_account_blake2(PALLET_SUBTENSOR, "Owner", &account);
            index_of.insert(sk.clone(), i);
            storage_keys.push(sk);
        }
        let mut coldkeys = vec![vec![0_u8; ACCOUNT_ID_LEN]; hotkeys.len()];
        for chunk in storage_keys.chunks(256) {
            for (key, value) in self.rpc.state_query_storage_at(chunk, at)? {
                let Some(&i) = index_of.get(&key) else {
                    continue;
                };
                coldkeys[i] = storage::decode_hotkey(&value)?;
            }
        }
        Ok(coldkeys)
    }

    /// Connect and load a signing key from a file (32 raw bytes or 64 hex chars).
    ///
    /// # Errors
    /// HTTP client build, key file read, or key decode failure.
    pub fn with_signing_key(endpoint: &str, key_file: &Path) -> Result<Self, ChainError> {
        let mut client = Self::connect(endpoint)?;
        client.signing_key = Some(load_signing_key(key_file)?);
        Ok(client)
    }

    /// Set the netuid used by [`ChainClient::metagraph_at`].
    pub fn set_netuid(&mut self, netuid: u16) {
        self.netuid = netuid;
    }

    /// Install the sr25519 mini-secret used to sign extrinsics.
    ///
    /// Callers typically derive this from a Bittensor wallet mnemonic. Without
    /// it, `set_weights` and `submit_timelocked_weights` fail closed rather
    /// than silently doing nothing.
    pub fn set_signing_key(&mut self, mini_secret: [u8; 32]) {
        self.signing_key = Some(mini_secret);
    }

    /// Whether a signing key is loaded. Never exposes the key itself.
    #[must_use]
    pub fn can_sign(&self) -> bool {
        self.signing_key.is_some()
    }

    /// Require a signing key, returning an error if none is loaded.
    fn require_key(&self) -> Result<[u8; 32], ChainError> {
        self.signing_key
            .ok_or_else(|| ChainError::Other("no signing key loaded (use with_signing_key)".into()))
    }

    /// Fetch live `(spec_version, transaction_version)` for signed extrinsics.
    ///
    /// Always trusts the chain tip — never compares against a local pin. Call
    /// indices / CRV4 shape still come from the committed lockfile; only the
    /// additional-signed runtime versions are dynamic.
    fn live_runtime_version(&self) -> Result<(u32, u32), ChainError> {
        let rt = self.rpc.state_get_runtime_version()?;
        tracing::debug!(
            spec = rt.spec_version,
            tx = rt.transaction_version,
            "live runtime version for signing"
        );
        Ok((rt.spec_version, rt.transaction_version))
    }

    /// Read `commit_reveal_weights_enabled` from hyperparams v3 runtime API.
    fn commit_reveal_enabled_from_hyperparams(
        &self,
        netuid: u16,
    ) -> Result<Option<bool>, ChainError> {
        let bytes = self.rpc.state_call(
            "SubnetInfoRuntimeApi_get_subnet_hyperparams_v3",
            &netuid.to_le_bytes(),
        )?;
        Ok(parse_commit_reveal_enabled_v3(&bytes))
    }

    /// Read a per-netuid `u64`, substituting a `ValueQuery` default when absent.
    ///
    /// Substrate omits `ValueQuery` keys whose value equals the pallet default,
    /// so an absent key is normal and must not be an error.
    fn read_netuid_u64(&self, item: &str, netuid: u16, default: u64) -> Result<u64, ChainError> {
        let key = storage::storage_map_key_u16(PALLET_SUBTENSOR, item, netuid);
        match self.rpc.state_get_storage(&key)? {
            Some(bytes) => storage::decode_u64(&bytes),
            None => Ok(default),
        }
    }

    /// Read a per-netuid `u16`, substituting a `ValueQuery` default when absent.
    fn read_netuid_u16(&self, item: &str, netuid: u16, default: u16) -> Result<u16, ChainError> {
        let key = storage::storage_map_key_u16(PALLET_SUBTENSOR, item, netuid);
        match self.rpc.state_get_storage(&key)? {
            Some(bytes) => storage::decode_u16(&bytes),
            None => Ok(default),
        }
    }

    /// Read a per-netuid `u16` that must exist (no meaningful default).
    fn read_netuid_u16_required(&self, item: &str, netuid: u16) -> Result<u16, ChainError> {
        let key = storage::storage_map_key_u16(PALLET_SUBTENSOR, item, netuid);
        let bytes = self.rpc.state_get_storage(&key)?.ok_or_else(|| {
            ChainError::Other(format!("{PALLET_SUBTENSOR}.{item}({netuid}) not found"))
        })?;
        storage::decode_u16(&bytes)
    }

    /// Read `SubnetOwnerHotkey(netuid)`; absent means the subnet does not exist.
    fn read_owner_hotkey(&self, netuid: u16, at: Option<&[u8; 32]>) -> Result<Vec<u8>, ChainError> {
        let key = storage::storage_map_key_u16(PALLET_SUBTENSOR, "SubnetOwnerHotkey", netuid);
        let raw = match at {
            Some(h) => self.rpc.state_get_storage_at(&key, h)?,
            None => self.rpc.state_get_storage(&key)?,
        };
        let bytes = raw.ok_or_else(|| {
            ChainError::Other(format!(
                "{PALLET_SUBTENSOR}.SubnetOwnerHotkey({netuid}) not found — subnet does not exist"
            ))
        })?;
        storage::decode_hotkey(&bytes)
    }

    /// Enumerate `Keys(netuid, uid) -> AccountId32` in ascending `uid` order.
    ///
    /// `Keys` is a double map, so the hotkeys are spread across one storage
    /// entry per neuron rather than a single `Vec`. Pages through
    /// `state_getKeysPaged` and batch-reads with `state_queryStorageAt`.
    fn enumerate_hotkeys(
        &self,
        netuid: u16,
        at: Option<&[u8; 32]>,
    ) -> Result<Vec<Vec<u8>>, ChainError> {
        let prefix = storage::storage_double_map_prefix_u16(PALLET_SUBTENSOR, "Keys", netuid);
        let all_keys = self.paged_keys(&prefix, "Keys", netuid, at)?;
        if all_keys.is_empty() {
            return Ok(Vec::new());
        }

        let mut by_uid: std::collections::BTreeMap<u16, Vec<u8>> =
            std::collections::BTreeMap::new();
        for chunk in all_keys.chunks(256) {
            for (key, value) in self.rpc.state_query_storage_at(chunk, at)? {
                let uid = storage::decode_double_map_k2(&key)?;
                by_uid.insert(uid, storage::decode_hotkey(&value)?);
            }
        }
        Ok(by_uid.into_values().collect())
    }

    /// Read `Axons(netuid, hotkey)`; `None` means the miner never served one.
    ///
    /// # Errors
    /// Bad hotkey length, transport, or decode failure.
    pub fn read_axon(&self, netuid: u16, hotkey: &[u8]) -> Result<Option<AxonInfo>, ChainError> {
        let account = account_id(hotkey)?;
        let key = storage::storage_double_map_key_u16_account(
            PALLET_SUBTENSOR,
            "Axons",
            netuid,
            &account,
        );
        match self.rpc.state_get_storage(&key)? {
            Some(bytes) => storage::decode_axon_info(&bytes).map(Some),
            None => Ok(None),
        }
    }

    /// Enumerate every published axon on `netuid` as `(hotkey, info)`.
    ///
    /// # Errors
    /// Transport or decode failure, or more than [`MAX_NEURONS`] entries.
    pub fn enumerate_axons(&self, netuid: u16) -> Result<Vec<(Vec<u8>, AxonInfo)>, ChainError> {
        let prefix = storage::storage_double_map_prefix_u16(PALLET_SUBTENSOR, "Axons", netuid);
        let all_keys = self.paged_keys(&prefix, "Axons", netuid, None)?;
        if all_keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(all_keys.len());
        for chunk in all_keys.chunks(256) {
            for (key, value) in self.rpc.state_query_storage_at(chunk, None)? {
                out.push((
                    storage::decode_double_map_account_k2(&key)?,
                    storage::decode_axon_info(&value)?,
                ));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// Page `state_getKeysPaged` to completion under `prefix`.
    fn paged_keys(
        &self,
        prefix: &[u8],
        item: &str,
        netuid: u16,
        at: Option<&[u8; 32]>,
    ) -> Result<Vec<Vec<u8>>, ChainError> {
        let mut start = prefix.to_vec();
        let mut all_keys: Vec<Vec<u8>> = Vec::new();
        loop {
            let page = self
                .rpc
                .state_get_keys_paged(prefix, KEYS_PAGE_SIZE, &start, at)?;
            if page.is_empty() {
                break;
            }
            let short = page.len() < KEYS_PAGE_SIZE as usize;
            if let Some(last) = page.last() {
                start.clone_from(last);
            }
            all_keys.extend(page);
            if short {
                break;
            }
            if all_keys.len() > MAX_NEURONS {
                return Err(ChainError::Other(format!(
                    "{PALLET_SUBTENSOR}.{item}({netuid}) exceeded {MAX_NEURONS} entries"
                )));
            }
        }
        Ok(all_keys)
    }

    /// Build and submit a `serve_axon` extrinsic publishing this miner's endpoint.
    ///
    /// Returns the extrinsic hash / subscription ID from the node.
    ///
    /// # Errors
    /// Missing signing key, transport failure, or runtime-version RPC failure.
    pub fn serve_axon(&self, params: &extrinsic::ServeAxonParams) -> Result<String, ChainError> {
        let (spec_version, tx_version) = self.live_runtime_version()?;
        let key = self.require_key()?;
        let genesis_hash = self.block_hash(0)?;
        let pubkey = extrinsic::derive_public_key(&key)?;
        let nonce = self.rpc.system_account_next_index(pubkey)?;
        let ext = extrinsic::build_and_sign_serve_axon(
            &key,
            nonce,
            &Era::Immortal,
            &genesis_hash,
            &genesis_hash,
            spec_version,
            tx_version,
            params,
        )?;
        self.submit_extrinsic(&ext)
    }

    /// Submit a signed extrinsic and return the extrinsic hash.
    fn submit_extrinsic(&self, ext: &[u8]) -> Result<String, ChainError> {
        tracing::debug!(
            len = ext.len(),
            "submitting extrinsic via author_submitExtrinsic"
        );
        self.rpc.author_submit_extrinsic(ext)
    }

    /// `LastUpdate[netuid][uid_of(hotkey)]` — block of the hotkey's last
    /// successful weight dispatch. `None` when the hotkey is unregistered or
    /// has never set weights.
    ///
    /// # Errors
    /// Transport or decode failure.
    fn last_weight_update(&self, netuid: u16, hotkey: &[u8]) -> Result<Option<u64>, ChainError> {
        let account = account_id(hotkey)?;
        let uid_key =
            storage::storage_double_map_key_u16_account(PALLET_SUBTENSOR, "Uids", netuid, &account);
        let Some(uid_bytes) = self.rpc.state_get_storage(&uid_key)? else {
            return Ok(None);
        };
        let uid = storage::decode_u16(&uid_bytes)?;
        let lu_key = storage::storage_map_key_u16(PALLET_SUBTENSOR, "LastUpdate", netuid);
        let Some(lu_bytes) = self.rpc.state_get_storage(&lu_key)? else {
            return Ok(None);
        };
        let blocks = storage::decode_vec_u64(&lu_bytes)?;
        Ok(blocks.get(usize::from(uid)).copied())
    }

    /// `WeightsSetRateLimit(netuid)` with the chain-wide default when the
    /// sparse storage key is absent. Heuristic — only used to classify an
    /// unconfirmed submit for logs/backoff, never to gate submission.
    fn weights_rate_limit(&self, netuid: u16) -> u64 {
        let key = storage::storage_map_key_u16(PALLET_SUBTENSOR, "WeightsSetRateLimit", netuid);
        self.rpc
            .state_get_storage(&key)
            .ok()
            .flatten()
            .and_then(|b| storage::decode_u64(&b).ok())
            .unwrap_or(DEFAULT_WEIGHTS_RATE_LIMIT)
    }

    /// Prove a weight extrinsic dispatched successfully by waiting for the
    /// hotkey's `LastUpdate` entry to advance. `author_submitExtrinsic` is
    /// fire-and-forget: a dispatch failure (e.g. `WeightsSetRateLimited`) is
    /// still included in a block, so pool acceptance alone is not evidence.
    ///
    /// An unchanged `LastUpdate` inside the rate-limit window maps to
    /// [`ChainError::RateLimited`] so callers back off instead of reporting a
    /// false success; any other unchanged outcome is a transient error.
    ///
    /// # Errors
    /// [`ChainError::RateLimited`] inside the rate-limit window,
    /// [`ChainError::Other`] on transport failure or missing confirmation.
    fn confirm_weight_update(
        &self,
        netuid: u16,
        hotkey: &[u8],
        before: Option<u64>,
        start_tip: u64,
    ) -> Result<(), ChainError> {
        let deadline = start_tip.saturating_add(WEIGHT_CONFIRM_BLOCKS);
        loop {
            std::thread::sleep(WEIGHT_CONFIRM_POLL);
            let tip = self.current_block()?;
            let now = self.last_weight_update(netuid, hotkey)?;
            if now != before {
                tracing::info!(
                    ?before,
                    ?now,
                    tip,
                    "weight extrinsic confirmed: LastUpdate advanced"
                );
                return Ok(());
            }
            if tip >= deadline {
                let window = self.weights_rate_limit(netuid);
                if let Some(b) = before {
                    let elapsed = tip.saturating_sub(b);
                    if elapsed < window {
                        return Err(ChainError::RateLimited {
                            retry_after_blocks: Some(window - elapsed),
                        });
                    }
                }
                return Err(ChainError::Other(format!(
                    "weight extrinsic unconfirmed: LastUpdate still {before:?} \
                     {WEIGHT_CONFIRM_BLOCKS} blocks after submit (tip {tip})"
                )));
            }
        }
    }
}

/// Extract `commit_reveal_weights_enabled` from SCALE hyperparams v3 bytes.
///
/// v3 is `Option<Vec<{name: String, value: Enum}>>`. We locate the UTF-8 field
/// name and read the following `Bool` variant (`0x00` tag + `u8` value).
pub(crate) fn parse_commit_reveal_enabled_v3(bytes: &[u8]) -> Option<bool> {
    const NEEDLE: &[u8] = b"commit_reveal_weights_enabled";
    let pos = bytes.windows(NEEDLE.len()).position(|w| w == NEEDLE)?;
    let after = pos.checked_add(NEEDLE.len())?;
    let tag = *bytes.get(after)?;
    let val = *bytes.get(after.checked_add(1)?)?;
    // Variant 0 = Bool in the hyperparam value enum (verified on Finney v445).
    if tag != 0 {
        return None;
    }
    match val {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

fn account_id(hotkey: &[u8]) -> Result<[u8; storage::ACCOUNT_ID_LEN], ChainError> {
    hotkey.try_into().map_err(|_| {
        ChainError::Other(format!(
            "hotkey must be {} bytes, got {}",
            storage::ACCOUNT_ID_LEN,
            hotkey.len()
        ))
    })
}

fn load_signing_key(path: &Path) -> Result<[u8; 32], ChainError> {
    let raw = std::fs::read(path)
        .map_err(|e| ChainError::Other(format!("read key file {}: {e}", path.display())))?;
    let bytes = if raw.len() == crypto::KEY_LEN {
        raw
    } else {
        let s = std::str::from_utf8(&raw)
            .map_err(|e| ChainError::Other(format!("key file not utf8: {e}")))?
            .trim();
        let s = s.strip_prefix("0x").unwrap_or(s);
        hex::decode(s).map_err(|e| ChainError::Other(format!("hex decode key: {e}")))?
    };
    if bytes.len() != crypto::KEY_LEN {
        return Err(ChainError::Other(format!(
            "signing key must be {} bytes, got {}",
            crypto::KEY_LEN,
            bytes.len()
        )));
    }
    let mut key = [0_u8; crypto::KEY_LEN];
    key.copy_from_slice(&bytes);
    Ok(key)
}

impl ChainClient for LiveChainClient {
    fn current_block(&self) -> Result<u64, ChainError> {
        let header = self.rpc.chain_get_header()?;
        let num = header
            .get("number")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ChainError::Other("header.number missing".into()))?;
        let s = num.strip_prefix("0x").unwrap_or(num);
        u64::from_str_radix(s, 16).map_err(|e| ChainError::Other(format!("block number hex: {e}")))
    }

    fn block_hash(&self, n: u64) -> Result<[u8; 32], ChainError> {
        self.rpc.chain_get_block_hash(n)
    }

    fn metagraph_at(&self, block_hash: &[u8; 32]) -> Result<Metagraph, ChainError> {
        self.metagraph_at_cached(block_hash)
    }

    fn subnet_owner_hotkey(&self, netuid: u16) -> Result<Vec<u8>, ChainError> {
        self.read_owner_hotkey(netuid, None)
    }

    fn axon(&self, netuid: u16, hotkey: &[u8]) -> Result<Option<AxonInfo>, ChainError> {
        self.read_axon(netuid, hotkey)
    }

    fn axons(&self, netuid: u16) -> Result<Vec<(Vec<u8>, AxonInfo)>, ChainError> {
        self.enumerate_axons(netuid)
    }

    fn commit_reveal_enabled(&self, netuid: u16) -> Result<bool, ChainError> {
        // Authoritative source is `SubnetInfoRuntimeApi` hyperparams. The
        // `CommitRevealWeightsEnabled` storage map is sparse — netuid 100 is
        // CR-on via the runtime API while the map key is absent (a raw storage
        // read would falsely report disabled and downgrade to `set_weights`).
        if let Ok(Some(v)) = self.commit_reveal_enabled_from_hyperparams(netuid) {
            return Ok(v);
        }
        // Fall back to sparse storage map (unit tests / older runtimes).
        let key =
            storage::storage_map_key_u16(PALLET_SUBTENSOR, "CommitRevealWeightsEnabled", netuid);
        match self.rpc.state_get_storage(&key)? {
            Some(bytes) => storage::decode_bool(&bytes),
            None => Ok(false),
        }
    }

    fn commit_reveal_version(&self, netuid: u16) -> Result<u16, ChainError> {
        let key =
            storage::storage_map_key_u16(PALLET_SUBTENSOR, "CommitRevealWeightsVersion", netuid);
        match self.rpc.state_get_storage(&key)? {
            Some(bytes) => storage::decode_u16(&bytes),
            None => Ok(COMMIT_REVEAL_VERSION),
        }
    }

    fn tempo(&self, netuid: u16) -> Result<u64, ChainError> {
        // No safe default: epoch math is wrong without the real tempo.
        self.read_netuid_u16_required("Tempo", netuid)
            .map(u64::from)
    }

    fn reveal_period_epochs(&self, netuid: u16) -> Result<u64, ChainError> {
        self.read_netuid_u16("RevealPeriodEpochs", netuid, DEFAULT_REVEAL_PERIOD_EPOCHS)
            .map(u64::from)
    }

    fn block_time(&self) -> Result<u64, ChainError> {
        let bytes = self.rpc.state_call("AuraApi_slot_duration", &[])?;
        storage::decode_u64(&bytes)
    }

    fn last_epoch_block(&self, netuid: u16) -> Result<u64, ChainError> {
        self.read_netuid_u64("LastEpochBlock", netuid, 0)
    }

    fn pending_epoch_at(&self, netuid: u16) -> Result<u64, ChainError> {
        self.read_netuid_u64("PendingEpochAt", netuid, 0)
    }

    fn subnet_epoch_index(&self, netuid: u16) -> Result<u64, ChainError> {
        self.read_netuid_u64("SubnetEpochIndex", netuid, 0)
    }

    fn blocks_since_last_step(&self, netuid: u16) -> Result<u64, ChainError> {
        self.read_netuid_u64("BlocksSinceLastStep", netuid, 0)
    }

    fn submit_timelocked_weights(
        &self,
        mecid: u8,
        payload: WeightsTlockPayload,
        reveal_round: u64,
    ) -> Result<(), ChainError> {
        if !self.commit_reveal_enabled(self.netuid)? {
            return Err(ChainError::CommitRevealDisabled {
                alternate: "set_weights",
            });
        }
        let (spec_version, tx_version) = self.live_runtime_version()?;
        let key = self.require_key()?;

        // CRV4: encrypt SCALE WeightsTlockPayload with Drand TLE (same wire
        // format as `bittensor_drand.get_encrypted_commit_v2`). Fail closed —
        // never submit plaintext or downgrade to set_weights while CR is on.
        let plaintext = payload.encode();
        let commit = tlock::encrypt_commit(&plaintext, reveal_round).map_err(|e| {
            ChainError::Other(format!(
                "CRV4 tlock encrypt failed (fail-closed, no set_weights): {e}"
            ))
        })?;

        let genesis_hash = self.block_hash(0)?;
        let pubkey = extrinsic::derive_public_key(&key)?;
        let nonce = self.rpc.system_account_next_index(pubkey)?;
        let ext = extrinsic::build_and_sign_commit_timelocked(
            &key,
            nonce,
            &Era::Immortal,
            &genesis_hash,
            &genesis_hash,
            spec_version,
            tx_version,
            self.netuid,
            mecid,
            &commit,
            reveal_round,
            COMMIT_REVEAL_VERSION,
        )?;
        tracing::info!(
            mecid,
            reveal_round,
            commit_len = commit.len(),
            "submitting commit_timelocked_mechanism_weights"
        );
        let before = self.last_weight_update(self.netuid, &payload.hotkey)?;
        let start_tip = self.current_block()?;
        // Inside WeightsSetRateLimit the dispatch is guaranteed to fail, so
        // return fast instead of pool-submitting a doomed extrinsic and
        // blocking ~4 blocks in confirm. The post-submit confirm stays the
        // source of truth when the window opens mid-flight.
        if let Some(b) = before {
            let window = self.weights_rate_limit(self.netuid);
            let elapsed = start_tip.saturating_sub(b);
            if elapsed < window {
                return Err(ChainError::RateLimited {
                    retry_after_blocks: Some(window - elapsed),
                });
            }
        }
        let tx_hash = self.submit_extrinsic(&ext)?;
        tracing::info!(%tx_hash, "commit accepted by pool; awaiting dispatch confirmation");
        self.confirm_weight_update(self.netuid, &payload.hotkey, before, start_tip)?;
        Ok(())
    }

    fn set_weights(
        &self,
        netuid: u16,
        uids: Vec<u16>,
        values: Vec<u16>,
        version_key: u64,
    ) -> Result<(), ChainError> {
        if self.commit_reveal_enabled(netuid)? {
            return Err(ChainError::Other(
                "set_weights refused: commit_reveal is enabled — never downgrade".into(),
            ));
        }
        let (spec_version, tx_version) = self.live_runtime_version()?;
        let key = self.require_key()?;
        let genesis_hash = self.block_hash(0)?;
        let pubkey = extrinsic::derive_public_key(&key)?;
        let nonce = self.rpc.system_account_next_index(pubkey)?;
        let ext = extrinsic::build_and_sign_set_weights(
            &key,
            nonce,
            &Era::Immortal,
            &genesis_hash,
            &genesis_hash,
            spec_version,
            tx_version,
            netuid,
            &uids,
            &values,
            version_key,
        )?;
        let hotkey = extrinsic::derive_public_key(&key)?;
        let before = self.last_weight_update(netuid, &hotkey)?;
        let start_tip = self.current_block()?;
        // Same pre-submit window check as the timelocked path: fail fast
        // instead of pool-submitting a dispatch that cannot land.
        if let Some(b) = before {
            let window = self.weights_rate_limit(netuid);
            let elapsed = start_tip.saturating_sub(b);
            if elapsed < window {
                return Err(ChainError::RateLimited {
                    retry_after_blocks: Some(window - elapsed),
                });
            }
        }
        let tx_hash = self.submit_extrinsic(&ext)?;
        tracing::info!(%tx_hash, "set_weights accepted by pool; awaiting dispatch confirmation");
        self.confirm_weight_update(netuid, &hotkey, before, start_tip)?;
        Ok(())
    }
}
