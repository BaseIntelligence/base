//! Live Bittensor JSON-RPC chain client with sr25519 signed extrinsics.
//!
//! Re-exports [`ChainClient`], [`ChainError`], [`Metagraph`], [`WeightsTlockPayload`]
//! from the `chain` crate and provides [`LiveChainClient`] — a full
//! implementation backed by blocking HTTPS JSON-RPC.

#![forbid(unsafe_code)]

mod extrinsic;
mod rpc;
mod storage;

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
    decode_hotkey, decode_metagraph, decode_u16, decode_u64, decode_vec_vec_u8,
    storage_double_map_key_u16_account, storage_double_map_key_u16_u16,
    storage_double_map_prefix_u16, storage_key, storage_map_key_identity, storage_map_key_twox64,
    storage_map_key_u16, ACCOUNT_ID_LEN,
};

use std::path::Path;

/// Lockfile-pinned spec version (`metadata/testnet.lock` → 443).
const SPEC_VERSION: u32 = 443;
/// Lockfile-pinned transaction version (1).
const TRANSACTION_VERSION: u32 = 1;
/// Lockfile-pinned commit-reveal version (CRV4 = 4).
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

/// Live chain client with sr25519 signed extrinsic submission.
pub struct LiveChainClient {
    rpc: LiveChainRpc,
    netuid: u16,
    spec_version: u32,
    tx_version: u32,
    signing_key: Option<[u8; 32]>,
}

impl std::fmt::Debug for LiveChainClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiveChainClient")
            .field("netuid", &self.netuid)
            .field("spec_version", &self.spec_version)
            .field("tx_version", &self.tx_version)
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
    /// # Errors
    /// HTTP client build failure.
    pub fn connect(endpoint: &str) -> Result<Self, ChainError> {
        let rpc = LiveChainRpc::connect(endpoint)?;
        Ok(Self {
            rpc,
            netuid: DEFAULT_NETUID,
            spec_version: SPEC_VERSION,
            tx_version: TRANSACTION_VERSION,
            signing_key: None,
        })
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

    /// Check live runtime version against lockfile pins before signing.
    fn check_runtime_version(&self) -> Result<(), ChainError> {
        let rt = self.rpc.state_get_runtime_version()?;
        tracing::debug!(
            spec = rt.spec_version,
            tx = rt.transaction_version,
            "runtime version check"
        );
        if rt.spec_version != self.spec_version {
            return Err(ChainError::Other(format!(
                "spec_version mismatch: live {} vs lockfile {} — refusing to sign",
                rt.spec_version, self.spec_version
            )));
        }
        if rt.transaction_version != self.tx_version {
            return Err(ChainError::Other(format!(
                "transaction_version mismatch: live {} vs lockfile {} — refusing to sign",
                rt.transaction_version, self.tx_version
            )));
        }
        Ok(())
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
    /// Runtime-version drift, missing signing key, transport failure.
    pub fn serve_axon(&self, params: &extrinsic::ServeAxonParams) -> Result<String, ChainError> {
        self.check_runtime_version()?;
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
            self.spec_version,
            self.tx_version,
            params,
        )?;
        self.submit_extrinsic(&ext)
    }

    /// Submit a signed extrinsic and return the extrinsic hash.
    fn submit_extrinsic(&self, ext: &[u8]) -> Result<String, ChainError> {
        tracing::debug!(len = ext.len(), "submitting extrinsic via author_submitExtrinsic");
        self.rpc.author_submit_extrinsic(ext)
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
        let netuid = self.netuid;
        // A zero hash means "tip"; callers that do not track a block pass it.
        let at = if block_hash == &[0_u8; 32] {
            None
        } else {
            Some(block_hash)
        };
        let keys = self.enumerate_hotkeys(netuid, at)?;
        let owner = self.read_owner_hotkey(netuid, at)?;
        Ok(storage::decode_metagraph(keys, owner, netuid))
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
        let key =
            storage::storage_map_key_u16(PALLET_SUBTENSOR, "CommitRevealWeightsEnabled", netuid);
        // `ValueQuery` with a `false` default: an absent key means disabled.
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
        self.check_runtime_version()?;
        let key = self.require_key()?;
        let genesis_hash = self.block_hash(0)?;
        let pubkey = extrinsic::derive_public_key(&key)?;
        let nonce = self.rpc.system_account_next_index(pubkey)?;
        let ext = extrinsic::build_and_sign_commit_timelocked(
            &key,
            nonce,
            &Era::Immortal,
            &genesis_hash,
            &genesis_hash,
            self.spec_version,
            self.tx_version,
            mecid,
            &payload,
            reveal_round,
        )?;
        self.submit_extrinsic(&ext)?;
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
        self.check_runtime_version()?;
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
            self.spec_version,
            self.tx_version,
            netuid,
            &uids,
            &values,
            version_key,
        )?;
        self.submit_extrinsic(&ext)?;
        Ok(())
    }
}
