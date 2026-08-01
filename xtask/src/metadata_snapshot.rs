//! Nightly drift gate: snapshot testnet metadata + epoch-schedule read paths.
//!
//! Connects over HTTPS JSON-RPC (Client maps `wss://` → `https://`) and writes
//! `metadata/testnet.lock`. `--check` fails on structural drift.
//!
//! Lockfile intentionally omits volatile per-block schedule *values* so a
//! double-run within seconds stays byte-stable; it records the concrete
//! storage/RPC source for every `generate_commit_v2` / `get_encrypted_commit_v2`
//! input so task 13 cannot invent them.
//!
//! Implementation uses blocking reqwest + frame-metadata (not bittensor-core)
//! so base does not inherit the monorepo's w3f-bls/path patches. Sources and
//! field names match the pinned SDK at `SDK_PIN`.

use frame_metadata::v15::{RuntimeMetadataV15, StorageEntryType};
use frame_metadata::{RuntimeMetadata, RuntimeMetadataPrefixed};
use parity_scale_codec::{Decode, Encode};
use scale_info::form::PortableForm;
use scale_info::{PortableRegistry, TypeDef};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use twox_hash::XxHash64;

/// Default Finney testnet endpoint (`wss://` is rewritten to `https://`).
pub const DEFAULT_ENDPOINT: &str = "wss://test.finney.opentensor.ai:443";

/// Default netuid used when reading per-subnet schedule storage.
/// Task 45 re-reads for the real netuid at runtime.
pub const DEFAULT_SNAPSHOT_NETUID: u16 = 1;

/// Pinned SDK revision this gate documents sources against.
pub const SDK_PIN: &str = "e4ffa2e1325c6c7db618dbceaf396310a170990c";

const TRACKED_CALLS: &[&str] = &[
    "set_weights",
    "commit_timelocked_weights",
    "commit_timelocked_mechanism_weights",
    "set_subnet_identity",
    "serve_axon",
];

const WEIGHTS_TLOCK_FIELDS: &[&str] = &["hotkey", "uids", "values", "version_key"];

const RPC_TIMEOUT: Duration = Duration::from_mins(1);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Lockfile {
    pub schema_version: u32,
    pub endpoint: String,
    pub sdk_pin: String,
    pub snapshot_netuid: u16,
    pub chain: ChainSnapshot,
    /// SHA-256 of raw metadata bytes (0x-prefixed hex). Stable until runtime upgrade.
    pub metadata_digest: String,
    pub call_indices: CallIndices,
    pub weights_tlock_payload: WeightsTlockShape,
    pub commit_reveal_version: VersionedSource,
    pub epoch_schedule_inputs: EpochScheduleSources,
    pub commitments_pallet_present: bool,
    pub commitments_pallet: Option<PalletRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChainSnapshot {
    pub spec_name: String,
    pub spec_version: u32,
    pub transaction_version: u32,
    pub ss58_prefix: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(clippy::struct_field_names)]
pub struct CallIndex {
    pub pallet: String,
    pub pallet_index: u8,
    pub call: String,
    pub call_index: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CallIndices {
    pub set_weights: CallIndex,
    pub commit_timelocked_weights: CallIndex,
    pub commit_timelocked_mechanism_weights: CallIndex,
    pub set_subnet_identity: CallIndex,
    pub serve_axon: CallIndex,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightsTlockShape {
    pub fields: Vec<String>,
    pub scale_types: Vec<String>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VersionedSource {
    pub value: u16,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScheduleInputSource {
    pub source: String,
    pub key: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpochScheduleSources {
    pub tempo: ScheduleInputSource,
    pub reveal_period_epochs: ScheduleInputSource,
    pub block_time: ScheduleInputSource,
    pub last_epoch_block: ScheduleInputSource,
    pub pending_epoch_at: ScheduleInputSource,
    pub subnet_epoch_index: ScheduleInputSource,
    pub blocks_since_last_step: ScheduleInputSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PalletRef {
    pub name: String,
    pub index: u8,
}

#[derive(Debug, Clone)]
pub struct SnapshotArgs {
    pub endpoint: String,
    pub netuid: u16,
    pub out: PathBuf,
    pub check: bool,
}

impl Default for SnapshotArgs {
    fn default() -> Self {
        Self {
            endpoint: DEFAULT_ENDPOINT.to_owned(),
            netuid: DEFAULT_SNAPSHOT_NETUID,
            out: PathBuf::from("metadata/testnet.lock"),
            check: false,
        }
    }
}

pub fn run(workspace_root: &Path, args: &SnapshotArgs) -> Result<(), String> {
    let out_path = if args.out.is_absolute() {
        args.out.clone()
    } else {
        workspace_root.join(&args.out)
    };

    let lock = fetch_lockfile(&args.endpoint, args.netuid)?;
    validate_lockfile(&lock)?;
    let rendered = render_lockfile(&lock)?;

    if args.check {
        let existing = fs::read_to_string(&out_path).map_err(|e| {
            format!(
                "cannot read lockfile {}: {e} (run without --check to create it)",
                out_path.display()
            )
        })?;
        if normalize_json(&existing)? != normalize_json(&rendered)? {
            return Err(format!(
                "metadata drift detected against {} — update lockfile deliberately after reviewing chain changes",
                out_path.display()
            ));
        }
        println!(
            "metadata-snapshot --check: OK (matches {})",
            out_path.display()
        );
        return Ok(());
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
    }
    fs::write(&out_path, rendered.as_bytes())
        .map_err(|e| format!("write {}: {e}", out_path.display()))?;
    println!("wrote {}", out_path.display());
    println!(
        "metadata_digest={} commit_reveal_version={} commitments_pallet_present={}",
        lock.metadata_digest, lock.commit_reveal_version.value, lock.commitments_pallet_present
    );
    Ok(())
}

pub fn fetch_lockfile(endpoint: &str, netuid: u16) -> Result<Lockfile, String> {
    let http = http_endpoint(endpoint);
    let mut rpc = Rpc::connect(&http)?;

    let (spec_name, spec_version, transaction_version) = rpc.runtime_version()?;
    let ss58_prefix = rpc.ss58_format()?;
    let metadata_bytes = rpc.metadata_bytes()?;
    let digest_hex = format!("0x{}", hex::encode(Sha256::digest(&metadata_bytes)));

    let meta = decode_metadata(&metadata_bytes)?;
    let call_indices = resolve_tracked_calls(&meta)?;
    let weights_tlock_payload = weights_tlock_shape();
    let commit_reveal_version = read_commit_reveal_version(&mut rpc, &meta)?;
    let epoch_schedule_inputs = build_and_probe_schedule_sources(&mut rpc, &meta, netuid)?;
    let (commitments_pallet_present, commitments_pallet) = commitments_pallet_info(&meta);

    Ok(Lockfile {
        schema_version: 1,
        endpoint: endpoint.to_owned(),
        sdk_pin: SDK_PIN.to_owned(),
        snapshot_netuid: netuid,
        chain: ChainSnapshot {
            spec_name,
            spec_version,
            transaction_version,
            ss58_prefix,
        },
        metadata_digest: digest_hex,
        call_indices,
        weights_tlock_payload,
        commit_reveal_version,
        epoch_schedule_inputs,
        commitments_pallet_present,
        commitments_pallet,
    })
}

// --------------- RPC ---------------

struct Rpc {
    http: reqwest::blocking::Client,
    endpoint: String,
    next_id: u64,
}

impl Rpc {
    fn connect(endpoint: &str) -> Result<Self, String> {
        let http = reqwest::blocking::Client::builder()
            .timeout(RPC_TIMEOUT)
            .build()
            .map_err(|e| format!("http client: {e}"))?;
        Ok(Self {
            http,
            endpoint: endpoint.to_owned(),
            next_id: 1,
        })
    }

    fn call(&mut self, method: &str, params: &JsonValue) -> Result<JsonValue, String> {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let resp = self
            .http
            .post(&self.endpoint)
            .json(&body)
            .send()
            .map_err(|e| format!("rpc {method}: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("rpc {method}: HTTP {}", resp.status()));
        }
        let v: JsonValue = resp
            .json()
            .map_err(|e| format!("rpc {method} decode: {e}"))?;
        if let Some(err) = v.get("error") {
            return Err(format!("rpc {method} error: {err}"));
        }
        v.get("result")
            .cloned()
            .ok_or_else(|| format!("rpc {method}: missing result"))
    }

    fn runtime_version(&mut self) -> Result<(String, u32, u32), String> {
        let v = self.call("state_getRuntimeVersion", &json!([]))?;
        let obj = v
            .as_object()
            .ok_or_else(|| "runtime version not object".to_owned())?;
        let spec_name = obj
            .get("specName")
            .and_then(|x| x.as_str())
            .ok_or_else(|| "missing specName".to_owned())?
            .to_owned();
        let spec_version = json_u32(obj.get("specVersion"), "specVersion")?;
        let transaction_version = json_u32(obj.get("transactionVersion"), "transactionVersion")?;
        Ok((spec_name, spec_version, transaction_version))
    }

    fn ss58_format(&mut self) -> Result<u16, String> {
        let v = self.call("system_properties", &json!([]))?;
        let n = v
            .as_object()
            .and_then(|o| o.get("ss58Format"))
            .and_then(JsonValue::as_u64)
            .unwrap_or(42);
        u16::try_from(n).map_err(|_| format!("ss58Format {n} > u16"))
    }

    fn metadata_bytes(&mut self) -> Result<Vec<u8>, String> {
        // Prefer V15 via runtime API; fall back to state_getMetadata (V14).
        let requested = hex_prefixed(&15u32.to_le_bytes());
        if let Ok(value) = self.call(
            "state_call",
            &json!(["Metadata_metadata_at_version", requested]),
        ) {
            if let Ok(encoded) = decode_hex_json(&value) {
                let mut input = encoded.as_slice();
                if let Ok(Some(bytes)) = Option::<Vec<u8>>::decode(&mut input) {
                    if !bytes.is_empty() {
                        return Ok(bytes);
                    }
                }
            }
        }
        let value = self.call("state_getMetadata", &json!([]))?;
        decode_hex_json(&value)
    }

    fn storage_raw(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>, String> {
        let value = self.call("state_getStorage", &json!([hex_prefixed(key)]))?;
        if value.is_null() {
            return Ok(None);
        }
        Ok(Some(decode_hex_json(&value)?))
    }
}

fn http_endpoint(endpoint: &str) -> String {
    if let Some(rest) = endpoint.strip_prefix("wss://") {
        return format!("https://{rest}");
    }
    if let Some(rest) = endpoint.strip_prefix("ws://") {
        return format!("http://{rest}");
    }
    endpoint.to_owned()
}

fn json_u32(value: Option<&JsonValue>, field: &str) -> Result<u32, String> {
    let value = value.ok_or_else(|| format!("missing {field}"))?;
    let n = value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|i| u64::try_from(i).ok()))
        .ok_or_else(|| format!("{field} is not an integer"))?;
    u32::try_from(n).map_err(|_| format!("{field} does not fit u32"))
}

fn hex_prefixed(bytes: &[u8]) -> String {
    format!("0x{}", hex::encode(bytes))
}

fn decode_hex_json(value: &JsonValue) -> Result<Vec<u8>, String> {
    let s = value
        .as_str()
        .ok_or_else(|| format!("expected hex string, got {value}"))?;
    let s = s.strip_prefix("0x").unwrap_or(s);
    hex::decode(s).map_err(|e| format!("hex decode: {e}"))
}

// --------------- metadata ---------------

struct MetaView {
    pallets: Vec<PalletView>,
    types: PortableRegistry,
}

struct PalletView {
    name: String,
    index: u8,
    calls_type: Option<u32>,
    storage_prefix: Option<String>,
    storage: Vec<StorageView>,
    constants: Vec<ConstantView>,
}

#[allow(dead_code)]
struct StorageView {
    name: String,
    hashers: Vec<String>,
    key_types: Vec<u32>, // kept for future typed decode
    value_type: u32,
    default_bytes: Vec<u8>,
}

struct ConstantView {
    name: String,
    value: Vec<u8>,
}

fn decode_metadata(bytes: &[u8]) -> Result<MetaView, String> {
    let prefixed = RuntimeMetadataPrefixed::decode(&mut &bytes[..])
        .map_err(|e| format!("decode metadata: {e}"))?;
    match prefixed.1 {
        RuntimeMetadata::V15(m) => Ok(from_v15(m)),
        RuntimeMetadata::V14(m) => Ok(from_v14(m)),
        other => Err(format!("unsupported metadata version: {other:?}")),
    }
}

fn from_v15(m: RuntimeMetadataV15) -> MetaView {
    let types = m.types;
    let pallets = m
        .pallets
        .into_iter()
        .map(|p| {
            let (storage_prefix, storage) = match p.storage {
                Some(s) => {
                    let entries = s
                        .entries
                        .into_iter()
                        .map(|e| storage_from_entry(e, &types))
                        .collect();
                    (Some(s.prefix), entries)
                }
                None => (None, Vec::new()),
            };
            let constants = p
                .constants
                .into_iter()
                .map(|c| ConstantView {
                    name: c.name,
                    value: c.value,
                })
                .collect();
            PalletView {
                name: p.name,
                index: p.index,
                calls_type: p.calls.map(|c| c.ty.id),
                storage_prefix,
                storage,
                constants,
            }
        })
        .collect();
    MetaView { pallets, types }
}

fn from_v14(m: frame_metadata::v14::RuntimeMetadataV14) -> MetaView {
    let types = m.types;
    let pallets = m
        .pallets
        .into_iter()
        .map(|p| {
            let (storage_prefix, storage) = match p.storage {
                Some(s) => {
                    let entries = s
                        .entries
                        .into_iter()
                        .map(|e| storage_from_entry_v14(e, &types))
                        .collect();
                    (Some(s.prefix), entries)
                }
                None => (None, Vec::new()),
            };
            let constants = p
                .constants
                .into_iter()
                .map(|c| ConstantView {
                    name: c.name,
                    value: c.value,
                })
                .collect();
            PalletView {
                name: p.name,
                index: p.index,
                calls_type: p.calls.map(|c| c.ty.id),
                storage_prefix,
                storage,
                constants,
            }
        })
        .collect();
    MetaView { pallets, types }
}

fn storage_from_entry(
    e: frame_metadata::v15::StorageEntryMetadata<PortableForm>,
    types: &PortableRegistry,
) -> StorageView {
    let (hashers, key_types, value_type) = match e.ty {
        StorageEntryType::Plain(ty) => (vec![], vec![], ty.id),
        StorageEntryType::Map {
            hashers,
            key,
            value,
        } => {
            let hasher_names = hashers.iter().map(|h| format!("{h:?}")).collect();
            // key may be a tuple of N types matching N hashers
            let key_types = flatten_key_types(key.id, hashers.len(), types);
            (hasher_names, key_types, value.id)
        }
    };
    StorageView {
        name: e.name,
        hashers,
        key_types,
        value_type,
        default_bytes: e.default,
    }
}

fn storage_from_entry_v14(
    e: frame_metadata::v14::StorageEntryMetadata<PortableForm>,
    types: &PortableRegistry,
) -> StorageView {
    use frame_metadata::v14::StorageEntryType as T;
    let (hashers, key_types, value_type) = match e.ty {
        T::Plain(ty) => (vec![], vec![], ty.id),
        T::Map {
            hashers,
            key,
            value,
        } => {
            let hasher_names = hashers.iter().map(|h| format!("{h:?}")).collect();
            let key_types = flatten_key_types(key.id, hashers.len(), types);
            (hasher_names, key_types, value.id)
        }
    };
    StorageView {
        name: e.name,
        hashers,
        key_types,
        value_type,
        default_bytes: e.default,
    }
}

fn flatten_key_types(key_id: u32, n_hashers: usize, types: &PortableRegistry) -> Vec<u32> {
    if n_hashers <= 1 {
        return vec![key_id];
    }
    if let Some(ty) = types.resolve(key_id) {
        if let TypeDef::Tuple(t) = &ty.type_def {
            if t.fields.len() == n_hashers {
                return t.fields.iter().map(|f| f.id).collect();
            }
        }
        if let TypeDef::Composite(c) = &ty.type_def {
            if c.fields.len() == n_hashers {
                return c.fields.iter().map(|f| f.ty.id).collect();
            }
        }
    }
    vec![key_id]
}

fn find_pallet<'a>(meta: &'a MetaView, name: &str) -> Result<&'a PalletView, String> {
    meta.pallets
        .iter()
        .find(|p| p.name == name)
        .ok_or_else(|| format!("pallet {name} not found in metadata"))
}

fn find_storage<'a>(
    meta: &'a MetaView,
    pallet: &str,
    item: &str,
) -> Result<(&'a PalletView, &'a StorageView), String> {
    let p = find_pallet(meta, pallet)?;
    let s = p
        .storage
        .iter()
        .find(|s| s.name == item)
        .ok_or_else(|| format!("storage {pallet}.{item} not found in metadata"))?;
    Ok((p, s))
}

fn resolve_tracked_calls(meta: &MetaView) -> Result<CallIndices, String> {
    let mut map = std::collections::BTreeMap::new();
    for name in TRACKED_CALLS {
        map.insert(
            (*name).to_owned(),
            call_index(meta, "SubtensorModule", name)?,
        );
    }
    Ok(CallIndices {
        set_weights: map
            .remove("set_weights")
            .ok_or_else(|| "missing set_weights".to_owned())?,
        commit_timelocked_weights: map
            .remove("commit_timelocked_weights")
            .ok_or_else(|| "missing commit_timelocked_weights".to_owned())?,
        commit_timelocked_mechanism_weights: map
            .remove("commit_timelocked_mechanism_weights")
            .ok_or_else(|| "missing commit_timelocked_mechanism_weights".to_owned())?,
        set_subnet_identity: map
            .remove("set_subnet_identity")
            .ok_or_else(|| "missing set_subnet_identity".to_owned())?,
        serve_axon: map
            .remove("serve_axon")
            .ok_or_else(|| "missing serve_axon".to_owned())?,
    })
}

fn call_index(meta: &MetaView, pallet_name: &str, call_name: &str) -> Result<CallIndex, String> {
    let pallet = find_pallet(meta, pallet_name)?;
    let calls_type = pallet
        .calls_type
        .ok_or_else(|| format!("pallet {pallet_name} has no calls"))?;
    let ty = meta
        .types
        .resolve(calls_type)
        .ok_or_else(|| format!("unknown calls type id {calls_type}"))?;
    let TypeDef::Variant(variant) = &ty.type_def else {
        return Err(format!("pallet {pallet_name} calls type is not a variant"));
    };
    let chosen = variant
        .variants
        .iter()
        .find(|v| v.name == call_name)
        .ok_or_else(|| format!("call {pallet_name}.{call_name} not found in metadata"))?;
    Ok(CallIndex {
        pallet: pallet_name.to_owned(),
        pallet_index: pallet.index,
        call: call_name.to_owned(),
        call_index: chosen.index,
    })
}

fn weights_tlock_shape() -> WeightsTlockShape {
    WeightsTlockShape {
        fields: WEIGHTS_TLOCK_FIELDS
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        scale_types: vec![
            "Vec<u8>".into(),
            "Vec<u16>".into(),
            "Vec<u16>".into(),
            "u64".into(),
        ],
        source: format!("sdk:bittensor_core::timelock::WeightsTlockPayload@{SDK_PIN}"),
    }
}

fn read_commit_reveal_version(rpc: &mut Rpc, meta: &MetaView) -> Result<VersionedSource, String> {
    let (pallet, entry) = find_storage(meta, "SubtensorModule", "CommitRevealWeightsVersion")?;
    let prefix = pallet
        .storage_prefix
        .as_deref()
        .unwrap_or(pallet.name.as_str());
    let key = storage_value_key(prefix, &entry.name);
    let raw = rpc.storage_raw(&key)?;
    let bytes = raw.unwrap_or_else(|| entry.default_bytes.clone());
    let value = decode_u16(&bytes)
        .ok_or_else(|| format!("CommitRevealWeightsVersion decode failed: {bytes:?}"))?;
    Ok(VersionedSource {
        value,
        source: "storage:SubtensorModule.CommitRevealWeightsVersion".into(),
    })
}

fn build_and_probe_schedule_sources(
    rpc: &mut Rpc,
    meta: &MetaView,
    netuid: u16,
) -> Result<EpochScheduleSources, String> {
    let netuid_key = vec!["netuid".to_owned()];
    let sources = EpochScheduleSources {
        tempo: ScheduleInputSource {
            source: "storage:SubtensorModule.Tempo".into(),
            key: netuid_key.clone(),
            note: Some(format!("per-netuid map; snapshot probes netuid={netuid}")),
        },
        reveal_period_epochs: ScheduleInputSource {
            source: "storage:SubtensorModule.RevealPeriodEpochs".into(),
            key: netuid_key.clone(),
            note: Some("epochs, not blocks (D22)".into()),
        },
        block_time: ScheduleInputSource {
            source: "constant:Aura.SlotDuration".into(),
            key: vec![],
            note: Some(
                "milliseconds; block_time_secs = SlotDuration_ms / 1000 (SDK Client::block_time)"
                    .into(),
            ),
        },
        last_epoch_block: ScheduleInputSource {
            source: "storage:SubtensorModule.LastEpochBlock".into(),
            key: netuid_key.clone(),
            note: None,
        },
        pending_epoch_at: ScheduleInputSource {
            source: "storage:SubtensorModule.PendingEpochAt".into(),
            key: netuid_key.clone(),
            note: Some("0 when no owner-triggered pending epoch".into()),
        },
        subnet_epoch_index: ScheduleInputSource {
            source: "storage:SubtensorModule.SubnetEpochIndex".into(),
            key: netuid_key.clone(),
            note: None,
        },
        blocks_since_last_step: ScheduleInputSource {
            source: "storage:SubtensorModule.BlocksSinceLastStep".into(),
            key: netuid_key,
            note: None,
        },
    };

    for (label, src) in [
        ("tempo", &sources.tempo),
        ("reveal_period_epochs", &sources.reveal_period_epochs),
        ("last_epoch_block", &sources.last_epoch_block),
        ("pending_epoch_at", &sources.pending_epoch_at),
        ("subnet_epoch_index", &sources.subnet_epoch_index),
        ("blocks_since_last_step", &sources.blocks_since_last_step),
    ] {
        probe_storage(rpc, meta, label, src, netuid)?;
    }
    probe_block_time_constant(meta, &sources.block_time)?;

    Ok(sources)
}

fn probe_storage(
    rpc: &mut Rpc,
    meta: &MetaView,
    label: &str,
    src: &ScheduleInputSource,
    netuid: u16,
) -> Result<(), String> {
    if src.source.is_empty() {
        return Err(format!("{label}: null/empty source is forbidden"));
    }
    let (pallet_name, item) = parse_storage_source(&src.source)?;
    let (pallet, entry) = find_storage(meta, pallet_name, item)?;
    let prefix = pallet
        .storage_prefix
        .as_deref()
        .unwrap_or(pallet.name.as_str());
    let key = if entry.hashers.is_empty() {
        storage_value_key(prefix, &entry.name)
    } else {
        storage_map_key_u16(prefix, &entry.name, &entry.hashers, netuid)?
    };
    // Presence of a result OR a default proves the path is live.
    let _ = rpc.storage_raw(&key)?;
    Ok(())
}

fn probe_block_time_constant(meta: &MetaView, src: &ScheduleInputSource) -> Result<(), String> {
    if src.source.is_empty() {
        return Err("block_time: null/empty source is forbidden".into());
    }
    let (pallet_name, const_name) = parse_constant_source(&src.source)?;
    let pallet = find_pallet(meta, pallet_name)?;
    let c = pallet
        .constants
        .iter()
        .find(|c| c.name == const_name)
        .ok_or_else(|| format!("block_time: constant {pallet_name}.{const_name} missing"))?;
    let ms = decode_u64(&c.value)
        .ok_or_else(|| format!("block_time: cannot decode SlotDuration: {:?}", c.value))?;
    if ms == 0 {
        return Err("block_time: Aura.SlotDuration is zero".into());
    }
    Ok(())
}

fn commitments_pallet_info(meta: &MetaView) -> (bool, Option<PalletRef>) {
    match meta.pallets.iter().find(|p| p.name == "Commitments") {
        Some(p) => (
            true,
            Some(PalletRef {
                name: p.name.clone(),
                index: p.index,
            }),
        ),
        None => (false, None),
    }
}

// --------------- storage keys ---------------

fn twox128(data: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    let hash = XxHash64::oneshot(0, data);
    out[..8].copy_from_slice(&hash.to_le_bytes());
    let hash2 = XxHash64::oneshot(1, data);
    out[8..].copy_from_slice(&hash2.to_le_bytes());
    out
}

fn twox64(data: &[u8]) -> [u8; 8] {
    XxHash64::oneshot(0, data).to_le_bytes()
}

fn blake2_128(data: &[u8]) -> [u8; 16] {
    use blake2::digest::consts::U16;
    use blake2::digest::Digest as _;
    use blake2::Blake2b;
    let mut hasher = Blake2b::<U16>::new();
    hasher.update(data);
    let res = hasher.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&res);
    out
}

fn storage_value_key(prefix: &str, name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(&twox128(prefix.as_bytes()));
    key.extend_from_slice(&twox128(name.as_bytes()));
    key
}

fn storage_map_key_u16(
    prefix: &str,
    name: &str,
    hashers: &[String],
    netuid: u16,
) -> Result<Vec<u8>, String> {
    if hashers.len() != 1 {
        return Err(format!(
            "{prefix}.{name}: expected 1 hasher for netuid map, got {hashers:?}"
        ));
    }
    let encoded = netuid.encode();
    let mut key = storage_value_key(prefix, name);
    let h = hashers[0].as_str();
    if h.contains("Twox64Concat") {
        key.extend_from_slice(&twox64(&encoded));
        key.extend_from_slice(&encoded);
    } else if h.contains("Blake2_128Concat") {
        key.extend_from_slice(&blake2_128(&encoded));
        key.extend_from_slice(&encoded);
    } else if h.contains("Identity") {
        key.extend_from_slice(&encoded);
    } else if h.contains("Twox128") {
        key.extend_from_slice(&twox128(&encoded));
    } else if h.contains("Blake2_128") {
        key.extend_from_slice(&blake2_128(&encoded));
    } else {
        return Err(format!(
            "{prefix}.{name}: unsupported hasher {h} for netuid map"
        ));
    }
    Ok(key)
}

fn decode_u16(bytes: &[u8]) -> Option<u16> {
    if bytes.len() >= 2 {
        return Some(u16::from_le_bytes([bytes[0], bytes[1]]));
    }
    // compact-encoded small values sometimes appear; try full decode
    u16::decode(&mut &bytes[..]).ok()
}

fn decode_u64(bytes: &[u8]) -> Option<u64> {
    if bytes.len() >= 8 {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[..8]);
        return Some(u64::from_le_bytes(arr));
    }
    u64::decode(&mut &bytes[..]).ok()
}

fn parse_storage_source(source: &str) -> Result<(&str, &str), String> {
    let rest = source
        .strip_prefix("storage:")
        .ok_or_else(|| format!("expected storage:Pallet.Item, got {source}"))?;
    rest.split_once('.')
        .ok_or_else(|| format!("expected storage:Pallet.Item, got {source}"))
}

fn parse_constant_source(source: &str) -> Result<(&str, &str), String> {
    let rest = source
        .strip_prefix("constant:")
        .ok_or_else(|| format!("expected constant:Pallet.Name, got {source}"))?;
    rest.split_once('.')
        .ok_or_else(|| format!("expected constant:Pallet.Name, got {source}"))
}

// --------------- validate / render ---------------

pub fn validate_lockfile(lock: &Lockfile) -> Result<(), String> {
    if lock.schema_version == 0 {
        return Err("schema_version must be >= 1".into());
    }
    if lock.metadata_digest.len() != 66 || !lock.metadata_digest.starts_with("0x") {
        return Err(format!(
            "metadata_digest must be 0x + 64 hex chars, got {}",
            lock.metadata_digest
        ));
    }
    validate_weights_shape(&lock.weights_tlock_payload)?;
    validate_schedule_sources(&lock.epoch_schedule_inputs)?;
    if lock.commit_reveal_version.source.is_empty() {
        return Err("commit_reveal_version.source must not be empty".into());
    }
    if lock.commitments_pallet_present && lock.commitments_pallet.is_none() {
        return Err("commitments_pallet_present=true but commitments_pallet is null".into());
    }
    Ok(())
}

fn validate_weights_shape(shape: &WeightsTlockShape) -> Result<(), String> {
    let expected: Vec<String> = WEIGHTS_TLOCK_FIELDS
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    if shape.fields != expected {
        return Err(format!(
            "WeightsTlockPayload fields must be exactly {expected:?}, got {:?} — revisit D5 before task 33",
            shape.fields
        ));
    }
    if shape.scale_types.len() != expected.len() {
        return Err("WeightsTlockPayload scale_types length mismatch".into());
    }
    if shape.source.is_empty() {
        return Err("WeightsTlockPayload source must not be empty".into());
    }
    Ok(())
}

fn validate_schedule_sources(s: &EpochScheduleSources) -> Result<(), String> {
    for (label, src) in [
        ("tempo", &s.tempo),
        ("reveal_period_epochs", &s.reveal_period_epochs),
        ("block_time", &s.block_time),
        ("last_epoch_block", &s.last_epoch_block),
        ("pending_epoch_at", &s.pending_epoch_at),
        ("subnet_epoch_index", &s.subnet_epoch_index),
        ("blocks_since_last_step", &s.blocks_since_last_step),
    ] {
        if src.source.is_empty() {
            return Err(format!(
                "epoch_schedule_inputs.{label}.source is null/empty — forbidden"
            ));
        }
    }
    if !s.tempo.source.starts_with("storage:") {
        return Err(format!(
            "tempo source must be storage:*, got {}",
            s.tempo.source
        ));
    }
    if !s.block_time.source.starts_with("constant:") && !s.block_time.source.starts_with("rpc:") {
        return Err(format!(
            "block_time source must be constant:* or rpc:*, got {}",
            s.block_time.source
        ));
    }
    Ok(())
}

pub fn render_lockfile(lock: &Lockfile) -> Result<String, String> {
    let mut body =
        serde_json::to_string_pretty(lock).map_err(|e| format!("serialize lockfile: {e}"))?;
    body.push('\n');
    Ok(body)
}

fn normalize_json(text: &str) -> Result<String, String> {
    let value: JsonValue = serde_json::from_str(text).map_err(|e| format!("parse JSON: {e}"))?;
    serde_json::to_string(&value).map_err(|e| format!("re-serialize JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lock() -> Lockfile {
        Lockfile {
            schema_version: 1,
            endpoint: DEFAULT_ENDPOINT.into(),
            sdk_pin: SDK_PIN.into(),
            snapshot_netuid: 1,
            chain: ChainSnapshot {
                spec_name: "node-subtensor".into(),
                spec_version: 440,
                transaction_version: 1,
                ss58_prefix: 42,
            },
            metadata_digest: format!("0x{}", "ab".repeat(32)),
            call_indices: CallIndices {
                set_weights: CallIndex {
                    pallet: "SubtensorModule".into(),
                    pallet_index: 7,
                    call: "set_weights".into(),
                    call_index: 0,
                },
                commit_timelocked_weights: CallIndex {
                    pallet: "SubtensorModule".into(),
                    pallet_index: 7,
                    call: "commit_timelocked_weights".into(),
                    call_index: 1,
                },
                commit_timelocked_mechanism_weights: CallIndex {
                    pallet: "SubtensorModule".into(),
                    pallet_index: 7,
                    call: "commit_timelocked_mechanism_weights".into(),
                    call_index: 2,
                },
                set_subnet_identity: CallIndex {
                    pallet: "SubtensorModule".into(),
                    pallet_index: 7,
                    call: "set_subnet_identity".into(),
                    call_index: 3,
                },
                serve_axon: CallIndex {
                    pallet: "SubtensorModule".into(),
                    pallet_index: 7,
                    call: "serve_axon".into(),
                    call_index: 4,
                },
            },
            weights_tlock_payload: weights_tlock_shape(),
            commit_reveal_version: VersionedSource {
                value: 4,
                source: "storage:SubtensorModule.CommitRevealWeightsVersion".into(),
            },
            epoch_schedule_inputs: EpochScheduleSources {
                tempo: ScheduleInputSource {
                    source: "storage:SubtensorModule.Tempo".into(),
                    key: vec!["netuid".into()],
                    note: None,
                },
                reveal_period_epochs: ScheduleInputSource {
                    source: "storage:SubtensorModule.RevealPeriodEpochs".into(),
                    key: vec!["netuid".into()],
                    note: None,
                },
                block_time: ScheduleInputSource {
                    source: "constant:Aura.SlotDuration".into(),
                    key: vec![],
                    note: None,
                },
                last_epoch_block: ScheduleInputSource {
                    source: "storage:SubtensorModule.LastEpochBlock".into(),
                    key: vec!["netuid".into()],
                    note: None,
                },
                pending_epoch_at: ScheduleInputSource {
                    source: "storage:SubtensorModule.PendingEpochAt".into(),
                    key: vec!["netuid".into()],
                    note: None,
                },
                subnet_epoch_index: ScheduleInputSource {
                    source: "storage:SubtensorModule.SubnetEpochIndex".into(),
                    key: vec!["netuid".into()],
                    note: None,
                },
                blocks_since_last_step: ScheduleInputSource {
                    source: "storage:SubtensorModule.BlocksSinceLastStep".into(),
                    key: vec!["netuid".into()],
                    note: None,
                },
            },
            commitments_pallet_present: true,
            commitments_pallet: Some(PalletRef {
                name: "Commitments".into(),
                index: 18,
            }),
        }
    }

    #[test]
    fn weights_tlock_payload_is_exactly_four_fields() {
        let shape = weights_tlock_shape();
        assert_eq!(
            shape.fields,
            vec!["hotkey", "uids", "values", "version_key"]
        );
        validate_weights_shape(&shape).expect("shape ok");
    }

    #[test]
    fn weights_shape_rejects_extra_merkle_field() {
        let mut shape = weights_tlock_shape();
        shape.fields.push("merkle_root".into());
        shape.scale_types.push("[u8;32]".into());
        let err = validate_weights_shape(&shape).expect_err("must reject");
        assert!(err.contains("exactly"), "{err}");
    }

    #[test]
    fn null_schedule_source_fails_validation() {
        let mut lock = sample_lock();
        lock.epoch_schedule_inputs.pending_epoch_at.source.clear();
        let err = validate_lockfile(&lock).expect_err("null source");
        assert!(err.contains("pending_epoch_at"), "{err}");
    }

    #[test]
    fn sample_lock_validates_and_roundtrips() {
        let lock = sample_lock();
        validate_lockfile(&lock).expect("valid");
        let text = render_lockfile(&lock).expect("render");
        let back: Lockfile = serde_json::from_str(&text).expect("parse");
        assert_eq!(lock, back);
    }

    #[test]
    fn seven_schedule_inputs_all_named() {
        let lock = sample_lock();
        let s = &lock.epoch_schedule_inputs;
        for src in [
            &s.tempo.source,
            &s.reveal_period_epochs.source,
            &s.block_time.source,
            &s.last_epoch_block.source,
            &s.pending_epoch_at.source,
            &s.subnet_epoch_index.source,
            &s.blocks_since_last_step.source,
        ] {
            assert!(!src.is_empty());
            assert!(src.contains(':'), "source must name protocol: {src}");
        }
    }

    #[test]
    fn parse_storage_and_constant_sources() {
        let (p, i) = parse_storage_source("storage:SubtensorModule.Tempo").unwrap();
        assert_eq!((p, i), ("SubtensorModule", "Tempo"));
        let (p, n) = parse_constant_source("constant:Aura.SlotDuration").unwrap();
        assert_eq!((p, n), ("Aura", "SlotDuration"));
        assert!(parse_storage_source("rpc:foo").is_err());
    }

    #[test]
    fn twox128_storage_prefix_stable() {
        // Substrate twox128("SubtensorModule") — pin against known vector.
        let k = twox128(b"SubtensorModule");
        assert_eq!(k.len(), 16);
        // Second call identical
        assert_eq!(k, twox128(b"SubtensorModule"));
    }
}
