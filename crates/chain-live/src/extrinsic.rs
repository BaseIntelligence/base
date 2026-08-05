//! sr25519 signed Substrate extrinsic builder (V4 format).

use chain::ChainError;
use parity_scale_codec::{Compact, Encode};
use schnorrkel::{signing_context, ExpansionMode, MiniSecretKey};

/// Pallet index for `SubtensorModule` (lockfile `call_indices`).
const PALLET_INDEX: u8 = 7;
/// Call index for `set_weights` (lockfile).
const CALL_SET_WEIGHTS: u8 = 0;
/// Call index for `commit_timelocked_mechanism_weights` (lockfile).
const CALL_COMMIT_MECHANISM_WEIGHTS: u8 = 118;
/// Call index for `serve_axon` (lockfile).
const CALL_SERVE_AXON: u8 = 4;

/// Arguments of `SubtensorModule::serve_axon`, in on-chain order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServeAxonParams {
    /// Subnet to publish on.
    pub netuid: u16,
    /// Bittensor version identifier.
    pub version: u32,
    /// Endpoint address as a numeric integer (IPv4 packed into the low 32 bits).
    pub ip: u128,
    /// Endpoint TCP port.
    pub port: u16,
    /// `4` or `6`; the pallet rejects anything else.
    pub ip_type: u8,
    /// TCP:0 or UDP:1.
    pub protocol: u8,
    /// Reserved by the pallet.
    pub placeholder1: u8,
    /// Reserved by the pallet.
    pub placeholder2: u8,
}

impl ServeAxonParams {
    /// IPv4 endpoint over TCP with all placeholders zeroed.
    #[must_use]
    pub fn ipv4(netuid: u16, version: u32, ip: std::net::Ipv4Addr, port: u16) -> Self {
        Self {
            netuid,
            version,
            ip: u128::from(u32::from(ip)),
            port,
            ip_type: 4,
            protocol: 0,
            placeholder1: 0,
            placeholder2: 0,
        }
    }
}

/// Extrinsic era.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Era {
    /// Immortal era — valid forever; `block_hash` = genesis hash in signing payload.
    Immortal,
    /// Mortal era — valid for `period` blocks starting at `phase`.
    Mortal {
        /// Period in blocks (rounded to next power of two, min 2).
        period: u64,
        /// Phase offset in blocks.
        phase: u64,
    },
}

impl Era {
    /// Encode as Substrate does (single byte).
    #[must_use]
    pub fn encode_era(&self) -> Vec<u8> {
        match self {
            Self::Immortal => vec![0x00],
            Self::Mortal { period, phase } => {
                let period = period.next_power_of_two().max(2);
                let quantize_factor = (period >> 2).max(1);
                let last = period - 1;
                let first = (last / 2 + 1).max(2);
                let factor = first.trailing_zeros();
                let factor_u8 = u8::try_from(factor).unwrap_or(0) & 0x0F;
                let phase_u8 = u8::try_from(phase / quantize_factor).unwrap_or(0) & 0x0F;
                vec![(factor_u8 << 4) | phase_u8]
            }
        }
    }
}

/// Derive the sr25519 public key from a 32-byte mini-secret.
///
/// # Errors
/// Invalid secret key bytes.
pub fn derive_public_key(secret: &[u8; 32]) -> Result<[u8; 32], ChainError> {
    let mini = MiniSecretKey::from_bytes(secret)
        .map_err(|e| ChainError::Other(format!("invalid secret key: {e}")))?;
    Ok(mini
        .expand(ExpansionMode::Ed25519)
        .to_keypair()
        .public
        .to_bytes())
}

/// Blake2b-256 — Substrate hashes signing payloads longer than 256 bytes.
fn blake2_256(data: &[u8]) -> [u8; 32] {
    use blake2::digest::{consts::U32, Digest};
    use blake2::Blake2b;
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(data);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Sign a payload with sr25519 under the `"substrate"` context.
///
/// When `payload.len() > 256`, Substrate signs `blake2_256(payload)` (same as
/// polkadot.js / `sp_runtime::generic::UncheckedExtrinsic`).
fn sign_payload(secret: &[u8; 32], payload: &[u8]) -> Result<[u8; 64], ChainError> {
    let mini = MiniSecretKey::from_bytes(secret)
        .map_err(|e| ChainError::Other(format!("invalid secret key: {e}")))?;
    let keypair = mini.expand(ExpansionMode::Ed25519).to_keypair();
    let ctx = signing_context(b"substrate");
    if payload.len() > 256 {
        let hash = blake2_256(payload);
        Ok(keypair.sign(ctx.bytes(&hash)).to_bytes())
    } else {
        Ok(keypair.sign(ctx.bytes(payload)).to_bytes())
    }
}

/// `CheckMetadataHash` mode: `Disabled` (Finney / subtensor `TxExtension`).
///
/// Encoded in the signed extrinsic after tip; additional-signed appends
/// `Option<MetadataHash>::None` (`0x00`) when disabled.
const METADATA_HASH_MODE_DISABLED: u8 = 0x00;

/// Build the signing payload for a V4 extrinsic.
///
/// Substrate / subtensor order:
/// `Call` ++ signed extras (`era`, `nonce`, `tip`, `CheckMetadataHash` mode) ++
/// additional signed (`spec_version`, `tx_version`, `genesis_hash`, `block_hash`,
/// `Option::None` metadata hash).
#[allow(clippy::too_many_arguments)]
fn signing_payload(
    era: &Era,
    nonce: u64,
    tip: u64,
    call: &[u8],
    spec_version: u32,
    tx_version: u32,
    genesis_hash: &[u8; 32],
    block_hash: &[u8; 32],
) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(call);
    payload.extend_from_slice(&era.encode_era());
    Compact(nonce).encode_to(&mut payload);
    Compact(tip).encode_to(&mut payload);
    payload.push(METADATA_HASH_MODE_DISABLED);
    payload.extend_from_slice(&spec_version.to_le_bytes());
    payload.extend_from_slice(&tx_version.to_le_bytes());
    payload.extend_from_slice(genesis_hash);
    payload.extend_from_slice(block_hash);
    // CheckMetadataHash additional signed: None when mode is Disabled.
    payload.push(0x00);
    payload
}

/// Build a signed V4 extrinsic from call bytes.
#[allow(clippy::too_many_arguments)]
fn build_signed_extrinsic(
    secret: &[u8; 32],
    era: &Era,
    nonce: u64,
    tip: u64,
    call: &[u8],
    spec_version: u32,
    tx_version: u32,
    genesis_hash: &[u8; 32],
    block_hash: &[u8; 32],
) -> Result<Vec<u8>, ChainError> {
    let payload = signing_payload(
        era,
        nonce,
        tip,
        call,
        spec_version,
        tx_version,
        genesis_hash,
        block_hash,
    );
    let sig = sign_payload(secret, &payload)?;
    let pubkey = derive_public_key(secret)?;

    let mut ext = Vec::new();
    ext.push(0x84); // version 4 + signed bit
    ext.push(0x00); // MultiAddress::Id
    ext.extend_from_slice(&pubkey);
    ext.push(0x01); // MultiSignature::Sr25519
    ext.extend_from_slice(&sig);
    ext.extend_from_slice(&era.encode_era());
    Compact(nonce).encode_to(&mut ext);
    Compact(tip).encode_to(&mut ext);
    ext.push(METADATA_HASH_MODE_DISABLED);
    ext.extend_from_slice(call);
    Ok(ext)
}

/// SCALE-encode the `set_weights` call bytes (pallet + call index + params).
#[must_use]
pub fn set_weights_call(netuid: u16, uids: &[u16], values: &[u16], version_key: u64) -> Vec<u8> {
    let mut call = Vec::new();
    call.push(PALLET_INDEX);
    call.push(CALL_SET_WEIGHTS);
    netuid.encode_to(&mut call);
    uids.encode_to(&mut call);
    values.encode_to(&mut call);
    version_key.encode_to(&mut call);
    call
}

/// SCALE-encode the `commit_timelocked_mechanism_weights` call bytes.
///
/// Runtime args (pallet 7 / call 118):
/// `netuid`, `mecid`, `commit` (`BoundedVec<u8>`), `reveal_round`, `commit_reveal_version`.
///
/// `commit` must be the **drand-timelock encrypted** blob (SDK
/// `get_encrypted_commit_v2`). Passing a raw [`WeightsTlockPayload`] SCALE blob
/// is only for unit fixtures — Finney will reject it at validation.
#[must_use]
pub fn commit_timelocked_call(
    netuid: u16,
    mecid: u8,
    commit: &[u8],
    reveal_round: u64,
    commit_reveal_version: u16,
) -> Vec<u8> {
    let mut call = Vec::new();
    call.push(PALLET_INDEX);
    call.push(CALL_COMMIT_MECHANISM_WEIGHTS);
    netuid.encode_to(&mut call);
    mecid.encode_to(&mut call);
    // BoundedVec<u8, N> encodes identically to Vec<u8> (compact len + bytes).
    commit.to_vec().encode_to(&mut call);
    reveal_round.encode_to(&mut call);
    commit_reveal_version.encode_to(&mut call);
    call
}

/// SCALE-encode the `serve_axon` call bytes.
#[must_use]
pub fn serve_axon_call(p: &ServeAxonParams) -> Vec<u8> {
    let mut call = Vec::new();
    call.push(PALLET_INDEX);
    call.push(CALL_SERVE_AXON);
    p.netuid.encode_to(&mut call);
    p.version.encode_to(&mut call);
    p.ip.encode_to(&mut call);
    p.port.encode_to(&mut call);
    p.ip_type.encode_to(&mut call);
    p.protocol.encode_to(&mut call);
    p.placeholder1.encode_to(&mut call);
    p.placeholder2.encode_to(&mut call);
    call
}

/// Build and sign a `serve_axon` extrinsic.
///
/// # Errors
/// Invalid signing key.
#[allow(clippy::too_many_arguments)]
pub fn build_and_sign_serve_axon(
    key: &[u8; 32],
    nonce: u64,
    era: &Era,
    genesis_hash: &[u8; 32],
    block_hash: &[u8; 32],
    spec_version: u32,
    tx_version: u32,
    params: &ServeAxonParams,
) -> Result<Vec<u8>, ChainError> {
    let call = serve_axon_call(params);
    build_signed_extrinsic(
        key,
        era,
        nonce,
        0,
        &call,
        spec_version,
        tx_version,
        genesis_hash,
        block_hash,
    )
}

/// Build and sign a `set_weights` extrinsic.
///
/// # Errors
/// Invalid signing key.
#[allow(clippy::too_many_arguments)]
pub fn build_and_sign_set_weights(
    key: &[u8; 32],
    nonce: u64,
    era: &Era,
    genesis_hash: &[u8; 32],
    block_hash: &[u8; 32],
    spec_version: u32,
    tx_version: u32,
    netuid: u16,
    uids: &[u16],
    values: &[u16],
    version_key: u64,
) -> Result<Vec<u8>, ChainError> {
    let call = set_weights_call(netuid, uids, values, version_key);
    build_signed_extrinsic(
        key,
        era,
        nonce,
        0,
        &call,
        spec_version,
        tx_version,
        genesis_hash,
        block_hash,
    )
}

/// Build and sign a `commit_timelocked_mechanism_weights` extrinsic.
///
/// # Errors
/// Invalid signing key.
#[allow(clippy::too_many_arguments)]
pub fn build_and_sign_commit_timelocked(
    key: &[u8; 32],
    nonce: u64,
    era: &Era,
    genesis_hash: &[u8; 32],
    block_hash: &[u8; 32],
    spec_version: u32,
    tx_version: u32,
    netuid: u16,
    mecid: u8,
    commit: &[u8],
    reveal_round: u64,
    commit_reveal_version: u16,
) -> Result<Vec<u8>, ChainError> {
    let call = commit_timelocked_call(netuid, mecid, commit, reveal_round, commit_reveal_version);
    build_signed_extrinsic(
        key,
        era,
        nonce,
        0,
        &call,
        spec_version,
        tx_version,
        genesis_hash,
        block_hash,
    )
}
