//! Epoch sealer + sealed-bundle HTTP (task 27). Core seal: `bundle::build_sealed_bundle`.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bundle::{
    build_sealed_bundle, EpochBundleV1, LeafV1, LocalTrustRoot, RawWeightBodyV1,
    SealParams as BundleSealParams,
};
use chain::ChainClient;
use crypto::{KEY_LEN, SIGNATURE_LEN};
use parity_scale_codec::Decode;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use trustroot::ChallengesBody;
use weights_api::{build_burn_fallback, build_latest, SealRecord};

use crate::api::GatewayState;
use crate::weights::{RawWeightRow, RawWeightStore};

/// Shared sealed-bundle store handle.
pub type SharedBundleStore = Arc<dyn BundleStore>;

/// Persisted sealed epoch bundles.
pub trait BundleStore: Send + Sync {
    /// Insert or return existing sealed bytes for `epoch` (idempotent).
    fn put_if_absent(&self, epoch: u64, bytes: Vec<u8>) -> Vec<u8>;
    /// Lookup by epoch.
    fn get_by_epoch(&self, epoch: u64) -> Option<Vec<u8>>;
    /// Lookup by merkle root.
    fn get_by_root(&self, root: &[u8; 32]) -> Option<Vec<u8>>;
    /// Highest sealed epoch.
    fn latest_epoch(&self) -> Option<u64>;
    /// Seal-time provenance for `epoch` (`computed_at`, revision).
    fn seal_record(&self, epoch: u64) -> Option<SealRecord>;
}

/// In-memory sealed-bundle store.
#[derive(Debug, Default)]
pub struct MemoryBundleStore {
    by_epoch: RwLock<BTreeMap<u64, Vec<u8>>>,
    by_root: RwLock<BTreeMap<[u8; 32], Vec<u8>>>,
    seals: RwLock<BTreeMap<u64, SealRecord>>,
}

impl MemoryBundleStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl BundleStore for MemoryBundleStore {
    fn put_if_absent(&self, epoch: u64, bytes: Vec<u8>) -> Vec<u8> {
        if let Some(existing) = self.by_epoch.read().get(&epoch) {
            return existing.clone();
        }
        let root = match EpochBundleV1::decode_bytes(&bytes) {
            Ok(b) => b.body.merkle_root,
            Err(_) => [0u8; 32],
        };
        let mut by_epoch = self.by_epoch.write();
        if let Some(existing) = by_epoch.get(&epoch) {
            return existing.clone();
        }
        by_epoch.insert(epoch, bytes.clone());
        drop(by_epoch);
        self.by_root.write().insert(root, bytes.clone());
        let mut seals = self.seals.write();
        // Revision counts accepted seals for the epoch; the first seal is 1 and
        // only a replacing seal of the same epoch can raise it.
        let revision = seals.get(&epoch).map_or(1, |prev| prev.revision + 1);
        seals.insert(epoch, SealRecord::now(revision));
        bytes
    }

    fn get_by_epoch(&self, epoch: u64) -> Option<Vec<u8>> {
        self.by_epoch.read().get(&epoch).cloned()
    }

    fn get_by_root(&self, root: &[u8; 32]) -> Option<Vec<u8>> {
        self.by_root.read().get(root).cloned()
    }

    fn latest_epoch(&self) -> Option<u64> {
        self.by_epoch.read().keys().next_back().copied()
    }

    fn seal_record(&self, epoch: u64) -> Option<SealRecord> {
        self.seals.read().get(&epoch).copied()
    }
}

/// Gateway seal parameters.
#[derive(Debug, Clone)]
pub struct SealParams {
    /// Epoch index.
    pub epoch: u64,
    /// Subnet netuid.
    pub netuid: u16,
    /// Inclusive epoch end block.
    pub block_b: u64,
    /// Gateway mini-secret.
    pub gateway_secret: [u8; KEY_LEN],
    /// Measurements digest.
    pub measurements_digest: [u8; 32],
}

/// Seal failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SealError {
    /// Bundle seal failure.
    #[error("{0}")]
    Bundle(String),
    /// Incomplete participant set (D24).
    #[error("incomplete participant set (D24)")]
    IncompleteParticipantSet,
    /// Bad stored leaf.
    #[error("bad raw-weight leaf: {0}")]
    BadLeaf(String),
    /// Codec.
    #[error("codec error: {0}")]
    Codec(String),
}

fn map_bundle_err(e: bundle::BundleError) -> SealError {
    match e {
        bundle::BundleError::IncompleteParticipantSet => SealError::IncompleteParticipantSet,
        other => SealError::Bundle(other.to_string()),
    }
}

/// Seal epoch: gather leaves, D24+aggregate+sign, persist (idempotent).
///
/// # Errors
///
/// [`SealError`] on incomplete set, bad leaves, or seal failure.
pub fn seal_epoch(
    chain: &dyn ChainClient,
    challenges: &ChallengesBody,
    weights: &dyn RawWeightStore,
    bundles: &dyn BundleStore,
    params: &SealParams,
) -> Result<EpochBundleV1, SealError> {
    if let Some(existing) = bundles.get_by_epoch(params.epoch) {
        return EpochBundleV1::decode_bytes(&existing).map_err(|e| SealError::Codec(e.to_string()));
    }
    let leaves = rows_to_leaves(&weights.list_for_epoch(params.epoch))?;
    let trust = LocalTrustRoot {
        challenges: challenges.clone(),
        measurements_digest: params.measurements_digest,
    };
    let bparams = BundleSealParams {
        epoch: params.epoch,
        netuid: params.netuid,
        block_b: params.block_b,
        gateway_secret: params.gateway_secret,
    };
    let bundle = build_sealed_bundle(chain, &trust, leaves, &bparams).map_err(map_bundle_err)?;
    let stored = bundles.put_if_absent(params.epoch, bundle.encode_bytes());
    EpochBundleV1::decode_bytes(&stored).map_err(|e| SealError::Codec(e.to_string()))
}

fn rows_to_leaves(rows: &[RawWeightRow]) -> Result<Vec<LeafV1>, SealError> {
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let body = RawWeightBodyV1::decode(&mut row.payload.as_slice())
            .map_err(|e| SealError::BadLeaf(format!("payload decode: {e}")))?;
        if body.epoch != row.epoch {
            return Err(SealError::BadLeaf("payload epoch mismatch".into()));
        }
        let challenge_sig: [u8; SIGNATURE_LEN] = row
            .challenge_sig
            .as_slice()
            .try_into()
            .map_err(|_| SealError::BadLeaf("challenge_sig length".into()))?;
        out.push(LeafV1 {
            challenge_id: body.challenge_id,
            miner_hotkey: body.miner_hotkey,
            epoch: body.epoch,
            score_or_absence: body.score_or_absence,
            challenge_sig,
        });
    }
    Ok(out)
}

/// Bundle serve routes.
pub fn bundle_router(state: GatewayState) -> Router {
    Router::new()
        .route("/v1/bundle/{epoch}", get(get_bundle_by_epoch))
        .route("/v1/bundle/root/{root}", get(get_bundle_by_root))
        .route("/v1/weights/latest", get(get_weights_latest))
        .with_state(state)
}

fn octet_or_404(bytes: Option<Vec<u8>>) -> Response {
    match bytes {
        Some(b) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/octet-stream")],
            b,
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "bundle not found" })),
        )
            .into_response(),
    }
}

async fn get_bundle_by_epoch(State(st): State<GatewayState>, Path(epoch): Path<u64>) -> Response {
    octet_or_404(st.bundles.get_by_epoch(epoch))
}

async fn get_bundle_by_root(
    State(st): State<GatewayState>,
    Path(root_hex): Path<String>,
) -> Response {
    match parse_root_hex(&root_hex) {
        Ok(root) => octet_or_404(st.bundles.get_by_root(&root)),
        Err(msg) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": msg })),
        )
            .into_response(),
    }
}

async fn get_weights_latest(State(st): State<GatewayState>) -> Response {
    let sealed = st.bundles.latest_epoch().and_then(|epoch| {
        Some((
            st.bundles.get_by_epoch(epoch)?,
            st.bundles.seal_record(epoch)?,
        ))
    });
    let Some((bytes, seal)) = sealed else {
        // Fail-closed: never 404 — serve uid-0 burn until a real seal exists.
        return (StatusCode::OK, Json(build_burn_fallback(st.seal_netuid))).into_response();
    };
    match EpochBundleV1::decode_bytes(&bytes) {
        Ok(bundle) => (StatusCode::OK, Json(build_latest(&bundle, seal))).into_response(),
        Err(_) => {
            // Corrupt sealed bytes: still burn rather than 5xx/404.
            (StatusCode::OK, Json(build_burn_fallback(st.seal_netuid))).into_response()
        }
    }
}

/// JSON body for `POST /v1/admin/seal`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealRequest {
    /// Epoch to seal.
    pub epoch: u64,
    /// Optional netuid (defaults to gateway seal context).
    pub netuid: Option<u16>,
    /// Optional inclusive epoch end block (defaults to the chain tip).
    pub block_b: Option<u64>,
}

/// Successful seal response.
#[derive(Debug, Clone, Serialize)]
pub struct SealResponse {
    /// Sealed epoch.
    pub epoch: u64,
    /// Merkle root hex.
    pub merkle_root: String,
    /// Final weight vector length.
    pub final_vector_len: usize,
}

/// Env: 64-hex mini-secret for bundle signing.
pub const GATEWAY_SK_ENV: &str = "BASE_GATEWAY_SK";
/// Env: path to 32-byte raw or 64-hex gateway mini-secret file.
pub const GATEWAY_SK_FILE_ENV: &str = "BASE_GATEWAY_SK_FILE";

/// Load gateway mini-secret from `BASE_GATEWAY_SK` or `BASE_GATEWAY_SK_FILE`.
///
/// # Errors
///
/// Missing env, bad length, or IO/decode failure.
pub fn load_gateway_secret() -> Result<[u8; KEY_LEN], String> {
    if let Ok(hex_str) = std::env::var(GATEWAY_SK_ENV) {
        return parse_sk_hex(&hex_str);
    }
    if let Ok(path) = std::env::var(GATEWAY_SK_FILE_ENV) {
        let raw = std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?;
        if raw.len() == KEY_LEN {
            let mut out = [0u8; KEY_LEN];
            out.copy_from_slice(&raw);
            return Ok(out);
        }
        let s = String::from_utf8_lossy(&raw);
        return parse_sk_hex(s.trim());
    }
    Err(format!(
        "set {GATEWAY_SK_ENV} (64 hex) or {GATEWAY_SK_FILE_ENV} (32 raw bytes or 64 hex)"
    ))
}

fn parse_sk_hex(s: &str) -> Result<[u8; KEY_LEN], String> {
    let s = s.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    let bytes = hex::decode(s).map_err(|e| format!("gateway sk hex: {e}"))?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| format!("gateway sk must be 32 bytes, got {}", v.len()))
}

/// Mount master-only `POST /v1/admin/seal`.
pub fn admin_seal_router(state: GatewayState) -> Router {
    Router::new()
        .route("/v1/admin/seal", post(post_admin_seal))
        .with_state(state)
}

async fn post_admin_seal(State(st): State<GatewayState>, Json(req): Json<SealRequest>) -> Response {
    let gateway_secret = match load_gateway_secret() {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": e })),
            )
                .into_response();
        }
    };
    let netuid = req.netuid.unwrap_or(st.seal_netuid);
    // No `block_b` means "seal at the chain tip"; a stale constant would pin the
    // metagraph to a block the caller never chose.
    let block_b = match req.block_b {
        Some(b) => b,
        None => match st.chain.current_block() {
            Ok(b) => b,
            Err(e) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({ "error": format!("chain tip unavailable: {e}") })),
                )
                    .into_response();
            }
        },
    };
    let params = SealParams {
        epoch: req.epoch,
        netuid,
        block_b,
        gateway_secret,
        measurements_digest: st.measurements_digest,
    };
    match seal_epoch(
        st.chain.as_ref(),
        st.challenges.as_ref(),
        st.weights.as_ref(),
        st.bundles.as_ref(),
        &params,
    ) {
        Ok(bundle) => (
            StatusCode::OK,
            Json(SealResponse {
                epoch: bundle.body.epoch,
                merkle_root: hex::encode(bundle.body.merkle_root),
                final_vector_len: bundle.body.final_vector.len(),
            }),
        )
            .into_response(),
        Err(SealError::IncompleteParticipantSet) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "incomplete participant set (D24)",
                "code": "incomplete_participant_set",
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

fn parse_root_hex(s: &str) -> Result<[u8; 32], String> {
    let s = s.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    if s.len() != 64 {
        return Err(format!("merkle root must be 64 hex chars, got {}", s.len()));
    }
    let bytes = hex::decode(s).map_err(|e| e.to_string())?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| format!("expected 32 bytes, got {}", v.len()))
}
