//! Gateway-side HTTP surface for [`crate::MinerEndpointBodyV1`].
//!
//! Behind the `server` feature so the miner CLI can depend on the body and its
//! signature without pulling in axum, sqlx, and a chain client.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use chain::{current_epoch_pre_run_coinbase, gather_schedule_state, ChainClient};
use crypto::{KEY_LEN, SIGNATURE_LEN};
use db::{upsert_miner_endpoint, NewMinerEndpoint, PgPool};
use serde::{Deserialize, Serialize};

use crate::{validate_base_url, verify_endpoint, MinerEndpointBodyV1, ENDPOINT_ROUTE};

/// Chain handle used to pin the epoch and check metagraph membership.
pub type SharedChain = Arc<dyn ChainClient + Send + Sync>;

/// Handler state: a chain handle and the endpoint table's pool.
#[derive(Clone)]
pub struct MinerEndpointState {
    chain: SharedChain,
    pool: PgPool,
    netuid: u16,
}

impl MinerEndpointState {
    /// Build state for one subnet.
    #[must_use]
    pub fn new(chain: SharedChain, pool: PgPool, netuid: u16) -> Self {
        Self {
            chain,
            pool,
            netuid,
        }
    }
}

/// Router carrying [`ENDPOINT_ROUTE`], ready to `merge` into the gateway app.
pub fn miner_endpoint_router(state: MinerEndpointState) -> Router {
    Router::new()
        .route(ENDPOINT_ROUTE, post(announce))
        .with_state(state)
}

/// `POST /v1/miners/endpoint` request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnounceRequest {
    /// Subnet netuid; must match the gateway's.
    pub netuid: u16,
    /// Announcing miner hotkey, 64 lowercase hex chars.
    pub miner_hotkey_hex: String,
    /// Public base URL, origin only.
    pub base_url: String,
    /// Chain epoch; must be the current one.
    pub epoch: u64,
    /// sr25519 signature over the SCALE body, 128 hex chars.
    pub signature_hex: String,
}

/// `POST /v1/miners/endpoint` success body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnounceResponse {
    /// Subnet netuid the row was stored under.
    pub netuid: u16,
    /// Miner hotkey hex, as stored (lowercase, unprefixed).
    pub miner_hotkey_hex: String,
    /// Base URL, as stored (verbatim from the signed body).
    pub base_url: String,
    /// Epoch the row was stored under.
    pub epoch: u64,
}

async fn announce(
    State(st): State<MinerEndpointState>,
    Json(req): Json<AnnounceRequest>,
) -> Result<Json<AnnounceResponse>, ApiError> {
    // 1. Shape. Nothing below should have to reason about malformed input.
    if req.netuid != st.netuid {
        return Err(ApiError::bad(format!(
            "netuid {} is not this gateway's netuid {}",
            req.netuid, st.netuid
        )));
    }
    let miner = parse_key_hex(&req.miner_hotkey_hex)?;
    let signature = parse_sig_hex(&req.signature_hex)?;
    validate_base_url(&req.base_url).map_err(|e| ApiError::bad(format!("base_url: {e}")))?;

    // 2. Epoch, from the same counter as the challenge daemon's pin, so a row
    //    written here is findable at the epoch it will be dispatched in.
    let current = current_chain_epoch(st.chain.as_ref(), st.netuid)?;
    if req.epoch != current {
        return Err(ApiError::bad(format!(
            "epoch {} is not the current chain epoch {current}",
            req.epoch
        )));
    }

    // 3. Registration. Without this an unregistered key could fill the table
    //    with URLs the dispatcher would then fan out to.
    require_registered(st.chain.as_ref(), &miner)?;

    // 4. Authenticity, over the exact fields that are about to be stored.
    let body = MinerEndpointBodyV1 {
        netuid: req.netuid,
        miner_hotkey: miner,
        base_url: req.base_url.clone().into_bytes(),
        epoch: req.epoch,
    };
    verify_endpoint(&body, &signature).map_err(|_| ApiError::unauthorized())?;

    // 5. Persist. Re-announcing in the same epoch replaces the row.
    let miner_hex = hex::encode(miner);
    let epoch = i64::try_from(req.epoch)
        .map_err(|_| ApiError::bad(format!("epoch {} does not fit in i64", req.epoch)))?;
    let netuid = i32::from(req.netuid);
    upsert_miner_endpoint(
        &st.pool,
        &NewMinerEndpoint {
            epoch,
            netuid,
            miner_hotkey: &miner_hex,
            base_url: &req.base_url,
            signature: &signature,
        },
    )
    .await
    .map_err(|e| ApiError::unavailable(format!("store endpoint: {e}")))?;

    tracing::info!(
        target: "miner_endpoint",
        event = "miner_endpoint_announced",
        netuid = req.netuid,
        epoch = req.epoch,
        miner = %miner_hex,
        base_url = %req.base_url,
        "stored miner base URL"
    );

    Ok(Json(AnnounceResponse {
        netuid: req.netuid,
        miner_hotkey_hex: miner_hex,
        base_url: req.base_url,
        epoch: req.epoch,
    }))
}

/// Current epoch from `gather_schedule_state` + `current_epoch_pre_run_coinbase`
/// — the pair `chainsnap::read_epoch_pin` and `attest-http` also use, so all
/// three number epochs identically.
fn current_chain_epoch(chain: &dyn ChainClient, netuid: u16) -> Result<u64, ApiError> {
    let state = gather_schedule_state(chain, netuid)
        .map_err(|e| ApiError::unavailable(format!("epoch schedule read: {e}")))?;
    Ok(current_epoch_pre_run_coinbase(&state, state.current_block))
}

/// 403 unless `miner` is a neuron at the chain tip.
fn require_registered(chain: &dyn ChainClient, miner: &[u8; KEY_LEN]) -> Result<(), ApiError> {
    let tip = chain
        .current_block()
        .map_err(|e| ApiError::unavailable(format!("current_block: {e}")))?;
    let block_hash = chain
        .block_hash(tip)
        .map_err(|e| ApiError::unavailable(format!("block_hash({tip}): {e}")))?;
    let metagraph = chain
        .metagraph_at(&block_hash)
        .map_err(|e| ApiError::unavailable(format!("metagraph_at: {e}")))?;
    if metagraph.hotkeys.iter().any(|h| h.as_slice() == miner) {
        return Ok(());
    }
    Err(ApiError {
        status: StatusCode::FORBIDDEN,
        msg: format!(
            "hotkey {} is not registered on netuid at block {tip}",
            hex::encode(miner)
        ),
    })
}

fn parse_key_hex(s: &str) -> Result<[u8; KEY_LEN], ApiError> {
    let bytes =
        hex::decode(s.trim()).map_err(|e| ApiError::bad(format!("miner_hotkey_hex: {e}")))?;
    <[u8; KEY_LEN]>::try_from(bytes.as_slice()).map_err(|_| {
        ApiError::bad(format!(
            "miner_hotkey_hex: expected {KEY_LEN} bytes, got {}",
            bytes.len()
        ))
    })
}

fn parse_sig_hex(s: &str) -> Result<[u8; SIGNATURE_LEN], ApiError> {
    let bytes = hex::decode(s.trim()).map_err(|e| ApiError::bad(format!("signature_hex: {e}")))?;
    <[u8; SIGNATURE_LEN]>::try_from(bytes.as_slice()).map_err(|_| {
        ApiError::bad(format!(
            "signature_hex: expected {SIGNATURE_LEN} bytes, got {}",
            bytes.len()
        ))
    })
}

/// Status + plain-text body, matching the shape `attest-http` returns.
struct ApiError {
    status: StatusCode,
    msg: String,
}

impl ApiError {
    fn bad(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            msg: msg.into(),
        }
    }

    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            msg: "signature does not verify under the announced hotkey".to_owned(),
        }
    }

    fn unavailable(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            msg: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, self.msg).into_response()
    }
}
