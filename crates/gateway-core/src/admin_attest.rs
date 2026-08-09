//! Master-only admin attestation grant (`POST /v1/admin/attest-grant`).
//!
//! The normal path to a `verified` row is the D10 quote pipeline: nonce → TDX
//! quote → DCAP verify → compose binding → insert. When the runtime has no TEE
//! (local testnet stack, `AGENT_CHALLENGE.md` §9.6) there is no quote to
//! verify, so the subnet owner records credit directly: this route appends a
//! `verified` attestation row with `quote = NULL` and an explicit
//! `reason: admin-exempt: …`, under the gateway's own control-plane hotkey.
//!
//! Access control matches `/v1/admin/seal`: Bearer token via
//! `BASE_GATEWAY_ADMIN_TOKEN` / `_FILE` (required when
//! `BASE_GATEWAY_REQUIRE_OWNER=1`). Do not treat network placement alone as
//! auth — production historically published these routes on `:80`/`:443`.
//! The written row is auditable in `attestation.reason`.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use crypto::KEY_LEN;
use serde::{Deserialize, Serialize};

/// Route path (mounted on the master-plane gateway only).
pub const ATTEST_GRANT_ROUTE: &str = "/v1/admin/attest-grant";

/// Outcome the grant writes; identical to a D10-`verified` row.
const OUTCOME_VERIFIED: &str = "verified";

/// Longest accepted `reason` (audit text), bytes.
const MAX_REASON_LEN: usize = 512;

/// Router state: the shared control-plane pool plus the identity the row is
/// attributed to (the gateway's own hotkey — this process IS the control
/// plane that verified, administratively, the runtime).
#[derive(Debug, Clone)]
pub struct AttestGrantState {
    /// Shared control-plane Postgres pool.
    pub pool: db::PgPool,
    /// Hotkey stored as `validator_hotkey` on written rows.
    pub validator_hotkey: [u8; KEY_LEN],
}

impl AttestGrantState {
    /// Build state from the shared pool and the gateway hotkey.
    #[must_use]
    pub const fn new(pool: db::PgPool, validator_hotkey: [u8; KEY_LEN]) -> Self {
        Self {
            pool,
            validator_hotkey,
        }
    }
}

/// JSON body for `POST /v1/admin/attest-grant`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttestGrantRequest {
    /// Chain epoch the credit binds to (attestation rows are epoch-keyed).
    pub epoch: u64,
    /// Miner hotkey, 64 lowercase hex (sr25519 public key).
    pub miner_hotkey_hex: String,
    /// Receipt public key the runner signs work receipts with, 64 hex.
    pub receipt_pk_hex: String,
    /// Mandatory audit note, stored verbatim in `attestation.reason`.
    pub reason: String,
}

/// Successful grant response.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AttestGrantResponse {
    /// Epoch the credit binds to.
    pub epoch: u64,
    /// Miner hotkey (lowercase hex) as stored.
    pub miner_hotkey_hex: String,
    /// Receipt key (lowercase hex) as stored.
    pub receipt_pk_hex: String,
    /// Attempt slot the row landed in.
    pub attempt: i32,
    /// Always `verified`.
    pub outcome: String,
}

/// Mount master-only `POST /v1/admin/attest-grant`.
pub fn admin_attest_grant_router(state: AttestGrantState) -> Router {
    Router::new()
        .route(ATTEST_GRANT_ROUTE, post(post_attest_grant))
        .with_state(state)
}

fn bad_request(msg: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": msg })),
    )
        .into_response()
}

/// Parse a 64-char hex public key field.
fn parse_hex32(field: &str, raw: &str) -> Result<[u8; KEY_LEN], String> {
    let raw = raw
        .strip_prefix("0x")
        .or_else(|| raw.strip_prefix("0X"))
        .unwrap_or(raw);
    if raw.len() != 64 {
        return Err(format!("{field} must be 64 hex chars, got {}", raw.len()));
    }
    let bytes = hex::decode(raw).map_err(|e| format!("{field} hex: {e}"))?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| format!("{field} must be 32 bytes, got {}", v.len()))
}

async fn post_attest_grant(
    State(st): State<AttestGrantState>,
    Json(req): Json<AttestGrantRequest>,
) -> Response {
    if req.miner_hotkey_hex.trim() != req.miner_hotkey_hex
        || req.receipt_pk_hex.trim() != req.receipt_pk_hex
        || req.reason.trim() != req.reason
    {
        return bad_request("fields must not carry surrounding whitespace");
    }
    let miner_hotkey = match parse_hex32("miner_hotkey_hex", &req.miner_hotkey_hex) {
        Ok(k) => k,
        Err(e) => return bad_request(&e),
    };
    let receipt_pk = match parse_hex32("receipt_pk_hex", &req.receipt_pk_hex) {
        Ok(k) => k,
        Err(e) => return bad_request(&e),
    };
    if req.reason.is_empty() {
        return bad_request("reason must be a non-empty audit note");
    }
    if req.reason.len() > MAX_REASON_LEN {
        return bad_request(&format!("reason exceeds {MAX_REASON_LEN} bytes"));
    }
    let reason = format!("admin-exempt: {}", req.reason);
    let Ok(epoch) = i64::try_from(req.epoch) else {
        return bad_request("epoch does not fit i64");
    };
    // Random per grant: this column carries the D10 nonce on quote-driven rows;
    // for an admin grant it is evidence entropy, not a replay pre-image.
    let nonce = crypto::generate_mini_secret();
    let miner_hotkey_hex = hex::encode(miner_hotkey);
    let validator_hotkey_hex = hex::encode(st.validator_hotkey);
    let row = db::NewAttestation {
        epoch,
        miner_hotkey: &miner_hotkey_hex,
        validator_hotkey: &validator_hotkey_hex,
        nonce: &nonce,
        outcome: OUTCOME_VERIFIED,
        quote: None,
        reason: Some(&reason),
        receipt_pk: Some(&receipt_pk),
    };
    match db::insert_attestation(&st.pool, &row).await {
        Ok(attempt) => (
            StatusCode::OK,
            Json(AttestGrantResponse {
                epoch: req.epoch,
                miner_hotkey_hex,
                receipt_pk_hex: hex::encode(receipt_pk),
                attempt,
                outcome: OUTCOME_VERIFIED.to_owned(),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}
