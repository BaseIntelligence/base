//! Compatibility shims so legacy Python validators can talk to the Rust gateway.
//!
//! Python agents call `POST /v1/validators/register`, heartbeat, subscriptions,
//! assignment pull, and `GET /v1/registry` against `master_url`. The Rust control
//! plane evaluates challenges on master and only needs validators to fetch
//! `/v1/weights/latest` — these routes keep the old client loops from 404ing.
//!
//! Auth: substrate-context sr25519 over the Python canonical string
//! (`METHOD\\nPATH\\nTIMESTAMP\\nNONCE\\nSHA256(body)`). Metagraph eligibility is
//! not enforced here (soft compat); signature + SS58 are required.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use keystore::ss58_decode;
use parking_lot::Mutex;
use schnorrkel::{signing_context, PublicKey, Signature};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use trustroot::ChallengesBody;

const HEARTBEAT_INTERVAL_SECS: i64 = 60;
const AUTH_TTL_SECS: i64 = 300;
const HDR_HOTKEY: &str = "x-hotkey";
const HDR_SIGNATURE: &str = "x-signature";
const HDR_NONCE: &str = "x-nonce";
const HDR_TIMESTAMP: &str = "x-timestamp";

/// Shared shim state (in-memory validator directory + trust-root challenges).
#[derive(Clone)]
pub struct CompatState {
    /// Registered validators keyed by SS58 hotkey.
    pub validators: Arc<Mutex<HashMap<String, ValidatorRec>>>,
    /// Owner-signed challenges (used by `GET /v1/registry`).
    pub challenges: Arc<ChallengesBody>,
}

impl CompatState {
    /// Build from the gateway trust-root challenges body.
    #[must_use]
    pub fn new(challenges: Arc<ChallengesBody>) -> Self {
        Self {
            validators: Arc::new(Mutex::new(HashMap::new())),
            challenges,
        }
    }
}

/// One in-memory validator row.
#[derive(Clone, Debug, Serialize)]
pub struct ValidatorRec {
    hotkey: String,
    uid: Option<i64>,
    status: String,
    capabilities: Vec<String>,
    subscriptions: Vec<String>,
    version: Option<String>,
    /// RFC3339 UTC timestamp.
    registered_at: String,
    /// RFC3339 UTC timestamp.
    last_heartbeat_at: Option<String>,
    last_heartbeat_sequence: i64,
    last_seen_meta: Value,
}

/// Mount Python-compat routes.
pub fn compat_router(state: CompatState) -> Router {
    Router::new()
        .route("/v1/validators/register", post(register))
        .route("/v1/validators/heartbeat", post(heartbeat))
        .route("/v1/validators/subscriptions", post(subscriptions))
        .route("/v1/validators/public", get(public_validators))
        .route("/v1/assignments/pull", post(assignments_pull))
        .route("/v1/assignments/{id}/progress", post(assignment_progress))
        .route("/v1/assignments/{id}/result", post(assignment_result))
        .route("/v1/registry", get(registry))
        .route(
            "/v1/weights/submission-observations",
            post(submission_observation),
        )
        .with_state(state)
}

#[derive(Deserialize)]
struct RegisterBody {
    #[serde(default = "default_caps")]
    capabilities: Vec<String>,
    version: Option<String>,
    last_seen_meta: Option<Value>,
}

fn default_caps() -> Vec<String> {
    vec!["cpu".into()]
}

#[derive(Deserialize)]
struct HeartbeatBody {
    #[serde(default)]
    sequence: i64,
    last_seen_meta: Option<Value>,
    capabilities: Option<Vec<String>>,
    version: Option<String>,
}

#[derive(Deserialize)]
struct SubscriptionsBody {
    #[serde(default)]
    slugs: Vec<String>,
}

async fn register(
    State(st): State<CompatState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let hotkey = match verify_signed("POST", "/v1/validators/register", &headers, &body) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let payload: RegisterBody = serde_json::from_slice(&body).unwrap_or(RegisterBody {
        capabilities: default_caps(),
        version: None,
        last_seen_meta: None,
    });
    let now = iso8601_now();
    let mut map = st.validators.lock();
    let (rec, created) = if let Some(existing) = map.get_mut(&hotkey) {
        existing.status = "online".into();
        existing.capabilities.clone_from(&payload.capabilities);
        if payload.version.is_some() {
            existing.version.clone_from(&payload.version);
        }
        if let Some(meta) = payload.last_seen_meta.clone() {
            existing.last_seen_meta = meta;
        }
        existing.last_heartbeat_at = Some(now.clone());
        (existing.clone(), false)
    } else {
        let rec = ValidatorRec {
            hotkey: hotkey.clone(),
            uid: None,
            status: "online".into(),
            capabilities: payload.capabilities,
            subscriptions: vec![],
            version: payload.version,
            registered_at: now.clone(),
            last_heartbeat_at: Some(now),
            last_heartbeat_sequence: 0,
            last_seen_meta: payload.last_seen_meta.unwrap_or_else(|| json!({})),
        };
        map.insert(hotkey, rec.clone());
        (rec, true)
    };
    drop(map);
    Json(json!({
        "validator": view(&rec),
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECS,
        "idempotent": !created,
    }))
    .into_response()
}

async fn heartbeat(
    State(st): State<CompatState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let hotkey = match verify_signed("POST", "/v1/validators/heartbeat", &headers, &body) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let payload: HeartbeatBody = serde_json::from_slice(&body).unwrap_or(HeartbeatBody {
        sequence: 0,
        last_seen_meta: None,
        capabilities: None,
        version: None,
    });
    let now = iso8601_now();
    let mut map = st.validators.lock();
    let Some(rec) = map.get_mut(&hotkey) else {
        return (StatusCode::NOT_FOUND, "validator not registered").into_response();
    };
    let idempotent = payload.sequence > 0 && payload.sequence == rec.last_heartbeat_sequence;
    if payload.sequence >= rec.last_heartbeat_sequence {
        rec.last_heartbeat_sequence = payload.sequence;
    }
    rec.status = "online".into();
    rec.last_heartbeat_at = Some(now.clone());
    if let Some(caps) = payload.capabilities {
        rec.capabilities = caps;
    }
    if payload.version.is_some() {
        rec.version = payload.version;
    }
    if let Some(meta) = payload.last_seen_meta {
        rec.last_seen_meta = meta;
    }
    let seq = rec.last_heartbeat_sequence;
    let status = rec.status.clone();
    drop(map);
    Json(json!({
        "status": status,
        "now": now,
        "sequence": seq,
        "idempotent": idempotent,
    }))
    .into_response()
}

async fn subscriptions(
    State(st): State<CompatState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let hotkey = match verify_signed("POST", "/v1/validators/subscriptions", &headers, &body) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let payload: SubscriptionsBody =
        serde_json::from_slice(&body).unwrap_or(SubscriptionsBody { slugs: vec![] });
    let mut map = st.validators.lock();
    let Some(rec) = map.get_mut(&hotkey) else {
        return (StatusCode::NOT_FOUND, "validator not registered").into_response();
    };
    rec.subscriptions = payload.slugs;
    let out = rec.clone();
    drop(map);
    Json(json!({
        "validator": view(&out),
        "subscriptions": out.subscriptions,
    }))
    .into_response()
}

async fn public_validators(State(st): State<CompatState>) -> Json<Value> {
    let map = st.validators.lock();
    let validators: Vec<Value> = map
        .values()
        .map(|r| {
            json!({
                "hotkey": r.hotkey,
                "uid": r.uid,
                "status": r.status,
                "online": r.status == "online",
                "capabilities": r.capabilities,
                "subscriptions": r.subscriptions,
                "last_heartbeat_at": r.last_heartbeat_at,
                "identity": null,
            })
        })
        .collect();
    Json(json!({ "validators": validators, "subnet": null }))
}

async fn assignments_pull(headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(r) = verify_signed("POST", "/v1/assignments/pull", &headers, &body) {
        return *r;
    }
    // Challenges run on master only; legacy agents get an empty work queue.
    Json(json!({ "api_version": "1.0", "assignments": [] })).into_response()
}

async fn assignment_progress(
    Path(id): Path<String>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let path = format!("/v1/assignments/{id}/progress");
    if let Err(r) = verify_signed("POST", &path, &headers, &body) {
        return *r;
    }
    Json(json!({
        "api_version": "1.0",
        "status": "running",
        "lease_deadline": null,
        "last_progress_at": iso8601_now(),
        "checkpoint_ref": null,
    }))
    .into_response()
}

async fn assignment_result(
    Path(id): Path<String>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let path = format!("/v1/assignments/{id}/result");
    if let Err(r) = verify_signed("POST", &path, &headers, &body) {
        return *r;
    }
    Json(json!({
        "api_version": "1.0",
        "status": "succeeded",
        "result_ref": null,
        "idempotent": true,
    }))
    .into_response()
}

async fn registry(State(st): State<CompatState>) -> Json<Value> {
    let challenges: Vec<Value> = st
        .challenges
        .challenges
        .iter()
        .filter_map(|c| {
            let slug = String::from_utf8(c.id.clone()).ok()?;
            let pct = f64::from(c.emission_share_bps) / 100.0;
            Some(json!({
                "slug": slug,
                "name": slug,
                "image": format!("ghcr.io/baseintelligence/base/{slug}"),
                "version": "1.0",
                "emission_percent": pct,
                "status": "active",
                "description": null,
                "metadata": {},
                "internal_base_url": format!("http://{slug}:8080"),
                "public_proxy_base_path": format!("/challenge/{slug}"),
                "required_capabilities": ["cpu"],
                "resources": {},
                "volumes": {},
                "env": {},
                "secrets": [],
            }))
        })
        .collect();
    Json(json!({
        "network": "base",
        "api_version": "1.0",
        "master_uid": 0,
        "challenges": challenges,
    }))
}

async fn submission_observation(headers: HeaderMap, body: axum::body::Bytes) -> Response {
    let hotkey = match verify_signed(
        "POST",
        "/v1/weights/submission-observations",
        &headers,
        &body,
    ) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    // Soft-accept: Python clients only require HTTP 2xx + JSON. Mirror the
    // fields `ValidatorSubmissionObservationResponse` documents.
    let payload: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
    let vector_id = payload
        .get("vector_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_owned();
    let vector_digest = payload
        .get("vector_digest")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();
    let outcome = payload
        .get("outcome")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_owned();
    let attempt = payload.get("attempt").and_then(Value::as_i64).unwrap_or(1);
    (
        StatusCode::CREATED,
        Json(json!({
            "observation_id": format!("compat-{hotkey}-{vector_id}"),
            "validator_hotkey": hotkey,
            "vector_id": vector_id,
            "vector_digest": vector_digest,
            "outcome": outcome,
            "attempt": attempt,
            "created_at": iso8601_now(),
            "idempotent": true,
            "accepted": true,
        })),
    )
        .into_response()
}

fn view(r: &ValidatorRec) -> Value {
    json!({
        "hotkey": r.hotkey,
        "uid": r.uid,
        "status": r.status,
        "capabilities": r.capabilities,
        "subscriptions": r.subscriptions,
        "version": r.version,
        "registered_at": r.registered_at,
        "last_heartbeat_at": r.last_heartbeat_at,
        "last_heartbeat_sequence": r.last_heartbeat_sequence,
        "last_seen_meta": r.last_seen_meta,
    })
}

fn verify_signed(
    method: &str,
    path: &str,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<String, Box<Response>> {
    let hotkey = header(headers, HDR_HOTKEY).ok_or_else(|| Box::new(unauthorized()))?;
    let sig_hex = header(headers, HDR_SIGNATURE).ok_or_else(|| Box::new(unauthorized()))?;
    let nonce = header(headers, HDR_NONCE).ok_or_else(|| Box::new(unauthorized()))?;
    let ts_raw = header(headers, HDR_TIMESTAMP).ok_or_else(|| Box::new(unauthorized()))?;
    let ts: i64 = ts_raw.parse().map_err(|_| Box::new(unauthorized()))?;
    let now = now_secs();
    if (now - ts).abs() > AUTH_TTL_SECS {
        return Err(Box::new(unauthorized()));
    }
    let (pk, _prefix) = ss58_decode(&hotkey).map_err(|_| Box::new(unauthorized()))?;
    let body_hash = hex::encode(Sha256::digest(body));
    let canonical = format!(
        "{}\n{}\n{}\n{}\n{}",
        method.to_uppercase(),
        path,
        ts_raw,
        nonce,
        body_hash
    );
    if !verify_sr25519(&pk, canonical.as_bytes(), &sig_hex) {
        return Err(Box::new(unauthorized()));
    }
    Ok(hotkey)
}

fn verify_sr25519(public: &[u8; 32], message: &[u8], signature: &str) -> bool {
    let raw = signature.strip_prefix("0x").unwrap_or(signature);
    let Ok(bytes) = hex::decode(raw) else {
        return false;
    };
    if bytes.len() != 64 {
        return false;
    }
    let Ok(pk) = PublicKey::from_bytes(public) else {
        return false;
    };
    let Ok(sig) = Signature::from_bytes(&bytes) else {
        return false;
    };
    let ctx = signing_context(b"substrate");
    pk.verify(ctx.bytes(message), &sig).is_ok()
}

fn header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
}

fn unauthorized() -> Response {
    (StatusCode::UNAUTHORIZED, "invalid signed request").into_response()
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

/// RFC3339 UTC with microsecond precision (no chrono; denied by cargo-deny).
fn iso8601_now() -> String {
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_micros()).unwrap_or(u64::MAX));
    let secs = micros / 1_000_000;
    let frac = micros % 1_000_000;
    let days = secs / 86_400;
    let tod = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    format!("{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}.{frac:06}Z")
}

fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use keystore::BITTENSOR_SS58_PREFIX;
    use rand_core::OsRng;
    use schnorrkel::{ExpansionMode, MiniSecretKey};
    use tower::ServiceExt;

    fn sign(sk: &schnorrkel::SecretKey, message: &[u8]) -> String {
        let pk = sk.to_public();
        let sig = sk.sign(signing_context(b"substrate").bytes(message), &pk);
        format!("0x{}", hex::encode(sig.to_bytes()))
    }

    #[test]
    fn verifies_bittensor_substrate_signature() {
        // Fixed vector from `bittensor.Keypair` (BIP39 abandon…about).
        let pub_hex = "66933bd1f37070ef87bd1198af3dacceb095237f803f3d32b173e6b425ed7972";
        let sig_hex = "e8352b5811042659ac694b8f48518051f0dff1f727bd893481c3b87bbc9d0d741239e7149d72f23d47da6d198024e5fbc8c535bef32716d6624da851a26ce78e";
        let body = br#"{"capabilities":["cpu"]}"#;
        let body_hash = hex::encode(Sha256::digest(body));
        let canonical = format!("POST\n/v1/validators/register\n1700000000\nn1\n{body_hash}");
        let mut pk = [0u8; 32];
        pk.copy_from_slice(&hex::decode(pub_hex).unwrap());
        assert!(verify_sr25519(&pk, canonical.as_bytes(), sig_hex));
    }

    #[tokio::test]
    async fn register_heartbeat_pull_registry() {
        let mini = MiniSecretKey::generate_with(OsRng);
        let sk = mini.expand(ExpansionMode::Ed25519);
        let pk = sk.to_public().to_bytes();
        let hotkey = keystore::ss58_encode(&pk, BITTENSOR_SS58_PREFIX);

        let challenges = Arc::new(ChallengesBody {
            challenges: vec![trustroot::ChallengeEntry {
                id: b"design".to_vec(),
                public_key: [1u8; 32],
                emission_share_bps: 5000,
                policy: trustroot::ParticipantPolicy::AllMetagraphHotkeys,
            }],
        });
        let app = compat_router(CompatState::new(challenges));

        let body = br#"{"capabilities":["cpu"],"version":"test"}"#;
        let ts = now_secs().to_string();
        let nonce = "abc123";
        let body_hash = hex::encode(Sha256::digest(body));
        let canonical = format!("POST\n/v1/validators/register\n{ts}\n{nonce}\n{body_hash}");
        let sig = sign(&sk, canonical.as_bytes());

        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/validators/register")
            .header(HDR_HOTKEY, &hotkey)
            .header(HDR_SIGNATURE, &sig)
            .header(HDR_NONCE, nonce)
            .header(HDR_TIMESTAMP, &ts)
            .header("content-type", "application/json")
            .body(Body::from(body.as_slice()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["heartbeat_interval_seconds"], 60);
        assert_eq!(v["validator"]["hotkey"], hotkey);

        // Empty assignment pull.
        let body2 = b"{}";
        let ts2 = now_secs().to_string();
        let nonce2 = "def456";
        let hash2 = hex::encode(Sha256::digest(body2));
        let can2 = format!("POST\n/v1/assignments/pull\n{ts2}\n{nonce2}\n{hash2}");
        let sig2 = sign(&sk, can2.as_bytes());
        let req2 = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/assignments/pull")
            .header(HDR_HOTKEY, &hotkey)
            .header(HDR_SIGNATURE, &sig2)
            .header(HDR_NONCE, nonce2)
            .header(HDR_TIMESTAMP, &ts2)
            .header("content-type", "application/json")
            .body(Body::from(body2.as_slice()))
            .unwrap();
        let resp2 = app.clone().oneshot(req2).await.unwrap();
        assert_eq!(resp2.status(), StatusCode::OK);

        let req3 = axum::http::Request::builder()
            .uri("/v1/registry")
            .body(Body::empty())
            .unwrap();
        let resp3 = app.oneshot(req3).await.unwrap();
        assert_eq!(resp3.status(), StatusCode::OK);
        let bytes3 = resp3.into_body().collect().await.unwrap().to_bytes();
        let reg: Value = serde_json::from_slice(&bytes3).unwrap();
        assert_eq!(reg["challenges"][0]["slug"], "design");
    }
}
