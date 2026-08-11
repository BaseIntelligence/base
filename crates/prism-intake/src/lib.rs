//! `prism-intake` — the HTTP front-end shared by the two PRISM intake
//! entry points, split out of `prism-challenge` for the per-crate non-test
//! LOC cap (the repo's standard crate-split pattern, same as
//! `prism-pipeline` and `prism-attribution`).
//!
//! Both `POST /v1/submissions` (real intake, in `prism-challenge`) and
//! `POST /v1/submissions/precheck` (advisory copy-gate, [`precheck_route`])
//! run the same front-end in the same order: parse the body
//! ([`parse_submission_body`]), expand a ZIP, validate the contract,
//! materialize a training-only architecture from the registry
//! ([`materialize_arch`]), then check metagraph membership
//! ([`metagraph_uid`]). Keeping that order in one place is what makes the
//! advisory verdict comparable to the one intake will return.
//!
//! The precheck route deliberately does **less** than intake: no submission
//! row, no 1-max gate, no pod. It is advisory only, and its quota
//! (`PRECHECK_DAILY_LIMIT` per coldkey per UTC day) is consumed before any
//! corpus work so a caller cannot use it as a free similarity oracle.

#![forbid(unsafe_code)]
// The fallible helpers return a ready-to-send `Response`, not an error type
// worth a prose `# Errors` section (same choice as `prism-attribution`).
#![allow(clippy::missing_errors_doc)]

mod bearer;

pub use bearer::{
    hash_matches, load_token_hashes, require_bearer, token_hash, BearerGate, CallerToken,
};

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{post, MethodRouter};
use axum::{Json, Router};
use serde_json::json;

use prism_pipeline::{
    ephemeral_candidate, evaluate_copy_precheck, precheck_json, precheck_quota_exceeded_json,
    precheck_skipped, quota_identity, quota_view, utc_day, SubmissionError, SubmissionRequest,
    PRECHECK_DAILY_LIMIT,
};
use prism_store::PrismStore;
use submission_gating::MetagraphCache;

/// Precheck state: the submission store (arch registry + quota + champion
/// corpus) and the metagraph cache (membership + coldkey).
#[derive(Clone)]
pub struct PrecheckState {
    /// Submission rows, arch registry, precheck quota.
    pub store: Arc<dyn PrismStore>,
    /// Cached metagraph snapshot. `None` disables the membership check
    /// (tests/dev).
    pub metagraph: Option<Arc<MetagraphCache>>,
}

/// Standalone router (offline tests / alternate mounts).
pub fn precheck_router(st: PrecheckState) -> Router {
    Router::new()
        .route("/v1/submissions/precheck", post(post_precheck))
        .with_state(st)
}

/// `POST /v1/submissions/precheck` with its own state applied — mounts into
/// a parent router of any state.
pub fn precheck_route<S>(
    store: Arc<dyn PrismStore>,
    metagraph: Option<Arc<MetagraphCache>>,
) -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    post(post_precheck).with_state(PrecheckState { store, metagraph })
}

/// A submission body: JSON sources, JSON + `zip_base64`, or raw
/// `application/zip` with `X-Miner-Hotkey` (+ optional `X-Prism-Arch-Id` for
/// transitional legacy training-only when `PRISM_ALLOW_LEGACY_INTAKE=1`).
pub fn parse_submission_body(
    headers: &axum::http::HeaderMap,
    body: &[u8],
) -> Result<SubmissionRequest, String> {
    let ct = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if ct.contains("application/zip") || ct.contains("application/x-zip-compressed") {
        let hk = headers
            .get("x-miner-hotkey")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| "X-Miner-Hotkey required for application/zip".to_owned())?;
        let arch_id = headers
            .get("x-prism-arch-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        // Keep raw ZIP bytes so recipe ≥ 2.0 AutoModel expand can apply the
        // patch; legacy paths still go through expand when allowed.
        let zip_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, body);
        return Ok(SubmissionRequest {
            miner_hotkey: hk.to_owned(),
            architecture_py: String::new(),
            training_py: String::new(),
            zip_base64: Some(zip_b64),
            arch_id,
            label: None,
            automodel_base: None,
            automodel_patch: None,
            prism_toml: None,
            packed_tree: None,
        });
    }
    serde_json::from_slice(body).map_err(|e| format!("invalid_json: {e}"))
}

/// Metagraph membership only (fail closed when configured but empty).
#[allow(clippy::result_large_err)] // mirrors other intake helpers returning `Response`
pub fn metagraph_uid(
    metagraph: Option<&MetagraphCache>,
    hotkey: &str,
) -> Result<Option<u32>, Response> {
    let Some(cache) = metagraph else {
        return Ok(None);
    };
    match cache.snapshot() {
        Some(view) => match view.uid_of_hex(hotkey) {
            Some(u) => Ok(Some(u)),
            None => Err(json_err(
                StatusCode::FORBIDDEN,
                "hotkey_not_in_metagraph",
                "miner hotkey is not registered on this subnet",
            )),
        },
        None => Err(json_err(
            StatusCode::SERVICE_UNAVAILABLE,
            "metagraph_unavailable",
            "metagraph snapshot not ready; retry shortly",
        )),
    }
}

/// The coldkey owning `hotkey` per the live snapshot (`None` off-chain), so
/// quota / corpus dedup can key on the owner rather than a rotatable hotkey.
#[must_use]
pub fn coldkey_of(metagraph: Option<&MetagraphCache>, hotkey: &str) -> Option<String> {
    metagraph
        .and_then(MetagraphCache::snapshot)
        .and_then(|v| v.coldkey_hex_of(hotkey))
}

/// Materialize a training-only request's architecture from the registry
/// (`Ok(())` for architecture submissions — nothing to pull).
pub async fn materialize_arch(
    store: &dyn PrismStore,
    req: &mut SubmissionRequest,
) -> Result<(), Response> {
    let Some(arch_id) = req
        .arch_id
        .as_deref()
        .map(str::trim)
        .map(str::to_ascii_lowercase)
    else {
        return Ok(());
    };
    match store.get_arch(&arch_id).await {
        Ok(Some(rec)) => {
            req.architecture_py = rec.architecture_py;
            req.arch_id = Some(arch_id);
            Ok(())
        }
        Ok(None) => Err(json_err(
            StatusCode::NOT_FOUND,
            "unknown_arch",
            "arch_id is not in the published architecture registry",
        )),
        Err(e) => Err(json_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "store",
            &e.to_string(),
        )),
    }
}

/// The shared front-end: parse → expand ZIP → validate → materialize arch →
/// normalize hotkey → membership. `Err` is the response to return as-is.
async fn front_end(
    st: &PrecheckState,
    headers: &axum::http::HeaderMap,
    body: &[u8],
) -> Result<SubmissionRequest, Response> {
    let mut req = parse_submission_body(headers, body)
        .map_err(|e| json_err(StatusCode::BAD_REQUEST, "invalid_submission", &e))?;
    prism_pipeline::expand_zip_fields(&mut req).map_err(|e| map_zip_err(&e))?;
    prism_pipeline::validate(&req).map_err(|e| map_submission_err(&e))?;
    if !prism_pipeline::is_automodel_request(&req) {
        materialize_arch(st.store.as_ref(), &mut req).await?;
        prism_recipe::check_contract(&req.architecture_py, &req.training_py)
            .map_err(|e| json_err(StatusCode::BAD_REQUEST, "contract", &e.to_string()))?;
    }
    req.miner_hotkey = req.miner_hotkey.trim().to_ascii_lowercase();
    metagraph_uid(st.metagraph.as_deref(), &req.miner_hotkey)?;
    Ok(req)
}

/// `POST /v1/submissions/precheck` — advisory copy-gate (same front-end as
/// intake), no submission row, no 1-max gate, no pod. Quota:
/// `PRECHECK_DAILY_LIMIT` per coldkey per UTC day, consumed before the
/// corpus comparison.
async fn post_precheck(
    State(st): State<PrecheckState>,
    headers: axum::http::HeaderMap,
    body: bytes::Bytes,
) -> Response {
    let req = match front_end(&st, &headers, body.as_ref()).await {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    let miner_coldkey = coldkey_of(st.metagraph.as_deref(), &req.miner_hotkey);
    let (identity, identity_kind) = quota_identity(&req.miner_hotkey, miner_coldkey.as_deref());
    let day = utc_day(now_ms() / 1000);
    let used = match st
        .store
        .precheck_quota_try_consume(&identity, &day, PRECHECK_DAILY_LIMIT)
        .await
    {
        Ok(Some(n)) => n,
        Ok(None) => {
            let used = st
                .store
                .precheck_quota_get(&identity, &day)
                .await
                .unwrap_or(PRECHECK_DAILY_LIMIT);
            let q = quota_view(day, used, identity_kind);
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(precheck_quota_exceeded_json(&q)),
            )
                .into_response();
        }
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let quota = quota_view(day, used, identity_kind);
    if req.arch_id.is_some() {
        return Json(precheck_json(&precheck_skipped(quota))).into_response();
    }
    let candidate = ephemeral_candidate(
        &req.miner_hotkey,
        miner_coldkey,
        &req.architecture_py,
        now_ms(),
    );
    let recent = st.store.list_champions(64).await.unwrap_or_default();
    let result = evaluate_copy_precheck(&candidate, &recent, quota);
    Json(precheck_json(&result)).into_response()
}

/// A rejected [`SubmissionRequest`] as a 400 with the contract error text.
#[must_use]
pub fn map_submission_err(err: &SubmissionError) -> Response {
    json_err(StatusCode::BAD_REQUEST, err.code(), &err.to_string())
}

/// Map expand/apply errors to stable API codes (`unsupported_layout`, …).
#[must_use]
pub fn map_zip_err(err: &str) -> Response {
    let code = if err.starts_with("unsupported_layout") {
        "unsupported_layout"
    } else if err.starts_with("recipe_version") {
        "recipe_version"
    } else if err.starts_with("pin ") || err.starts_with("pin mismatch") || err.contains("pin ") {
        "pin"
    } else if err.contains("patch") || err.contains("conflict") {
        "patch"
    } else {
        "zip"
    };
    json_err(StatusCode::BAD_REQUEST, code, err)
}

/// The API's uniform error envelope.
#[must_use]
pub fn json_err(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": message,
            "code": code,
        })),
    )
        .into_response()
}

/// Unix epoch milliseconds (saturating; never panics on a skewed clock).
#[must_use]
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}
