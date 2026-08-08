//! Axum funding routes (merge into a challenge router).

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use crate::error::FundingError;
use crate::service::FundingService;

/// HTTP state.
#[derive(Clone)]
pub struct FundingHttpState {
    /// Service.
    pub service: Arc<FundingService>,
}

/// Funding routes under `/v1/funding/*`.
pub fn funding_router(state: FundingHttpState) -> Router {
    Router::new()
        .route("/v1/funding/quote", get(get_quote).post(post_quote))
        .route("/v1/funding/status", get(get_status))
        .route("/v1/funding/admin/credits", get(admin_credits))
        .with_state(Arc::new(state))
}

#[derive(Debug, Deserialize)]
pub struct QuoteQuery {
    pub challenge_id: Option<String>,
    pub hotkey: String,
}

#[derive(Debug, Deserialize)]
pub struct QuoteBody {
    pub challenge_id: Option<String>,
    pub hotkey: String,
}

async fn get_quote(
    State(st): State<Arc<FundingHttpState>>,
    Query(q): Query<QuoteQuery>,
) -> Response {
    quote_inner(&st, q.challenge_id.as_deref(), &q.hotkey).await
}

async fn post_quote(
    State(st): State<Arc<FundingHttpState>>,
    Json(body): Json<QuoteBody>,
) -> Response {
    quote_inner(&st, body.challenge_id.as_deref(), &body.hotkey).await
}

async fn quote_inner(st: &FundingHttpState, challenge_id: Option<&str>, hotkey: &str) -> Response {
    if let Err(resp) = check_challenge(st, challenge_id) {
        return resp;
    }
    match st.service.quote(hotkey).await {
        Ok(q) => (StatusCode::OK, Json(q)).into_response(),
        Err(e) => err_response(e),
    }
}

async fn get_status(
    State(st): State<Arc<FundingHttpState>>,
    Query(q): Query<QuoteQuery>,
) -> Response {
    if let Err(resp) = check_challenge(&st, q.challenge_id.as_deref()) {
        return resp;
    }
    match st.service.status(&q.hotkey).await {
        Ok(s) => (StatusCode::OK, Json(s)).into_response(),
        Err(e) => err_response(e),
    }
}

async fn admin_credits(State(st): State<Arc<FundingHttpState>>, headers: HeaderMap) -> Response {
    if !admin_ok(&st, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error":"unauthorized"})),
        )
            .into_response();
    }
    match st.service.list_credits().await {
        Ok(list) => (StatusCode::OK, Json(json!({"credits": list}))).into_response(),
        Err(e) => err_response(e),
    }
}

fn check_challenge(st: &FundingHttpState, challenge_id: Option<&str>) -> Result<(), Response> {
    if let Some(cid) = challenge_id {
        if cid != st.service.challenge_id() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": format!(
                        "challenge_id mismatch: got {cid}, expected {}",
                        st.service.challenge_id()
                    )
                })),
            )
                .into_response());
        }
    }
    Ok(())
}

fn admin_ok(st: &FundingHttpState, headers: &HeaderMap) -> bool {
    let Some(expected) = st.service.cfg().admin_token.as_deref() else {
        return false;
    };
    let Some(auth) = headers.get(axum::http::header::AUTHORIZATION) else {
        return false;
    };
    let Ok(s) = auth.to_str() else {
        return false;
    };
    s.strip_prefix("Bearer ").is_some_and(|t| t == expected)
}

fn err_response(e: FundingError) -> Response {
    let code = match &e {
        FundingError::Ineligible(_) => StatusCode::FORBIDDEN,
        FundingError::Credit(_) | FundingError::Payment(_) => StatusCode::PAYMENT_REQUIRED,
        FundingError::Config(_) => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::BAD_REQUEST,
    };
    (code, Json(json!({"error": e.to_string()}))).into_response()
}
