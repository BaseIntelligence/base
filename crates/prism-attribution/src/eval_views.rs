//! The v3 read-only eval routes: raw metric rows per zone, the anchor set
//! registry, and the anchor pre-registration commits.
//!
//! Read-only counterparts to [`crate::zone_b`] — same split rationale (the
//! per-crate non-test LOC cap) and the same mounting pattern: each route
//! carries its own narrow state, so it drops into the challenge router
//! without widening `AppState`.
//!
//! Auth: as documented for `/anchors` and `/preregistration`, the API ships
//! no bearer middleware at this layer. These are reads of already-public
//! competition state (organizer measurements, anchor commits); front them at
//! the gateway where the deployment needs it.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, MethodRouter};
use axum::{Json, Router};
use serde::Deserialize;

use prism_eval_store::{anchors_view, prereg_view, zone_a_view, zone_b_view};
use prism_store::eval::EvalStore;
use prism_store::PrismStore;

use crate::json_err;

/// Metrics state: the submission store (existence check) and the eval store
/// (Zone A rows, Zone B report chain).
#[derive(Debug, Clone)]
pub struct MetricsState {
    /// Submission rows.
    pub submissions: Arc<dyn PrismStore>,
    /// Zone A metrics + Zone B report chain.
    pub eval: Arc<dyn EvalStore>,
}

/// Standalone router (offline tests / alternate mounts).
pub fn eval_views_router(st: MetricsState) -> Router {
    let eval = Arc::clone(&st.eval);
    Router::new()
        .route("/v1/submissions/{id}/metrics", get(get_metrics))
        .with_state(st)
        .route("/v1/anchors", anchors_route(Arc::clone(&eval)))
        .route("/v1/preregistration", prereg_route(eval))
}

/// `GET /v1/submissions/{id}/metrics` with its own state applied.
pub fn metrics_route<S>(
    submissions: Arc<dyn PrismStore>,
    eval: Arc<dyn EvalStore>,
) -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    get(get_metrics).with_state(MetricsState { submissions, eval })
}

/// `GET /v1/anchors` with its own state applied.
pub fn anchors_route<S>(eval: Arc<dyn EvalStore>) -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    get(get_anchors).with_state(eval)
}

/// `GET /v1/preregistration` with its own state applied.
pub fn prereg_route<S>(eval: Arc<dyn EvalStore>) -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    get(get_preregistration).with_state(eval)
}

#[derive(Debug, Deserialize)]
struct MetricsQuery {
    /// `a` (organizer-measured `org.*` rows) | `b` (participant-reported
    /// chain); default `a`.
    zone: Option<String>,
}

/// `GET /v1/submissions/{id}/metrics?zone=a|b` — raw metric rows. Zone B is
/// always labelled participant-reported and never feeds scoring.
async fn get_metrics(
    State(st): State<MetricsState>,
    Path(id): Path<String>,
    Query(q): Query<MetricsQuery>,
) -> Response {
    match st.submissions.get(&id).await {
        Ok(Some(_)) => {}
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "not_found", "submission not found"),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
    match q.zone.as_deref().unwrap_or("a") {
        "a" => {
            let (metrics, mirrors) = match st.eval.eval_run(&id).await {
                Ok(Some(r)) => (
                    st.eval.eval_metrics(&r.run_id).await.unwrap_or_default(),
                    st.eval.eval_mirrors(&r.run_id).await.unwrap_or_default(),
                ),
                _ => (Vec::new(), Vec::new()),
            };
            Json(zone_a_view(&metrics, &mirrors)).into_response()
        }
        "b" => match st.eval.metric_reports(&id).await {
            Ok(reports) => Json(zone_b_view(&reports)).into_response(),
            Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
        },
        other => json_err(
            StatusCode::BAD_REQUEST,
            "invalid_zone",
            &format!("zone must be 'a' or 'b', got '{other}'"),
        ),
    }
}

/// `GET /v1/anchors` — every known anchor set with status.
async fn get_anchors(State(eval): State<Arc<dyn EvalStore>>) -> Response {
    match eval.anchor_sets().await {
        Ok(sets) => Json(anchors_view(&sets)).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}

/// `GET /v1/preregistration` — anchor pre-registration hash-commits.
async fn get_preregistration(State(eval): State<Arc<dyn EvalStore>>) -> Response {
    match eval.preregistrations().await {
        Ok(entries) => Json(prereg_view(&entries)).into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    }
}
