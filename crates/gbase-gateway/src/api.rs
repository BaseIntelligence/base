//! Operational backend registry HTTP API (routing only — D18).
//!
//! Paths align with validator `MASTER_ONLY_PATHS`: `/v1/admin/backends`.
//! Request bodies use `#[serde(deny_unknown_fields)]` so `signing_key` → 400.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use gbase_trustroot::ChallengesBody;
use serde::Deserialize;
use uuid::Uuid;

use crate::registry::{BackendView, CreateBackend, Registry, RegistryError};
use crate::sealer::{MemoryBundleStore, SharedBundleStore};
use crate::weights::{MemoryRawWeightStore, SharedWeightStore};

/// Shared axum state for registry + proxy + weights + sealed bundles.
#[derive(Clone)]
pub struct GatewayState {
    /// Backend registry.
    pub registry: Arc<Registry>,
    /// Outbound HTTP client used by the reverse proxy.
    pub client: reqwest::Client,
    /// Local owner-signed challenges body (D18).
    pub challenges: Arc<ChallengesBody>,
    /// Append-only raw-weight store.
    pub weights: SharedWeightStore,
    /// Sealed epoch bundles.
    pub bundles: SharedBundleStore,
}

impl GatewayState {
    /// Empty trust root and in-memory stores.
    ///
    /// # Errors
    ///
    /// When the reqwest client cannot be built.
    pub fn new(registry: Arc<Registry>) -> Result<Self, String> {
        Self::with_parts(
            registry,
            Arc::new(ChallengesBody::default()),
            Arc::new(MemoryRawWeightStore::new()),
            Arc::new(MemoryBundleStore::new()),
        )
    }

    /// Injected trust root, weight store, and bundle store.
    ///
    /// # Errors
    ///
    /// When the reqwest client cannot be built.
    pub fn with_parts(
        registry: Arc<Registry>,
        challenges: Arc<ChallengesBody>,
        weights: SharedWeightStore,
        bundles: SharedBundleStore,
    ) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            registry,
            client,
            challenges,
            weights,
            bundles,
        })
    }
}

/// Query for `GET /v1/admin/backends`.
#[derive(Debug, Deserialize)]
pub struct ListQuery {
    /// Optional challenge filter.
    pub challenge_id: Option<String>,
}

/// Mount registry CRUD under `/v1/admin/backends`.
pub fn registry_router(state: GatewayState) -> Router {
    Router::new()
        .route(
            "/v1/admin/backends",
            get(list_backends).post(create_backend),
        )
        .route(
            "/v1/admin/backends/{id}",
            get(get_backend).delete(delete_backend),
        )
        .with_state(state)
}

async fn create_backend(
    State(st): State<GatewayState>,
    Json(body): Json<CreateBackend>,
) -> Result<(StatusCode, Json<BackendView>), ApiError> {
    let view = st.registry.create(&body)?;
    Ok((StatusCode::CREATED, Json(view)))
}

async fn list_backends(
    State(st): State<GatewayState>,
    Query(q): Query<ListQuery>,
) -> Json<Vec<BackendView>> {
    Json(st.registry.list(q.challenge_id.as_deref()))
}

async fn get_backend(
    State(st): State<GatewayState>,
    Path(id): Path<Uuid>,
) -> Result<Json<BackendView>, ApiError> {
    Ok(Json(st.registry.get(id)?))
}

async fn delete_backend(
    State(st): State<GatewayState>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    st.registry.delete(id)?;
    Ok(StatusCode::NO_CONTENT)
}

/// API error → HTTP.
#[derive(Debug)]
pub struct ApiError(pub RegistryError);

impl From<RegistryError> for ApiError {
    fn from(value: RegistryError) -> Self {
        Self(value)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            RegistryError::NotFound(_) | RegistryError::NoBackends(_) => StatusCode::NOT_FOUND,
            RegistryError::Duplicate { .. } => StatusCode::CONFLICT,
            RegistryError::Invalid(_) => StatusCode::BAD_REQUEST,
        };
        let body = serde_json::json!({
            "error": self.0.to_string(),
        });
        (status, Json(body)).into_response()
    }
}
