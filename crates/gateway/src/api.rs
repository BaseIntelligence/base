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
use serde::Deserialize;
use trustroot::ChallengesBody;
use uuid::Uuid;

use crate::sealer::SharedBundleStore;
use crate::weights::SharedWeightStore;
use gateway_registry::{BackendView, CreateBackend, Registry, RegistryError};

/// Shared chain handle; `Send + Sync` because axum state crosses tasks.
pub type SharedChain = Arc<dyn chain::ChainClient + Send + Sync>;

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
    /// Measurements digest used at seal (must match validator local root).
    pub measurements_digest: [u8; 32],
    /// Default netuid for admin seal.
    pub seal_netuid: u16,
    /// Chain the seal pins against (metagraph, block hash, tip).
    pub chain: SharedChain,
    /// `frame-ancestors` allowlist for the viewer lockdown headers the proxy
    /// re-applies to miner-controlled `/challenge/*/v1/view/*` responses.
    pub view_frame_ancestors: String,
}

impl GatewayState {
    /// Injected chain, trust root, weight store, and bundle store.
    ///
    /// # Errors
    ///
    /// When the reqwest client cannot be built.
    pub fn with_parts(
        registry: Arc<Registry>,
        chain: SharedChain,
        challenges: Arc<ChallengesBody>,
        weights: SharedWeightStore,
        bundles: SharedBundleStore,
    ) -> Result<Self, String> {
        Self::with_parts_seal(
            registry,
            chain,
            challenges,
            weights,
            bundles,
            trustroot::measurements_digest(&trustroot::MeasurementsBody::default()),
            1,
        )
    }

    /// [`Self::with_parts`] plus the seal measurements digest and netuid.
    ///
    /// # Errors
    ///
    /// When the reqwest client cannot be built.
    pub fn with_parts_seal(
        registry: Arc<Registry>,
        chain: SharedChain,
        challenges: Arc<ChallengesBody>,
        weights: SharedWeightStore,
        bundles: SharedBundleStore,
        measurements_digest: [u8; 32],
        seal_netuid: u16,
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
            measurements_digest,
            seal_netuid,
            chain,
            view_frame_ancestors: std::env::var(crate::gw_config::keys::VIEW_FRAME_ANCESTORS)
                .unwrap_or_else(|_| design_sanitize::default_frame_ancestors().to_owned()),
        })
    }

    /// Override the viewer `frame-ancestors` allowlist (tests, custom fronts).
    #[must_use]
    pub fn with_view_frame_ancestors(mut self, frame_ancestors: impl Into<String>) -> Self {
        self.view_frame_ancestors = frame_ancestors.into();
        self
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
