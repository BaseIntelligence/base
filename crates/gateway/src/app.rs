//! Composed axum application: health + registry API + challenge proxy + raw weights + bundles + site.

use std::sync::Arc;

use axum::Router;
use metrics_exporter_prometheus::PrometheusHandle;

use crate::api::{registry_router, GatewayState};
use crate::proxy::proxy_router;
use crate::sealer::{admin_seal_router, bundle_router};
use crate::tls::TlsConfig;
use crate::weights::weights_router;
use crate::GatewayError;

/// Build the full gateway router (health + registry + proxy + raw weights + bundles + site).
///
/// # Errors
///
/// Telemetry router construction failures.
pub fn build_router(
    metrics: PrometheusHandle,
    state: GatewayState,
    tls: &TlsConfig,
) -> Result<Router, GatewayError> {
    tracing::info!(
        event = "gateway_tls_mode",
        mode = tls.mode_label(),
        tls_enabled = tls.enabled,
        "TLS ownership is this process only (D20); ACME via rustls-acme is task 42"
    );

    let site = site_api::site_router(site_api::SiteState::new(
        Arc::clone(&state.registry),
        state.client.clone(),
        Some(Arc::clone(&state.chain)),
        state.seal_netuid,
    ));

    let health = telemetry::health_router(metrics)?;
    let app = health
        .merge(registry_router(state.clone()))
        .merge(weights_router(state.clone()))
        .merge(bundle_router(state.clone()))
        .merge(admin_seal_router(state.clone()))
        .merge(site)
        .merge(proxy_router(state));
    Ok(app)
}
