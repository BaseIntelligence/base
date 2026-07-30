//! Composed axum application: health + registry API + challenge proxy + raw weights + bundles.

use axum::Router;
use metrics_exporter_prometheus::PrometheusHandle;

use crate::api::{registry_router, GatewayState};
use crate::proxy::proxy_router;
use crate::sealer::bundle_router;
use crate::tls::TlsConfig;
use crate::weights::weights_router;
use crate::GatewayError;

/// Build the full gateway router (health + registry + proxy + raw weights + bundles).
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

    let health = telemetry::health_router(metrics)?;
    let app = health
        .merge(registry_router(state.clone()))
        .merge(weights_router(state.clone()))
        .merge(bundle_router(state.clone()))
        .merge(proxy_router(state));
    Ok(app)
}
