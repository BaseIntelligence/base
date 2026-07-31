//! HTTP pack catalog + stripped pack delivery for miners.
//!
//! Routes (mounted at challenge root, proxied via gateway `/challenge/{id}/*`):
//! - `GET /v1/catalog` → pin, `catalog_digest`, entries
//! - `GET /v1/packs/{pack_id}` → application/gzip stripped pack tar

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_pack::{
    export_stripped_tar_gz, load_catalog, load_pack, materialize_catalog, Catalog, CatalogError,
    CatalogManifest, DEEPAGENT_PIN, PACKS_DIR_NAME,
};
use axum::extract::{Path as AxumPath, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};
use thiserror::Error;

/// In-memory pack catalog backed by a verified cache directory.
#[derive(Debug, Clone)]
pub struct PackCatalogState {
    cache_dir: PathBuf,
    catalog: Catalog,
}

/// Pack serve / bootstrap failures.
#[derive(Debug, Error)]
pub enum PackServeError {
    /// Catalog materialize/load failed.
    #[error("catalog: {0}")]
    Catalog(#[from] CatalogError),
    /// I/O.
    #[error("io {path}: {message}")]
    Io { path: PathBuf, message: String },
    /// Pack missing from catalog.
    #[error("pack not found: {0}")]
    NotFound(String),
    /// Export failure.
    #[error("export: {0}")]
    Export(String),
    /// Source bootstrap empty.
    #[error("pack source empty or missing: {0}")]
    EmptySource(String),
}

impl PackCatalogState {
    /// Materialize `source_dir` into `cache_dir` and load the verified catalog.
    ///
    /// # Errors
    /// Propagates materialize/load failures.
    pub fn open_from_source(source_dir: &Path, cache_dir: &Path) -> Result<Self, PackServeError> {
        ensure_pack_source(source_dir)?;
        materialize_catalog(source_dir, cache_dir)?;
        let catalog = load_catalog(cache_dir)?;
        Ok(Self {
            cache_dir: cache_dir.to_path_buf(),
            catalog,
        })
    }

    /// Load an already-materialized cache (no re-copy).
    ///
    /// # Errors
    /// Catalog load / integrity.
    pub fn open_cache(cache_dir: &Path) -> Result<Self, PackServeError> {
        let catalog = load_catalog(cache_dir)?;
        Ok(Self {
            cache_dir: cache_dir.to_path_buf(),
            catalog,
        })
    }

    /// Borrow verified catalog.
    #[must_use]
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// JSON document for `GET /v1/catalog`.
    #[must_use]
    pub fn catalog_json(&self) -> Value {
        let m: &CatalogManifest = self.catalog.manifest();
        let entries: Vec<Value> = m
            .entries
            .iter()
            .map(|e| {
                json!({
                    "pack_id": e.pack_id,
                    "pack_digest": e.pack_digest,
                    "environment_image_digest": e.environment_image_digest,
                })
            })
            .collect();
        json!({
            "pin": m.pin,
            "catalog_digest": m.catalog_digest,
            "entry_count": m.entries.len(),
            "entries": entries,
            "deepagent_pin_const": DEEPAGENT_PIN,
        })
    }

    /// Export stripped pack as gzip tar bytes.
    ///
    /// # Errors
    /// Missing pack or export failure.
    pub fn pack_tar_gz(&self, pack_id: &str) -> Result<Vec<u8>, PackServeError> {
        let known = self.catalog.pack_ids();
        if !known.iter().any(|id| id.as_str() == pack_id) {
            return Err(PackServeError::NotFound(pack_id.to_owned()));
        }
        let dir = self.cache_dir.join(PACKS_DIR_NAME).join(pack_id);
        let pack = load_pack(&dir).map_err(|e| PackServeError::Export(e.to_string()))?;
        export_stripped_tar_gz(&pack).map_err(|e| PackServeError::Export(e.to_string()))
    }

    /// True when catalog has at least one pack.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        !self.catalog.is_empty()
    }
}

/// Ensure pack source exists and is non-empty, optionally filling from seed / auto paths.
///
/// Env:
/// - `BASE_PACK_SEED` — directory to copy when `source` is empty
/// - `BASE_HF_AUTO_PULL=1` — if seed unset, try well-known local HF pull path
///
/// # Errors
/// Empty source after bootstrap attempts.
pub fn ensure_pack_source(source: &Path) -> Result<(), PackServeError> {
    if source_has_packs(source) {
        return Ok(());
    }

    if let Ok(seed) = std::env::var("BASE_PACK_SEED") {
        let seed_path = PathBuf::from(seed);
        if source_has_packs(&seed_path) {
            copy_dir_contents(&seed_path, source)?;
            if source_has_packs(source) {
                return Ok(());
            }
        }
    }

    let auto = std::env::var("BASE_HF_AUTO_PULL").unwrap_or_default();
    if matches!(auto.as_str(), "1" | "true" | "TRUE" | "yes") {
        let candidates = [
            PathBuf::from("/tmp/da_m18c_hf_pull/tasks"),
            PathBuf::from("/var/lib/base/pack-seed"),
        ];
        for c in candidates {
            if source_has_packs(&c) {
                copy_dir_contents(&c, source)?;
                if source_has_packs(source) {
                    return Ok(());
                }
            }
        }
    }

    Err(PackServeError::EmptySource(source.display().to_string()))
}

fn source_has_packs(dir: &Path) -> bool {
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() && p.join("task.toml").is_file() {
            return true;
        }
    }
    false
}

fn copy_dir_contents(from: &Path, to: &Path) -> Result<(), PackServeError> {
    fs::create_dir_all(to).map_err(|e| PackServeError::Io {
        path: to.to_path_buf(),
        message: e.to_string(),
    })?;
    for entry in fs::read_dir(from).map_err(|e| PackServeError::Io {
        path: from.to_path_buf(),
        message: e.to_string(),
    })? {
        let entry = entry.map_err(|e| PackServeError::Io {
            path: from.to_path_buf(),
            message: e.to_string(),
        })?;
        let src = entry.path();
        let name = entry.file_name();
        let dst = to.join(&name);
        if src.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else if src.is_file() {
            fs::copy(&src, &dst).map_err(|e| PackServeError::Io {
                path: dst,
                message: e.to_string(),
            })?;
        }
    }
    Ok(())
}

fn copy_dir_recursive(from: &Path, to: &Path) -> Result<(), PackServeError> {
    fs::create_dir_all(to).map_err(|e| PackServeError::Io {
        path: to.to_path_buf(),
        message: e.to_string(),
    })?;
    for entry in fs::read_dir(from).map_err(|e| PackServeError::Io {
        path: from.to_path_buf(),
        message: e.to_string(),
    })? {
        let entry = entry.map_err(|e| PackServeError::Io {
            path: from.to_path_buf(),
            message: e.to_string(),
        })?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        if src.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else if src.is_file() {
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent).map_err(|e| PackServeError::Io {
                    path: parent.to_path_buf(),
                    message: e.to_string(),
                })?;
            }
            fs::copy(&src, &dst).map_err(|e| PackServeError::Io {
                path: dst,
                message: e.to_string(),
            })?;
        }
    }
    Ok(())
}

/// Axum routes for pack catalog + download.
pub fn pack_routes(state: Arc<PackCatalogState>) -> Router {
    Router::new()
        .route("/v1/catalog", get(get_catalog))
        .route("/v1/packs/{pack_id}", get(get_pack))
        .with_state(state)
}

async fn get_catalog(State(state): State<Arc<PackCatalogState>>) -> Json<Value> {
    Json(state.catalog_json())
}

async fn get_pack(
    State(state): State<Arc<PackCatalogState>>,
    AxumPath(pack_id): AxumPath<String>,
) -> Response {
    match state.pack_tar_gz(&pack_id) {
        Ok(bytes) => {
            let mut res = Response::new(axum::body::Body::from(bytes));
            *res.status_mut() = StatusCode::OK;
            res.headers_mut().insert(
                header::CONTENT_TYPE,
                header::HeaderValue::from_static("application/gzip"),
            );
            if let Ok(val) =
                header::HeaderValue::from_str(&format!("attachment; filename=\"{pack_id}.tar.gz\""))
            {
                res.headers_mut().insert(header::CONTENT_DISPOSITION, val);
            }
            res
        }
        Err(PackServeError::NotFound(_)) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "pack_not_found", "pack_id": pack_id})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "pack_export_failed", "message": e.to_string()})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn seed_fixture_source(dest: &Path) {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../agent-pack/tests/fixtures/minimal-ok");
        let pack_dest = dest.join("minimal-ok");
        copy_dir_recursive(&fixture, &pack_dest).expect("seed");
    }

    #[tokio::test]
    async fn catalog_and_pack_routes_ok() {
        let tmp = tempfile::tempdir().expect("tmp");
        let source = tmp.path().join("source");
        let cache = tmp.path().join("cache");
        fs::create_dir_all(&source).unwrap();
        seed_fixture_source(&source);
        let state = Arc::new(PackCatalogState::open_from_source(&source, &cache).expect("open"));
        assert!(state.is_ready());
        let app = pack_routes(state.clone());

        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/catalog")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert!(v["entry_count"].as_u64().unwrap() >= 1);
        assert!(!v["catalog_digest"].as_str().unwrap().is_empty());
        assert_eq!(v["entries"][0]["pack_id"], "minimal-ok");

        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/packs/minimal-ok")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        assert!(bytes.len() > 32);
        assert_eq!(&bytes[0..2], &[0x1f, 0x8b]);
    }
}
