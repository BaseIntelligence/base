//! Verified-bundle mirror store and `GET /v1/bundle/root/{root}` (task 30).
//!
//! Validators persist SCALE bytes only after independent verification succeeds
//! far enough that the merkle root is trustworthy (`Match` / `VectorMismatch`).
//! Peers fetch by content-addressed root; the response is self-authenticating.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use gbase_bundle::EpochBundleV1;

/// Shared verified-bundle mirror handle.
pub type SharedMirrorStore = Arc<MemoryMirrorStore>;

/// In-memory store of verified sealed bundles, keyed by merkle root and epoch.
#[derive(Debug, Default)]
pub struct MemoryMirrorStore {
    by_root: RwLock<BTreeMap<[u8; 32], Vec<u8>>>,
    by_epoch: RwLock<BTreeMap<u64, [u8; 32]>>,
}

impl MemoryMirrorStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Shared empty store.
    #[must_use]
    pub fn shared() -> SharedMirrorStore {
        Arc::new(Self::new())
    }

    /// Persist SCALE bytes under `root` / `epoch` (idempotent per root).
    ///
    /// Call only after the bundle has been verified (or inputs verified for class A).
    pub fn put_verified(&self, epoch: u64, root: [u8; 32], bytes: Vec<u8>) {
        {
            let mut by_root = self.by_root.write().unwrap_or_else(std::sync::PoisonError::into_inner);
            by_root.entry(root).or_insert(bytes);
        }
        let mut by_epoch = self.by_epoch.write().unwrap_or_else(std::sync::PoisonError::into_inner);
        by_epoch.entry(epoch).or_insert(root);
    }

    /// Lookup by merkle root.
    #[must_use]
    pub fn get_by_root(&self, root: &[u8; 32]) -> Option<Vec<u8>> {
        self.by_root
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(root)
            .cloned()
    }

    /// Lookup by epoch (returns root then bytes).
    #[must_use]
    pub fn get_by_epoch(&self, epoch: u64) -> Option<Vec<u8>> {
        let root = {
            let by_epoch = self.by_epoch.read().unwrap_or_else(std::sync::PoisonError::into_inner);
            by_epoch.get(&epoch).copied()?
        };
        self.get_by_root(&root)
    }

    /// Whether any bundle is stored under `root`.
    #[must_use]
    pub fn contains_root(&self, root: &[u8; 32]) -> bool {
        self.by_root
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(root)
    }
}

/// Axum routes that re-serve verified bundles by content-addressed root.
pub fn mirror_router(store: SharedMirrorStore) -> Router {
    Router::new()
        .route("/v1/bundle/root/{root}", get(get_bundle_by_root))
        .with_state(store)
}

fn octet_or_404(bytes: Option<Vec<u8>>) -> Response {
    match bytes {
        Some(b) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/octet-stream")],
            b,
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "bundle not found" })),
        )
            .into_response(),
    }
}

async fn get_bundle_by_root(
    State(store): State<SharedMirrorStore>,
    Path(root_hex): Path<String>,
) -> Response {
    match parse_root_hex(&root_hex) {
        Ok(root) => octet_or_404(store.get_by_root(&root)),
        Err(msg) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": msg })),
        )
            .into_response(),
    }
}

/// Parse a 32-byte merkle root from hex (optional `0x`, case-insensitive).
///
/// # Errors
///
/// Wrong length or non-hex input.
pub fn parse_root_hex(s: &str) -> Result<[u8; 32], String> {
    let s = s.trim();
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    if s.len() != 64 {
        return Err(format!("merkle root must be 64 hex chars, got {}", s.len()));
    }
    let bytes = hex::decode(s).map_err(|e| e.to_string())?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| format!("expected 32 bytes, got {}", v.len()))
}

/// Encode merkle root as lowercase hex (no `0x`).
#[must_use]
pub fn root_hex(root: &[u8; 32]) -> String {
    hex::encode(root)
}

/// Decode SCALE bytes and return `(epoch, merkle_root)` when well-formed.
#[must_use]
pub fn bundle_identity(bytes: &[u8]) -> Option<(u64, [u8; 32])> {
    let b = EpochBundleV1::decode_bytes(bytes).ok()?;
    Some((b.body.epoch, b.body.merkle_root))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_root_hex_roundtrip() {
        let root = [0xabu8; 32];
        let h = root_hex(&root);
        assert_eq!(h.len(), 64);
        assert_eq!(parse_root_hex(&h).unwrap(), root);
        assert_eq!(parse_root_hex(&format!("0x{h}")).unwrap(), root);
    }

    #[test]
    fn store_put_get() {
        let store = MemoryMirrorStore::new();
        let root = [1u8; 32];
        store.put_verified(7, root, vec![9, 9, 9]);
        assert_eq!(store.get_by_root(&root).unwrap(), vec![9, 9, 9]);
        assert_eq!(store.get_by_epoch(7).unwrap(), vec![9, 9, 9]);
        // idempotent
        store.put_verified(7, root, vec![1, 2, 3]);
        assert_eq!(store.get_by_root(&root).unwrap(), vec![9, 9, 9]);
    }
}
