//! Integration: PackCatalogState + HTTP pack routes (stripped delivery).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_challenge::{pack_routes, PackCatalogState};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use flate2::read::GzDecoder;
use http_body_util::BodyExt;
use tar::Archive;
use tempfile::tempdir;
use tower::ServiceExt;

fn fixture_pack() -> PathBuf {
    let real = PathBuf::from("/tmp/da_m18c_hf_pull/tasks/realpr-more-itertools-1136");
    if real.join("task.toml").is_file() {
        return real;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../agent-pack/tests/fixtures/minimal-ok")
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let src = e.path();
        let dst = to.join(e.file_name());
        if src.is_dir() {
            copy_dir(&src, &dst);
        } else if src.is_file() {
            fs::copy(&src, &dst).unwrap();
        }
    }
}

fn stage_source(tmp: &Path) -> (PathBuf, String) {
    let src_pack = fixture_pack();
    let pack_id = src_pack.file_name().unwrap().to_string_lossy().into_owned();
    let source = tmp.join("source");
    copy_dir(&src_pack, &source.join(&pack_id));
    (source, pack_id)
}

#[test]
fn open_from_source_catalog_json_and_tar_gz() {
    let tmp = tempdir().unwrap();
    let (source, pack_id) = stage_source(tmp.path());
    let cache = tmp.path().join("cache");

    let state = PackCatalogState::open_from_source(&source, &cache).expect("open");
    assert!(state.is_ready());
    assert!(state.catalog().len() >= 1);

    let cat = state.catalog_json();
    assert!(!cat["pin"].as_str().unwrap().is_empty());
    assert!(!cat["catalog_digest"].as_str().unwrap().is_empty());
    assert!(cat["entry_count"].as_u64().unwrap() >= 1);
    let ids: Vec<&str> = cat["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["pack_id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&pack_id.as_str()), "ids={ids:?}");

    let bytes = state.pack_tar_gz(&pack_id).expect("tar");
    assert_eq!(&bytes[0..2], &[0x1f, 0x8b]);
    assert!(bytes.len() > 64);

    // Unpack and assert no solution/tests
    let unpack = tmp.path().join("unpack");
    fs::create_dir_all(&unpack).unwrap();
    {
        let dec = GzDecoder::new(bytes.as_slice());
        let mut archive = Archive::new(dec);
        archive.unpack(&unpack).expect("unpack");
    }
    assert!(!unpack.join("solution").exists());
    assert!(!unpack.join("tests").exists());
    assert!(unpack.join("task.toml").is_file() || find_task_toml(&unpack).is_some());

    let err = state.pack_tar_gz("does-not-exist").unwrap_err();
    assert!(err.to_string().contains("not found"));
}

#[tokio::test]
async fn http_catalog_and_pack_routes() {
    let tmp = tempdir().unwrap();
    let (source, pack_id) = stage_source(tmp.path());
    let cache = tmp.path().join("cache");
    let state = Arc::new(PackCatalogState::open_from_source(&source, &cache).unwrap());
    let app = pack_routes(state);

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
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(v["entry_count"].as_u64().unwrap() >= 1);
    assert!(!v["catalog_digest"].as_str().unwrap().is_empty());

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/packs/{pack_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let ct = res
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.contains("gzip"), "ct={ct}");
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&bytes[0..2], &[0x1f, 0x8b]);

    // Gunzip + list paths — no solution/tests
    let mut dec = GzDecoder::new(bytes.as_ref());
    let mut raw = Vec::new();
    dec.read_to_end(&mut raw).unwrap();
    let mut archive = Archive::new(raw.as_slice());
    for entry in archive.entries().unwrap() {
        let entry = entry.unwrap();
        let path = entry.path().unwrap().to_string_lossy().replace('\\', "/");
        let norm = path.trim_start_matches("./");
        assert!(!norm.starts_with("solution/"), "leaked {norm}");
        assert!(!norm.starts_with("tests/"), "leaked {norm}");
    }

    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/packs/missing-pack-xyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

fn find_task_toml(dir: &Path) -> Option<PathBuf> {
    if dir.join("task.toml").is_file() {
        return Some(dir.join("task.toml"));
    }
    for e in fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(f) = find_task_toml(&p) {
                return Some(f);
            }
        }
    }
    None
}
