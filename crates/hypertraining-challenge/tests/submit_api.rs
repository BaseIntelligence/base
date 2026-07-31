//! Todo 14: miner POST /v1/submissions (brief §7) — axum oneshot + fixture tree inject.
//!
//! No real git clone: sealed admission uses in-memory path→bytes maps only.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use hypertraining_challenge::{
    example_valid_request, submission_router, SubmissionRequest, SubmissionService,
};
use hypertraining_sealed::{
    sealed_symbol_ast_hash, sha256_hex, DatasetPin, SealedSurfaceV1, SegmentPin,
    DEFAULT_SEALED_SYMBOL_KEYS,
};
use serde_json::{json, Value};
use tower::ServiceExt;

fn sealed_fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hypertraining-sealed/tests/fixtures")
}

fn load_tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut map = BTreeMap::new();
    load_tree_rec(dir, dir, &mut map);
    map
}

fn load_tree_rec(root: &Path, dir: &Path, map: &mut BTreeMap<String, Vec<u8>>) {
    for ent in fs::read_dir(dir).expect("read_dir") {
        let ent = ent.expect("entry");
        let path = ent.path();
        if path.is_dir() {
            load_tree_rec(root, &path, map);
        } else {
            let rel = path
                .strip_prefix(root)
                .expect("strip")
                .to_string_lossy()
                .replace('\\', "/");
            map.insert(rel, fs::read(&path).expect("read"));
        }
    }
}

fn baseline_manifest(files: &BTreeMap<String, Vec<u8>>) -> SealedSurfaceV1 {
    let mut m = SealedSurfaceV1::with_pins(
        "basedeadbeef",
        DatasetPin {
            corpus: "fineweb-edu".into(),
            revision: "rev1".into(),
            order_seed: 42,
        },
        SegmentPin {
            tokens: 1_000_000,
            gbs: 8,
            seq_len: 2048,
        },
    );
    for path in [
        "megatron/core/datasets/blended.py",
        "megatron/core/num_microbatches_calculator.py",
    ] {
        let bytes = files.get(path).unwrap_or_else(|| panic!("missing {path}"));
        m.denylist_hashes.insert(path.to_owned(), sha256_hex(bytes));
    }
    let training = std::str::from_utf8(
        files
            .get("megatron/training/training.py")
            .expect("training.py"),
    )
    .expect("utf8");
    for key in DEFAULT_SEALED_SYMBOL_KEYS {
        let h = sealed_symbol_ast_hash(key, training).expect("ast hash");
        m.sealed_symbols.insert((*key).to_owned(), h);
    }
    m
}

async fn post_json(body: &Value) -> (StatusCode, Value) {
    let svc = Arc::new(SubmissionService::default());
    let app = submission_router(svc);
    let bytes = serde_json::to_vec(body).unwrap();
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from(bytes))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = res.status();
    let raw = res.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&raw).unwrap_or(json!(null));
    (status, v)
}

/// S1 happy: valid brief §7 JSON → 202 Accepted.
#[tokio::test]
async fn s1_valid_submission_accepted() {
    let req = example_valid_request();
    let body = serde_json::to_value(&req).unwrap();
    let (st, v) = post_json(&body).await;
    assert_eq!(st, StatusCode::ACCEPTED, "body={v}");
    assert_eq!(v["status"], "accepted");
    assert_eq!(v["challenge_id"], "hypertraining");
    assert!(
        v["submission_id"].as_str().unwrap().starts_with("ht-sub-"),
        "{v}"
    );
}

/// S2 edge: missing required field → 400.
#[tokio::test]
async fn s2_missing_tree_sha_400() {
    let mut body = serde_json::to_value(example_valid_request()).unwrap();
    body.as_object_mut().unwrap().remove("tree_sha");
    let (st, v) = post_json(&body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "body={v}");
    assert_eq!(v["code"], "invalid_json");
}

/// S2b empty body → 400.
#[tokio::test]
async fn s2b_empty_body_400() {
    let svc = Arc::new(SubmissionService::default());
    let app = submission_router(svc);
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/submissions")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

/// S3 policy: `allow_tf32` true when default policy forbids → 400.
#[tokio::test]
async fn s3_allow_tf32_true_rejected() {
    let mut req = example_valid_request();
    req.precision_attestation.allow_tf32 = true;
    let body = serde_json::to_value(&req).unwrap();
    let (st, v) = post_json(&body).await;
    assert_eq!(st, StatusCode::BAD_REQUEST, "body={v}");
    assert_eq!(v["code"], "attestation_rejected");
    let msg = v["error"].as_str().unwrap_or("");
    assert!(
        msg.contains("allow_tf32") || msg.contains("TF32") || msg.contains("tf32"),
        "msg={msg}"
    );
}

/// S4 fixture tree inject: sealed admit without network/git clone.
#[test]
fn s4_fixture_tree_inject_admits_without_clone() {
    let root = sealed_fixtures_root().join("good_fork");
    assert!(
        root.is_dir(),
        "sealed good_fork fixture missing at {}",
        root.display()
    );
    let files = load_tree(&root);
    let manifest = baseline_manifest(&files);
    let changed = vec!["megatron/core/fusions/softmax.py".to_owned()];
    let svc = SubmissionService::default();
    svc.admit_with_fixture_tree(&changed, &files, &manifest)
        .expect("fixture admit must succeed offline");
    // Still accept HTTP body independently of tree fetch.
    let q = svc
        .accept(example_valid_request())
        .expect("accept after fixture admit");
    assert_eq!(q.request.topology.tp, 4);
}

/// Regression: health stays up next to submit routes.
#[tokio::test]
async fn s_health_ok() {
    let app = submission_router(Arc::new(SubmissionService::default()));
    let res = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

/// Ensure request type round-trips the normative example shape.
#[test]
fn s_request_json_shape_matches_brief() {
    let raw = r#"{
      "repo_url": "https://example.invalid/miner/fork.git",
      "commit_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "topology": { "tp": 4, "pp": 2, "ep": 8, "cp": 1 },
      "precision_attestation": {
        "format": "fp8_e4m3",
        "accumulate_dtype": "fp32",
        "accumulate_interval": 128,
        "scaling_recipe": "delayed",
        "allow_tf32": false
      }
    }"#;
    let req: SubmissionRequest = serde_json::from_str(raw).expect("parse brief §7");
    assert_eq!(req.topology.ep, 8);
    assert!(!req.precision_attestation.allow_tf32);
}

/// Real-surface: bind TCP, curl-equivalent reqwest POST (no git).
#[tokio::test]
async fn s_surface_bind_and_http_client() {
    use hypertraining_challenge::example_valid_request;
    use std::net::SocketAddr;
    use tokio::net::TcpListener;

    let svc = Arc::new(SubmissionService::default());
    let app = submission_router(svc);
    let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let health = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), reqwest::StatusCode::OK);

    let ok = client
        .post(format!("http://{addr}/v1/submissions"))
        .json(&example_valid_request())
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status(), reqwest::StatusCode::ACCEPTED);
    let v: Value = ok.json().await.unwrap();
    assert_eq!(v["status"], "accepted");

    let bad = client
        .post(format!("http://{addr}/v1/submissions"))
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), reqwest::StatusCode::BAD_REQUEST);
}
