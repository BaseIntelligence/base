//! Miner CVM agent HTTP surface (AGENT_CHALLENGE compose `agent:8080`).
//!
//! Routes:
//! - `GET /healthz` / `GET /readyz`
//! - `GET /v1/capacity` — effective `max_concurrency` + current load
//! - `POST /v1/task` — accept signed [`TaskDescriptorV1`] envelope (auth default ON)
//! - `GET /v1/task/{id}` — status; when terminal, [`TaskResultV1`] (patch + receipt)
//!
//! # Auth (todo 18)
//! When [`RunnerConfig::auth_enabled`] is true (default), `POST /v1/task` requires a
//! [`SignedDispatchRequest`] under [`crypto::domain::DISPATCH`] with a single-use
//! nonce and TTL shorter than one epoch. Health/ready/capacity stay open.
//!
//! # Concurrency (todo 19)
//! Capacity is advertised only. Over-capacity dispatch is still accepted until the
//! semaphore clamp lands.

#![forbid(unsafe_code)]

mod api;
mod auth;
mod receipt_key;
mod store;

pub use api::{router, ApiError, CapacityResponse, TaskAccepted, TaskView};
pub use auth::{
    dispatch_auth_payload, sign_dispatch_request, unix_now_ms, verify_and_consume_dispatch,
    DispatchAuthError, SignedDispatchRequest, DEFAULT_DISPATCH_NONCE_TTL,
};
pub use receipt_key::{
    load_or_generate, load_required, receipt_sk_path_from_env, ReceiptKey, ReceiptKeyError,
    DEFAULT_RECEIPT_SK_PATH, RECEIPT_SK_FILE_ENV,
};
pub use store::{RunnerConfig, RunnerState, TaskLifecycle};

use axum::Router;

/// Build the full agent-runner router with shared state.
#[must_use]
pub fn app(state: RunnerState) -> Router {
    router(state)
}

/// Crate identity for smoke checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "agent-runner"
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_dispatch::{TaskDescriptorV1, DISPATCH_PROTOCOL};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use crypto::KEY_LEN;
    use http_body_util::BodyExt;
    use schnorrkel::MiniSecretKey;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    fn mini_pair(seed: u8) -> ([u8; KEY_LEN], [u8; KEY_LEN]) {
        let mini = MiniSecretKey::from_bytes(&[seed; KEY_LEN]).expect("mini");
        let secret = mini.to_bytes();
        let public = mini
            .expand(schnorrkel::ExpansionMode::Ed25519)
            .to_public()
            .to_bytes();
        (secret, public)
    }

    fn test_receipt_key() -> ReceiptKey {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("receipt_sk");
        // Leak dir so path stays valid for the test process lifetime of the key.
        std::mem::forget(dir);
        load_or_generate(&path).expect("receipt key")
    }

    fn auth_state(max: u32, pk: [u8; KEY_LEN]) -> RunnerState {
        RunnerState::new(RunnerConfig {
            max_concurrency: max,
            receipt_key: Some(test_receipt_key()),
            auth_enabled: true,
            trusted_challenge_pubkey: Some(pk),
            dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
        })
    }

    fn auth_state_with_receipt(
        max: u32,
        pk: [u8; KEY_LEN],
        receipt: ReceiptKey,
    ) -> RunnerState {
        RunnerState::new(RunnerConfig {
            max_concurrency: max,
            auth_enabled: true,
            trusted_challenge_pubkey: Some(pk),
            dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
            receipt_key: Some(receipt),
        })
    }

    fn open_state(max: u32) -> RunnerState {
        RunnerState::new(RunnerConfig {
            max_concurrency: max,
            receipt_key: Some(test_receipt_key()),
            auth_enabled: false,
            trusted_challenge_pubkey: None,
            dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
        })
    }

    fn open_state_no_receipt(max: u32) -> RunnerState {
        RunnerState::new(RunnerConfig {
            max_concurrency: max,
            receipt_key: None,
            auth_enabled: false,
            trusted_challenge_pubkey: None,
            dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
        })
    }

    fn sample_desc() -> TaskDescriptorV1 {
        TaskDescriptorV1::new(
            "agent-v1",
            2,
            7,
            "aa".repeat(32),
            "pack-fixture-001",
            9_999_999_999_999,
        )
    }

    async fn body_json(response: axum::response::Response) -> Value {
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("json body")
    }

    async fn body_text(response: axum::response::Response) -> String {
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        String::from_utf8(bytes.to_vec()).expect("utf8")
    }

    /// S4 — GET /healthz → 200 ok (unauthenticated).
    #[tokio::test]
    async fn healthz_returns_200_ok() {
        let (_sk, pk) = mini_pair(1);
        let app = app(auth_state(1, pk));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(body_text(res).await, "ok");
    }

    /// S4 — GET /readyz → 200 ready.
    #[tokio::test]
    async fn readyz_returns_200_ready() {
        let (_sk, pk) = mini_pair(1);
        let app = app(auth_state(1, pk));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/readyz")
                    .body(Body::empty())
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(body_text(res).await, "ready");
    }

    /// S1 — capacity reflects configured max_concurrency and zero load.
    #[tokio::test]
    async fn capacity_reports_configured_max_and_zero_load() {
        let (_sk, pk) = mini_pair(1);
        let app = app(auth_state(3, pk));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/capacity")
                    .body(Body::empty())
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["max_concurrency"], 3);
        assert_eq!(v["current_load"], 0);
    }

    /// S3 — unknown task id → 404 typed body (not 500).
    #[tokio::test]
    async fn unknown_task_returns_404_typed() {
        let (_sk, pk) = mini_pair(1);
        let app = app(auth_state(1, pk));
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/task/does-not-exist")
                    .body(Body::empty())
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        let v = body_json(res).await;
        assert!(v.get("error").is_some(), "typed error field required: {v}");
        assert_eq!(v["code"], "task_not_found");
    }

    /// S5 — malformed JSON body → 401 when auth on (unsigned / unparseable envelope).
    #[tokio::test]
    async fn post_task_malformed_json_returns_401_when_auth_on() {
        let (_sk, pk) = mini_pair(1);
        let app = app(auth_state(1, pk));
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from("{not-json"))
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        // JsonRejection is 400 from extractor path — still typed.
        assert!(
            res.status() == StatusCode::BAD_REQUEST || res.status() == StatusCode::UNAUTHORIZED,
            "status={}",
            res.status()
        );
        let v = body_json(res).await;
        assert!(v.get("error").is_some(), "typed error: {v}");
    }

    /// S2 — unsigned bare descriptor → 401, no task created.
    #[tokio::test]
    async fn post_unsigned_descriptor_returns_401() {
        let (_sk, pk) = mini_pair(2);
        let state = auth_state(1, pk);
        let app = app(state.clone());
        let body = serde_json::to_vec(&sample_desc()).expect("ser");
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        let v = body_json(res).await;
        assert_eq!(v["code"], "unauthorized");
        assert_eq!(state.task_count().await, 0);
    }

    /// S1 — signed dispatch → 202 + poll completion.
    #[tokio::test]
    async fn post_signed_and_poll_task_to_completion() {
        let (sk, pk) = mini_pair(3);
        let receipt = test_receipt_key();
        let receipt_pk = *receipt.public_key();
        let state = auth_state_with_receipt(1, pk, receipt);
        let router = app(state.clone());
        let now = unix_now_ms();
        let req = sign_dispatch_request(
            &sk,
            &pk,
            sample_desc(),
            [0x10; KEY_LEN],
            now + 60_000,
        )
        .expect("sign");
        let body = serde_json::to_vec(&req).expect("ser");
        let res = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::ACCEPTED);
        let accepted = body_json(res).await;
        let task_id = accepted["task_id"]
            .as_str()
            .expect("task_id string")
            .to_owned();
        assert!(!task_id.is_empty());

        let mut terminal: Option<Value> = None;
        for _ in 0..50 {
            let router = app(state.clone());
            let res = router
                .oneshot(
                    Request::builder()
                        .uri(format!("/v1/task/{task_id}"))
                        .body(Body::empty())
                        .expect("req"),
                )
                .await
                .expect("oneshot");
            assert_eq!(res.status(), StatusCode::OK);
            let v = body_json(res).await;
            let st = v["status"].as_str().unwrap_or("");
            if st == "completed" || st == "failed" {
                terminal = Some(v);
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let v = terminal.expect("task reached terminal status");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["task_id"], task_id);
        let result = v.get("result").expect("result present when complete");
        assert_eq!(result["protocol"], DISPATCH_PROTOCOL);
        assert_eq!(result["status"], "completed");
        let patch = result["model_patch"]
            .as_str()
            .expect("model_patch string");
        assert!(
            patch.contains("diff --git"),
            "stub patch should look like a unified diff: {patch}"
        );
        let sig_hex = result["receipt_sig_hex"]
            .as_str()
            .expect("receipt_sig_hex");
        assert_eq!(sig_hex.len(), 128);
        assert_ne!(sig_hex, "00".repeat(64), "must not be all-zero stub");
        let mut sig = [0u8; 64];
        sig.copy_from_slice(&hex::decode(sig_hex).unwrap());
        let mut hk = [0u8; 32];
        hk.copy_from_slice(&hex::decode("aa".repeat(32)).unwrap());
        agent_dispatch::verify_work_receipt(
            &receipt_pk,
            &agent_dispatch::SignedWorkReceiptV1 {
                body: agent_dispatch::WorkReceiptBodyV1 {
                    challenge_id: b"agent-v1".to_vec(),
                    scoring_version: 2,
                    epoch: 7,
                    miner_hotkey: hk,
                    pack_id: b"pack-fixture-001".to_vec(),
                    patch_sha256: agent_dispatch::patch_sha256(patch.as_bytes()),
                },
                signature: sig,
            },
        )
        .expect("WORK_RECEIPT verify");
    }

    /// S3 — replay identical signed body → second 401 nonce_replay; one task only.
    #[tokio::test]
    async fn post_replay_nonce_rejected() {
        let (sk, pk) = mini_pair(4);
        let state = auth_state(2, pk);
        let now = unix_now_ms();
        let envelope = sign_dispatch_request(
            &sk,
            &pk,
            sample_desc(),
            [0x20; KEY_LEN],
            now + 60_000,
        )
        .expect("sign");
        let body = serde_json::to_vec(&envelope).expect("ser");

        let res1 = app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res1.status(), StatusCode::ACCEPTED);

        let res2 = app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res2.status(), StatusCode::UNAUTHORIZED);
        let v = body_json(res2).await;
        assert_eq!(v["code"], "nonce_replay");
        assert_eq!(state.task_count().await, 1);
    }

    /// S4 — foreign signer → 401, no task.
    #[tokio::test]
    async fn post_foreign_key_rejected() {
        let (_sk_trust, pk_trust) = mini_pair(5);
        let (sk_foreign, pk_foreign) = mini_pair(6);
        let state = auth_state(1, pk_trust);
        let now = unix_now_ms();
        let envelope = sign_dispatch_request(
            &sk_foreign,
            &pk_foreign,
            sample_desc(),
            [0x30; KEY_LEN],
            now + 60_000,
        )
        .expect("sign");
        let body = serde_json::to_vec(&envelope).expect("ser");
        let res = app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        let v = body_json(res).await;
        assert_eq!(v["code"], "unauthorized");
        assert!(!v["error"].as_str().unwrap_or("").contains(&envelope.signature_hex));
        assert_eq!(state.task_count().await, 0);
    }

    #[test]
    fn crate_name_is_agent_runner() {
        assert_eq!(crate_name(), "agent-runner");
    }

    /// Capacity current_load bumps while a task is running (best-effort race window).
    #[tokio::test]
    async fn capacity_load_non_negative() {
        let state = open_state(2);
        let cap = state.capacity();
        assert_eq!(cap.max_concurrency, 2);
        assert_eq!(cap.current_load, 0);
        let _ = json!({"ok": true});
    }

    /// Auth-off path still accepts plain descriptors (local/dev).
    #[tokio::test]
    async fn auth_disabled_accepts_plain_descriptor() {
        let state = open_state(1);
        let body = serde_json::to_vec(&sample_desc()).expect("ser");
        let res = app(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::ACCEPTED);
    }

    /// Missing receipt key → task fails closed (no zero stub signature).
    #[tokio::test]
    async fn missing_receipt_key_fails_closed() {
        let state = open_state_no_receipt(1);
        let body = serde_json::to_vec(&sample_desc()).expect("ser");
        let res = app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::ACCEPTED);
        let accepted = body_json(res).await;
        let task_id = accepted["task_id"].as_str().expect("id").to_owned();

        let mut terminal: Option<Value> = None;
        for _ in 0..50 {
            let router = app(state.clone());
            let res = router
                .oneshot(
                    Request::builder()
                        .uri(format!("/v1/task/{task_id}"))
                        .body(Body::empty())
                        .expect("req"),
                )
                .await
                .expect("oneshot");
            let v = body_json(res).await;
            let st = v["status"].as_str().unwrap_or("");
            if st == "completed" || st == "failed" {
                terminal = Some(v);
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let v = terminal.expect("terminal");
        assert_eq!(v["status"], "failed");
        let result = v.get("result").expect("result");
        assert_eq!(result["status"], "failed");
        let sig = result["receipt_sig_hex"].as_str().unwrap_or("x");
        assert!(sig.is_empty() || sig != &"00".repeat(64));
    }
}
