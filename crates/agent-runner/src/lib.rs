//! Miner CVM agent HTTP surface (`AGENT_CHALLENGE` compose `agent:8080`).
//!
//! Routes:
//! - `GET /healthz` / `GET /readyz`
//! - `GET /v1/capacity` — effective `max_concurrency` (clamped `1..=5`) + current load
//! - `POST /v1/task` — accept signed [`TaskDescriptorV1`] envelope (auth default ON)
//! - `GET /v1/task/{id}` — status; when terminal, [`TaskResultV1`] (patch + receipt)
//!
//! # Auth (todo 18)
//! When [`RunnerConfig::auth_enabled`] is true (default), `POST /v1/task` requires a
//! [`SignedDispatchRequest`] under [`crypto::domain::DISPATCH`] with a single-use
//! nonce and TTL shorter than one epoch. Health/ready/capacity stay open.
//!
//! # Concurrency (todo 19)
//! Miner-declared `max_concurrency` is clamped to `1..=5` via [`clamp_concurrency`].
//! Accept acquires a semaphore permit; over-capacity returns **HTTP 503** with code
//! `capacity_exhausted` (retryable; never unbounded-queued). Capacity reports the
//! effective clamped max and occupied slot count.
//!
//! # Pack execution + egress (todo 21)
//! [`ExecutionBackend::Docker`] pulls a digest-pinned env image via allowlisted
//! socket-proxy, runs the reference agent, collects `/logs/artifacts/model.patch`,
//! signs a work receipt, and tears down `gbase-verify-agent-*` containers.
//! Default egress posture is [`DEFAULT_AGENT_EGRESS_POSTURE`] (**OPEN**): no network
//! lockdown. Stripping protects grading-channel integrity, not miner honesty (D19).

#![forbid(unsafe_code)]

mod api;
mod auth;
mod egress;
mod executor;
mod receipt_key;
mod store;

pub use api::{router, ApiError, CapacityResponse, TaskAccepted, TaskView};
pub use auth::{
    dispatch_auth_payload, sign_dispatch_request, unix_now_ms, verify_and_consume_dispatch,
    DispatchAuthError, SignedDispatchRequest, DEFAULT_DISPATCH_NONCE_TTL,
};
pub use egress::{AgentEgressPosture, DEFAULT_AGENT_EGRESS_POSTURE, EGRESS_POSTURE_OPEN_LABEL};
pub use executor::{
    count_agent_containers, execute_pack, load_stripped, reference_agent_cmd, resolve_timeout_sec,
    DockerExecConfig, ExecOutcome, ExecutionBackend, AGENT_CONTAINER_PREFIX,
    MODEL_KEY_CONTAINER_PATH, MODEL_KEY_FILE_ENV, MODEL_PATCH_REL,
};
pub use receipt_key::{
    load_or_generate, load_required, receipt_sk_path_from_env, ReceiptKey, ReceiptKeyError,
    DEFAULT_RECEIPT_SK_PATH, RECEIPT_SK_FILE_ENV,
};
pub use store::{
    clamp_concurrency, CapacityExhausted, RunnerConfig, RunnerState, TaskLifecycle,
    MAX_CONCURRENCY_BOUND, MIN_CONCURRENCY,
};

use axum::Router;

/// Build the full agent-runner router with shared state.
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
    use agent_dispatch::{patch_sha256, TaskDescriptorV1, TaskStatusV1, DISPATCH_PROTOCOL};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use crypto::KEY_LEN;
    use http_body_util::BodyExt;
    use schnorrkel::MiniSecretKey;
    use serde_json::{json, Value};
    use std::time::Duration;
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
        std::mem::forget(dir);
        load_or_generate(&path).expect("receipt key")
    }

    fn cfg_base(max: u32) -> RunnerConfig {
        RunnerConfig {
            max_concurrency: max,
            auth_enabled: false,
            trusted_challenge_pubkey: None,
            dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
            receipt_key: Some(test_receipt_key()),
            execution: ExecutionBackend::Stub {
                hold: Duration::ZERO,
            },
            egress_posture: DEFAULT_AGENT_EGRESS_POSTURE,
        }
    }

    fn auth_state(max: u32, pk: [u8; KEY_LEN]) -> RunnerState {
        RunnerState::new(RunnerConfig {
            auth_enabled: true,
            trusted_challenge_pubkey: Some(pk),
            ..cfg_base(max)
        })
    }

    fn auth_state_with_receipt(max: u32, pk: [u8; KEY_LEN], receipt: ReceiptKey) -> RunnerState {
        RunnerState::new(RunnerConfig {
            max_concurrency: max,
            auth_enabled: true,
            trusted_challenge_pubkey: Some(pk),
            dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
            receipt_key: Some(receipt),
            execution: ExecutionBackend::Stub {
                hold: Duration::ZERO,
            },
            egress_posture: DEFAULT_AGENT_EGRESS_POSTURE,
        })
    }

    fn open_state(max: u32) -> RunnerState {
        RunnerState::new(cfg_base(max))
    }

    fn open_state_hold(max: u32, hold: Duration) -> RunnerState {
        RunnerState::new(RunnerConfig {
            execution: ExecutionBackend::Stub { hold },
            ..cfg_base(max)
        })
    }

    fn open_state_no_receipt(max: u32) -> RunnerState {
        RunnerState::new(RunnerConfig {
            receipt_key: None,
            ..cfg_base(max)
        })
    }

    fn sample_desc() -> TaskDescriptorV1 {
        TaskDescriptorV1::new(
            "agent-v1",
            2,
            7,
            "aa".repeat(32),
            "pack-fixture-001",
            unix_now_ms() + 3_600_000,
        )
    }

    fn sample_desc_deadline(deadline_unix_ms: u64) -> TaskDescriptorV1 {
        TaskDescriptorV1::new(
            "agent-v1",
            2,
            7,
            "aa".repeat(32),
            "pack-fixture-001",
            deadline_unix_ms,
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

    async fn post_plain(state: &RunnerState, desc: &TaskDescriptorV1) -> axum::response::Response {
        let body = serde_json::to_vec(desc).expect("ser");
        app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .expect("req"),
            )
            .await
            .expect("oneshot")
    }

    async fn poll_terminal(state: &RunnerState, task_id: &str) -> Value {
        let mut terminal: Option<Value> = None;
        for _ in 0..100 {
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
            if matches!(st, "completed" | "failed" | "timed_out") {
                terminal = Some(v);
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        terminal.expect("task reached terminal status")
    }

    /// S3 — clamp edges: 0→1, 1→1, 5→5, 6→5, 9999→5.
    #[test]
    fn clamp_concurrency_bounds_1_to_5() {
        assert_eq!(clamp_concurrency(0), 1);
        assert_eq!(clamp_concurrency(1), 1);
        assert_eq!(clamp_concurrency(3), 3);
        assert_eq!(clamp_concurrency(5), 5);
        assert_eq!(clamp_concurrency(6), 5);
        assert_eq!(clamp_concurrency(9999), 5);
        assert_eq!(MIN_CONCURRENCY, 1);
        assert_eq!(MAX_CONCURRENCY_BOUND, 5);
    }

    /// Egress posture locked OPEN (todo 21).
    #[test]
    fn egress_default_is_open_documented_label() {
        assert_eq!(DEFAULT_AGENT_EGRESS_POSTURE, AgentEgressPosture::Open);
        assert_eq!(
            DEFAULT_AGENT_EGRESS_POSTURE.as_str(),
            EGRESS_POSTURE_OPEN_LABEL
        );
        assert!(!DEFAULT_AGENT_EGRESS_POSTURE.network_disabled());
        let st = open_state(1);
        assert_eq!(st.egress_posture(), AgentEgressPosture::Open);
    }

    /// S2 — capacity reports effective clamp for 0 and 9999 (no panic).
    #[tokio::test]
    async fn capacity_reports_clamped_effective_max() {
        let zero = open_state(0);
        let cap0 = zero.capacity_async().await;
        assert_eq!(cap0.max_concurrency, 1);
        assert_eq!(cap0.current_load, 0);
        assert_eq!(zero.effective_max_concurrency(), 1);

        let huge = open_state(9999);
        let cap_h = huge.capacity_async().await;
        assert_eq!(cap_h.max_concurrency, 5);
        assert_eq!(cap_h.current_load, 0);
        assert_eq!(huge.effective_max_concurrency(), 5);

        let app0 = app(open_state(0));
        let res = app0
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
        assert_eq!(v["max_concurrency"], 1);

        let app_h = app(open_state(9999));
        let res = app_h
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
        assert_eq!(v["max_concurrency"], 5);
    }

    /// S1 — max=2: two accepted while held; third 503 `capacity_exhausted`; after free, next 202.
    #[tokio::test]
    async fn over_capacity_refused_then_succeeds_when_slot_frees() {
        let state = open_state_hold(2, Duration::from_millis(400));
        let before = state.task_count().await;

        let r1 = post_plain(&state, &sample_desc()).await;
        assert_eq!(r1.status(), StatusCode::ACCEPTED, "first must accept");
        let r2 = post_plain(&state, &sample_desc()).await;
        assert_eq!(r2.status(), StatusCode::ACCEPTED, "second must accept");

        tokio::time::sleep(Duration::from_millis(30)).await;
        let cap = state.capacity_async().await;
        assert_eq!(cap.max_concurrency, 2);
        assert_eq!(cap.current_load, 2, "both slots occupied");

        let r3 = post_plain(&state, &sample_desc()).await;
        assert_eq!(
            r3.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "third must be refused while full"
        );
        let v3 = body_json(r3).await;
        assert_eq!(v3["code"], "capacity_exhausted");
        assert!(v3.get("error").is_some());
        assert_eq!(state.task_count().await, before + 2);

        tokio::time::sleep(Duration::from_millis(500)).await;
        let cap_after = state.capacity_async().await;
        assert_eq!(cap_after.current_load, 0, "slots freed after completion");

        let r4 = post_plain(&state, &sample_desc()).await;
        assert_eq!(
            r4.status(),
            StatusCode::ACCEPTED,
            "dispatch succeeds once a slot frees"
        );
        assert_eq!(state.task_count().await, before + 3);
    }

    /// Deadline already passed → `timed_out`, no patch, signed receipt.
    #[tokio::test]
    async fn past_deadline_returns_timed_out_with_signed_receipt() {
        let receipt = test_receipt_key();
        let receipt_pk = *receipt.public_key();
        let state = RunnerState::new(RunnerConfig {
            receipt_key: Some(receipt),
            ..cfg_base(1)
        });
        let desc = sample_desc_deadline(1);
        let res = post_plain(&state, &desc).await;
        assert_eq!(res.status(), StatusCode::ACCEPTED);
        let accepted = body_json(res).await;
        let task_id = accepted["task_id"].as_str().expect("id").to_owned();
        let v = poll_terminal(&state, &task_id).await;
        assert_eq!(v["status"], "timed_out");
        let result = v.get("result").expect("result");
        assert_eq!(result["status"], "timed_out");
        assert!(result.get("model_patch").is_none() || result["model_patch"].is_null());
        let sig_hex = result["receipt_sig_hex"].as_str().expect("sig");
        assert_eq!(sig_hex.len(), 128);
        let digest = patch_sha256(b"");
        assert_eq!(result["patch_sha256_hex"], hex::encode(digest));
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
                    patch_sha256: digest,
                },
                signature: sig,
            },
        )
        .expect("timed-out receipt still verifies");
        let _ = TaskStatusV1::TimedOut;
    }

    /// Completed stub: `patch_sha256` matches returned bytes.
    #[tokio::test]
    async fn completed_patch_sha256_matches_bytes() {
        let state = open_state(1);
        let res = post_plain(&state, &sample_desc()).await;
        assert_eq!(res.status(), StatusCode::ACCEPTED);
        let accepted = body_json(res).await;
        let task_id = accepted["task_id"].as_str().expect("id").to_owned();
        let v = poll_terminal(&state, &task_id).await;
        assert_eq!(v["status"], "completed");
        let result = v.get("result").expect("result");
        let patch = result["model_patch"].as_str().expect("patch");
        assert!(patch.contains("diff --git"));
        assert!(!patch.is_empty());
        let expected = hex::encode(patch_sha256(patch.as_bytes()));
        assert_eq!(result["patch_sha256_hex"], expected);
    }

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
        assert!(
            res.status() == StatusCode::BAD_REQUEST || res.status() == StatusCode::UNAUTHORIZED,
            "status={}",
            res.status()
        );
        let v = body_json(res).await;
        assert!(v.get("error").is_some(), "typed error: {v}");
    }

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

    #[tokio::test]
    async fn post_signed_and_poll_task_to_completion() {
        let (sk, pk) = mini_pair(3);
        let receipt = test_receipt_key();
        let receipt_pk = *receipt.public_key();
        let state = auth_state_with_receipt(1, pk, receipt);
        let router = app(state.clone());
        let now = unix_now_ms();
        let req = sign_dispatch_request(&sk, &pk, sample_desc(), [0x10; KEY_LEN], now + 60_000)
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

        let v = poll_terminal(&state, &task_id).await;
        assert_eq!(v["status"], "completed");
        assert_eq!(v["task_id"], task_id);
        let result = v.get("result").expect("result present when complete");
        assert_eq!(result["protocol"], DISPATCH_PROTOCOL);
        assert_eq!(result["status"], "completed");
        let patch = result["model_patch"].as_str().expect("model_patch string");
        assert!(
            patch.contains("diff --git"),
            "stub patch should look like a unified diff: {patch}"
        );
        let sig_hex = result["receipt_sig_hex"].as_str().expect("receipt_sig_hex");
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

    #[tokio::test]
    async fn post_replay_nonce_rejected() {
        let (sk, pk) = mini_pair(4);
        let state = auth_state(2, pk);
        let now = unix_now_ms();
        let envelope =
            sign_dispatch_request(&sk, &pk, sample_desc(), [0x20; KEY_LEN], now + 60_000)
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
        assert!(!v["error"]
            .as_str()
            .unwrap_or("")
            .contains(&envelope.signature_hex));
        assert_eq!(state.task_count().await, 0);
    }

    #[test]
    fn crate_name_is_agent_runner() {
        assert_eq!(crate_name(), "agent-runner");
    }

    #[tokio::test]
    async fn capacity_load_non_negative() {
        let state = open_state(2);
        let cap = state.capacity();
        assert_eq!(cap.max_concurrency, 2);
        assert_eq!(cap.current_load, 0);
        let _ = json!({"ok": true});
    }

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

        let v = poll_terminal(&state, &task_id).await;
        assert_eq!(v["status"], "failed");
        let result = v.get("result").expect("result");
        assert_eq!(result["status"], "failed");
        let sig = result["receipt_sig_hex"].as_str().unwrap_or("x");
        assert!(sig.is_empty() || sig != "00".repeat(64));
    }

    #[tokio::test]
    async fn signed_dispatch_over_capacity_returns_503() {
        let (sk, pk) = mini_pair(7);
        let state = RunnerState::new(RunnerConfig {
            max_concurrency: 1,
            auth_enabled: true,
            trusted_challenge_pubkey: Some(pk),
            dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
            receipt_key: Some(test_receipt_key()),
            execution: ExecutionBackend::Stub {
                hold: Duration::from_millis(300),
            },
            egress_posture: DEFAULT_AGENT_EGRESS_POSTURE,
        });
        let now = unix_now_ms();
        let env1 = sign_dispatch_request(&sk, &pk, sample_desc(), [0x41; KEY_LEN], now + 60_000)
            .expect("sign1");
        let env2 = sign_dispatch_request(&sk, &pk, sample_desc(), [0x42; KEY_LEN], now + 60_000)
            .expect("sign2");

        let r1 = app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&env1).unwrap()))
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(r1.status(), StatusCode::ACCEPTED);

        tokio::time::sleep(Duration::from_millis(20)).await;

        let r2 = app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/task")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&env2).unwrap()))
                    .expect("req"),
            )
            .await
            .expect("oneshot");
        assert_eq!(r2.status(), StatusCode::SERVICE_UNAVAILABLE);
        let v = body_json(r2).await;
        assert_eq!(v["code"], "capacity_exhausted");
        assert_eq!(state.task_count().await, 1);
    }
}
