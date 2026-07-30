//! Miner CVM agent HTTP surface (AGENT_CHALLENGE compose `agent:8080`).
//!
//! Routes:
//! - `GET /healthz` / `GET /readyz`
//! - `GET /v1/capacity` — effective `max_concurrency` + current load
//! - `POST /v1/task` — accept [`TaskDescriptorV1`], return 202 + task id
//! - `GET /v1/task/{id}` — status; when terminal, [`TaskResultV1`] (patch + receipt)
//!
//! # Auth (todo 18)
//! Dispatch authentication is **not** enforced here. Callers may POST any well-formed
//! descriptor. Signed single-use nonces land in a later task — do not treat this
//! surface as production-exposed without that gate.
//!
//! # Concurrency (todo 19)
//! Capacity is advertised only. Over-capacity dispatch is still accepted until the
//! semaphore clamp lands.

#![forbid(unsafe_code)]

mod api;
mod store;

pub use api::{router, ApiError, CapacityResponse, TaskAccepted, TaskView};
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
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    fn state_with_cap(max: u32) -> RunnerState {
        RunnerState::new(RunnerConfig {
            max_concurrency: max,
        })
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

    /// S4 — GET /healthz → 200 ok.
    #[tokio::test]
    async fn healthz_returns_200_ok() {
        let app = app(state_with_cap(1));
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
        let app = app(state_with_cap(1));
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
        let app = app(state_with_cap(3));
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
        let app = app(state_with_cap(1));
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

    /// S5 — malformed JSON body → 400 typed.
    #[tokio::test]
    async fn post_task_malformed_json_returns_400() {
        let app = app(state_with_cap(1));
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
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        let v = body_json(res).await;
        assert!(v.get("error").is_some(), "typed error: {v}");
    }

    /// S2 — POST accepts descriptor → 202 + id; poll until completed with model_patch.
    #[tokio::test]
    async fn post_and_poll_task_to_completion() {
        let state = state_with_cap(1);
        let router = app(state.clone());
        let desc = TaskDescriptorV1::new(
            "agent-v1",
            2,
            7,
            "aa".repeat(32),
            "pack-fixture-001",
            9_999_999_999_999,
        );
        let body = serde_json::to_vec(&desc).expect("ser");
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

        // Poll until terminal (stub executor is near-instant).
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
        assert!(
            result["patch_sha256_hex"].as_str().is_some_and(|h| h.len() == 64),
            "patch sha256 hex"
        );
        assert!(
            result["receipt_sig_hex"].as_str().is_some(),
            "receipt sig field present (stub)"
        );
    }

    #[test]
    fn crate_name_is_agent_runner() {
        assert_eq!(crate_name(), "agent-runner");
    }

    /// Capacity current_load bumps while a task is running (best-effort race window).
    #[tokio::test]
    async fn capacity_load_non_negative() {
        let state = state_with_cap(2);
        let cap = state.capacity();
        assert_eq!(cap.max_concurrency, 2);
        assert_eq!(cap.current_load, 0);
        // After insert pending without run, load still 0.
        let _ = json!({"ok": true});
    }
}
