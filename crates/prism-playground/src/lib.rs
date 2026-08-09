//! Operator playground: `POST /v1/admin/playground/complete`.
//!
//! Fail-closed behind the same admin bearer as retry (`PRISM_ADMIN_TOKENS_FILE`).
//! Runs inference against a master-parked checkpoint (harvested from Lium)
//! for the current top publication or a specified `submission_id`.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{post, MethodRouter};
use axum::Json;
use prism_intake::json_err;
use prism_store::PrismStore;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::warn;

/// Narrow state for the playground route.
#[derive(Clone)]
pub struct PlaygroundState {
    /// Submission / publication store.
    pub store: Arc<dyn PrismStore>,
    /// Path to `playground_infer.py` (embedded harness copy on disk, or env).
    pub infer_script: PathBuf,
}

/// Mountable `POST /v1/admin/playground/complete`.
pub fn playground_route<S>(st: PlaygroundState) -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    post(post_complete).with_state(st)
}

#[derive(Debug, Deserialize)]
pub struct CompleteRequest {
    /// Prompt text (required).
    pub prompt: String,
    /// `top` (default) | submission id.
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_true")]
    pub return_logprobs: bool,
    #[serde(default = "default_top_logprobs")]
    pub top_logprobs: u32,
}

fn default_model() -> String {
    "top".into()
}
fn default_max_tokens() -> u32 {
    64
}
fn default_true() -> bool {
    true
}
fn default_top_logprobs() -> u32 {
    5
}

async fn post_complete(
    State(st): State<PlaygroundState>,
    Json(body): Json<CompleteRequest>,
) -> Response {
    if body.prompt.trim().is_empty() {
        return json_err(StatusCode::BAD_REQUEST, "invalid_request", "empty prompt");
    }
    if body.max_tokens == 0 || body.max_tokens > 512 {
        return json_err(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "max_tokens must be 1..=512",
        );
    }
    let resolved = match resolve_model(&st.store, &body.model).await {
        Ok(r) => r,
        Err((code, kind, msg)) => return json_err(code, kind, &msg),
    };
    let ckpt = resolved.checkpoint.clone();
    if !ckpt.is_file() {
        return json_err(
            StatusCode::SERVICE_UNAVAILABLE,
            "inference_unavailable",
            &format!("no checkpoint at {}", ckpt.display()),
        );
    }
    // Sim stub: refuse real torch load but return structured diagnostics.
    if looks_like_sim_stub(&ckpt) {
        return (
            StatusCode::OK,
            Json(json!({
                "model": {
                    "kind": resolved.kind,
                    "submission_id": resolved.submission_id,
                    "arch_id": resolved.arch_id,
                    "bpb": resolved.bpb,
                    "repo_path": "top-model",
                },
                "text": "",
                "tokens": [],
                "logprobs": [],
                "diagnostics": {
                    "sim_stub": true,
                    "checkpoint": ckpt.display().to_string(),
                    "note": "parked artifact is a Sim stub; real inference needs a Lium-harvested checkpoint.pt",
                },
            })),
        )
            .into_response();
    }
    if !st.infer_script.is_file() {
        return json_err(
            StatusCode::SERVICE_UNAVAILABLE,
            "inference_unavailable",
            "playground_infer.py not found on master",
        );
    }
    match run_infer(&st.infer_script, &resolved, &body).await {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err(e) => {
            warn!(error = %e, "playground infer failed");
            json_err(StatusCode::SERVICE_UNAVAILABLE, "inference_unavailable", &e)
        }
    }
}

struct ResolvedModel {
    kind: &'static str,
    submission_id: String,
    arch_id: Option<String>,
    bpb: Option<f64>,
    checkpoint: PathBuf,
    architecture_py: String,
    training_py: String,
}

async fn resolve_model(
    store: &Arc<dyn PrismStore>,
    model: &str,
) -> Result<ResolvedModel, (StatusCode, &'static str, String)> {
    let submission_id = if model == "top" {
        let Some(pub_row) = store
            .last_publication()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, "store", e.to_string()))?
        else {
            return Err((
                StatusCode::NOT_FOUND,
                "no_top_model",
                "no top-model publication yet".into(),
            ));
        };
        pub_row.submission_id
    } else {
        model.to_owned()
    };
    let row = store
        .get(&submission_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, "store", e.to_string()))?
        .ok_or((
            StatusCode::NOT_FOUND,
            "unknown_submission",
            submission_id.clone(),
        ))?;
    let checkpoint = prism_artifacts::checkpoint_path_for(&submission_id);
    Ok(ResolvedModel {
        kind: if model == "top" { "top" } else { "submission" },
        submission_id,
        arch_id: row.arch_id,
        bpb: row.bpb,
        checkpoint,
        architecture_py: row.architecture_py,
        training_py: row.training_py,
    })
}

fn looks_like_sim_stub(path: &Path) -> bool {
    std::fs::read(path)
        .ok()
        .is_some_and(|b| b.starts_with(b"PRISM_SIM_CKPT"))
}

async fn run_infer(
    script: &Path,
    model: &ResolvedModel,
    body: &CompleteRequest,
) -> Result<Value, String> {
    let work = tempfile_dir()?;
    std::fs::write(work.join("architecture.py"), &model.architecture_py)
        .map_err(|e| e.to_string())?;
    std::fs::write(work.join("training.py"), &model.training_py).map_err(|e| e.to_string())?;
    let req_path = work.join("request.json");
    let req = json!({
        "prompt": body.prompt,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "return_logprobs": body.return_logprobs,
        "top_logprobs": body.top_logprobs,
        "checkpoint": model.checkpoint,
        "workdir": work,
        "submission_id": model.submission_id,
        "arch_id": model.arch_id,
        "bpb": model.bpb,
        "kind": model.kind,
    });
    std::fs::write(
        &req_path,
        serde_json::to_vec(&req).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let child = tokio::process::Command::new("python3")
        .arg(script)
        .arg(&req_path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("spawn python: {e}"))?;
    let out = tokio::time::timeout(Duration::from_mins(2), child.wait_with_output())
        .await
        .map_err(|_| "infer timed out".to_owned())?
        .map_err(|e| format!("infer wait: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "infer exit {:?}: {}",
            out.status.code(),
            err.chars().take(400).collect::<String>()
        ));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("infer json: {e}"))
}

fn tempfile_dir() -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join(format!(
        "prism-play-{}-{}",
        std::process::id(),
        prism_intake::now_ms()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Default path for the infer script (harness checkout next to the binary cwd).
#[must_use]
pub fn default_infer_script() -> PathBuf {
    if let Ok(p) = std::env::var("PRISM_PLAYGROUND_INFER_SCRIPT") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../prism-recipe/harness/playground_infer.py")
}
