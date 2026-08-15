//! Top-model GitHub publisher: each new global-best bpb model is published
//! to the public `BaseIntelligence/prism` repo under `top-model/`
//! (architecture.py + training.py + METRICS.json + README.md block) via the
//! GitHub contents API.
//!
//! Token discipline: the GitHub token is read from a deploy secret **file**
//! (`PRISM_TOPMODEL_GITHUB_TOKEN_FILE`, e.g. `deploy/secrets/github/token`),
//! never from env text, never logged. When the file is absent/empty the
//! publisher is `None` and the orchestrator skips publishing entirely
//! (graceful no-op).

/// Whether top-model publish must include a parked checkpoint.
///
/// Default **true** (`PRISM_TOPMODEL_REQUIRE_WEIGHTS` unset/`1`). Set to
/// `0`/`false`/`no` for source-only publish (legacy / tests).
#[must_use]
pub fn require_topmodel_weights() -> bool {
    !matches!(
        std::env::var("PRISM_TOPMODEL_REQUIRE_WEIGHTS")
            .unwrap_or_else(|_| "1".into())
            .trim(),
        "0" | "false" | "FALSE" | "no" | "NO"
    )
}

/// Repo directory that always mirrors the current global top model.
pub const TOPMODEL_REPO_PATH: &str = "top-model";

const DEFAULT_API_BASE: &str = "https://api.github.com";
const DEFAULT_REPO: &str = "BaseIntelligence/prism";
const DEFAULT_BRANCH: &str = "main";

/// Publish errors (transport / API). Fail-closed: no publication is
/// recorded by the caller when these fire.
#[derive(Debug, thiserror::Error)]
pub enum PublishError {
    /// HTTP/IO.
    #[error("transport: {0}")]
    Transport(String),
    /// GitHub API non-2xx.
    #[error("github api: {0}")]
    Api(String),
}

/// One top-model publish payload.
#[derive(Debug, Clone)]
pub struct TopModelRequest {
    /// Submission that set the global best.
    pub submission_id: String,
    /// Registry arch (when linked).
    pub arch_id: Option<String>,
    /// Miner hotkey that set the best.
    pub owner_hotkey: String,
    /// Global-best bpb.
    pub bpb: f64,
    /// architecture.py (registry source for training-only entries).
    pub architecture_py: String,
    /// training.py.
    pub training_py: String,
    /// Harness metrics blob (telemetry summary extracted for the README).
    pub metrics_json: Option<serde_json::Value>,
    /// Master-parked checkpoint path (harvested from the Lium pod).
    /// When absent and weights are required, publish is refused.
    pub checkpoint_path: Option<std::path::PathBuf>,
    /// Extra source files for Hub publish (AutoModel novelty / tree_blob
    /// modules). Paths are relative; HF publisher nests them under `sources/`.
    pub extra_files: Vec<(String, Vec<u8>)>,
}

/// GitHub contents-API publisher. Token is never `Debug`/`Display`'d.
pub struct TopModelPublisher {
    pub(crate) http: reqwest::Client,
    pub(crate) api_base: String,
    pub(crate) repo: String,
    pub(crate) branch: String,
}

impl std::fmt::Debug for TopModelPublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately redacts the HTTP client (holds the bearer token).
        f.debug_struct("TopModelPublisher")
            .field("api_base", &self.api_base)
            .field("repo", &self.repo)
            .field("branch", &self.branch)
            .field("http", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl TopModelPublisher {
    /// Build from the deploy secret file (`PRISM_TOPMODEL_GITHUB_TOKEN_FILE`).
    /// `None` when the env var is unset or the file missing/empty — the
    /// top-model step then no-ops. Repo/branch overridable for tests via
    /// `PRISM_TOPMODEL_REPO` / `PRISM_TOPMODEL_BRANCH`.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let path = std::env::var("PRISM_TOPMODEL_GITHUB_TOKEN_FILE").ok()?;
        let token = std::fs::read_to_string(path).ok()?.trim().to_owned();
        if token.len() < 8 {
            return None;
        }
        let repo = std::env::var("PRISM_TOPMODEL_REPO").unwrap_or_else(|_| DEFAULT_REPO.into());
        let branch =
            std::env::var("PRISM_TOPMODEL_BRANCH").unwrap_or_else(|_| DEFAULT_BRANCH.into());
        Self::with_config(token, DEFAULT_API_BASE, repo, branch).ok()
    }

    /// Explicit config (tests / wiremock).
    pub fn with_config(
        token: impl Into<String>,
        api_base: impl Into<String>,
        repo: impl Into<String>,
        branch: impl Into<String>,
    ) -> Result<Self, PublishError> {
        let token = token.into();
        if token.trim().is_empty() {
            return Err(PublishError::Transport("empty token".into()));
        }
        let mut headers = reqwest::header::HeaderMap::new();
        let mut hv = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        hv.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, hv);
        headers.insert(
            reqwest::header::USER_AGENT,
            reqwest::header::HeaderValue::from_static("base-prism-topmodel/0.1"),
        );
        headers.insert(
            reqwest::header::ACCEPT,
            reqwest::header::HeaderValue::from_static("application/vnd.github+json"),
        );
        let http = reqwest::Client::builder()
            .default_headers(headers)
            // Checkpoint uploads can be large; keep a generous ceiling.
            .timeout(std::time::Duration::from_mins(10))
            .build()
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            api_base: api_base.into().trim_end_matches('/').to_owned(),
            repo: repo.into(),
            branch: branch.into(),
        })
    }

    /// Publish `top-model/` sources (+ optional checkpoint). Returns last commit sha.
    ///
    /// When `PRISM_TOPMODEL_REQUIRE_WEIGHTS` is unset/true (default), a missing
    /// or unloadable checkpoint fails the publish (no journal). Set the env to
    /// `0`/`false` to allow source-only publish (legacy / tests).
    pub async fn publish(&self, req: &TopModelRequest) -> Result<String, PublishError> {
        let arch = req.arch_id.as_deref().unwrap_or("arch-unregistered");
        let require_weights = require_topmodel_weights();
        let mut weight_note = String::new();
        if let Some(path) = &req.checkpoint_path {
            let meta = self.publish_checkpoint(path, arch, req.bpb).await?;
            weight_note = format!(
                "checkpoint sha256=`{}` ({} bytes){}",
                meta.sha256,
                meta.bytes,
                meta.release_tag
                    .as_deref()
                    .map(|t| format!(" release=`{t}`"))
                    .unwrap_or_default()
            );
        } else if require_weights {
            return Err(PublishError::Transport(
                "checkpoint missing: secure receive (SSH harvest or admin /v1/admin/artifacts/.../receive) required, or set PRISM_TOPMODEL_REQUIRE_WEIGHTS=0"
                    .into(),
            ));
        }
        let files: Vec<(String, String)> = vec![
            (
                format!("{TOPMODEL_REPO_PATH}/architecture.py"),
                req.architecture_py.clone(),
            ),
            (
                format!("{TOPMODEL_REPO_PATH}/training.py"),
                req.training_py.clone(),
            ),
            (
                format!("{TOPMODEL_REPO_PATH}/METRICS.json"),
                serde_json::to_string_pretty(&metrics_blob(req)).unwrap_or_else(|_| "{}".into()),
            ),
            (
                format!("{TOPMODEL_REPO_PATH}/README.md"),
                readme_block(req, &weight_note),
            ),
        ];
        let mut last_sha = String::new();
        for (path, body) in files {
            last_sha = self
                .put_file_public(&path, body.as_bytes(), arch, req.bpb)
                .await?;
        }
        Ok(last_sha)
    }
}

fn metrics_blob(req: &TopModelRequest) -> serde_json::Value {
    let tele = req
        .metrics_json
        .as_ref()
        .and_then(|m| m.get("telemetry").cloned());
    serde_json::json!({
        "submission_id": req.submission_id,
        "arch_id": req.arch_id,
        "owner_hotkey": req.owner_hotkey,
        "bpb": req.bpb,
        "n_params": req.metrics_json.as_ref().and_then(|m| m.get("n_params")),
        "tokens_seen": req.metrics_json.as_ref().and_then(|m| m.get("tokens_seen")),
        "wall_clock_seconds": req.metrics_json.as_ref().and_then(|m| m.get("wall_clock_seconds")),
        "telemetry": tele,
    })
}

fn readme_block(req: &TopModelRequest, weight_note: &str) -> String {
    let series_len = req
        .metrics_json
        .as_ref()
        .and_then(|m| m.pointer("/telemetry/loss_series"))
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    let finish = req
        .metrics_json
        .as_ref()
        .and_then(|m| m.pointer("/telemetry/finish_reason"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    let n_params = req
        .metrics_json
        .as_ref()
        .and_then(|m| m.get("n_params"))
        .and_then(serde_json::Value::as_u64)
        .map_or("unknown".into(), |n| n.to_string());
    let weights = if weight_note.is_empty() {
        "source-only (no checkpoint parked on master)".to_owned()
    } else {
        weight_note.to_owned()
    };
    format!(
        "# PRISM top model\n\n\
         Published by the Base master on every new global-best bpb. This\n\
         directory always mirrors the current champion; history lives in git.\n\n\
         | field | value |\n|---|---|\n\
         | arch_id | `{}` |\n\
         | owner_hotkey | `{}…` |\n\
         | bpb | `{:.6}` |\n\
         | n_params | {} |\n\
         | submission | `{}` |\n\
         | telemetry points | {} |\n\
         | finish reason | `{}` |\n\
         | weights | {} |\n\n\
         Files: `architecture.py`, `training.py`, `METRICS.json`, `ARTIFACT.json`\n\
         (checkpoint sha / release pointer). Large weights ship on the\n\
         `prism-top-model` GitHub Release when they exceed the contents API.\n",
        req.arch_id.as_deref().unwrap_or("arch-unregistered"),
        req.owner_hotkey.chars().take(12).collect::<String>(),
        req.bpb,
        n_params,
        req.submission_id,
        series_len,
        finish,
        weights,
    )
}

/// Serialize unit tests that mutate `PRISM_TOPMODEL_REQUIRE_WEIGHTS`.
#[cfg(test)]
pub(crate) static TOPMODEL_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn topmodel_env_lock() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::PoisonError;
    TOPMODEL_ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::await_holding_lock)]
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn req() -> TopModelRequest {
        TopModelRequest {
            submission_id: "subm-1".into(),
            arch_id: Some("arch_0123456789abcdef".into()),
            owner_hotkey: "ab".repeat(32),
            bpb: 1.234,
            architecture_py: "def build_model(ctx):\n    pass\n".into(),
            training_py: "def train(model, ctx):\n    return {}\n".into(),
            metrics_json: Some(serde_json::json!({
                "n_params": 12_000_000,
                "telemetry": {"finish_reason": "finish_evaluation", "loss_series": [{"step": 1, "loss": 2.0}]},
            })),
            checkpoint_path: None,
            extra_files: Vec::new(),
        }
    }

    #[tokio::test]
    async fn publishes_all_files_and_returns_commit_sha() {
        let _lock = topmodel_env_lock();
        std::env::set_var("PRISM_TOPMODEL_REQUIRE_WEIGHTS", "0");
        let server = MockServer::start().await;
        for f in [
            "architecture.py",
            "training.py",
            "METRICS.json",
            "README.md",
        ] {
            Mock::given(method("GET"))
                .and(path(format!(
                    "/repos/BaseIntelligence/prism/contents/top-model/{f}"
                )))
                .respond_with(ResponseTemplate::new(404))
                .mount(&server)
                .await;
            Mock::given(method("PUT"))
                .and(path(format!(
                    "/repos/BaseIntelligence/prism/contents/top-model/{f}"
                )))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "commit": {"sha": format!("sha-{f}")}
                })))
                .mount(&server)
                .await;
        }
        let p =
            TopModelPublisher::with_config("tok", server.uri(), "BaseIntelligence/prism", "main")
                .unwrap();
        let sha = p.publish(&req()).await.unwrap();
        assert_eq!(sha, "sha-README.md");
    }

    #[tokio::test]
    async fn api_error_is_typed() {
        let _lock = topmodel_env_lock();
        std::env::set_var("PRISM_TOPMODEL_REQUIRE_WEIGHTS", "0");
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .respond_with(ResponseTemplate::new(422).set_body_string("validation failed"))
            .mount(&server)
            .await;
        let p = TopModelPublisher::with_config("tok", server.uri(), "o/r", "main").unwrap();
        let err = p.publish(&req()).await.unwrap_err();
        assert!(matches!(err, PublishError::Api(_)), "{err}");
    }

    #[test]
    fn readme_summarizes_metrics() {
        let md = readme_block(&req(), "");
        assert!(md.contains("arch_0123456789abcdef"));
        assert!(md.contains("1.234000"));
        assert!(md.contains("finish_evaluation"));
        assert!(md.contains("telemetry points | 1"));
    }

    #[test]
    fn from_env_graceful_without_file() {
        // No env var → None (no panic, no network).
        assert!(TopModelPublisher::from_env().is_none());
    }
}
