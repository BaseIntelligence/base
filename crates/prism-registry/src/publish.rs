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

use base64::Engine;
use serde::Deserialize;

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
}

/// GitHub contents-API publisher. Token is never `Debug`/`Display`'d.
pub struct TopModelPublisher {
    http: reqwest::Client,
    api_base: String,
    repo: String,
    branch: String,
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

#[derive(Debug, Deserialize)]
struct ContentGet {
    sha: String,
}

#[derive(Debug, Deserialize)]
struct ContentPut {
    commit: CommitInfo,
}

#[derive(Debug, Deserialize)]
struct CommitInfo {
    sha: String,
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
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            api_base: api_base.into().trim_end_matches('/').to_owned(),
            repo: repo.into(),
            branch: branch.into(),
        })
    }

    /// Publish the four `top-model/` files; returns the last commit sha.
    pub async fn publish(&self, req: &TopModelRequest) -> Result<String, PublishError> {
        let arch = req.arch_id.as_deref().unwrap_or("arch-unregistered");
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
            (format!("{TOPMODEL_REPO_PATH}/README.md"), readme_block(req)),
        ];
        let mut last_sha = String::new();
        for (path, body) in files {
            last_sha = self.put_file(&path, &body, arch, req.bpb).await?;
        }
        Ok(last_sha)
    }

    /// Create-or-update one file via the contents API.
    async fn put_file(
        &self,
        path: &str,
        body: &str,
        arch: &str,
        bpb: f64,
    ) -> Result<String, PublishError> {
        let url = format!("{}/repos/{}/contents/{}", self.api_base, self.repo, path);
        let existing_sha: Option<String> = match self
            .http
            .get(&url)
            .query(&[("ref", self.branch.as_str())])
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                resp.json::<ContentGet>().await.ok().map(|c| c.sha)
            }
            Ok(_) => None, // 404 → create
            Err(e) => return Err(PublishError::Transport(e.to_string())),
        };
        let mut put = serde_json::json!({
            "message": format!("top-model: {arch} bpb={bpb:.4} ({path})"),
            "content": base64::engine::general_purpose::STANDARD.encode(body),
            "branch": self.branch,
        });
        if let Some(sha) = existing_sha {
            put["sha"] = serde_json::Value::String(sha);
        }
        let resp = self
            .http
            .put(&url)
            .json(&put)
            .send()
            .await
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(PublishError::Api(format!(
                "{status}: {}",
                &text.chars().take(200).collect::<String>()
            )));
        }
        let out: ContentPut = resp
            .json()
            .await
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        Ok(out.commit.sha)
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

fn readme_block(req: &TopModelRequest) -> String {
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
         | finish reason | `{}` |\n\n\
         Files: `architecture.py`, `training.py`, `METRICS.json` (full harness\n\
         metrics incl. telemetry loss series).\n",
        req.arch_id.as_deref().unwrap_or("arch-unregistered"),
        req.owner_hotkey.chars().take(12).collect::<String>(),
        req.bpb,
        n_params,
        req.submission_id,
        series_len,
        finish,
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
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
        }
    }

    #[tokio::test]
    async fn publishes_all_files_and_returns_commit_sha() {
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
        let md = readme_block(&req());
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
