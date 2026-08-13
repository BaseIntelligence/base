//! HuggingFace Hub top-model publisher.
//!
//! When a submission becomes the new global-best bpb, source files
//! (`architecture.py`, `training.py`, `METRICS.json`, `README.md`) are
//! committed to `PRISM_TOPMODEL_HF_REPO` (default
//! `BaseIntelligence/prism-top-model`) via the Hub ndjson commit API.
//!
//! Token discipline mirrors GitHub: read from
//! `PRISM_TOPMODEL_HF_TOKEN_FILE` (e.g. `deploy/secrets/huggingface/token`),
//! never from env text. Absent/empty → graceful no-op.

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use tracing::info;

use crate::publish::{PublishError, TopModelRequest};

const DEFAULT_API_BASE: &str = "https://huggingface.co";
const DEFAULT_REPO: &str = "BaseIntelligence/prism-top-model";
const DEFAULT_REVISION: &str = "main";

/// HuggingFace Hub publisher (token never `Debug`/`Display`'d).
pub struct HfTopModelPublisher {
    http: reqwest::Client,
    api_base: String,
    repo: String,
    revision: String,
}

impl std::fmt::Debug for HfTopModelPublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTopModelPublisher")
            .field("api_base", &self.api_base)
            .field("repo", &self.repo)
            .field("revision", &self.revision)
            .field("http", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl HfTopModelPublisher {
    /// Configured Hub repo id (`org/name`).
    #[must_use]
    pub fn repo_id(&self) -> &str {
        &self.repo
    }

    /// `None` when the token file env is unset/empty.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let path = std::env::var("PRISM_TOPMODEL_HF_TOKEN_FILE").ok()?;
        let token = std::fs::read_to_string(path).ok()?.trim().to_owned();
        if token.len() < 8 {
            return None;
        }
        let repo = std::env::var("PRISM_TOPMODEL_HF_REPO").unwrap_or_else(|_| DEFAULT_REPO.into());
        let revision =
            std::env::var("PRISM_TOPMODEL_HF_REVISION").unwrap_or_else(|_| DEFAULT_REVISION.into());
        Self::with_config(token, DEFAULT_API_BASE, repo, revision).ok()
    }

    /// Explicit config (tests / wiremock).
    pub fn with_config(
        token: impl Into<String>,
        api_base: impl Into<String>,
        repo: impl Into<String>,
        revision: impl Into<String>,
    ) -> Result<Self, PublishError> {
        let token = token.into();
        if token.trim().is_empty() {
            return Err(PublishError::Transport("empty hf token".into()));
        }
        let mut headers = reqwest::header::HeaderMap::new();
        let mut hv = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        hv.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, hv);
        headers.insert(
            reqwest::header::USER_AGENT,
            reqwest::header::HeaderValue::from_static("base-prism-topmodel-hf/0.1"),
        );
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_mins(10))
            .build()
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            api_base: api_base.into().trim_end_matches('/').to_owned(),
            repo: repo.into(),
            revision: revision.into(),
        })
    }

    /// Ensure the model repo exists, then commit top-model sources.
    ///
    /// # Errors
    /// Transport / Hub API failures.
    pub async fn publish(&self, req: &TopModelRequest) -> Result<String, PublishError> {
        self.ensure_repo().await?;
        let arch = req.arch_id.as_deref().unwrap_or("arch-unregistered");
        let readme = format!(
            "# PRISM top model (HuggingFace)\n\n\
             Global-best bpb champion published by the Base master.\n\n\
             | field | value |\n|---|---|\n\
             | arch_id | `{arch}` |\n\
             | bpb | `{:.6}` |\n\
             | submission | `{}` |\n\
             | owner_hotkey | `{}…` |\n\n\
             Companion GitHub publish (when configured) lives under\n\
             `BaseIntelligence/prism` `top-model/`.\n",
            req.bpb,
            req.submission_id,
            req.owner_hotkey.chars().take(12).collect::<String>(),
        );
        let metrics = serde_json::to_string_pretty(&serde_json::json!({
            "submission_id": req.submission_id,
            "arch_id": req.arch_id,
            "owner_hotkey": req.owner_hotkey,
            "bpb": req.bpb,
            "n_params": req.metrics_json.as_ref().and_then(|m| m.get("n_params")),
            "tokens_seen": req.metrics_json.as_ref().and_then(|m| m.get("tokens_seen")),
            "wall_clock_seconds": req.metrics_json.as_ref().and_then(|m| m.get("wall_clock_seconds")),
            "battery": req.metrics_json.as_ref().and_then(|m| m.get("battery")),
            "eval_tier": req.metrics_json.as_ref().and_then(|m| m.get("eval_tier")),
            "flow": req.metrics_json.as_ref().and_then(|m| m.get("flow")),
        }))
        .unwrap_or_else(|_| "{}".into());
        let files: [(&str, &[u8]); 4] = [
            ("architecture.py", req.architecture_py.as_bytes()),
            ("training.py", req.training_py.as_bytes()),
            ("METRICS.json", metrics.as_bytes()),
            ("README.md", readme.as_bytes()),
        ];
        let oid = self
            .commit_files(&format!("top-model: {arch} bpb={:.4}", req.bpb), &files)
            .await?;
        info!(
            submission_id = %req.submission_id,
            repo = %self.repo,
            commit = %oid,
            "top model published to HuggingFace"
        );
        Ok(oid)
    }

    async fn ensure_repo(&self) -> Result<(), PublishError> {
        let (org, name) = split_repo(&self.repo)?;
        let url = format!("{}/api/repos/create", self.api_base);
        let body = serde_json::json!({
            "name": name,
            "organization": org,
            "private": false,
            "type": "model",
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        let status = resp.status();
        // 409 / already exists → ok; 200/201 → created.
        if status.is_success() || status.as_u16() == 409 {
            return Ok(());
        }
        let text = resp.text().await.unwrap_or_default();
        // Hub sometimes returns 400 "You already created this repository".
        if text.to_ascii_lowercase().contains("already") {
            return Ok(());
        }
        Err(PublishError::Api(format!(
            "hf create repo {status}: {text}"
        )))
    }

    async fn commit_files(
        &self,
        summary: &str,
        files: &[(&str, &[u8])],
    ) -> Result<String, PublishError> {
        let url = format!(
            "{}/api/models/{}/commit/{}",
            self.api_base, self.repo, self.revision
        );
        let mut ndjson = String::new();
        ndjson.push_str(
            &serde_json::json!({
                "key": "header",
                "value": {"summary": summary, "description": ""}
            })
            .to_string(),
        );
        ndjson.push('\n');
        for (path, bytes) in files {
            let line = serde_json::json!({
                "key": "file",
                "value": {
                    "content": B64.encode(bytes),
                    "path": path,
                    "encoding": "base64",
                }
            });
            ndjson.push_str(&line.to_string());
            ndjson.push('\n');
        }
        let resp = self
            .http
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
            .body(ndjson)
            .send()
            .await
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(PublishError::Api(format!("hf commit {status}: {body}")));
        }
        let v: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| PublishError::Api(e.to_string()))?;
        Ok(v.get("commitOid")
            .and_then(|x| x.as_str())
            .unwrap_or("ok")
            .to_owned())
    }
}

fn split_repo(repo: &str) -> Result<(&str, &str), PublishError> {
    let mut parts = repo.splitn(2, '/');
    let org = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| PublishError::Transport("hf repo missing org".into()))?;
    let name = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| PublishError::Transport("hf repo missing name".into()))?;
    Ok((org, name))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn req() -> TopModelRequest {
        TopModelRequest {
            submission_id: "subm-hf".into(),
            arch_id: Some("arch_hf".into()),
            owner_hotkey: "cd".repeat(32),
            bpb: 1.1,
            architecture_py: "def build_model(ctx):\n    pass\n".into(),
            training_py: "def train(model, ctx):\n    return {}\n".into(),
            metrics_json: Some(serde_json::json!({
                "n_params": 1,
                "battery": {"g1": {"status": "ok"}},
                "flow": "v3",
                "eval_tier": "public",
            })),
            checkpoint_path: None,
        }
    }

    #[tokio::test]
    async fn commits_sources_via_ndjson() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/repos/create"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/models/BaseIntelligence/prism-top-model/commit/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "commitOid": "hfoid123",
                "commitUrl": "https://huggingface.co/BaseIntelligence/prism-top-model/commit/hfoid123"
            })))
            .mount(&server)
            .await;
        let p = HfTopModelPublisher::with_config(
            "hf_tok_test",
            server.uri(),
            "BaseIntelligence/prism-top-model",
            "main",
        )
        .unwrap();
        let oid = p.publish(&req()).await.unwrap();
        assert_eq!(oid, "hfoid123");
    }

    #[test]
    fn from_env_graceful_without_file() {
        assert!(HfTopModelPublisher::from_env().is_none());
    }

    #[test]
    fn split_repo_ok() {
        assert_eq!(
            split_repo("BaseIntelligence/prism-top-model").unwrap(),
            ("BaseIntelligence", "prism-top-model")
        );
    }
}
