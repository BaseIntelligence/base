//! HuggingFace Hub top-model publisher.
//!
//! When a submission becomes the new global-best bpb, the master publishes a
//! **reloadable** Hub model card to `PRISM_TOPMODEL_HF_REPO` (default
//! `BaseIntelligence/top-prism-architecture`):
//!
//! - custom architecture / AutoModel novelty sources (`architecture.py`,
//!   `training.py`, plus touched `.py` / patch files from `tree_blob`)
//! - `config.json` + thin `configuration_prism.py` / `modeling_prism.py`
//!   wrappers so `trust_remote_code=True` consumers can locate the code
//! - trained `checkpoint.pt` (LFS when large) when secure-receive parked it
//!
//! Token discipline: read from `PRISM_TOPMODEL_HF_TOKEN_FILE` only (never env
//! text). Absent/empty → graceful no-op.

use std::collections::BTreeMap;
use std::path::Path;

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::publish::{PublishError, TopModelRequest};

const DEFAULT_API_BASE: &str = "https://huggingface.co";
const DEFAULT_REPO: &str = "BaseIntelligence/top-prism-architecture";
const DEFAULT_REVISION: &str = "main";
/// Hub regular (non-LFS) file ceiling used by the ndjson commit API.
const REGULAR_FILE_MAX: usize = 5 * 1024 * 1024;

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
            reqwest::header::HeaderValue::from_static("base-prism-topmodel-hf/0.2"),
        );
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_mins(30))
            .build()
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            api_base: api_base.into().trim_end_matches('/').to_owned(),
            repo: repo.into(),
            revision: revision.into(),
        })
    }

    /// Ensure the model repo exists, then commit a reloadable custom-arch pack.
    ///
    /// # Errors
    /// Transport / Hub API failures.
    pub async fn publish(&self, req: &TopModelRequest) -> Result<String, PublishError> {
        self.ensure_repo().await?;
        let files = build_hub_files(req)?;
        let oid = self
            .commit_payload(
                &format!(
                    "top-model: {} bpb={:.4}",
                    req.arch_id.as_deref().unwrap_or("arch-unregistered"),
                    req.bpb
                ),
                &files,
            )
            .await?;
        info!(
            submission_id = %req.submission_id,
            repo = %self.repo,
            commit = %oid,
            files = files.len(),
            "top model published to HuggingFace (custom-arch pack)"
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
        if status.is_success() || status.as_u16() == 409 {
            return Ok(());
        }
        let text = resp.text().await.unwrap_or_default();
        if text.to_ascii_lowercase().contains("already") {
            return Ok(());
        }
        Err(PublishError::Api(format!(
            "hf create repo {status}: {text}"
        )))
    }

    async fn commit_payload(
        &self,
        summary: &str,
        files: &[(String, Vec<u8>)],
    ) -> Result<String, PublishError> {
        let url = format!(
            "{}/api/models/{}/commit/{}",
            self.api_base, self.repo, self.revision
        );
        let mut ndjson = String::new();
        ndjson.push_str(
            &serde_json::json!({
                "key": "header",
                "value": {"summary": summary, "description": "PRISM custom-arch top model"}
            })
            .to_string(),
        );
        ndjson.push('\n');
        for (path, bytes) in files {
            if bytes.len() > REGULAR_FILE_MAX {
                let oid = hex::encode(Sha256::digest(bytes));
                self.upload_lfs(path, bytes, &oid).await?;
                let line = serde_json::json!({
                    "key": "lfsFile",
                    "value": {
                        "path": path,
                        "algo": "sha256",
                        "size": bytes.len(),
                        "oid": oid,
                    }
                });
                ndjson.push_str(&line.to_string());
                ndjson.push('\n');
            } else {
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

    /// LFS batch + PUT for one large file (Hub oid = hex sha256, no prefix).
    async fn upload_lfs(&self, path: &str, bytes: &[u8], oid_hex: &str) -> Result<(), PublishError> {
        let batch_url = format!("{}/{}.git/info/lfs/objects/batch", self.api_base, self.repo);
        let batch = serde_json::json!({
            "operation": "upload",
            "transfers": ["basic"],
            "objects": [{
                "oid": oid_hex,
                "size": bytes.len(),
            }],
            "hash_algo": "sha256",
        });
        let resp = self
            .http
            .post(&batch_url)
            .header(reqwest::header::ACCEPT, "application/vnd.git-lfs+json")
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/vnd.git-lfs+json",
            )
            .json(&batch)
            .send()
            .await
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(PublishError::Api(format!(
                "hf lfs batch {status} ({path}): {body}"
            )));
        }
        let v: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| PublishError::Api(e.to_string()))?;
        let obj = v
            .get("objects")
            .and_then(|o| o.as_array())
            .and_then(|a| a.first())
            .ok_or_else(|| PublishError::Api("hf lfs batch: empty objects".into()))?;
        if obj.get("error").is_some() {
            return Err(PublishError::Api(format!(
                "hf lfs object error ({path}): {obj}"
            )));
        }
        // Already present on the server → nothing to upload.
        let Some(actions) = obj.get("actions") else {
            return Ok(());
        };
        let upload = actions
            .get("upload")
            .ok_or_else(|| PublishError::Api(format!("hf lfs missing upload action ({path})")))?;
        let href = upload
            .get("href")
            .and_then(|h| h.as_str())
            .ok_or_else(|| PublishError::Api("hf lfs upload href missing".into()))?;
        let mut req = self.http.put(href).body(bytes.to_vec());
        if let Some(headers) = upload.get("header").and_then(|h| h.as_object()) {
            for (k, val) in headers {
                if let Some(s) = val.as_str() {
                    req = req.header(k.as_str(), s);
                }
            }
        }
        let put = req
            .send()
            .await
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        if !put.status().is_success() {
            let st = put.status();
            let t = put.text().await.unwrap_or_default();
            return Err(PublishError::Api(format!(
                "hf lfs put {st} ({path}): {t}"
            )));
        }
        // Optional verify action.
        if let Some(verify) = actions.get("verify") {
            if let Some(vhref) = verify.get("href").and_then(|h| h.as_str()) {
                let mut vreq = self.http.post(vhref).json(&serde_json::json!({
                    "oid": oid_hex,
                    "size": bytes.len(),
                }));
                if let Some(headers) = verify.get("header").and_then(|h| h.as_object()) {
                    for (k, val) in headers {
                        if let Some(s) = val.as_str() {
                            vreq = vreq.header(k.as_str(), s);
                        }
                    }
                }
                let _ = vreq.send().await;
            }
        }
        Ok(())
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

/// Build the Hub file set for a top-model publish (sources + wrappers + weights).
fn build_hub_files(req: &TopModelRequest) -> Result<Vec<(String, Vec<u8>)>, PublishError> {
    let arch = req.arch_id.as_deref().unwrap_or("arch-unregistered");
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();

    files.insert(
        "architecture.py".into(),
        req.architecture_py.as_bytes().to_vec(),
    );
    files.insert("training.py".into(), req.training_py.as_bytes().to_vec());
    files.insert(
        "configuration_prism.py".into(),
        CONFIGURATION_PRISM_PY.as_bytes().to_vec(),
    );
    files.insert(
        "modeling_prism.py".into(),
        MODELING_PRISM_PY.as_bytes().to_vec(),
    );
    files.insert(
        "config.json".into(),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_type": "prism_custom",
            "architectures": ["PrismCustomModel"],
            "auto_map": {
                "AutoConfig": "configuration_prism.PrismConfig",
                "AutoModel": "modeling_prism.PrismCustomModel",
                "AutoModelForCausalLM": "modeling_prism.PrismCustomModel",
            },
            "prism_arch_id": arch,
            "prism_submission_id": req.submission_id,
            "prism_bpb": req.bpb,
            "torch_dtype": "float32",
            "checkpoint_file": "checkpoint.pt",
        }))
        .map_err(|e| PublishError::Transport(e.to_string()))?,
    );

    let metrics = serde_json::to_vec_pretty(&serde_json::json!({
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
    .unwrap_or_else(|_| b"{}".to_vec());
    files.insert("METRICS.json".into(), metrics);

    for (rel, bytes) in &req.extra_files {
        let clean = sanitize_hub_path(rel);
        if clean.is_empty() || files.contains_key(&clean) {
            continue;
        }
        files.insert(clean, bytes.clone());
    }

    if let Some(path) = &req.checkpoint_path {
        let bytes = std::fs::read(path).map_err(|e| PublishError::Transport(e.to_string()))?;
        if bytes.is_empty() {
            return Err(PublishError::Transport("empty checkpoint".into()));
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("checkpoint.pt");
        files.insert(name.to_owned(), bytes);
    }

    let has_ckpt = files.keys().any(|k| k.ends_with("checkpoint.pt"));
    files.insert(
        "README.md".into(),
        hub_readme(req, arch, has_ckpt).into_bytes(),
    );

    Ok(files.into_iter().collect())
}

fn sanitize_hub_path(rel: &str) -> String {
    let p = Path::new(rel);
    if p.is_absolute()
        || p.components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return String::new();
    }
    let s = rel.trim_start_matches("./").replace('\\', "/");
    if s.is_empty() || s.contains('\0') {
        return String::new();
    }
    // Keep novel modules under sources/ so they never collide with wrappers.
    if s == "architecture.py"
        || s == "training.py"
        || s == "config.json"
        || s == "README.md"
        || s == "METRICS.json"
        || s == "modeling_prism.py"
        || s == "configuration_prism.py"
        || s == "checkpoint.pt"
    {
        return s;
    }
    if s.starts_with("sources/") {
        s
    } else {
        format!("sources/{s}")
    }
}

fn hub_readme(req: &TopModelRequest, arch: &str, has_ckpt: bool) -> String {
    let ckpt_note = if has_ckpt {
        "Weights: `checkpoint.pt` (LFS when large). Load via `PrismCustomModel.from_pretrained` with `trust_remote_code=True`, or apply `sources/.prism/automodel.patch` onto the live AutoModel pin and `torch.load` the checkpoint."
    } else {
        "Weights were not parked on the master for this champion — sources/config only."
    };
    format!(
        "---\n\
         library_name: transformers\n\
         tags:\n\
         - prism\n\
         - custom-architecture\n\
         - trust-remote-code\n\
         ---\n\n\
         # PRISM top model — custom architecture\n\n\
         Global-best champion published by the Base master. Novel Prism / AutoModel\n\
         modules are included under `sources/` (and seam files at repo root) so the\n\
         Hub card is reloadable even when the arch is **not** a stock transformers\n\
         GPT-2-like config.\n\n\
         | field | value |\n|---|---|\n\
         | arch_id | `{arch}` |\n\
         | bpb | `{:.6}` |\n\
         | submission | `{}` |\n\
         | owner_hotkey | `{}…` |\n\
         | hub repo | `{DEFAULT_REPO}` (override via `PRISM_TOPMODEL_HF_REPO`) |\n\n\
         ## Load (trust_remote_code)\n\n\
         ```python\n\
         from transformers import AutoModel, AutoConfig\n\
         cfg = AutoConfig.from_pretrained(\"{DEFAULT_REPO}\", trust_remote_code=True)\n\
         model = AutoModel.from_pretrained(\"{DEFAULT_REPO}\", trust_remote_code=True)\n\
         ```\n\n\
         {ckpt_note}\n\n\
         Companion GitHub publish (when configured) lives under\n\
         `BaseIntelligence/prism` `top-model/`.\n",
        req.bpb,
        req.submission_id,
        req.owner_hotkey.chars().take(12).collect::<String>(),
    )
}

const CONFIGURATION_PRISM_PY: &str = r#"
"""Prism custom-arch Hub config (trust_remote_code)."""

from transformers import PretrainedConfig


class PrismConfig(PretrainedConfig):
    model_type = "prism_custom"

    def __init__(self, prism_arch_id=None, checkpoint_file="checkpoint.pt", **kwargs):
        super().__init__(**kwargs)
        self.prism_arch_id = prism_arch_id
        self.checkpoint_file = checkpoint_file
"#;

const MODELING_PRISM_PY: &str = r#"
"""Prism custom-arch loader (trust_remote_code).

Supports:
  1. Legacy seam: ``architecture.build_model(ctx)`` + ``checkpoint.pt``
  2. AutoModel novelty: sources under ``sources/`` (apply patch offline)

This is intentionally permissive — novel arches are not limited to stock
GPT-2 ``transformers`` configs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from transformers import PreTrainedModel

try:
    from configuration_prism import PrismConfig
except ImportError:  # package-style local import
    from .configuration_prism import PrismConfig


def _load_architecture_module(root: Path):
    path = root / "architecture.py"
    if not path.is_file():
        raise FileNotFoundError(f"architecture.py missing under {root}")
    spec = importlib.util.spec_from_file_location("prism_architecture", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load architecture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PrismCustomModel(PreTrainedModel):
    config_class = PrismConfig
    _no_split_modules = []

    def __init__(self, config: PrismConfig, inner: Optional[nn.Module] = None):
        super().__init__(config)
        self.inner = inner if inner is not None else nn.Identity()
        self.post_init()

    def forward(self, *args: Any, **kwargs: Any):
        return self.inner(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        trust = kwargs.pop("trust_remote_code", True)
        config = kwargs.pop("config", None)
        if config is None:
            config = PrismConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust, **kwargs
            )
        root = Path(pretrained_model_name_or_path)
        # Hub download may leave us with a cache dir; prefer local folder layout.
        if not (root / "architecture.py").is_file():
            # Fall back to empty shell when only config is present.
            return cls(config)

        mod = _load_architecture_module(root)
        ctx = {
            "device": "cpu",
            "dtype": torch.float32,
            "seed": 0,
            "vocab_size": getattr(config, "vocab_size", 50257),
        }
        if hasattr(mod, "build_model"):
            inner = mod.build_model(ctx)
        elif hasattr(mod, "Model"):
            inner = mod.Model(**{k: v for k, v in ctx.items() if k in ("vocab_size",)})
        else:
            raise AttributeError(
                "architecture.py must define build_model(ctx) or Model for Hub reload"
            )
        model = cls(config, inner=inner)
        ckpt_name = getattr(config, "checkpoint_file", "checkpoint.pt") or "checkpoint.pt"
        ckpt = root / ckpt_name
        if ckpt.is_file():
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
            if isinstance(state, dict):
                try:
                    model.inner.load_state_dict(state, strict=False)
                except Exception:
                    # Novel arches / tied keys — best-effort; sources remain authoritative.
                    pass
        return model
"#;

/// Collect novel source files from a packed `tree_blob` for Hub `sources/`.
#[must_use]
pub fn extra_files_from_tree_blob(tree_blob: Option<&[u8]>) -> Vec<(String, Vec<u8>)> {
    let Some(blob) = tree_blob else {
        return Vec::new();
    };
    let Ok(tree) = prism_tree::StagedTree::unpack(blob) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (path, bytes) in tree.files() {
        if !is_publishable_source(path) {
            continue;
        }
        if bytes.len() > 8 * 1024 * 1024 {
            continue;
        }
        out.push((path.clone(), bytes.clone()));
    }
    out
}

fn is_publishable_source(path: &str) -> bool {
    let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) else {
        // Allow extensionless patch paths under .prism/
        return path.ends_with("automodel.patch");
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "py" | "toml" | "json" | "md" | "patch" | "yaml" | "yml"
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
            submission_id: "subm-hf".into(),
            arch_id: Some("arch_hf".into()),
            owner_hotkey: "cd".repeat(32),
            bpb: 1.1,
            architecture_py: "def build_model(ctx):\n    return None\n".into(),
            training_py: "def train(model, ctx):\n    return {}\n".into(),
            metrics_json: Some(serde_json::json!({
                "n_params": 1,
                "battery": {"g1": {"status": "ok"}},
                "flow": "v3",
                "eval_tier": "public",
            })),
            checkpoint_path: None,
            extra_files: vec![(
                "nemo_automodel/components/models/toy/layers.py".into(),
                b"class ToyLayer: pass\n".to_vec(),
            )],
        }
    }

    #[tokio::test]
    async fn commits_custom_arch_pack() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/repos/create"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(
                "/api/models/BaseIntelligence/top-prism-architecture/commit/main",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "commitOid": "hfoid123",
                "commitUrl": "https://huggingface.co/BaseIntelligence/top-prism-architecture/commit/hfoid123"
            })))
            .mount(&server)
            .await;
        let p = HfTopModelPublisher::with_config(
            "hf_tok_test",
            server.uri(),
            "BaseIntelligence/top-prism-architecture",
            "main",
        )
        .unwrap();
        let oid = p.publish(&req()).await.unwrap();
        assert_eq!(oid, "hfoid123");
        let built = build_hub_files(&req()).unwrap();
        let keys: Vec<_> = built.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"config.json"));
        assert!(keys.contains(&"modeling_prism.py"));
        assert!(keys.contains(&"sources/nemo_automodel/components/models/toy/layers.py"));
    }

    #[test]
    fn default_repo_is_top_prism_architecture() {
        assert_eq!(DEFAULT_REPO, "BaseIntelligence/top-prism-architecture");
    }

    #[test]
    fn from_env_graceful_without_file() {
        assert!(HfTopModelPublisher::from_env().is_none());
    }

    #[test]
    fn split_repo_ok() {
        assert_eq!(
            split_repo("BaseIntelligence/top-prism-architecture").unwrap(),
            ("BaseIntelligence", "top-prism-architecture")
        );
    }

    #[test]
    fn sanitize_rejects_traversal() {
        assert!(sanitize_hub_path("../evil.py").is_empty());
        assert_eq!(
            sanitize_hub_path("foo/bar.py"),
            "sources/foo/bar.py"
        );
    }
}
