//! Top-model weight publish: park metadata under `top-model/` and upload the
//! checkpoint as a GitHub Release asset (contents API is unsuitable for large
//! `.pt` blobs). Fail-closed: callers must not journal a publication when
//! these helpers return `Err`.

use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::publish::{PublishError, TopModelPublisher, TOPMODEL_REPO_PATH};

/// Release tag that always points at the current champion checkpoint.
pub const TOPMODEL_RELEASE_TAG: &str = "prism-top-model";

/// Max bytes uploaded via the contents API (base64 expands ~4/3).
const CONTENTS_API_MAX: usize = 40 * 1024 * 1024;

#[derive(Deserialize)]
struct ContentGet {
    sha: String,
}

#[derive(Deserialize)]
struct ContentPut {
    commit: CommitInfo,
}

#[derive(Deserialize)]
struct CommitInfo {
    sha: String,
}

#[derive(Deserialize)]
struct RelId {
    id: u64,
}

#[derive(Deserialize)]
struct CreatedRelease {
    upload_url: String,
    html_url: String,
}

#[derive(Debug, Clone)]
pub struct CheckpointMeta {
    pub sha256: String,
    pub bytes: usize,
    pub file_name: String,
    pub release_tag: Option<String>,
    pub release_asset_url: Option<String>,
    pub contents_path: Option<String>,
}

impl TopModelPublisher {
    /// Publish checkpoint bytes + `ARTIFACT.json` under `top-model/`.
    ///
    /// Small stubs (Sim / tiny models) may land via the contents API; larger
    /// weights go to the mutable `prism-top-model` release. Returns metadata
    /// for the journal / README.
    pub async fn publish_checkpoint(
        &self,
        checkpoint: &Path,
        arch: &str,
        bpb: f64,
    ) -> Result<CheckpointMeta, PublishError> {
        let bytes =
            std::fs::read(checkpoint).map_err(|e| PublishError::Transport(e.to_string()))?;
        if bytes.is_empty() {
            return Err(PublishError::Transport("empty checkpoint".into()));
        }
        let mut h = Sha256::new();
        h.update(&bytes);
        let sha = hex::encode(h.finalize());
        let file_name = checkpoint
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("checkpoint.pt")
            .to_owned();

        let mut meta = CheckpointMeta {
            sha256: sha.clone(),
            bytes: bytes.len(),
            file_name: file_name.clone(),
            release_tag: None,
            release_asset_url: None,
            contents_path: None,
        };

        if bytes.len() <= CONTENTS_API_MAX {
            let path = format!("{TOPMODEL_REPO_PATH}/{file_name}");
            self.put_file_public(&path, &bytes, arch, bpb).await?;
            meta.contents_path = Some(path);
        } else {
            let (tag, url) = self
                .upsert_release_asset(&file_name, &bytes, arch, bpb)
                .await?;
            meta.release_tag = Some(tag);
            meta.release_asset_url = Some(url);
        }

        let artifact = serde_json::json!({
            "sha256": sha,
            "bytes": bytes.len(),
            "file_name": file_name,
            "release_tag": meta.release_tag,
            "release_asset_url": meta.release_asset_url,
            "contents_path": meta.contents_path,
            "arch": arch,
            "bpb": bpb,
        });
        let body = serde_json::to_vec_pretty(&artifact)
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        self.put_file_public(
            &format!("{TOPMODEL_REPO_PATH}/ARTIFACT.json"),
            &body,
            arch,
            bpb,
        )
        .await?;
        Ok(meta)
    }

    /// Contents-API put for arbitrary bytes (text or small binary).
    pub(crate) async fn put_file_public(
        &self,
        path: &str,
        body: &[u8],
        arch: &str,
        bpb: f64,
    ) -> Result<String, PublishError> {
        use base64::Engine;
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
            Ok(_) => None,
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

    async fn upsert_release_asset(
        &self,
        file_name: &str,
        bytes: &[u8],
        arch: &str,
        bpb: f64,
    ) -> Result<(String, String), PublishError> {
        // Delete prior release with the same tag (mutable champion pointer).
        let tag_url = format!(
            "{}/repos/{}/releases/tags/{TOPMODEL_RELEASE_TAG}",
            self.api_base, self.repo
        );
        if let Ok(resp) = self.http.get(&tag_url).send().await {
            if resp.status().is_success() {
                if let Ok(rel) = resp.json::<RelId>().await {
                    let _ = self
                        .http
                        .delete(format!(
                            "{}/repos/{}/releases/{}",
                            self.api_base, self.repo, rel.id
                        ))
                        .send()
                        .await;
                }
            }
        }
        let create_url = format!("{}/repos/{}/releases", self.api_base, self.repo);
        let body = serde_json::json!({
            "tag_name": TOPMODEL_RELEASE_TAG,
            "name": format!("PRISM top model ({arch})"),
            "body": format!("Global-best checkpoint bpb={bpb:.6} arch={arch}"),
            "target_commitish": self.branch,
        });
        let resp = self
            .http
            .post(&create_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(PublishError::Api(format!(
                "create release {status}: {}",
                &text.chars().take(200).collect::<String>()
            )));
        }
        let created: CreatedRelease = resp
            .json()
            .await
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        // upload_url looks like .../assets{?name,label}
        let upload_base = created
            .upload_url
            .split('{')
            .next()
            .unwrap_or(&created.upload_url);
        let upload = format!("{upload_base}?name={file_name}");
        // GitHub upload host differs from api.github.com — use the URL as-is
        // with the same auth headers (reqwest client already has them).
        let up = self
            .http
            .post(&upload)
            .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
            .body(bytes.to_vec())
            .send()
            .await
            .map_err(|e| PublishError::Transport(e.to_string()))?;
        if !up.status().is_success() {
            let status = up.status();
            let text = up.text().await.unwrap_or_default();
            return Err(PublishError::Api(format!(
                "upload asset {status}: {}",
                &text.chars().take(200).collect::<String>()
            )));
        }
        Ok((TOPMODEL_RELEASE_TAG.to_owned(), created.html_url))
    }
}
