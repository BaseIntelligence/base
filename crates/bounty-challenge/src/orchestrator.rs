//! Pipeline worker (compress → similarity). Leaf emit lives in [`crate::emit`].

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use bounty_store::{BountyStore, BugPatch, BugStatus, StageEvent};
use challenge_agentic::AgenticBackend;
use serde_json::json;
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::compress::compress_video;
use crate::similarity::{review_similarity, SimilarityKind};

/// Runtime knobs for the bounty worker.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Max concurrent compress / review jobs.
    pub max_concurrent: u32,
    /// Force deterministic sim backends (CI) — skip host ffmpeg.
    pub force_sim: bool,
    /// Artifacts root (`BOUNTY_ARTIFACTS_ROOT`).
    pub artifacts_root: PathBuf,
    /// Worker poll when queue empty.
    pub worker_poll: Duration,
    /// Similarity corpus lookback (ms).
    pub corpus_lookback_ms: u64,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 2,
            force_sim: false,
            artifacts_root: PathBuf::from("/var/lib/bounty/artifacts"),
            worker_poll: Duration::from_secs(2),
            corpus_lookback_ms: 24 * 60 * 60 * 1000,
        }
    }
}

/// Background pipeline orchestrator.
pub struct Orchestrator {
    /// Persistence.
    pub store: Arc<dyn BountyStore>,
    /// Config.
    pub config: OrchestratorConfig,
    /// Agentic backend.
    pub agentic: Arc<dyn AgenticBackend>,
}

impl std::fmt::Debug for Orchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Orchestrator")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Orchestrator {
    /// Construct.
    #[must_use]
    pub fn new(
        store: Arc<dyn BountyStore>,
        config: OrchestratorConfig,
        agentic: Arc<dyn AgenticBackend>,
    ) -> Self {
        Self {
            store,
            config,
            agentic,
        }
    }

    /// Process one claimed bug through compress → similarity.
    pub async fn process_bug(&self, id: &str) -> Result<(), String> {
        let bug = self
            .store
            .get_bug(id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "bug not found".to_owned())?;
        let raw_path = bug
            .video_path
            .clone()
            .ok_or_else(|| "missing raw video_path".to_owned())?;
        let raw = PathBuf::from(&raw_path);
        let out_dir = self.config.artifacts_root.join(id);
        let out = out_dir.join("video.mp4");

        let compressed = match compress_video(&raw, &out, self.config.force_sim).await {
            Ok(c) => c,
            Err(e) => {
                let _ = self
                    .store
                    .apply(
                        id,
                        &BugPatch {
                            status: Some(BugStatus::Failed),
                            reject_reason: Some(format!("compress: {e}")),
                            ..BugPatch::default()
                        },
                        Some(&StageEvent {
                            stage: "failed".into(),
                            detail: Some(json!({"reason": "compress", "error": e.to_string()})),
                            at_ms: 0,
                        }),
                    )
                    .await;
                return Err(format!("compress: {e}"));
            }
        };

        let _ = tokio::fs::remove_file(&raw).await;

        self.store
            .apply(
                id,
                &BugPatch {
                    status: Some(BugStatus::AgenticReview),
                    video_path: Some(compressed.video_path.display().to_string()),
                    video_sha256: Some(compressed.sha256.clone()),
                    video_bytes: Some(compressed.bytes),
                    ..BugPatch::default()
                },
                Some(&StageEvent {
                    stage: "agentic_review".into(),
                    detail: Some(json!({
                        "video_sha256": compressed.sha256,
                        "video_bytes": compressed.bytes,
                    })),
                    at_ms: 0,
                }),
            )
            .await
            .map_err(|e| e.to_string())?;

        let now = now_ms();
        let since = now.saturating_sub(self.config.corpus_lookback_ms);
        let corpus = self
            .store
            .list_similarity_corpus(since, &bug.miner_hotkey, bug.miner_coldkey.as_deref(), 64)
            .await
            .map_err(|e| e.to_string())?;

        let review_dir = out_dir.join("review");
        let verdict = match review_similarity(self.agentic.as_ref(), &bug, &corpus, &review_dir)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                let _ = self
                    .store
                    .apply(
                        id,
                        &BugPatch {
                            status: Some(BugStatus::Failed),
                            reject_reason: Some(format!("similarity: {e}")),
                            ..BugPatch::default()
                        },
                        Some(&StageEvent {
                            stage: "failed".into(),
                            detail: Some(json!({"reason": "similarity", "error": e.to_string()})),
                            at_ms: 0,
                        }),
                    )
                    .await;
                return Err(format!("similarity: {e}"));
            }
        };

        let verdict_json = serde_json::to_value(&verdict).unwrap_or(json!({}));
        match verdict.kind {
            SimilarityKind::Duplicate => {
                self.store
                    .apply(
                        id,
                        &BugPatch {
                            status: Some(BugStatus::Rejected),
                            agentic_verdict: Some(verdict_json),
                            nearest_id: verdict.nearest_id.clone(),
                            reject_reason: Some("duplicate_24h".into()),
                            ..BugPatch::default()
                        },
                        Some(&StageEvent {
                            stage: "rejected".into(),
                            detail: Some(json!({"reason": "duplicate_24h", "verdict": verdict})),
                            at_ms: 0,
                        }),
                    )
                    .await
                    .map_err(|e| e.to_string())?;
            }
            SimilarityKind::Novel => {
                self.store
                    .apply(
                        id,
                        &BugPatch {
                            status: Some(BugStatus::PendingAdmin),
                            agentic_verdict: Some(verdict_json),
                            nearest_id: verdict.nearest_id.clone(),
                            ..BugPatch::default()
                        },
                        Some(&StageEvent {
                            stage: "pending_admin".into(),
                            detail: Some(json!({"verdict": verdict})),
                            at_ms: 0,
                        }),
                    )
                    .await
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    /// Worker loop: claim uploaded → process.
    pub async fn run_worker(self: Arc<Self>, permit: Arc<Semaphore>) {
        loop {
            let Ok(guard) = permit.acquire().await else {
                sleep(self.config.worker_poll).await;
                continue;
            };
            match self.store.claim_next().await {
                Ok(Some(bug)) => {
                    let id = bug.id.clone();
                    info!(bug_id = %id, "claimed bug for processing");
                    if let Err(e) = self.process_bug(&id).await {
                        warn!(bug_id = %id, error = %e, "bug processing failed");
                    }
                }
                Ok(None) => {
                    drop(guard);
                    sleep(self.config.worker_poll).await;
                }
                Err(e) => {
                    warn!(error = %e, "claim_next failed");
                    drop(guard);
                    sleep(self.config.worker_poll).await;
                }
            }
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::similarity::BountySimAgent;
    use bounty_store::{BugRow, MemoryBountyStore};

    #[tokio::test]
    async fn process_novel_path_with_sim() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(MemoryBountyStore::new());
        let raw = dir.path().join("raw.mp4");
        tokio::fs::write(&raw, b"video-bytes").await.unwrap();
        let row = BugRow {
            id: "bug1".into(),
            miner_hotkey: "aa".repeat(32),
            miner_coldkey: None,
            app_id: "demo".into(),
            title: "unique title xyz".into(),
            description: "unique description abc".into(),
            steps: None,
            status: BugStatus::Uploaded,
            agentic_verdict: None,
            nearest_id: None,
            video_sha256: None,
            video_bytes: None,
            video_path: Some(raw.display().to_string()),
            reject_reason: None,
            epoch: 1,
            created_at_ms: 1,
            updated_at_ms: 1,
        };
        store.insert_bug(&row).await.unwrap();
        let claimed = store.claim_next().await.unwrap().unwrap();
        let orch = Orchestrator::new(
            store.clone(),
            OrchestratorConfig {
                force_sim: true,
                artifacts_root: dir.path().to_path_buf(),
                ..OrchestratorConfig::default()
            },
            Arc::new(BountySimAgent::new()),
        );
        orch.process_bug(&claimed.id).await.unwrap();
        let done = store.get_bug("bug1").await.unwrap().unwrap();
        assert_eq!(done.status, BugStatus::PendingAdmin);
        assert!(done.video_sha256.is_some());
    }
}
