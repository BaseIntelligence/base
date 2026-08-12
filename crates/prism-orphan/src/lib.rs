//! Mid-flight orphan detection for Prism after control-plane restarts.
//!
//! Workers register in [`ActiveJobs`] for the whole `run_row` hold. On boot
//! (and periodically) [`reconcile_once`] fails provisioning/running rows whose
//! worker is gone, best-effort terminates the pod when the BYOK vault still
//! has a key, and requeues post-measure review stages so they can resume.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]

mod active;
mod logs;
mod reconcile;

pub use active::{ActiveGuard, ActiveJobs};
pub use logs::{logs_json, LogBuffer, LogLine};
pub use reconcile::{
    midflight_rows, reconcile_once, ReconcileReport, CLASS_ORPHAN, REASON_CONTROL_PLANE_RESTART,
    REASON_HARNESS_DETACHED,
};

/// Default detach grace for periodic reconcile (seconds). Boot uses zero.
pub const DEFAULT_ORPHAN_GRACE_SECS: u64 = 90;
/// Log / heartbeat poll while a pod is measuring.
pub const DEFAULT_LOG_POLL_SECS: u64 = 20;

/// Spawn [`watch_pod_logs`]; returns the stop sender (send `true` to end).
pub fn spawn_log_watch(
    logs: std::sync::Arc<LogBuffer>,
    store: std::sync::Arc<dyn prism_store::PrismStore>,
    backend: std::sync::Arc<dyn prism_lium::EvalJobBackend>,
    active: std::sync::Arc<ActiveJobs>,
    submission_id: String,
    pod_id: String,
    poll_secs: u64,
) -> tokio::sync::watch::Sender<bool> {
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        watch_pod_logs(
            logs,
            store,
            backend,
            active,
            submission_id,
            pod_id,
            stop_rx,
            poll_secs,
        )
        .await;
    });
    stop_tx
}

/// Poll harness logs + heartbeat stage events until `cancel` fires.
pub async fn watch_pod_logs(
    logs: std::sync::Arc<LogBuffer>,
    store: std::sync::Arc<dyn prism_store::PrismStore>,
    backend: std::sync::Arc<dyn prism_lium::EvalJobBackend>,
    active: std::sync::Arc<ActiveJobs>,
    submission_id: String,
    pod_id: String,
    mut cancel: tokio::sync::watch::Receiver<bool>,
    poll_secs: u64,
) {
    let period = std::time::Duration::from_secs(poll_secs.max(5));
    loop {
        if *cancel.borrow() {
            break;
        }
        active.touch(&submission_id);
        match backend.harvest_logs(&pod_id).await {
            Ok(text) if !text.trim().is_empty() => {
                logs.replace_tail(&submission_id, &text);
            }
            _ => {}
        }
        let _ = store
            .apply(
                &submission_id,
                &prism_store::StatePatch::default(),
                Some(&prism_store::StageEvent {
                    stage: prism_store::Stage::Running,
                    detail: Some(serde_json::json!({
                        "heartbeat": true,
                        "pod_id": pod_id,
                    })),
                    at_ms: 0,
                }),
            )
            .await;
        tokio::select! {
            _ = cancel.changed() => {
                if *cancel.borrow() { break; }
            }
            () = tokio::time::sleep(period) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_trait::async_trait;
    use prism_lium::{
        EvalJobBackend, EvalReceipt, Instance, InstanceSpec, LiumError, Offer, RemoteExecResult,
    };
    use prism_store::{FinalScore, MemoryPrismStore, PrismStore, Stage, SubmissionState};

    struct NoopBackend;
    #[async_trait]
    impl EvalJobBackend for NoopBackend {
        async fn list_offers(
            &self,
            _max_price_per_hour: Option<f64>,
        ) -> Result<Vec<Offer>, LiumError> {
            Ok(vec![])
        }
        async fn provision(&self, _spec: &InstanceSpec) -> Result<Instance, LiumError> {
            Err(LiumError::Exec("noop".into()))
        }
        async fn terminate(&self, _instance_id: &str) -> Result<(), LiumError> {
            Ok(())
        }
        async fn verify_terminated(&self, _instance_id: &str) -> Result<bool, LiumError> {
            Ok(true)
        }
        async fn exec_eval(
            &self,
            _instance_id: &str,
            _architecture_py: &str,
            _training_py: &str,
            _tree_blob: Option<&[u8]>,
        ) -> Result<RemoteExecResult, LiumError> {
            Err(LiumError::Exec("noop".into()))
        }
    }

    fn row(id: &str, status: Stage, pod: Option<&str>) -> SubmissionState {
        SubmissionState {
            id: id.into(),
            miner_hotkey: "a".repeat(64),
            miner_coldkey: None,
            epoch: 1,
            netuid: 1,
            status,
            architecture_py: "a".into(),
            training_py: "t".into(),
            tree_blob: None,
            label: None,
            pod_id: pod.map(str::to_owned),
            pod_provider: pod.map(|_| "lium".into()),
            receipt: None,
            metrics_json: None,
            bpb: None,
            arch_id: None,
            review: None,
            similarity: None,
            final_score: None,
            retry_count: 0,
            error_detail: None,
            created_at_ms: 1,
            updated_at_ms: 1,
        }
    }

    #[tokio::test]
    async fn boot_reconcile_marks_running_orphan_failed() {
        let store: Arc<dyn PrismStore> = Arc::new(MemoryPrismStore::default());
        let r = row("deadbeefdeadbeef", Stage::Running, Some("pod-1"));
        store.insert_queued(&r).await.unwrap();
        store
            .apply(
                &r.id,
                &prism_store::StatePatch {
                    status: Some(Stage::Running),
                    pod_id: Some("pod-1".into()),
                    pod_provider: Some("lium".into()),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        let active = Arc::new(ActiveJobs::new());
        let be: Arc<dyn EvalJobBackend> = Arc::new(NoopBackend);
        let report = reconcile_once(store.as_ref(), &active, None, be, 0, true, None)
            .await
            .unwrap();
        assert_eq!(report.failed, 1);
        let got = store.get(&r.id).await.unwrap().unwrap();
        assert_eq!(got.status, Stage::Failed);
        let err = got.error_detail.unwrap();
        assert!(err.contains(REASON_CONTROL_PLANE_RESTART));
        assert!(
            err.to_ascii_lowercase().contains("lium pod")
                && (err.contains("stop") || err.contains("Stop") || err.contains("resubmit"))
        );
        assert!(matches!(got.final_score, Some(FinalScore::NoScore(_))));
    }

    #[tokio::test]
    async fn active_worker_not_orphaned() {
        let store: Arc<dyn PrismStore> = Arc::new(MemoryPrismStore::default());
        let r = row("aabbccddeeff0011", Stage::Running, Some("pod-2"));
        store.insert_queued(&r).await.unwrap();
        store
            .apply(
                &r.id,
                &prism_store::StatePatch {
                    status: Some(Stage::Running),
                    pod_id: Some("pod-2".into()),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
        let active = Arc::new(ActiveJobs::new());
        let _g = active.enter(&r.id);
        let be: Arc<dyn EvalJobBackend> = Arc::new(NoopBackend);
        let report = reconcile_once(store.as_ref(), &active, None, be, 90, false, None)
            .await
            .unwrap();
        assert_eq!(report.failed, 0);
        assert_eq!(
            store.get(&r.id).await.unwrap().unwrap().status,
            Stage::Running
        );
    }

    #[tokio::test]
    async fn post_measure_orphan_requeues() {
        let store: Arc<dyn PrismStore> = Arc::new(MemoryPrismStore::default());
        let mut r = row("1122334455667788", Stage::LlmReview, Some("pod-3"));
        let metrics = RemoteExecResult {
            bpb: 1.2,
            tokens_seen: 1,
            wall_clock_seconds: 1.0,
            gpu_type: None,
            notes: String::new(),
            n_params: None,
            val_rows: None,
            telemetry: None,
            metrics_version: None,
            tokens_seen_source: None,
            probe_curve: None,
            pod_manifest: None,
            netns: None,
            harness_files_sha256: None,
            eval_tier: None,
            extra: std::collections::BTreeMap::new(),
        };
        let receipt = EvalReceipt {
            provider: "lium".into(),
            pod_id: "pod-3".into(),
            image_digest: String::new(),
            submission_hash: "h".into(),
            metrics_hash: "m".into(),
            termination_verified: true,
        };
        r.metrics_json = serde_json::to_value(&metrics).ok();
        r.receipt = Some(receipt.clone());
        r.bpb = Some(1.2);
        store.insert_queued(&r).await.unwrap();
        store
            .apply(
                &r.id,
                &prism_store::StatePatch {
                    status: Some(Stage::LlmReview),
                    metrics_json: r.metrics_json.clone(),
                    receipt: Some(receipt),
                    bpb: Some(1.2),
                    pod_id: Some("pod-3".into()),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
        let active = Arc::new(ActiveJobs::new());
        let be: Arc<dyn EvalJobBackend> = Arc::new(NoopBackend);
        let report = reconcile_once(store.as_ref(), &active, None, be, 0, true, None)
            .await
            .unwrap();
        assert_eq!(report.requeued, 1);
        assert_eq!(
            store.get(&r.id).await.unwrap().unwrap().status,
            Stage::Queued
        );
    }

    #[test]
    fn log_buffer_since_cursor() {
        let b = LogBuffer::new();
        b.append("s1", "line-a");
        b.append("s1", "line-b");
        let (lines, next) = b.since("s1", 0);
        assert_eq!(lines.len(), 2);
        assert!(next > 0);
        let (more, _) = b.since("s1", next);
        assert!(more.is_empty());
    }
}
