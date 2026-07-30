//! In-memory task store + stub executor for todo 17.

use std::collections::HashMap;
use std::sync::Arc;

use agent_dispatch::{
    patch_sha256, TaskDescriptorV1, TaskResultV1, TaskStatusV1, DISPATCH_PROTOCOL,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::api::CapacityResponse;

/// Runner process configuration (env-driven at the binary boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunnerConfig {
    /// Advertised max concurrent tasks (todo 19 will clamp/enforce).
    pub max_concurrency: u32,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 1,
        }
    }
}

/// Lifecycle status exposed on `GET /v1/task/{id}` (broader than wire terminal status).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskLifecycle {
    /// Accepted, not yet started.
    Pending,
    /// Stub/real executor holds a slot.
    Running,
    /// Finished successfully (result present).
    Completed,
    /// Finished with failure (result present).
    Failed,
}

#[derive(Debug, Clone)]
struct TaskRecord {
    lifecycle: TaskLifecycle,
    result: Option<TaskResultV1>,
}

/// Shared runner state (cloneable axum `State`).
#[derive(Clone)]
pub struct RunnerState {
    config: RunnerConfig,
    inner: Arc<Mutex<StoreInner>>,
}

struct StoreInner {
    tasks: HashMap<String, TaskRecord>,
    /// Tasks currently in [`TaskLifecycle::Running`].
    running: u32,
}

impl RunnerState {
    /// New empty store with the given capacity advertisement.
    #[must_use]
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            config,
            inner: Arc::new(Mutex::new(StoreInner {
                tasks: HashMap::new(),
                running: 0,
            })),
        }
    }

    /// Snapshot capacity for `GET /v1/capacity`.
    #[must_use]
    pub fn capacity(&self) -> CapacityResponse {
        // Try lock; if contended report last-known via blocking path in async handlers.
        // Sync helper used from tests — prefer try_lock then 0.
        let load = self
            .inner
            .try_lock()
            .map(|g| g.running)
            .unwrap_or(0);
        CapacityResponse {
            max_concurrency: self.config.max_concurrency,
            current_load: load,
        }
    }

    /// Async capacity (accurate under load).
    pub async fn capacity_async(&self) -> CapacityResponse {
        let g = self.inner.lock().await;
        CapacityResponse {
            max_concurrency: self.config.max_concurrency,
            current_load: g.running,
        }
    }

    /// Enqueue descriptor, spawn stub executor, return assigned task id.
    pub async fn accept_task(&self, descriptor: TaskDescriptorV1) -> String {
        let task_id = Uuid::new_v4().to_string();
        {
            let mut g = self.inner.lock().await;
            g.tasks.insert(
                task_id.clone(),
                TaskRecord {
                    lifecycle: TaskLifecycle::Pending,
                    result: None,
}
            );
        }
        let state = self.clone();
        let id = task_id.clone();
        tokio::spawn(async move {
            state.run_stub(id, descriptor).await;
        });
        task_id
    }

    /// Lookup task view fields.
    pub async fn get_task(
        &self,
        id: &str,
    ) -> Option<(TaskLifecycle, Option<TaskResultV1>)> {
        let g = self.inner.lock().await;
        g.tasks
            .get(id)
            .map(|r| (r.lifecycle, r.result.clone()))
    }

    async fn run_stub(&self, task_id: String, descriptor: TaskDescriptorV1) {
        {
            let mut g = self.inner.lock().await;
            if let Some(rec) = g.tasks.get_mut(&task_id) {
                rec.lifecycle = TaskLifecycle::Running;
                g.running = g.running.saturating_add(1);
            }
        }

        // Yield so polls can observe Running / load if desired.
        tokio::task::yield_now().await;

        let patch = stub_model_patch(&descriptor.pack_id);
        let digest = patch_sha256(patch.as_bytes());
        let result = TaskResultV1 {
            protocol: DISPATCH_PROTOCOL.into(),
            challenge_id: descriptor.challenge_id.clone(),
            scoring_version: descriptor.scoring_version,
            epoch: descriptor.epoch,
            miner_hotkey_hex: descriptor.miner_hotkey_hex.clone(),
            pack_id: descriptor.pack_id.clone(),
            status: TaskStatusV1::Completed,
            model_patch: Some(patch),
            patch_sha256_hex: hex::encode(digest),
            // Stub receipt sig — real CVM key + sign_work_receipt in later todos.
            receipt_sig_hex: "00".repeat(64),
        };

        let mut g = self.inner.lock().await;
        let was_running = g
            .tasks
            .get(&task_id)
            .is_some_and(|r| r.lifecycle == TaskLifecycle::Running);
        if was_running {
            g.running = g.running.saturating_sub(1);
        }
        if let Some(rec) = g.tasks.get_mut(&task_id) {
            rec.lifecycle = TaskLifecycle::Completed;
            rec.result = Some(result);
        }
    }
}

fn stub_model_patch(pack_id: &str) -> String {
    format!(
        "diff --git a/README.md b/README.md\n\
         --- a/README.md\n\
         +++ b/README.md\n\
         @@ -1 +1,2 @@\n\
          # pack\n\
         +stub patch for {pack_id}\n"
    )
}
