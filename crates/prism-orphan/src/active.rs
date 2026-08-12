//! In-process registry of submissions currently held by a worker.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Live worker holds keyed by submission id.
#[derive(Debug, Default)]
pub struct ActiveJobs {
    inner: Mutex<HashMap<String, Instant>>,
}

impl ActiveJobs {
    /// Empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `id` until the returned guard drops.
    #[must_use]
    pub fn enter(self: &std::sync::Arc<Self>, id: &str) -> ActiveGuard {
        if let Ok(mut g) = self.inner.lock() {
            g.insert(id.to_owned(), Instant::now());
        }
        ActiveGuard {
            jobs: std::sync::Arc::clone(self),
            id: id.to_owned(),
        }
    }

    /// Refresh heartbeat timestamp (log poller / stage transitions).
    pub fn touch(&self, id: &str) {
        if let Ok(mut g) = self.inner.lock() {
            if let Some(t) = g.get_mut(id) {
                *t = Instant::now();
            }
        }
    }

    /// True when a worker currently holds `id`.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.inner.lock().ok().is_some_and(|g| g.contains_key(id))
    }

    /// Seconds since last touch, when registered.
    #[must_use]
    pub fn age_secs(&self, id: &str) -> Option<u64> {
        let g = self.inner.lock().ok()?;
        let t = g.get(id)?;
        Some(t.elapsed().as_secs())
    }
}

/// RAII unregister for [`ActiveJobs::enter`].
#[derive(Debug)]
pub struct ActiveGuard {
    jobs: std::sync::Arc<ActiveJobs>,
    id: String,
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        if let Ok(mut g) = self.jobs.inner.lock() {
            g.remove(&self.id);
        }
    }
}
