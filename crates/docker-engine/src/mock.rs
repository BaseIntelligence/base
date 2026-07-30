//! In-memory Docker mock for unit tests (records calls).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::allowlist::Allowlist;
use crate::api::DockerApi;
use crate::error::DockerError;
use crate::types::ContainerSummary;

/// In-memory Docker mock for unit tests (records calls).
#[derive(Debug, Clone)]
pub struct MockDocker {
    inner: Arc<Mutex<MockState>>,
    allowlist: Allowlist,
}

#[derive(Debug, Default)]
struct MockState {
    containers: Vec<ContainerSummary>,
    calls: Vec<(String, String)>,
    pull_ok: bool,
    fail_create: bool,
    auto_recreate: bool,
    wait_status: i64,
    logs: String,
}

impl Default for MockDocker {
    fn default() -> Self { Self::new() }
}

impl MockDocker {
    /// Empty mock; pulls succeed. Uses **updater** allowlist for `raw_call`.
    #[must_use]
    pub fn new() -> Self { Self::with_allowlist(Allowlist::updater()) }

    /// Mock bound to an explicit allowlist (verifier path tests).
    #[must_use]
    pub fn with_allowlist(allowlist: Allowlist) -> Self {
        Self {
            inner: Arc::new(Mutex::new(MockState { pull_ok: true, auto_recreate: true, ..MockState::default() })),
            allowlist,
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, MockState>, DockerError> {
        self.inner.lock().map_err(|e| DockerError::Api(e.to_string()))
    }

    fn record(&self, method: &str, path: String) -> Result<(), DockerError> {
        self.lock()?.calls.push((method.to_owned(), path));
        Ok(())
    }

    /// Seed a running container.
    pub fn seed(&self, c: ContainerSummary) {
        if let Ok(mut g) = self.lock() { g.containers.push(c); }
    }

    /// Recorded `(METHOD, path)` calls.
    #[must_use]
    pub fn calls(&self) -> Vec<(String, String)> {
        self.lock().map(|g| g.calls.clone()).unwrap_or_default()
    }

    /// Force pull failures.
    pub fn set_pull_ok(&self, ok: bool) {
        if let Ok(mut g) = self.lock() { g.pull_ok = ok; }
    }

    /// Force create failures.
    pub fn set_fail_create(&self, fail: bool) {
        if let Ok(mut g) = self.lock() { g.fail_create = fail; }
    }

    /// Set wait status / logs for one-shot run tests.
    pub fn set_run_result(&self, status: i64, logs: impl Into<String>) {
        if let Ok(mut g) = self.lock() { g.wait_status = status; g.logs = logs.into(); }
    }

    /// Raw call through the allowlist (S3 tests).
    ///
    /// # Errors
    /// [`DockerError::NotAllowlisted`] when blocked.
    pub fn raw_call(&self, method: &str, path: &str) -> Result<(), DockerError> {
        self.allowlist.assert_allowed(method, path)?;
        self.record(method, path.to_owned())
    }
}

impl DockerApi for MockDocker {
    fn list_containers(&self) -> Result<Vec<ContainerSummary>, DockerError> {
        let mut g = self.lock()?;
        g.calls.push(("GET".into(), "/containers/json?all=true".into()));
        Ok(g.containers.clone())
    }
    fn inspect_container(&self, id_or_name: &str) -> Result<ContainerSummary, DockerError> {
        let mut g = self.lock()?;
        g.calls.push(("GET".into(), format!("/containers/{id_or_name}/json")));
        g.containers.iter().find(|c| c.id == id_or_name || c.name == id_or_name).cloned()
            .ok_or_else(|| DockerError::Api(format!("not found: {id_or_name}")))
    }
    fn pull_image(&self, image: &str) -> Result<(), DockerError> {
        let mut g = self.lock()?;
        g.calls.push(("POST".into(), format!("/images/create?fromImage={image}")));
        if g.pull_ok { Ok(()) } else { Err(DockerError::Api("pull failed".into())) }
    }
    fn stop_container(&self, id_or_name: &str) -> Result<(), DockerError> {
        self.record("POST", format!("/containers/{id_or_name}/stop"))
    }
    fn create_container(&self, name: &str, image: &str, labels: &HashMap<String, String>) -> Result<String, DockerError> {
        let mut g = self.lock()?;
        g.calls.push(("POST".into(), format!("/containers/create?name={name}")));
        if g.fail_create { return Err(DockerError::Api("create failed".into())); }
        let id = format!("id-{name}");
        if g.auto_recreate {
            g.containers.retain(|c| c.name != name);
            g.containers.push(ContainerSummary {
                id: id.clone(), name: name.to_owned(), image: image.to_owned(),
                compose_project: labels.get("com.docker.compose.project").cloned(),
                compose_service: labels.get("com.docker.compose.service").cloned(),
            });
        }
        Ok(id)
    }
    fn start_container(&self, id_or_name: &str) -> Result<(), DockerError> {
        self.record("POST", format!("/containers/{id_or_name}/start"))
    }
    fn rename_container(&self, id_or_name: &str, new_name: &str) -> Result<(), DockerError> {
        let mut g = self.lock()?;
        g.calls.push(("POST".into(), format!("/containers/{id_or_name}/rename?name={new_name}")));
        if let Some(c) = g.containers.iter_mut().find(|c| c.id == id_or_name || c.name == id_or_name) {
            new_name.clone_into(&mut c.name);
        }
        Ok(())
    }
    fn wait_container(&self, id_or_name: &str) -> Result<i64, DockerError> {
        let mut g = self.lock()?;
        g.calls.push(("POST".into(), format!("/containers/{id_or_name}/wait")));
        Ok(g.wait_status)
    }
    fn container_logs(&self, id_or_name: &str) -> Result<String, DockerError> {
        let mut g = self.lock()?;
        g.calls.push(("GET".into(), format!("/containers/{id_or_name}/logs?stdout=true&stderr=true")));
        Ok(g.logs.clone())
    }
    fn remove_container(&self, id_or_name: &str) -> Result<(), DockerError> {
        let mut g = self.lock()?;
        g.calls.push(("DELETE".into(), format!("/containers/{id_or_name}?force=true")));
        g.containers.retain(|c| c.id != id_or_name && c.name != id_or_name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_call_deny_does_not_record() {
        let d = MockDocker::new();
        assert!(matches!(d.raw_call("DELETE", "/volumes/data"), Err(DockerError::NotAllowlisted { .. })));
        assert!(d.calls().is_empty());
    }

    #[test]
    fn verifier_mock_allows_delete_container() {
        let d = MockDocker::with_allowlist(Allowlist::verifier());
        d.raw_call("DELETE", "/containers/abc?force=true").expect("rm");
        assert_eq!(d.calls().len(), 1);
    }

    #[test]
    fn wait_logs_remove_roundtrip() {
        let d = MockDocker::new();
        d.set_run_result(0, "ok\n");
        let id = d.create_container("t", "img@sha256:aa", &HashMap::new()).expect("c");
        d.start_container(&id).expect("s");
        assert_eq!(d.wait_container(&id).expect("w"), 0);
        assert_eq!(d.container_logs(&id).expect("l"), "ok\n");
        d.remove_container(&id).expect("r");
    }
}
