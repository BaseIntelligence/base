//! Docker Engine HTTP client restricted to socket-proxy allowlist.
//!
//! tecnativa/docker-socket-proxy with `CONTAINERS=1 IMAGES=1 POST=1` (all else 0).
//! This client **never** issues methods/paths outside [`ALLOWED_ROUTES`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// HTTP method + path-prefix pairs permitted for the updater.
///
/// Paths are matched as Engine API routes **without** the optional `/v1.xx` prefix.
pub const ALLOWED_ROUTES: &[(&str, &str)] = &[
    ("GET", "/containers/json"),
    ("GET", "/containers/"),  // /containers/{id}/json
    ("POST", "/containers/"), // create, start, stop, rename, …
    ("POST", "/images/create"),
    ("GET", "/images/"),
];

/// Docker client errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DockerError {
    /// Method/path not on the allowlist (also covers DELETE /volumes etc.).
    #[error("docker API call not allowlisted: {method} {path}")]
    NotAllowlisted {
        /// HTTP method.
        method: String,
        /// Request path.
        path: String,
    },
    /// Transport or HTTP status failure.
    #[error("docker API error: {0}")]
    Api(String),
    /// JSON decode failure.
    #[error("docker API JSON: {0}")]
    Json(String),
}

/// Minimal container summary from list/inspect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerSummary {
    /// Container id.
    pub id: String,
    /// First name without leading `/`.
    pub name: String,
    /// Image reference as reported by Docker.
    pub image: String,
    /// Compose project label if present.
    pub compose_project: Option<String>,
    /// Compose service label if present.
    pub compose_service: Option<String>,
}

/// Abstraction over Docker Engine operations used by the state machine.
pub trait DockerApi: Send + Sync {
    /// List containers (all=true).
    ///
    /// # Errors
    /// Propagates [`DockerError`].
    fn list_containers(&self) -> Result<Vec<ContainerSummary>, DockerError>;

    /// Inspect one container by id or name.
    ///
    /// # Errors
    /// Propagates [`DockerError`].
    fn inspect_container(&self, id_or_name: &str) -> Result<ContainerSummary, DockerError>;

    /// Pull image by reference (`repo@sha256:…`).
    ///
    /// # Errors
    /// Propagates [`DockerError`].
    fn pull_image(&self, image: &str) -> Result<(), DockerError>;

    /// Stop container.
    ///
    /// # Errors
    /// Propagates [`DockerError`].
    fn stop_container(&self, id_or_name: &str) -> Result<(), DockerError>;

    /// Create container from image with the same name (caller stops/removes old first).
    ///
    /// # Errors
    /// Propagates [`DockerError`].
    fn create_container(
        &self,
        name: &str,
        image: &str,
        labels: &HashMap<String, String>,
    ) -> Result<String, DockerError>;

    /// Start container by id or name.
    ///
    /// # Errors
    /// Propagates [`DockerError`].
    fn start_container(&self, id_or_name: &str) -> Result<(), DockerError>;

    /// Remove stopped container (uses POST? No — DELETE is often disabled on proxy).
    /// We use rename + create pattern: rename old to `.old`, create new, start new,
    /// stop old. Removal of `.old` is optional and only via allowlisted POST rename
    /// path; we **do not** call DELETE.
    ///
    /// Rename container.
    ///
    /// # Errors
    /// Propagates [`DockerError`].
    fn rename_container(&self, id_or_name: &str, new_name: &str) -> Result<(), DockerError>;
}

/// Returns true if method+path is permitted.
#[must_use]
pub fn is_allowlisted(method: &str, path: &str) -> bool {
    let method = method.to_ascii_uppercase();
    let path = strip_api_version(path);
    for (m, prefix) in ALLOWED_ROUTES {
        if method == *m && path_matches(path, prefix) {
            return true;
        }
    }
    false
}

fn strip_api_version(path: &str) -> &str {
    // /v1.43/containers/json → /containers/json
    let bytes = path.as_bytes();
    if bytes.len() > 4 && bytes[0] == b'/' && bytes[1] == b'v' {
        if let Some(idx) = path[1..].find('/') {
            return &path[1 + idx..];
        }
    }
    path
}

fn path_matches(path: &str, prefix: &str) -> bool {
    if prefix.ends_with('/') {
        path == prefix.trim_end_matches('/') || path.starts_with(prefix)
    } else {
        path == prefix || path.starts_with(&format!("{prefix}?"))
    }
}

/// Guard that rejects non-allowlisted calls before any HTTP is sent.
///
/// # Errors
/// [`DockerError::NotAllowlisted`] when the method/path pair is blocked.
pub fn assert_allowlisted(method: &str, path: &str) -> Result<(), DockerError> {
    if is_allowlisted(method, path) {
        Ok(())
    } else {
        Err(DockerError::NotAllowlisted {
            method: method.to_owned(),
            path: path.to_owned(),
        })
    }
}

/// Live HTTP client talking to docker-socket-proxy.
#[derive(Debug, Clone)]
pub struct AllowlistClient {
    base: String,
    http: reqwest::blocking::Client,
}

impl AllowlistClient {
    /// Create a client for `base` (e.g. `http://socket-proxy:2375`).
    ///
    /// # Errors
    /// When the reqwest client cannot be built.
    pub fn new(base: impl Into<String>) -> Result<Self, DockerError> {
        let http = reqwest::blocking::Client::builder()
            .build()
            .map_err(|e| DockerError::Api(e.to_string()))?;
        Ok(Self {
            base: base.into().trim_end_matches('/').to_owned(),
            http,
        })
    }

    fn request(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value, DockerError> {
        assert_allowlisted(method, path)?;
        let url = format!("{}{path}", self.base);
        let mut builder = match method {
            "GET" => self.http.get(&url),
            "POST" => self.http.post(&url),
            other => {
                return Err(DockerError::NotAllowlisted {
                    method: other.to_owned(),
                    path: path.to_owned(),
                });
            }
        };
        if let Some(b) = body {
            builder = builder.json(&b);
        }
        let resp = builder
            .send()
            .map_err(|e| DockerError::Api(e.to_string()))?;
        let status = resp.status();
        let text = resp.text().map_err(|e| DockerError::Api(e.to_string()))?;
        if !status.is_success() {
            return Err(DockerError::Api(format!("HTTP {status}: {text}")));
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| DockerError::Json(e.to_string()))
    }
}

impl DockerApi for AllowlistClient {
    fn list_containers(&self) -> Result<Vec<ContainerSummary>, DockerError> {
        let path = "/containers/json?all=true";
        let val = self.request("GET", path, None)?;
        parse_container_list(&val)
    }

    fn inspect_container(&self, id_or_name: &str) -> Result<ContainerSummary, DockerError> {
        let path = format!("/containers/{id_or_name}/json");
        let val = self.request("GET", &path, None)?;
        Ok(parse_inspect(&val))
    }

    fn pull_image(&self, image: &str) -> Result<(), DockerError> {
        // POST /images/create?fromImage=repo&tag= — for digest pins use fromImage=repo@sha256
        let encoded = urlencoding_loose(image);
        let path = format!("/images/create?fromImage={encoded}");
        let _ = self.request("POST", &path, None)?;
        Ok(())
    }

    fn stop_container(&self, id_or_name: &str) -> Result<(), DockerError> {
        let path = format!("/containers/{id_or_name}/stop");
        let _ = self.request("POST", &path, None)?;
        Ok(())
    }

    fn create_container(
        &self,
        name: &str,
        image: &str,
        labels: &HashMap<String, String>,
    ) -> Result<String, DockerError> {
        let path = format!("/containers/create?name={name}");
        let body = serde_json::json!({
            "Image": image,
            "Labels": labels,
        });
        let val = self.request("POST", &path, Some(body))?;
        val.get("Id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| DockerError::Json("create response missing Id".into()))
    }

    fn start_container(&self, id_or_name: &str) -> Result<(), DockerError> {
        let path = format!("/containers/{id_or_name}/start");
        let _ = self.request("POST", &path, None)?;
        Ok(())
    }

    fn rename_container(&self, id_or_name: &str, new_name: &str) -> Result<(), DockerError> {
        let path = format!("/containers/{id_or_name}/rename?name={new_name}");
        let _ = self.request("POST", &path, None)?;
        Ok(())
    }
}

fn urlencoding_loose(s: &str) -> String {
    // Minimal encoding for image refs in query strings.
    s.replace('@', "%40")
        .replace(':', "%3A")
        .replace('/', "%2F")
}

fn parse_container_list(val: &Value) -> Result<Vec<ContainerSummary>, DockerError> {
    let arr = val
        .as_array()
        .ok_or_else(|| DockerError::Json("expected array".into()))?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        out.push(summary_from_list_item(item));
    }
    Ok(out)
}

fn summary_from_list_item(item: &Value) -> ContainerSummary {
    let id = item
        .get("Id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let names = item
        .get("Names")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let name = names
        .first()
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim_start_matches('/')
        .to_owned();
    let image = item
        .get("Image")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let labels = item.get("Labels").cloned().unwrap_or(Value::Null);
    ContainerSummary {
        id,
        name,
        image,
        compose_project: label_str(&labels, "com.docker.compose.project"),
        compose_service: label_str(&labels, "com.docker.compose.service"),
    }
}

fn parse_inspect(val: &Value) -> ContainerSummary {
    let id = val
        .get("Id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let name = val
        .get("Name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim_start_matches('/')
        .to_owned();
    let image = val
        .pointer("/Config/Image")
        .and_then(Value::as_str)
        .or_else(|| val.get("Image").and_then(Value::as_str))
        .unwrap_or("")
        .to_owned();
    let labels = val
        .pointer("/Config/Labels")
        .cloned()
        .unwrap_or(Value::Null);
    ContainerSummary {
        id,
        name,
        image,
        compose_project: label_str(&labels, "com.docker.compose.project"),
        compose_service: label_str(&labels, "com.docker.compose.service"),
    }
}

fn label_str(labels: &Value, key: &str) -> Option<String> {
    labels.get(key).and_then(Value::as_str).map(str::to_owned)
}

/// In-memory Docker mock for unit tests (records calls).
#[derive(Debug, Default, Clone)]
pub struct MockDocker {
    inner: Arc<Mutex<MockState>>,
}

#[derive(Debug, Default)]
struct MockState {
    containers: Vec<ContainerSummary>,
    calls: Vec<(String, String)>,
    pull_ok: bool,
    fail_create: bool,
    /// When set, recreate swaps image on the named container.
    auto_recreate: bool,
}

impl MockDocker {
    /// Empty mock; pulls succeed by default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MockState {
                pull_ok: true,
                auto_recreate: true,
                ..MockState::default()
            })),
        }
    }

    /// Seed a running container.
    pub fn seed(&self, c: ContainerSummary) {
        if let Ok(mut g) = self.inner.lock() {
            g.containers.push(c);
        }
    }

    /// Recorded `(METHOD, path)` calls.
    #[must_use]
    pub fn calls(&self) -> Vec<(String, String)> {
        self.inner
            .lock()
            .map(|g| g.calls.clone())
            .unwrap_or_default()
    }

    /// Force pull failures.
    pub fn set_pull_ok(&self, ok: bool) {
        if let Ok(mut g) = self.inner.lock() {
            g.pull_ok = ok;
        }
    }

    /// Force create failures (simulates recreate error).
    pub fn set_fail_create(&self, fail: bool) {
        if let Ok(mut g) = self.inner.lock() {
            g.fail_create = fail;
        }
    }

    /// Attempt a raw call through the allowlist (for S3 tests).
    ///
    /// # Errors
    /// [`DockerError::NotAllowlisted`] when blocked.
    pub fn raw_call(&self, method: &str, path: &str) -> Result<(), DockerError> {
        assert_allowlisted(method, path)?;
        if let Ok(mut g) = self.inner.lock() {
            g.calls.push((method.to_owned(), path.to_owned()));
        }
        Ok(())
    }
}

impl DockerApi for MockDocker {
    fn list_containers(&self) -> Result<Vec<ContainerSummary>, DockerError> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| DockerError::Api(e.to_string()))?;
        g.calls
            .push(("GET".into(), "/containers/json?all=true".into()));
        Ok(g.containers.clone())
    }

    fn inspect_container(&self, id_or_name: &str) -> Result<ContainerSummary, DockerError> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| DockerError::Api(e.to_string()))?;
        g.calls
            .push(("GET".into(), format!("/containers/{id_or_name}/json")));
        g.containers
            .iter()
            .find(|c| c.id == id_or_name || c.name == id_or_name)
            .cloned()
            .ok_or_else(|| DockerError::Api(format!("not found: {id_or_name}")))
    }

    fn pull_image(&self, image: &str) -> Result<(), DockerError> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| DockerError::Api(e.to_string()))?;
        let path = format!("/images/create?fromImage={image}");
        g.calls.push(("POST".into(), path));
        if g.pull_ok {
            Ok(())
        } else {
            Err(DockerError::Api("pull failed".into()))
        }
    }

    fn stop_container(&self, id_or_name: &str) -> Result<(), DockerError> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| DockerError::Api(e.to_string()))?;
        g.calls
            .push(("POST".into(), format!("/containers/{id_or_name}/stop")));
        Ok(())
    }

    fn create_container(
        &self,
        name: &str,
        image: &str,
        labels: &HashMap<String, String>,
    ) -> Result<String, DockerError> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| DockerError::Api(e.to_string()))?;
        g.calls
            .push(("POST".into(), format!("/containers/create?name={name}")));
        if g.fail_create {
            return Err(DockerError::Api("create failed".into()));
        }
        let id = format!("id-{name}");
        if g.auto_recreate {
            // Replace any container with this name.
            g.containers.retain(|c| c.name != name);
            g.containers.push(ContainerSummary {
                id: id.clone(),
                name: name.to_owned(),
                image: image.to_owned(),
                compose_project: labels.get("com.docker.compose.project").cloned(),
                compose_service: labels.get("com.docker.compose.service").cloned(),
            });
        }
        Ok(id)
    }

    fn start_container(&self, id_or_name: &str) -> Result<(), DockerError> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| DockerError::Api(e.to_string()))?;
        g.calls
            .push(("POST".into(), format!("/containers/{id_or_name}/start")));
        Ok(())
    }

    fn rename_container(&self, id_or_name: &str, new_name: &str) -> Result<(), DockerError> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| DockerError::Api(e.to_string()))?;
        g.calls.push((
            "POST".into(),
            format!("/containers/{id_or_name}/rename?name={new_name}"),
        ));
        if let Some(c) = g
            .containers
            .iter_mut()
            .find(|c| c.id == id_or_name || c.name == id_or_name)
        {
            new_name.clone_into(&mut c.name);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_accepts_container_and_image_routes() {
        assert!(is_allowlisted("GET", "/containers/json"));
        assert!(is_allowlisted("GET", "/v1.43/containers/json"));
        assert!(is_allowlisted("POST", "/images/create?fromImage=x"));
        assert!(is_allowlisted("POST", "/containers/abc/stop"));
        assert!(is_allowlisted("POST", "/containers/create?name=x"));
    }

    #[test]
    fn allowlist_rejects_volumes_delete_and_networks() {
        assert!(!is_allowlisted("DELETE", "/volumes/foo"));
        assert!(!is_allowlisted("GET", "/volumes"));
        assert!(!is_allowlisted("POST", "/networks/create"));
        assert!(!is_allowlisted("DELETE", "/containers/abc"));
        assert!(assert_allowlisted("DELETE", "/volumes/x").is_err());
    }
}
