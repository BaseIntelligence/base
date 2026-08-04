//! Live HTTP client for docker-socket-proxy / Engine HTTP.

use std::collections::HashMap;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::allowlist::Allowlist;
use crate::api::DockerApi;
use crate::error::DockerError;
use crate::types::{ContainerSummary, RunResult, RunSpec};
use crate::{is_owned_name, DESIGN_OWNED_NAME_PREFIX, OWNED_NAME_PREFIX};

/// Live HTTP client restricted by an [`Allowlist`].
#[derive(Debug, Clone)]
pub struct AllowlistClient {
    base: String,
    http: reqwest::blocking::Client,
    allowlist: Allowlist,
}

impl AllowlistClient {
    /// Client for `base` with the **updater** allowlist.
    ///
    /// # Errors
    /// When the reqwest client cannot be built.
    pub fn new(base: impl Into<String>) -> Result<Self, DockerError> {
        Self::with_allowlist(base, Allowlist::updater())
    }

    /// Client with an explicit allowlist profile.
    ///
    /// # Errors
    /// When the reqwest client cannot be built.
    pub fn with_allowlist(
        base: impl Into<String>,
        allowlist: Allowlist,
    ) -> Result<Self, DockerError> {
        let http = reqwest::blocking::Client::builder()
            .build()
            .map_err(|e| DockerError::Api(e.to_string()))?;
        Ok(Self {
            base: base.into().trim_end_matches('/').to_owned(),
            http,
            allowlist,
        })
    }

    /// Allowlist profile this client enforces.
    #[must_use]
    pub const fn allowlist(&self) -> Allowlist {
        self.allowlist
    }

    /// Create a container with an explicit command (agent / verifier one-shot).
    ///
    /// # Errors
    /// Propagates [`DockerError`].
    pub fn create_container_cmd(
        &self,
        name: &str,
        image: &str,
        cmd: &[&str],
        labels: &HashMap<String, String>,
    ) -> Result<String, DockerError> {
        self.create_inner(name, image, labels, Some(cmd), None, &[], false, None)
    }

    /// Pull → create → start → wait → logs → remove (one-shot).
    ///
    /// # Errors
    /// Propagates [`DockerError`] from any step.
    pub fn run_once(&self, image: &str, cmd: &[&str]) -> Result<RunResult, DockerError> {
        self.pull_image(image)?;
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let id =
            self.create_container_cmd(&format!("base-run-{ns}"), image, cmd, &HashMap::new())?;
        let result = (|| {
            self.start_container(&id)?;
            Ok(RunResult {
                status_code: self.wait_container(&id)?,
                logs: self.container_logs(&id)?,
            })
        })();
        let _ = self.remove_container(&id);
        result
    }

    /// Owned verifier/design run: name prefix enforced, binds/env/hardening, always removed.
    ///
    /// # Errors
    /// [`DockerError::BadName`] when the name lacks an owned prefix;
    /// [`DockerError::Timeout`] when `timeout_sec` elapses; other API errors.
    pub fn run_owned(&self, spec: &RunSpec) -> Result<RunResult, DockerError> {
        if !is_owned_name(&spec.name) {
            return Err(DockerError::BadName {
                name: spec.name.clone(),
                required_prefix: format!("{OWNED_NAME_PREFIX}|{DESIGN_OWNED_NAME_PREFIX}"),
            });
        }
        // Best-effort pull: local digest-pinned images may 404 from registry.
        let _ = self.pull_image(&spec.image);
        let cmd_refs: Vec<&str> = spec.cmd.iter().map(String::as_str).collect();
        let id = self.create_owned(spec, cmd_refs.as_slice())?;
        let result = self.run_lifecycle(&id, spec.timeout_sec);
        let _ = self.remove_container(&id);
        result
    }

    /// List containers whose name starts with an owned prefix.
    ///
    /// # Errors
    /// Propagates list failures.
    pub fn list_owned(&self) -> Result<Vec<ContainerSummary>, DockerError> {
        Ok(self
            .list_containers()?
            .into_iter()
            .filter(|c| is_owned_name(&c.name))
            .collect())
    }

    /// Force-remove every owned container (best-effort per id).
    ///
    /// # Errors
    /// First list failure; individual remove errors are ignored.
    pub fn cleanup_owned(&self) -> Result<usize, DockerError> {
        let owned = self.list_owned()?;
        let n = owned.len();
        for c in owned {
            let _ = self.remove_container(&c.id);
            if !c.name.is_empty() {
                let _ = self.remove_container(&c.name);
            }
        }
        Ok(n)
    }

    fn run_lifecycle(&self, id: &str, timeout_sec: Option<u64>) -> Result<RunResult, DockerError> {
        self.start_container(id)?;
        let status_code = match timeout_sec {
            None => self.wait_container(id)?,
            Some(secs) => self.wait_with_timeout(id, secs)?,
        };
        let logs = self.container_logs(id).unwrap_or_default();
        Ok(RunResult { status_code, logs })
    }

    fn wait_with_timeout(&self, id: &str, timeout_sec: u64) -> Result<i64, DockerError> {
        // Poll inspect instead of POST /wait: socket-proxy / HAProxy often
        // reset long-lived wait streams (~30–60s), which breaks multi-minute
        // design-runtime LLM runs.
        let deadline = std::time::Instant::now() + Duration::from_secs(timeout_sec);
        loop {
            let insp = self.request_json("GET", &format!("/containers/{id}/json"), None)?;
            let running = insp
                .pointer("/State/Running")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !running {
                return Ok(insp
                    .pointer("/State/ExitCode")
                    .and_then(Value::as_i64)
                    .unwrap_or(1));
            }
            if std::time::Instant::now() >= deadline {
                let _ = self.stop_container(id);
                return Err(DockerError::Timeout { timeout_sec });
            }
            thread::sleep(Duration::from_secs(2));
        }
    }

    fn create_owned(&self, spec: &RunSpec, cmd: &[&str]) -> Result<String, DockerError> {
        let mut body = serde_json::json!({
            "Image": spec.image,
            "Labels": {},
            "Cmd": cmd,
            "Tty": true,
            "AttachStdout": true,
            "AttachStderr": true,
        });
        if !spec.env.is_empty() {
            body["Env"] = serde_json::json!(spec.env);
        }
        if spec.network_disabled {
            body["NetworkDisabled"] = Value::Bool(true);
        }
        if let Some(wd) = &spec.working_dir {
            body["WorkingDir"] = Value::String(wd.clone());
        }
        if let Some(user) = &spec.user {
            body["User"] = Value::String(user.clone());
        }
        let mut host = serde_json::Map::new();
        if !spec.binds.is_empty() {
            host.insert("Binds".into(), serde_json::json!(spec.binds));
        }
        if spec.readonly_rootfs {
            host.insert("ReadonlyRootfs".into(), Value::Bool(true));
        }
        if spec.cap_drop_all {
            host.insert("CapDrop".into(), serde_json::json!(["ALL"]));
        }
        if spec.no_new_privileges {
            host.insert(
                "SecurityOpt".into(),
                serde_json::json!(["no-new-privileges:true"]),
            );
        }
        if let Some(n) = spec.pids_limit {
            host.insert("PidsLimit".into(), Value::Number(n.into()));
        }
        if let Some(m) = spec.memory_bytes {
            host.insert("Memory".into(), Value::Number(m.into()));
        }
        if let Some(m) = spec.memory_swap_bytes {
            host.insert("MemorySwap".into(), Value::Number(m.into()));
        }
        if let Some(n) = spec.nano_cpus {
            host.insert("NanoCpus".into(), Value::Number(n.into()));
        }
        if let Some(mode) = &spec.network_mode {
            if !spec.network_disabled {
                host.insert("NetworkMode".into(), Value::String(mode.clone()));
            }
        }
        if !spec.tmpfs.is_empty() {
            let mut map = serde_json::Map::new();
            for (path, opts) in &spec.tmpfs {
                map.insert(path.clone(), Value::String(opts.clone()));
            }
            host.insert("Tmpfs".into(), Value::Object(map));
        }
        if !host.is_empty() {
            body["HostConfig"] = Value::Object(host);
        }
        let val = self.request_json(
            "POST",
            &format!("/containers/create?name={}", spec.name),
            Some(body),
        )?;
        val.get("Id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| DockerError::Json("create response missing Id".into()))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_inner(
        &self,
        name: &str,
        image: &str,
        labels: &HashMap<String, String>,
        cmd: Option<&[&str]>,
        binds: Option<&[String]>,
        env: &[String],
        network_disabled: bool,
        working_dir: Option<&str>,
    ) -> Result<String, DockerError> {
        let mut body = serde_json::json!({ "Image": image, "Labels": labels });
        if let Some(c) = cmd {
            body["Cmd"] = serde_json::json!(c);
            body["Tty"] = Value::Bool(true);
            body["AttachStdout"] = Value::Bool(true);
            body["AttachStderr"] = Value::Bool(true);
        }
        if !env.is_empty() {
            body["Env"] = serde_json::json!(env);
        }
        if network_disabled {
            body["NetworkDisabled"] = Value::Bool(true);
        }
        if let Some(wd) = working_dir {
            body["WorkingDir"] = Value::String(wd.to_owned());
        }
        if let Some(b) = binds {
            if !b.is_empty() {
                body["HostConfig"] = serde_json::json!({ "Binds": b });
            }
        }
        let val = self.request_json(
            "POST",
            &format!("/containers/create?name={name}"),
            Some(body),
        )?;
        val.get("Id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| DockerError::Json("create response missing Id".into()))
    }

    fn post_ok(&self, path: &str) -> Result<(), DockerError> {
        let _ = self.request_json("POST", path, None)?;
        Ok(())
    }

    fn request_json(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value, DockerError> {
        let text = self.request_raw(method, path, body)?;
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| DockerError::Json(e.to_string()))
    }

    fn request_raw(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<String, DockerError> {
        self.allowlist.assert_allowed(method, path)?;
        let url = format!("{}{path}", self.base);
        let mut b = match method {
            "GET" => self.http.get(&url),
            "POST" => self.http.post(&url),
            "DELETE" => self.http.delete(&url),
            other => {
                return Err(DockerError::NotAllowlisted {
                    method: other.to_owned(),
                    path: path.to_owned(),
                })
            }
        };
        if let Some(body) = body {
            b = b.json(&body);
        }
        let resp = b.send().map_err(|e| DockerError::Api(e.to_string()))?;
        let status = resp.status();
        let text = resp.text().map_err(|e| DockerError::Api(e.to_string()))?;
        if !status.is_success() {
            return Err(DockerError::Api(format!("HTTP {status}: {text}")));
        }
        Ok(text)
    }
}

impl DockerApi for AllowlistClient {
    fn list_containers(&self) -> Result<Vec<ContainerSummary>, DockerError> {
        let val = self.request_json("GET", "/containers/json?all=true", None)?;
        let arr = val
            .as_array()
            .ok_or_else(|| DockerError::Json("expected array".into()))?;
        Ok(arr.iter().map(|i| summary(i, true)).collect())
    }
    fn inspect_container(&self, id_or_name: &str) -> Result<ContainerSummary, DockerError> {
        Ok(summary(
            &self.request_json("GET", &format!("/containers/{id_or_name}/json"), None)?,
            false,
        ))
    }
    fn pull_image(&self, image: &str) -> Result<(), DockerError> {
        let enc = image
            .replace('@', "%40")
            .replace(':', "%3A")
            .replace('/', "%2F");
        let _ = self.request_raw("POST", &format!("/images/create?fromImage={enc}"), None)?;
        Ok(())
    }
    fn stop_container(&self, id_or_name: &str) -> Result<(), DockerError> {
        self.post_ok(&format!("/containers/{id_or_name}/stop"))
    }
    fn create_container(
        &self,
        name: &str,
        image: &str,
        labels: &HashMap<String, String>,
    ) -> Result<String, DockerError> {
        self.create_inner(name, image, labels, None, None, &[], false, None)
    }
    fn start_container(&self, id_or_name: &str) -> Result<(), DockerError> {
        self.post_ok(&format!("/containers/{id_or_name}/start"))
    }
    fn rename_container(&self, id_or_name: &str, new_name: &str) -> Result<(), DockerError> {
        self.post_ok(&format!("/containers/{id_or_name}/rename?name={new_name}"))
    }
    fn wait_container(&self, id_or_name: &str) -> Result<i64, DockerError> {
        self.request_json("POST", &format!("/containers/{id_or_name}/wait"), None)?
            .get("StatusCode")
            .and_then(Value::as_i64)
            .ok_or_else(|| DockerError::Json("wait response missing StatusCode".into()))
    }
    fn container_logs(&self, id_or_name: &str) -> Result<String, DockerError> {
        self.request_raw(
            "GET",
            &format!("/containers/{id_or_name}/logs?stdout=true&stderr=true"),
            None,
        )
    }
    fn remove_container(&self, id_or_name: &str) -> Result<(), DockerError> {
        let _ = self.request_raw(
            "DELETE",
            &format!("/containers/{id_or_name}?force=true"),
            None,
        )?;
        Ok(())
    }
}

fn summary(val: &Value, list: bool) -> ContainerSummary {
    let name = if list {
        val.get("Names")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
    } else {
        val.get("Name").and_then(Value::as_str)
    }
    .unwrap_or("")
    .trim_start_matches('/')
    .to_owned();
    let image = if list {
        val.get("Image").and_then(Value::as_str)
    } else {
        val.pointer("/Config/Image")
            .and_then(Value::as_str)
            .or_else(|| val.get("Image").and_then(Value::as_str))
    }
    .unwrap_or("")
    .to_owned();
    let labels = if list {
        val.get("Labels").cloned()
    } else {
        val.pointer("/Config/Labels").cloned()
    }
    .unwrap_or(Value::Null);
    let lab = |k: &str| labels.get(k).and_then(Value::as_str).map(str::to_owned);
    ContainerSummary {
        id: val
            .get("Id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        name,
        image,
        compose_project: lab("com.docker.compose.project"),
        compose_service: lab("com.docker.compose.service"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_allowlisted_verb_fails_closed_without_daemon() {
        let c = AllowlistClient::with_allowlist("http://127.0.0.1:1", Allowlist::verifier())
            .expect("c");
        assert!(matches!(
            c.allowlist().assert_allowed("DELETE", "/volumes/data"),
            Err(DockerError::NotAllowlisted { .. })
        ));
        assert!(matches!(
            c.request_raw("POST", "/networks/create", None),
            Err(DockerError::NotAllowlisted { method, path })
                if method == "POST" && path == "/networks/create"
        ));
    }

    #[test]
    fn updater_client_defaults_to_updater_allowlist() {
        let c = AllowlistClient::new("http://socket-proxy:2375").expect("c");
        assert_eq!(c.allowlist(), Allowlist::updater());
        assert!(c.allowlist().is_allowed("POST", "/containers/x/stop"));
        assert!(!c.allowlist().is_allowed("DELETE", "/containers/x"));
    }

    #[test]
    fn run_owned_rejects_name_without_prefix() {
        let c = AllowlistClient::with_allowlist("http://127.0.0.1:1", Allowlist::verifier())
            .expect("c");
        let err = c
            .run_owned(&RunSpec {
                name: "evil".into(),
                image: "busybox@sha256:dead".into(),
                cmd: vec!["true".into()],
                binds: vec![],
                env: vec![],
                network_disabled: false,
                working_dir: None,
                timeout_sec: Some(1),
                readonly_rootfs: false,
                cap_drop_all: false,
                no_new_privileges: false,
                pids_limit: None,
                memory_bytes: None,
                memory_swap_bytes: None,
                nano_cpus: None,
                network_mode: None,
                user: None,
                tmpfs: vec![],
            })
            .expect_err("bad name");
        assert!(matches!(err, DockerError::BadName { .. }));
    }

    #[test]
    fn design_prefix_is_owned() {
        assert!(crate::is_owned_name("base-design-run-1"));
        let spec = RunSpec::design_hardened(
            "base-design-x".into(),
            "img@sha256:ab".into(),
            vec!["true".into()],
        );
        assert!(spec.readonly_rootfs);
        assert!(spec.cap_drop_all);
        assert_eq!(spec.user.as_deref(), Some("65532:65532"));
    }
}
