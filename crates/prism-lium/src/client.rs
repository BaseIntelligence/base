//! Real Lium HTTPS client + SSH-backed live eval.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use lium_rent_pool::RentPool;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;
use tokio::time::sleep;
use tracing::{debug, info, warn};

fn rent_pool() -> &'static RentPool {
    static POOL: OnceLock<RentPool> = OnceLock::new();
    POOL.get_or_init(RentPool::new)
}

use crate::ssh::{
    parse_ssh_target, resolve_private_key, ssh_exec, ssh_exec_allow_fail, ssh_exec_stdin,
    truncate_tail, SshTarget,
};
use crate::{EvalJobBackend, HARNESS_LOG_RETAIN_BYTES, LIUM_API_BASE_URL, MIN_LIFETIME_HOURS};
use prism_lium_harness::{
    classify_log, detach_launch_cmd, eval_assets_dir, harness_env_pairs, harness_upload_tar,
    parse_harness_probe, parse_metrics_output, random_seed_hex, HarnessProgress,
    EVAL_ASSETS_POD_DIR, HARNESS_ABSENT, HARNESS_BOOTSTRAP, HARNESS_EXTRACT_CMD, HARNESS_PROBE_CMD,
    TRAIN_DONE_MARKER,
};
use prism_lium_types::{CostGuardrailError, LiumError};
use prism_lium_types::{
    GpuPreference, Instance, InstanceSpec, LiumSshConfig, Offer, RemoteExecResult,
};

// cu13.0.2-dinD: sshd from image init (empty startup). Other marketplace
// images lack a stable sshd under Lium's metachar-free startup rules.
const RECIPES_TEMPLATE_IMAGE: &str = "daturaai/pytorch";
const RECIPES_TEMPLATE_TAG: &str = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind";
const RECIPES_TEMPLATE_NAME: &str = "prism-recipe-v9";
const RECIPES_TEMPLATE_STARTUP: &str = "";

const RUNNING_STATUSES: &[&str] = &["RUNNING", "RUNNING_SSH", "READY"];
const TERMINAL_FAIL_STATUSES: &[&str] = &[
    "FAILED",
    "ERROR",
    "CREATION_FAILED",
    "TERMINATED",
    "DELETED",
    "STOPPED",
];
const RATE_LIMIT_RETRIES: u32 = 5;
const RATE_LIMIT_BASE_MS: u64 = 2_000;
const DEPS_INSTALL_TIMEOUT_SECS: u64 = 1_200;

/// Async Lium REST client (`X-API-Key`).
pub struct LiumClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    ssh: LiumSshConfig,
}

impl std::fmt::Debug for LiumClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiumClient")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("ssh_private_key_path", &self.ssh.private_key_path)
            .finish()
    }
}

impl LiumClient {
    /// # Errors
    /// Empty key or HTTP client build failure.
    pub fn new(api_key: impl Into<String>) -> Result<Self, LiumError> {
        Self::with_base_url(api_key, LIUM_API_BASE_URL)
    }

    /// Custom base URL (tests / wiremock).
    ///
    /// # Errors
    /// Empty key or HTTP client build failure.
    pub fn with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Result<Self, LiumError> {
        Self::with_config(api_key, base_url, LiumSshConfig::default_live())
    }

    /// # Errors
    /// Empty key or HTTP client build failure.
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        ssh: LiumSshConfig,
    ) -> Result<Self, LiumError> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(LiumError::Api("empty LIUM_API_KEY".into()));
        }
        let mut headers = HeaderMap::new();
        let mut hv = HeaderValue::from_str(&api_key)
            .map_err(|e| LiumError::Api(format!("invalid api key header: {e}")))?;
        hv.set_sensitive(true);
        headers.insert("X-API-Key", hv);
        headers.insert(
            reqwest::header::USER_AGENT,
            HeaderValue::from_static("prism-lium/0.1 (base; +https://lium.io)"),
        );
        headers.insert(
            reqwest::header::ACCEPT,
            HeaderValue::from_static("application/json"),
        );
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| LiumError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
            ssh,
        })
    }

    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn set_ssh_private_key_path(&mut self, path: PathBuf) {
        self.ssh.private_key_path = Some(path);
    }

    fn validate_spec(spec: &InstanceSpec) -> Result<(), CostGuardrailError> {
        if spec.max_lifetime_hours <= 0.0 {
            return Err(CostGuardrailError::LifetimeMissing);
        }
        if spec.max_lifetime_hours < MIN_LIFETIME_HOURS {
            return Err(CostGuardrailError::LifetimeBelowFloor);
        }
        if spec.max_price_per_hour <= 0.0 {
            return Err(CostGuardrailError::PriceMissing);
        }
        Ok(())
    }

    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, LiumError> {
        let url = format!("{}{path}", self.base_url);
        let mut attempt = 0u32;
        loop {
            let mut builder = self.http.request(method.clone(), &url);
            if let Some(b) = body {
                builder = builder.json(b);
            }
            let resp = builder
                .send()
                .await
                .map_err(|e| LiumError::Transport(sanitize_err(&e.to_string(), &self.api_key)))?;
            let status = resp.status();
            let retry_after = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            let text = resp
                .text()
                .await
                .map_err(|e| LiumError::Transport(sanitize_err(&e.to_string(), &self.api_key)))?;
            // Rent POSTs are gated by `RentPool` — do not multi-retry (each
            // attempt burns Lium's 60/h budget). Other endpoints keep backoff.
            let is_rent = path.contains("/rent");
            if status.as_u16() == 429 {
                let secs = retry_after.or_else(|| lium_rent_pool::parse_retry_secs(&text));
                if is_rent {
                    rent_pool().note_429(secs.unwrap_or(5));
                } else if attempt < RATE_LIMIT_RETRIES {
                    attempt = attempt.saturating_add(1);
                    let wait_ms = secs.map_or(RATE_LIMIT_BASE_MS << (attempt - 1).min(3), |s| {
                        s.saturating_mul(1000).max(50)
                    });
                    warn!(%path, attempt, wait_ms, "lium 429; backing off");
                    sleep(Duration::from_millis(wait_ms)).await;
                    continue;
                }
            }
            if !status.is_success() {
                return Err(LiumError::Api(format!(
                    "{method} {path} -> {status}: {}",
                    truncate(&sanitize_err(&text, &self.api_key), 200)
                )));
            }
            if text.trim().is_empty() {
                return Ok(Value::Null);
            }
            return serde_json::from_str(&text).map_err(|e| LiumError::Api(format!("json: {e}")));
        }
    }

    fn parse_offers(v: &Value) -> Vec<Offer> {
        v.as_array()
            .cloned()
            .unwrap_or_else(|| get_array(v, &["executors", "data"]))
            .iter()
            .filter_map(parse_one_offer)
            .collect()
    }

    async fn list_pods_raw(&self) -> Result<Vec<Value>, LiumError> {
        let v = self.request(reqwest::Method::GET, "/pods", None).await?;
        Ok(v.as_array()
            .cloned()
            .unwrap_or_else(|| get_array(&v, &["pods"])))
    }

    pub async fn get_pod_raw(&self, instance_id: &str) -> Result<Value, LiumError> {
        self.request(reqwest::Method::GET, &format!("/pods/{instance_id}"), None)
            .await
    }

    pub async fn status(&self, instance_id: &str) -> Result<Instance, LiumError> {
        let v = self.get_pod_raw(instance_id).await?;
        Ok(parse_instance(&v, instance_id))
    }

    pub async fn ensure_ssh_key(
        &self,
        public_key: &str,
        name: Option<&str>,
    ) -> Result<Value, LiumError> {
        let normalized = public_key.trim();
        let v = self
            .request(reqwest::Method::GET, "/ssh-keys", None)
            .await?;
        let keys = v
            .as_array()
            .cloned()
            .unwrap_or_else(|| get_array(&v, &["ssh_keys"]));
        if let Some(key) = keys.iter().find(|k| {
            k.get("public_key")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .trim()
                == normalized
        }) {
            return Ok(key.clone());
        }
        let mut body = serde_json::json!({ "public_key": public_key });
        if let Some(n) = name {
            body["name"] = Value::String(n.to_owned());
        }
        self.request(reqwest::Method::POST, "/ssh-keys", Some(&body))
            .await
    }

    pub async fn ensure_template(
        &self,
        name: &str,
        docker_image: &str,
        docker_image_tag: Option<&str>,
        startup_commands: Option<&str>,
    ) -> Result<String, LiumError> {
        let v = self
            .request(reqwest::Method::GET, "/templates", None)
            .await?;
        let templates = v
            .as_array()
            .cloned()
            .unwrap_or_else(|| get_array(&v, &["templates"]));
        for tmpl in templates {
            if get_str(&tmpl, &["name"]) == Some(name) {
                if let Some(id) = get_str(&tmpl, &["id"]) {
                    return Ok(id.to_owned());
                }
            }
        }
        let mut body = serde_json::json!({
            "name": name,
            "docker_image": docker_image,
            "internal_ports": [22],
            "is_private": true,
            "container_start_immediately": true,
        });
        if let Some(tag) = docker_image_tag {
            body["docker_image_tag"] = serde_json::Value::String(tag.to_owned());
        }
        if let Some(cmd) = startup_commands {
            body["startup_commands"] = serde_json::Value::String(cmd.to_owned());
        }
        let created = self
            .request(reqwest::Method::POST, "/templates", Some(&body))
            .await?;
        created
            .get("id")
            .and_then(|x| x.as_str())
            .map(str::to_owned)
            .ok_or_else(|| LiumError::Api("template create missing id".into()))
    }

    async fn resolve_template_id(&self, spec: &InstanceSpec) -> Result<String, LiumError> {
        if let Some(id) = &spec.template_id {
            if !id.is_empty() {
                return Ok(id.clone());
            }
        }
        let name = spec
            .template_name
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or(RECIPES_TEMPLATE_NAME);
        self.ensure_template(
            name,
            RECIPES_TEMPLATE_IMAGE,
            Some(RECIPES_TEMPLATE_TAG),
            Some(RECIPES_TEMPLATE_STARTUP),
        )
        .await
    }

    /// Account balance (USD) when available.
    pub async fn balance(&self) -> Result<f64, LiumError> {
        let v = self
            .request(reqwest::Method::GET, "/users/me", None)
            .await?;
        v.get("balance")
            .and_then(|x| x.as_f64())
            .or_else(|| get_str(&v, &["balance"]).and_then(|s| s.parse().ok()))
            .ok_or_else(|| LiumError::Api("users/me missing balance".into()))
    }

    pub async fn wait_until_running(&self, instance_id: &str) -> Result<Instance, LiumError> {
        let timeout = Duration::from_secs(self.ssh.running_timeout_secs.max(30));
        let start = Instant::now();
        let mut last = String::new();
        loop {
            let inst = self.status(instance_id).await?;
            let st = inst.status.to_ascii_uppercase();
            if st != last {
                info!(%instance_id, status = %st, "lium pod status");
                last = st.clone();
            }
            if RUNNING_STATUSES.iter().any(|s| st == *s) {
                return Ok(inst);
            }
            if TERMINAL_FAIL_STATUSES.iter().any(|s| st.contains(s)) {
                return Err(LiumError::Api(format!(
                    "pod {instance_id} terminal status {st}"
                )));
            }
            if start.elapsed() >= timeout {
                return Err(LiumError::Api(format!(
                    "pod {instance_id} not RUNNING within {}s (last {st})",
                    timeout.as_secs()
                )));
            }
            sleep(Duration::from_secs(5)).await;
        }
    }

    async fn cleanup_after_rent(&self, id: &str) {
        if let Err(e) = self.terminate(id).await {
            warn!(error = %e, pod_id = %id, "lium cleanup terminate failed");
        }
        let _ = self.verify_terminated(id).await;
    }

    async fn find_pod_id_by_name(&self, name: &str) -> Option<String> {
        self.list_pods_raw().await.ok()?.into_iter().find_map(|p| {
            (get_str(&p, &["pod_name", "name"]).unwrap_or("") == name)
                .then(|| get_str(&p, &["id"]).map(str::to_owned))?
        })
    }

    /// Terminate every pod still listed under `name` (429/orphan storms).
    async fn reclaim_pods_named(&self, name: &str) {
        for _ in 0..8 {
            let Some(id) = self.find_pod_id_by_name(name).await else {
                return;
            };
            self.cleanup_after_rent(&id).await;
        }
    }

    async fn resolve_ssh_target(&self, instance_id: &str) -> Result<SshTarget, LiumError> {
        let raw = self.get_pod_raw(instance_id).await?;
        let cmd = raw
            .get("ssh_connect_cmd")
            .and_then(|x| x.as_str())
            .unwrap_or("");
        parse_ssh_target(cmd, &raw).ok_or_else(|| {
            LiumError::Exec(format!(
                "could not parse ssh target for pod {instance_id} from {cmd:?}"
            ))
        })
    }

    /// Live recipe eval: wait RUNNING → stage harness → **detach** `main.py`
    /// → poll `harness.log` (survives control-plane restart / SSH drop).
    ///
    /// Private-tier assets: when `PRISM_EVAL_ASSETS_DIR` is set, the poller
    /// stages the pack on [`TRAIN_DONE_MARKER`] (exact raw line).
    async fn exec_eval_live(
        &self,
        instance_id: &str,
        architecture_py: &str,
        training_py: &str,
        tree_blob: Option<&[u8]>,
    ) -> Result<RemoteExecResult, LiumError> {
        let _running = self.wait_until_running(instance_id).await?;
        let target = self.resolve_ssh_target(instance_id).await?;
        let key = resolve_private_key(self.ssh.private_key_path.as_deref())?;
        let gpu_type = self.gpu_smoke(&target, &key).await?;
        self.ensure_python_deps(&target, &key).await?;

        let train_cap_secs = (self.ssh.train_hours_cap * 3600.0) as u64;
        let timeout_secs = train_cap_secs.saturating_add(3600);
        let tar = harness_upload_tar(architecture_py, training_py, tree_blob)?;
        let (att, rty) = (self.ssh.ssh_attempts, self.ssh.ssh_retry_secs);
        ssh_exec_stdin(&target, &key, HARNESS_EXTRACT_CMD, &tar, att, rty, 300).await?;
        let assets = eval_assets_dir();
        let pairs = harness_env_pairs(self.ssh.train_hours_cap, &gpu_type, assets.is_some());
        #[allow(clippy::format_collect)]
        let env: String = pairs
            .iter()
            .map(|(k, v)| format!("export {k}='{v}'\n"))
            .collect();
        let launch = format!(
            "{HARNESS_BOOTSTRAP}{}",
            detach_launch_cmd(&env, timeout_secs)
        );
        let out = ssh_exec_allow_fail(&target, &key, &launch, att, rty, 120).await?;
        if !out.stdout.contains("DETACH_STARTED")
            && !out.stdout.contains("DETACH_ALREADY")
            && !out.stdout.contains("DETACH_DONE")
        {
            return Err(LiumError::Exec(format!(
                "detach launch failed: {}",
                truncate_tail(&out.stderr, 400)
            )));
        }
        self.poll_detached_eval(
            instance_id,
            &target,
            &key,
            assets.as_deref(),
            train_cap_secs.saturating_add(3900),
        )
        .await
    }

    /// Reattach: probe harness, refuse if absent, else poll to terminal.
    async fn resume_eval_live(&self, instance_id: &str) -> Result<RemoteExecResult, LiumError> {
        let _running = self.wait_until_running(instance_id).await?;
        let target = self.resolve_ssh_target(instance_id).await?;
        let key = resolve_private_key(self.ssh.private_key_path.as_deref())?;
        let (att, rty) = (self.ssh.ssh_attempts, self.ssh.ssh_retry_secs);
        let probe_out =
            ssh_exec_allow_fail(&target, &key, HARNESS_PROBE_CMD, att.max(1), rty, 60).await?;
        let probe = parse_harness_probe(&probe_out.stdout);
        if !probe.attachable() {
            return Err(LiumError::Exec(format!(
                "{HARNESS_ABSENT}: no harness on pod {instance_id}"
            )));
        }
        let assets = eval_assets_dir();
        let train_cap_secs = (self.ssh.train_hours_cap * 3600.0) as u64;
        self.poll_detached_eval(
            instance_id,
            &target,
            &key,
            assets.as_deref(),
            train_cap_secs.saturating_add(3900),
        )
        .await
    }

    /// Poll `harness.log` until metrics, staging assets on train-done.
    async fn poll_detached_eval(
        &self,
        instance_id: &str,
        target: &SshTarget,
        key: &Path,
        assets: Option<&Path>,
        timeout_secs: u64,
    ) -> Result<RemoteExecResult, LiumError> {
        let start = Instant::now();
        let period = Duration::from_secs(20);
        let mut staged = false;
        let marker = std::env::var("PRISM_TRAIN_DONE_MARKER")
            .unwrap_or_else(|_| TRAIN_DONE_MARKER.to_owned());
        loop {
            if start.elapsed() >= Duration::from_secs(timeout_secs) {
                let h = self
                    .harvest_logs_inner(instance_id)
                    .await
                    .unwrap_or_default();
                return Err(LiumError::Exec(format!(
                    "harness poll timed out after {timeout_secs}s; harvested: {}",
                    truncate_tail(&h, 4000)
                )));
            }
            let rty = self.ssh.ssh_retry_secs;
            let probe_out = ssh_exec_allow_fail(target, key, HARNESS_PROBE_CMD, 1, rty, 45).await?;
            let probe = parse_harness_probe(&probe_out.stdout);
            if probe.assets_ready {
                staged = true;
            }
            let log = self
                .harvest_logs_inner(instance_id)
                .await
                .unwrap_or_default();
            // Prefer exact configured marker when present in full log harvest.
            let train_line = log.lines().any(|l| l.trim_end() == marker);
            match classify_log(&log, assets.is_some(), staged || probe.assets_ready) {
                HarnessProgress::Done(res) => {
                    if assets.is_some() && !staged && !probe.assets_ready {
                        return Err(LiumError::Exec(
                            "eval assets configured but harness never emitted the train-done marker — refusing public-tier result under PRISM_EVAL_ASSETS_DIR".into(),
                        ));
                    }
                    let tier_ok = matches!(res.eval_tier.as_deref(), Some("public" | "private"));
                    if (staged || probe.assets_ready) && !tier_ok {
                        return Err(LiumError::Exec(format!(
                            "eval assets staged but harness reported eval_tier={:?} (want \"public\"|\"private\")",
                            res.eval_tier
                        )));
                    }
                    return Ok(*res);
                }
                HarnessProgress::NeedsAssets | HarnessProgress::Running
                    if assets.is_some()
                        && !staged
                        && !probe.assets_ready
                        && (train_line || probe.train_done) =>
                {
                    if let Some(a) = assets {
                        self.stage_eval_assets(target, key, a).await?;
                        staged = true;
                        info!("eval assets staged post-train (detached poll)");
                    }
                }
                HarnessProgress::Failed(msg) => {
                    return Err(LiumError::Exec(msg));
                }
                HarnessProgress::Running | HarnessProgress::NeedsAssets => {
                    if !probe.pid_alive && probe.has_log && !probe.terminal {
                        // Process died without terminal markers.
                        let _ = parse_metrics_output(&log, 1, &log)?;
                        return Err(LiumError::Exec(format!(
                            "harness exited without EVAL_OK; harvested: {}",
                            truncate_tail(&log, 4000)
                        )));
                    }
                }
            }
            sleep(period).await;
        }
    }

    async fn instance_running_live(&self, instance_id: &str) -> Result<bool, LiumError> {
        match self.status(instance_id).await {
            Ok(inst) => {
                let st = inst.status.to_ascii_uppercase();
                Ok(RUNNING_STATUSES.iter().any(|s| st == *s))
            }
            Err(LiumError::Api(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Stage the operator assets into the pod workdir post-train in a single
    /// ssh round-trip: the tar stream goes over stdin while the remote
    /// command extracts it, writes the client-generated `SECRET_SEED`, and
    /// finally touches `.ready` (last — the harness gate treats `.ready` as
    /// the go signal; `set -e` keeps a partial stage from ever touching it).
    ///
    /// The whole assets tree rides along, so the G5 natural-document packs
    /// under [`prism_recipe::NATURAL_PACK_REL`] reach the pod on this same
    /// path with no transport of their own; they are the largest thing in
    /// the tree, which is what [`prism_recipe::MAX_EVAL_ASSETS_PACKED_BYTES`]
    /// is measured against.
    async fn stage_eval_assets(
        &self,
        target: &SshTarget,
        key: &Path,
        assets: &Path,
    ) -> Result<(), LiumError> {
        let tar = tokio::process::Command::new("tar")
            .args(["-cz", "-C"])
            .arg(assets)
            .arg(".")
            .output()
            .await
            .map_err(|e| LiumError::Exec(format!("tar assets: {e}")))?;
        if !tar.status.success() {
            return Err(LiumError::Exec(format!(
                "tar assets: {}",
                truncate(&String::from_utf8_lossy(&tar.stderr), 200)
            )));
        }
        if tar.stdout.len() > prism_recipe::MAX_EVAL_ASSETS_PACKED_BYTES {
            return Err(LiumError::Exec("eval assets exceed the packed cap".into()));
        }
        let seed = random_seed_hex()?;
        let cmd = format!(
            "set -e; d={EVAL_ASSETS_POD_DIR}; mkdir -p $d; tar -xz -C $d; printf '%s' '{seed}' > $d/SECRET_SEED; touch $d/.ready"
        );
        let (att, rty) = (self.ssh.ssh_attempts, self.ssh.ssh_retry_secs);
        ssh_exec_stdin(target, key, &cmd, &tar.stdout, att, rty, 300).await?;
        Ok(())
    }

    async fn ensure_python_deps(&self, target: &SshTarget, key: &Path) -> Result<(), LiumError> {
        const VERIFY: &str =
            "python3 -c 'import transformers, datasets, pyarrow; print(\"DEPS_OK\")'";
        const INSTALL: &str = "set -e\ncommand -v pip >/dev/null 2>&1 || { apt-get update -q; DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-pip; }\npython3 -c 'import transformers, datasets, pyarrow' 2>/dev/null || pip install --break-system-packages --root-user-action=ignore 'transformers==4.44.2' 'datasets==3.0.2' 'pyarrow==17.0.0'\npython3 -c 'import transformers, datasets, pyarrow; print(\"DEPS_OK\")'\n";
        let (a, r) = (self.ssh.ssh_attempts, self.ssh.ssh_retry_secs);
        let out =
            ssh_exec_allow_fail(target, key, INSTALL, a, r, DEPS_INSTALL_TIMEOUT_SECS).await?;
        if out.stdout.contains("DEPS_OK") {
            return Ok(());
        }
        let v = ssh_exec_allow_fail(target, key, VERIFY, a.max(3), r, 120).await?;
        if v.stdout.contains("DEPS_OK") {
            return Ok(());
        }
        Err(LiumError::Exec(format!(
            "deps install failed (code {}): {}",
            out.returncode,
            truncate_tail(
                &format!(
                    "{}\n{}\n---\n{}\n{}",
                    out.stdout, out.stderr, v.stdout, v.stderr
                ),
                HARNESS_LOG_RETAIN_BYTES
            )
        )))
    }

    async fn harvest_logs_inner(&self, instance_id: &str) -> Result<String, LiumError> {
        let target = self.resolve_ssh_target(instance_id).await?;
        let key = resolve_private_key(self.ssh.private_key_path.as_deref())?;
        let cmd = format!(
            "tail -c {HARNESS_LOG_RETAIN_BYTES} /tmp/prism_eval/harness.log 2>/dev/null || true"
        );
        let out = ssh_exec_allow_fail(&target, &key, &cmd, 1, self.ssh.ssh_retry_secs, 45).await?;
        Ok(truncate_tail(&out.stdout, HARNESS_LOG_RETAIN_BYTES))
    }

    async fn gpu_smoke(&self, target: &SshTarget, key: &Path) -> Result<String, LiumError> {
        let (att, rty) = (self.ssh.ssh_attempts, self.ssh.ssh_retry_secs);
        let smoke = ssh_exec(target, key, "nvidia-smi -L && echo SMOKE_OK", att, rty, 60).await?;
        if !smoke.stdout.contains("SMOKE_OK") {
            return Err(LiumError::Exec(format!(
                "nvidia-smi smoke failed: {}",
                truncate(&smoke.stderr, 200)
            )));
        }
        Ok(smoke
            .stdout
            .lines()
            .find(|l| l.contains("GPU") || l.contains("NVIDIA") || l.contains("RTX"))
            .unwrap_or("GPU unknown")
            .trim()
            .to_owned())
    }
}

/// First string value found at any of `keys` (top-level object lookups).
fn get_str<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| v.get(k).and_then(|x| x.as_str()))
}

/// First array found at any of `keys` (bare arrays pass through at call sites).
fn get_array(v: &Value, keys: &[&str]) -> Vec<Value> {
    keys.iter()
        .find_map(|k| v.get(k).and_then(|x| x.as_array()))
        .cloned()
        .unwrap_or_default()
}

fn parse_one_offer(item: &Value) -> Option<Offer> {
    let id = get_str(item, &["id", "executor_id"])?.to_owned();
    // Live API uses machine_name + price_per_gpu (not gpu_type / price_per_hour).
    let gpu_type = get_str(item, &["gpu_type", "gpu_name", "machine_name"])
        .or_else(|| item.pointer("/machine/gpu_type").and_then(|x| x.as_str()))
        .unwrap_or("UNKNOWN")
        .to_owned();
    let raw_count = item
        .get("gpu_count")
        .and_then(|x| x.as_u64())
        .or_else(|| item.get("available_gpu_count").and_then(|x| x.as_u64()))
        .or_else(|| item.get("gpus").and_then(|x| x.as_u64()))
        .unwrap_or(1) as u32;
    // Prefer the higher of the numeric field and any `8x` / `x8` label so a
    // lying `gpu_count: 1` on a multi-GPU host cannot pass the single-GPU filter.
    let gpu_count = prism_lium_types::effective_gpu_count(raw_count, &gpu_type);
    let price = item
        .get("price_per_hour")
        .or_else(|| item.get("price_per_gpu"))
        .or_else(|| item.get("price"))
        .or_else(|| item.pointer("/price/per_gpu_hour"))
        .and_then(|x| x.as_f64())
        .or_else(|| {
            get_str(item, &["price_per_hour", "price_per_gpu"]).and_then(|s| s.parse().ok())
        })
        .unwrap_or(f64::MAX);
    Some(Offer {
        id,
        gpu_type,
        gpu_count,
        price_per_hour: price,
        provider: "lium".into(),
    })
}

fn parse_instance(v: &Value, fallback_id: &str) -> Instance {
    Instance {
        id: get_str(v, &["id", "pod_id"])
            .unwrap_or(fallback_id)
            .to_owned(),
        status: get_str(v, &["status", "state"])
            .unwrap_or("UNKNOWN")
            .to_owned(),
        provider: "lium".into(),
        gpu_type: get_str(v, &["gpu_type"])
            .or_else(|| v.pointer("/executor/gpu_type").and_then(|x| x.as_str()))
            .map(str::to_owned),
        ssh_connect_cmd: get_str(v, &["ssh_connect_cmd"]).map(str::to_owned),
    }
}

fn sanitize_err(msg: &str, key: &str) -> String {
    if key.is_empty() {
        msg.to_owned()
    } else {
        msg.replace(key, "<redacted>")
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_owned()
    } else {
        format!("{}…", &s[..n])
    }
}

fn extract_pod_id(v: &Value) -> Option<String> {
    v.get("id")
        .or_else(|| v.get("pod_id"))
        .or_else(|| v.pointer("/pod/id"))
        .and_then(|x| x.as_str())
        .map(str::to_owned)
}

#[async_trait]
impl EvalJobBackend for LiumClient {
    async fn list_offers(&self, max_price_per_hour: Option<f64>) -> Result<Vec<Offer>, LiumError> {
        let v = self
            .request(reqwest::Method::GET, "/executors", None)
            .await?;
        let mut offers = Self::parse_offers(&v);
        if let Some(max) = max_price_per_hour {
            offers.retain(|o| o.price_per_hour <= max);
        }
        debug!(count = offers.len(), "lium list_offers");
        Ok(offers)
    }

    async fn provision(&self, spec: &InstanceSpec) -> Result<Instance, LiumError> {
        Self::validate_spec(spec)?;
        if spec.ssh_public_keys.is_empty() {
            return Err(LiumError::Api(
                "Lium rent requires at least one SSH public key".into(),
            ));
        }

        let key_name = spec
            .ssh_key_name
            .as_deref()
            .unwrap_or("prism-mission-worker");
        for pk in &spec.ssh_public_keys {
            self.ensure_ssh_key(pk, Some(key_name)).await?;
        }

        let mut offers = self.list_offers(Some(spec.max_price_per_hour)).await?;
        // Hard-reject multi-GPU hosts when requesting a single GPU (and exact
        // match otherwise). The old `gpu_count >= requested` filter + rent
        // fallback to `selected.gpu_count` could silently rent 8×5090.
        offers.retain(|o| o.matches_gpu_count(spec.gpu_count));
        let pref = GpuPreference::default_prism();
        offers.sort_by(|a, b| {
            pref.rank(&a.gpu_type)
                .cmp(&pref.rank(&b.gpu_type))
                .then_with(|| {
                    // Prefer true singles (effective count) then cheaper.
                    let ac = prism_lium_types::effective_gpu_count(a.gpu_count, &a.gpu_type);
                    let bc = prism_lium_types::effective_gpu_count(b.gpu_count, &b.gpu_type);
                    ac.cmp(&bc).then_with(|| {
                        a.price_per_hour
                            .partial_cmp(&b.price_per_hour)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                })
        });
        let candidates: Vec<Offer> = match &spec.preferred_offer_id {
            Some(pref_id) => {
                let matched: Vec<Offer> = offers
                    .into_iter()
                    .filter(|o| &o.id == pref_id)
                    .take(1)
                    .collect();
                if matched.is_empty() {
                    return Err(LiumError::Api(format!(
                        "preferred offer {pref_id} missing or multi-GPU rejected"
                    )));
                }
                matched
            }
            None => offers.into_iter().take(10).collect(),
        };
        if candidates.is_empty() {
            return Err(CostGuardrailError::NoCapacity.into());
        }

        let lifetime = spec.max_lifetime_hours.ceil() as u64;
        let template_id = self.resolve_template_id(spec).await?;
        // Never rent more GPUs than requested — even if the marketplace
        // offer advertises a larger host (those are filtered above).
        let rent_gpu_count = spec.gpu_count.max(1);
        let make_body = || {
            serde_json::json!({
                "pod_name": spec.name,
                "user_public_key": spec.ssh_public_keys,
                "termination_hours": lifetime.max(1),
                "gpu_count": rent_gpu_count,
                "template_id": template_id,
            })
        };

        let mut last_err = String::from("no offer tried");
        for selected in &candidates {
            if selected.price_per_hour > spec.max_price_per_hour {
                continue;
            }
            let effective =
                prism_lium_types::effective_gpu_count(selected.gpu_count, &selected.gpu_type);
            info!(
                offer_id = %selected.id,
                gpu = %selected.gpu_type,
                gpu_count = effective,
                rent_gpu_count,
                price = selected.price_per_hour,
                %template_id,
                "lium rent"
            );
            let body = make_body();
            let permit = rent_pool().take().await;
            let rented = self
                .request(
                    reqwest::Method::POST,
                    &format!("/executors/{}/rent", selected.id),
                    Some(&body),
                )
                .await;
            if let Err(e) = &rented {
                let es = e.to_string();
                if lium_rent_pool::is_rate_limited(&es) {
                    permit.rate_limited(&es);
                }
            }
            match rented {
                Ok(v) => {
                    let id = match extract_pod_id(&v) {
                        Some(id) => Some(id),
                        None => self.find_pod_id_by_name(&spec.name).await,
                    };
                    let Some(id) = id else {
                        last_err = "could not determine provisioned pod id from rent".into();
                        self.reclaim_pods_named(&spec.name).await;
                        continue;
                    };
                    match self.wait_until_running(&id).await {
                        Ok(inst) => return Ok(inst),
                        Err(e) => {
                            last_err = format!("offer {} wait_running: {e}", selected.id);
                            self.cleanup_after_rent(&id).await;
                            self.reclaim_pods_named(&spec.name).await;
                        }
                    }
                }
                Err(e) => {
                    last_err = e.to_string();
                    // 429/transport can still leave a PENDING pod under our name.
                    self.reclaim_pods_named(&spec.name).await;
                    // A rent 429 consumed one of Lium's hourly calls. The pool
                    // already recorded the cooldown; return after orphan
                    // cleanup instead of walking more candidates.
                    if lium_rent_pool::is_rate_limited(&last_err) {
                        return Err(LiumError::Api(last_err));
                    }
                }
            }
        }
        self.reclaim_pods_named(&spec.name).await;
        if last_err == "no offer tried" {
            return Err(CostGuardrailError::NoCapacity.into());
        }
        Err(LiumError::Api(last_err))
    }

    async fn terminate(&self, instance_id: &str) -> Result<(), LiumError> {
        let url = format!("{}/pods/{instance_id}", self.base_url);
        let resp = self
            .http
            .delete(&url)
            .send()
            .await
            .map_err(|e| LiumError::Transport(sanitize_err(&e.to_string(), &self.api_key)))?;
        if resp.status().as_u16() == 404 || resp.status().is_success() {
            return Ok(());
        }
        Err(LiumError::Api(format!(
            "DELETE /pods/{instance_id} -> {}",
            resp.status()
        )))
    }

    async fn verify_terminated(&self, instance_id: &str) -> Result<bool, LiumError> {
        Ok(!self
            .list_pods_raw()
            .await?
            .iter()
            .any(|p| p.get("id").and_then(|x| x.as_str()) == Some(instance_id)))
    }

    async fn exec_eval(
        &self,
        instance_id: &str,
        architecture_py: &str,
        training_py: &str,
        tree_blob: Option<&[u8]>,
    ) -> Result<RemoteExecResult, LiumError> {
        self.exec_eval_live(instance_id, architecture_py, training_py, tree_blob)
            .await
    }

    async fn instance_running(&self, instance_id: &str) -> Result<bool, LiumError> {
        self.instance_running_live(instance_id).await
    }

    async fn resume_eval(&self, instance_id: &str) -> Result<RemoteExecResult, LiumError> {
        self.resume_eval_live(instance_id).await
    }

    async fn harvest_logs(&self, instance_id: &str) -> Result<String, LiumError> {
        self.harvest_logs_inner(instance_id).await
    }

    async fn harvest_artifacts(
        &self,
        instance_id: &str,
        dest_dir: &Path,
        _seed: &[u8],
        n_params: Option<u64>,
    ) -> Result<PathBuf, LiumError> {
        let submission_id = dest_dir
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| LiumError::Integrity("park dir missing submission_id".into()))?;
        let target = self.resolve_ssh_target(instance_id).await?;
        let key = resolve_private_key(self.ssh.private_key_path.as_deref())?;
        crate::artifacts::harvest_checkpoint_ssh(
            &target,
            &key,
            dest_dir,
            submission_id,
            self.ssh.ssh_attempts,
            self.ssh.ssh_retry_secs,
            n_params,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::ASSETS_ENV_LOCK;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn cost_guard_refuses_before_network() {
        let c = LiumClient::with_base_url("test-key", "http://127.0.0.1:1").unwrap();
        let spec = InstanceSpec {
            name: "x".into(),
            max_lifetime_hours: 0.0,
            max_price_per_hour: 1.0,
            gpu_count: 1,
            image_digest: None,
            ssh_public_keys: vec!["ssh-ed25519 AAAA".into()],
            ssh_key_name: None,
            preferred_offer_id: None,
            template_id: None,
            template_name: None,
        };
        let err = c.provision(&spec).await.unwrap_err();
        assert!(matches!(
            err,
            LiumError::Cost(CostGuardrailError::LifetimeMissing)
        ));
    }

    #[tokio::test]
    async fn provision_requires_ssh_keys() {
        let c = LiumClient::with_base_url("test-key", "http://127.0.0.1:1").unwrap();
        let mut spec = InstanceSpec {
            name: "x".into(),
            max_lifetime_hours: 1.0,
            max_price_per_hour: 1.0,
            gpu_count: 1,
            image_digest: None,
            ssh_public_keys: vec![],
            ssh_key_name: None,
            preferred_offer_id: None,
            template_id: None,
            template_name: None,
        };
        let err = c.provision(&spec).await.unwrap_err();
        assert!(matches!(err, LiumError::Api(_)));
        let _ = &mut spec;
    }

    #[tokio::test]
    async fn list_offers_filters_price() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/executors"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"id": "a", "gpu_type": "NVIDIA A100", "gpu_count": 1, "price_per_hour": 0.5},
                {"id": "b", "gpu_type": "NVIDIA H100", "gpu_count": 1, "price_per_hour": 5.0}
            ])))
            .mount(&server)
            .await;
        let c = LiumClient::with_base_url("test-key", server.uri()).unwrap();
        let offers = c.list_offers(Some(1.0)).await.unwrap();
        assert_eq!(offers.len(), 1);
        assert_eq!(offers[0].id, "a");
    }

    #[tokio::test]
    async fn request_retries_on_429_then_succeeds() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/executors"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("Retry-After", "0")
                    .set_body_string("Too many requests"),
            )
            .up_to_n_times(1)
            .expect(1..)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/executors"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"id": "a", "gpu_type": "NVIDIA A100", "gpu_count": 1, "price_per_hour": 0.5}
            ])))
            .mount(&server)
            .await;
        let c = LiumClient::with_base_url("test-key", server.uri()).unwrap();
        let offers = c.list_offers(None).await.unwrap();
        assert_eq!(offers.len(), 1);
        assert_eq!(offers[0].id, "a");
    }

    fn provision_spec() -> InstanceSpec {
        InstanceSpec {
            name: "x".into(),
            max_lifetime_hours: 1.0,
            max_price_per_hour: 2.5,
            gpu_count: 1,
            image_digest: None,
            ssh_public_keys: vec!["ssh-ed25519 AAAA".into()],
            ssh_key_name: None,
            preferred_offer_id: None,
            template_id: None,
            template_name: None,
        }
    }

    async fn mount_rent_path(server: &MockServer, offer_id: &str, pod_id: &str) {
        Mock::given(method("POST"))
            .and(path(format!("/executors/{offer_id}/rent")))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"id": pod_id})),
            )
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("/pods/{pod_id}")))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id": pod_id, "status": "RUNNING"})),
            )
            .mount(server)
            .await;
    }

    async fn mount_common(server: &MockServer, offers: serde_json::Value) {
        Mock::given(method("GET"))
            .and(path("/ssh-keys"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
            .mount(server)
            .await;
        Mock::given(method("POST"))
            .and(path("/ssh-keys"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"id": "k"})))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/templates"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(
                    serde_json::json!([{"name": RECIPES_TEMPLATE_NAME, "id": "tmpl1"}]),
                ),
            )
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/executors"))
            .respond_with(ResponseTemplate::new(200).set_body_json(offers))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn provision_prefers_5090_over_cheaper_a100() {
        let server = MockServer::start().await;
        mount_common(
            &server,
            serde_json::json!([
                {"id": "cheap-a100", "gpu_type": "NVIDIA A100-SXM4-80GB", "gpu_count": 1, "price_per_hour": 0.5},
                {"id": "pin-5090", "gpu_type": "NVIDIA GeForce RTX 5090", "gpu_count": 1, "price_per_hour": 2.0},
                {"id": "mid-h100", "gpu_type": "NVIDIA H100", "gpu_count": 1, "price_per_hour": 1.5}
            ]),
        )
        .await;
        // Every candidate rents fine: the returned pod proves which offer the
        // sort put first (5090 despite being the priciest within the cap).
        mount_rent_path(&server, "pin-5090", "pod-5090").await;
        mount_rent_path(&server, "cheap-a100", "pod-a100").await;
        mount_rent_path(&server, "mid-h100", "pod-h100").await;
        let c = LiumClient::with_base_url("test-key", server.uri()).unwrap();
        let inst = c.provision(&provision_spec()).await.unwrap();
        assert_eq!(inst.id, "pod-5090");
    }

    #[tokio::test]
    async fn provision_never_selects_8x_5090_when_1x_available() {
        let server = MockServer::start().await;
        mount_common(
            &server,
            serde_json::json!([
                {"id": "eight-5090", "gpu_type": "NVIDIA GeForce RTX 5090", "gpu_count": 8, "price_per_hour": 0.48},
                {"id": "eight-label", "gpu_type": "8x RTX 5090", "gpu_count": 1, "price_per_hour": 0.45},
                {"id": "one-5090", "gpu_type": "NVIDIA GeForce RTX 5090", "gpu_count": 1, "price_per_hour": 2.0}
            ]),
        )
        .await;
        mount_rent_path(&server, "one-5090", "pod-1x").await;
        // If multi-GPU slipped through, these would be rented instead.
        mount_rent_path(&server, "eight-5090", "pod-8x").await;
        mount_rent_path(&server, "eight-label", "pod-8x-label").await;
        let c = LiumClient::with_base_url("test-key", server.uri()).unwrap();
        let inst = c.provision(&provision_spec()).await.unwrap();
        assert_eq!(inst.id, "pod-1x");
    }

    #[tokio::test]
    async fn provision_rejects_all_multi_gpu_offers() {
        let server = MockServer::start().await;
        mount_common(
            &server,
            serde_json::json!([
                {"id": "eight-5090", "gpu_type": "NVIDIA GeForce RTX 5090", "gpu_count": 8, "price_per_hour": 0.48},
                {"id": "eight-label", "gpu_type": "8× RTX 5090", "gpu_count": 1, "price_per_hour": 0.45}
            ]),
        )
        .await;
        mount_rent_path(&server, "eight-5090", "pod-8x").await;
        mount_rent_path(&server, "eight-label", "pod-8x-label").await;
        let c = LiumClient::with_base_url("test-key", server.uri()).unwrap();
        let err = c.provision(&provision_spec()).await.unwrap_err();
        assert!(
            matches!(err, LiumError::Cost(CostGuardrailError::NoCapacity))
                || err.to_string().contains("NoCapacity")
                || err.to_string().to_ascii_lowercase().contains("capacity"),
            "got {err}"
        );
    }

    #[tokio::test]
    async fn provision_falls_back_in_declared_order_when_no_5090() {
        let server = MockServer::start().await;
        mount_common(
            &server,
            serde_json::json!([
                {"id": "cheap-a100", "gpu_type": "NVIDIA A100-SXM4-80GB", "gpu_count": 1, "price_per_hour": 0.5},
                {"id": "mid-h100", "gpu_type": "NVIDIA H100", "gpu_count": 1, "price_per_hour": 1.5},
                {"id": "weak-l4", "gpu_type": "NVIDIA L4", "gpu_count": 1, "price_per_hour": 0.2}
            ]),
        )
        .await;
        mount_rent_path(&server, "cheap-a100", "pod-a100").await;
        mount_rent_path(&server, "mid-h100", "pod-h100").await;
        mount_rent_path(&server, "weak-l4", "pod-l4").await;
        let c = LiumClient::with_base_url("test-key", server.uri()).unwrap();
        let inst = c.provision(&provision_spec()).await.unwrap();
        assert_eq!(inst.id, "pod-h100");
    }

    #[tokio::test]
    async fn provision_reclaims_orphan_pod_on_429() {
        // Rent 429 must still terminate any PENDING pod under our name
        // (dd4343ae backoff alone left orphans billing).
        let server = MockServer::start().await;
        mount_common(
            &server,
            serde_json::json!([{
                "id": "only",
                "gpu_type": "NVIDIA GeForce RTX 5090",
                "gpu_count": 1,
                "price_per_hour": 1.0
            }]),
        )
        .await;
        Mock::given(method("POST"))
            .and(path("/executors/only/rent"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/pods"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"id": "orphan-1", "pod_name": "x", "status": "PENDING"}
            ])))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/pods"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
            .mount(&server)
            .await;
        let _delete = Mock::given(method("DELETE"))
            .and(path("/pods/orphan-1"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount_as_scoped(&server)
            .await;
        let c = LiumClient::with_base_url("test-key", server.uri()).unwrap();
        let err = c.provision(&provision_spec()).await.unwrap_err();
        assert!(
            err.to_string().contains("429")
                || err.to_string().to_ascii_lowercase().contains("rate"),
            "got {err}"
        );
    }

    #[test]
    fn harness_env_pairs_forward_numeric_test_values_only() {
        std::env::set_var("PRISM_TEST_TRAIN_MINUTES", "15");
        std::env::set_var("PRISM_TEST_MAX_PARAMS", "2000000");
        std::env::set_var("PRISM_TEST_EVAL_CAPS", "0");
        std::env::set_var("PRISM_EVAL_G5_N_ITEMS", "1");
        std::env::set_var("PRISM_FLOW", "v3");
        let pairs = harness_env_pairs(6.0, "NVIDIA GeForce RTX 5090", false);
        assert!(pairs
            .iter()
            .any(|(k, v)| *k == "PRISM_TEST_TRAIN_MINUTES" && v == "15"));
        assert!(pairs
            .iter()
            .any(|(k, v)| *k == "PRISM_TEST_MAX_PARAMS" && v == "2000000"));
        assert!(pairs
            .iter()
            .any(|(k, v)| *k == "PRISM_TEST_EVAL_CAPS" && v == "0"));
        assert!(pairs
            .iter()
            .any(|(k, v)| *k == "PRISM_EVAL_G5_N_ITEMS" && v == "1"));
        assert!(pairs.iter().any(|(k, v)| *k == "PRISM_FLOW" && v == "v3"));
        std::env::set_var("PRISM_TEST_TRAIN_MINUTES", "15'; rm -rf /; '");
        let pairs = harness_env_pairs(6.0, "NVIDIA GeForce RTX 5090", false);
        assert!(!pairs.iter().any(|(_, v)| v.contains("rm -rf")));
        // Reject non-allowlisted flow tokens (injection / typo surface).
        std::env::set_var("PRISM_FLOW", "v3; rm -rf /");
        let pairs = harness_env_pairs(6.0, "NVIDIA GeForce RTX 5090", false);
        assert!(!pairs.iter().any(|(k, _)| *k == "PRISM_FLOW"));
        std::env::remove_var("PRISM_TEST_TRAIN_MINUTES");
        std::env::remove_var("PRISM_TEST_MAX_PARAMS");
        std::env::remove_var("PRISM_TEST_EVAL_CAPS");
        std::env::remove_var("PRISM_EVAL_G5_N_ITEMS");
        std::env::remove_var("PRISM_FLOW");
        let pairs = harness_env_pairs(6.0, "NVIDIA GeForce RTX 5090' OR '1", false);
        assert!(pairs
            .iter()
            .any(|(k, v)| *k == "PRISM_GPU_TYPE" && v == "NVIDIA GeForce RTX 5090 OR 1"));
        // assets_pending advertises the pod dir; the default omits it.
        assert!(!pairs.iter().any(|(k, _)| *k == "PRISM_EVAL_ASSETS_DIR"));
        let pairs = harness_env_pairs(6.0, "SIM", true);
        assert!(pairs
            .iter()
            .any(|(k, v)| *k == "PRISM_EVAL_ASSETS_DIR" && v == "/tmp/prism_eval/eval-assets"));
    }

    #[test]
    fn train_done_marker_match_is_exact_line_only() {
        // Miner stdout is relayed with the `[harness] v3| ` prefix, so a
        // forged marker inside miner output must NOT match.
        let m = TRAIN_DONE_MARKER;
        assert!("[harness] v3| PHASE_TRAIN_DONE".trim_end() != m);
        assert!(" PHASE_TRAIN_DONE".trim_end() != m);
        assert_eq!("PHASE_TRAIN_DONE".trim_end(), m);
        assert_eq!("PHASE_TRAIN_DONE\r".trim_end(), m);
    }

    #[test]
    fn eval_assets_dir_requires_existing_dir() {
        let _guard = ASSETS_ENV_LOCK.lock().unwrap();
        std::env::remove_var("PRISM_EVAL_ASSETS_DIR");
        assert!(eval_assets_dir().is_none());
        std::env::set_var("PRISM_EVAL_ASSETS_DIR", "/definitely/not/a/dir");
        assert!(eval_assets_dir().is_none());
        let dir = std::env::temp_dir().join(format!("prism-assets-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("PRISM_EVAL_ASSETS_DIR", &dir);
        assert_eq!(eval_assets_dir().as_deref(), Some(dir.as_path()));
        std::env::remove_var("PRISM_EVAL_ASSETS_DIR");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn debug_redacts_key() {
        let c = LiumClient::with_base_url("super-secret-key-xyz", "http://example").unwrap();
        let s = format!("{c:?}");
        assert!(!s.contains("super-secret"));
        assert!(s.contains("<redacted>"));
    }

    /// Minimal ustar reader for assertions: validates magic + checksum per
    /// header and returns the regular-file entries `(path, contents)`.
    fn parse_tar_files(tar: &[u8]) -> Vec<(String, Vec<u8>)> {
        let mut out = Vec::new();
        let mut off = 0;
        while off + 512 <= tar.len() {
            let h = &tar[off..off + 512];
            if h.iter().all(|&b| b == 0) {
                break;
            }
            assert_eq!(&h[257..262], b"ustar", "bad ustar magic");
            let sum: usize = h
                .iter()
                .enumerate()
                .map(|(i, &b)| {
                    if (148..156).contains(&i) {
                        32
                    } else {
                        b as usize
                    }
                })
                .sum();
            let stored =
                usize::from_str_radix(std::str::from_utf8(&h[148..154]).unwrap(), 8).unwrap();
            assert_eq!(stored, sum, "ustar checksum mismatch");
            let end = h[..100].iter().position(|&b| b == 0).unwrap_or(100);
            let name = std::str::from_utf8(&h[..end]).unwrap().to_owned();
            let size = usize::from_str_radix(
                std::str::from_utf8(&h[124..135])
                    .unwrap()
                    .trim_end_matches('\0'),
                8,
            )
            .unwrap();
            assert_eq!(h[156], b'0', "only regular files are archived");
            off += 512;
            out.push((name, tar[off..off + size].to_vec()));
            off += size.div_ceil(512) * 512;
        }
        out
    }

    #[test]
    fn harness_upload_tar_has_exact_files_with_identical_contents() {
        let files = parse_tar_files(&harness_upload_tar("# arch", "# train", None).unwrap());
        let got: std::collections::BTreeMap<&str, &[u8]> = files
            .iter()
            .map(|(p, c)| (p.as_str(), c.as_slice()))
            .collect();
        assert_eq!(got.len(), prism_recipe::HARNESS_FILES.len() + 2);
        for (path, contents) in prism_recipe::HARNESS_FILES {
            assert_eq!(
                got.get(path).copied(),
                Some(contents.as_bytes()),
                "tar entry {path} mismatch"
            );
        }
        assert_eq!(
            got.get("architecture.py").copied(),
            Some(b"# arch".as_slice())
        );
        assert_eq!(got.get("training.py").copied(), Some(b"# train".as_slice()));
        // Pod layout anchors at the workdir root (byte-identical to the old
        // base64 upload): entrypoint + package dirs + banned-pattern list.
        assert!(got.contains_key("main.py"));
        assert!(got.contains_key("cheatguard_patterns.json"));
        assert!(got.keys().any(|p| p.starts_with("prismlib/")));
        assert!(got.keys().any(|p| p.starts_with("eval/")));
    }

    #[test]
    fn natural_packs_ride_the_post_train_assets_stage() {
        // `stage_eval_assets` tars the operator assets dir wholesale, so the
        // G5 natural packs need no transport of their own — but they must be
        // addressed relative to that dir, and the on-pod adapter has to
        // resolve the very same relative path.
        let rel = std::path::Path::new(prism_recipe::NATURAL_PACK_REL);
        assert!(rel.is_relative(), "pack path must be assets-dir relative");
        let staged = std::path::Path::new(EVAL_ASSETS_POD_DIR).join(rel);
        assert!(staged.starts_with(EVAL_ASSETS_POD_DIR));
        // The harness resolves the same relative path on the pod.
        let adapter = prism_recipe::HARNESS_FILES
            .iter()
            .find(|(p, _)| *p == "eval/natural_docs.py")
            .map(|(_, c)| *c)
            .expect("natural_docs.py is part of the harness package");
        assert!(adapter.contains(prism_recipe::NATURAL_PACK_REL));
    }

    #[test]
    fn harness_upload_tar_is_byte_deterministic() {
        let a = harness_upload_tar("arch", "train", None).unwrap();
        let b = harness_upload_tar("arch", "train", None).unwrap();
        assert_eq!(a, b, "two builds must produce identical bytes");
        assert!(a.len() > 100_000, "the real harness set is large");
    }

    #[test]
    fn harness_upload_tar_stages_full_source_tree_under_submission() {
        let mut files = std::collections::BTreeMap::new();
        files.insert("architecture.py".into(), b"import kernels\n".to_vec());
        files.insert(
            "training.py".into(),
            b"def train(m,c):\n    return {}\n".to_vec(),
        );
        files.insert(
            "kernels/flash.py".into(),
            b"def attend():\n    return 1\n".to_vec(),
        );
        files.insert("tokenizer/tokenizer.json".into(), b"{}".to_vec());
        let blob = prism_tree::StagedTree::new(files, "training.py".into())
            .pack()
            .unwrap();
        let tar = harness_upload_tar("ignored", "ignored", Some(&blob)).unwrap();
        let got = parse_tar_files(&tar);
        let names: std::collections::BTreeSet<_> = got.iter().map(|(p, _)| p.as_str()).collect();
        assert!(names.contains("submission/kernels/flash.py"));
        assert!(names.contains("submission/tokenizer/tokenizer.json"));
        assert!(names.contains("main.py"));
        assert!(!names.contains("architecture.py"));
    }

    #[test]
    fn ssh_argv_stays_small_regardless_of_harness_size() {
        // BUG-5 regression: the harness payload (~240 KB as base64) blew the
        // 128 KB MAX_ARG_STRLEN when embedded in the remote command. It must
        // ride ssh stdin as a tar; argv holds only fixed-size commands.
        use std::fmt::Write as _;
        assert!(HARNESS_EXTRACT_CMD.len() < 1024);
        let mut env = String::new();
        for (k, v) in harness_env_pairs(6.0, "NVIDIA GeForce RTX 5090", true) {
            let _ = writeln!(env, "export {k}='{v}'");
        }
        let remote = format!("{HARNESS_BOOTSTRAP}{}", detach_launch_cmd(&env, 25200));
        assert!(
            remote.len() < 8 * 1024,
            "run argv is {} bytes",
            remote.len()
        );
        assert!(remote.contains("setsid"));
        assert!(!remote.contains("base64"), "no payload embedding in argv");
        // The stdin payload scales with miner size; argv never does.
        let big = harness_upload_tar(&"a".repeat(200_000), &"t".repeat(200_000), None).unwrap();
        assert!(big.len() > 400_000);
        assert!(HARNESS_EXTRACT_CMD.len() < 1024);
    }

    #[test]
    fn random_seed_hex_reads_exactly_16_bytes() {
        // Regression: fs::read("/dev/urandom") blocks forever on a char
        // device (no EOF) — the seed path must complete with 16 bytes.
        let s = random_seed_hex().unwrap();
        assert_eq!(s.len(), 32);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
