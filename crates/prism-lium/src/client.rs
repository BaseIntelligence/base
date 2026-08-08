//! Real Lium HTTPS client + SSH-backed live eval.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;
use tokio::time::sleep;
use tracing::{debug, info, warn};

use crate::error::{CostGuardrailError, LiumError};
use crate::ssh::{
    parse_ssh_target, resolve_private_key, ssh_exec, ssh_exec_allow_fail, truncate_tail, SshTarget,
};
use crate::types::{GpuPreference, Instance, InstanceSpec, LiumSshConfig, Offer, RemoteExecResult};
use crate::{EvalJobBackend, HARNESS_LOG_RETAIN_BYTES, LIUM_API_BASE_URL, MIN_LIFETIME_HOURS};

/// Pod image: Lium-owned DinD variant pulses its own dockerd init and never
/// Only `daturaai/*-dind` pods deliver a reachable sshd on this marketplace
/// (vanilla pytorch has none; cu12.8-dinD needs `service ssh start`, whose
/// process-exit kills sshd when the startup job ends — metachar rejection
/// makes keep-alive chains impossible). The cuda**13.0.2**-dinD tag runs sshd
/// from its own init: sleeps keep ssh alive across verify + exec phases.
const RECIPES_TEMPLATE_IMAGE: &str = "daturaai/pytorch";
const RECIPES_TEMPLATE_TAG: &str = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind";
/// Recipe template ns (v9: the cu13 DinD shape that ssh-verifies with an
/// EMPTY startup script; proven ssh-stable ≥7 min on a 5070 Ti pod).
const RECIPES_TEMPLATE_NAME: &str = "prism-recipe-v9";
/// Boot: nothing — sshd comes up by itself in the cu13 dinD image.
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

/// Async Lium REST client. API key only in `X-API-Key` header.
pub struct LiumClient {
    http: reqwest::Client,
    base_url: String,
    /// Stored but never Debug/Display'd.
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
    /// Build a client. `api_key` must be non-empty.
    ///
    /// # Errors
    /// Empty key or HTTP client build failure.
    pub fn new(api_key: impl Into<String>) -> Result<Self, LiumError> {
        Self::with_base_url(api_key, LIUM_API_BASE_URL)
    }

    /// Build with custom base URL (tests / wiremock).
    ///
    /// # Errors
    /// Empty key or HTTP client build failure.
    pub fn with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Result<Self, LiumError> {
        Self::with_config(api_key, base_url, LiumSshConfig::default_live())
    }

    /// Build with SSH config for live eval.
    ///
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
        // Lium edge WAF returns 403 for empty/missing User-Agent.
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

    /// Never expose key.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Override private key path after construction.
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

    async fn request_json(&self, method: reqwest::Method, path: &str) -> Result<Value, LiumError> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .request(method.clone(), &url)
            .send()
            .await
            .map_err(|e| LiumError::Transport(sanitize_err(&e.to_string(), &self.api_key)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| LiumError::Transport(sanitize_err(&e.to_string(), &self.api_key)))?;
        if !status.is_success() {
            return Err(LiumError::Api(format!(
                "{method} {path} -> {status}: {}",
                truncate(&sanitize_err(&text, &self.api_key), 200)
            )));
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| LiumError::Api(format!("json: {e}")))
    }

    async fn request_json_body(
        &self,
        method: reqwest::Method,
        path: &str,
        body: &Value,
    ) -> Result<Value, LiumError> {
        let url = format!("{}{path}", self.base_url);
        let resp = self
            .http
            .request(method.clone(), &url)
            .json(body)
            .send()
            .await
            .map_err(|e| LiumError::Transport(sanitize_err(&e.to_string(), &self.api_key)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| LiumError::Transport(sanitize_err(&e.to_string(), &self.api_key)))?;
        if !status.is_success() {
            return Err(LiumError::Api(format!(
                "{method} {path} -> {status}: {}",
                truncate(&sanitize_err(&text, &self.api_key), 200)
            )));
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| LiumError::Api(format!("json: {e}")))
    }

    /// Parse executors list into offers.
    fn parse_offers(v: &Value) -> Vec<Offer> {
        let items = v
            .as_array()
            .cloned()
            .or_else(|| v.get("executors").and_then(|x| x.as_array().cloned()))
            .or_else(|| v.get("data").and_then(|x| x.as_array().cloned()))
            .unwrap_or_default();
        let mut out = Vec::new();
        for item in items {
            if let Some(o) = parse_one_offer(&item) {
                out.push(o);
            }
        }
        out
    }

    async fn list_pods_raw(&self) -> Result<Vec<Value>, LiumError> {
        let v = self.request_json(reqwest::Method::GET, "/pods").await?;
        Ok(v.as_array()
            .cloned()
            .or_else(|| v.get("pods").and_then(|x| x.as_array().cloned()))
            .unwrap_or_default())
    }

    /// GET /pods/{id} raw JSON.
    pub async fn get_pod_raw(&self, instance_id: &str) -> Result<Value, LiumError> {
        self.request_json(reqwest::Method::GET, &format!("/pods/{instance_id}"))
            .await
    }

    /// Parsed instance status.
    pub async fn status(&self, instance_id: &str) -> Result<Instance, LiumError> {
        let v = self.get_pod_raw(instance_id).await?;
        Ok(parse_instance(&v, instance_id))
    }

    /// Ensure SSH public key is registered with Lium (idempotent).
    pub async fn ensure_ssh_key(
        &self,
        public_key: &str,
        name: Option<&str>,
    ) -> Result<Value, LiumError> {
        let normalized = public_key.trim();
        let v = self.request_json(reqwest::Method::GET, "/ssh-keys").await?;
        let keys = v
            .as_array()
            .cloned()
            .or_else(|| v.get("ssh_keys").and_then(|x| x.as_array().cloned()))
            .unwrap_or_default();
        for key in keys {
            let pk = key
                .get("public_key")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .trim();
            if pk == normalized {
                return Ok(key);
            }
        }
        let mut body = serde_json::json!({ "public_key": public_key });
        if let Some(n) = name {
            body["name"] = Value::String(n.to_owned());
        }
        self.request_json_body(reqwest::Method::POST, "/ssh-keys", &body)
            .await
    }

    /// Ensure a named template exists; return its id (idempotent).
    pub async fn ensure_template(
        &self,
        name: &str,
        docker_image: &str,
        docker_image_tag: Option<&str>,
        startup_commands: Option<&str>,
    ) -> Result<String, LiumError> {
        let v = self
            .request_json(reqwest::Method::GET, "/templates")
            .await?;
        let templates = v
            .as_array()
            .cloned()
            .or_else(|| v.get("templates").and_then(|x| x.as_array().cloned()))
            .unwrap_or_default();
        for tmpl in templates {
            let n = tmpl.get("name").and_then(|x| x.as_str()).unwrap_or("");
            if n == name {
                if let Some(id) = tmpl.get("id").and_then(|x| x.as_str()) {
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
        // Metachar-free keep-alive; Lium rejects shell metachar startup chains.
        if let Some(cmd) = startup_commands {
            body["startup_commands"] = serde_json::Value::String(cmd.to_owned());
        }
        let created = self
            .request_json_body(reqwest::Method::POST, "/templates", &body)
            .await?;
        created
            .get("id")
            .and_then(|x| x.as_str())
            .map(str::to_owned)
            .ok_or_else(|| LiumError::Api("template create missing id".into()))
    }

    /// Resolve template id from spec (explicit id, name, or default e2e template).
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
        let v = self.request_json(reqwest::Method::GET, "/users/me").await?;
        v.get("balance")
            .and_then(|x| x.as_f64())
            .or_else(|| {
                v.get("balance")
                    .and_then(|x| x.as_str())
                    .and_then(|s| s.parse().ok())
            })
            .ok_or_else(|| LiumError::Api("users/me missing balance".into()))
    }

    /// Poll until RUNNING (or fail).
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

    async fn cleanup_after_rent(&self, pod_id: Option<&str>) {
        if let Some(id) = pod_id {
            if let Err(e) = self.terminate(id).await {
                warn!(error = %e, "lium cleanup terminate failed");
            }
            let _ = self.verify_terminated(id).await;
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

    /// Live recipe eval: wait RUNNING → SSH nvidia-smi → upload harness +
    /// miner sources → run [`prism_recipe`] harness → parse `METRICS_JSON`.
    ///
    /// The harness trains the miner's code on the pinned fineweb-edu shard and
    /// scores the frozen val cut; no metric comes from hashing sources.
    async fn exec_eval_live(
        &self,
        instance_id: &str,
        architecture_py: &str,
        training_py: &str,
    ) -> Result<RemoteExecResult, LiumError> {
        let _running = self.wait_until_running(instance_id).await?;
        let target = self.resolve_ssh_target(instance_id).await?;
        let key = resolve_private_key(self.ssh.private_key_path.as_deref())?;

        let gpu_type = self.gpu_smoke(&target, &key).await?;

        let arch_b64 = base64_encode(architecture_py.as_bytes());
        let train_b64 = base64_encode(training_py.as_bytes());
        let harness_b64 = base64_encode(prism_recipe::HARNESS_PY.as_bytes());
        let train_cap_secs = (self.ssh.train_hours_cap * 3600.0) as u64;
        let remote = format!(
            "set -e
command -v pip >/dev/null 2>&1 || apt-get update -q
command -v pip >/dev/null 2>&1 || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-pip
python3 -c 'import torch' 2>/dev/null || echo 'torch stopping'
python3 -c 'import transformers' 2>/dev/null || pip install \
  --break-system-packages --root-user-action=ignore \
  'transformers==4.44.2' 'datasets==3.0.2' 'pyarrow==17.0.0'
mkdir -p /tmp/prism_eval
echo '{harness_b64}' | base64 -d > /tmp/prism_eval/prism_harness.py
echo '{arch_b64}' | base64 -d > /tmp/prism_eval/architecture.py
echo '{train_b64}' | base64 -d > /tmp/prism_eval/training.py
cd /tmp/prism_eval
# Persist full harness output on-pod so timeout / stuck-sweep can harvest the
# fatal tail even when the long-lived SSH session is killed without pipes.
set +e
PRISM_DATASET_URL='{dataset_url}' \
PRISM_DATASET_SHA256='{dataset_sha}' \
PRISM_MAX_TRAIN_STEPS='{steps}' \
PRISM_TRAIN_HOURS_CAP='{train_hours}' \
PRISM_GPU_TYPE='{gpu_type}' \
{test_env}timeout --kill-after=60 {timeout_secs} python3 prism_harness.py \
  > /tmp/prism_eval/harness.log 2>&1
ec=$?
set -e
# Surface the log tail on the SSH channel for the happy path + failure parse.
tail -c 524288 /tmp/prism_eval/harness.log || true
exit $ec\n",
            harness_b64 = harness_b64,
            arch_b64 = arch_b64,
            train_b64 = train_b64,
            dataset_url = prism_recipe::DATASET_URL,
            dataset_sha = prism_recipe::dataset_sha256(),
            steps = prism_recipe::MAX_TRAIN_STEPS,
            train_hours = self.ssh.train_hours_cap,
            gpu_type = gpu_type.replace('\'', ""),
            test_env = test_mode_env(),
            timeout_secs = train_cap_secs.saturating_add(3600),
        );

        let out = match ssh_exec_allow_fail(
            &target,
            &key,
            &remote,
            1,
            self.ssh.ssh_retry_secs,
            train_cap_secs.saturating_add(3900),
        )
        .await
        {
            Ok(o) => o,
            Err(e) => {
                // Session timed out / dropped — second SSH pulls the on-pod log.
                let harvested = self
                    .harvest_logs_inner(instance_id)
                    .await
                    .unwrap_or_default();
                return Err(LiumError::Exec(format!(
                    "harness transport: {e}; harvested: {}",
                    truncate_tail(&harvested, HARNESS_LOG_RETAIN_BYTES)
                )));
            }
        };
        if !out.stdout.contains("EVAL_OK") {
            let mut detail = out.stdout.clone();
            if !out.stderr.is_empty() {
                detail.push_str("\n--- stderr ---\n");
                detail.push_str(&out.stderr);
            }
            if detail.trim().is_empty() {
                detail = self
                    .harvest_logs_inner(instance_id)
                    .await
                    .unwrap_or_default();
            }
            return Err(LiumError::Exec(format!(
                "harness failed (code {}): {}",
                out.returncode,
                truncate_tail(&detail, HARNESS_LOG_RETAIN_BYTES)
            )));
        }
        let line = out
            .stdout
            .lines()
            .find(|l| l.starts_with("METRICS_JSON="))
            .ok_or_else(|| LiumError::Exec("harness EVAL_OK without METRICS_JSON".into()))?;
        let v: RemoteExecResult = serde_json::from_str(&line["METRICS_JSON=".len()..])
            .map_err(|e| LiumError::Exec(format!("metrics json: {e}")))?;
        if !v.bpb.is_finite() || v.bpb <= 0.0 {
            return Err(LiumError::Exec(format!(
                "harness bpb not finite: {}",
                v.bpb
            )));
        }
        Ok(v)
    }

    /// SSH-fetch the on-pod harness log tail (empty when missing / unreachable).
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
        let smoke = ssh_exec(
            target,
            key,
            "nvidia-smi -L && echo SMOKE_OK",
            self.ssh.ssh_attempts,
            self.ssh.ssh_retry_secs,
            60,
        )
        .await?;
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

/// Optional test-mode env lines for the pod harness (`KEY='v' \\\n` pairs).
///
/// Forwards `PRISM_TEST_TRAIN_MINUTES` / `PRISM_TEST_MAX_PARAMS` from the
/// master process env so staging can run short/tiny evals. Values are
/// forwarded only when they parse as plain numerics — the remote string is
/// shell, so anything else would be an injection vector.
fn test_mode_env() -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    if let Ok(v) = std::env::var("PRISM_TEST_TRAIN_MINUTES") {
        if v.trim().parse::<f64>().is_ok() {
            let _ = writeln!(out, "PRISM_TEST_TRAIN_MINUTES='{}' \\", v.trim());
        }
    }
    if let Ok(v) = std::env::var("PRISM_TEST_MAX_PARAMS") {
        if v.trim().parse::<u64>().is_ok() {
            let _ = writeln!(out, "PRISM_TEST_MAX_PARAMS='{}' \\", v.trim());
        }
    }
    out
}

fn base64_encode(bytes: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(T[((n >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(T[(n & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn parse_one_offer(item: &Value) -> Option<Offer> {
    let id = item
        .get("id")
        .or_else(|| item.get("executor_id"))
        .and_then(|x| x.as_str())?
        .to_owned();
    // Live API uses machine_name + price_per_gpu (not gpu_type / price_per_hour).
    let gpu_type = item
        .get("gpu_type")
        .or_else(|| item.get("gpu_name"))
        .or_else(|| item.get("machine_name"))
        .or_else(|| item.pointer("/machine/gpu_type"))
        .and_then(|x| x.as_str())
        .unwrap_or("UNKNOWN")
        .to_owned();
    let gpu_count = item
        .get("gpu_count")
        .and_then(|x| x.as_u64())
        .or_else(|| item.get("available_gpu_count").and_then(|x| x.as_u64()))
        .or_else(|| item.get("gpus").and_then(|x| x.as_u64()))
        .unwrap_or(1) as u32;
    let price = item
        .get("price_per_hour")
        .or_else(|| item.get("price_per_gpu"))
        .or_else(|| item.get("price"))
        .or_else(|| item.pointer("/price/per_gpu_hour"))
        .and_then(|x| x.as_f64())
        .or_else(|| {
            item.get("price_per_hour")
                .or_else(|| item.get("price_per_gpu"))
                .and_then(|x| x.as_str())
                .and_then(|s| s.parse().ok())
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
    let id = v
        .get("id")
        .or_else(|| v.get("pod_id"))
        .and_then(|x| x.as_str())
        .unwrap_or(fallback_id)
        .to_owned();
    let status = v
        .get("status")
        .or_else(|| v.get("state"))
        .and_then(|x| x.as_str())
        .unwrap_or("UNKNOWN")
        .to_owned();
    let gpu_type = v
        .get("gpu_type")
        .or_else(|| v.pointer("/executor/gpu_type"))
        .and_then(|x| x.as_str())
        .map(str::to_owned);
    let ssh_connect_cmd = v
        .get("ssh_connect_cmd")
        .and_then(|x| x.as_str())
        .map(str::to_owned);
    Instance {
        id,
        status,
        provider: "lium".into(),
        gpu_type,
        ssh_connect_cmd,
    }
}

fn sanitize_err(msg: &str, key: &str) -> String {
    if key.is_empty() {
        return msg.to_owned();
    }
    msg.replace(key, "<redacted>")
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
            .request_json(reqwest::Method::GET, "/executors")
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
        // GPU pin first (RTX 5090 → ordered fallback), price only breaks ties
        // within the same rank: cheapest-first would silently drift evals
        // onto weaker GPUs whenever the pinned SKU is not the cheapest offer.
        let pref = GpuPreference::default_prism();
        offers.sort_by(|a, b| {
            pref.rank(&a.gpu_type)
                .cmp(&pref.rank(&b.gpu_type))
                .then_with(|| {
                    a.price_per_hour
                        .partial_cmp(&b.price_per_hour)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let candidates: Vec<Offer> = match &spec.preferred_offer_id {
            Some(pref) => offers
                .into_iter()
                .filter(|o| &o.id == pref)
                .take(1)
                .collect(),
            None => offers
                .into_iter()
                .filter(|o| o.gpu_count >= spec.gpu_count)
                .take(10)
                .collect(),
        };
        if candidates.is_empty() {
            return Err(CostGuardrailError::NoCapacity.into());
        }

        let lifetime = spec.max_lifetime_hours.ceil() as u64;
        let template_id = self.resolve_template_id(spec).await?;
        let make_body = |gpu_count: u32| {
            serde_json::json!({
                "pod_name": spec.name,
                "user_public_key": spec.ssh_public_keys,
                "termination_hours": lifetime.max(1),
                "gpu_count": gpu_count,
                "template_id": template_id,
            })
        };

        let mut last_err = String::from("no offer tried");
        for selected in &candidates {
            if selected.price_per_hour > spec.max_price_per_hour {
                continue;
            }
            info!(
                offer_id = %selected.id,
                gpu = %selected.gpu_type,
                price = selected.price_per_hour,
                %template_id,
                "lium rent"
            );
            // Try the requested split first; on "no GPU splitting allowed"
            // retry the whole node (per-GPU price is unchanged).
            let mut split_choices = vec![spec.gpu_count];
            if selected.gpu_count != spec.gpu_count {
                split_choices.push(selected.gpu_count);
            }
            let mut rented: Result<Value, LiumError> = Err(LiumError::Api("unrented".into()));
            for gcount in split_choices {
                rented = self
                    .request_json_body(
                        reqwest::Method::POST,
                        &format!("/executors/{}/rent", selected.id),
                        &make_body(gcount),
                    )
                    .await;
                if let Err(e) = &rented {
                    if e.to_string().contains("splitting") {
                        continue;
                    }
                }
                break;
            }
            let mut pod_id: Option<String> = None;
            match rented {
                Ok(v) => {
                    pod_id = extract_pod_id(&v);
                    if pod_id.is_none() {
                        if let Ok(pods) = self.list_pods_raw().await {
                            for p in pods {
                                let name = p
                                    .get("pod_name")
                                    .or_else(|| p.get("name"))
                                    .and_then(|x| x.as_str())
                                    .unwrap_or("");
                                if name == spec.name {
                                    if let Some(id) = p.get("id").and_then(|x| x.as_str()) {
                                        pod_id = Some(id.to_owned());
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    let Some(id) = pod_id.clone() else {
                        last_err = "could not determine provisioned pod id from rent".into();
                        self.cleanup_after_rent(None).await;
                        continue;
                    };
                    // Wait for RUNNING here so a CREATION_FAILED offer falls
                    // through to the next candidate instead of poisoning exec.
                    match self.wait_until_running(&id).await {
                        Ok(inst) => return Ok(inst),
                        Err(e) => {
                            last_err = format!("offer {} wait_running: {e}", selected.id);
                            self.cleanup_after_rent(Some(&id)).await;
                        }
                    }
                }
                Err(e) => {
                    last_err = e.to_string();
                    self.cleanup_after_rent(pod_id.as_deref()).await;
                }
            }
        }
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
        let pods = self.list_pods_raw().await?;
        for p in pods {
            if p.get("id").and_then(|x| x.as_str()) == Some(instance_id) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    async fn exec_eval(
        &self,
        instance_id: &str,
        architecture_py: &str,
        training_py: &str,
    ) -> Result<RemoteExecResult, LiumError> {
        self.exec_eval_live(instance_id, architecture_py, training_py)
            .await
    }

    async fn harvest_logs(&self, instance_id: &str) -> Result<String, LiumError> {
        self.harvest_logs_inner(instance_id).await
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
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

    #[test]
    fn test_mode_env_forwards_numeric_values_only() {
        std::env::set_var("PRISM_TEST_TRAIN_MINUTES", "15");
        std::env::set_var("PRISM_TEST_MAX_PARAMS", "2000000");
        let out = test_mode_env();
        assert!(out.contains("PRISM_TEST_TRAIN_MINUTES='15'"), "{out}");
        assert!(out.contains("PRISM_TEST_MAX_PARAMS='2000000'"), "{out}");
        std::env::set_var("PRISM_TEST_TRAIN_MINUTES", "15'; rm -rf /; '");
        let out = test_mode_env();
        assert!(!out.contains("rm -rf"), "{out}");
        std::env::remove_var("PRISM_TEST_TRAIN_MINUTES");
        std::env::remove_var("PRISM_TEST_MAX_PARAMS");
        assert!(test_mode_env().is_empty());
    }

    #[test]
    fn debug_redacts_key() {
        let c = LiumClient::with_base_url("super-secret-key-xyz", "http://example").unwrap();
        let s = format!("{c:?}");
        assert!(!s.contains("super-secret"));
        assert!(s.contains("<redacted>"));
    }

    #[test]
    fn base64_roundtrip_smoke() {
        let s = base64_encode(b"hello");
        assert_eq!(s, "aGVsbG8=");
    }
}
