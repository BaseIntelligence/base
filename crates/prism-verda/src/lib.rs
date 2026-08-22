//! Verda serverless batch jobs as a Prism [`EvalJobBackend`].
//!
//! Miners pay with OAuth client credentials + an inference token. The
//! container image and job command are operator-pinned (digest + embedded
//! `job_server.py`). No SSH.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc, clippy::doc_markdown)]

mod image;

pub use image::{
    hex_sha256, job_command, job_server_env_pairs, job_server_sha256, pinned_image, require_digest,
    CAPACITY_NOTE, CAPACITY_POLICY, EXPOSED_PORT, HEALTH_PATH, JOB_SERVER_PY,
};

use std::time::Duration;

use async_trait::async_trait;
use prism_lium::{
    EvalJobBackend, Instance, InstanceSpec, LiumError, Offer, RemoteExecResult, MIN_LIFETIME_HOURS,
};
use prism_lium_harness::{harness_env_pairs, harness_upload_tar, parse_metrics_output};
use prism_lium_types::CostGuardrailError;
use serde_json::{json, Value};

/// Public API base (OAuth + job-deployments).
pub const VERDA_API_BASE: &str = "https://api.verda.com/v1";
/// Batch job trigger / poll host.
pub const VERDA_TASKS_BASE: &str = "https://tasks.datacrunch.io";

/// Miner Verda credentials (never logged).
#[derive(Clone)]
pub struct VerdaCreds {
    /// Cloud API client id.
    pub client_id: String,
    /// Cloud API client secret.
    pub client_secret: String,
    /// Inference / tasks bearer.
    pub inference_key: String,
}

impl std::fmt::Debug for VerdaCreds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerdaCreds")
            .field("client_id", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Live Verda backend (one miner account).
pub struct VerdaClient {
    creds: VerdaCreds,
    api_base: String,
    tasks_base: String,
    http: reqwest::Client,
    train_hours_cap: f64,
}

impl std::fmt::Debug for VerdaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerdaClient")
            .field("api_base", &self.api_base)
            .finish_non_exhaustive()
    }
}

impl VerdaClient {
    /// Production endpoints.
    pub fn new(creds: VerdaCreds) -> Result<Self, LiumError> {
        Self::with_bases(creds, VERDA_API_BASE, VERDA_TASKS_BASE)
    }

    /// Test / isolated bases.
    pub fn with_bases(
        creds: VerdaCreds,
        api_base: &str,
        tasks_base: &str,
    ) -> Result<Self, LiumError> {
        if creds.client_id.trim().is_empty()
            || creds.client_secret.trim().is_empty()
            || creds.inference_key.trim().is_empty()
        {
            return Err(LiumError::Api("missing_verda_credentials".into()));
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_mins(1))
            .user_agent("prism-verda/0.1")
            .build()
            .map_err(|e| LiumError::Transport(e.to_string()))?;
        let train_hours_cap = std::env::var("PRISM_TEST_TRAIN_MINUTES")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|m| *m > 0.0)
            .map(|m| m / 60.0)
            .or_else(|| {
                std::env::var("PRISM_TRAIN_HOURS_CAP")
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|h| *h > 0.0)
            })
            .unwrap_or(prism_recipe::TRAIN_HOURS_CAP);
        Ok(Self {
            creds,
            api_base: api_base.trim_end_matches('/').to_owned(),
            tasks_base: tasks_base.trim_end_matches('/').to_owned(),
            http,
            train_hours_cap,
        })
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn deadline_secs(lifetime_hours: f64) -> u64 {
        let h = lifetime_hours.max(MIN_LIFETIME_HOURS);
        ((h * 3600.0).ceil() as u64).max(600)
    }

    async fn access_token(&self) -> Result<String, LiumError> {
        let res = self
            .http
            .post(format!("{}/oauth2/token", self.api_base))
            .json(&json!({
                "grant_type": "client_credentials",
                "client_id": self.creds.client_id,
                "client_secret": self.creds.client_secret,
            }))
            .send()
            .await
            .map_err(|e| LiumError::Transport(e.to_string()))?;
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(LiumError::Api(format!(
                "oauth2/token -> {status} {}",
                redact(&body)
            )));
        }
        let v: Value =
            serde_json::from_str(&body).map_err(|e| LiumError::Api(format!("oauth json: {e}")))?;
        v.get("access_token")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| LiumError::Api("oauth2/token missing access_token".into()))
    }

    async fn api_json(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<(reqwest::StatusCode, Value), LiumError> {
        let token = self.access_token().await?;
        let mut req = self
            .http
            .request(method, format!("{}{path}", self.api_base))
            .bearer_auth(&token);
        if let Some(b) = body {
            req = req.json(&b);
        }
        let res = req
            .send()
            .await
            .map_err(|e| LiumError::Transport(e.to_string()))?;
        let status = res.status();
        let text = res.text().await.unwrap_or_default();
        let json = if text.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&text).unwrap_or(Value::String(text.clone()))
        };
        if status.as_u16() == 429 || is_capacity_status(status, &text) {
            return Err(CostGuardrailError::NoCapacity.into());
        }
        Ok((status, json))
    }

    /// Snapshot used by unit tests (digest + operator command hash).
    #[must_use]
    pub fn deployment_body(name: &str, image: &str, compute_name: &str, deadline: u64) -> Value {
        Self::deployment_body_with_train(
            name,
            image,
            compute_name,
            deadline,
            prism_recipe::TRAIN_HOURS_CAP,
        )
    }

    fn deployment_body_with_train(
        name: &str,
        image: &str,
        compute_name: &str,
        deadline: u64,
        train_hours: f64,
    ) -> Value {
        let mut env: Vec<Value> = harness_env_pairs(train_hours, compute_name, false)
            .into_iter()
            .map(|(key, value)| {
                json!({
                    "name": key,
                    "value_or_reference_to_secret": value,
                    "type": "plain",
                })
            })
            .collect();
        for (key, value) in job_server_env_pairs() {
            env.push(json!({
                "name": key,
                "value_or_reference_to_secret": value,
                "type": "plain",
            }));
        }
        json!({
            "name": name,
            "containers": [{
                "image": image,
                "exposed_port": EXPOSED_PORT,
                "healthcheck": {
                    "enabled": true,
                    "port": EXPOSED_PORT,
                    "path": HEALTH_PATH,
                },
                "entrypoint_overrides": {
                    "enabled": true,
                    "cmd": job_command(),
                },
                "env": env,
            }],
            "compute": { "name": compute_name, "size": 1 },
            "scaling": {
                "max_replica_count": 1,
                "queue_message_ttl_seconds": 86400,
                "deadline_seconds": deadline,
            },
            "container_registry_settings": { "is_private": false },
        })
    }

    fn pick_compute(resources: &[Value]) -> Result<(String, String), LiumError> {
        // Dashboard "1× B200 available" is not always `is_available` + `size=1`
        // on `/serverless-compute-resources`. Pin B200 by name first; create
        // with `compute.size=1`. Only H200/H100 if no B200 SKU exists at all.
        let named = |needles: &[&str], require_avail: bool, want_size: Option<u64>| {
            resources.iter().find_map(|r| {
                let name = r.get("name").and_then(Value::as_str)?;
                let avail = r
                    .get("is_available")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let size = r.get("size").and_then(Value::as_u64);
                if require_avail && !avail {
                    return None;
                }
                if want_size.is_some() && size != want_size {
                    return None;
                }
                let low = name.to_ascii_lowercase();
                needles
                    .iter()
                    .any(|n| low.contains(n))
                    .then(|| (name.to_owned(), name.to_owned()))
            })
        };
        if let Some(want) = std::env::var("PRISM_VERDA_COMPUTE")
            .ok()
            .filter(|s| !s.trim().is_empty())
        {
            let needle = want.to_ascii_lowercase();
            if let Some(hit) = named(&[needle.as_str()], false, Some(1))
                .or_else(|| named(&[needle.as_str()], false, None))
            {
                return Ok(hit);
            }
            return Err(LiumError::from(CostGuardrailError::NoCapacity));
        }
        named(&["b200"], true, Some(1))
            .or_else(|| named(&["b200"], true, None))
            .or_else(|| named(&["b200"], false, Some(1)))
            .or_else(|| named(&["b200"], false, None))
            .or_else(|| named(&["h200"], true, Some(1)))
            .or_else(|| named(&["h100"], true, Some(1)))
            .ok_or_else(|| LiumError::from(CostGuardrailError::NoCapacity))
    }

    async fn list_compute(&self) -> Result<Vec<Value>, LiumError> {
        let (status, v) = self
            .api_json(reqwest::Method::GET, "/serverless-compute-resources", None)
            .await?;
        if !status.is_success() {
            return Err(LiumError::Api(format!(
                "compute-resources -> {status} {}",
                redact(&v.to_string())
            )));
        }
        match v {
            Value::Array(a) => Ok(a),
            other => other
                .get("compute_resources")
                .and_then(Value::as_array)
                .cloned()
                .ok_or_else(|| LiumError::Api("compute-resources: expected array".into())),
        }
    }

    async fn wait_running(&self, name: &str) -> Result<(), LiumError> {
        for _ in 0..40 {
            let (status, v) = self
                .api_json(
                    reqwest::Method::GET,
                    &format!("/job-deployments/{name}/status"),
                    None,
                )
                .await?;
            if status.is_success() {
                let st = v
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if matches!(st.as_str(), "running" | "healthy" | "ready") {
                    return Ok(());
                }
            }
            tokio::time::sleep(Duration::from_secs(15)).await;
        }
        Err(LiumError::Api(format!(
            "verda job-deployment {name} not running"
        )))
    }

    async fn tasks_endpoint(&self, name: &str) -> Result<String, LiumError> {
        let (status, v) = self
            .api_json(
                reqwest::Method::GET,
                &format!("/job-deployments/{name}"),
                None,
            )
            .await?;
        if status.is_success() {
            if let Some(u) = v.get("endpoint_base_url").and_then(Value::as_str) {
                return Ok(u.trim_end_matches('/').to_owned());
            }
        }
        Ok(format!("{}/{name}", self.tasks_base))
    }

    async fn post_job(&self, name: &str, tar: &[u8]) -> Result<String, LiumError> {
        let ep = self.tasks_endpoint(name).await?;
        let res = self
            .http
            .post(format!("{ep}/job"))
            .bearer_auth(&self.creds.inference_key)
            .header("X-Inference-Id", name)
            .header("Content-Type", "application/octet-stream")
            .body(tar.to_vec())
            .send()
            .await
            .map_err(|e| LiumError::Transport(e.to_string()))?;
        let status = res.status();
        let text = res.text().await.unwrap_or_default();
        if status.as_u16() == 429 || is_capacity_status(status, &text) {
            return Err(CostGuardrailError::NoCapacity.into());
        }
        if !status.is_success() {
            return Err(LiumError::Api(format!(
                "tasks job -> {status} {}",
                redact(&text)
            )));
        }
        Ok(name.to_owned())
    }

    async fn poll_result(
        &self,
        name: &str,
        timeout: Duration,
    ) -> Result<RemoteExecResult, LiumError> {
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(LiumError::Exec("verda job deadline".into()));
            }
            let res = self
                .http
                .get(format!("{}/status/{name}", self.tasks_base))
                .bearer_auth(&self.creds.inference_key)
                .header("X-Inference-Id", name)
                .send()
                .await
                .map_err(|e| LiumError::Transport(e.to_string()))?;
            let text = res.text().await.unwrap_or_default();
            let st = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    v.get("Status")
                        .or_else(|| v.get("status"))
                        .and_then(Value::as_str)
                        .map(str::to_ascii_lowercase)
                })
                .unwrap_or_default();
            let inflight = matches!(
                st.as_str(),
                "queue"
                    | "queued"
                    | "initialized"
                    | "running"
                    | "inference"
                    | "starting"
                    | "pending"
                    | ""
            );
            if inflight {
                tokio::time::sleep(Duration::from_secs(20)).await;
                continue;
            }
            let res = self
                .http
                .get(format!("{}/result/{name}", self.tasks_base))
                .bearer_auth(&self.creds.inference_key)
                .header("X-Inference-Id", name)
                .send()
                .await
                .map_err(|e| LiumError::Transport(e.to_string()))?;
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            if status.as_u16() == 404 || body.contains("result not found") {
                tokio::time::sleep(Duration::from_secs(20)).await;
                continue;
            }
            return parse_job_result(&body);
        }
    }
}

fn parse_job_result(body: &str) -> Result<RemoteExecResult, LiumError> {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(m) = v.get("metrics").and_then(Value::as_str) {
            if !m.is_empty() {
                return parse_metrics_output(&format!("METRICS_JSON={m}\nEVAL_OK\n"), 0, "");
            }
        }
        if v.get("bpb").is_some() {
            return serde_json::from_value(v)
                .map_err(|e| LiumError::Exec(format!("result metrics: {e}")));
        }
        let log = v.get("log").and_then(Value::as_str).unwrap_or(body);
        return parse_metrics_output(log, 0, "");
    }
    parse_metrics_output(body, 0, "")
}

fn is_capacity_status(status: reqwest::StatusCode, text: &str) -> bool {
    let l = text.to_ascii_lowercase();
    status.as_u16() == 503
        || l.contains("no_capacity")
        || l.contains("out of capacity")
        || l.contains("sold out")
        || l.contains("no available")
}

fn redact(s: &str) -> String {
    let mut t = s.to_owned();
    for needle in ["client_secret", "access_token", "inference"] {
        if t.to_ascii_lowercase().contains(needle) {
            t = format!("<{needle} redacted len={}>", s.len());
            break;
        }
    }
    if t.len() > 400 {
        t.truncate(400);
    }
    t
}

#[async_trait]
impl EvalJobBackend for VerdaClient {
    async fn list_offers(&self, _max_price_per_hour: Option<f64>) -> Result<Vec<Offer>, LiumError> {
        let rows = self.list_compute().await?;
        Ok(rows
            .iter()
            .filter_map(|r| {
                let name = r.get("name").and_then(Value::as_str)?;
                Some(Offer {
                    id: name.to_owned(),
                    gpu_type: name.to_owned(),
                    gpu_count: 1,
                    price_per_hour: 0.0,
                    provider: "verda".into(),
                    ..Offer::default()
                })
            })
            .collect())
    }

    async fn provision(&self, spec: &InstanceSpec) -> Result<Instance, LiumError> {
        if spec.max_lifetime_hours <= 0.0 {
            return Err(CostGuardrailError::LifetimeMissing.into());
        }
        if spec.max_lifetime_hours < MIN_LIFETIME_HOURS {
            return Err(CostGuardrailError::LifetimeBelowFloor.into());
        }
        let image = if let Some(d) = spec.image_digest.as_deref().filter(|s| !s.is_empty()) {
            require_digest(d)?
        } else {
            pinned_image()?
        };
        let compute = self.list_compute().await?;
        let (compute_name, gpu_type) = Self::pick_compute(&compute)?;
        let deadline = Self::deadline_secs(spec.max_lifetime_hours);
        let body = Self::deployment_body_with_train(
            &spec.name,
            &image,
            &compute_name,
            deadline,
            self.train_hours_cap,
        );
        // Never honor miner image/cmd: body is built here only.
        debug_assert!(body["containers"][0]["image"] == image);
        let (status, v) = self
            .api_json(reqwest::Method::POST, "/job-deployments", Some(body))
            .await?;
        if !status.is_success() {
            let msg = v.to_string();
            if lium_rent_pool::is_no_capacity(&msg) || is_capacity_status(status, &msg) {
                return Err(CostGuardrailError::NoCapacity.into());
            }
            return Err(LiumError::Api(format!(
                "create job-deployment -> {status} {}",
                redact(&msg)
            )));
        }
        Ok(Instance {
            id: spec.name.clone(),
            status: "PROVISIONING".into(),
            provider: "verda".into(),
            gpu_type: Some(gpu_type),
            ssh_connect_cmd: None,
        })
    }

    async fn terminate(&self, instance_id: &str) -> Result<(), LiumError> {
        let (status, v) = self
            .api_json(
                reqwest::Method::DELETE,
                &format!("/job-deployments/{instance_id}"),
                None,
            )
            .await?;
        if status.is_success() || status.as_u16() == 404 {
            return Ok(());
        }
        Err(LiumError::Api(format!(
            "delete job-deployment -> {status} {}",
            redact(&v.to_string())
        )))
    }

    async fn verify_terminated(&self, instance_id: &str) -> Result<bool, LiumError> {
        let (status, _) = self
            .api_json(
                reqwest::Method::GET,
                &format!("/job-deployments/{instance_id}"),
                None,
            )
            .await?;
        Ok(status.as_u16() == 404)
    }

    async fn exec_eval(
        &self,
        instance_id: &str,
        architecture_py: &str,
        training_py: &str,
        tree_blob: Option<&[u8]>,
    ) -> Result<RemoteExecResult, LiumError> {
        self.wait_running(instance_id).await?;
        let tar = harness_upload_tar(architecture_py, training_py, tree_blob)?;
        let _ = harness_env_pairs(self.train_hours_cap, "verda", false);
        self.post_job(instance_id, &tar).await?;
        let timeout = Duration::from_secs(Self::deadline_secs(self.train_hours_cap + 1.5));
        self.poll_result(instance_id, timeout).await
    }

    async fn resume_eval(&self, instance_id: &str) -> Result<RemoteExecResult, LiumError> {
        let timeout = Duration::from_secs(Self::deadline_secs(self.train_hours_cap + 1.5));
        self.poll_result(instance_id, timeout).await
    }

    async fn instance_running(&self, instance_id: &str) -> Result<bool, LiumError> {
        let (status, v) = self
            .api_json(
                reqwest::Method::GET,
                &format!("/job-deployments/{instance_id}/status"),
                None,
            )
            .await?;
        if !status.is_success() {
            return Ok(false);
        }
        let st = v
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        Ok(matches!(st.as_str(), "running" | "healthy" | "ready"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_payload_is_digest_and_operator_cmd() {
        let image =
            "docker.io/x/y@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let body = VerdaClient::deployment_body("prism-abc", image, "B200", 14400);
        assert_eq!(body["containers"][0]["image"], image);
        assert_eq!(body["compute"]["size"], 1);
        assert_eq!(body["compute"]["name"], "B200");
        let cmd = body["containers"][0]["entrypoint_overrides"]["cmd"]
            .as_array()
            .unwrap();
        assert_eq!(cmd[0], "python3");
        assert!(cmd.iter().all(|a| a.as_str().unwrap().len() < 253));
        assert_eq!(
            body["containers"][0]["entrypoint_overrides"]["enabled"],
            true
        );
        let env = body["containers"][0]["env"].as_array().unwrap();
        assert!(env.iter().any(|e| e["name"] == "PRISM_JS_C00"));
        assert_eq!(job_server_sha256().len(), 64);
        assert!(require_digest("nginx:latest").is_err());
    }

    #[test]
    fn pick_b200_then_fallback() {
        let rows = vec![
            json!({"name": "H100", "size": 1, "is_available": true}),
            json!({"name": "1x B200", "size": 1, "is_available": true}),
        ];
        let (n, _) = VerdaClient::pick_compute(&rows).unwrap();
        assert!(n.to_ascii_lowercase().contains("b200"));
        let mixed = vec![
            json!({"name": "B200", "size": 2, "is_available": true}),
            json!({"name": "B200", "size": 1, "is_available": true}),
        ];
        let (n, _) = VerdaClient::pick_compute(&mixed).unwrap();
        assert!(n.to_ascii_lowercase().contains("b200"));
        let queued = vec![json!({"name": "8x B200", "size": 8, "is_available": false})];
        let (n, _) = VerdaClient::pick_compute(&queued).unwrap();
        assert!(n.to_ascii_lowercase().contains("b200"));
        let none = vec![json!({"name": "L4", "size": 1, "is_available": true})];
        assert!(VerdaClient::pick_compute(&none).is_err());
    }

    #[test]
    fn pick_honors_compute_override() {
        let rows = vec![
            json!({"name": "B200", "size": 1, "is_available": true}),
            json!({"name": "L40S", "size": 1, "is_available": true}),
        ];
        std::env::set_var("PRISM_VERDA_COMPUTE", "L40S");
        let (n, _) = VerdaClient::pick_compute(&rows).unwrap();
        std::env::remove_var("PRISM_VERDA_COMPUTE");
        assert!(n.to_ascii_lowercase().contains("l40"));
    }

    #[tokio::test]
    async fn create_no_capacity_is_queued_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let api = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth2/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "t"
            })))
            .mount(&api)
            .await;
        Mock::given(method("GET"))
            .and(path("/serverless-compute-resources"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
            .mount(&api)
            .await;
        let c = VerdaClient::with_bases(
            VerdaCreds {
                client_id: "id".into(),
                client_secret: "sec".into(),
                inference_key: "inf".into(),
            },
            &api.uri(),
            &api.uri(),
        )
        .unwrap();
        let spec = InstanceSpec {
            name: "prism-t".into(),
            max_lifetime_hours: 2.0,
            ..InstanceSpec::default()
        };
        let err = c.provision(&spec).await.unwrap_err();
        assert!(
            matches!(err, LiumError::Cost(CostGuardrailError::NoCapacity))
                || lium_rent_pool::is_no_capacity(&err.to_string()),
            "{err}"
        );
    }

    #[test]
    fn creds_debug_redacted() {
        let c = VerdaCreds {
            client_id: "id".into(),
            client_secret: "SECRETVALUE".into(),
            inference_key: "INF".into(),
        };
        let d = format!("{c:?}");
        assert!(!d.contains("SECRETVALUE"));
        assert!(!d.contains("INF"));
    }
}
