//! Containerized review backend: runs the `challenge-review` image via Docker
//! (same hardening pattern as `design-sandbox`) with the submitted agent and
//! the most-similar harness mounted read-only, verdict written to `/out`.
//!
//! The container gets `AGENTIC_ENABLE_RUN_COMMAND=1` so the LLM may use the
//! sandboxed `run_command` tool (required for diffs / grep / AST probes in
//! prod). Mitigations against prompt-injected `OpenRouter` key theft:
//! - key is mounted as a **file** (`OPENROUTER_API_KEY_FILE`), never put in
//!   the container's initial process environ (`/proc/<pid>/environ` is a
//!   boot-time snapshot and cannot be scrubbed by `unsetenv`);
//! - `run_command` children get `env_clear` and refuse procfs + secrets paths.
//!
//! With no `OpenRouter` key the inner agent is the deterministic `SimAgent`
//! and the container runs with networking disabled.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use challenge_agentic::{AgenticBackend, AgenticError, AgenticVerdict, ReviewRequest};
use challenge_ast::{fingerprint_source, top_k_nearest, Fingerprint};
use docker_engine::{Allowlist, AllowlistClient, RunSpec, DESIGN_OWNED_NAME_PREFIX};

/// Compose/local tag for the review image (digests pin via deploy).
pub const DEFAULT_REVIEW_IMAGE: &str = "design-review:0.1.0";

/// Request payload path inside the container.
pub const CONTAINER_REQUEST_PATH: &str = "/work/_review_request.json";
/// Verdict output path inside the container.
pub const CONTAINER_VERDICT_PATH: &str = "/out/verdict.json";
/// `OpenRouter` key file path inside the container (RO secrets mount).
pub const CONTAINER_KEY_FILE: &str = "/run/review-secrets/openrouter_api_key";
/// Secrets mount point inside the container.
pub const CONTAINER_SECRETS_MOUNT: &str = "/run/review-secrets";

/// `DockerAgent` settings.
#[derive(Debug, Clone)]
pub struct DockerAgentConfig {
    /// Docker engine (socket-proxy) base URL.
    pub docker_base: String,
    /// Review image ref (digest-pinned in deploy).
    pub image: String,
    /// `OpenRouter` key handed to the inner agent (file mount, never logged).
    pub openrouter_key: Option<String>,
    /// Optional `OpenRouter` base override.
    pub openrouter_base: Option<String>,
    /// Optional `OpenRouter` model override.
    pub openrouter_model: Option<String>,
    /// Container wall-clock timeout.
    pub timeout_sec: u64,
}

impl Default for DockerAgentConfig {
    fn default() -> Self {
        Self {
            docker_base: "http://socket-proxy:2375".into(),
            image: DEFAULT_REVIEW_IMAGE.into(),
            openrouter_key: None,
            openrouter_base: None,
            openrouter_model: None,
            timeout_sec: 600,
        }
    }
}

/// Review verdict runner over Docker.
#[derive(Debug)]
pub struct DockerAgent {
    /// Engine client (verifier allowlist) — exposed for boot probes.
    pub client: AllowlistClient,
    cfg: DockerAgentConfig,
}

impl DockerAgent {
    /// New with verifier allowlist client.
    ///
    /// # Errors
    /// Client build failure.
    pub fn new(cfg: DockerAgentConfig) -> Result<Self, AgenticError> {
        let client = AllowlistClient::with_allowlist(&cfg.docker_base, Allowlist::verifier())
            .map_err(|e| AgenticError::Tool(format!("docker client: {e}")))?;
        Ok(Self { client, cfg })
    }

    /// Container env (key never logged by callers; never put the raw key in env).
    ///
    /// Prod keeps `AGENTIC_ENABLE_RUN_COMMAND=1`: agentic review needs shell
    /// probes. Do not disable without an alternate inspection path; secrets
    /// stay out of process environ and `run_command` denies procfs + the
    /// secrets mount.
    fn container_env(&self, key_file: bool) -> Vec<String> {
        let mut env = vec![
            "AGENTIC_ENABLE_RUN_COMMAND=1".to_owned(),
            format!("REVIEW_REQUEST_PATH={CONTAINER_REQUEST_PATH}"),
            format!("REVIEW_OUT_PATH={CONTAINER_VERDICT_PATH}"),
        ];
        if key_file {
            env.push(format!("OPENROUTER_API_KEY_FILE={CONTAINER_KEY_FILE}"));
        }
        if let Some(b) = &self.cfg.openrouter_base {
            env.push(format!("OPENROUTER_BASE_URL={b}"));
        }
        if let Some(m) = &self.cfg.openrouter_model {
            env.push(format!("OPENROUTER_MODEL={m}"));
        }
        env
    }

    fn build_spec(&self, work: &Path, out_dir: &Path, secrets_dir: Option<&Path>) -> RunSpec {
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let label: String = work
            .file_name()
            .map_or_else(|| "run".into(), |n| n.to_string_lossy().into_owned())
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
            .take(24)
            .collect();
        let mut spec = RunSpec::design_hardened(
            format!("{DESIGN_OWNED_NAME_PREFIX}rev-{label}-{ns}"),
            self.cfg.image.clone(),
            vec![],
        );
        let mut binds = vec![
            format!("{}:/work:ro", work.display()),
            format!("{}:/out:rw", out_dir.display()),
        ];
        if let Some(secrets) = secrets_dir {
            binds.push(format!(
                "{}:{CONTAINER_SECRETS_MOUNT}:ro",
                secrets.display()
            ));
        }
        spec.binds = binds;
        spec.env = self.container_env(secrets_dir.is_some());
        spec.network_mode = None;
        // Sim inner agent needs no network at all.
        spec.network_disabled = self.cfg.openrouter_key.is_none();
        spec.working_dir = Some("/work".into());
        spec.timeout_sec = Some(self.cfg.timeout_sec);
        spec.memory_bytes = Some(1024 * 1024 * 1024);
        spec.memory_swap_bytes = Some(1024 * 1024 * 1024);
        spec
    }
}

/// Stage `_similar/` (most-similar harness) + `_review_request.json` into the
/// already-staged `workdir`; create the writable sibling `<workdir>.out`.
/// Returns the out dir. Pure host-side I/O (unit-testable, no daemon).
///
/// # Errors
/// Staging I/O failures.
pub fn stage_review_mounts(req: &ReviewRequest) -> Result<PathBuf, AgenticError> {
    let work = req
        .workdir
        .canonicalize()
        .map_err(|e| AgenticError::Tool(format!("workdir: {e}")))?;
    // Most-similar corpus harness by AST fingerprint (byte hash fallback).
    if let Some((id, source)) = most_similar(req) {
        let sim_dir = work.join("_similar");
        std::fs::create_dir_all(&sim_dir)
            .map_err(|e| AgenticError::Tool(format!("similar: {e}")))?;
        std::fs::write(sim_dir.join("agent.py"), source)
            .map_err(|e| AgenticError::Tool(format!("similar: {e}")))?;
        std::fs::write(sim_dir.join("NEAREST_ID"), id)
            .map_err(|e| AgenticError::Tool(format!("similar: {e}")))?;
    }
    let payload = serde_json::to_string_pretty(&req.to_container())
        .map_err(|e| AgenticError::Parse(e.to_string()))?;
    std::fs::write(work.join("_review_request.json"), payload)
        .map_err(|e| AgenticError::Tool(format!("request: {e}")))?;
    let out_dir = work.with_file_name(format!(
        "{}.out",
        work.file_name()
            .map_or_else(|| "review".into(), |n| n.to_string_lossy().into_owned())
    ));
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).map_err(|e| AgenticError::Tool(format!("out: {e}")))?;
    Ok(out_dir)
}

/// Stage the `OpenRouter` key into a host-side secrets dir (sibling of `out_dir`)
/// for a RO bind into the review container. Never logs the key.
///
/// # Errors
/// Staging I/O failures.
pub fn stage_review_secrets(out_dir: &Path, key: &str) -> Result<PathBuf, AgenticError> {
    let secrets = out_dir.with_file_name(format!(
        "{}.secrets",
        out_dir
            .file_name()
            .map_or_else(|| "review.out".into(), |n| n.to_string_lossy().into_owned())
    ));
    let _ = std::fs::remove_dir_all(&secrets);
    std::fs::create_dir_all(&secrets).map_err(|e| AgenticError::Tool(format!("secrets: {e}")))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&secrets, std::fs::Permissions::from_mode(0o700));
    }
    let path = secrets.join("openrouter_api_key");
    std::fs::write(&path, key).map_err(|e| AgenticError::Tool(format!("secrets: {e}")))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o400));
    }
    Ok(secrets)
}

fn most_similar(req: &ReviewRequest) -> Option<(String, String)> {
    let cand = req
        .primary_relpaths
        .iter()
        .find_map(|rel| std::fs::read_to_string(req.workdir.join(rel)).ok())?;
    let fp = fingerprint_source(&cand).ok()?;
    let corpus: Vec<(String, Fingerprint)> = req
        .corpus
        .iter()
        .filter_map(|e| {
            fingerprint_source(&e.source)
                .ok()
                .map(|f| (e.id.clone(), f))
        })
        .collect();
    let nearest = top_k_nearest(&fp, &corpus, 1).into_iter().next()?;
    req.corpus
        .iter()
        .find(|e| e.id == nearest.id)
        .map(|e| (e.id.clone(), e.source.clone()))
}

#[async_trait]
impl AgenticBackend for DockerAgent {
    async fn review(&self, req: &ReviewRequest) -> Result<AgenticVerdict, AgenticError> {
        let out_dir = stage_review_mounts(req)?;
        let secrets_dir = match &self.cfg.openrouter_key {
            Some(k) => Some(stage_review_secrets(&out_dir, k)?),
            None => None,
        };
        let spec = self.build_spec(&req.workdir, &out_dir, secrets_dir.as_deref());
        let client = self.client.clone();
        let run = tokio::task::spawn_blocking(move || client.run_owned(&spec))
            .await
            .map_err(|e| AgenticError::Transport(format!("join: {e}")))?
            .map_err(|e| AgenticError::Provider(format!("review container: {e}")))?;
        let verdict_path = out_dir.join("verdict.json");
        let text = std::fs::read_to_string(&verdict_path).map_err(|e| {
            AgenticError::NoVerdict(format!(
                "verdict missing (exit={}): {e}; logs: {}",
                run.status_code,
                tail(&run.logs)
            ))
        })?;
        let _ = std::fs::remove_dir_all(&out_dir);
        if let Some(s) = secrets_dir {
            let _ = std::fs::remove_dir_all(&s);
        }
        if run.status_code != 0 {
            return Err(AgenticError::Provider(format!(
                "review exit={}: {}",
                run.status_code,
                tail(&run.logs)
            )));
        }
        serde_json::from_str(&text).map_err(|e| AgenticError::Parse(format!("verdict json: {e}")))
    }
}

fn tail(logs: &str) -> String {
    let t: String = logs.chars().take(600).collect();
    t.replace(['\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use challenge_agentic::CorpusEntry;
    use tempfile::tempdir;

    #[test]
    fn stage_mounts_writes_request_similar_and_out() {
        let dir = tempdir().unwrap();
        let work = dir.path().join("agentic-run1");
        std::fs::create_dir_all(&work).unwrap();
        let cand = "def run(task, llm, out):\n    out.write_page('index.html', '<html>a</html>')\n";
        std::fs::write(work.join("agent.py"), cand).unwrap();
        let req = ReviewRequest {
            workdir: work.clone(),
            primary_relpaths: vec!["agent.py".into()],
            corpus: vec![CorpusEntry {
                id: "harness:v".into(),
                source: cand.to_owned(),
            }],
            metrics_relpath: None,
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: "design".into(),
        };
        let out = stage_review_mounts(&req).unwrap();
        assert!(work.join("_review_request.json").is_file());
        assert_eq!(
            std::fs::read_to_string(work.join("_similar/NEAREST_ID")).unwrap(),
            "harness:v"
        );
        assert!(work.join("_similar/agent.py").is_file());
        assert!(out.is_dir());
        // Container request carries corpus + rules, no host workdir leak.
        let v: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(work.join("_review_request.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(v["corpus"][0]["id"], "harness:v");
        assert_eq!(v["domain_rules"], "design");
        assert!(v.get("workdir").is_none());
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn container_env_uses_key_file_not_environ() {
        let agent = DockerAgent::new(DockerAgentConfig {
            openrouter_key: Some("sk-test-not-a-real-key".into()),
            ..DockerAgentConfig::default()
        })
        .unwrap();
        let env = agent.container_env(true);
        assert!(env.iter().any(|e| e == "AGENTIC_ENABLE_RUN_COMMAND=1"));
        let expected = format!("OPENROUTER_API_KEY_FILE={CONTAINER_KEY_FILE}");
        assert!(
            env.iter().any(|e| e == &expected),
            "missing key-file env; got {env:?}"
        );
        assert!(
            !env.iter().any(|e| e.starts_with("OPENROUTER_API_KEY=")),
            "raw key must not enter container environ (proc environ is boot-fixed)"
        );
        assert!(!env.iter().any(|e| e.contains("sk-test-not-a-real-key")));
    }

    #[test]
    fn stage_secrets_writes_key_file() {
        let dir = tempdir().unwrap();
        let out = dir.path().join("agentic-run1.out");
        std::fs::create_dir_all(&out).unwrap();
        let secrets = stage_review_secrets(&out, "sk-test-not-a-real-key").unwrap();
        let key_path = secrets.join("openrouter_api_key");
        assert_eq!(
            std::fs::read_to_string(&key_path).unwrap(),
            "sk-test-not-a-real-key"
        );
        let _ = std::fs::remove_dir_all(&secrets);
    }

    #[test]
    fn build_spec_binds_secrets_mount() {
        let dir = tempdir().unwrap();
        let work = dir.path().join("agentic-run1");
        let out = dir.path().join("agentic-run1.out");
        let secrets = dir.path().join("agentic-run1.out.secrets");
        std::fs::create_dir_all(&work).unwrap();
        std::fs::create_dir_all(&out).unwrap();
        std::fs::create_dir_all(&secrets).unwrap();
        let agent = DockerAgent::new(DockerAgentConfig {
            openrouter_key: Some("sk-test-not-a-real-key".into()),
            ..DockerAgentConfig::default()
        })
        .unwrap();
        let spec = agent.build_spec(&work, &out, Some(&secrets));
        assert!(spec
            .binds
            .iter()
            .any(|b| b.contains(":ro") && b.contains(CONTAINER_SECRETS_MOUNT)));
        assert!(spec
            .env
            .iter()
            .any(|e| e.starts_with("OPENROUTER_API_KEY_FILE=")));
        assert!(!spec
            .env
            .iter()
            .any(|e| e.starts_with("OPENROUTER_API_KEY=")));
        assert!(!spec.network_disabled);
    }
}
