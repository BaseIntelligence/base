//! Harbor pack execution via allowlisted Docker (todo 21).
//!
//! Pulls a digest-pinned environment image, runs a reference agent command against
//! the stripped instruction, collects `/logs/artifacts/model.patch`, enforces the
//! wall deadline, and tears down owned containers. Model API keys are mounted as
//! files only — never logged.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use agent_dispatch::TaskStatusV1;
use agent_pack::{load_pack, StrippedDescriptor};
use docker_engine::{
    Allowlist, AllowlistClient, DockerApi, DockerError, RunResult, RunSpec, OWNED_NAME_PREFIX,
};
use thiserror::Error;

use crate::auth::unix_now_ms;
use crate::egress::{AgentEgressPosture, DEFAULT_AGENT_EGRESS_POSTURE};

/// Env var name inside the agent container pointing at the mounted key file path.
pub const MODEL_KEY_FILE_ENV: &str = "MODEL_KEY_FILE";
/// In-container mount path for the miner-supplied model key (file, mode 0600 on host).
pub const MODEL_KEY_CONTAINER_PATH: &str = "/run/base/model_key";
/// Artifact path produced by the agent (Harbor contract).
pub const MODEL_PATCH_REL: &str = "artifacts/model.patch";

/// Prefix for agent one-shot containers (subset of [`OWNED_NAME_PREFIX`]).
pub const AGENT_CONTAINER_PREFIX: &str = "base-verify-agent-";

/// Configuration for Docker-backed pack execution.
#[derive(Debug, Clone)]
pub struct DockerExecConfig {
    /// Docker Engine HTTP base (socket-proxy), e.g. `http://127.0.0.1:2375`.
    pub docker_base: String,
    /// Digest-pinned environment image (`name@sha256:…`).
    pub environment_image: String,
    /// Host directory of Harbor packs (`{pack_root}/{pack_id}/task.toml`).
    pub pack_root: PathBuf,
    /// Staging root for binds (instruction + logs).
    pub work_root: PathBuf,
    /// Optional host path to miner model key file (mounted read-only).
    pub model_key_path: Option<PathBuf>,
    /// Egress posture (default [`DEFAULT_AGENT_EGRESS_POSTURE`]).
    pub egress: AgentEgressPosture,
    /// Override agent argv; default is the reference agent that writes a unified diff.
    pub agent_cmd: Option<Vec<String>>,
    /// Optional pack catalog base URL (`…/v1/packs/{id}`).
    pub pack_catalog_url: Option<String>,
}

impl Default for DockerExecConfig {
    fn default() -> Self {
        Self {
            docker_base: "http://127.0.0.1:2375".into(),
            environment_image: String::new(),
            pack_root: PathBuf::from("/var/lib/base/packs"),
            work_root: PathBuf::from("/tmp/base-agent-work"),
            model_key_path: None,
            egress: DEFAULT_AGENT_EGRESS_POSTURE,
            agent_cmd: None,
            pack_catalog_url: None,
        }
    }
}

/// How the runner produces `model.patch`.
#[derive(Debug, Clone)]
pub enum ExecutionBackend {
    /// Deterministic stub patch (unit tests / no Docker). Honors deadline.
    Stub {
        /// Artificial hold before completion.
        hold: Duration,
    },
    /// Real allowlisted Docker path.
    Docker(DockerExecConfig),
}

impl Default for ExecutionBackend {
    fn default() -> Self {
        Self::Stub {
            hold: Duration::ZERO,
        }
    }
}

/// Outcome of one pack execution attempt (before receipt signing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecOutcome {
    /// Terminal wire status.
    pub status: TaskStatusV1,
    /// Unified diff when produced; `None` on timeout / hard failure without patch.
    pub model_patch: Option<String>,
}

/// Pack execution failures (mapped to `Failed` or `TimedOut`).
#[derive(Debug, Error)]
pub enum ExecError {
    /// Pack directory missing or invalid.
    #[error("pack load: {0}")]
    Pack(String),
    /// Staging I/O.
    #[error("staging: {0}")]
    Staging(String),
    /// Docker / allowlist failure.
    #[error("docker: {0}")]
    Docker(#[from] DockerError),
    /// Wall deadline exceeded.
    #[error("deadline exceeded")]
    Deadline,
    /// Agent finished without a usable patch.
    #[error("empty model.patch")]
    EmptyPatch,
}

/// Resolve wall-clock timeout seconds from descriptor deadline and pack budget.
#[must_use]
pub fn resolve_timeout_sec(deadline_unix_ms: u64, pack_deadline_sec: u64, now_unix_ms: u64) -> u64 {
    if deadline_unix_ms <= now_unix_ms {
        return 0;
    }
    let remaining_ms = deadline_unix_ms.saturating_sub(now_unix_ms);
    let remaining_sec = remaining_ms.div_ceil(1000).max(1);
    let pack = pack_deadline_sec.max(1);
    remaining_sec.min(pack)
}

/// Default reference agent: write a non-empty unified diff to the Harbor artifact path.
///
/// Does not read or print model key contents — only checks that the path env is set
/// when a key was mounted.
#[must_use]
pub fn reference_agent_cmd(pack_id: &str) -> Vec<String> {
    let script = format!(
        r#"set -euo pipefail
mkdir -p /logs/artifacts
# Model key path only — never log file bytes
if [ -n "${{MODEL_KEY_FILE:-}}" ]; then
  test -f "$MODEL_KEY_FILE"
fi
cat > /logs/artifacts/model.patch <<'EOF'
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 # pack
+reference agent patch for {pack_id}
EOF
test -s /logs/artifacts/model.patch
"#
    );
    vec!["bash".into(), "-lc".into(), script]
}

/// Load stripped projection for `pack_id` under `pack_root`.
///
/// # Errors
/// Missing directory or invalid Harbor pack.
pub fn load_stripped(pack_root: &Path, pack_id: &str) -> Result<StrippedDescriptor, ExecError> {
    let dir = pack_root.join(pack_id);
    let pack = load_pack(&dir).map_err(|e| ExecError::Pack(e.to_string()))?;
    let stripped = pack.strip();
    stripped
        .assert_total_keys()
        .map_err(|e| ExecError::Pack(e.to_string()))?;
    Ok(stripped)
}

/// Execute one pack under the configured backend.
pub fn execute_pack(
    backend: &ExecutionBackend,
    pack_id: &str,
    deadline_unix_ms: u64,
) -> ExecOutcome {
    match backend {
        ExecutionBackend::Stub { hold } => execute_stub(pack_id, deadline_unix_ms, *hold),
        ExecutionBackend::Docker(cfg) => match execute_docker(cfg, pack_id, deadline_unix_ms) {
            Ok(o) => o,
            Err(ExecError::Deadline) => ExecOutcome {
                status: TaskStatusV1::TimedOut,
                model_patch: None,
            },
            Err(e) => {
                tracing::error!(
                    event = "pack_exec_failed",
                    error = %e,
                    pack_id = %pack_id,
                    "pack execution failed"
                );
                ExecOutcome {
                    status: TaskStatusV1::Failed,
                    model_patch: None,
                }
            }
        },
    }
}

fn execute_stub(pack_id: &str, deadline_unix_ms: u64, hold: Duration) -> ExecOutcome {
    let now = unix_now_ms();
    let timeout_sec = resolve_timeout_sec(deadline_unix_ms, u64::MAX / 4, now);
    if timeout_sec == 0 {
        return ExecOutcome {
            status: TaskStatusV1::TimedOut,
            model_patch: None,
        };
    }
    let budget = Duration::from_secs(timeout_sec);
    let start = Instant::now();
    if !hold.is_zero() {
        std::thread::sleep(hold.min(budget.saturating_add(Duration::from_millis(50))));
    }
    if start.elapsed() >= budget || unix_now_ms() >= deadline_unix_ms {
        return ExecOutcome {
            status: TaskStatusV1::TimedOut,
            model_patch: None,
        };
    }
    ExecOutcome {
        status: TaskStatusV1::Completed,
        model_patch: Some(stub_model_patch(pack_id)),
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

fn execute_docker(
    cfg: &DockerExecConfig,
    pack_id: &str,
    deadline_unix_ms: u64,
) -> Result<ExecOutcome, ExecError> {
    crate::fetch_pack::ensure_pack(&cfg.pack_root, cfg.pack_catalog_url.as_deref(), pack_id)
        .map_err(|e| ExecError::Pack(e.to_string()))?;
    let stripped = load_stripped(&cfg.pack_root, pack_id)?;
    let now = unix_now_ms();
    let timeout_sec = resolve_timeout_sec(deadline_unix_ms, stripped.deadline_sec, now);
    if timeout_sec == 0 {
        return Err(ExecError::Deadline);
    }
    if cfg.environment_image.is_empty() {
        return Err(ExecError::Pack(
            "environment_image empty (digest-pinned image required)".into(),
        ));
    }

    let client = AllowlistClient::with_allowlist(&cfg.docker_base, Allowlist::verifier())?;
    let stage = stage_agent_workdir(&cfg.work_root, pack_id, &stripped.instruction)?;
    let name = unique_agent_name(pack_id);
    let mut binds = vec![
        format!("{}:/instruction:ro", stage.instruction_dir.display()),
        format!("{}:/logs", stage.logs_dir.display()),
    ];
    let mut env = vec![
        "ARTIFACTS_DIR=/logs/artifacts".into(),
        "INSTRUCTION_FILE=/instruction/instruction.md".into(),
    ];
    if let Some(key_path) = &cfg.model_key_path {
        // Mount path only — never put key bytes in env or logs.
        binds.push(format!(
            "{}:{}:ro",
            key_path.display(),
            MODEL_KEY_CONTAINER_PATH
        ));
        env.push(format!("{MODEL_KEY_FILE_ENV}={MODEL_KEY_CONTAINER_PATH}"));
        tracing::info!(
            event = "model_key_mounted",
            path = %MODEL_KEY_CONTAINER_PATH,
            "model key file mounted (contents not logged)"
        );
    }

    let cmd = cfg
        .agent_cmd
        .clone()
        .unwrap_or_else(|| reference_agent_cmd(pack_id));

    let spec = RunSpec {
        name: name.clone(),
        image: cfg.environment_image.clone(),
        cmd,
        binds,
        env,
        network_disabled: cfg.egress.network_disabled(),
        working_dir: Some("/".into()),
        timeout_sec: Some(timeout_sec),
    };

    // Best-effort pull then run; always remove this name afterward.
    let run_result = client.run_owned(&spec);
    let _ = client.remove_container(&name);
    // Sweep any siblings left under the agent prefix (this pack attempt).
    let _ = cleanup_agent_owned(&client);

    let outcome = match run_result {
        Ok(r) => outcome_from_run(&stage, r)?,
        Err(DockerError::Timeout { .. }) => {
            let _ = fs::remove_dir_all(&stage.root);
            return Ok(ExecOutcome {
                status: TaskStatusV1::TimedOut,
                model_patch: None,
            });
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&stage.root);
            return Err(ExecError::Docker(e));
        }
    };
    let _ = fs::remove_dir_all(&stage.root);
    Ok(outcome)
}

fn outcome_from_run(stage: &AgentStage, _run: RunResult) -> Result<ExecOutcome, ExecError> {
    let patch_path = stage.logs_dir.join(MODEL_PATCH_REL);
    let bytes = fs::read(&patch_path).map_err(|_| ExecError::EmptyPatch)?;
    if bytes.is_empty() {
        return Err(ExecError::EmptyPatch);
    }
    let text = String::from_utf8(bytes).map_err(|e| ExecError::Staging(e.to_string()))?;
    if !text.contains("diff --git") && !text.starts_with("--- ") {
        // Still accept non-empty text that looks like a patch body.
        if text.trim().is_empty() {
            return Err(ExecError::EmptyPatch);
        }
    }
    Ok(ExecOutcome {
        status: TaskStatusV1::Completed,
        model_patch: Some(text),
    })
}

struct AgentStage {
    root: PathBuf,
    instruction_dir: PathBuf,
    logs_dir: PathBuf,
}

fn stage_agent_workdir(
    work_root: &Path,
    pack_id: &str,
    instruction: &str,
) -> Result<AgentStage, ExecError> {
    fs::create_dir_all(work_root).map_err(|e| ExecError::Staging(e.to_string()))?;
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let root = work_root.join(format!("agent-{pack_id}-{ns}"));
    let instruction_dir = root.join("instruction");
    let logs_dir = root.join("logs");
    let artifacts = logs_dir.join("artifacts");
    for d in [&instruction_dir, &artifacts] {
        fs::create_dir_all(d).map_err(|e| ExecError::Staging(e.to_string()))?;
    }
    fs::write(instruction_dir.join("instruction.md"), instruction)
        .map_err(|e| ExecError::Staging(e.to_string()))?;
    Ok(AgentStage {
        root,
        instruction_dir,
        logs_dir,
    })
}

fn unique_agent_name(pack_id: &str) -> String {
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let safe: String = pack_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .take(48)
        .collect();
    format!("{AGENT_CONTAINER_PREFIX}{safe}-{ns}")
}

fn cleanup_agent_owned(client: &AllowlistClient) -> Result<usize, DockerError> {
    // OWNED_NAME_PREFIX covers base-verify-agent-*; remove only agent-prefixed.
    let owned = client.list_owned()?;
    let mut n = 0usize;
    for c in owned {
        if c.name.starts_with(AGENT_CONTAINER_PREFIX)
            || c.name
                .trim_start_matches('/')
                .starts_with(AGENT_CONTAINER_PREFIX)
        {
            let _ = client.remove_container(&c.id);
            if !c.name.is_empty() {
                let _ = client.remove_container(&c.name);
            }
            n = n.saturating_add(1);
        }
    }
    let _ = OWNED_NAME_PREFIX; // document relationship
    Ok(n)
}

/// Count agent-owned containers (QA helper).
///
/// # Errors
/// Docker list failures.
pub fn count_agent_containers(docker_base: &str) -> Result<usize, DockerError> {
    let client = AllowlistClient::with_allowlist(docker_base, Allowlist::verifier())?;
    let owned = client.list_owned()?;
    Ok(owned
        .iter()
        .filter(|c| {
            let n = c.name.trim_start_matches('/');
            n.starts_with(AGENT_CONTAINER_PREFIX)
        })
        .count())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_dispatch::TaskStatusV1;

    #[test]
    fn resolve_timeout_zero_when_deadline_passed() {
        assert_eq!(resolve_timeout_sec(1000, 60, 2000), 0);
    }

    #[test]
    fn resolve_timeout_clamps_to_pack_budget() {
        let now = 1_000_000u64;
        // 100s remaining, pack 10s → 10
        assert_eq!(resolve_timeout_sec(now + 100_000, 10, now), 10);
        // 5s remaining, pack 60s → 5
        assert_eq!(resolve_timeout_sec(now + 5_000, 60, now), 5);
    }

    #[test]
    fn stub_respects_past_deadline() {
        let o = execute_pack(
            &ExecutionBackend::Stub {
                hold: Duration::ZERO,
            },
            "pack-x",
            1, // long past
        );
        assert_eq!(o.status, TaskStatusV1::TimedOut);
        assert!(o.model_patch.is_none());
    }

    #[test]
    fn stub_completes_with_diff_when_deadline_far() {
        let o = execute_pack(
            &ExecutionBackend::Stub {
                hold: Duration::ZERO,
            },
            "pack-y",
            unix_now_ms() + 60_000,
        );
        assert_eq!(o.status, TaskStatusV1::Completed);
        let p = o.model_patch.expect("patch");
        assert!(p.contains("diff --git"));
        assert!(p.contains("pack-y"));
    }

    #[test]
    fn stub_hold_past_budget_times_out() {
        let deadline = unix_now_ms() + 200;
        let o = execute_pack(
            &ExecutionBackend::Stub {
                hold: Duration::from_millis(800),
            },
            "pack-z",
            deadline,
        );
        assert_eq!(o.status, TaskStatusV1::TimedOut);
        assert!(o.model_patch.is_none());
    }

    #[test]
    fn reference_agent_cmd_mentions_pack_and_not_key_bytes() {
        let cmd = reference_agent_cmd("realpr-httpx-3672");
        assert_eq!(cmd[0], "bash");
        let script = &cmd[2];
        assert!(script.contains("model.patch"));
        assert!(script.contains("realpr-httpx-3672"));
        assert!(script.contains("MODEL_KEY_FILE"));
        // Must not embed a sample secret.
        assert!(!script.contains("sk-"));
    }

    #[test]
    fn agent_name_uses_owned_prefix() {
        let n = unique_agent_name("foo/bar");
        assert!(n.starts_with(AGENT_CONTAINER_PREFIX));
        assert!(n.starts_with(OWNED_NAME_PREFIX));
    }
}
