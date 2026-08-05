//! Two-phase design sandbox (install → run) via docker-engine, plus SimSandbox.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::implicit_hasher)]
#![allow(clippy::doc_markdown)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use design_harness::{stage_work_files, HarnessBundle, REQUIRED_PAGES};
use docker_engine::{AllowlistClient, RunResult, RunSpec, DESIGN_OWNED_NAME_PREFIX};
use thiserror::Error;

/// Default design-runtime image (compose / local tag; digest pins via GHCR).
pub const DEFAULT_RUNTIME_IMAGE: &str = "design-runtime:0.1.0";

/// Sandbox errors.
#[derive(Debug, Error)]
pub enum SandboxError {
    /// IO.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Docker.
    #[error("docker: {0}")]
    Docker(String),
    /// Phase failed.
    #[error("phase {phase}: exit={status}: {logs}")]
    PhaseFailed {
        /// install|run.
        phase: &'static str,
        /// Exit code.
        status: i64,
        /// Logs.
        logs: String,
    },
    /// Missing output.
    #[error("missing output: {0}")]
    MissingOutput(String),
}

/// Collected `/out` pages.
#[derive(Debug, Clone)]
pub struct SandboxOutput {
    /// Page path → HTML.
    pub pages: HashMap<String, String>,
    /// Install logs.
    pub install_logs: String,
    /// Run logs.
    pub run_logs: String,
}

/// Work root after a successful install phase (or sim staging).
#[derive(Debug)]
pub struct SandboxSession {
    /// Staging directory for this run.
    pub work_root: PathBuf,
    /// Install-phase stdout/stderr.
    pub install_logs: String,
}

/// Execution backend.
pub trait SandboxBackend: Send + Sync + std::fmt::Debug {
    /// Install phase (venv + pip). Emits progress between phases when used by orchestrator.
    ///
    /// # Errors
    /// Backend failures.
    fn install(
        &self,
        bundle: &HarnessBundle,
        round_id: u64,
        run_id: &str,
        prompt: &str,
        llm_proxy: &str,
    ) -> Result<SandboxSession, SandboxError>;

    /// Run phase on a prepared session.
    ///
    /// # Errors
    /// Backend failures.
    fn run_session(
        &self,
        session: SandboxSession,
        round_id: u64,
        run_id: &str,
        prompt: &str,
        llm_proxy: &str,
    ) -> Result<SandboxOutput, SandboxError>;

    /// Install then run (convenience for tests).
    ///
    /// # Errors
    /// Backend failures.
    fn execute(
        &self,
        bundle: &HarnessBundle,
        round_id: u64,
        run_id: &str,
        prompt: &str,
        llm_proxy: &str,
    ) -> Result<SandboxOutput, SandboxError> {
        let session = self.install(bundle, round_id, run_id, prompt, llm_proxy)?;
        let install_logs = session.install_logs.clone();
        let mut out = self.run_session(session, round_id, run_id, prompt, llm_proxy)?;
        if out.install_logs.is_empty() {
            out.install_logs = install_logs;
        }
        Ok(out)
    }
}

/// Live Docker two-phase sandbox.
#[derive(Debug)]
pub struct DockerSandbox {
    /// Engine client.
    pub client: AllowlistClient,
    /// Image ref.
    pub image: String,
    /// Network mode.
    pub network_mode: String,
    /// Install timeout secs.
    pub install_timeout_sec: u64,
    /// Run timeout secs.
    pub run_timeout_sec: u64,
    /// Staging root.
    pub staging_root: PathBuf,
}

impl DockerSandbox {
    /// New with verifier allowlist client.
    ///
    /// # Errors
    /// Client build.
    pub fn new(docker_base: &str, staging_root: PathBuf) -> Result<Self, SandboxError> {
        let client =
            AllowlistClient::with_allowlist(docker_base, docker_engine::Allowlist::verifier())
                .map_err(|e| SandboxError::Docker(e.to_string()))?;
        Ok(Self {
            client,
            image: DEFAULT_RUNTIME_IMAGE.into(),
            network_mode: "design-sandbox-egress".into(),
            install_timeout_sec: 180,
            run_timeout_sec: design_challenge_task::agent_run_timeout_secs(),
            staging_root,
        })
    }

    fn stage(&self, bundle: &HarnessBundle, run_id: &str) -> Result<PathBuf, SandboxError> {
        let work = self.staging_root.join(run_id);
        let _ = fs::remove_dir_all(&work);
        fs::create_dir_all(work.join("out/pages"))?;
        fs::create_dir_all(work.join("out/assets"))?;
        let staged = stage_work_files(bundle);
        for (rel, body) in staged.files {
            let path = work.join(&rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, body)?;
        }
        // Side-channel for run-phase injection; removed after apply (never logged).
        if !bundle.env_vars.is_empty() {
            let raw = serde_json::to_vec(&bundle.env_vars)
                .map_err(|e| SandboxError::Docker(format!("env encode: {e}")))?;
            fs::write(work.join(".miner_env.json"), raw)?;
        }
        Ok(work)
    }

    fn run_phase(
        &self,
        name: &str,
        work: &Path,
        cmd: Vec<String>,
        env: Vec<String>,
        timeout: u64,
        readonly: bool,
    ) -> Result<RunResult, SandboxError> {
        let mut spec = RunSpec::design_hardened(name.to_owned(), self.image.clone(), cmd);
        spec.binds = vec![format!("{}:/work:rw", work.display())];
        spec.env = env;
        spec.network_mode = Some(self.network_mode.clone());
        spec.timeout_sec = Some(timeout);
        spec.readonly_rootfs = readonly;
        spec.working_dir = Some("/work".into());
        self.client
            .run_owned(&spec)
            .map_err(|e| SandboxError::Docker(e.to_string()))
    }

    fn collect_out(work: &Path) -> Result<HashMap<String, String>, SandboxError> {
        let mut pages = HashMap::new();
        for req in REQUIRED_PAGES {
            let p = work.join("out/pages").join(req);
            if !p.is_file() {
                return Err(SandboxError::MissingOutput((*req).into()));
            }
            pages.insert((*req).into(), fs::read_to_string(p)?);
        }
        Ok(pages)
    }
}

impl SandboxBackend for DockerSandbox {
    fn install(
        &self,
        bundle: &HarnessBundle,
        _round_id: u64,
        run_id: &str,
        _prompt: &str,
        _llm_proxy: &str,
    ) -> Result<SandboxSession, SandboxError> {
        let work = self.stage(bundle, run_id)?;
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let install_name = format!("{DESIGN_OWNED_NAME_PREFIX}inst-{run_id}-{ns}");
        // Create a venv, then `pip install -e`. When the harness declares no
        // third-party deps, allow install to succeed even if PyPI egress is
        // unavailable — `design_harness.py` loads `agent.py` by path.
        let install = self.run_phase(
            &install_name,
            &work,
            vec![
                "bash".into(),
                "-lc".into(),
                r#"set -e
python -m venv /work/.venv
if /work/.venv/bin/pip install -e /work; then
  echo pip-install-ok
  exit 0
fi
python - <<'PY'
import pathlib, sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore
raw = pathlib.Path("/work/pyproject.toml").read_bytes()
deps = tomllib.loads(raw.decode()).get("project", {}).get("dependencies") or []
if deps:
    print("pip-install-failed with dependencies:", deps, file=sys.stderr)
    sys.exit(1)
print("pip-install-skipped-no-deps")
PY"#
                .into(),
            ],
            vec![
                "HTTP_PROXY=http://design-egress-proxy:8094".into(),
                "HTTPS_PROXY=http://design-egress-proxy:8094".into(),
                "DESIGN_PROXY_MODE=pypi".into(),
            ],
            self.install_timeout_sec,
            false,
        )?;
        if install.status_code != 0 {
            return Err(SandboxError::PhaseFailed {
                phase: "install",
                status: install.status_code,
                logs: install.logs,
            });
        }
        Ok(SandboxSession {
            work_root: work,
            install_logs: install.logs,
        })
    }

    fn run_session(
        &self,
        session: SandboxSession,
        round_id: u64,
        run_id: &str,
        prompt: &str,
        llm_proxy: &str,
    ) -> Result<SandboxOutput, SandboxError> {
        let install_logs = session.install_logs;
        let work = session.work_root;
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let run_name = format!("{DESIGN_OWNED_NAME_PREFIX}run-{run_id}-{ns}");
        // Miner env is applied only on the run phase. Values are never logged.
        let mut env = vec![
            format!("DESIGN_LLM_PROXY={llm_proxy}"),
            format!("DESIGN_RUN_ID={run_id}"),
            format!("DESIGN_ROUND_ID={round_id}"),
            format!("DESIGN_PROMPT={prompt}"),
            "DESIGN_WORK_ROOT=/work".into(),
            "DESIGN_OUT_ROOT=/work/out".into(),
            "DESIGN_PROXY_MODE=llm".into(),
            "HTTP_PROXY=http://design-egress-proxy:8094".into(),
            "HTTPS_PROXY=http://design-egress-proxy:8094".into(),
        ];
        // Re-read env from staged files is not possible; callers must put
        // validated env on the bundle before install/run. We pass via a
        // side channel file written at stage time when present.
        if let Ok(raw) = fs::read_to_string(work.join(".miner_env.json")) {
            if let Ok(map) =
                serde_json::from_str::<std::collections::BTreeMap<String, String>>(&raw)
            {
                for (k, v) in map {
                    env.push(format!("{k}={v}"));
                }
            }
            let _ = fs::remove_file(work.join(".miner_env.json"));
        }
        let run = self.run_phase(
            &run_name,
            &work,
            vec![
                "bash".into(),
                "-lc".into(),
                "/work/.venv/bin/python /work/design_harness.py".into(),
            ],
            env,
            self.run_timeout_sec,
            true,
        )?;
        if run.status_code != 0 {
            return Err(SandboxError::PhaseFailed {
                phase: "run",
                status: run.status_code,
                logs: run.logs,
            });
        }
        Ok(SandboxOutput {
            pages: Self::collect_out(&work)?,
            install_logs,
            run_logs: run.logs,
        })
    }
}

/// CI / offline sandbox: stages the real harness and runs `agent.py` via host Python.
///
/// A local stub LLM on loopback answers `POST /v1/chat` with HTML so the SDK path
/// (`base_design.LlmClient` → `out.write_page`) is exercised without Docker.
#[derive(Debug, Default)]
pub struct SimSandbox;

impl SimSandbox {
    /// New.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn stage(bundle: &HarnessBundle, run_id: &str) -> Result<PathBuf, SandboxError> {
        let root = std::env::temp_dir().join(format!("design-sim-{run_id}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("out/pages"))?;
        fs::create_dir_all(root.join("out/assets"))?;
        for (rel, body) in stage_work_files(bundle).files {
            let path = root.join(&rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, body)?;
        }
        if !bundle.env_vars.is_empty() {
            let raw = serde_json::to_vec(&bundle.env_vars)
                .map_err(|e| SandboxError::Docker(format!("env encode: {e}")))?;
            fs::write(root.join(".miner_env.json"), raw)?;
        }
        Ok(root)
    }

    fn python_bin() -> Result<&'static str, SandboxError> {
        for cand in ["python3", "python"] {
            if std::process::Command::new(cand)
                .arg("-V")
                .output()
                .is_ok_and(|o| o.status.success())
            {
                return Ok(cand);
            }
        }
        Err(SandboxError::MissingOutput(
            "python3 required for SimSandbox harness execution".into(),
        ))
    }
}

impl SandboxBackend for SimSandbox {
    fn install(
        &self,
        bundle: &HarnessBundle,
        _round_id: u64,
        run_id: &str,
        _prompt: &str,
        _llm_proxy: &str,
    ) -> Result<SandboxSession, SandboxError> {
        let work = Self::stage(bundle, run_id)?;
        Ok(SandboxSession {
            work_root: work,
            install_logs: "sim-install-ok (staged agent.py + base_design)".into(),
        })
    }

    fn run_session(
        &self,
        session: SandboxSession,
        round_id: u64,
        run_id: &str,
        prompt: &str,
        _llm_proxy: &str,
    ) -> Result<SandboxOutput, SandboxError> {
        let work = session.work_root;
        if work.as_os_str().is_empty() {
            return Err(SandboxError::MissingOutput("sim work root".into()));
        }
        let (proxy_url, stub) = spawn_stub_llm_proxy()?;
        let py = Self::python_bin()?;
        let mut cmd = std::process::Command::new(py);
        cmd.current_dir(&work)
            .env("PYTHONPATH", &work)
            .env("DESIGN_WORK_ROOT", &work)
            .env("DESIGN_OUT_ROOT", work.join("out"))
            .env("DESIGN_LLM_PROXY", &proxy_url)
            .env("DESIGN_RUN_ID", run_id)
            .env("DESIGN_ROUND_ID", round_id.to_string())
            .env("DESIGN_PROMPT", prompt);
        if let Ok(raw) = fs::read_to_string(work.join(".miner_env.json")) {
            if let Ok(map) =
                serde_json::from_str::<std::collections::BTreeMap<String, String>>(&raw)
            {
                for (k, v) in map {
                    cmd.env(k, v);
                }
            }
            let _ = fs::remove_file(work.join(".miner_env.json"));
        }
        let out = cmd
            .arg(work.join("design_harness.py"))
            .output()
            .map_err(|e| SandboxError::Docker(format!("spawn python: {e}")))?;
        drop(stub);
        let logs = format!(
            "sim-run-ok\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if !out.status.success() {
            return Err(SandboxError::PhaseFailed {
                phase: "run",
                status: i64::from(out.status.code().unwrap_or(1)),
                logs,
            });
        }
        Ok(SandboxOutput {
            pages: DockerSandbox::collect_out(&work)?,
            install_logs: session.install_logs,
            run_logs: logs,
        })
    }
}

/// Tiny loopback chat stub used only by [`SimSandbox`].
struct StubLlmGuard {
    addr: std::net::SocketAddr,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Drop for StubLlmGuard {
    fn drop(&mut self) {
        // Wake a blocked `accept` so the stub thread can exit.
        let _ = std::net::TcpStream::connect(self.addr);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

fn spawn_stub_llm_proxy() -> Result<(String, StubLlmGuard), SandboxError> {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").map_err(SandboxError::Io)?;
    let addr = listener.local_addr().map_err(SandboxError::Io)?;
    let join = std::thread::spawn(move || {
        // Baseline agent makes one chat call per required page (+ wake-up connect).
        for _ in 0..16 {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            let mut buf = vec![0_u8; 65536];
            let n = stream.read(&mut buf).unwrap_or(0);
            if n == 0 {
                break;
            }
            let req = String::from_utf8_lossy(&buf[..n]);
            let body_start = req.find("\r\n\r\n").map_or(req.len(), |i| i + 4);
            let payload = &req[body_start..];
            let page = REQUIRED_PAGES
                .iter()
                .find(|p| payload.contains(*p))
                .copied()
                .unwrap_or("index.html");
            let html = format!(
                "<!DOCTYPE html><html data-agent=\"sim-stub\" data-page=\"{page}\">\
                 <head><meta charset=\"utf-8\"/><title>{page}</title></head>\
                 <body><h1>{page}</h1><p>sim stub llm</p></body></html>"
            );
            let content = serde_json::json!({ "content": html }).to_string();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                content.len(),
                content
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    Ok((
        format!("http://{addr}"),
        StubLlmGuard {
            addr,
            join: Some(join),
        },
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use design_harness::HarnessBundle;
    use std::collections::BTreeMap;

    const BASELINE_AGENT: &str =
        include_str!("../../../docs/external-miner/examples/design-baseline/agent.py");
    const BASELINE_PYPROJECT: &str =
        include_str!("../../../docs/external-miner/examples/design-baseline/pyproject.toml");

    #[test]
    fn sim_runs_baseline_agent_py() {
        let b = HarnessBundle {
            miner_hotkey: "ab".repeat(32),
            agent_py: BASELINE_AGENT.into(),
            pyproject_toml: BASELINE_PYPROJECT.into(),
            extra_files: BTreeMap::new(),
            env_vars: BTreeMap::new(),
        };
        let out = SimSandbox::new()
            .execute(
                &b,
                1,
                "run-baseline",
                "Build a three-page landing for a B2B analytics SaaS",
                "http://unused",
            )
            .unwrap();
        assert_eq!(out.pages.len(), 3);
        assert!(out.pages.contains_key("index.html"));
        assert!(out.install_logs.contains("sim-install"));
        assert!(out.run_logs.contains("sim-run-ok"));
        let index = out.pages.get("index.html").unwrap();
        assert!(
            index.contains("design-baseline") || index.contains("sim-stub"),
            "expected baseline/stub marker in generated HTML"
        );
        assert!(index.to_lowercase().contains("<html"));
    }
}
