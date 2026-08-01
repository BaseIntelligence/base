//! `agent-runner` — miner CVM HTTP task API (`agent:8080`).
//!
//! Loads the CVM-local work-receipt key from `BASE_RECEIPT_SK_FILE` (mode 0600
//! mount). Dispatch auth (todo 18) is on by default when a trusted challenge
//! pubkey is configured. Concurrency is clamped to 1..=5 and enforced with a
//! semaphore (todo 19). Pack execution uses allowlisted Docker when
//! `BASE_DOCKER_BASE` + `BASE_ENVIRONMENT_IMAGE` + `BASE_PACK_ROOT` are set;
//! otherwise the deterministic stub backend is used. Default egress posture is
//! OPEN (todo 21).

#![forbid(unsafe_code)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use agent_runner::{
    app, clamp_concurrency, load_or_generate, load_required, receipt_sk_path_from_env,
    AgentEgressPosture, DockerExecConfig, ExecutionBackend, RunnerConfig, RunnerState,
    DEFAULT_AGENT_EGRESS_POSTURE, DEFAULT_DISPATCH_NONCE_TTL, DEFAULT_RECEIPT_SK_PATH,
    RECEIPT_SK_FILE_ENV,
};
use clap::Parser;
use crypto::KEY_LEN;
use tokio::net::TcpListener;

/// Miner agent-runner CLI.
#[derive(Debug, Parser)]
#[command(
    name = "agent-runner",
    about = "Miner CVM agent task API (capacity + dispatch + pack execution + work-receipt)"
)]
struct Cli {
    /// Bind address (compose publishes agent:8080).
    #[arg(long, env = "BASE_RUNNER_BIND", default_value = "0.0.0.0:8080")]
    bind: SocketAddr,
    /// Miner-declared max concurrency (clamped to 1..=5 at runtime).
    #[arg(long, env = "BASE_MAX_CONCURRENCY", default_value_t = 1)]
    max_concurrency: u32,
    /// Path to the CVM-local receipt mini-secret (mode 0600 file).
    #[arg(long, env = "BASE_RECEIPT_SK_FILE", default_value = DEFAULT_RECEIPT_SK_PATH)]
    receipt_sk_file: PathBuf,
    /// When set, generate the receipt key if the file is missing (local/dev only).
    #[arg(long, env = "BASE_RECEIPT_SK_GENERATE", default_value_t = false)]
    receipt_sk_generate: bool,
    /// Disable dispatch auth (local/dev only). Default: auth on when pubkey set.
    #[arg(long, env = "BASE_DISPATCH_AUTH_DISABLE", default_value_t = false)]
    dispatch_auth_disable: bool,
    /// Trusted challenge public key (64 hex) for dispatch auth.
    #[arg(long, env = "BASE_TRUSTED_CHALLENGE_PUBKEY")]
    trusted_challenge_pubkey: Option<String>,
    /// Docker Engine HTTP base (socket-proxy). When set with image + pack root → Docker backend.
    #[arg(long, env = "BASE_DOCKER_BASE")]
    docker_base: Option<String>,
    /// Digest-pinned environment image for pack runs (`name@sha256:…`).
    #[arg(long, env = "BASE_ENVIRONMENT_IMAGE")]
    environment_image: Option<String>,
    /// Host directory of Harbor packs (`{root}/{pack_id}/`).
    #[arg(long, env = "BASE_PACK_ROOT")]
    pack_root: Option<PathBuf>,
    /// Pack catalog HTTP base (challenge via gateway), e.g. <http://gateway:8080/challenge/agent-v1>
    #[arg(long, env = "BASE_PACK_CATALOG_URL")]
    pack_catalog_url: Option<String>,
    /// Staging root for agent binds.
    #[arg(
        long,
        env = "BASE_AGENT_WORK_ROOT",
        default_value = "/tmp/base-agent-work"
    )]
    work_root: PathBuf,
    /// Miner-supplied model API key file (mounted into agent; never logged).
    #[arg(long, env = "BASE_MODEL_KEY_FILE")]
    model_key_file: Option<PathBuf>,
    /// Real agent command as a JSON array, e.g. `["bash","-lc","python /agent/run.py"]`.
    /// Replaces the built-in reference command, which is a placeholder that
    /// writes a canned patch; without this flag no miner can run a real agent.
    #[arg(long, env = "BASE_AGENT_CMD")]
    agent_cmd: Option<String>,
    /// Egress posture: `open` (default) or `allowlisted_proxy`.
    #[arg(long, env = "BASE_AGENT_EGRESS", default_value = "open")]
    egress: String,
}

fn main() -> ExitCode {
    let _ = telemetry::init_tracing();
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("agent-runner: {msg}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(serve(cli))
}

fn parse_pubkey_hex(s: &str) -> Result<[u8; KEY_LEN], String> {
    let bytes = hex::decode(s.trim()).map_err(|e| format!("trusted pubkey hex: {e}"))?;
    if bytes.len() != KEY_LEN {
        return Err(format!("trusted pubkey must be {KEY_LEN} bytes"));
    }
    let mut out = [0u8; KEY_LEN];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn parse_egress(s: &str) -> Result<AgentEgressPosture, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "open" | "" => Ok(AgentEgressPosture::Open),
        "allowlisted_proxy" | "proxy" => Ok(AgentEgressPosture::AllowlistedProxy),
        other => Err(format!(
            "unknown egress posture {other:?} (expected open|allowlisted_proxy)"
        )),
    }
}

fn parse_agent_cmd(raw: &str) -> Result<Vec<String>, String> {
    let value: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| format!("BASE_AGENT_CMD is not JSON: {e}"))?;
    let items = value
        .as_array()
        .ok_or_else(|| "BASE_AGENT_CMD must be a JSON array of strings".to_owned())?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let s = item
            .as_str()
            .ok_or_else(|| "BASE_AGENT_CMD entries must be strings".to_owned())?;
        out.push(s.to_owned());
    }
    if out.is_empty() {
        return Err("BASE_AGENT_CMD must not be an empty array".into());
    }
    Ok(out)
}

fn build_execution(cli: &Cli) -> Result<ExecutionBackend, String> {
    match (
        cli.docker_base.as_ref(),
        cli.environment_image.as_ref(),
        cli.pack_root.as_ref(),
    ) {
        (Some(base), Some(image), Some(root)) => {
            if image.is_empty() {
                return Err("BASE_ENVIRONMENT_IMAGE must be non-empty".into());
            }
            if let Some(key) = &cli.model_key_file {
                if !key.is_file() {
                    return Err(format!(
                        "BASE_MODEL_KEY_FILE not a file: {}",
                        key.display()
                    ));
                }
            }
            let agent_cmd = cli
                .agent_cmd
                .as_deref()
                .map(parse_agent_cmd)
                .transpose()?;
            Ok(ExecutionBackend::Docker(DockerExecConfig {
                docker_base: base.clone(),
                environment_image: image.clone(),
                pack_root: root.clone(),
                work_root: cli.work_root.clone(),
                model_key_path: cli.model_key_file.clone(),
                egress: parse_egress(&cli.egress)?,
                agent_cmd,
                pack_catalog_url: cli.pack_catalog_url.clone(),
            }))
        }
        (None, None, None) => Ok(ExecutionBackend::Stub {
            hold: Duration::ZERO,
        }),
        _ => Err(
            "Docker pack execution requires BASE_DOCKER_BASE + BASE_ENVIRONMENT_IMAGE + BASE_PACK_ROOT (or omit all three for stub)"
                .into(),
        ),
    }
}

async fn serve(cli: Cli) -> Result<(), String> {
    let path = if cli.receipt_sk_file.as_os_str().is_empty() {
        receipt_sk_path_from_env()
    } else {
        cli.receipt_sk_file.clone()
    };
    let key = if cli.receipt_sk_generate {
        load_or_generate(&path).map_err(|e| e.to_string())?
    } else {
        load_required(&path).map_err(|e| {
            format!(
                "{e} (path={}, set {RECEIPT_SK_FILE_ENV} or pass --receipt-sk-generate for local)",
                path.display()
            )
        })?
    };
    let public_hex = key.public_key_hex();

    let trusted = match &cli.trusted_challenge_pubkey {
        Some(h) => Some(parse_pubkey_hex(h)?),
        None => None,
    };
    let auth_enabled = !cli.dispatch_auth_disable;
    if auth_enabled && trusted.is_none() {
        return Err(
            "dispatch auth enabled but BASE_TRUSTED_CHALLENGE_PUBKEY unset (or pass --dispatch-auth-disable)"
                .into(),
        );
    }

    let egress_posture = parse_egress(&cli.egress)?;
    let execution = build_execution(&cli)?;
    let declared = cli.max_concurrency;
    let effective = clamp_concurrency(declared);
    let state = RunnerState::new(RunnerConfig {
        max_concurrency: declared,
        auth_enabled,
        trusted_challenge_pubkey: trusted,
        dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
        receipt_key: Some(key),
        execution,
        egress_posture,
    });
    let router = app(state);
    let listener = TcpListener::bind(cli.bind)
        .await
        .map_err(|e| format!("bind {}: {e}", cli.bind))?;
    let actual = listener.local_addr().map_err(|e| e.to_string())?;
    tracing::info!(
        event = "agent_runner_listening",
        %actual,
        declared_max_concurrency = declared,
        effective_max_concurrency = effective,
        receipt_public_key_hex = %public_hex,
        auth_enabled,
        egress_posture = egress_posture.as_str(),
        default_egress = DEFAULT_AGENT_EGRESS_POSTURE.as_str(),
        "agent-runner HTTP surface"
    );
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_agent_cmd;

    #[test]
    fn agent_cmd_json_array_of_strings() {
        let cmd = parse_agent_cmd(r#"["bash","-lc","echo hi"]"#).expect("ok");
        assert_eq!(cmd, vec!["bash", "-lc", "echo hi"]);
    }

    #[test]
    fn agent_cmd_rejects_non_array_and_empty_array_and_non_strings() {
        for raw in ["{}", "[]", "[\"bash\", 1]", "not json"] {
            assert!(
                parse_agent_cmd(raw).is_err(),
                "{raw} must be rejected: an invalid agent command silently changes \
                 the measured compose away from what the operator signed"
            );
        }
    }
}
