//! SSH target parsing + remote exec for Lium pods (no secrets in logs).

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde_json::Value;
use tokio::process::Command;
use tokio::time::sleep;
use tracing::debug;

use crate::error::LiumError;

/// Parsed SSH connect target from Lium pod payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SshTarget {
    pub user: String,
    pub host: String,
    pub port: u16,
}

/// Parse `ssh_connect_cmd` and optional `ports_mapping` from pod JSON.
#[must_use]
pub fn parse_ssh_target(ssh_connect_cmd: &str, raw: &Value) -> Option<SshTarget> {
    let user_host = regex_user_host(ssh_connect_cmd);
    let mut port = regex_port(ssh_connect_cmd);
    if port.is_none() {
        if let Some(mapping) = raw.get("ports_mapping") {
            let external = mapping
                .get("22")
                .or_else(|| mapping.get(22.to_string().as_str()));
            // serde_json object keys are strings; also try numeric-as-string
            let external = external.or_else(|| {
                mapping.as_object().and_then(|m| {
                    m.iter()
                        .find(|(k, _)| k == &"22" || k.as_str() == "22")
                        .map(|(_, v)| v)
                })
            });
            if let Some(ext) = external {
                port = ext
                    .as_u64()
                    .or_else(|| ext.as_str().and_then(|s| s.parse().ok()))
                    .map(|p| p as u16);
            }
        }
    }
    let (user, host) = user_host?;
    Some(SshTarget {
        user,
        host,
        port: port.unwrap_or(22),
    })
}

fn regex_user_host(cmd: &str) -> Option<(String, String)> {
    // user@host
    let bytes = cmd.as_bytes();
    let at = bytes.iter().position(|&b| b == b'@')?;
    // walk left for user
    let mut start = at;
    while start > 0 {
        let c = bytes[start - 1];
        if c.is_ascii_alphanumeric() || c == b'.' || c == b'-' || c == b'_' {
            start -= 1;
        } else {
            break;
        }
    }
    if start == at {
        return None;
    }
    let user = cmd[start..at].to_owned();
    // walk right for host
    let mut end = at + 1;
    while end < bytes.len() {
        let c = bytes[end];
        if c.is_ascii_alphanumeric() || c == b'.' || c == b'-' {
            end += 1;
        } else {
            break;
        }
    }
    if end == at + 1 {
        return None;
    }
    let host = cmd[at + 1..end].to_owned();
    Some((user, host))
}

fn regex_port(cmd: &str) -> Option<u16> {
    // -p NNNN
    let bytes = cmd.as_bytes();
    let mut i = 0;
    while i + 2 < bytes.len() {
        if bytes[i] == b'-' && bytes[i + 1] == b'p' {
            let mut j = i + 2;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            let start = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > start {
                if let Ok(p) = cmd[start..j].parse::<u16>() {
                    return Some(p);
                }
            }
        }
        i += 1;
    }
    None
}

/// Resolve private key path from explicit path or `LIUM_SSH_PRIVATE_KEY` env.
pub fn resolve_private_key(explicit: Option<&Path>) -> Result<PathBuf, LiumError> {
    if let Some(p) = explicit {
        if p.is_file() {
            return Ok(p.to_path_buf());
        }
        return Err(LiumError::Exec(format!(
            "ssh private key missing: {}",
            p.display()
        )));
    }
    if let Ok(p) = std::env::var("LIUM_SSH_PRIVATE_KEY") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Ok(pb);
        }
        return Err(LiumError::Exec(format!(
            "LIUM_SSH_PRIVATE_KEY not a file: {}",
            pb.display()
        )));
    }
    let default = PathBuf::from("/root/.config/prism-mission/lium_ssh_ed25519");
    if default.is_file() {
        return Ok(default);
    }
    Err(LiumError::Exec(
        "no SSH private key: set LIUM_SSH_PRIVATE_KEY or pass path".into(),
    ))
}

/// Run a remote command over SSH with retries.
pub async fn ssh_exec(
    target: &SshTarget,
    private_key: &Path,
    remote_cmd: &str,
    attempts: u32,
    retry_secs: u64,
    timeout_secs: u64,
) -> Result<SshExecOutput, LiumError> {
    let mut last_err = String::new();
    for attempt in 1..=attempts.max(1) {
        let mut cmd = Command::new("ssh");
        cmd.arg("-i")
            .arg(private_key)
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("UserKnownHostsFile=/dev/null")
            .arg("-o")
            .arg("ConnectTimeout=15")
            .arg("-o")
            .arg("ServerAliveInterval=10")
            .arg("-o")
            .arg("ServerAliveCountMax=60")
            .arg("-o")
            .arg("TCPKeepAlive=yes")
            .arg("-o")
            .arg("BatchMode=yes")
            .arg("-p")
            .arg(target.port.to_string())
            .arg(format!("{}@{}", target.user, target.host))
            .arg(remote_cmd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let fut = cmd.output();
        let result = tokio::time::timeout(Duration::from_secs(timeout_secs), fut).await;
        match result {
            Ok(Ok(out)) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                debug!(
                    attempt,
                    code = out.status.code().unwrap_or(-1),
                    stdout_len = stdout.len(),
                    "ssh exec finished"
                );
                if out.status.success() {
                    return Ok(SshExecOutput {
                        returncode: out.status.code().unwrap_or(0),
                        stdout,
                        stderr: truncate_str(&stderr, 4000),
                    });
                }
                last_err = format!(
                    "ssh exit {:?}: {}",
                    out.status.code(),
                    truncate_str(&stderr, 200)
                );
            }
            Ok(Err(e)) => {
                last_err = format!("ssh spawn: {e}");
            }
            Err(_) => {
                last_err = "ssh timed out".into();
            }
        }
        if attempt < attempts {
            sleep(Duration::from_secs(retry_secs)).await;
        }
    }
    Err(LiumError::Exec(last_err))
}

/// Run SSH and return stdout/stderr even when the remote exit code is non-zero.
///
/// Still errors on spawn failure / timeout after retries. Used when the caller
/// wants to interpret remote failure (e.g. missing torch) and fall back.
pub async fn ssh_exec_allow_fail(
    target: &SshTarget,
    private_key: &Path,
    remote_cmd: &str,
    attempts: u32,
    retry_secs: u64,
    timeout_secs: u64,
) -> Result<SshExecOutput, LiumError> {
    let mut last_err = String::new();
    for attempt in 1..=attempts.max(1) {
        let mut cmd = Command::new("ssh");
        cmd.arg("-i")
            .arg(private_key)
            .arg("-o")
            .arg("StrictHostKeyChecking=no")
            .arg("-o")
            .arg("UserKnownHostsFile=/dev/null")
            .arg("-o")
            .arg("ConnectTimeout=15")
            .arg("-o")
            .arg("ServerAliveInterval=10")
            .arg("-o")
            .arg("ServerAliveCountMax=60")
            .arg("-o")
            .arg("TCPKeepAlive=yes")
            .arg("-o")
            .arg("BatchMode=yes")
            .arg("-p")
            .arg(target.port.to_string())
            .arg(format!("{}@{}", target.user, target.host))
            .arg(remote_cmd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let fut = cmd.output();
        let result = tokio::time::timeout(Duration::from_secs(timeout_secs), fut).await;
        match result {
            Ok(Ok(out)) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                return Ok(SshExecOutput {
                    returncode: out.status.code().unwrap_or(-1),
                    stdout,
                    stderr: truncate_str(&stderr, 4000),
                });
            }
            Ok(Err(e)) => {
                last_err = format!("ssh spawn: {e}");
            }
            Err(_) => {
                last_err = "ssh timed out".into();
            }
        }
        if attempt < attempts {
            sleep(Duration::from_secs(retry_secs)).await;
        }
    }
    Err(LiumError::Exec(last_err))
}

/// Successful SSH run output.
#[derive(Debug, Clone)]
pub struct SshExecOutput {
    #[allow(dead_code)]
    pub returncode: i32,
    pub stdout: String,
    pub stderr: String,
}

fn truncate_str(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_owned()
    } else {
        format!("{}…", &s[..n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_ssh_connect_cmd() {
        let raw = json!({});
        let t = parse_ssh_target("ssh root@216.81.200.32 -p 40298", &raw).unwrap();
        assert_eq!(t.user, "root");
        assert_eq!(t.host, "216.81.200.32");
        assert_eq!(t.port, 40298);
    }

    #[test]
    fn parse_port_from_mapping() {
        let raw = json!({"ports_mapping": {"22": 22001}});
        let t = parse_ssh_target("ssh ubuntu@10.0.0.1", &raw).unwrap();
        assert_eq!(t.port, 22001);
        assert_eq!(t.user, "ubuntu");
    }
}
