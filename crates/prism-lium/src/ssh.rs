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
    let (user, host) = regex_user_host(ssh_connect_cmd)?;
    let mut port = regex_port(ssh_connect_cmd);
    if port.is_none() {
        // serde_json object keys are strings; `/ports_mapping/22` covers both
        // numeric and string-typed external port values.
        port = raw
            .pointer("/ports_mapping/22")
            .and_then(|ext| {
                ext.as_u64()
                    .or_else(|| ext.as_str().and_then(|s| s.parse().ok()))
            })
            .map(|p| p as u16);
    }
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
    let env = std::env::var("LIUM_SSH_PRIVATE_KEY").ok();
    let candidate = explicit
        .map(Path::to_path_buf)
        .or_else(|| env.as_deref().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("/root/.config/prism-mission/lium_ssh_ed25519"));
    if candidate.is_file() {
        return Ok(candidate);
    }
    if explicit.is_some() || env.is_some() {
        return Err(LiumError::Exec(format!(
            "ssh private key missing: {}",
            candidate.display()
        )));
    }
    Err(LiumError::Exec(
        "no SSH private key: set LIUM_SSH_PRIVATE_KEY or pass path".into(),
    ))
}

/// Shared ssh invocation (BatchMode, keepalives, piped stdout/stderr).
fn ssh_command(target: &SshTarget, private_key: &Path, remote_cmd: &str) -> Command {
    let port = target.port.to_string();
    let mut cmd = Command::new("ssh");
    cmd.arg("-i").arg(private_key).args([
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=60",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "BatchMode=yes",
        "-p",
        &port,
    ]);
    cmd.arg(format!("{}@{}", target.user, target.host))
        .arg(remote_cmd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    cmd
}

/// Retry driver shared by the exec variants. `stdin` is streamed to the
/// child (then closed) when present — keep remote stdout/stderr small while
/// stdin is in flight (no concurrent drain until the write finishes).
#[allow(clippy::too_many_arguments)]
async fn exec_with_retries(
    target: &SshTarget,
    private_key: &Path,
    remote_cmd: &str,
    stdin: Option<&[u8]>,
    allow_fail: bool,
    attempts: u32,
    retry_secs: u64,
    timeout_secs: u64,
) -> Result<SshExecOutput, LiumError> {
    let mut last_err = String::new();
    for attempt in 1..=attempts.max(1) {
        let mut cmd = ssh_command(target, private_key, remote_cmd);
        if stdin.is_some() {
            cmd.stdin(Stdio::piped());
        }
        let run = async move {
            let mut child = cmd.spawn().map_err(|e| format!("ssh spawn: {e}"))?;
            if let Some(bytes) = stdin {
                if let Some(mut pipe) = child.stdin.take() {
                    use tokio::io::AsyncWriteExt as _;
                    pipe.write_all(bytes)
                        .await
                        .map_err(|e| format!("ssh stdin: {e}"))?;
                    drop(pipe);
                }
            }
            child
                .wait_with_output()
                .await
                .map_err(|e| format!("ssh wait: {e}"))
        };
        match tokio::time::timeout(Duration::from_secs(timeout_secs), run).await {
            Ok(Ok(out)) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                debug!(
                    attempt,
                    code = out.status.code().unwrap_or(-1),
                    stdout_len = stdout.len(),
                    "ssh exec finished"
                );
                if allow_fail || out.status.success() {
                    return Ok(SshExecOutput {
                        returncode: out.status.code().unwrap_or(-1),
                        stdout,
                        stderr: truncate_tail(&stderr, crate::HARNESS_LOG_RETAIN_BYTES),
                    });
                }
                last_err = format!(
                    "ssh exit {:?}: {}",
                    out.status.code(),
                    truncate_tail(&stderr, 200)
                );
            }
            Ok(Err(e)) => last_err = e,
            Err(_) => last_err = "ssh timed out".into(),
        }
        if attempt < attempts {
            sleep(Duration::from_secs(retry_secs)).await;
        }
    }
    Err(LiumError::Exec(last_err))
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
    exec_with_retries(
        target,
        private_key,
        remote_cmd,
        None,
        false,
        attempts,
        retry_secs,
        timeout_secs,
    )
    .await
}

/// Run SSH and return stdout/stderr even when the remote exit code is
/// non-zero. Still errors on spawn failure / timeout after retries; used when
/// the caller interprets remote failure itself (best-effort log harvest).
pub async fn ssh_exec_allow_fail(
    target: &SshTarget,
    private_key: &Path,
    remote_cmd: &str,
    attempts: u32,
    retry_secs: u64,
    timeout_secs: u64,
) -> Result<SshExecOutput, LiumError> {
    exec_with_retries(
        target,
        private_key,
        remote_cmd,
        None,
        true,
        attempts,
        retry_secs,
        timeout_secs,
    )
    .await
}

/// Run a remote command streaming `stdin` bytes (e.g. a tar archive for
/// post-train eval-asset staging); errors on non-zero exit after retries.
pub async fn ssh_exec_stdin(
    target: &SshTarget,
    private_key: &Path,
    remote_cmd: &str,
    stdin: &[u8],
    attempts: u32,
    retry_secs: u64,
    timeout_secs: u64,
) -> Result<SshExecOutput, LiumError> {
    exec_with_retries(
        target,
        private_key,
        remote_cmd,
        Some(stdin),
        false,
        attempts,
        retry_secs,
        timeout_secs,
    )
    .await
}

/// Run a remote command invoking `on_line` for each stdout line as it
/// arrives (merge stderr into stdout remote-side with `2>&1`). Single
/// attempt; the full output is returned and a non-zero exit is NOT an
/// error — the caller interprets the payload (harness run pattern).
pub async fn ssh_exec_streaming(
    target: &SshTarget,
    private_key: &Path,
    remote_cmd: &str,
    timeout_secs: u64,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> Result<SshExecOutput, LiumError> {
    use tokio::io::AsyncBufReadExt as _;
    let run = async {
        let mut child = ssh_command(target, private_key, remote_cmd)
            .spawn()
            .map_err(|e| format!("ssh spawn: {e}"))?;
        let stdout = child.stdout.take().ok_or("ssh stdout pipe")?;
        let mut lines = tokio::io::BufReader::new(stdout).lines();
        let mut acc = String::new();
        while let Some(line) = lines
            .next_line()
            .await
            .map_err(|e| format!("ssh read: {e}"))?
        {
            on_line(&line);
            acc.push_str(&line);
            acc.push('\n');
        }
        let status = child.wait().await.map_err(|e| format!("ssh wait: {e}"))?;
        Ok((acc, status.code().unwrap_or(-1)))
    };
    match tokio::time::timeout(Duration::from_secs(timeout_secs), run).await {
        Ok(Ok((stdout, returncode))) => Ok(SshExecOutput {
            returncode,
            stdout,
            stderr: String::new(),
        }),
        Ok(Err(e)) => Err(LiumError::Exec(e)),
        Err(_) => Err(LiumError::Exec("ssh timed out".into())),
    }
}

/// Successful SSH run output.
#[derive(Debug, Clone)]
pub struct SshExecOutput {
    #[allow(dead_code)]
    pub returncode: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Keep the **tail** of a log (fatals / tracebacks), UTF-8 safe.
#[must_use]
pub fn truncate_tail(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_owned();
    }
    let mut start = s.len().saturating_sub(n);
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    format!("…{}", &s[start..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn truncate_tail_keeps_suffix() {
        let s = format!("{}FATAL_TRACEBACK", "x".repeat(100));
        let t = truncate_tail(&s, 20);
        assert!(t.starts_with('…'));
        assert!(t.ends_with("FATAL_TRACEBACK"));
        assert!(t.ends_with(&s[s.len() - 20..]));
    }

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
