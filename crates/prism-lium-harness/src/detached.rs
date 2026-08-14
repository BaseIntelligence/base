//! Detached pod harness: survive control-plane SSH drops and allow reattach.

use prism_lium_types::{LiumError, RemoteExecResult};

/// Parent-emitted train-done marker (exact raw line; miner stdout is prefixed).
pub const TRAIN_DONE_MARKER: &str = "PHASE_TRAIN_DONE";
/// Error token when the pod has no harness to reattach to.
pub const HARNESS_ABSENT: &str = "harness_absent";

/// Classify a harvested harness log for the poll / resume loop.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum HarnessProgress {
    /// Still training / evaluating.
    Running,
    /// Train finished; master must stage eval assets (private tier).
    NeedsAssets,
    /// Terminal success / cap breach payload available.
    Done(Box<RemoteExecResult>),
    /// Process ended without a parseable terminal marker.
    Failed(String),
}

/// Build the remote command that starts `main.py` detached (idempotent).
#[must_use]
pub fn detach_launch_cmd(env_exports: &str, timeout_secs: u64) -> String {
    // `setsid` + stdin closed: SSH session death must not kill the harness.
    format!(
        r#"set -e
cd /tmp/prism_eval
if [ -f harness.pid ] && kill -0 "$(cat harness.pid)" 2>/dev/null; then
  echo DETACH_ALREADY
  exit 0
fi
if grep -qE '^(EVAL_OK|CAP_EXCEEDED)$' harness.log 2>/dev/null; then
  echo DETACH_DONE
  exit 0
fi
rm -f harness.exit
: > harness.log
{env_exports}nohup setsid bash -c 'timeout --kill-after=60 {timeout_secs} python3 main.py; echo $? > harness.exit' \
  >> harness.log 2>&1 < /dev/null &
echo $! > harness.pid
echo DETACH_STARTED
"#
    )
}

/// Probe command: pid alive / terminal markers / assets ready.
pub const HARNESS_PROBE_CMD: &str = r#"set +e
cd /tmp/prism_eval 2>/dev/null || { echo 'STATE=absent'; exit 0; }
alive=0
if [ -f harness.pid ] && kill -0 "$(cat harness.pid)" 2>/dev/null; then alive=1; fi
done=0
grep -qE '^(EVAL_OK|CAP_EXCEEDED)$' harness.log 2>/dev/null && done=1
train=0
grep -qxF 'PHASE_TRAIN_DONE' harness.log 2>/dev/null && train=1
ready=0
[ -f eval-assets/.ready ] && ready=1
log=0
[ -f harness.log ] && log=1
echo "STATE=ok alive=$alive done=$done train=$train ready=$ready log=$log"
"#;

/// Harvest metrics + status markers without relying on a fixed-byte log tail.
///
/// v3 `METRICS_JSON=` is often a single line ≫32 KiB (G5/G2 prompts embedded).
/// A `tail -c 32768` of `harness.log` can keep `EVAL_OK` while dropping the
/// `METRICS_JSON=` prefix, so classify sees a failed run after hours of GPU.
/// Prefer the harness sidecar (`metrics.json`), else `grep` the full line.
pub const HARNESS_HARVEST_CMD: &str = r"set +e
cd /tmp/prism_eval 2>/dev/null || exit 0
if [ -f metrics.json ]; then
  printf 'METRICS_JSON='
  cat metrics.json
  printf '\n'
elif [ -f harness.log ]; then
  grep -m1 '^METRICS_JSON=' harness.log 2>/dev/null || true
fi
grep -E '^(EVAL_OK|CAP_EXCEEDED|PHASE_TRAIN_DONE)$' harness.log 2>/dev/null || true
tail -c 8192 harness.log 2>/dev/null || true
";

/// Parsed probe from [`HARNESS_PROBE_CMD`].
#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct HarnessProbe {
    pub present: bool,
    pub pid_alive: bool,
    pub terminal: bool,
    pub train_done: bool,
    pub assets_ready: bool,
    pub has_log: bool,
}

impl HarnessProbe {
    /// True when a resume poll can continue (or harvest a finished log).
    #[must_use]
    pub const fn attachable(self) -> bool {
        self.present && (self.pid_alive || self.terminal || self.has_log || self.train_done)
    }
}

/// Parse [`HARNESS_PROBE_CMD`] stdout.
#[must_use]
pub fn parse_harness_probe(stdout: &str) -> HarnessProbe {
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.starts_with("STATE="))
        .unwrap_or("");
    if line.contains("STATE=absent") || line.is_empty() {
        return HarnessProbe::default();
    }
    let flag = |k: &str| line.contains(&format!("{k}=1"));
    HarnessProbe {
        present: true,
        pid_alive: flag("alive"),
        terminal: flag("done"),
        train_done: flag("train"),
        assets_ready: flag("ready"),
        has_log: flag("log"),
    }
}

/// Progress from log text + whether assets still need staging.
#[must_use]
pub fn classify_log(log: &str, assets_configured: bool, assets_ready: bool) -> HarnessProgress {
    if let Ok(m) = parse_metrics_output(log, 0, log) {
        return HarnessProgress::Done(Box::new(m));
    }
    let train_done = log.lines().any(|l| l.trim_end() == TRAIN_DONE_MARKER);
    if assets_configured && train_done && !assets_ready {
        return HarnessProgress::NeedsAssets;
    }
    if log
        .lines()
        .any(|l| l.trim_end() == "EVAL_OK" || l.trim_end() == "CAP_EXCEEDED")
    {
        return HarnessProgress::Failed(truncate(log, 4000));
    }
    HarnessProgress::Running
}

/// Parse harness output: `EVAL_OK` / `CAP_EXCEEDED` + `METRICS_JSON=`.
pub fn parse_metrics_output(
    stdout: &str,
    returncode: i32,
    stderr_tail: &str,
) -> Result<RemoteExecResult, LiumError> {
    if !stdout.contains("EVAL_OK") && !stdout.contains("CAP_EXCEEDED") {
        return Err(LiumError::Exec(format!(
            "harness failed (code {}): {}",
            returncode,
            truncate(stderr_tail, 4000)
        )));
    }
    let line = stdout
        .lines()
        .find(|l| l.starts_with("METRICS_JSON="))
        .ok_or_else(|| LiumError::Exec("harness EVAL_OK without METRICS_JSON".into()))?;
    let v: RemoteExecResult = serde_json::from_str(&line["METRICS_JSON=".len()..])
        .map_err(|e| LiumError::Exec(format!("metrics json: {e}")))?;
    if v.extra
        .get("cap_exceeded")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        return Ok(v);
    }
    if !v.bpb.is_finite() || v.bpb <= 0.0 {
        return Err(LiumError::Exec(format!(
            "harness bpb not finite: {}",
            v.bpb
        )));
    }
    Ok(v)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    s[s.len().saturating_sub(max)..].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_parses_flags() {
        let p = parse_harness_probe("noise\nSTATE=ok alive=1 done=0 train=1 ready=0 log=1\n");
        assert!(p.present && p.pid_alive && p.train_done && p.has_log && !p.terminal);
        assert!(p.attachable());
        assert!(!parse_harness_probe("STATE=absent\n").present);
    }

    #[test]
    fn classify_needs_assets_and_done() {
        let log = "PHASE_TRAIN_DONE\n";
        assert_eq!(classify_log(log, true, false), HarnessProgress::NeedsAssets);
        let good = "noise\nMETRICS_JSON={\"bpb\":2.5,\"tokens_seen\":1,\"wall_clock_seconds\":1.0,\"gpu_type\":null,\"notes\":\"n\",\"eval_tier\":\"public\"}\nEVAL_OK\n";
        assert!(matches!(
            classify_log(good, false, false),
            HarnessProgress::Done(_)
        ));
    }

    #[test]
    fn parse_metrics_output_gates_eval_ok_and_bpb() {
        let good = "noise\nMETRICS_JSON={\"bpb\":2.5,\"tokens_seen\":1,\"wall_clock_seconds\":1.0,\"gpu_type\":null,\"notes\":\"n\",\"eval_tier\":\"public\"}\nEVAL_OK\n";
        let v = parse_metrics_output(good, 0, "").unwrap();
        assert!((v.bpb - 2.5).abs() < 1e-9);
        assert!(parse_metrics_output("no ok line", 1, "boom")
            .unwrap_err()
            .to_string()
            .contains("harness failed"));
        let bad = "METRICS_JSON={\"bpb\":-1.0,\"tokens_seen\":1,\"wall_clock_seconds\":1.0,\"gpu_type\":null,\"notes\":\"n\"}\nEVAL_OK\n";
        assert!(parse_metrics_output(bad, 0, "")
            .unwrap_err()
            .to_string()
            .contains("bpb"));
    }

    #[test]
    fn parse_metrics_output_accepts_cap_exceeded_terminal() {
        let out = "noise\nMETRICS_JSON={\"bpb\":0.0,\"tokens_seen\":0,\"wall_clock_seconds\":3.0,\"gpu_type\":null,\"notes\":\"parameter cap exceeded\",\"n_params\":999000000,\"cap_exceeded\":true}\nCAP_EXCEEDED\n";
        let v = parse_metrics_output(out, 3, "").unwrap();
        assert_eq!(
            v.extra.get("cap_exceeded").and_then(|x| x.as_bool()),
            Some(true)
        );
        let missing_flag = "METRICS_JSON={\"bpb\":0.0,\"tokens_seen\":0,\"wall_clock_seconds\":3.0,\"gpu_type\":null,\"notes\":\"n\"}\nCAP_EXCEEDED\n";
        assert!(parse_metrics_output(missing_flag, 3, "")
            .unwrap_err()
            .to_string()
            .contains("not finite"));
    }

    #[test]
    fn detach_launch_uses_setsid() {
        let cmd = detach_launch_cmd("export FOO='bar'\n", 100);
        assert!(cmd.contains("setsid"));
        assert!(cmd.contains("harness.pid"));
        assert!(cmd.contains("DETACH_ALREADY"));
    }

    #[test]
    fn classify_log_survives_huge_metrics_json_line() {
        // Regression: v3 battery METRICS_JSON often ≫32 KiB; a log-tail harvest
        // that keeps EVAL_OK but drops the METRICS_JSON= prefix must not happen
        // when the full line is present (sidecar / grep harvest).
        let pad = "x".repeat(40_000);
        let metrics = format!(
            r#"{{"bpb":2.5,"tokens_seen":1,"wall_clock_seconds":1.0,"gpu_type":null,"notes":"{pad}","eval_tier":"public"}}"#
        );
        assert!(metrics.len() > 40_000);
        let log = format!("noise\nMETRICS_JSON={metrics}\nEVAL_OK\n");
        match classify_log(&log, false, false) {
            HarnessProgress::Done(m) => assert!((m.bpb - 2.5).abs() < 1e-9),
            other => panic!("expected Done, got {other:?}"),
        }
        // Truncated tail that still has EVAL_OK but lost the METRICS_JSON= prefix
        // (historical harvest bug) remains Failed — fix is harvest, not parse.
        let truncated_tail = format!("{}\nEVAL_OK\n", &metrics[metrics.len() - 2000..]);
        assert!(
            matches!(
                classify_log(&truncated_tail, false, false),
                HarnessProgress::Failed(_)
            ),
            "EVAL_OK without METRICS_JSON= prefix must stay Failed"
        );
    }

    #[test]
    fn harvest_cmd_prefers_sidecar_and_greps_metrics() {
        assert!(HARNESS_HARVEST_CMD.contains("metrics.json"));
        assert!(HARNESS_HARVEST_CMD.contains("grep -m1 '^METRICS_JSON='"));
        assert!(HARNESS_HARVEST_CMD.contains("EVAL_OK|CAP_EXCEEDED|PHASE_TRAIN_DONE"));
        assert!(!HARNESS_HARVEST_CMD.contains("tail -c 32768"));
    }
}
